"""Resumable optimizer state machine for TARDIS training."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol, cast

import numpy as np
import torch
import torch.distributed as dist
from numpy.typing import NDArray
from torch import nn
from torch.amp.grad_scaler import GradScaler
from torch.nn.parallel import DistributedDataParallel

from tardis.models.factory import (
    load_tardis_temporal_state_dict,
    tardis_forward_migration_names,
    tardis_temporal_state_dict,
)
from tardis.models.tardis import TARDISModel
from tardis.training.curriculum import CurriculumPoint, CurriculumSchedule
from tardis.training.validation import (
    ValidationCheckpointSelector,
    ValidationMetric,
    ValidationScore,
)
from tardis.utils.checkpoint import CHECKPOINT_SCHEMA_VERSION, atomic_torch_save


@dataclass(frozen=True, slots=True)
class ObjectiveOutput:
    """One differentiable objective and its detached logging components."""

    total: torch.Tensor
    losses: Mapping[str, torch.Tensor]
    metrics: Mapping[str, torch.Tensor] = field(default_factory=dict)


class TrainingObjective(Protocol):
    def __call__(
        self,
        model: nn.Module,
        batch: object,
        point: CurriculumPoint,
        generator: torch.Generator,
    ) -> ObjectiveOutput: ...


@dataclass(frozen=True, slots=True)
class TrainEngineOptions:
    learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-2
    gradient_accumulation_steps: int = 4
    gradient_clip_norm: float = 1.0
    warmup_steps: int = 500
    total_optimizer_steps: int = 12_000
    precision: str = "bf16"
    ema_decay: float = 0.999

    def __post_init__(self) -> None:
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("optimizer learning rate/weight decay are invalid")
        if self.gradient_accumulation_steps <= 0 or self.gradient_clip_norm <= 0:
            raise ValueError("gradient accumulation and clipping must be positive")
        if self.warmup_steps < 0 or self.total_optimizer_steps <= 0:
            raise ValueError("scheduler step counts are invalid")
        if self.precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be fp32, fp16, or bf16")
        if not 0 <= self.ema_decay < 1:
            raise ValueError("ema_decay must be in [0, 1)")


@dataclass(frozen=True, slots=True)
class TrainStepResult:
    total_loss: float
    losses: dict[str, float]
    optimizer_updated: bool
    skipped_nonfinite: bool
    gradient_norm: float | None
    learning_rate: float
    micro_step: int
    optimizer_step: int
    stage: str
    metrics: dict[str, float] = field(default_factory=dict)


class ModelEMA:
    """Checkpointable average over the complete deployable temporal state."""

    def __init__(self, model: nn.Module, *, decay: float) -> None:
        if not 0 <= decay < 1:
            raise ValueError("EMA decay must be in [0, 1)")
        self.decay = decay
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in ema_parameter_map(model).items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        parameters = dict(model.named_parameters())
        if set(parameters).issuperset(self.shadow) is False:
            raise ValueError("EMA parameter names do not match model")
        for name, average in self.shadow.items():
            parameter = parameters[name].detach()
            average.lerp_(parameter.to(device=average.device, dtype=average.dtype), 1 - self.decay)

    def state_dict(self) -> dict[str, object]:
        return {
            "decay": self.decay,
            "shadow": {name: value.clone() for name, value in self.shadow.items()},
        }

    def load_state_dict(self, state: Mapping[str, object], model: nn.Module) -> None:
        if set(state) != {"decay", "shadow"} or float(cast(float, state["decay"])) != self.decay:
            raise ValueError("EMA state is incompatible")
        raw_shadow = state["shadow"]
        if not isinstance(raw_shadow, Mapping):
            raise ValueError("EMA shadow parameter names are incompatible")
        expected_names = set(self.shadow)
        received_names = set(raw_shadow)
        missing = expected_names - received_names
        unexpected = received_names - expected_names
        allowed_missing = tardis_forward_migration_names(expected_names)
        if not missing.issubset(allowed_missing) or unexpected:
            raise ValueError(
                "EMA shadow parameter names are incompatible; "
                f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        migrated_shadow = dict(raw_shadow)
        parameters = dict(model.named_parameters())
        for name in missing:
            source_name = _ema_forward_migration_source(name)
            source_value = None if source_name is None else raw_shadow.get(source_name)
            if (
                isinstance(source_value, torch.Tensor)
                and source_value.shape == parameters[name].shape
            ):
                migrated_shadow[name] = source_value.detach().to(device="cpu", copy=True)
            else:
                migrated_shadow[name] = parameters[name].detach().to(
                    device="cpu", copy=True
                )
        parameters = dict(model.named_parameters())
        for name, value in migrated_shadow.items():
            if not isinstance(name, str) or not isinstance(value, torch.Tensor):
                raise ValueError("EMA shadow entries must be named tensors")
            parameter = parameters[name]
            if value.shape != parameter.shape:
                raise ValueError(f"EMA tensor shape mismatch for {name!r}")
            self.shadow[name] = value.to(device=parameter.device, dtype=parameter.dtype).clone()


def _ema_forward_migration_source(name: str) -> str | None:
    if ".output_projection." in name:
        return None
    if name.startswith("keyframe_residual_dit."):
        return name.replace("keyframe_residual_dit.", "residual_dit.", 1)
    if name.startswith("transition_lite_corrector."):
        return name.replace("transition_lite_corrector.", "lite_corrector.", 1)
    return None


def ema_parameter_map(model: nn.Module) -> dict[str, nn.Parameter]:
    """Return the complete deployable EMA state for a training model."""

    return {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.is_floating_point()
        and (
            (isinstance(model, TARDISModel) and not name.startswith("priors."))
            or (not isinstance(model, TARDISModel) and parameter.requires_grad)
        )
    }


class TrainEngine:
    """Own optimizer, curriculum, exact resume state, and validation selection."""

    def __init__(
        self,
        model: nn.Module,
        *,
        objective: TrainingObjective,
        options: TrainEngineOptions,
        curriculum: CurriculumSchedule,
        generator: torch.Generator,
        selector: ValidationCheckpointSelector | None = None,
    ) -> None:
        unwrapped_model = _unwrap_execution_model(model)
        trainable = [
            parameter for parameter in unwrapped_model.parameters() if parameter.requires_grad
        ]
        if not trainable:
            raise ValueError("training model has no trainable parameters")
        self.execution_model = model
        self.unwrapped_model = unwrapped_model
        self.objective = objective
        self.options = options
        self.curriculum = curriculum
        self.generator = generator
        self.selector = selector or ValidationCheckpointSelector()
        self.optimizer = torch.optim.AdamW(
            trainable,
            lr=options.learning_rate,
            weight_decay=options.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=_learning_rate_multiplier(
                warmup_steps=options.warmup_steps,
                total_steps=options.total_optimizer_steps,
            ),
        )
        device = _model_device(unwrapped_model)
        scaler_enabled = options.precision == "fp16" and device.type == "cuda"
        self.scaler = GradScaler("cuda", enabled=scaler_enabled)
        self.ema = ModelEMA(unwrapped_model, decay=options.ema_decay)
        self.micro_step = 0
        self.optimizer_step = 0
        self.accumulation_index = 0
        self.nonfinite_ledger: list[tuple[int, tuple[str, ...]]] = []
        self.optimizer.zero_grad(set_to_none=True)

    def train_microbatch(
        self,
        batch: object,
        *,
        batch_ids: tuple[str, ...] = (),
    ) -> TrainStepResult:
        self.execution_model.train(True)
        point = self.curriculum.at_step(self.optimizer_step)
        device = _model_device(self.unwrapped_model)
        autocast_dtype = _autocast_dtype(self.options.precision)
        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=autocast_dtype is not None and device.type in {"cpu", "cuda"},
        ):
            output = self.objective(self.execution_model, batch, point, self.generator)
        _validate_objective(output)
        self.micro_step += 1
        detached_total = float(output.total.detach().float().item())
        detached_losses = {
            name: float(loss.detach().float().item()) for name, loss in output.losses.items()
        }
        detached_metrics = {
            name: float(metric.detach().float().item()) for name, metric in output.metrics.items()
        }
        if not _all_ranks_finite(math.isfinite(detached_total), device):
            self.optimizer.zero_grad(set_to_none=True)
            self.accumulation_index = 0
            self.nonfinite_ledger.append((self.micro_step, tuple(batch_ids)))
            return self._result(
                detached_total,
                detached_losses,
                detached_metrics,
                point,
                optimizer_updated=False,
                skipped_nonfinite=True,
                gradient_norm=None,
            )

        scaled_loss = output.total / self.options.gradient_accumulation_steps
        torch.autograd.backward(self.scaler.scale(scaled_loss))
        self.accumulation_index += 1
        if self.accumulation_index < self.options.gradient_accumulation_steps:
            return self._result(
                detached_total,
                detached_losses,
                detached_metrics,
                point,
                optimizer_updated=False,
                skipped_nonfinite=False,
                gradient_norm=None,
            )

        self.scaler.unscale_(self.optimizer)
        gradient_norm_tensor = torch.nn.utils.clip_grad_norm_(
            self.unwrapped_model.parameters(),
            self.options.gradient_clip_norm,
        )
        gradient_norm = float(gradient_norm_tensor.detach().float().item())
        if not math.isfinite(gradient_norm):
            self.optimizer.zero_grad(set_to_none=True)
            self.accumulation_index = 0
            self.nonfinite_ledger.append((self.micro_step, tuple(batch_ids)))
            self.scaler.update()
            return self._result(
                detached_total,
                detached_losses,
                detached_metrics,
                point,
                optimizer_updated=False,
                skipped_nonfinite=True,
                gradient_norm=gradient_norm,
            )

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.accumulation_index = 0
        self.optimizer_step += 1
        self.ema.update(self.unwrapped_model)
        _update_objective_teacher(self.objective, self.unwrapped_model)
        return self._result(
            detached_total,
            detached_losses,
            detached_metrics,
            point,
            optimizer_updated=True,
            skipped_nonfinite=False,
            gradient_norm=gradient_norm,
        )

    def state_dict(self, *, epoch: int, next_batch_index: int) -> dict[str, object]:
        if epoch < 0 or next_batch_index < 0:
            raise ValueError("checkpoint position cannot be negative")
        return {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "kind": "tardis_train",
            "epoch": epoch,
            "next_batch_index": next_batch_index,
            "engine_options": asdict(self.options),
            "curriculum_durations": self.curriculum.durations,
            "model": _model_checkpoint_state(self.unwrapped_model),
            "objective": _objective_checkpoint_state(self.objective),
            "optimizer": deepcopy(self.optimizer.state_dict()),
            "scheduler": deepcopy(self.scheduler.state_dict()),
            "scaler": deepcopy(self.scaler.state_dict()),
            "ema": deepcopy(self.ema.state_dict()),
            "micro_step": self.micro_step,
            "optimizer_step": self.optimizer_step,
            "accumulation_index": self.accumulation_index,
            "gradients": _gradient_state(self.unwrapped_model),
            **self.stochastic_state_dict(),
            "selector": _selector_state(self.selector),
            "nonfinite_ledger": list(self.nonfinite_ledger),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> tuple[int, int]:
        _validate_train_state(state, self.options, self.curriculum)
        _restore_model_checkpoint_state(
            self.unwrapped_model,
            _mapping(state["model"], "model"),
        )
        _restore_objective_checkpoint_state(self.objective, state["objective"])
        self.optimizer.load_state_dict(dict(_mapping(state["optimizer"], "optimizer")))
        self.scheduler.load_state_dict(dict(_mapping(state["scheduler"], "scheduler")))
        self.scaler.load_state_dict(dict(_mapping(state["scaler"], "scaler")))
        self.ema.load_state_dict(_mapping(state["ema"], "ema"), self.unwrapped_model)
        self.micro_step = int(cast(int, state["micro_step"]))
        self.optimizer_step = int(cast(int, state["optimizer_step"]))
        self.accumulation_index = int(cast(int, state["accumulation_index"]))
        _restore_gradients(self.unwrapped_model, _mapping(state["gradients"], "gradients"))
        self.load_stochastic_state_dict(state)
        _restore_selector(self.selector, _mapping(state["selector"], "selector"))
        ledger = state["nonfinite_ledger"]
        if not isinstance(ledger, list):
            raise ValueError("nonfinite ledger must be a list")
        self.nonfinite_ledger = [
            (int(item[0]), tuple(str(value) for value in item[1])) for item in ledger
        ]
        return int(cast(int, state["epoch"])), int(cast(int, state["next_batch_index"]))

    def stochastic_state_dict(self) -> dict[str, object]:
        """Capture the process-local random streams required for exact rank resume."""

        return {
            "generator_state": self.generator.get_state().clone(),
            "torch_rng_state": torch.get_rng_state().clone(),
            "cuda_rng_state": (
                [state.clone() for state in torch.cuda.get_rng_state_all()]
                if torch.cuda.is_available()
                else []
            ),
            "python_rng_state": random.getstate(),
            "numpy_rng_state": _numpy_rng_state(),
            "objective_rank_state": _objective_rank_state(self.objective),
        }

    def load_stochastic_state_dict(self, state: Mapping[str, object]) -> None:
        """Restore one rank's generator and process RNG streams."""

        required = {
            "generator_state",
            "torch_rng_state",
            "cuda_rng_state",
            "python_rng_state",
            "numpy_rng_state",
            "objective_rank_state",
        }
        if not required.issubset(state):
            raise ValueError(f"stochastic state is missing keys: {sorted(required - set(state))}")
        generator_state = state["generator_state"]
        torch_rng_state = state["torch_rng_state"]
        if not isinstance(generator_state, torch.Tensor) or not isinstance(
            torch_rng_state, torch.Tensor
        ):
            raise ValueError("checkpoint RNG states must be tensors")
        self.generator.set_state(generator_state.cpu())
        torch.set_rng_state(torch_rng_state.cpu())
        cuda_rng_state = state["cuda_rng_state"]
        if torch.cuda.is_available() and isinstance(cuda_rng_state, list) and cuda_rng_state:
            torch.cuda.set_rng_state_all(cuda_rng_state)
        random.setstate(cast(tuple[Any, ...], state["python_rng_state"]))
        _restore_numpy_rng_state(_mapping(state["numpy_rng_state"], "numpy_rng_state"))
        _restore_objective_rank_state(self.objective, state["objective_rank_state"])

    def save_epoch(
        self,
        checkpoint_dir: Any,
        *,
        epoch: int,
        next_batch_index: int,
        validation_metrics: Mapping[str, Mapping[str, float]],
    ) -> bool:
        improved = self.selector.update(validation_metrics, epoch=epoch)
        payload = self.state_dict(epoch=epoch, next_batch_index=next_batch_index)
        score = self.selector.best_score
        payload["validation_score"] = _score_state(score) if score is not None else None
        from pathlib import Path

        directory = Path(checkpoint_dir)
        atomic_torch_save(payload, directory / "latest.pt")
        if improved:
            atomic_torch_save(payload, directory / "best.pt")
        return improved

    def _result(
        self,
        total_loss: float,
        losses: dict[str, float],
        metrics: dict[str, float],
        point: CurriculumPoint,
        *,
        optimizer_updated: bool,
        skipped_nonfinite: bool,
        gradient_norm: float | None,
    ) -> TrainStepResult:
        return TrainStepResult(
            total_loss=total_loss,
            losses=losses,
            metrics=metrics,
            optimizer_updated=optimizer_updated,
            skipped_nonfinite=skipped_nonfinite,
            gradient_norm=gradient_norm,
            learning_rate=float(self.optimizer.param_groups[0]["lr"]),
            micro_step=self.micro_step,
            optimizer_step=self.optimizer_step,
            stage=point.stage.value,
        )


def _learning_rate_multiplier(*, warmup_steps: int, total_steps: int) -> Callable[[int], float]:
    def multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max((step + 1) / warmup_steps, 1.0e-8)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    return multiplier


def _autocast_dtype(precision: str) -> torch.dtype | None:
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    return None


def _model_device(model: nn.Module) -> torch.device:
    parameter = next(model.parameters(), None)
    return torch.device("cpu") if parameter is None else parameter.device


def _all_ranks_finite(local_finite: bool, device: torch.device) -> bool:
    if not dist.is_available() or not dist.is_initialized():
        return local_finite
    flag = torch.tensor(int(local_finite), device=device, dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def _unwrap_execution_model(model: nn.Module) -> nn.Module:
    current = model
    seen: set[int] = set()
    while True:
        if id(current) in seen:
            raise ValueError("training model wrappers contain a cycle")
        seen.add(id(current))
        if isinstance(current, DistributedDataParallel):
            current = current.module
            continue
        original = getattr(current, "_orig_mod", None)
        if isinstance(original, nn.Module):
            current = original
            continue
        return current


def _validate_objective(output: ObjectiveOutput) -> None:
    if output.total.ndim != 0 or not output.total.is_floating_point():
        raise ValueError("training objective total must be a floating scalar")
    if not output.losses:
        raise ValueError("training objective must expose component losses")
    if any(loss.ndim != 0 for loss in output.losses.values()):
        raise ValueError("training objective component losses must be scalar")
    if any(metric.ndim != 0 for metric in output.metrics.values()):
        raise ValueError("training objective metrics must be scalar")


def _gradient_state(model: nn.Module) -> dict[str, torch.Tensor | None]:
    return {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _model_checkpoint_state(model: nn.Module) -> dict[str, torch.Tensor]:
    if isinstance(model, TARDISModel):
        return tardis_temporal_state_dict(model)
    return deepcopy(model.state_dict())


def _restore_model_checkpoint_state(
    model: nn.Module,
    state: Mapping[str, object],
) -> None:
    if isinstance(model, TARDISModel):
        load_tardis_temporal_state_dict(model, _tensor_mapping(state, "model"))
        return
    model.load_state_dict(state)


def _objective_checkpoint_state(objective: TrainingObjective) -> dict[str, object] | None:
    state_function = getattr(objective, "state_dict", None)
    if state_function is None:
        return None
    if not callable(state_function):
        raise ValueError("training objective state_dict must be callable")
    raw_state = cast(Callable[[], object], state_function)()
    if not isinstance(raw_state, Mapping):
        raise ValueError("training objective state_dict must return a mapping")
    return deepcopy(dict(raw_state))


def _update_objective_teacher(objective: TrainingObjective, model: nn.Module) -> None:
    update_function = getattr(objective, "update_teacher", None)
    if update_function is None:
        return
    if not callable(update_function):
        raise ValueError("training objective update_teacher must be callable")
    cast(Callable[[nn.Module], object], update_function)(model)


def _restore_objective_checkpoint_state(
    objective: TrainingObjective,
    state: object,
) -> None:
    load_function = getattr(objective, "load_state_dict", None)
    if state is None:
        if load_function is not None:
            raise ValueError("checkpoint has no state for the stateful training objective")
        return
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint objective state must be a mapping or None")
    if not callable(load_function):
        raise ValueError("checkpoint objective state requires a stateful training objective")
    cast(Callable[[Mapping[str, object]], object], load_function)(cast(Mapping[str, object], state))


def _objective_rank_state(objective: TrainingObjective) -> dict[str, object] | None:
    state_function = getattr(objective, "rank_state_dict", None)
    if state_function is None:
        return None
    if not callable(state_function):
        raise ValueError("training objective rank_state_dict must be callable")
    raw_state = cast(Callable[[], object], state_function)()
    if not isinstance(raw_state, Mapping):
        raise ValueError("training objective rank_state_dict must return a mapping")
    return deepcopy(dict(raw_state))


def _restore_objective_rank_state(
    objective: TrainingObjective,
    state: object,
) -> None:
    load_function = getattr(objective, "load_rank_state_dict", None)
    if state is None:
        if load_function is not None:
            raise ValueError("checkpoint has no rank state for the rank-stateful objective")
        return
    if not isinstance(state, Mapping) or not callable(load_function):
        raise ValueError("checkpoint objective rank state is incompatible")
    cast(Callable[[Mapping[str, object]], object], load_function)(cast(Mapping[str, object], state))


def _restore_gradients(model: nn.Module, state: Mapping[str, object]) -> None:
    parameters = {
        name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    if set(state) != set(parameters):
        raise ValueError("checkpoint gradient names do not match model")
    for name, value in state.items():
        if value is None:
            parameters[name].grad = None
        elif isinstance(value, torch.Tensor) and value.shape == parameters[name].shape:
            parameters[name].grad = value.to(
                device=parameters[name].device,
                dtype=parameters[name].dtype,
            )
        else:
            raise ValueError(f"invalid checkpoint gradient for {name!r}")


def _numpy_rng_state() -> dict[str, object]:
    state = cast(tuple[str, NDArray[np.uint32], int, int, float], np.random.get_state())
    values = np.asarray(state[1], dtype=np.uint32)
    return {
        "name": str(state[0]),
        "values": torch.from_numpy(values.astype(np.int64)),
        "position": int(state[2]),
        "has_gauss": int(state[3]),
        "cached_gaussian": float(state[4]),
    }


def _restore_numpy_rng_state(state: Mapping[str, object]) -> None:
    values = state.get("values")
    if not isinstance(values, torch.Tensor):
        raise ValueError("NumPy RNG values must be a tensor")
    np.random.set_state(
        (
            str(state["name"]),
            values.cpu().numpy().astype(np.uint32),
            int(cast(int, state["position"])),
            int(cast(int, state["has_gauss"])),
            float(cast(float, state["cached_gaussian"])),
        )
    )


def _selector_state(selector: ValidationCheckpointSelector) -> dict[str, object]:
    return {
        "baselines": {str(metric): value for metric, value in selector.baselines.items()},
        "tolerance": selector.tolerance,
        "pareto_tolerance": selector.pareto_tolerance,
        "best_epoch": selector.best_epoch,
        "best_score": _score_state(selector.best_score),
    }


def _score_state(score: ValidationScore | None) -> dict[str, object] | None:
    if score is None:
        return None
    return {
        "source_metrics": score.source_metrics,
        "average_metrics": score.average_metrics,
        "normalized_metrics": score.normalized_metrics,
        "composite": score.composite,
    }


def _restore_selector(selector: ValidationCheckpointSelector, state: Mapping[str, object]) -> None:
    if float(cast(float, state["tolerance"])) != selector.tolerance:
        raise ValueError("checkpoint selector tolerance does not match")
    if float(cast(float, state["pareto_tolerance"])) != selector.pareto_tolerance:
        raise ValueError("checkpoint selector Pareto tolerance does not match")
    raw_baselines = _mapping(state["baselines"], "selector baselines")
    checkpoint_baselines = {
        ValidationMetric(str(key)): float(cast(float, value))
        for key, value in raw_baselines.items()
    }
    current_baselines = {
        ValidationMetric(metric): float(value) for metric, value in selector.baselines.items()
    }
    if checkpoint_baselines != current_baselines:
        raise ValueError("checkpoint validation baselines do not match")
    selector.best_epoch = (
        None if state["best_epoch"] is None else int(cast(int, state["best_epoch"]))
    )
    raw_score = state["best_score"]
    if raw_score is None:
        selector.best_score = None
        return
    score = _mapping(raw_score, "best validation score")
    selector.best_score = ValidationScore(
        source_metrics=_nested_float_mapping(score["source_metrics"]),
        average_metrics=_float_mapping(score["average_metrics"]),
        normalized_metrics=_float_mapping(score["normalized_metrics"]),
        composite=float(cast(float, score["composite"])),
    )


def _validate_train_state(
    state: Mapping[str, object],
    options: TrainEngineOptions,
    curriculum: CurriculumSchedule,
) -> None:
    required = {
        "schema_version",
        "kind",
        "epoch",
        "next_batch_index",
        "engine_options",
        "curriculum_durations",
        "model",
        "objective",
        "optimizer",
        "scheduler",
        "scaler",
        "ema",
        "micro_step",
        "optimizer_step",
        "accumulation_index",
        "gradients",
        "generator_state",
        "torch_rng_state",
        "cuda_rng_state",
        "python_rng_state",
        "numpy_rng_state",
        "objective_rank_state",
        "selector",
        "nonfinite_ledger",
    }
    if not required.issubset(state):
        raise ValueError(f"training checkpoint is missing keys: {sorted(required - set(state))}")
    if state["schema_version"] != CHECKPOINT_SCHEMA_VERSION or state["kind"] != "tardis_train":
        raise ValueError("training checkpoint schema or kind is incompatible")
    if dict(_mapping(state["engine_options"], "engine options")) != asdict(options):
        raise ValueError("training checkpoint engine options do not match")
    if tuple(cast(tuple[int, ...], state["curriculum_durations"])) != curriculum.durations:
        raise ValueError("training checkpoint curriculum does not match")


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return cast(Mapping[str, object], value)


def _tensor_mapping(value: Mapping[str, object], name: str) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, torch.Tensor):
            raise ValueError(f"{name} entries must be named tensors")
        tensors[key] = item
    return tensors


def _float_mapping(value: object) -> dict[str, float]:
    return {
        str(key): float(cast(float, item))
        for key, item in _mapping(value, "metric mapping").items()
    }


def _nested_float_mapping(value: object) -> dict[str, dict[str, float]]:
    return {
        str(key): _float_mapping(item)
        for key, item in _mapping(value, "nested metric mapping").items()
    }
