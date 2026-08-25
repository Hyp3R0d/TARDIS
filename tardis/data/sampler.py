"""Deterministic sample requests for training epochs."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Mapping
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SampleRequest:
    source: str
    source_index: int
    sample_seed: int
    global_position: int
    epoch: int
    replacement_attempt: int = 0


class BalancedVirtualEpoch:
    """Generate a source-balanced, stateless diagnostic schedule.

    This schedule is retained for focused sampler tests. Production training uses
    :class:`CyclicTrainingEpoch` for bounded, no-replacement coverage windows.
    """

    def __init__(
        self,
        source_sizes: Mapping[str, int],
        *,
        steps_per_epoch: int,
        global_batch_size: int,
        seed: int,
    ) -> None:
        if not source_sizes:
            raise ValueError("source_sizes cannot be empty")
        if any(size <= 0 for size in source_sizes.values()):
            raise ValueError("every source size must be positive")
        if steps_per_epoch <= 0 or global_batch_size <= 0:
            raise ValueError("steps_per_epoch and global_batch_size must be positive")
        self.source_sizes = dict(sorted(source_sizes.items()))
        self.sources = tuple(self.source_sizes)
        self.steps_per_epoch = steps_per_epoch
        self.global_batch_size = global_batch_size
        self.seed = seed

    def requests(self, *, epoch: int, rank: int, world_size: int) -> Iterator[SampleRequest]:
        if epoch < 0:
            raise ValueError("epoch cannot be negative")
        if world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size)")
        if self.global_batch_size % world_size != 0:
            raise ValueError("global_batch_size must be divisible by world_size")
        total = self.steps_per_epoch * self.global_batch_size
        source_offset = _bounded_hash(self.seed, epoch, "source-offset") % len(self.sources)
        for global_position in range(rank, total, world_size):
            source = self.sources[(global_position + source_offset) % len(self.sources)]
            source_size = self.source_sizes[source]
            source_index = (
                _bounded_hash(
                    self.seed,
                    epoch,
                    global_position,
                    source,
                    "index",
                )
                % source_size
            )
            sample_seed = _bounded_hash(
                self.seed,
                epoch,
                global_position,
                source,
                "sample",
            )
            yield SampleRequest(
                source=source,
                source_index=source_index,
                sample_seed=sample_seed,
                global_position=global_position,
                epoch=epoch,
            )

    def replacement(self, request: SampleRequest, *, attempt: int) -> SampleRequest:
        if attempt <= 0:
            raise ValueError("replacement attempt must be positive")
        source_size = self.source_sizes[request.source]
        if source_size <= 1:
            raise ValueError(f"source {request.source!r} cannot provide a distinct replacement")
        offset = 1 + (
            _bounded_hash(
                self.seed,
                request.epoch,
                request.global_position,
                request.source,
                attempt,
                "replacement",
            )
            % (source_size - 1)
        )
        source_index = (request.source_index + offset) % source_size
        sample_seed = _bounded_hash(request.sample_seed, attempt, "replacement-seed")
        return SampleRequest(
            source=request.source,
            source_index=source_index,
            sample_seed=sample_seed,
            global_position=request.global_position,
            epoch=request.epoch,
            replacement_attempt=attempt,
        )


class FullTrainingEpoch:
    """Shuffle and consume the complete union of source-local training records.

    The schedule is stateless and rank-shardable: serial requests contain every record
    exactly once, followed only by the minimum deterministic padding required to make
    global batches and gradient-accumulation windows rectangular. This keeps DDP and
    exact checkpoint resume well-defined without silently truncating a source.
    """

    def __init__(
        self,
        source_sizes: Mapping[str, int],
        *,
        global_batch_size: int,
        gradient_accumulation_steps: int,
        seed: int,
    ) -> None:
        if not source_sizes:
            raise ValueError("source_sizes cannot be empty")
        if any(size <= 0 for size in source_sizes.values()):
            raise ValueError("every source size must be positive")
        if global_batch_size <= 0 or gradient_accumulation_steps <= 0:
            raise ValueError("batch size and accumulation steps must be positive")
        self.source_sizes = dict(sorted(source_sizes.items()))
        self.sources = tuple(self.source_sizes)
        self.global_batch_size = global_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.seed = seed
        alignment = global_batch_size * gradient_accumulation_steps
        record_count = sum(self.source_sizes.values())
        self.steps_per_epoch = ((record_count + alignment - 1) // alignment) * (
            alignment // global_batch_size
        )
        self.total_records = self.steps_per_epoch * global_batch_size
        self.padding_records = self.total_records - record_count

    def requests(self, *, epoch: int, rank: int, world_size: int) -> Iterator[SampleRequest]:
        if epoch < 0:
            raise ValueError("epoch cannot be negative")
        if world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size)")
        if self.global_batch_size % world_size != 0:
            raise ValueError("global_batch_size must be divisible by world_size")

        records = [
            (source, source_index)
            for source in self.sources
            for source_index in range(self.source_sizes[source])
        ]
        # Hash sorting avoids process-global RNG state and gives every worker the same
        # epoch permutation without materializing a shared sampler object.
        records.sort(
            key=lambda item: (
                _bounded_hash(self.seed, epoch, item[0], item[1], "full-epoch-order"),
                item[0],
                item[1],
            )
        )
        if self.padding_records:
            records.extend(records[: self.padding_records])

        for global_position in range(rank, self.total_records, world_size):
            source, source_index = records[global_position]
            sample_seed = _bounded_hash(
                self.seed,
                epoch,
                global_position,
                source,
                source_index,
                "full-epoch-sample",
            )
            yield SampleRequest(
                source=source,
                source_index=source_index,
                sample_seed=sample_seed,
                global_position=global_position,
                epoch=epoch,
            )

    def replacement(self, request: SampleRequest, *, attempt: int) -> SampleRequest:
        if attempt <= 0:
            raise ValueError("replacement attempt must be positive")
        source_size = self.source_sizes[request.source]
        if source_size <= 1:
            raise ValueError(f"source {request.source!r} cannot provide a distinct replacement")
        offset = 1 + (
            _bounded_hash(
                self.seed,
                request.epoch,
                request.global_position,
                request.source,
                attempt,
                "full-epoch-replacement",
            )
            % (source_size - 1)
        )
        source_index = (request.source_index + offset) % source_size
        return SampleRequest(
            source=request.source,
            source_index=source_index,
            sample_seed=_bounded_hash(request.sample_seed, attempt, "replacement-seed"),
            global_position=request.global_position,
            epoch=request.epoch,
            replacement_attempt=attempt,
        )


class CyclicTrainingEpoch:
    """Consume a time-bounded window from a continuous shuffled dataset cycle.

    Consecutive epochs advance through one deterministic permutation of the complete
    three-source training union. A record is not repeated until every record has been
    requested once; a new independently shuffled coverage cycle then begins. This gives
    wall-clock-bounded epochs without reverting to random sampling with replacement.
    """

    def __init__(
        self,
        source_sizes: Mapping[str, int],
        *,
        steps_per_epoch: int,
        global_batch_size: int,
        seed: int,
    ) -> None:
        if not source_sizes:
            raise ValueError("source_sizes cannot be empty")
        if any(size <= 0 for size in source_sizes.values()):
            raise ValueError("every source size must be positive")
        if steps_per_epoch <= 0 or global_batch_size <= 0:
            raise ValueError("steps_per_epoch and global_batch_size must be positive")
        self.source_sizes = dict(sorted(source_sizes.items()))
        self.sources = tuple(self.source_sizes)
        self.steps_per_epoch = steps_per_epoch
        self.global_batch_size = global_batch_size
        self.seed = seed
        self.records_per_cycle = sum(self.source_sizes.values())
        self.records_per_epoch = steps_per_epoch * global_batch_size

    def requests(self, *, epoch: int, rank: int, world_size: int) -> Iterator[SampleRequest]:
        if epoch < 0:
            raise ValueError("epoch cannot be negative")
        if world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size)")
        if self.global_batch_size % world_size != 0:
            raise ValueError("global_batch_size must be divisible by world_size")

        cycle_records: dict[int, list[tuple[str, int]]] = {}
        epoch_offset = epoch * self.records_per_epoch
        for global_position in range(rank, self.records_per_epoch, world_size):
            absolute_position = epoch_offset + global_position
            cycle, cycle_position = divmod(absolute_position, self.records_per_cycle)
            records = cycle_records.get(cycle)
            if records is None:
                records = self._cycle_records(cycle)
                cycle_records[cycle] = records
            source, source_index = records[cycle_position]
            yield SampleRequest(
                source=source,
                source_index=source_index,
                sample_seed=_bounded_hash(
                    self.seed,
                    cycle,
                    cycle_position,
                    source,
                    source_index,
                    "cyclic-epoch-sample",
                ),
                global_position=global_position,
                epoch=epoch,
            )

    def replacement(self, request: SampleRequest, *, attempt: int) -> SampleRequest:
        if attempt <= 0:
            raise ValueError("replacement attempt must be positive")
        source_size = self.source_sizes[request.source]
        if source_size <= 1:
            raise ValueError(f"source {request.source!r} cannot provide a distinct replacement")
        offset = 1 + (
            _bounded_hash(
                self.seed,
                request.epoch,
                request.global_position,
                request.source,
                attempt,
                "cyclic-epoch-replacement",
            )
            % (source_size - 1)
        )
        return SampleRequest(
            source=request.source,
            source_index=(request.source_index + offset) % source_size,
            sample_seed=_bounded_hash(request.sample_seed, attempt, "replacement-seed"),
            global_position=request.global_position,
            epoch=request.epoch,
            replacement_attempt=attempt,
        )

    def _cycle_records(self, cycle: int) -> list[tuple[str, int]]:
        records = [
            (source, source_index)
            for source in self.sources
            for source_index in range(self.source_sizes[source])
        ]
        records.sort(
            key=lambda item: (
                _bounded_hash(self.seed, cycle, item[0], item[1], "cyclic-epoch-order"),
                item[0],
                item[1],
            )
        )
        return records


def _bounded_hash(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode()
    digest = hashlib.blake2b(payload, digest_size=8, person=b"TARDISSample").digest()
    return int.from_bytes(digest, "big")
