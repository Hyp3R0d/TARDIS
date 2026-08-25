"""Low-overhead CPU/GPU resource sampling."""

from __future__ import annotations

import os
import resource
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from statistics import fmean

import torch

_BYTES_PER_MIB = 1024 * 1024


@dataclass(frozen=True, slots=True)
class ResourceSnapshot:
    timestamp: float
    gpu_allocated_mb: float
    gpu_reserved_mb: float
    gpu_total_mb: float
    gpu_utilization_percent: float
    process_rss_mb: float


@dataclass(frozen=True, slots=True)
class ResourceSummary:
    sample_count: int
    peak_allocated_mb: float
    peak_reserved_mb: float
    peak_process_rss_mb: float
    mean_gpu_utilization_percent: float


def _nvidia_utilization(device_index: int) -> float:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--id={device_index}",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        return float(completed.stdout.strip().splitlines()[0])
    except (FileNotFoundError, IndexError, ValueError, subprocess.SubprocessError):
        return 0.0


def sample_resources(device: torch.device | None = None) -> ResourceSnapshot:
    """Sample the current process and one CUDA device without synchronizing kernels."""
    resolved = device or torch.device("cuda", 0)
    allocated = reserved = total = utilization = 0.0
    if resolved.type == "cuda" and torch.cuda.is_available():
        index = resolved.index or 0
        allocated = torch.cuda.memory_allocated(index) / _BYTES_PER_MIB
        reserved = torch.cuda.memory_reserved(index) / _BYTES_PER_MIB
        total = torch.cuda.get_device_properties(index).total_memory / _BYTES_PER_MIB
        utilization = _nvidia_utilization(index)
    maximum_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mb = (
        float(maximum_rss) / 1024.0 if os.name == "posix" else float(maximum_rss) / _BYTES_PER_MIB
    )
    return ResourceSnapshot(time.time(), allocated, reserved, total, utilization, rss_mb)


class ResourceMonitor:
    """Collect snapshots synchronously or in a bounded background thread."""

    def __init__(
        self,
        sample_fn: Callable[[], ResourceSnapshot] = sample_resources,
        interval_seconds: float = 1.0,
    ) -> None:
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self._sample_fn = sample_fn
        self._interval = interval_seconds
        self._samples: list[ResourceSnapshot] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def sample_once(self) -> ResourceSnapshot:
        snapshot = self._sample_fn()
        with self._lock:
            self._samples.append(snapshot)
        return snapshot

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="resource-monitor", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            self.sample_once()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self._interval * 2))

    def summary(self) -> ResourceSummary:
        with self._lock:
            samples = tuple(self._samples)
        if not samples:
            return ResourceSummary(0, 0.0, 0.0, 0.0, 0.0)
        return ResourceSummary(
            sample_count=len(samples),
            peak_allocated_mb=max(item.gpu_allocated_mb for item in samples),
            peak_reserved_mb=max(item.gpu_reserved_mb for item in samples),
            peak_process_rss_mb=max(item.process_rss_mb for item in samples),
            mean_gpu_utilization_percent=fmean(item.gpu_utilization_percent for item in samples),
        )

    def __enter__(self) -> ResourceMonitor:
        self.start()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.stop()
