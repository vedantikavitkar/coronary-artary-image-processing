"""Latency measurement utilities for the cardiology coronary pipeline.

The assignment mandates a hard budget of 36 ms per frame, measured from the
moment the input frame reaches the pipeline until the final processed frame
exists.  This module provides the instrumentation used to prove that budget:

    from latency import BUDGET_MS, StageTimer, measure

    with StageTimer() as t:                     # optional per-stage breakdown
        out = pipeline.process(frame)
    print(t.stages)                             # {'ingest': 0.8, ...}

    report = measure(pipeline.process, frames)  # avg/min/max/p95/FPS + pass/fail
    print(report.table())

Nothing in here touches the processing logic; it only times it.
"""

from __future__ import annotations

import json
import os
import platform
import statistics
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# Mandatory per-frame budget from the assignment.
BUDGET_MS = 36.0


class LatencyBudgetExceeded(RuntimeError):
    """Raised by :func:`enforce` when a frame exceeds the mandatory budget."""


# ---------------------------------------------------------------------------
# hardware description
# ---------------------------------------------------------------------------
def _cpu_name() -> str:
    try:
        if os.name == "nt":
            import winreg

            key = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key) as handle:
                return str(winreg.QueryValueEx(handle, "ProcessorNameString")[0])
    except Exception:
        pass
    return platform.processor() or platform.machine() or "unknown"


def _total_ram_gb() -> Optional[float]:
    try:
        if os.name == "nt":
            import ctypes

            class _MemStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            status = _MemStatus()
            status.dwLength = ctypes.sizeof(_MemStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return round(status.ullTotalPhys / 1024 ** 3, 1)
    except Exception:
        pass
    try:  # POSIX fallback
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return round(pages * page_size / 1024 ** 3, 1)
    except Exception:
        return None


def host_load(window: float = 0.25) -> Optional[float]:
    """Whole-machine CPU utilisation in percent, sampled over ``window`` s.

    Used to qualify the timing runs: on a busy host the wall-clock latency is
    inflated by descheduling even when the pipeline itself is fast.
    """
    try:
        if os.name == "nt":
            import ctypes

            class _FileTime(ctypes.Structure):
                _fields_ = [("low", ctypes.c_ulong), ("high", ctypes.c_ulong)]

                def value(self) -> int:
                    return (self.high << 32) | self.low

            idle, kernel, user = _FileTime(), _FileTime(), _FileTime()
            get_times = ctypes.windll.kernel32.GetSystemTimes
            if not get_times(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
                return None
            start = (kernel.value() + user.value(), idle.value())
            time.sleep(window)
            if not get_times(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
                return None
            end = (kernel.value() + user.value(), idle.value())
        else:
            def _cpu() -> Tuple[int, int]:
                with open("/proc/stat", "r", encoding="ascii") as handle:
                    parts = handle.readline().split()
                values = [int(v) for v in parts[1:]]
                return sum(values), values[3]

            start = _cpu()
            time.sleep(window)
            end = _cpu()
        total = (end[0] - start[0]) or 1
        idle = (end[1] - start[1]) or 0
        return max(0.0, min(100.0, 100.0 * (1.0 - idle / total)))
    except Exception:
        return None


def hardware_info() -> Dict[str, Any]:
    """Hardware / software configuration used for the timing run."""
    info: Dict[str, Any] = {
        "cpu": _cpu_name(),
        "logical_cores": os.cpu_count(),
        "ram_gb": _total_ram_gb(),
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "python": platform.python_version(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["torch_threads"] = torch.get_num_threads()
        info["cuda_available"] = bool(torch.cuda.is_available())
    except Exception:
        pass
    try:
        import cv2

        info["opencv"] = cv2.__version__
        info["opencv_threads"] = cv2.getNumThreads()
    except Exception:
        pass
    try:
        import numpy as np

        info["numpy"] = np.__version__
    except Exception:
        pass
    return info


# ---------------------------------------------------------------------------
# timers
# ---------------------------------------------------------------------------
class StageTimer:
    """Collects a per-stage millisecond breakdown of one frame.

    ``StageTimer`` is a context manager.  Stage names are registered with
    :meth:`stage`, which is itself a context manager::

        with StageTimer() as t:
            with t.stage('enhance'):
                ...
    """

    __slots__ = ("stages", "_stack")

    def __init__(self) -> None:
        self.stages: Dict[str, float] = {}
        self._stack: List[tuple] = []

    def __enter__(self) -> "StageTimer":
        return self

    def __exit__(self, *exc) -> bool:
        return False

    @contextmanager
    def stage(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.stages[name] = self.stages.get(name, 0.0) + (time.perf_counter() - start) * 1000.0


class LatencyMeter:
    """Accumulates end-to-end (and optionally per-stage) frame latencies."""

    def __init__(self, budget_ms: float = BUDGET_MS) -> None:
        self.budget_ms = float(budget_ms)
        self.samples: List[float] = []
        self.cpu_samples: List[float] = []
        self.stage_samples: Dict[str, List[float]] = {}

    # -- recording ---------------------------------------------------------
    def record(self, latency_ms: float, stages: Optional[Dict[str, float]] = None,
               cpu_ms: Optional[float] = None) -> None:
        self.samples.append(float(latency_ms))
        if cpu_ms is not None:
            self.cpu_samples.append(float(cpu_ms))
        for key, value in (stages or {}).items():
            self.stage_samples.setdefault(key, []).append(float(value))

    def time(self, fn: Callable[..., Any], *args, **kwargs) -> Any:
        """Run ``fn`` and record its wall-clock latency in milliseconds."""
        start = time.perf_counter()
        cstart = time.process_time()
        result = fn(*args, **kwargs)
        self.record((time.perf_counter() - start) * 1000.0,
                    cpu_ms=(time.process_time() - cstart) * 1000.0)
        return result

    def reset(self) -> None:
        self.samples.clear()
        self.cpu_samples.clear()
        self.stage_samples.clear()

    # -- reporting ---------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    @property
    def violations(self) -> List[float]:
        return [s for s in self.samples if s > self.budget_ms]

    def report(self) -> "LatencyReport":
        return LatencyReport.from_samples(self.samples, self.stage_samples, self.budget_ms,
                                          cpu_samples=self.cpu_samples)


@dataclass
class LatencyReport:
    """Summary statistics of a timing run."""

    name: str = "pipeline"
    frames: int = 0
    avg_ms: float = 0.0
    min_ms: float = 0.0
    max_ms: float = 0.0
    median_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    std_ms: float = 0.0
    fps: float = 0.0
    budget_ms: float = BUDGET_MS
    violations: int = 0
    passed: bool = True
    resolution: str = ""
    cpu_avg_ms: float = 0.0
    host_load_pct: Optional[float] = None
    stages_ms: Dict[str, float] = field(default_factory=dict)
    hardware: Dict[str, Any] = field(default_factory=dict)

    # -- construction ------------------------------------------------------
    @classmethod
    def from_samples(
        cls,
        samples: Sequence[float],
        stage_samples: Optional[Dict[str, Sequence[float]]] = None,
        budget_ms: float = BUDGET_MS,
        name: str = "pipeline",
        resolution: str = "",
        hardware: Optional[Dict[str, Any]] = None,
        cpu_samples: Optional[Sequence[float]] = None,
    ) -> "LatencyReport":
        samples = [float(s) for s in samples]
        if not samples:
            raise ValueError("no latency samples recorded")
        ordered = sorted(samples)
        n = len(samples)
        avg = sum(samples) / n

        def _pct(p: float) -> float:
            if n == 1:
                return ordered[0]
            pos = p * (n - 1)
            lo = int(pos)
            hi = min(lo + 1, n - 1)
            frac = pos - lo
            return ordered[lo] * (1 - frac) + ordered[hi] * frac

        stages = {}
        for key, values in (stage_samples or {}).items():
            if len(values):
                stages[key] = sum(float(v) for v in values) / len(values)

        violations = sum(1 for s in samples if s > budget_ms)
        cpu_avg = float(statistics.fmean(cpu_samples)) if cpu_samples else 0.0
        return cls(
            name=name,
            frames=n,
            avg_ms=avg,
            min_ms=ordered[0],
            max_ms=ordered[-1],
            median_ms=_pct(0.5),
            p95_ms=_pct(0.95),
            p99_ms=_pct(0.99),
            std_ms=statistics.pstdev(samples) if n > 1 else 0.0,
            fps=1000.0 / avg if avg > 0 else float("inf"),
            budget_ms=budget_ms,
            violations=violations,
            passed=violations == 0,
            resolution=resolution,
            cpu_avg_ms=cpu_avg,
            stages_ms=stages,
            hardware=hardware or {},
        )

    # -- rendering ---------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def line(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"{self.name:<22} {self.resolution:<11} "
            f"avg {self.avg_ms:7.2f} ms | min {self.min_ms:7.2f} ms | "
            f"max {self.max_ms:7.2f} ms | p95 {self.p95_ms:7.2f} ms | "
            f"{self.fps:6.1f} FPS | budget {self.budget_ms:.0f} ms | {status}"
            + ("" if self.passed else f" ({self.violations} frame(s) over)")
        )

    def table(self) -> str:
        rows = [
            ("Frames measured", f"{self.frames}"),
            ("Average latency", f"{self.avg_ms:.2f} ms"),
            ("Minimum latency", f"{self.min_ms:.2f} ms"),
            ("Maximum latency", f"{self.max_ms:.2f} ms"),
            ("Median latency", f"{self.median_ms:.2f} ms"),
            ("P95 latency", f"{self.p95_ms:.2f} ms"),
            ("P99 latency", f"{self.p99_ms:.2f} ms"),
            ("Std deviation", f"{self.std_ms:.2f} ms"),
            ("Throughput", f"{self.fps:.1f} FPS"),
            ("Latency budget", f"{self.budget_ms:.0f} ms"),
            ("Frames over budget", f"{self.violations}"),
            ("CPU time / frame", f"{self.cpu_avg_ms:.1f} ms (all threads)" if self.cpu_avg_ms else "n/a"),
            ("Host CPU load", f"{self.host_load_pct:.0f} %" if self.host_load_pct is not None else "n/a"),
            ("Result", "PASS" if self.passed else "FAIL"),
        ]
        width = max(len(r[0]) for r in rows)
        body = "\n".join(f"  {k.ljust(width)} : {v}" for k, v in rows)
        stages = ""
        if self.stages_ms:
            stages = "\n  Per-stage breakdown (mean ms)\n" + "\n".join(
                f"    {k.ljust(width)} : {v:6.2f} ms" for k, v in sorted(
                    self.stages_ms.items(), key=lambda kv: -kv[1])
            )
        return f"{self.name} @ {self.resolution}\n{body}{stages}"


# ---------------------------------------------------------------------------
# benchmark driver
# ---------------------------------------------------------------------------
def measure(
    fn: Callable[[Any], Any],
    items: Sequence[Any],
    warmup: int = 5,
    budget_ms: float = BUDGET_MS,
    name: str = "pipeline",
    resolution: str = "",
    collect_stages: bool = True,
    hardware: bool = True,
    frames: Optional[int] = None,
    stages_of: Optional[Callable[[Any, StageTimer], Any]] = None,
) -> LatencyReport:
    """Time ``fn`` once per item and return a :class:`LatencyReport`.

    ``frames`` cycles through ``items`` until that many frames have been timed
    (a cine loop over the available images); the default times one pass over
    every item.  ``warmup`` untimed calls are executed first so that lazily
    initialised state (thread pools, LUTs, kernels) and the file/page cache do
    not pollute the measurement.

    ``stages_of`` optionally takes over the timed call: it receives the item
    and a :class:`StageTimer` whose per-stage timings are recorded.  It is how
    a pipeline that owns its own stages still contributes to the breakdown.
    """
    items = list(items)
    if not items:
        raise ValueError("measure() needs at least one input frame")

    total = len(items) if frames is None else max(1, int(frames))
    load = host_load() if hardware else None
    warm = min(max(int(warmup), 0), total - 1) if total > 1 else 0
    for i in range(warm):
        fn(items[i % len(items)])

    meter = LatencyMeter(budget_ms)
    for i in range(total):
        item = items[i % len(items)]
        timer = StageTimer() if (collect_stages or stages_of is not None) else None
        start = time.perf_counter()
        cstart = time.process_time()
        if stages_of is not None:
            stages_of(item, timer)
        elif timer is not None:
            with timer:
                fn(item)
        else:
            fn(item)
        elapsed = (time.perf_counter() - start) * 1000.0
        cpu = (time.process_time() - cstart) * 1000.0
        meter.record(elapsed, timer.stages if timer is not None else None, cpu_ms=cpu)

    report = LatencyReport.from_samples(
        meter.samples,
        meter.stage_samples,
        budget_ms,
        name=name,
        resolution=resolution,
        hardware=hardware_info() if hardware else {},
        cpu_samples=meter.cpu_samples,
    )
    report.host_load_pct = load
    return report


def enforce(latency: Any, budget_ms: float = BUDGET_MS) -> None:
    """Raise :class:`LatencyBudgetExceeded` when the budget is not met.

    Accepts a single latency in milliseconds or a :class:`LatencyReport`.
    """
    if isinstance(latency, LatencyReport):
        if not latency.passed:
            raise LatencyBudgetExceeded(
                f"{latency.name}: {latency.violations} of {latency.frames} frames exceed the "
                f"{latency.budget_ms:.0f} ms budget "
                f"(avg {latency.avg_ms:.2f} ms, max {latency.max_ms:.2f} ms)"
            )
        return
    if float(latency) > budget_ms:
        raise LatencyBudgetExceeded(
            f"frame latency {float(latency):.2f} ms exceeds the {budget_ms:.0f} ms budget"
        )


def write_report(report: Any, path: Optional[str] = None,
                 json_path: Optional[str] = None) -> List[str]:
    """Persist one or more reports as Markdown and/or JSON.

    ``report`` may be a single :class:`LatencyReport` or a sequence of them.
    Returns the list of paths that were written.
    """
    reports = [report] if isinstance(report, LatencyReport) else list(report)
    if not reports:
        return []
    written: List[str] = []
    for target, as_json in ((path, False), (json_path, True)):
        if not target:
            continue
        folder = os.path.dirname(os.path.abspath(target))
        if folder:
            os.makedirs(folder, exist_ok=True)
        if as_json or not target.lower().endswith(".md"):
            payload = json.dumps([r.to_dict() for r in reports], indent=2)
        else:
            payload = report_to_markdown(reports)
        with open(target, "w", encoding="utf-8") as handle:
            handle.write(payload + "\n")
        written.append(target)
    return written


def report_to_markdown(reports: Sequence[LatencyReport]) -> str:
    """Render one or more reports as a Markdown section."""
    if not reports:
        return ""
    hw = reports[0].hardware or {}
    lines = ["## Latency benchmark", ""]
    lines.append("| Pipeline | Resolution | Frames | Avg (ms) | Min (ms) | Max (ms) | P95 (ms) | FPS | Budget (ms) | Result |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for r in reports:
        lines.append(
            f"| {r.name} | {r.resolution} | {r.frames} | {r.avg_ms:.2f} | {r.min_ms:.2f} | "
            f"{r.max_ms:.2f} | {r.p95_ms:.2f} | {r.fps:.1f} | {r.budget_ms:.0f} | "
            f"{'PASS' if r.passed else 'FAIL'} |"
        )
    lines.append("")
    for r in reports:
        if r.stages_ms:
            lines.append(f"**{r.name} — per-stage mean latency (ms)**")
            lines.append("")
            lines.append("| Stage | Mean (ms) |")
            lines.append("| --- | --- |")
            for k, v in sorted(r.stages_ms.items(), key=lambda kv: -kv[1]):
                lines.append(f"| {k} | {v:.2f} |")
            lines.append("")
    for r in reports:
        if r.cpu_avg_ms:
            lines.append(f"**{r.name} — {r.cpu_avg_ms:.1f} ms of CPU time per frame summed over all "
                         f"threads**")
            lines.append("")
    if hw:
        lines.append("**Hardware / software configuration**")
        lines.append("")
        for key in ("cpu", "logical_cores", "ram_gb", "platform", "python",
                    "torch", "torch_threads", "cuda_available", "opencv",
                    "opencv_threads", "numpy"):
            if key in hw:
                lines.append(f"- {key.replace('_', ' ')}: {hw[key]}")
        lines.append("")
    return "\n".join(lines)


__all__ = [
    "BUDGET_MS",
    "LatencyBudgetExceeded",
    "StageTimer",
    "LatencyMeter",
    "LatencyReport",
    "hardware_info",
    "host_load",
    "measure",
    "enforce",
    "write_report",
    "report_to_markdown",
]
