"""Latency benchmark for the real-time coronary pipeline.

Measures wall-clock latency per frame (input image in, all outputs rendered)
and reports average / minimum / maximum / median / p95 latency, the resulting
throughput in frames per second, the per-stage breakdown and the hardware the
numbers were taken on, so the 36 ms budget can be verified on a given machine.

    python benchmark.py                                    # example/ images
    python benchmark.py --input datasets/synthetic/cine --frames 200
    python benchmark.py --include-deepsa                   # also time the original path
    python benchmark.py --report latency_report.md --json latency_report.json

The original DeepSA path is only used for reference (``--include-deepsa``): it
needs several seconds per frame on CPU and is therefore timed on one frame.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from PIL import Image

from latency import BUDGET_MS, LatencyBudgetExceeded, enforce, measure, write_report
from realtime import CoronaryRealtimePipeline

IMAGE_SUFFIXES = (".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp", ".webp")


def collect_inputs(path: str) -> List[Any]:
    """Load every image under ``path`` (a file or a directory) as a PIL image."""
    if os.path.isfile(path):
        return [Image.open(path)]
    files = []
    for root, _, names in os.walk(path):
        for name in sorted(names):
            if name.lower().endswith(IMAGE_SUFFIXES):
                files.append(os.path.join(root, name))
    if not files:
        raise SystemExit(f"no images found under {path}")
    return [Image.open(f) for f in files]


def _resolution(frames: Sequence[Any]) -> str:
    sizes = {frame.size for frame in frames}
    if len(sizes) == 1:
        width, height = sizes.pop()
        return f"{width}x{height}"
    return f"{len(sizes)} sizes"


def time_deepsa(frames: Sequence[Any]) -> Optional[Dict[str, Any]]:
    """Reference timing of the original model path (one frame, best of two)."""
    try:
        import torch  # noqa: F401

        import demo
    except Exception as error:  # pragma: no cover - optional dependency path
        return {"error": f"unavailable: {error}"}

    samples = []
    for _ in range(2):
        start = time.perf_counter()
        demo.predict(frames[0])
        samples.append((time.perf_counter() - start) * 1000.0)
    return {
        "name": "DeepSA (original predict())",
        "frames": 1,
        "min_ms": min(samples),
        "avg_ms": sum(samples) / len(samples),
        "max_ms": max(samples),
        "fps": 1000.0 / (sum(samples) / len(samples)),
    }


def _run(pipeline: CoronaryRealtimePipeline, frame: Any, timer: StageTimer) -> Any:
    return pipeline.process(frame, timer=timer).comparison


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="example", help="image file or directory")
    parser.add_argument("--frames", type=int, default=120, help="number of timed frames")
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--budget", type=float, default=BUDGET_MS, help="per-frame budget in ms")
    parser.add_argument("--threads", type=int, default=None, help="cv2.setNumThreads value")
    parser.add_argument("--resolution", type=int, default=512,
                        help="longest side used for the timed frames (0 = native size)")
    parser.add_argument("--report", default="", help="write a markdown report to this path")
    parser.add_argument("--json", dest="json_path", default="", help="write raw results as JSON")
    parser.add_argument("--include-deepsa", action="store_true",
                        help="also time the original DeepSA path for reference")
    parser.add_argument("--save-dir", default="", help="write the outputs of the first frame here")
    parser.add_argument("--strict", action="store_true",
                        help="exit non-zero when the budget is exceeded")
    args = parser.parse_args()

    frames = collect_inputs(args.input)
    if args.resolution:
        size = args.resolution
        frames = [frame.resize((size, size), Image.BILINEAR) if max(frame.size) != size
                  else frame for frame in frames]
    pipeline = CoronaryRealtimePipeline(budget_ms=args.budget, cv_threads=args.threads)

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        first = pipeline.process(frames[0])
        for name in ("raw", "processed", "enhanced", "segmentation", "comparison"):
            getattr(first, name).save(os.path.join(args.save_dir, f"{name}.png"))
        if first.enhanced_16bit is not None:
            first.enhanced_16bit.save(os.path.join(args.save_dir, "enhanced_16bit.png"))
        print(f"wrote sample outputs to {args.save_dir}")

    print(f"benchmarking {args.frames} frames from {args.input} "
          f"at {_resolution(frames)} against a {args.budget:.0f} ms budget")
    report = measure(
        lambda frame: pipeline.process(frame).comparison,
        frames,
        warmup=args.warmup,
        budget_ms=args.budget,
        name="realtime",
        resolution=_resolution(frames),
        collect_stages=False,
        frames=args.frames,
        stages_of=lambda frame, timer: _run(pipeline, frame, timer),
    )
    print(report.table())

    reports = [report]
    if args.include_deepsa:
        deepsa = time_deepsa(frames)
        if deepsa:
            print()
            print(f"reference: {deepsa['name']}: {deepsa['avg_ms']:.0f} ms/frame "
                  f"({deepsa['fps']:.2f} FPS) -> {deepsa['avg_ms'] / args.budget:.0f}x the budget")

    if args.report or args.json_path:
        written = write_report(reports, args.report or None, args.json_path or None)
        for path in written:
            print(f"wrote {path}")

    if args.strict:
        try:
            enforce(report)
        except LatencyBudgetExceeded as error:
            print(str(error), file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
