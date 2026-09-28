"""Objective quality evaluation of the real-time pipeline.

Runs the pipeline over a dataset that carries ground-truth masks (the phantom
frames produced by ``make_synthetic_dataset.py``) and reports

  * mask quality        - precision / recall / F1 against the vessel mask,
  * contrast            - CNR = (mean_vessel - mean_local_background)
                                      / std_local_background, for the raw
                          frame and for the enhanced frame (and the gain),
  * structure suppression - the mean level of every background structure
                          (ribs, spine, lung fields) relative to the level of
                          the vessel tree, before and after the pipeline.  A
                          value near 0 means the structure is gone, 1 means it
                          is as bright as the vessels.

    python make_synthetic_dataset.py --frames 40
    python evaluate_quality.py --input datasets/synthetic
    python evaluate_quality.py --input datasets/synthetic --json quality.json
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

from realtime import CoronaryRealtimePipeline

STRUCTURES = ("vessel", "ribs", "spine", "lung")


def _to_u8(image: np.ndarray) -> np.ndarray:
    """Percentile-window a frame to 8 bit so the levels are comparable."""
    arr = np.asarray(image, dtype=np.float32)
    lo, hi = np.percentile(arr, (0.5, 99.5))
    return np.clip((arr - lo) * 255.0 / max(hi - lo, 1e-6), 0, 255).astype(np.uint8)


def load_pairs(root: str) -> List[Tuple[str, str]]:
    """Frame path + vessel-mask path for every frame of a dataset."""
    cine = os.path.join(root, "cine")
    if not os.path.isdir(cine):
        cine = root
    masks = os.path.join(root, "masks") if os.path.isdir(os.path.join(root, "masks")) else root
    pairs = []
    for name in sorted(os.listdir(cine)):
        if not name.lower().endswith((".png", ".tif", ".tiff", ".jpg", ".jpeg")):
            continue
        base = name[6:] if name.startswith("frame_") else name
        pairs.append((os.path.join(cine, name), os.path.join(masks, f"vessel_{base}")))
    return [(f, m) for f, m in pairs if os.path.exists(m)]


def _cnr(image: np.ndarray, mask: np.ndarray) -> float:
    """Contrast-to-noise ratio of ``mask`` against its local surroundings."""
    if not mask.any():
        return 0.0
    local = cv2.GaussianBlur(np.asarray(image, np.float32), (0, 0), 8.0)
    background = local[~mask]
    if background.size == 0:
        return 0.0
    return float(image[mask].mean() - background.mean()) / max(float(background.std()), 1e-6)


def evaluate(pairs: Sequence[Tuple[str, str]], pipeline: CoronaryRealtimePipeline) -> Dict[str, Any]:
    per_frame: List[Dict[str, float]] = []
    for frame_path, mask_path in pairs:
        frame = np.array(Image.open(frame_path))
        base = os.path.basename(frame_path)
        gt = {
            key: np.array(Image.open(
                os.path.join(os.path.dirname(mask_path), f"{key}_{base[6:]}")).convert("L")) > 0
            for key in STRUCTURES if os.path.exists(
                os.path.join(os.path.dirname(mask_path), f"{key}_{base[6:]}"))
        }
        gt["vessel"] = np.array(Image.open(mask_path).convert("L")) > 0

        result = pipeline.process(frame)
        pred = np.asarray(result.segmentation, dtype=np.uint8) > 0
        truth = gt["vessel"]
        tp = int(np.count_nonzero(pred & truth))
        fp = int(np.count_nonzero(pred & ~truth))
        fn = int(np.count_nonzero(~pred & truth))
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)

        raw = _to_u8(frame)
        enhanced = np.asarray(result.enhanced, np.float32)
        processed = np.asarray(result.processed, np.float32)
        cnr_raw = _cnr(raw, truth)
        cnr_out = _cnr(enhanced, truth)
        vessel_level = float(raw[truth].mean()) or 1.0

        row: Dict[str, float] = {
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / max(precision + recall, 1e-9),
            "cnr_raw": cnr_raw,
            "cnr_enhanced": cnr_out,
            "cnr_gain": cnr_out / max(cnr_raw, 1e-6),
            "vessel_survival": float(processed[truth].mean() - np.median(processed)),
            "rib_components_removed": float(result.metrics.get("rib_components_removed", 0.0)),
            "latency_ms": float(result.latency_ms),
            "vessel_area_pct": float(truth.mean() * 100.0),
            "mask_area_pct": float(pred.mean() * 100.0),
        }
        for key, mask in gt.items():
            if not mask.any():
                continue
            level_raw = float(raw[mask].mean()) / vessel_level
            mean_all = float(enhanced.mean())
            mean_all_proc = float(processed.mean())
            row[f"{key}_raw_rel"] = level_raw
            row[f"{key}_processed_rel"] = (float(processed[mask].mean()) - mean_all_proc) / vessel_level
            row[f"{key}_enhanced_rel"] = (float(enhanced[mask].mean()) - mean_all) / vessel_level
            row[f"{key}_mask_frac"] = float(pred[mask].mean())
        per_frame.append(row)

    keys = per_frame[0].keys()
    return {key: float(np.mean([row[key] for row in per_frame])) for key in keys}


def format_report(results: Dict[str, Any], frames: int) -> str:
    lines = [f"frames evaluated: {frames}", "",
             f"mask precision {results['precision']:.3f} | recall {results['recall']:.3f} | "
             f"F1 {results['f1']:.3f}",
             f"vessel area {results['vessel_area_pct']:.2f}% | mask area {results['mask_area_pct']:.2f}%",
             f"CNR raw {results['cnr_raw']:.2f} -> enhanced {results['cnr_enhanced']:.2f} "
             f"({results['cnr_gain']:.2f}x)",
             f"vessel signal surviving background subtraction: {results['vessel_survival']:.1f} levels",
             f"mean latency {results['latency_ms']:.1f} ms", "",
             "structure level relative to the vessel tree (1.0 = as bright as the vessels, "
             "0.0 = removed)",
             "| structure | raw | background suppressed | enhanced | in mask |",
             "| --- | --- | --- | --- | --- |"]
    for key in STRUCTURES:
        lines.append(
            f"| {key} | {results.get(f'{key}_raw_rel', 0.0):.2f} | "
            f"{results.get(f'{key}_processed_rel', 0.0):.2f} | "
            f"{results.get(f'{key}_enhanced_rel', 0.0):.2f} | "
            f"{results.get(f'{key}_mask_frac', 0.0) * 100.0:.1f}% |"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=os.path.join("datasets", "synthetic"))
    parser.add_argument("--limit", type=int, default=0, help="evaluate only the first N frames")
    parser.add_argument("--json", dest="json_path", default="", help="write the metrics as JSON")
    parser.add_argument("--set", dest="overrides", default="",
                        help="pipeline overrides, e.g. 'vessel_size=256,vessel_thresh=0.2'")
    args = parser.parse_args()

    pairs = load_pairs(args.input)
    if args.limit:
        pairs = pairs[:args.limit]
    if not pairs:
        raise SystemExit(f"no frames with ground-truth masks found under {args.input}")

    overrides: Dict[str, Any] = {}
    for item in filter(None, args.overrides.split(",")):
        key, _, value = item.partition("=")
        try:
            overrides[key] = int(value) if value.lstrip("-").isdigit() else float(value)
        except ValueError:
            overrides[key] = {"true": True, "false": False}.get(value.lower(), value)

    results = evaluate(pairs, CoronaryRealtimePipeline(**overrides))
    print(f"dataset: {args.input} ({len(pairs)} frames)\n")
    print(format_report(results, len(pairs)))
    if args.json_path:
        with open(args.json_path, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
        print(f"\nwrote {args.json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
