"""Synthetic coronary angiography dataset generator.

The assignment allows legally usable / synthetic data, and a phantom with
known ground truth is the only way to *measure* background suppression and
vessel preservation quantitatively.  Each sample contains

    cine/frame_XXXX.png      16-bit grayscale frame (uint16, PNG)
    masks/vessel_XXXX.png    coronary tree mask (ground truth)
    masks/ribs_XXXX.png      rib mask
    masks/spine_XXXX.png     spine mask
    masks/lung_XXXX.png      lung field mask
    masks/background_XXXX.png every unwanted structure (ribs | spine | lung)

The anatomy is built from primitives with the same appearance as real
angiography: a bright contrast-filled coronary tree, curved rib bands, a
broad bright spine, dark lung fields, a bright cardiac silhouette, an
illumination gradient, detector vignetting, Poisson/Gaussian noise and
optional low-dose blur.

Usage
-----
    python make_synthetic_dataset.py                 # 40 frames into datasets/synthetic
    python make_synthetic_dataset.py --frames 200 --size 512 --seed 7
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PIL import Image

LEVELS_16 = 4096


# ---------------------------------------------------------------------------
# anatomy primitives
# ---------------------------------------------------------------------------
def _bezier(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, steps: int) -> np.ndarray:
    t = np.linspace(0.0, 1.0, steps, dtype=np.float32)[:, None]
    p0, p1, p2 = p0[None, :], p1[None, :], p2[None, :]
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2


def _draw_tube(canvas: np.ndarray, mask: np.ndarray, points: np.ndarray,
               r0: float, r1: float, value: float) -> None:
    """Draw a tapered tube along ``points`` into ``canvas`` and ``mask``."""
    pts = np.asarray(points, dtype=np.float32)
    if len(pts) < 2:
        return
    for i in range(len(pts) - 1):
        frac = i / max(1, len(pts) - 2)
        radius = r0 + (r1 - r0) * frac
        cv2.line(canvas, tuple(np.round(pts[i]).astype(int)),
                 tuple(np.round(pts[i + 1]).astype(int)), value, max(1, int(round(radius))),
                 cv2.LINE_AA)
        cv2.line(mask, tuple(np.round(pts[i]).astype(int)),
                 tuple(np.round(pts[i + 1]).astype(int)), 255, max(1, int(round(radius))),
                 cv2.LINE_AA)


def _vessel_tree(rng: np.random.Generator, size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Recursive tapering coronary tree rooted near the cardiac apex."""
    canvas = np.zeros((size, size), np.float32)
    mask = np.zeros((size, size), np.uint8)
    depth_limit = 3

    def branch(start, control, end, radius, depth):
        pts = _bezier(np.array(start, np.float32), np.array(control, np.float32),
                      np.array(end, np.float32), 48)
        pts = np.clip(pts, 4, size - 5)
        r_end = max(0.7, radius * rng.uniform(0.55, 0.75))
        _draw_tube(canvas, mask, pts, radius, r_end, 1.0)
        if depth >= depth_limit or radius < 1.2:
            return
        for _ in range(rng.integers(2, 4)):
            idx = int(rng.integers(len(pts) // 3, len(pts) * 2 // 3))
            base = pts[idx]
            direction = base - pts[max(0, idx - 3)]
            norm = float(np.linalg.norm(direction)) or 1.0
            direction = direction / norm
            angle = rng.uniform(0.45, 1.0) * (1 if rng.random() < 0.5 else -1)
            rot = np.array([[math.cos(angle), -math.sin(angle)],
                            [math.sin(angle), math.cos(angle)]], np.float32)
            new_dir = rot @ direction
            length = float(rng.uniform(0.10, 0.22) * size)
            tip = base + new_dir * length + rng.normal(0.0, 5.0, 2).astype(np.float32)
            ctrl = base + new_dir * (length * 0.5) + rng.normal(0.0, 6.0, 2).astype(np.float32)
            branch(base, ctrl, tip, r_end, depth + 1)

    roots = [
        ((0.32, 0.60), (0.34, 0.48), (0.26, 0.34)),
        ((0.56, 0.62), (0.64, 0.52), (0.74, 0.42)),
    ]
    for root, control, tip in roots:
        start = (root[0] * size + rng.normal(0, 8), root[1] * size + rng.normal(0, 8))
        ctrl = (control[0] * size + rng.normal(0, 10), control[1] * size + rng.normal(0, 10))
        end = (tip[0] * size + rng.normal(0, 12), tip[1] * size + rng.normal(0, 12))
        branch(start, ctrl, end, float(rng.uniform(2.0, 2.8)), 0)

    canvas = cv2.GaussianBlur(canvas, (0, 0), 0.7)
    return canvas, mask


def _ribs(rng: np.random.Generator, size: int) -> Tuple[np.ndarray, np.ndarray]:
    canvas = np.zeros((size, size), np.float32)
    mask = np.zeros((size, size), np.uint8)
    slope = rng.uniform(0.35, 0.65)
    for index in range(9):
        y0 = (index - 1.2) * size / 8.0 + rng.normal(0, 6)
        thickness = float(rng.uniform(5.0, 11.0))
        for side in (0, 1):
            x_start = -0.05 * size if side == 0 else 0.55 * size
            x_end = 0.62 * size if side == 0 else 1.05 * size
            p0 = (x_start, y0)
            p2 = (x_end, y0 + slope * (x_end - x_start) + rng.normal(0, 8))
            p1 = ((p0[0] + p2[0]) / 2, (p0[1] + p2[1]) / 2 - size * 0.06)
            pts = _bezier(np.array(p0, np.float32), np.array(p1, np.float32),
                          np.array(p2, np.float32), 64)
            value = float(rng.uniform(0.55, 0.95))
            _draw_tube(canvas, mask, pts, thickness, thickness * 0.85, value)
    return cv2.GaussianBlur(canvas, (0, 0), 1.1), mask


def _spine(size: int) -> Tuple[np.ndarray, np.ndarray]:
    canvas = np.zeros((size, size), np.float32)
    mask = np.zeros((size, size), np.uint8)
    x0 = 0.5 * size
    top = (0.18, 0.86)
    bottom = (0.42, 0.96)
    pts = np.array([[x0 + (top[0] - x0) * (1 - t) + (bottom[0] - x0) * t,
                     top[1] * size + (bottom[1] - top[1]) * size * t] for t in np.linspace(0, 1, 40)])
    cv2.fillPoly(canvas, [np.round(pts).astype(np.int32)], 0.85)
    cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 255)
    canvas = cv2.GaussianBlur(canvas, (0, 0), 6.0)
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
    return canvas, mask


def _lungs(rng: np.random.Generator, size: int) -> Tuple[np.ndarray, np.ndarray]:
    canvas = np.full((size, size), 0.32, np.float32)
    mask = np.zeros((size, size), np.uint8)
    for side in (0, 1):
        centre = (int(0.20 * size if side == 0 else 0.80 * size), int(0.40 * size))
        axes = (int(0.20 * size), int(0.30 * size))
        cv2.ellipse(canvas, centre, axes, float(rng.uniform(-18, 18)), 0, 360, 0.0, -1)
        cv2.ellipse(mask, centre, axes, 0, 0, 360, 255, -1)
    canvas = cv2.GaussianBlur(canvas, (0, 0), 7.0)
    return canvas, mask


def _heart(rng: np.random.Generator, size: int) -> np.ndarray:
    canvas = np.zeros((size, size), np.float32)
    centre = (int(0.52 * size + rng.normal(0, 6)), int(0.55 * size + rng.normal(0, 6)))
    axes = (int(0.24 * size), int(0.30 * size))
    cv2.ellipse(canvas, centre, axes, float(rng.uniform(-15, 15)), 0, 360, 0.30, -1)
    return cv2.GaussianBlur(canvas, (0, 0), 9.0)


# ---------------------------------------------------------------------------
# frame synthesis
# ---------------------------------------------------------------------------
def render_frame(rng: np.random.Generator, size: int = 512, noise: float = 1.0,
                 motion: float = 0.0, low_dose: bool = False) -> Dict[str, np.ndarray]:
    """Render one phantom frame plus its ground-truth masks."""
    heart = _heart(rng, size)
    lung, lung_mask = _lungs(rng, size)
    spine, spine_mask = _spine(size)
    rib, rib_mask = _ribs(rng, size)
    vessel, vessel_mask = _vessel_tree(rng, size)

    if motion > 0.0:
        shift = (int(round(motion * size * 0.03)), int(round(motion * size * 0.02)))
        vessel = np.roll(np.roll(vessel, shift[0], axis=1), shift[1], axis=0)
        vessel_mask = np.roll(np.roll(vessel_mask, shift[0], axis=1), shift[1], axis=0)
        vessel_mask[:shift[1], :] = 0
        vessel_mask[-shift[1]:, :] = 0
        vessel_mask[:, :shift[0]] = 0
        vessel_mask[:, -shift[0]:] = 0

    scene = 0.10 + 0.55 * heart + 0.45 * lung + 0.55 * spine + 0.65 * rib
    scene = scene + 0.95 * vessel

    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    gradient = 0.18 * (xx / size) + 0.10 * (yy / size)
    vignette = 1.0 - 0.35 * (((xx - size / 2) / (size / 2)) ** 2
                             + ((yy - size / 2) / (size / 2)) ** 2) * 0.5
    scene = (scene + gradient) * np.clip(vignette, 0.35, 1.0)
    scene = np.clip(scene, 0.0, 1.6)

    if low_dose:
        scene = cv2.GaussianBlur(scene, (0, 0), 1.1)

    poisson = rng.poisson(np.clip(scene, 0, None) * 900.0) / 900.0
    noisy = poisson + rng.normal(0.0, 0.035 * noise, scene.shape).astype(np.float32)
    noisy = np.clip(noisy, 0.0, None)

    peak = float(np.percentile(noisy, 99.5))
    frame16 = np.clip(noisy / max(peak, 1e-6) * (LEVELS_16 - 1), 0, LEVELS_16 - 1).astype(np.uint16)

    vessel_mask = (vessel_mask > 0).astype(np.uint8) * 255
    vessel_mask = cv2.erode(vessel_mask, np.ones((3, 3), np.uint8))
    background_mask = cv2.bitwise_or(rib_mask, spine_mask)
    background_mask = cv2.bitwise_or(background_mask, cv2.bitwise_and(lung_mask, 255 - vessel_mask))
    return {
        "frame": frame16,
        "vessel": vessel_mask,
        "ribs": rib_mask,
        "spine": spine_mask,
        "lung": lung_mask,
        "background": background_mask,
    }


def build_dataset(root: str, frames: int = 40, size: int = 512, seed: int = 0,
                  noise: float = 1.0, motion: float = 0.0, low_dose: bool = False) -> str:
    """Write ``frames`` phantom frames and masks under ``root``."""
    rng = np.random.default_rng(seed)
    cine = os.path.join(root, "cine")
    masks = os.path.join(root, "masks")
    os.makedirs(cine, exist_ok=True)
    os.makedirs(masks, exist_ok=True)
    for index in range(frames):
        sample = render_frame(rng, size=size, noise=noise,
                              motion=motion if index else 0.0, low_dose=low_dose)
        Image.fromarray(sample["frame"], mode="I;16").save(
            os.path.join(cine, f"frame_{index:04d}.png"))
        for key in ("vessel", "ribs", "spine", "lung", "background"):
            Image.fromarray(sample[key], mode="L").save(
                os.path.join(masks, f"{key}_{index:04d}.png"))
    return root


def load_pairs(root: str) -> List[Tuple[str, str]]:
    cine = os.path.join(root, "cine")
    masks = os.path.join(root, "masks")
    frames = sorted(f for f in os.listdir(cine) if f.endswith(".png"))
    return [(os.path.join(cine, f), os.path.join(masks, f"vessel_{f[6:]}")) for f in frames]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=os.path.join("datasets", "synthetic"))
    parser.add_argument("--frames", type=int, default=40)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--noise", type=float, default=1.0)
    parser.add_argument("--motion", type=float, default=0.5)
    parser.add_argument("--low-dose", action="store_true")
    args = parser.parse_args()
    root = build_dataset(args.root, frames=args.frames, size=args.size, seed=args.seed,
                         noise=args.noise, motion=args.motion, low_dose=args.low_dose)
    total = sum(os.path.getsize(os.path.join(dp, f))
                for dp, _, fs in os.walk(root) for f in fs)
    print(f"wrote {args.frames} frames to {root} ({total / 1024 / 1024:.1f} MiB)")


if __name__ == "__main__":
    main()
