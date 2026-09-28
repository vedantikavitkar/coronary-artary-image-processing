"""Real-time coronary pipeline (<= 36 ms/frame on CPU).

This module is *additive*: ``demo.py`` / ``utils.py`` keep the original
DeepSA subtraction model untouched.  It provides a purely OpenCV/NumPy
pipeline that runs the same conceptual stages fast enough for cine loops:

    1. ingest        - 16-bit safe normalisation (uint16 / float / uint8 / PIL)
    2. preprocess     - median denoise + CLAHE contrast enhancement
    3. background     - dual-scale (fine + coarse) background estimation and
                        subtraction, which removes the slowly varying anatomy:
                        ribs, spine, lung fields, diaphragm, collaterals-free
                        soft tissue
    4. vesselness     - multi-scale Frangi ridge filter (Hessian based) on the
                        background-subtracted residual
    5. cleanup        - area filter + straight-line/rib rejection on connected
                        components, leaving the coronary tree
    6. present        - background-suppressed frame, enhanced coronary frame,
                        binary mask and a raw|processed|enhanced comparison
                        strip

Typical measured cost on an Intel Core i5-1155G7 (4C/8T, no GPU) is
~10-15 ms per 512x512 frame, i.e. comfortably inside the 36 ms budget.
Use ``benchmark.py`` to reproduce the numbers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

from latency import BUDGET_MS, StageTimer

__all__ = ["CoronaryRealtimePipeline", "PipelineResult", "BUDGET_MS"]


# ---------------------------------------------------------------------------
# result container
# ---------------------------------------------------------------------------
@dataclass
class PipelineResult:
    """Everything the pipeline produced for one frame, plus its timing."""

    raw: Image.Image
    processed: Image.Image
    enhanced: Image.Image
    segmentation: Image.Image
    comparison: Image.Image
    latency_ms: float = 0.0
    budget_ms: float = BUDGET_MS
    input_bit_depth: int = 8
    output_bit_depth: int = 8
    enhanced_16bit: Optional[Image.Image] = None
    stages_ms: Dict[str, float] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def within_budget(self) -> bool:
        return self.latency_ms <= self.budget_ms


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _as_array(frame: Any) -> np.ndarray:
    """Return a 2-D numpy view of a PIL image / array / path, bit depth kept."""
    if isinstance(frame, Image.Image):
        mode = frame.mode
        if mode in ("I;16", "I;16B", "I;16L", "I;16N", "F", "L", "I"):
            return np.array(frame)
        if mode in ("RGB", "RGBA", "BGR;4", "BGRA"):
            arr = np.array(frame)
            code = cv2.COLOR_RGB2GRAY if mode == "RGB" else cv2.COLOR_RGBA2GRAY
            return cv2.cvtColor(arr, code)
        return np.array(frame.convert("L"))
    if isinstance(frame, (str, bytes)) or hasattr(frame, "__fspath__"):
        return _as_array(Image.open(frame))
    arr = np.asarray(frame)
    if arr.ndim == 3:
        arr = arr[..., 0] if arr.shape[2] == 1 else cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2-D frame, got shape {arr.shape}")
    return arr


def _normalise_to_uint8(arr: np.ndarray, low_pct: float, high_pct: float) -> Tuple[np.ndarray, float, float]:
    """Window the dynamic range to 8 bit.

    16-bit cine frames use only part of the 0..65535 code range; a fixed
    rescale would crush the contrast, so the window is taken from
    percentiles of a strided subsample (cheap: ~5% of the pixels).
    """
    if arr.dtype == np.uint8:
        return arr, 0.0, 255.0
    if arr.dtype == bool:
        return arr.astype(np.uint8) * 255, 0.0, 255.0
    values = arr.astype(np.float32, copy=False)
    step = max(1, int(np.sqrt(values.size / 16000.0)))
    lo, hi = np.percentile(values.reshape(-1)[::step], (low_pct, high_pct))
    span = float(hi - lo)
    if span < 1e-6:
        return np.zeros(values.shape, np.uint8), float(lo), float(lo) + 1.0
    values -= lo
    values *= 255.0 / span
    np.clip(values, 0.0, 255.0, out=values)
    return values.astype(np.uint8), float(lo), float(hi)


def _stretch_lut(img: np.ndarray, low_pct: float, high_pct: float, pre: Optional[np.ndarray] = None) -> np.ndarray:
    """256-entry intensity-stretch LUT derived from the image histogram.

    Histogram based percentiles cost ~0.3 ms on 512x512, while
    ``np.percentile`` over a float image costs several milliseconds, and the
    result is applied to the whole frame with a single ``cv2.LUT``.  When
    ``pre`` is given (a gamma LUT) the two tables are composed so that the
    frame still only needs one lookup.
    """
    hist = cv2.calcHist([img], [0], None, [256], [0, 256]).ravel()
    total = float(hist.sum())
    if total <= 0:
        lut = np.arange(256, dtype=np.uint8)
        return lut if pre is None else pre[lut]
    cumulative = np.cumsum(hist)
    lo = int(np.searchsorted(cumulative, total * (low_pct / 100.0)))
    hi = int(np.searchsorted(cumulative, total * (high_pct / 100.0)))
    if hi <= lo:
        lo, hi = 0, 255
    scale = 255.0 / float(hi - lo)
    lut = (np.arange(256, dtype=np.float32) - lo) * scale
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    return lut if pre is None else pre[lut]



def _odd(value: int) -> int:
    value = int(value)
    return value if value % 2 == 1 else value + 1


def _gaussian_derivative(sigma: float, order: int) -> np.ndarray:
    """1-D Gaussian derivative kernel, normalised like Frangi's filters."""
    radius = max(1, int(np.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    g = np.exp(-(x * x) / (2.0 * sigma * sigma))
    if order == 0:
        kernel = g / g.sum()
    elif order == 1:
        kernel = (-x / sigma ** 2) * g
        norm = -np.sum(x * kernel)
        kernel = kernel / norm if abs(norm) > 1e-12 else kernel
    else:
        kernel = ((x * x - sigma ** 2) / sigma ** 4) * g
        norm = np.sum(x * x * kernel)
        kernel = kernel / norm if abs(norm) > 1e-12 else kernel
    return kernel.astype(np.float32)


def _vesselness(
    response: np.ndarray,
    sigmas: Sequence[float],
    beta: float = 0.5,
    border: int = cv2.BORDER_REPLICATE,
) -> np.ndarray:
    """Multi-scale Frangi vesselness (bright ridges) in [0, 1].

    ``R_b`` (anisotropy) suppresses blobs and step edges, the ``S^2`` term
    suppresses low-contrast noise, and the per-scale ``c`` is taken from the
    image statistics so the response always uses its full dynamic range.

    The eigenvalue bookkeeping is done with three identities instead of an
    explicit sort, which removes the two ``np.where`` copies per scale::

        delta = |mu1 - mu2|                  (eigenvalue separation)
        S     = |mu1| + |mu2| = max(|dxx+dyy|, delta)
        R_b   = (S - delta) / (S + delta)     (min/max, both magnitudes)
        S2    = mu1^2 + mu2^2 = dxx^2 + dyy^2 + 2*dxy^2
        bright ridge  <=>  mu1 < 0  <=>  dxx + dyy < -delta
    """
    best = np.zeros_like(response)
    two_beta_sq = 2.0 * beta * beta
    for sigma in sigmas:
        smooth = cv2.GaussianBlur(response, (0, 0), sigma, borderType=border)
        g0 = _gaussian_derivative(sigma, 0)
        g1 = _gaussian_derivative(sigma, 1)
        g2 = _gaussian_derivative(sigma, 2)
        dxx = cv2.sepFilter2D(smooth, -1, g2, g0, borderType=border)
        dyy = cv2.sepFilter2D(smooth, -1, g0, g2, borderType=border)
        dxy = cv2.sepFilter2D(smooth, -1, g1, g1, borderType=border)

        trace = cv2.add(dxx, dyy)
        gap = cv2.subtract(dxx, dyy)
        quad = cv2.add(cv2.multiply(gap, gap), cv2.multiply(dxy, dxy, scale=4.0))
        delta = cv2.sqrt(quad)
        total = cv2.max(cv2.absdiff(trace, 0.0), delta)

        ratio = cv2.divide(cv2.subtract(total, delta), cv2.add(total, delta))
        norm = np.exp(-cv2.multiply(ratio, ratio, scale=1.0 / two_beta_sq))

        peak = float(np.quantile(np.abs(trace).reshape(-1)[::3], 0.995))
        c = 0.35 * peak
        if c < 1e-9:
            continue
        s2 = cv2.add(cv2.add(cv2.multiply(dxx, dxx), cv2.multiply(dyy, dyy)),
                     cv2.multiply(dxy, dxy, scale=2.0))
        strength = cv2.subtract(1.0, cv2.exp(-cv2.multiply(s2, 1.0 / (2.0 * c * c))))
        np.multiply(norm, strength, out=norm)
        norm *= (trace < -delta)
        np.maximum(best, norm, out=best)
    return best


def _component_moments(labels: np.ndarray, count: int) -> Dict[str, np.ndarray]:
    """Area, centroid and second moments for every labelled component."""
    ys, xs = np.nonzero(labels)
    flat = labels[ys, xs].astype(np.int32)
    area = np.bincount(flat, minlength=count).astype(np.float64)
    sx = np.bincount(flat, weights=xs.astype(np.float64), minlength=count)
    sy = np.bincount(flat, weights=ys.astype(np.float64), minlength=count)
    sxx = np.bincount(flat, weights=(xs * xs).astype(np.float64), minlength=count)
    syy = np.bincount(flat, weights=(ys * ys).astype(np.float64), minlength=count)
    sxy = np.bincount(flat, weights=(xs * ys).astype(np.float64), minlength=count)
    with np.errstate(divide="ignore", invalid="ignore"):
        cx = sx / area
        cy = sy / area
        mxx = sxx / area - cx * cx
        myy = syy / area - cy * cy
        mxy = sxy / area - cx * cy
    return {"area": area, "cx": cx, "cy": cy, "mxx": mxx, "myy": myy, "mxy": mxy}


def _rib_components(cx: np.ndarray, cy: np.ndarray, area: np.ndarray,
                    mxx: np.ndarray, myy: np.ndarray, mxy: np.ndarray,
                    shape: Tuple[int, int],
                    min_length_ratio: float = 0.30,
                    max_thickness_ratio: float = 0.05,
                    max_ratio: float = 0.22,
                    border_px: int = 3,
                    require_border: bool = True,
                    min_area: float = 12.0) -> np.ndarray:
    """Names of components that look like a rib: long, thin, straight, border.

    Coronary arteries are curved and branching, so their second-moment fit is
    never a thin long strip, while a rib is close to a straight band that runs
    across the frame.  The mask is morphologically closed before this test, so
    a rib appears as one elongated component instead of a chain of blobs.
    """
    h, w = shape
    max_side = float(max(h, w))
    min_length = min_length_ratio * max_side
    max_thickness = max_thickness_ratio * max_side
    with np.errstate(divide="ignore", invalid="ignore"):
        common = np.sqrt(np.abs(4.0 * mxx * myy - mxy * mxy)) / 2.0
        diff = (mxx - myy) / 2.0
        lam1 = diff + common
        lam2 = diff - common
        ratio = np.sqrt(np.maximum(lam2, 0.0) / np.maximum(lam1, 1e-9))
        length = 4.0 * np.sqrt(np.maximum(lam1, 0.0))
        thickness = 4.0 * np.sqrt(np.maximum(lam2, 0.0))
    touches = (cx <= border_px) | (cy <= border_px) | (cx >= w - 1 - border_px) | (cy >= h - 1 - border_px)
    is_rib = (
        (length > min_length)
        & (thickness < max_thickness)
        & (ratio < max_ratio)
    )
    if require_border:
        is_rib &= touches
    idx = np.nonzero(is_rib & (area >= min_area))[0]
    return idx[idx > 0].astype(np.int32)


# ---------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------
class CoronaryRealtimePipeline:
    """Background suppression + coronary enhancement within the 36 ms budget.

    Parameters mirror the processing stages; defaults are tuned for 512x512
    coronary angiograms and are resolution independent (all kernel sizes and
    scale factors are expressed relative to the frame size).
    """

    def __init__(
        self,
        vessel_size: int = 192,
        sigmas: Sequence[float] = (1.0, 2.0),
        beta: float = 0.5,
        denoise_ksize: int = 3,
        clahe_clip: float = 2.5,
        clahe_grid: int = 8,
        bg_fine: int = 6,
        bg_coarse: int = 26,
        bg_size: int = 192,
        rib_line_ratio: float = 0.12,
        vessel_thresh: float = 0.20,
        min_component_area: int = 6,
        close_px: int = 5,
        suppress_ribs: bool = True,
        rib_margin_px: int = 2,
        rib_min_length_ratio: float = 0.30,
        rib_max_thickness_ratio: float = 0.06,
        rib_max_ratio: float = 0.25,
        rib_require_border: bool = True,
        unsharp_amount: float = 1.7,
        gamma: float = 1.8,
        gain_base: float = 0.45,
        gain_scale: float = 1.8,
        mask_base: float = 0.6,
        mask_gain: float = 0.7,
        output_smooth: float = 0.0,
        make_comparison: bool = True,
        comparison_max_height: int = 320,
        keep_16bit_output: bool = True,
        budget_ms: float = BUDGET_MS,
        cv_threads: Optional[int] = None,
        warmup: bool = True,
    ) -> None:
        self.vessel_size = int(vessel_size)
        self.sigmas = tuple(float(s) for s in sigmas)
        self.beta = float(beta)
        self.denoise_ksize = _odd(denoise_ksize)
        self.clahe_clip = float(clahe_clip)
        self.clahe_grid = int(clahe_grid)
        self.bg_fine = max(2, int(bg_fine))
        self.bg_coarse = max(self.bg_fine + 2, int(bg_coarse))
        self.bg_size = int(bg_size)
        self.rib_line_ratio = float(rib_line_ratio)
        self.vessel_thresh = float(vessel_thresh)
        self.min_component_area = int(min_component_area)
        self.close_px = int(close_px)
        self.suppress_ribs = bool(suppress_ribs)
        self.rib_margin_px = int(rib_margin_px)
        self.rib_min_length_ratio = float(rib_min_length_ratio)
        self.rib_max_thickness_ratio = float(rib_max_thickness_ratio)
        self.rib_max_ratio = float(rib_max_ratio)
        self.rib_require_border = bool(rib_require_border)
        self.unsharp_amount = float(unsharp_amount)
        self.gamma = float(gamma)
        self.gain_base = float(gain_base) * 255.0
        self.gain_scale = float(gain_scale)
        self.mask_base = float(mask_base) * 255.0
        self.mask_gain = float(mask_gain)
        self.output_smooth = float(output_smooth)
        self.make_comparison = bool(make_comparison)
        self.comparison_max_height = int(comparison_max_height)
        self.keep_16bit_output = bool(keep_16bit_output)
        self.budget_ms = float(budget_ms)
        if cv_threads is not None:
            cv2.setNumThreads(int(cv_threads))
        self._clahe = cv2.createCLAHE(clipLimit=self.clahe_clip,
                                      tileGridSize=(self.clahe_grid, self.clahe_grid))
        self._gamma_lut = self._build_gamma_lut(self.gamma)
        self._close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_odd(self.close_px),) * 2)
        self._seg_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self._rotations: Dict[Tuple[float, Tuple[int, int]], np.ndarray] = {}
        self._header: Optional[np.ndarray] = None
        if warmup:
            self.warmup()

    def warmup(self, size: int = 192, repeats: int = 2) -> None:
        """Pay the one-off costs (thread pool spin-up, LUT/kernel setup, first
        touch of every buffer) before the first real frame is timed, so that the
        first user-visible frame is already within budget.
        """
        frame = np.linspace(0, 4000, size * size).reshape(size, size).astype(np.uint16)
        for _ in range(max(1, repeats)):
            self.process(frame)

    # -- lookup tables -----------------------------------------------------
    @staticmethod
    def _build_gamma_lut(gamma: float) -> np.ndarray:
        x = np.arange(256, dtype=np.float32) / 255.0
        lut = np.clip(np.power(x, gamma) * 255.0 + 0.5, 0, 255).astype(np.uint8)
        return lut

    @staticmethod
    def _to_pil(arr: np.ndarray) -> Image.Image:
        arr = np.ascontiguousarray(arr)
        if arr.ndim == 3:
            if arr.shape[2] == 4:
                return Image.fromarray(arr, mode="RGBA")
            return Image.fromarray(arr, mode="RGB")
        if arr.dtype == np.uint16:
            return Image.fromarray(arr, mode="I;16")
        return Image.fromarray(arr, mode="L")

    # -- stages ------------------------------------------------------------
    def _rotation(self, angle: float, shape: Tuple[int, int]) -> np.ndarray:
        key = (angle, shape)
        cached = self._rotations.get(key)
        if cached is None:
            h, w = shape
            cached = cv2.getRotationMatrix2D(((w - 1) * 0.5, (h - 1) * 0.5), angle, 1.0)
            self._rotations[key] = cached
        return cached

    def _line_background(self, small: np.ndarray) -> np.ndarray:
        """Illumination of long straight bright bands (ribs, straight edges).

        A morphological *opening* with a line structuring element only keeps a
        structure that contains a straight segment of at least that length, so
        the max over the four principal orientations estimates the rib
        illumination: a rib keeps its full value, while a curved coronary
        branch - which contains no such segment - stays at zero and therefore
        survives the subtraction.  The four line elements are evaluated in
        rotated copies of the small analysis image, where a plain 1xN
        rectangle can be used, and every operation costs a few hundredths of a
        millisecond at this resolution.
        """
        h, w = small.shape
        length = max(3, int(round(self.rib_line_ratio * max(h, w))))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, length))
        best = small
        for angle in (0.0, 45.0, 90.0, 135.0):
            if angle == 0.0:
                opened = cv2.morphologyEx(small, cv2.MORPH_OPEN, kernel)
            elif angle == 90.0:
                opened = cv2.transpose(cv2.morphologyEx(cv2.transpose(small),
                                                        cv2.MORPH_OPEN, kernel))
            else:
                rotated = cv2.warpAffine(small, self._rotation(angle, (h, w)), (w, h),
                                         flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
                opened = cv2.warpAffine(
                    cv2.morphologyEx(rotated, cv2.MORPH_OPEN, kernel),
                    self._rotation(180.0 - angle, (h, w)), (w, h),
                    flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
            best = cv2.max(best, opened)
        return best

    def _background(self, gray: np.ndarray) -> np.ndarray:
        """Background estimate (8-bit in, 8-bit out).

        Three estimates are fused in a small analysis domain, so only one
        full-resolution resample is needed:

        * a fine scale that absorbs ribs and other thin bright bands,
        * a coarse scale that absorbs the spine, lung fields and diaphragm,
          i.e. everything that varies slowly compared with a vessel,
        * an oriented line background that absorbs long straight ridges,
          which is what actually distinguishes a rib from a curved artery.
        """
        h, w = gray.shape[:2]
        ratio = self.bg_size / float(max(h, w))
        aw, ah = max(8, int(round(w * ratio))), max(8, int(round(h * ratio)))
        small = cv2.resize(gray, (aw, ah), interpolation=cv2.INTER_AREA) if ratio < 1.0 else gray

        fw, fh = max(2, aw // self.bg_fine), max(2, ah // self.bg_fine)
        cw, ch = max(2, aw // self.bg_coarse), max(2, ah // self.bg_coarse)
        fine = cv2.GaussianBlur(cv2.resize(small, (fw, fh), interpolation=cv2.INTER_AREA),
                                (0, 0), 1.5)
        coarse = cv2.GaussianBlur(cv2.resize(small, (cw, ch), interpolation=cv2.INTER_AREA),
                                  (0, 0), 2.5)
        coarse = cv2.resize(coarse, (fw, fh), interpolation=cv2.INTER_LINEAR)
        mixed = cv2.addWeighted(fine, 0.65, coarse, 0.35, 0.0)
        mixed = cv2.resize(mixed, (aw, ah), interpolation=cv2.INTER_LINEAR)

        if self.rib_line_ratio > 0.0:
            mixed = cv2.max(mixed, self._line_background(small))
        if ratio >= 1.0:
            return mixed
        return cv2.resize(mixed, (w, h), interpolation=cv2.INTER_LINEAR)

    def _vessel_mask(self, residual: np.ndarray, vesselness: np.ndarray) -> Tuple[np.ndarray, int, int]:
        """Threshold the vesselness, drop specks and rib-shaped components.

        The thresholded map is closed with a small elliptical element first so
        that a rib - which the ridge filter breaks into a chain of blobs -
        becomes a single elongated component that the moment test can reject,
        while a coronary branch stays short and compact.
        """
        bright = cv2.compare(vesselness, self.vessel_thresh, cv2.CMP_GE)
        positive = cv2.compare(residual, 0.0, cv2.CMP_GT)
        mask = cv2.bitwise_and(bright, positive)
        if self.close_px > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._close_kernel)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        if count <= 1:
            return mask, 0, 0
        moments = _component_moments(labels, count)
        removed = 0
        drop = np.zeros(count, dtype=bool)
        drop[1:] = moments["area"][1:] < self.min_component_area
        if self.suppress_ribs:
            ribs = _rib_components(
                moments["cx"],
                moments["cy"],
                moments["area"],
                moments["mxx"],
                moments["myy"],
                moments["mxy"],
                residual.shape,
                min_length_ratio=self.rib_min_length_ratio,
                max_thickness_ratio=self.rib_max_thickness_ratio,
                max_ratio=self.rib_max_ratio,
                border_px=self.rib_margin_px,
                require_border=self.rib_require_border,
            )
            drop[ribs] = True
            removed = int(ribs.size)
        if drop.any():
            keep = np.where(drop[labels], 0, labels)
            mask = (keep > 0).astype(np.uint8)
        return mask, removed, count - 1

    def _comparison(self, raw: np.ndarray, processed: np.ndarray, enhanced: np.ndarray) -> np.ndarray:
        """Build the ``original | suppressed | enhanced`` strip.

        The preview is reduced by an *integer* factor, which lets OpenCV use
        the fast box path of ``INTER_AREA`` (a 512 -> 320 resample costs five
        times as much as 512 -> 256), and the label bar is rasterised once and
        copied in, so only the three panels are touched per frame.
        """
        h, w = raw.shape[:2]
        bar = 6
        pad = 26
        panels = (raw, processed, enhanced)
        if self.comparison_max_height and h > self.comparison_max_height:
            factor = max(1, int(round(h / float(self.comparison_max_height))))
            nw, nh = max(1, w // factor), max(1, h // factor)
            if (nw, nh) != (w, h):
                panels = tuple(cv2.resize(panel, (nw, nh), interpolation=cv2.INTER_AREA)
                               for panel in panels)
            h, w = nh, nw
        shape = (h + pad, 3 * w + 2 * bar, 3)
        canvas = np.empty(shape, np.uint8)
        canvas[:] = 12
        for index, panel in enumerate(panels):
            x0 = index * (w + bar)
            cv2.cvtColor(panel, cv2.COLOR_GRAY2BGR, dst=canvas[pad:, x0:x0 + w, :])
        if self._header is None or self._header.shape != (pad, shape[1], 3):
            self._header = self._label_bar(pad, shape[1], w, bar)
        canvas[:pad] = self._header
        return canvas

    def _label_bar(self, pad: int, width: int, panel_w: int, bar: int) -> np.ndarray:
        header = np.empty((pad, width, 3), np.uint8)
        header[:] = 12
        font = cv2.FONT_HERSHEY_SIMPLEX
        for text, x in (("ORIGINAL / RAW", 4),
                        ("BACKGROUND SUPPRESSED", panel_w + bar + 4),
                        ("CORONARY ENHANCED", 2 * (panel_w + bar) + 4)):
            cv2.putText(header, text, (x, 18), font, 0.45, (210, 214, 220), 1, cv2.LINE_8)
        return header

    # -- public API --------------------------------------------------------
    def process(self, frame: Any, timer: Optional[StageTimer] = None) -> PipelineResult:
        """Run the full pipeline for one frame and return all outputs."""
        timer = timer or StageTimer()

        with timer.stage("ingest"):
            arr = _as_array(frame)
            bit_depth = 16 if arr.dtype == np.uint16 else (8 if arr.dtype == np.uint8 else 32)
            gray8, lo, hi = _normalise_to_uint8(arr, 0.5, 99.5)
            h, w = gray8.shape[:2]

        with timer.stage("preprocess"):
            denoised = cv2.medianBlur(gray8, self.denoise_ksize) if self.denoise_ksize > 1 else gray8
            equalised = self._clahe.apply(denoised)
            base = equalised.astype(np.float32)

        with timer.stage("background_subtraction"):
            background = self._background(equalised)
            residual = cv2.subtract(base, background, dtype=cv2.CV_32F)

        with timer.stage("vessel_enhancement"):
            scale = min(1.0, self.vessel_size / float(max(h, w)))
            if scale < 1.0:
                nw, nh = max(32, int(round(w * scale))), max(32, int(round(h * scale)))
                small = cv2.resize(residual, (nw, nh), interpolation=cv2.INTER_AREA)
            else:
                nw, nh = w, h
                small = residual
            peak = max(float(small.max()), -float(small.min()))
            norm = small * (1.0 / peak) if peak > 1e-6 else small
            sigma_scale = min(1.0, self.vessel_size / 512.0)
            sigmas = tuple(s * max(0.75, sigma_scale) for s in self.sigmas)
            vesselness = _vesselness(norm, sigmas, beta=self.beta)
            top = float(np.quantile(vesselness.reshape(-1)[::2], 0.995))
            if top > 1e-6:
                vesselness *= 1.0 / top
            np.clip(vesselness, 0.0, 1.0, out=vesselness)

        with timer.stage("rib_vessel_cleanup"):
            mask_small, ribs_removed, components = self._vessel_mask(norm, vesselness)
            if (nw, nh) != (w, h):
                vesselness_full = cv2.resize(vesselness, (w, h), interpolation=cv2.INTER_LINEAR)
                mask_full = cv2.resize(mask_small, (w, h), interpolation=cv2.INTER_NEAREST)
            else:
                vesselness_full = vesselness
                mask_full = mask_small

        with timer.stage("output"):
            v8 = cv2.convertScaleAbs(vesselness_full, alpha=self.gain_scale, beta=self.gain_base)
            m8 = cv2.convertScaleAbs(mask_full, alpha=self.mask_gain, beta=self.mask_base)
            blend8 = cv2.multiply(v8, m8, scale=1.0 / 255.0)

            processed_u8 = cv2.convertScaleAbs(residual, alpha=self.unsharp_amount, beta=128.0)
            processed8 = cv2.LUT(processed_u8,
                                 _stretch_lut(processed_u8, 1.0, 99.0, self._gamma_lut))

            enhanced = cv2.multiply(base, blend8, scale=1.0 / 255.0, dtype=cv2.CV_32F)
            enhanced = cv2.add(enhanced, cv2.multiply(residual, m8,
                                                      scale=self.unsharp_amount / 255.0,
                                                      dtype=cv2.CV_32F))
            if self.output_smooth > 0.0:
                enhanced = cv2.GaussianBlur(enhanced, (0, 0), self.output_smooth)
            enhanced_u8 = cv2.convertScaleAbs(enhanced, alpha=1.0)
            enhanced8 = cv2.LUT(enhanced_u8,
                                _stretch_lut(enhanced_u8, 0.5, 99.5, self._gamma_lut))

            seg = cv2.morphologyEx(cv2.convertScaleAbs(mask_full, alpha=255.0),
                                   cv2.MORPH_CLOSE, self._seg_kernel)

            enhanced16 = None
            if bit_depth == 16 and self.keep_16bit_output:
                span = max(hi - lo, 1.0)
                enhanced16 = (enhanced8.astype(np.float32) * (span / 255.0) + lo)
                enhanced16 = np.clip(enhanced16, 0, 65535).astype(np.uint16)

            comparison = (self._comparison(gray8, processed8, enhanced8)
                          if self.make_comparison else np.zeros((1, 1, 3), np.uint8))

        enhanced_16 = self._to_pil(enhanced16) if enhanced16 is not None else None
        result = PipelineResult(
            raw=self._to_pil(gray8),
            processed=self._to_pil(processed8),
            enhanced=self._to_pil(enhanced8),
            segmentation=self._to_pil(seg),
            comparison=self._to_pil(comparison),
            latency_ms=sum(timer.stages.values()),
            budget_ms=self.budget_ms,
            input_bit_depth=bit_depth,
            output_bit_depth=16 if enhanced_16 is not None else 8,
            enhanced_16bit=enhanced_16,
            stages_ms=dict(timer.stages),
            metrics={
                "components": float(components),
                "rib_components_removed": float(ribs_removed),
                "vessel_area_ratio": float(np.count_nonzero(mask_full)) / float(mask_full.size),
                "mean_vesselness": float(cv2.mean(vesselness_full)[0]),
            },
        )
        return result


def build_default_pipeline(**kwargs: Any) -> CoronaryRealtimePipeline:
    """Factory used by ``demo.py`` and ``benchmark.py``."""
    return CoronaryRealtimePipeline(**kwargs)
