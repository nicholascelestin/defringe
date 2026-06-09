"""Shared helpers for the defringe algorithms.

Pure domain logic — numpy / skimage / scipy only, no I/O. RGB uint8 in, arrays out.
Both passes work in CIELAB, keep L* untouched, and carry untouched pixels through
bit-exact (alpha == 0 -> original pixel verbatim).
"""
import numpy as np
from skimage.color import rgb2lab, lab2rgb
from scipy.ndimage import gaussian_filter, zoom


def to_lab(rgb_u8):
    """RGB uint8 (or float 0-255) -> CIELAB float."""
    return rgb2lab(np.asarray(rgb_u8, np.float32) / 255.0)


def lab_to_rgb(lab):
    """CIELAB float -> RGB float 0-255 (clipped, not yet uint8)."""
    return np.clip(lab2rgb(lab) * 255.0, 0, 255)


def composite(out_float, src_float, alpha):
    """Carry untouched pixels (alpha == 0) through bit-exact; -> uint8."""
    return np.where((alpha > 0)[..., None], out_float, src_float).astype(np.uint8)


def soft_step(x, thr, width):
    """Smooth 0->1 ramp centred at `thr`, reaching 0/1 exactly at thr-+width.

    A drop-in soft replacement for a hard `x > thr` boolean: a value near the
    threshold contributes partially and varies *continuously* as the input
    jitters, instead of flipping 0<->1 frame to frame (the main flicker source).
    Smoothstep (not logistic) so it is exactly 0 below the band and 1 above it,
    preserving sparsity and the bit-exact `alpha == 0` passthrough. `width <= 0`
    recovers the hard step (x > thr) at the input's own precision, so a width=0
    pass is bit-identical to the original comparison. Output is float32 to match
    the float32 Lab pipeline (a float64 weight would silently upcast downstream
    blurs and perturb results at float32 epsilon). `thr` and `width` may be
    scalars or arrays broadcastable to `x` (e.g. a per-component limit)."""
    x = np.asarray(x)                                      # compare at native precision
    width = np.asarray(width, np.float64)
    safe_w = np.where(width > 0, width, 1.0)               # avoid /0 where hard
    t = np.clip((x - thr) / (2.0 * safe_w) + 0.5, 0.0, 1.0)
    smooth = t * t * (3.0 - 2.0 * t)
    return np.where(width > 0, smooth, x > thr).astype(np.float32)


def hue_weight(va, vb, target, power):
    """Chroma magnitude weighted by how closely (va,vb) points at `target`."""
    ta, tb = target
    n = (ta * ta + tb * tb) ** 0.5
    mag = np.hypot(va, vb)
    cos = (va * ta + vb * tb) / (mag * n + 1e-6)
    return mag * np.clip(cos, 0, 1) ** power


def blur(x, sigma, ds=1):
    """Gaussian blur, with an optional fast path for large sigma.

    For ds>1 the blur is computed on a copy downsampled by `ds` at sigma/ds, then
    upsampled back — visually identical for smooth large-sigma tone estimates at a
    fraction of the cost. ds=1 is the exact gaussian_filter.
    """
    if ds <= 1 or sigma < 2 * ds:
        return gaussian_filter(x, sigma)
    h, w = x.shape
    small = zoom(x, 1.0 / ds, order=1, mode="reflect")
    small = gaussian_filter(small, sigma / ds)
    return zoom(small, (h / small.shape[0], w / small.shape[1]), order=1, mode="reflect")


def normconv(x, w, sigma, ds=1, den=None):
    """Trust-weighted blur (normalised convolution).

    `den` lets callers share one weight-blur across channels that use the same `w`.
    """
    if den is None:
        den = blur(w, sigma, ds) + 1e-6
    return blur(x * w, sigma, ds) / den
