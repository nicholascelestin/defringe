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
