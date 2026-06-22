"""Shared resolution-relative → pixel geometry — the one contract the numpy reference
(defringe_numpy) and its torch/ONNX twin (defringe_torch) must agree on byte-for-byte.

Spatial params are resolution-relative so a setting transfers across resolutions:
`cast_radius`, `feather`, `repair_spread`, `tone_correction_radius` are per-mille of the
frame DIAGONAL; `min_area` is ppm of frame AREA. They convert to pixels from the frame's
own size. The two calibration constants tie the box-sum / square-dilation geometry to the
former Euclidean reach — defined here once so the twin can't silently drift from the
reference (the conformance suite would catch it, but only after the fact).
"""

# Spatial keys measured in per-mille of the diagonal (min_area is handled separately).
DIAG_KEYS = ("cast_radius", "feather", "repair_spread", "tone_correction_radius")

REACH_CALIB = 0.8          # square-dilation radius = reach_px * this (calibrates to Euclidean reach)
REACH_FEATHER_CALIB = 0.4  # reach gaussian sigma = radius_softness * reach_px * this


def relative_to_px(params, h, w):
    """Copy of `params` with spatial keys converted to pixels for an h×w frame:
    DIAG_KEYS from ‰-of-diagonal, `min_area` from ppm-of-area (→ px²)."""
    diag = (w * w + h * h) ** 0.5
    out = dict(params)
    for k in DIAG_KEYS:
        out[k] = params[k] * diag / 1000.0
    out["min_area"] = params["min_area"] * (h * w) / 1e6
    return out


def area_window(min_area_px):
    """Odd box-sum window side holding ~`min_area_px` px — the Minimum-Area count kernel."""
    return 2 * max(1, int(round(min_area_px ** 0.5))) + 1
