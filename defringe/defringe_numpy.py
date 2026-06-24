from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from skimage.color import rgb2lab, lab2rgb
from scipy.ndimage import gaussian_filter, maximum_filter, uniform_filter

from . import geometry
from .parameters import GREEN_DEFAULTS, PURPLE_DEFAULTS

__all__ = ["defringe", "green_cast", "purple_cast", "Cast",
           "RED_REF", "GREEN_REF", "MAGENTA_REF"]

RED_REF, GREEN_REF, MAGENTA_REF = 5.0, 200.0, 315.0
KEPT_WEIGHT = 0.5
TRUST_FLOOR = 1e-3
NEGLIGIBLE_REACH = 1e-3


@dataclass(frozen=True)
class Cast:
    image: NDArray[np.uint8]
    alpha: NDArray[np.float32]
    casters_kept: NDArray[np.bool_]
    casters_found: NDArray[np.bool_]


def defringe(rgb, green=None, purple=None):
    # Green first: purple fringe can itself cast green fringe, so removing purple first would
    # orphan that green fringe (no caster left near it) and leave it uncorrectable.
    green, purple = green or {}, purple or {}
    cleaned = green_cast(rgb, **green)
    return purple_cast(cleaned.image, **purple).image


def green_cast(rgb, **kw):
    p = {**GREEN_DEFAULTS, **kw}
    src = np.asarray(rgb, np.float32)
    lab = rgb_to_lab(src)
    a, b = lab[..., 1], lab[..., 2]
    hue, chroma = to_polar(a, b)
    casters = (chroma > p["caster_min_chroma"]) & in_band(signed_hue_offset(hue, RED_REF), p["caster_hue_lo"], p["caster_hue_hi"])
    fringe = (chroma > p["fringe_min_chroma"]) & in_band(signed_hue_offset(hue, GREEN_REF), p["fringe_hue_lo"], p["fringe_hue_hi"])
    strength = ramp(chroma, p["fringe_min_chroma"], p["full_strength_span"])
    return _cast_defringe(src, lab, casters, fringe, strength, p)


def purple_cast(rgb, **kw):
    p = {**PURPLE_DEFAULTS, **kw}
    src = np.asarray(rgb, np.float32)
    lab = rgb_to_lab(src)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    _, chroma = to_polar(a, b)
    casters = L > p["caster_min_lightness"]
    scene_a, scene_b = scene_tone(L, a, b, p["caster_min_lightness"])
    excess_a, excess_b = a - scene_a, b - scene_b
    excess = np.hypot(excess_a, excess_b)
    off_target = angle_from_axis(excess_a, excess_b, np.radians(MAGENTA_REF + p["target_hue"]))
    within_cone = soft_step(-off_target, -(p["hue_halfwidth"] + 1e-6), p["hue_softness"])
    fringe = (chroma > p["fringe_min_chroma"]) & (excess > p["excess_thresh"]) & (within_cone > 0)
    strength = ramp(excess, p["excess_thresh"], p["full_strength_span"]) * within_cone
    return _cast_defringe(src, lab, casters, fringe, strength, p)


# ── casting-shadow engine ────────────────────────────────────────────────────

def _cast_defringe(src, lab, casters, fringe, strength, p):
    geom = geometry.relative_to_px(p, *lab.shape[:2])
    caster_field = survive_min_area(casters, geom["min_area"], p["area_softness"])
    casters_kept = caster_field > KEPT_WEIGHT
    reach = cast_reach(caster_field, geom["cast_radius"], p["radius_softness"])
    reachable = (reach > NEGLIGIBLE_REACH) & ~casters_kept
    shadow = (fringe & reachable).astype(np.float32) * reach
    alpha = feathered_alpha(shadow * strength, geom["repair_spread"], geom["feather"], p["max_opacity"])
    image = repair_chroma(src, lab, alpha, reach, geom["tone_correction_radius"], p["tone_directionality"])
    return Cast(image, alpha, casters_kept, casters)


def survive_min_area(casters, min_area, softness):
    casters = casters.astype(np.float32)
    window = geometry.area_window(min_area)
    neighbours = uniform_filter(casters, window) * float(window * window)
    return casters * soft_step(neighbours, min_area - 0.5, max(softness, 1e-6) * min_area)


def cast_reach(caster_field, reach_px, softness):
    radius = max(1, round(reach_px * geometry.REACH_CALIB))
    blur = max(softness * reach_px * geometry.REACH_FEATHER_CALIB, 1e-3)
    dilated = maximum_filter(caster_field, size=2 * radius + 1)
    return np.clip(gaussian_filter(dilated, blur), 0.0, 1.0)


def grow_region(field, radius_px):
    radius = round(radius_px)
    return maximum_filter(field, size=2 * radius + 1)


def feathered_alpha(seed, spread_px, feather_px, max_opacity):
    grown = grow_region(seed, spread_px)
    return np.clip(gaussian_filter(grown, feather_px), 0.0, max_opacity)


def tone_trust(alpha, reach_weight, directionality):
    donor = (1.0 - directionality * reach_weight).astype(np.float32)
    return ((1 - alpha) + TRUST_FLOOR) * donor


def repair_chroma(src, lab, alpha, reach_weight, tone_radius, directionality):
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    trust = tone_trust(alpha, reach_weight, directionality)
    tone_a, tone_b = local_tone(a, b, trust, tone_radius)
    corrected = np.stack([L, lerp(a, tone_a, alpha), lerp(b, tone_b, alpha)], -1)
    return composite(lab_to_rgb(corrected), src, alpha)


# ── trust-weighted tone ──────────────────────────────────────────────────────

def local_tone(a, b, trust, sigma):
    trust_blurred = gaussian_filter(trust, sigma)
    return (trust_weighted_blur(a, trust, sigma, trust_blurred),
            trust_weighted_blur(b, trust, sigma, trust_blurred))


def trust_weighted_blur(signal, trust, sigma, trust_blurred=None):
    if trust_blurred is None:
        trust_blurred = gaussian_filter(trust, sigma)
    return safe_divide(gaussian_filter(signal * trust, sigma), trust_blurred)


def scene_tone(L, a, b, blown):
    lit = L * (L < blown)
    weight = lit.sum()
    return safe_divide((lit * a).sum(), weight), safe_divide((lit * b).sum(), weight)


# ── maths ────────────────────────────────────────────────────────────────────

def soft_step(x, threshold, width):
    if width <= 0:
        return (x > threshold).astype(np.float32)
    width = np.float64(width)
    return smoothstep(threshold - width, threshold + width, x).astype(np.float32)


def smoothstep(lo, hi, x):
    t = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def lerp(start, end, t):
    return start + t * (end - start)


def ramp(value, threshold, span):
    return np.clip((value - threshold) / max(span, 1e-3), 0.0, 1.0)


def in_band(x, lo, hi):
    return (x >= lo) & (x <= hi)


def signed_hue_offset(hue, reference):
    return ((hue - reference + 180.0) % 360.0) - 180.0


def angle_from_axis(vec_a, vec_b, axis_rad):
    axis_a, axis_b = np.cos(axis_rad), np.sin(axis_rad)
    along = vec_a * axis_a + vec_b * axis_b
    across = -vec_a * axis_b + vec_b * axis_a
    return np.degrees(np.arctan2(np.abs(across), along))


def safe_divide(numerator, denominator):
    return numerator / (denominator + 1e-6)


# ── colour ───────────────────────────────────────────────────────────────────

def rgb_to_lab(rgb_u8):
    return rgb2lab(np.asarray(rgb_u8, np.float32) / 255.0)


def lab_to_rgb(lab):
    return np.clip(lab2rgb(lab) * 255.0, 0, 255)


def to_polar(a, b):
    return np.degrees(np.arctan2(b, a)) % 360.0, np.hypot(a, b)


def composite(corrected, original, alpha):
    touched = (alpha > 0)[..., np.newaxis]
    return np.where(touched, corrected, original).astype(np.uint8)
