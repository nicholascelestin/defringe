"""Green/purple fringe removal in CIELAB. Both passes share one casting-shadow model
-- find the caster source, find its nearby chroma fringe, pull that chroma to the clean
local tone -- and differ only in how they define caster/fringe/strength (shared back
end: cast_defringe). Chain green-first; purple fringe can itself cast green fringe.

    g, _ = green_cast(rgb); out, _ = purple_cast(g)
"""
import numpy as np
from skimage.color import rgb2lab, lab2rgb
from scipy.ndimage import gaussian_filter, maximum_filter, uniform_filter

__all__ = ["green_cast", "purple_cast", "cast_defringe",
           "GREEN_CAST", "PURPLE_CAST", "RED_REF", "GREEN_REF", "MAGENTA_REF"]


# ── colour-space + shared helpers ────────────────────────────────────────────

def rgb_to_lab(rgb_u8):
    """RGB uint8 (or float 0-255) -> CIELAB float."""
    return rgb2lab(np.asarray(rgb_u8, np.float32) / 255.0)


def lab_to_rgb(lab):
    """CIELAB float -> RGB float 0-255 (clipped, not yet uint8)."""
    return np.clip(lab2rgb(lab) * 255.0, 0, 255)


def composite(out_float, src_float, alpha):
    """Carry untouched pixels (alpha == 0) through bit-exact; -> uint8."""
    return np.where((alpha > 0)[..., None], out_float, src_float).astype(np.uint8)


def soft_step(x, thr, width):
    """Soft, temporally stable stand-in for `x > thr` -> float32 in [0, 1];
    width <= 0 falls back to the exact hard comparison."""
    x = np.asarray(x)
    width = np.asarray(width, np.float64)
    hard_step = x > thr
    # The clamp pins smoothstep to exactly 0/1 outside the band, so untouched pixels
    # stay at literal 0 for the bit-exact alpha==0 passthrough downstream.
    half_width = np.where(width > 0, width, 1.0)          # dummy where hard; avoids /0
    low, high = thr - half_width, thr + half_width
    t = np.clip((x - low) / (high - low), 0.0, 1.0)
    smoothstep = t * t * (3.0 - 2.0 * t)
    # float32, not float64: a wider weight would upcast the downstream Lab blurs.
    return np.where(width > 0, smoothstep, hard_step).astype(np.float32)


def normconv(signal, weight, sigma, denom=None):
    """Trust-weighted blur: blur(signal*weight) / blur(weight). `denom` shares one
    weight-blur across channels with the same `weight`."""
    if denom is None:
        denom = gaussian_filter(weight, sigma) + 1e-6
    return gaussian_filter(signal * weight, sigma) / denom


def local_tone(a, b, trust, sigma):
    """Trust-weighted local mean of the two chroma channels (the clean tone fringe is
    pulled toward); the shared `trust` lets one denominator blur serve both."""
    den = gaussian_filter(trust, sigma) + 1e-6
    return normconv(a, trust, sigma, den), normconv(b, trust, sigma, den)


# ── shared casting-shadow engine ─────────────────────────────────────────────

def cast_defringe(src, lab, caster, fringe, strength, p):
    """Shared back end: gate `fringe` by reach to a nearby `caster`, build a feathered
    alpha from `strength`, pull chroma to the trusted local tone (L kept).
    -> (uint8, {alpha, caster, caster_raw, shadow}); untouched pixels bit-exact.

    Geometry is scale-space and resolution-relative: the spatial params arrive as
    fractions of the frame and convert to px from the frame's own size, so a setting
    transfers across resolutions. `min_area` is ppm of frame AREA; `cast_radius`,
    `feather`, `repair_spread`, `tone_correction_radius` are per-mille of the DIAGONAL.
    The area test is a box-sum count + soft threshold (not connected components) and the
    reach is a square dilation + gaussian feather (not a Euclidean distance transform) --
    both linear/local, so the ONNX port matches and resolution scaling is exact.
    caster_raw is the raw detection; caster is what survived the area test."""
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    H, W = L.shape
    diag = (W * W + H * H) ** 0.5
    min_area = p["min_area"] * (H * W) / 1e6                     # ppm of area  -> px^2
    reach = p["cast_radius"] * diag / 1000.0                    # per-mille of diagonal -> px
    feather = p["feather"] * diag / 1000.0
    spread = p["repair_spread"] * diag / 1000.0
    tone_radius = p["tone_correction_radius"] * diag / 1000.0

    cf = caster.astype(np.float32)
    # Minimum Area as a local bright-pixel count (box-sum), soft-thresholded into a
    # continuous per-pixel caster weight -- marginal-size casters ramp in, not pop.
    wa = 2 * max(1, int(round(min_area ** 0.5))) + 1
    count = uniform_filter(cf, wa) * float(wa * wa)
    caster_w = cf * soft_step(count, min_area - 0.5, max(p["area_softness"], 1e-6) * min_area)
    caster_mask = caster_w > 0.5

    # Cast Reach as a square dilation of the kept-caster field (which carries the area
    # weight) plus a gaussian feather; 0.8/0.4 calibrate the square footprint to the
    # former Euclidean reach. reach_weight ~ 1 over caster + reach, decaying outward.
    R = max(1, int(round(reach * 0.8)))
    reach_weight = np.clip(gaussian_filter(maximum_filter(caster_w, size=2 * R + 1),
                                           max(p["radius_softness"] * reach * 0.4, 1e-3)), 0, 1)
    candidate = fringe & (reach_weight > 1e-3) & (~caster_mask)
    shadow_strength = candidate.astype(np.float32) * reach_weight
    donor = np.ones(L.shape, np.float32)
    if p["tone_directionality"] > 0:                            # bias tone donors off the caster side
        donor = (1.0 - p["tone_directionality"] * reach_weight).astype(np.float32)

    alpha_seed = shadow_strength * strength
    sp = int(round(spread))
    if sp > 0:
        alpha_seed = maximum_filter(alpha_seed, size=2 * sp + 1)
    alpha = np.clip(gaussian_filter(alpha_seed, feather), 0, p["max_opacity"])
    # 1e-3 keeps faint trust everywhere so corrected regions still get a tone estimate
    trust = ((1 - alpha) + 1e-3) * donor
    tone_a, tone_b = local_tone(a, b, trust, tone_radius)
    out = lab_to_rgb(np.stack([L, a + alpha * (tone_a - a), b + alpha * (tone_b - b)], -1))
    out = composite(out, src, alpha)
    return out, {"alpha": alpha, "caster": caster_mask, "caster_raw": caster, "shadow": shadow_strength > 0}


# ── green pass ───────────────────────────────────────────────────────────────

RED_REF, GREEN_REF = 5.0, 200.0

GREEN_CAST = dict(
    caster_min_chroma=21.0,
    # Hue bands are signed-degree offsets from a reference hue (RED_REF / GREEN_REF),
    # which sidesteps red's 0/360 wraparound; negative = toward purple, positive = orange.
    caster_hue_lo=-54.0,
    caster_hue_hi=23.0,
    fringe_hue_lo=-80.0,
    fringe_hue_hi=90.0,
    fringe_min_chroma=4.5,
    # Spatial knobs are resolution-relative (see cast_defringe): cast_radius/feather/
    # repair_spread/tone_correction_radius are per-mille of the diagonal; min_area is
    # ppm of frame area. Values below are the former 1080p pixel defaults, converted.
    cast_radius=4.0,              # ‰ of diagonal
    min_area=10.0,                # ppm of frame area
    full_strength_span=9.5,
    max_opacity=0.7,
    repair_spread=0.45,           # was 1 px
    feather=0.91,                 # was 2 px
    tone_correction_radius=7.26,  # was 16 px
    # *_softness: soft-step ramp width as a fraction of the matching threshold.
    area_softness=0.4,
    radius_softness=0.4,
    tone_directionality=0.5,
)


def green_cast(rgb, **kw):
    """Remove green fringe cast by saturated red/purple sources. -> (uint8, info)."""
    p = {**GREEN_CAST, **kw}
    src = np.asarray(rgb, np.float32)
    lab = rgb_to_lab(src)
    a, b = lab[..., 1], lab[..., 2]
    hue = np.degrees(np.arctan2(b, a)) % 360.0
    chroma = np.hypot(a, b)
    hue_from_red = ((hue - RED_REF + 180.0) % 360.0) - 180.0
    hue_from_green = ((hue - GREEN_REF + 180.0) % 360.0) - 180.0

    caster = (chroma > p["caster_min_chroma"]) & (hue_from_red >= p["caster_hue_lo"]) & (hue_from_red <= p["caster_hue_hi"])
    fringe = (chroma > p["fringe_min_chroma"]) & (hue_from_green >= p["fringe_hue_lo"]) & (hue_from_green <= p["fringe_hue_hi"])
    strength = np.clip((chroma - p["fringe_min_chroma"]) / max(p["full_strength_span"], 1e-3), 0, 1)
    return cast_defringe(src, lab, caster, fringe, strength, p)


# ── purple pass ──────────────────────────────────────────────────────────────

MAGENTA_REF = 315.0   # magenta-violet; the fringe excess is measured along this hue

PURPLE_CAST = dict(
    caster_min_lightness=88.0,
    fringe_min_chroma=6.8,
    cast_radius=3.63,             # per-mille of diagonal (was 8 px @1080p); see cast_defringe
    min_area=4.82,               # ppm of frame area      (was 10 px^2 @1080p)
    full_strength_span=10.0,
    max_opacity=0.85,
    repair_spread=0.91,          # was 2 px
    feather=0.91,                # was 2 px
    tone_correction_radius=7.26, # was 16 px
    # *_softness: soft-step ramp width as a fraction of the matching threshold.
    area_softness=0.4,
    radius_softness=0.4,
    tone_directionality=0.7,
    target_hue=-15.0,         # signed deg from magenta (315): 0 = magenta, + red, - violet
    excess_thresh=16.0,
    hue_halfwidth=90.0,       # +-deg accepted around target_hue; 90 ~ gate off
    hue_softness=20.0,        # feather on that edge, in DEGREES (not a fraction)
)


def purple_cast(rgb, **kw):
    """Remove purple fringe cast by blown highlights. -> (uint8, info)."""
    p = {**PURPLE_CAST, **kw}
    src = np.asarray(rgb, np.float32)
    lab = rgb_to_lab(src)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    chroma = np.hypot(a, b)

    caster = L > p["caster_min_lightness"]

    # Magenta fringe = an EXCESS over the scene's lighting tone (luminance-weighted mean
    # of a/b over lit, non-blown pixels), not an absolute hue band -- so a warm-lit scene
    # cancels its own warmth and only the blue-shifted fringe stands out.
    target_angle = np.radians(MAGENTA_REF + p["target_hue"])
    axis_a, axis_b = np.cos(target_angle), np.sin(target_angle)
    lit_weight = L * (L < p["caster_min_lightness"])
    weight_sum = float(lit_weight.sum()) + 1e-6
    lighting_a = float((lit_weight * a).sum() / weight_sum)
    lighting_b = float((lit_weight * b).sum() / weight_sum)
    excess_a, excess_b = a - lighting_a, b - lighting_b
    magenta_excess = excess_a * axis_a + excess_b * axis_b
    # angular gate on the excess vector, soft-edged so it doesn't flip frame to frame
    off_axis = -excess_a * axis_b + excess_b * axis_a
    off_axis_deg = np.degrees(np.arctan2(np.abs(off_axis), magenta_excess))
    hue_weight = soft_step(-off_axis_deg, -(p["hue_halfwidth"] + 1e-6), p["hue_softness"])
    fringe = (chroma > p["fringe_min_chroma"]) & (magenta_excess > p["excess_thresh"]) & (hue_weight > 0)
    strength = np.clip((magenta_excess - p["excess_thresh"]) / max(p["full_strength_span"], 1e-3), 0, 1) * hue_weight
    return cast_defringe(src, lab, caster, fringe, strength, p)
