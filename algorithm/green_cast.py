"""Green-fringe removal via a casting-shadow model, in CIELAB.

Green fringe behaves like a *shadow* cast by a saturated RED (or, more weakly,
PURPLE) source: it hugs the source's edge, bleeds into the neighbouring non-red
area, and is bounded by the source's size. So: find red/purple caster blobs ->
look for valley-green in their cast reach -> feathered, chroma-weighted alpha ->
pull the green chroma to local tone (L kept).

Independent of purple defringe(); the recommended full clean is green_cast() then
purple defringe(), each staying in its own hue lane.
"""
import numpy as np
from scipy.ndimage import (gaussian_filter, maximum_filter, label,
                           distance_transform_edt)

from ._common import to_lab, lab_to_rgb, soft_step

# 200deg blue-green/teal: the behind-focus sibling of axial CA (reference only;
# the caster/shadow model below is hue-band based, not single-target).
GREEN = (-0.94, -0.34)

# Both hue bands are signed offsets from a characteristic hue: red for the warm
# caster, blue-green/teal (~the GREEN direction, 200deg) for the cool shadow.
RED_REF, GREEN_REF = 5.0, 200.0

DEFAULTS = dict(
    cast_chr=21.0,      # caster saturation threshold (red & purple sources)
    band_lo=-54.0,      # caster hue band, signed deg from red(5deg): purple edge
    band_hi=23.0,       #   ...orange edge
    green_lo=-80.0,     # shadow hue band, signed deg from teal-green (GREEN_REF):
    green_hi=90.0,      #   green/yellow edge .. blue/violet edge (== abs hue 120..290)
    green_chr=4.5,      # green chroma floor (lower = catches fainter shadow)
    cast_radius=9,      # reach from the source edge to search for shadow (px)
    min_area=19,        # ignore caster blobs smaller than this (px)
    full_strength_span=9.5,  # chroma above green_chr (Minimum Chroma) at which correction
                             #   reaches full strength; ramp is anchored at the floor (0 there)
    max_strength=0.7,            # alpha cap (opacity, NOT spatial)
    repair_spread=1,             # spatial: dilate the corrected region by this many px
    feather=2.0,                 # alpha feather sigma (px; softens, slightly spreads)
    tone_correction_radius=16.0, # local-tone estimate radius (px)
    # soft decision widths (0 = hard threshold, identical to before; raise to
    # trade a little selectivity for temporal stability / less flicker). Baked to
    # a conservative "light" level: ~18% less static flicker, ~20% less alpha
    # jitter on frames 235-246, with only a small coverage increase.
    area_soft=0.3,      # min_area ramp (fraction of min_area)
    radius_soft=0.2,    # cast_radius distance falloff (fraction of cast_radius)
)


def green_cast(rgb, *, return_debug=False, **kw):
    """Remove green fringe cast by saturated red/purple sources. -> (uint8, info).

    info always has {alpha, caster, shadow} (caster mask + accepted-shadow mask),
    so the UI can render the casters-and-shadows view. Untouched pixels carry
    through bit-exact.
    """
    p = {**DEFAULTS, **kw}
    src = np.asarray(rgb, np.float32)
    lab = to_lab(src)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    hue = np.degrees(np.arctan2(b, a)) % 360.0
    chroma = np.hypot(a, b)
    hs = ((hue - RED_REF + 180.0) % 360.0) - 180.0      # signed dist from red
    gs = ((hue - GREEN_REF + 180.0) % 360.0) - 180.0    # signed dist from teal-green

    # 1. casters: saturated red/purple sources (uniform weight)
    caster = (chroma > p["cast_chr"]) & (hs >= p["band_lo"]) & (hs <= p["band_hi"])
    # 2. valley-green shadow candidates. Repair strength is a floor-anchored ramp:
    #    0 at green_chr (Minimum Chroma), full at green_chr + full_strength_span -
    #    so the onset is smooth without a separate soft chroma gate.
    green = (chroma > p["green_chr"]) & (gs >= p["green_lo"]) & (gs <= p["green_hi"])

    keepS = np.zeros_like(chroma, np.float32)
    caster_big = np.zeros_like(chroma, bool)
    lbl, n = label(caster)
    if n:
        areas = np.bincount(lbl.ravel()); areas[0] = 0
        # soft min-area: include marginal casters, weight contribution by area.
        # The -0.5 keeps width=0 identical to the original `area >= min_area`.
        big = soft_step(areas.astype(np.float32), p["min_area"] - 0.5,
                        p["area_soft"] * p["min_area"])[lbl] > 0
        lbl, n = label(big); areas = np.bincount(lbl.ravel()); areas[0] = 0
        caster_w = soft_step(areas.astype(np.float32), p["min_area"] - 0.5,
                             p["area_soft"] * p["min_area"]); caster_w[0] = 0.0
        caster_big = big
        # 3. one EDT gives the cast reach and the nearest caster per pixel
        dist, (iy, ix) = distance_transform_edt(~big, return_indices=True)
        # soft cast_radius: falloff half-width is a fraction of the reach; widen
        # the candidate seed so the ramp has room
        rw = p["radius_soft"] * p["cast_radius"]
        cand = green & (dist <= p["cast_radius"] + 2.0 * rw) & ~big
        radius_w = soft_step(-dist, -(p["cast_radius"] + 1e-6), rw)
        # every green candidate within reach is a cast shadow (no size gate),
        # weighted by the nearest caster's area
        keepS = cand.astype(np.float32) * caster_w[lbl[iy, ix]] * radius_w

    # 4. chroma-weighted alpha: spatially grow (dilate), feather, cap
    abase = keepS * np.clip((chroma - p["green_chr"]) / max(p["full_strength_span"], 1e-3), 0, 1)
    if p["repair_spread"] > 0:
        abase = maximum_filter(abase, size=int(2 * p["repair_spread"] + 1))
    alpha = np.clip(gaussian_filter(abase, p["feather"]), 0, p["max_strength"])
    # 5. repair: pull green chroma to the local (trust-weighted) tone; keep L
    trust = (1 - alpha) + 1e-3
    den = gaussian_filter(trust, p["tone_correction_radius"]) + 1e-6
    ta = gaussian_filter(a * trust, p["tone_correction_radius"]) / den
    tb = gaussian_filter(b * trust, p["tone_correction_radius"]) / den
    out = lab_to_rgb(np.stack([L, a + alpha * (ta - a), b + alpha * (tb - b)], -1))
    out = np.where((alpha > 0)[..., None], out, src).astype(np.uint8)
    return out, {"alpha": alpha, "caster": caster_big, "shadow": keepS > 0}
