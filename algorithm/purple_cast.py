"""Purple-fringe (axial CA) removal via the green-cast casting-shadow model.

EXPERIMENTAL sibling of purple_defringe(). Instead of an edge-gated detector, it
treats axial chromatic aberration the way green_cast treats green fringe -- as a
*shadow* cast by a source. Here the source ("caster") is the blown near-white
highlight that physically causes axial CA, and the artifact ("shadow") is the
magenta-violet fringe hugging it. Mirrors green_cast as closely as possible:
find bright caster blobs -> look for magenta in their cast reach -> feathered,
chroma-weighted alpha -> pull the magenta chroma to local tone (L kept).

Does NOT touch purple_defringe(); this is a parallel pass to compare and play with.
"""
import numpy as np
from scipy.ndimage import (gaussian_filter, maximum_filter, label,
                           distance_transform_edt)

from ._common import to_lab, lab_to_rgb, soft_step

# Magenta-violet (~315deg, the -45deg axial-CA direction). The shadow hue band is
# signed offsets from here, mirroring green_cast's GREEN_REF.
MAGENTA_REF = 315.0

DEFAULTS = dict(
    bright_L=88.0,      # caster: min L* for a near-white (blown highlight) source
    mag_lo=-15.0,       # shadow hue band, signed deg from magenta (MAGENTA_REF):
    mag_hi=30.0,        #   blue/violet edge .. magenta/red edge
    mag_chr=6.8,        # magenta chroma floor (lower = catches fainter fringe)
    cast_radius=8,      # reach from the highlight edge to search for fringe (px)
    min_area=10,        # ignore caster (highlight) blobs smaller than this (px)
    full_strength_span=23.5,     # chroma above mag_chr at which correction is full
    max_strength=0.85,           # alpha cap (opacity, NOT spatial)
    repair_spread=2,             # spatial: dilate the corrected region by this many px
    feather=2.0,                 # alpha feather sigma (px)
    tone_correction_radius=16.0, # local-tone estimate radius (px)
    area_soft=0.4,      # min_area ramp (fraction of min_area)
    radius_soft=0.4,    # cast_radius distance falloff (fraction of cast_radius)
)


def purple_cast(rgb, **kw):
    """Remove purple fringe cast by blown highlights (green-cast model). -> (uint8, info).

    info has {alpha, caster, shadow} (caster mask + accepted-fringe mask) for the
    casters-and-shadows view. Untouched pixels carry through bit-exact.
    """
    p = {**DEFAULTS, **kw}
    src = np.asarray(rgb, np.float32)
    lab = to_lab(src)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    hue = np.degrees(np.arctan2(b, a)) % 360.0
    chroma = np.hypot(a, b)
    ms = ((hue - MAGENTA_REF + 180.0) % 360.0) - 180.0   # signed dist from magenta

    # 1. casters: near-white blown highlights (the physical axial-CA source)
    caster = L > p["bright_L"]
    # 2. magenta-violet fringe candidates
    mag = (chroma > p["mag_chr"]) & (ms >= p["mag_lo"]) & (ms <= p["mag_hi"])

    keepS = np.zeros_like(chroma, np.float32)
    caster_big = np.zeros_like(chroma, bool)
    lbl, n = label(caster)
    if n:
        areas = np.bincount(lbl.ravel()); areas[0] = 0
        # soft min-area on the highlight blobs (same trick as green_cast)
        big = soft_step(areas.astype(np.float32), p["min_area"] - 0.5,
                        p["area_soft"] * p["min_area"])[lbl] > 0
        lbl, n = label(big); areas = np.bincount(lbl.ravel()); areas[0] = 0
        caster_w = soft_step(areas.astype(np.float32), p["min_area"] - 0.5,
                             p["area_soft"] * p["min_area"]); caster_w[0] = 0.0
        caster_big = big
        # 3. one EDT gives the cast reach and the nearest highlight per pixel
        dist, (iy, ix) = distance_transform_edt(~big, return_indices=True)
        rw = p["radius_soft"] * p["cast_radius"]
        cand = mag & (dist <= p["cast_radius"] + 2.0 * rw) & ~big
        radius_w = soft_step(-dist, -(p["cast_radius"] + 1e-6), rw)
        # every magenta candidate within reach is a cast fringe, weighted by the
        # nearest highlight's area
        keepS = cand.astype(np.float32) * caster_w[lbl[iy, ix]] * radius_w

    # 4. chroma-weighted alpha: spatially grow (dilate), feather, cap
    abase = keepS * np.clip((chroma - p["mag_chr"]) / max(p["full_strength_span"], 1e-3), 0, 1)
    if p["repair_spread"] > 0:
        abase = maximum_filter(abase, size=int(2 * p["repair_spread"] + 1))
    alpha = np.clip(gaussian_filter(abase, p["feather"]), 0, p["max_strength"])
    # 5. repair: pull magenta chroma to the local (trust-weighted) tone; keep L
    trust = (1 - alpha) + 1e-3
    den = gaussian_filter(trust, p["tone_correction_radius"]) + 1e-6
    ta = gaussian_filter(a * trust, p["tone_correction_radius"]) / den
    tb = gaussian_filter(b * trust, p["tone_correction_radius"]) / den
    out = lab_to_rgb(np.stack([L, a + alpha * (ta - a), b + alpha * (tb - b)], -1))
    out = np.where((alpha > 0)[..., None], out, src).astype(np.uint8)
    return out, {"alpha": alpha, "caster": caster_big, "shadow": keepS > 0}
