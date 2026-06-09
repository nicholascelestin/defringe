"""Green-fringe removal via a casting-shadow model, in CIELAB.

Green fringe behaves like a *shadow* cast by a saturated RED (or, more weakly,
PURPLE) source: it hugs the source's edge, bleeds into the neighbouring non-red
area, and is bounded by the source's size. So: find red/purple caster blobs ->
look for valley-green in their cast reach -> reject any shadow larger than
SIZE_FACTOR x its caster (a real object, not a cast) -> feathered, chroma-weighted
alpha -> pull the green chroma to local tone (L kept).

Independent of purple defringe(); the recommended full clean is green_cast() then
purple defringe(), each staying in its own hue lane.
"""
import numpy as np
from scipy.ndimage import (gaussian_filter, maximum_filter, label,
                           distance_transform_edt, maximum as label_max)

from ._common import to_lab, lab_to_rgb

# 200deg blue-green/teal: the behind-focus sibling of axial CA (reference only;
# the caster/shadow model below is hue-band based, not single-target).
GREEN = (-0.94, -0.34)

DEFAULTS = dict(
    cast_chr=21.0,      # caster saturation threshold (red & purple sources)
    band_lo=-54.0,      # caster hue band, signed deg from red(5deg): purple edge
    band_hi=23.0,       #   ...orange edge
    purple_str=1.0,     # purple caster strength vs red (1 = equal, <1 = weaker)
    purple_dist=48.0,   # deg over which strength ramps red -> purple
    green_lo=120.0,     # valley-green hue range of the shadow
    green_hi=290.0,
    green_chr=4.5,      # green chroma floor (lower = catches fainter shadow)
    cast_radius=18,     # reach from the source edge to search for shadow (px)
    min_area=19,        # ignore caster blobs smaller than this (px)
    size_factor=3.2,    # reject green shadow > this x its caster's area
    ch_ref=14.0,        # green chroma -> full alpha (strength)
    amax=0.7,           # alpha cap (strength/opacity, NOT spatial)
    grow=1,             # spatial: dilate the alpha region by this many px
    feather=2.0,        # alpha feather sigma (px; softens, slightly spreads)
    tone_sig=16.0,      # local-tone estimate radius (px)
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
    hs = ((hue - 5.0 + 180.0) % 360.0) - 180.0          # signed dist from red (5deg)

    # 1. casters: saturated red->purple sources, weighted (purple weaker)
    caster = (chroma > p["cast_chr"]) & (hs >= p["band_lo"]) & (hs <= p["band_hi"])
    strength = np.where(hs >= 0, 1.0,
        np.clip(1.0 - (-hs) / max(p["purple_dist"], 1e-3) * (1.0 - p["purple_str"]),
                p["purple_str"], 1.0)) * caster
    # 2. valley-green shadow candidates
    green = (chroma > p["green_chr"]) & (hue >= p["green_lo"]) & (hue <= p["green_hi"])

    keepS = np.zeros_like(chroma, np.float32)
    caster_big = np.zeros_like(chroma, bool)
    lbl, n = label(caster)
    if n:
        areas = np.bincount(lbl.ravel()); areas[0] = 0
        big = (areas >= p["min_area"])[lbl]
        lbl, n = label(big); areas = np.bincount(lbl.ravel()); areas[0] = 0
        caster_big = big
        # 3. one EDT gives BOTH the cast reach and the nearest caster per pixel
        dist, (iy, ix) = distance_transform_edt(~big, return_indices=True)
        cand = green & (dist <= p["cast_radius"]) & ~big
        g_lbl, ng = label(cand)
        if ng:
            carea = areas[lbl[iy, ix]]
            comp_size = np.bincount(g_lbl.ravel())[1:]
            comp_caster = label_max(carea, g_lbl, index=np.arange(1, ng + 1))
            keep = np.concatenate([[False],
                comp_size <= p["size_factor"] * np.maximum(comp_caster, 1)])[g_lbl]
            keepS = keep * strength[iy, ix]

    # 4. chroma-weighted alpha: spatially grow (dilate), feather, cap
    abase = keepS * np.clip(chroma / max(p["ch_ref"], 1e-3), 0, 1)
    if p["grow"] > 0:
        abase = maximum_filter(abase, size=int(2 * p["grow"] + 1))
    alpha = np.clip(gaussian_filter(abase, p["feather"]), 0, p["amax"])
    # 5. repair: pull green chroma to the local (trust-weighted) tone; keep L
    trust = (1 - alpha) + 1e-3
    den = gaussian_filter(trust, p["tone_sig"]) + 1e-6
    ta = gaussian_filter(a * trust, p["tone_sig"]) / den
    tb = gaussian_filter(b * trust, p["tone_sig"]) / den
    out = lab_to_rgb(np.stack([L, a + alpha * (ta - a), b + alpha * (tb - b)], -1))
    out = np.where((alpha > 0)[..., None], out, src).astype(np.uint8)
    return out, {"alpha": alpha, "caster": caster_big, "shadow": keepS > 0}
