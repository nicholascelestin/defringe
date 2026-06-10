"""Purple-fringe (axial chromatic-aberration) removal, in CIELAB.

A *fringe* is a pixel whose chroma deviates from its surroundings toward the
magenta-violet of axial CA, sitting on a strong luminance edge that borders a
near-white (blown) highlight. Detection produces a soft per-pixel weight (the
"field"); repair neutralises the fringe chroma toward local tone, leaving L*
untouched, so nothing darkens.

  detect(rgb)        -> {field, lab}
    1 anomaly   magenta-ward chroma deviation from a neighbour average (SEED)
    2 presence  absolute magenta-ward chroma                          (EXTENT)
    3 detected  grow seed across the band, capped by presence
    4 gate      strong luminance edge, bordering near-white, not blob-interior
    5 field     detected, faded inward from the gate, kept to the dark side
  repair(lab, field) -> rgb     neutralise fringe chroma by a field-derived alpha
  defringe(rgb)      -> uint8   detect once, repair iterated (each pass gentler)
"""
import numpy as np
from skimage.color import rgb2lab
from skimage.filters import scharr
from scipy.ndimage import (gaussian_filter, maximum_filter, binary_erosion,
                           distance_transform_edt, label)

from ._common import to_lab, lab_to_rgb, hue_weight, blur, normconv, soft_step

# Target fringe hue in Lab (a = green<->magenta, b = blue<->yellow).
# -45deg magenta-violet: centred on real axial CA, rejecting warm-red (b>0) and
# cool-teal/blue (a<0), so no separate colour guard is needed.
PURPLE = (0.71, -0.71)

DEFAULTS = dict(
    # --- detect ---
    target=PURPLE,        # fringe hue to detect/repair, as a Lab (a,b) direction
    hue_power=5.0,        # SEED hue selectivity (higher = tighter to `target`)
    grow_hue_power=None,  # GROW-ceiling selectivity; None = use hue_power
    presence_Lref=None,   # luminance-compensate the grow ceiling (None = off)
    presence_Lcap=3.0,
    ref_sigma=5.0,        # neighbour-average radius for the anomaly reference
    spread_win=7,         # grow the anomaly seed across the band (px, odd)
    edge_pct=90.0,        # contrast percentile that counts as a strong edge
    min_edge=15,          # drop edge fragments smaller than this (px)
    anomaly_thr=6.0,      # Lab-chroma to count as "coloured" (blob-interior test)
    int_erode=2,          # blob-interior erosion depth
    region_guard=False,   # suppress field in the deep interior of large same-hue blobs
    region_guard_thr=3.0,
    region_guard_erode=3,
    require_bright=True,   # gate on a near-white neighbour (purple)
    bright_win=9,         # light-side search window (px)
    bright_L=90.0,        # min L* for the bright "light side" (near-white)
    tol=20.0,             # soft reach radius inward from the edge (px)
    # --- fix ---
    field_ref=12.0,       # field value -> peak alpha
    strength=1.0,         # peak alpha at full field
    floor=0.15,           # min alpha where the map is active
    floor_ramp=4.0,       # field span over which the floor fades in
    fill_L_hi=85.0,       # dark-side cutoff: brighter than this L* is skipped
    fill_L_soft=25.0,     # softness of that cutoff
    fold_mix=0.5,         # how much surrounding tone to fold into the residual
    tone_sigma=25.0,      # radius of the surrounding-tone estimate
    blur_ds=2,            # downsample factor for the large tone blurs (1 = exact)
    normalize=True,       # re-centre residual onto local tone (kills brown bias)
    passes=2,             # repair iterations (field detected once, reused)
    pass_decay=0.6,       # amplitude multiplier per extra pass (gentler each time)
    # soft decision widths (0 = hard threshold, identical to before; raise to
    # trade a little selectivity for temporal stability / less flicker). Baked
    # to the levels tuned in the app for purple flicker reduction.
    edge_soft=0.45,       # edge percentile ramp (fraction of the threshold)
    anom_soft=0.0,        # anomaly-interior ramp (Lab chroma units)
    bright_soft=8.0,      # bright-side L* ramp (L* units)
)


def detect(rgb, **kw):
    """Return {'field', 'lab'} for an RGB array. field = per-pixel fringe weight."""
    p = {**DEFAULTS, **kw}
    lab = to_lab(rgb)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]

    ref_a = gaussian_filter(a, p["ref_sigma"])
    ref_b = gaussian_filter(b, p["ref_sigma"])
    anomaly = hue_weight(a - ref_a, b - ref_b, p["target"], p["hue_power"])
    grow_hp = p["hue_power"] if p["grow_hue_power"] is None else p["grow_hue_power"]
    presence = hue_weight(a, b, p["target"], grow_hp)
    if p["presence_Lref"] is not None:
        presence = presence * np.clip(p["presence_Lref"] / (L + 1.0), 1.0, p["presence_Lcap"])
    detected = np.minimum(maximum_filter(anomaly, size=p["spread_win"]), presence)

    edge = scharr(L)
    thr = np.percentile(edge, p["edge_pct"])
    # soft edge band around the (still global) percentile: pixels near the
    # threshold fade rather than flip, damping the frame-wide percentile breathing
    edge_w = soft_step(edge, thr, p["edge_soft"] * thr)
    lab_id, n = label(edge_w > 0)
    if n:
        sizes = np.bincount(lab_id.ravel()); sizes[0] = 0
        edge_w = edge_w * (sizes >= p["min_edge"])[lab_id]
    # interior of coloured blobs to subtract. Soft path = blur + ramp; width=0
    # falls back to the exact original binary erosion.
    if p["anom_soft"] > 0:
        interior_w = soft_step(gaussian_filter(anomaly, float(p["int_erode"]) + 1.0),
                               p["anomaly_thr"], p["anom_soft"])
    else:
        interior_w = binary_erosion(anomaly > p["anomaly_thr"],
                                    iterations=p["int_erode"]).astype(np.float32)
    gate_w = edge_w * (1.0 - interior_w)
    if p["require_bright"]:
        gate_w = gate_w * soft_step(maximum_filter(L, size=p["bright_win"]),
                                    p["bright_L"], p["bright_soft"])

    gate = gate_w > 0
    if (p["edge_soft"] or p["bright_soft"] or p["anom_soft"]) and gate.any():
        # propagate the soft gate strength from each pixel's nearest gate pixel
        dist, (iy, ix) = distance_transform_edt(~gate, return_indices=True)
        reach = np.exp(-(dist ** 2) / (2.0 * p["tol"] ** 2)) * gate_w[iy, ix]
    else:
        dist = distance_transform_edt(~gate)
        reach = np.exp(-(dist ** 2) / (2.0 * p["tol"] ** 2))
    darkside = np.clip((p["fill_L_hi"] - L) / p["fill_L_soft"], 0, 1)
    field = detected * reach * darkside

    if p["region_guard"]:
        body = binary_erosion(presence > p["region_guard_thr"], iterations=p["region_guard_erode"])
        field = field * ~body
    return {"field": field, "lab": lab}


def repair(lab, field, base_rgb=None, **kw):
    """Neutralise fringe chroma by a field-derived alpha; keep L (no darkening)."""
    p = {**DEFAULTS, **kw}
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]

    ramp = np.clip(field / p["field_ref"], 0, 1)
    floor_w = np.clip(field / p["floor_ramp"], 0, 1)
    alpha = np.clip(p["floor"] * floor_w + (p["strength"] - p["floor"]) * ramp, 0, 1)

    ta, tb = p["target"]
    nrm = (ta * ta + tb * tb) ** 0.5
    ua, ub = ta / nrm, tb / nrm
    ra, rb = a.copy(), b.copy()
    rem = np.clip(ra * ua + rb * ub, 0, None)
    ra, rb = ra - rem * ua, rb - rem * ub
    sig, ds = p["tone_sigma"], p["blur_ds"]
    trust = (1.0 - alpha) + 1e-3
    den = blur(trust, sig, ds) + 1e-6
    ta_ = normconv(a, trust, sig, ds, den)
    tb_ = normconv(b, trust, sig, ds, den)
    if p["normalize"]:
        ra = ra + (ta_ - blur(ra, sig, ds))
        rb = rb + (tb_ - blur(rb, sig, ds))
    fa = (1 - p["fold_mix"]) * ra + p["fold_mix"] * ta_
    fb = (1 - p["fold_mix"]) * rb + p["fold_mix"] * tb_
    a2 = a + alpha * (fa - a)
    b2 = b + alpha * (fb - b)

    out = lab_to_rgb(np.stack([L, a2, b2], -1))
    if base_rgb is not None:
        out = np.where((alpha > 0)[..., None], out, base_rgb)
    return out, alpha


def defringe(rgb, *, return_debug=False, **kw):
    """Detect once, then repair (iterated, each pass gentler). -> (uint8, info).

    info has {field, alpha}. With return_debug it additionally carries the
    detection field for the UI; the field is in info either way.
    """
    p = {**DEFAULTS, **kw}
    d = detect(rgb, **kw)
    field, lab = d["field"], d["lab"]
    base = np.asarray(rgb, np.float32)
    fixed = alpha = None
    for i in range(max(1, p["passes"])):
        amp = p["pass_decay"] ** i
        fixed, alpha = repair(lab, field, base_rgb=base, **{**kw,
                              "strength": p["strength"] * amp,
                              "floor": p["floor"] * amp})
        if i + 1 < p["passes"]:
            lab = rgb2lab(fixed / 255.0)        # float64, matches the original
            base = fixed
    return fixed.astype(np.uint8), {"field": field, "alpha": alpha}
