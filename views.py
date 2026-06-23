import json

import numpy as np
from scipy.ndimage import distance_transform_edt
import matplotlib.cm as cm

import defringe_numpy as alg

CASTER_RED, CASTER_GOLD, FRINGE_GREEN = (230., 60., 50.), (255., 215., 0.), (0., 255., 120.)


def detect_view(base, cast, caster_color, reach):
    dimmed = 0.45 * base
    kept, found = cast.casters_kept, cast.casters_found
    rejected = found & ~kept
    if rejected.any():
        dimmed[rejected] = 0.78 * dimmed[rejected] + 0.22 * np.array(caster_color)
    if kept.any():
        dimmed[kept] = 0.5 * dimmed[kept] + 0.5 * np.array(caster_color)
    lit = cast.alpha > 0.05
    dimmed[lit] = 0.18 * dimmed[lit] + 0.82 * np.array(FRINGE_GREEN)
    out = np.clip(dimmed, 0, 255).astype(np.uint8)
    h, w = base.shape[:2]
    _draw_reach(out, kept, reach * (w * w + h * h) ** 0.5 / 1000.0)   # cast_radius is ‰ of diagonal
    return out


def overlay(rgb, alpha, threshold=0.02):
    tint = cm.turbo(np.clip(alpha, 0, 1))[..., :3] * 255
    tinted = (alpha > threshold)[..., np.newaxis]
    return np.where(tinted, 0.2 * rgb + 0.8 * tint, rgb).astype(np.uint8)


def stat_line(name, cast):
    alpha = cast.alpha
    return f"{name} sel(a>0.1): {100 * np.mean(alpha > 0.1):.3f}%  a-max {alpha.max():.2f}"


def green_wheel_config(reg):
    ids = {e["name"]: e["elem_id"] for e in reg}
    return json.dumps({"maxChroma": 60, "wedges": [
        {"ref": alg.RED_REF, "color": "#c0392b",
         "chroma": ids["caster_min_chroma"], "lo": ids["caster_hue_lo"], "hi": ids["caster_hue_hi"]},
        {"ref": alg.GREEN_REF, "color": "#148f77",
         "chroma": ids["fringe_min_chroma"], "lo": ids["fringe_hue_lo"], "hi": ids["fringe_hue_hi"]},
    ], "sizeReach": {"area": ids["min_area"], "reach": ids["cast_radius"]},
        "repair": {"strength": ids["max_opacity"], "spread": ids["repair_spread"], "feather": ids["feather"]}})


def purple_wheel_config(reg):
    ids = {e["name"]: e["elem_id"] for e in reg}
    return json.dumps({"maxChroma": 60, "layout": "source",
                       "labels": {"left": "Casters", "right": "Shadows"}, "wedges": [
        {"ref": alg.MAGENTA_REF, "color": "#9b59b6",
         "chroma": ids["fringe_min_chroma"], "center": ids["target_hue"], "halfwidth": ids["hue_halfwidth"]},
    ], "source": {"lightness": ids["caster_min_lightness"], "area": ids["min_area"], "reach": ids["cast_radius"]},
        "excessRing": {"threshold": ids["excess_thresh"]},
        "repair": {"strength": ids["max_opacity"], "spread": ids["repair_spread"], "feather": ids["feather"]}})


def _draw_reach(img, caster, reach):
    if not caster.any():
        return
    reach = max(1, int(round(reach)))
    dist = distance_transform_edt(~caster)
    img[(dist >= reach - 0.5) & (dist <= reach + 0.5)] = (255, 255, 255)
