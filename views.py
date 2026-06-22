"""Render helpers for the tuner: turn a pass's output + detection info into the images and
colour-wheel config the UI shows. Pure presentation — numpy/matplotlib in, image/JSON out;
no Gradio components, no event wiring."""
import json

import numpy as np
from scipy.ndimage import distance_transform_edt
import matplotlib.cm as cm

import defringe_numpy as alg

CASTER_RED, CASTER_GOLD, FRINGE_GREEN = (230., 60., 50.), (255., 215., 0.), (0., 255., 120.)


def overlay(rgb, alpha, thr=0.02):
    """Tint corrected pixels by their alpha (turbo colormap) over the original frame."""
    over = cm.turbo(np.clip(alpha, 0, 1))[..., :3] * 255
    return np.where((alpha > thr)[..., None], 0.2 * rgb + 0.8 * over, rgb).astype(np.uint8)


def _draw_reach(img, caster, reach):
    """White contour at `reach` px from the casters — the region searched for fringe."""
    if not caster.any():
        return
    reach = max(1, int(round(reach)))
    dist = distance_transform_edt(~caster)
    img[(dist >= reach - 0.5) & (dist <= reach + 0.5)] = (255, 255, 255)


def detect_view(base, cast, caster_color, reach):
    """Detect image over a dimmed frame: accepted casters bright, the ones min_area
    rejected faint, fringe green, and a white ring at the cast reach around casters."""
    dm = 0.45 * base
    caster, raw = cast.caster, cast.caster_raw
    rejected = raw & ~caster
    if rejected.any():
        dm[rejected] = 0.78 * dm[rejected] + 0.22 * np.array(caster_color)   # too small for min_area
    if caster.any():
        dm[caster] = 0.5 * dm[caster] + 0.5 * np.array(caster_color)
    lit = cast.alpha > 0.05
    dm[lit] = 0.18 * dm[lit] + 0.82 * np.array(FRINGE_GREEN)
    out = np.clip(dm, 0, 255).astype(np.uint8)
    H, W = base.shape[:2]
    _draw_reach(out, caster, reach * (W * W + H * H) ** 0.5 / 1000.0)   # reach is ‰ of diagonal -> px
    return out


def stat_line(name, cast):
    a = cast.alpha
    return f"{name} sel(a>0.1): {100 * np.mean(a > 0.1):.3f}%  a-max {a.max():.2f}"


def green_wheel_config(reg):
    """Build the wheel's data-config from the green registry (call after the sliders exist)."""
    gid = {e["key"]: e["elem_id"] for e in reg}
    return json.dumps({"maxChroma": 60, "wedges": [
        {"ref": alg.RED_REF, "color": "#c0392b",
         "chroma": gid["caster_min_chroma"], "lo": gid["caster_hue_lo"], "hi": gid["caster_hue_hi"]},
        {"ref": alg.GREEN_REF, "color": "#148f77",
         "chroma": gid["fringe_min_chroma"], "lo": gid["fringe_hue_lo"], "hi": gid["fringe_hue_hi"]},
    ], "sizeReach": {"area": gid["min_area"], "reach": gid["cast_radius"]},
        "repair": {"strength": gid["max_opacity"], "spread": gid["repair_spread"], "feather": gid["feather"]}})


def purple_wheel_config(reg):
    """Magenta-fringe cone wedge: one wedge around MAGENTA_REF whose axis is Target Hue,
    angular half-width is Hue Range, and inner radius is Minimum Chroma. The caster is a
    brightness threshold (not a hue region), so it isn't on the wheel; Excess Threshold is
    a directional excess, so it stays a slider. Centred on neutral (scene-tone offset omitted)."""
    pid = {e["key"]: e["elem_id"] for e in reg}
    # Same disc geometry & radial scale (60) as the green wheel, so the carved magenta slice lands
    # exactly where the full disc sat -> the wheel stays put across a tab switch.
    return json.dumps({"maxChroma": 60, "layout": "source",
                       "labels": {"left": "Casters", "right": "Shadows"}, "wedges": [
        {"ref": alg.MAGENTA_REF, "color": "#9b59b6",
         "chroma": pid["fringe_min_chroma"], "center": pid["target_hue"], "halfwidth": pid["hue_halfwidth"]},
    ], "source": {"lightness": pid["caster_min_lightness"], "area": pid["min_area"], "reach": pid["cast_radius"]},
        "excessRing": {"threshold": pid["excess_thresh"]},
        "repair": {"strength": pid["max_opacity"], "spread": pid["repair_spread"], "feather": pid["feather"]}})
