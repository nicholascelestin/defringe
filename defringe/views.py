import json

import numpy as np
from scipy.ndimage import distance_transform_edt
import matplotlib.cm as cm

from . import defringe_numpy as alg

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


def green_wheel_mount(reg):
    bind = _Bindings(reg)
    config = {"maxChroma": 60, "wedges": [
        {"ref": alg.RED_REF, "color": "#c0392b",
         "chroma": bind("caster_min_chroma"), "lo": bind("caster_hue_lo"), "hi": bind("caster_hue_hi")},
        {"ref": alg.GREEN_REF, "color": "#148f77",
         "chroma": bind("fringe_min_chroma"), "lo": bind("fringe_hue_lo"), "hi": bind("fringe_hue_hi")},
    ], "sizeReach": {"area": bind("min_area"), "reach": bind("cast_radius")},
        "repair": {"strength": bind("max_opacity"), "spread": bind("repair_spread"), "feather": bind("feather")}}
    return _wheel_mount(config, bind.sliders)


def purple_wheel_mount(reg):
    bind = _Bindings(reg)
    config = {"maxChroma": 60, "layout": "source",
              "labels": {"left": "Casters", "right": "Shadows"}, "wedges": [
        {"ref": alg.MAGENTA_REF, "color": "#9b59b6",
         "chroma": bind("fringe_min_chroma"), "center": bind("target_hue"), "halfwidth": bind("hue_halfwidth")},
    ], "source": {"lightness": bind("caster_min_lightness"), "area": bind("min_area"), "reach": bind("cast_radius")},
        "excessRing": {"threshold": bind("excess_thresh")},
        "repair": {"strength": bind("max_opacity"), "spread": bind("repair_spread"), "feather": bind("feather")}}
    return _wheel_mount(config, bind.sliders)


class _Bindings:
    # As the config is built, key it by each param's own name (the wheel's opaque key) and record which
    # slider that name maps to — so the host's key->slider map is a byproduct of construction, and no
    # Gradio elem_id ever lands in the wheel's config.
    def __init__(self, reg):
        self._elem_id = {e["name"]: e["elem_id"] for e in reg}
        self.sliders = {}                     # wheel key (param name) -> slider elem_id

    def __call__(self, name):
        self.sliders[name] = self._elem_id[name]
        return name


def _wheel_mount(config, param_map):
    return ("<div class='defringe-wheel-mount' "
            f"data-config='{json.dumps(config)}' "
            f"data-param-map='{json.dumps(param_map)}'></div>")


def _draw_reach(img, caster, reach):
    if not caster.any():
        return
    reach = max(1, int(round(reach)))
    dist = distance_transform_edt(~caster)
    img[(dist >= reach - 0.5) & (dist <= reach + 0.5)] = (255, 255, 255)
