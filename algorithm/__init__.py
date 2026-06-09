"""Defringe — the canonical domain logic for this project.

Two independent passes, run on RGB uint8 frames in CIELAB; both keep L* and carry
untouched pixels through bit-exact:

    green_cast(rgb)  -> (uint8, info)   green fringe cast by red/purple sources
    defringe(rgb)    -> (uint8, info)   purple/magenta fringe near blown highlights

They occupy different hue lanes. The recommended full clean chains them in order:

    g, _ = green_cast(rgb)
    out, _ = defringe(g)

Everything here is pure (numpy/skimage/scipy). Video/file I/O lives in `video_io.py`.
"""
from . import green_cast as _green_mod
from . import purple_defringe as _purple_mod
from .green_cast import green_cast, GREEN
from .purple_defringe import defringe, detect, repair, PURPLE

# config dicts (single source of truth for the tuner's slider defaults)
GREEN_CAST = _green_mod.DEFAULTS
PURPLE_DEFAULTS = _purple_mod.DEFAULTS

__all__ = [
    "green_cast", "GREEN", "GREEN_CAST",
    "defringe", "detect", "repair", "PURPLE", "PURPLE_DEFAULTS",
]
