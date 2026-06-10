"""Defringe — the canonical domain logic for this project.

Two independent passes, run on RGB uint8 frames in CIELAB; both keep L* and carry
untouched pixels through bit-exact:

    green_cast(rgb)   -> (uint8, info)  green fringe cast by red/purple sources
    purple_cast(rgb)  -> (uint8, info)  magenta fringe cast by blown highlights

Both use the same casting-shadow model (caster source -> nearby chroma artifact ->
pull to local tone). They occupy different hue lanes; the full clean chains them:

    g, _ = green_cast(rgb)
    out, _ = purple_cast(g)

(The previous edge-gated purple detector is archived in ../archive/purple_defringe.py.)

Everything here is pure (numpy/skimage/scipy). Video/file I/O lives in `video_io.py`.
"""
from . import green_cast as _green_mod
from . import purple_cast as _purple_cast_mod
from .green_cast import green_cast, GREEN
from .purple_cast import purple_cast, MAGENTA_REF

# config dicts (single source of truth for the tuner's slider defaults)
GREEN_CAST = _green_mod.DEFAULTS
PURPLE_CAST = _purple_cast_mod.DEFAULTS

__all__ = [
    "green_cast", "GREEN", "GREEN_CAST",
    "purple_cast", "MAGENTA_REF", "PURPLE_CAST",
]
