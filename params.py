"""The parameter model: one slider registry that drives everything downstream.

Each slider declares the algorithm key it controls, so a single registry serves the
pipeline call (`build_params` / `_split`), on-disk persistence (`user_defaults.json`,
keyed s0.. and label-guarded), profile import/export, and the JS reset / colour-wheel
(both key off the slider elem_ids). REG is creation order (persistence); GREEN_REG /
PURPLE_REG feed each pass's kwargs.
"""
import json
from pathlib import Path
import tempfile

import gradio as gr

import defringe_numpy as alg

G, PC = alg.GREEN_CAST, alg.PURPLE_CAST          # per-pass built-in defaults (profile fallback)
DEFAULTS_FILE = Path(__file__).with_name("user_defaults.json")


def _load_saved():
    try:
        return json.loads(DEFAULTS_FILE.read_text())   # {persist_key: {"label", "value"}}
    except Exception:
        return {}


SAVED = _load_saved()
REG, GREEN_REG, PURPLE_REG = [], [], []
_sn = [0]


def _elem_id(persist):
    """The slider's DOM id — the one place the `def_` convention is spelled (the JS
    assets match `[id^="def_"]`). Everything Python reads it off the registry entry."""
    return f"def_{persist}"


def S(reg, key, lo, hi, step, label, info=None):
    """Slider for algorithm param `key`, default seeded from the pass's defaults
    then user_defaults.json; registered for the kwargs map, persistence, and reset."""
    defaults = G if reg is GREEN_REG else PC
    assert key in defaults, \
        f"slider {label!r}: {key!r} is not a {'green' if reg is GREEN_REG else 'purple'} param"
    persist = f"s{_sn[0]}"; _sn[0] += 1
    base = defaults[key]                               # the pass's built-in default (profile fallback)
    default = base
    saved = SAVED.get(persist)
    if saved and saved.get("label") == label:          # label guard: ignore stale keys
        default = saved["value"]
    eid = _elem_id(persist)
    comp = gr.Slider(lo, hi, value=default, step=step, label=label, info=info, elem_id=eid)
    entry = {"persist": persist, "elem_id": eid, "label": label, "key": key, "comp": comp, "base": base}
    reg.append(entry); REG.append(entry)
    return comp


def build_params(reg, vals):
    """Map a pass's slider values to its algorithm kwargs (registry order)."""
    return {e["key"]: v for e, v in zip(reg, vals)}


def split(vals):
    """Split the flat slider-value tuple into (green_kwargs, purple_kwargs)."""
    return (build_params(GREEN_REG, vals[:len(GREEN_REG)]),
            build_params(PURPLE_REG, vals[len(GREEN_REG):]))


def _profile_dict(vals):
    return {c["persist"]: {"label": c["label"], "value": v} for c, v in zip(REG, vals)}


def export_profile(*vals):
    """Write the current settings to a JSON file the browser downloads. Does NOT touch the
    active profile — it's just an export you keep on disk and re-import later."""
    path = Path(tempfile.gettempdir()) / "defringe_profile.json"
    path.write_text(json.dumps(_profile_dict(vals), indent=2))
    return str(path)


def import_profile(file):
    """Load a profile JSON: apply it to every slider, copy it to DEFAULTS_FILE so it's the
    active profile on the next app start, and hand RESET_JS a fresh {elem_id: value} map.
    Outputs: one value per REG slider, then the reset blob, then a status line."""
    blank = [gr.update()] * len(REG)
    if not file:
        return blank + ["{}", "no file selected"]
    try:
        raw = json.loads(Path(file).read_text())
    except Exception as e:
        return blank + ["{}", f"✗ couldn't read profile: {e}"]
    DEFAULTS_FILE.write_text(json.dumps(raw, indent=2))   # becomes the active profile on next load
    vals, applied = [], 0
    for c in REG:
        s = raw.get(c["persist"])
        if isinstance(s, dict) and s.get("label") == c["label"]:
            vals.append(s["value"]); applied += 1
        else:
            vals.append(c["base"])                        # not in the profile -> pass default
    client = {c["elem_id"]: v for c, v in zip(REG, vals)}
    return vals + [json.dumps(client), f"✓ loaded {Path(file).name} ({applied} settings) — active profile"]
