import json
from pathlib import Path
import tempfile

import gradio as gr

DEFAULTS_FILE = Path(__file__).with_name("user_defaults.json")


def _load_saved():
    try:
        return json.loads(DEFAULTS_FILE.read_text())
    except Exception:
        return {}


SAVED = _load_saved()
REG, GREEN_REG, PURPLE_REG = [], [], []


def _elem_id(uid):
    return f"def_{uid}"


def slider(spec, reg, tag):
    # uid is pass-qualified ('green_'/'purple_'): the param NAME is the shared algorithm kwarg,
    # but green and purple reuse 11 names, so the slider's identity must stay distinct.
    uid = f"{tag}_{spec.name}"
    eid = _elem_id(uid)
    value = SAVED.get(uid, spec.default)
    comp = gr.Slider(spec.lo, spec.hi, value=value, step=spec.step,
                     label=spec.label, info=spec.help, elem_id=eid)
    entry = {"name": spec.name, "uid": uid, "elem_id": eid, "comp": comp, "default": spec.default}
    reg.append(entry); REG.append(entry)
    return comp


def build_params(reg, vals):
    return {entry["name"]: v for entry, v in zip(reg, vals)}


def split(vals):
    return (build_params(GREEN_REG, vals[:len(GREEN_REG)]),
            build_params(PURPLE_REG, vals[len(GREEN_REG):]))


def export_profile(*vals):
    path = Path(tempfile.gettempdir()) / "defringe_profile.json"
    path.write_text(json.dumps({entry["uid"]: v for entry, v in zip(REG, vals)}, indent=2))
    return str(path)


def import_profile(file):
    blank = [gr.update()] * len(REG)
    if not file:
        return blank + ["{}", "no file selected"]
    try:
        loaded = json.loads(Path(file).read_text())
    except Exception as e:
        return blank + ["{}", f"✗ couldn't read profile: {e}"]
    DEFAULTS_FILE.write_text(json.dumps(loaded, indent=2))   # becomes the active profile on next load
    vals, applied = [], 0
    for entry in REG:
        if entry["uid"] in loaded:
            vals.append(loaded[entry["uid"]]); applied += 1
        else:
            vals.append(entry["default"])
    reset_values = {entry["elem_id"]: v for entry, v in zip(REG, vals)}
    return vals + [json.dumps(reset_values), f"✓ loaded {Path(file).name} ({applied} settings) — active profile"]
