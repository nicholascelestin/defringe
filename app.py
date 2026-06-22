#!/usr/bin/env python3
"""Defringe tuner — a tabbed Gradio app over defringe_algorithm.py.

Source: pick a still or extract a 10s video clip. Green / Purple: tune each pass
(toggleable) live. Temporal: flicker stats over 6 frames. ONNX export: bake the
current settings into a portable model. Every slider declares its algorithm key,
so one registry drives the pipeline call, the persistence, and the live reset.

Run:  .venv/bin/python app.py
"""
import os
import sys
import json
import shutil
import tempfile
import subprocess
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt
import gradio as gr
import matplotlib
matplotlib.use("Agg")               # headless: render plots to arrays, no display
import matplotlib.cm as cm
import matplotlib.pyplot as plt

import video_io
import defringe_algorithm as alg

DEFAULT_SECS = 10        # clip length to grab into memory, in seconds
ONNX_PATH = "cast_defringe.onnx"            # exported model (green->purple cast)
ONNX_PREVIEW = "onnx_preview.mp4"           # last ONNX-on-clip render (gitignored)
G, PC = alg.GREEN_CAST, alg.PURPLE_CAST

if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
    print("⚠ ffmpeg/ffprobe not on PATH — video upload needs them "
          "(brew install ffmpeg / apt install ffmpeg). Images still work.")

CASTER_RED, CASTER_GOLD, FRINGE_GREEN = (230., 60., 50.), (255., 215., 0.), (0., 255., 120.)


def _overlay(rgb, alpha, thr=0.02):
    over = cm.turbo(np.clip(alpha, 0, 1))[..., :3] * 255
    return np.where((alpha > thr)[..., None], 0.2 * rgb + 0.8 * over, rgb).astype(np.uint8)


def _draw_reach(img, caster, reach):
    """White contour at `reach` px from the casters — the region searched for fringe."""
    if not caster.any():
        return
    reach = max(1, int(round(reach)))
    dist = distance_transform_edt(~caster)
    img[(dist >= reach - 0.5) & (dist <= reach + 0.5)] = (255, 255, 255)


def detect_view(base, info, caster_color, reach):
    """Detect image over a dimmed frame: accepted casters bright, the ones min_area
    rejected faint, fringe green, and a white ring at the cast reach around casters."""
    dm = 0.45 * base
    caster, raw = info["caster"], info["caster_raw"]
    rejected = raw & ~caster
    if rejected.any():
        dm[rejected] = 0.78 * dm[rejected] + 0.22 * np.array(caster_color)   # too small for min_area
    if caster.any():
        dm[caster] = 0.5 * dm[caster] + 0.5 * np.array(caster_color)
    lit = info["alpha"] > 0.05
    dm[lit] = 0.18 * dm[lit] + 0.82 * np.array(FRINGE_GREEN)
    out = np.clip(dm, 0, 255).astype(np.uint8)
    H, W = base.shape[:2]
    _draw_reach(out, caster, reach * (W * W + H * H) ** 0.5 / 1000.0)   # reach is ‰ of diagonal -> px
    return out


def stat_line(name, info):
    a = info["alpha"]
    return f"{name} sel(a>0.1): {100 * np.mean(a > 0.1):.3f}%  a-max {a.max():.2f}"


def current_frame(still, clip, idx):
    """The frame the pipeline tunes on: an uploaded still, else the current clip frame."""
    if still is not None:
        return still
    if clip is not None and len(clip):
        return clip[int(np.clip(idx, 0, len(clip) - 1))]
    return None


# --- slider registry: each slider knows its algorithm key, so one map serves the
# pipeline call, persistence (user_defaults.json, keyed s0.., label-guarded), and
# the JS reset. REG is creation order (persistence); GREEN_REG/PURPLE_REG feed the
# per-pass kwargs. ---
DEFAULTS_FILE = Path(__file__).with_name("user_defaults.json")


def _load_saved():
    try:
        return json.loads(DEFAULTS_FILE.read_text())   # {persist_key: {"label", "value"}}
    except Exception:
        return {}


SAVED = _load_saved()
REG, GREEN_REG, PURPLE_REG = [], [], []
_sn = [0]


def S(reg, key, lo, hi, step, label, info=None):
    """Slider for algorithm param `key`, default seeded from the pass's defaults
    then user_defaults.json; registered for the kwargs map, persistence, and reset."""
    persist = f"s{_sn[0]}"; _sn[0] += 1
    base = (G if reg is GREEN_REG else PC)[key]        # the pass's built-in default (profile fallback)
    default = base
    saved = SAVED.get(persist)
    if saved and saved.get("label") == label:        # label guard: ignore stale keys
        default = saved["value"]
    comp = gr.Slider(lo, hi, value=default, step=step, label=label, info=info, elem_id=f"def_{persist}")
    entry = {"persist": persist, "label": label, "key": key, "comp": comp, "base": base}
    reg.append(entry); REG.append(entry)
    return comp


def build_params(reg, vals):
    """Map a pass's slider values to its algorithm kwargs (registry order)."""
    return {e["key"]: v for e, v in zip(reg, vals)}


def restart_app():
    """Re-exec the process: a fresh server on the same port (also picks up code edits).
    The in-memory clip is lost; saved defaults persist on disk."""
    os.execv(sys.executable, [sys.executable, *sys.argv])


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
    client = {f"def_{c['persist']}": v for c, v in zip(REG, vals)}
    return vals + [json.dumps(client), f"✓ loaded {Path(file).name} ({applied} settings) — active profile"]


def _split(vals):
    return build_params(GREEN_REG, vals[:len(GREEN_REG)]), build_params(PURPLE_REG, vals[len(GREEN_REG):])


def run_chain(still, clip, idx, g_on, pc_on, *vals):
    """frame -> [green] -> [purple]; returns the six tab-1 / tab-2 outputs."""
    rgb = current_frame(still, clip, idx)
    if rgb is None:
        return None, None, "no frame", None, None, "no frame"
    green, purple = _split(vals)

    if g_on:
        gout, ginfo = alg.green_cast(rgb, **green)
        g_detect = [(detect_view(rgb, ginfo, CASTER_RED, green["cast_radius"]), "casters & reach"),
                    (_overlay(rgb, ginfo["alpha"]), "alpha")]
        g_stat = stat_line("green", ginfo)
    else:
        gout, g_stat = rgb, "green DISABLED (passthrough)"
        g_detect = [(rgb, "casters (red) + shadows (green)"), (rgb, "alpha")]

    if pc_on:
        pcout, pcinfo = alg.purple_cast(gout, **purple)
        pc_detect = [(detect_view(gout, pcinfo, CASTER_GOLD, purple["cast_radius"]), "casters & reach"),
                     (_overlay(gout, pcinfo["alpha"]), "alpha")]
        pc_stat = stat_line("purple", pcinfo)
    else:
        pcout, pc_stat = gout, "purple DISABLED (passthrough)"
        pc_detect = [(gout, "casters (gold) + fringe (green)"), (gout, "alpha")]

    g_compare = [(rgb, "input"), (gout, "green-corrected → tab 2")]
    pc_compare = [(gout, "green-corrected"), (pcout, "final")]
    return (g_compare, g_detect, g_stat,
            pc_compare, pc_detect, pc_stat)


def on_upload(path):
    """Route an upload: an image becomes the tuning frame; a video reveals the seek /
    seconds / extract controls. Returns updates across the Source tab."""
    hide, off = gr.update(visible=False), gr.update(interactive=False)
    if not path:                                          # cleared
        return (None, None, None, "", None, None,
                hide, hide, hide, hide, hide, hide, off, off)
    if video_io.is_video(path):
        w, h, nb, fps = video_io.probe(path)
        dur = nb / fps if fps else 0
        info = f"video {w}x{h} {fps:.0f}fps {dur:.0f}s — set seconds & seek, then Extract"
        return (None, None, path, info, video_io.read_frame(path, 0.0, max_w=960), None,
                gr.update(visible=True, maximum=max(0, dur), value=0),       # seek
                gr.update(visible=True), gr.update(visible=True),            # secs/extract
                hide, hide, hide, off,                    # clip scrubbers + temporal btn (need a clip)
                gr.update(interactive=True))              # onnx run: a video is enough (whole-video scope)
    img = video_io.read_image(path)
    return (img, None, None, f"image {img.shape[1]}x{img.shape[0]}", None, img,
            hide, hide, hide, hide, hide, hide, off, off)


def seek_preview(video_path, start_sec):
    """Single frame at the seek point, so you can see where the clip will start."""
    if not video_path:
        return None
    return video_io.read_frame(video_path, float(start_sec), max_w=960)


def extract_clip(video_path, start_sec, secs):
    """Decode `secs` seconds from `video_path` into memory at full res; the clip becomes
    the active source (clears any uploaded still)."""
    if not video_path:
        return (None, None, "Upload a video first.", None, gr.update(), gr.update(),
                gr.update(), gr.update(), gr.update(interactive=False), gr.update(interactive=False), 25.0)
    frames, fps = video_io.read_clip(video_path, float(start_sec), float(secs))
    n = len(frames)
    info = (f"{n} frames @ {fps:.0f}fps from {start_sec:.0f}s "
            f"({frames.shape[2]}x{frames.shape[1]}, ~{frames.nbytes/1e6:.0f} MB)")
    scrub = gr.update(maximum=max(0, n - 1), value=0, visible=True)
    return (None, frames, info, frames[0], gr.update(maximum=max(0, n - 1), value=0),
            scrub, scrub, scrub, gr.update(interactive=True), gr.update(interactive=True), fps)


def pick_preview(still, clip, idx):
    f = current_frame(still, clip, idx)
    return f if f is not None else np.zeros((80, 80, 3), np.uint8)


def mirror_frame(v):
    """Keep the four frame sliders (tabs 0/1/2/3) in lockstep."""
    return v, v, v


TEMPORAL_N = 6   # current frame + next 5


def _pair_plot(green_pp, purple_pp, i0):
    """Line plot of per-frame-pair correction flicker (on corrected pixels)."""
    fig, ax = plt.subplots(figsize=(5.2, 2.6), dpi=100)
    xs = range(len(green_pp))
    labels = [f"{i0+k}→{i0+k+1}" for k in xs]
    ax.plot(xs, green_pp, "-o", color="#2ca02c", label="green")
    ax.plot(xs, purple_pp, "-o", color="#9467bd", label="purple")
    ax.set_xticks(list(xs)); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("mean |Δcorrection| /255", fontsize=8)
    ax.set_title("flicker per frame-pair (corrected px)", fontsize=9)
    ax.set_ylim(bottom=0); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


def _stage_metrics(out, inp, alphas, n):
    """Flicker stats for one stage: out vs its input, weighted by its alpha."""
    corr = [out[i] - inp[i] for i in range(n)]
    d_in = [np.abs(inp[i] - inp[i-1]).mean(-1) for i in range(1, n)]
    d_corr = [np.abs(corr[i] - corr[i-1]).mean(-1) for i in range(1, n)]
    reg = [(alphas[i] > 0) | (alphas[i-1] > 0) for i in range(1, n)]
    d_a = [np.abs(alphas[i] - alphas[i-1]) for i in range(1, n)]
    tog = [(alphas[i] > 0.1) != (alphas[i-1] > 0.1) for i in range(1, n)]
    static = [float(d_corr[i][(d_in[i] < 2.0) & reg[i]].mean())
              for i in range(n-1) if ((d_in[i] < 2.0) & reg[i]).any()]
    dalpha = [float(d_a[i][reg[i]].mean()) for i in range(n-1) if reg[i].any()]
    per_pair = [float(d_corr[i][reg[i]].mean()) if reg[i].any() else 0.0 for i in range(n-1)]
    return dict(static=float(np.mean(static)) if static else 0.0,
                dalpha=float(np.mean(dalpha)) if dalpha else 0.0,
                toggle=float(np.mean([t.mean() for t in tog]) * 100),
                cov=float(np.mean([(a > 0).mean() for a in alphas]) * 100),
                per_pair=per_pair)


def temporal_analyze(still, clip, idx, g_on, pc_on, *vals):
    """Run the live pipeline over the current frame + next 5; report flicker."""
    if clip is None or len(clip) == 0:
        return None, None, "Extract a video clip in Tab 0 first.", None
    i0 = int(np.clip(idx, 0, len(clip) - 1))
    sel = clip[i0:i0 + TEMPORAL_N]
    n = len(sel)
    if n < 2:
        return None, None, (f"Frame {i0} is too close to the clip end "
                            f"({n} frame(s)); pick an earlier start frame."), None
    green, purple = _split(vals)
    src, greens, finals, ga, pa = [], [], [], [], []
    for fr in sel:
        if g_on:
            gout, ginfo = alg.green_cast(fr, **green); galpha = ginfo["alpha"]
        else:
            gout, galpha = fr, np.zeros(fr.shape[:2], np.float32)
        if pc_on:
            pout, pinfo = alg.purple_cast(gout, **purple); palpha = pinfo["alpha"]
        else:
            pout, palpha = gout, np.zeros(np.asarray(gout).shape[:2], np.float32)
        src.append(np.asarray(fr, np.float32)); greens.append(np.asarray(gout, np.float32))
        finals.append(np.asarray(pout, np.float32)); ga.append(galpha); pa.append(palpha)

    gm = _stage_metrics(greens, src, ga, n)
    pm = _stage_metrics(finals, greens, pa, n)
    d_in = float(np.mean([np.abs(src[i] - src[i-1]).mean() for i in range(1, n)]))

    # heatmap: per-pixel temporal std of the final correction, isolating
    # algorithm-induced flicker from real scene motion.
    corr = np.stack([finals[i] - src[i] for i in range(n)], 0)
    fmap = corr.std(0).mean(-1)
    fmax = float(fmap.max())
    heat = (cm.turbo(np.clip(fmap / (fmax + 1e-6), 0, 1))[..., :3] * 255).astype(np.uint8)

    plot = _pair_plot(gm["per_pair"], pm["per_pair"], i0)
    gallery = [(finals[k].astype(np.uint8), f"frame {i0 + k}") for k in range(n)]
    stats = (
        f"**Frames {i0}–{i0+n-1}** ({n} frames) · input motion baseline **{d_in:.2f}**/255\n\n"
        f"| stage | static-flicker | Δalpha | toggle % | coverage % |\n"
        f"|---|---|---|---|---|\n"
        f"| green | {gm['static']:.3f} | {gm['dalpha']:.3f} | {gm['toggle']:.2f} | {gm['cov']:.2f} |\n"
        f"| purple | {pm['static']:.3f} | {pm['dalpha']:.3f} | {pm['toggle']:.2f} | {pm['cov']:.2f} |\n\n"
        f"*static-flicker* = correction jump where the input barely moved (the cleanest "
        f"flicker signal); *Δalpha* = mean per-frame alpha change on corrected pixels; "
        f"*toggle* = % of frame flipping in/out of correction per pair. "
        f"Heatmap = temporal std of the final correction (peak **{fmax:.1f}**/255; "
        f"turbo: blue = stable, red = flickering)."
    )
    return heat, plot, stats, gallery


def export_onnx(still, clip, idx, g_on, pc_on, *vals, progress=gr.Progress()):
    """Export the green->purple pipeline to ONNX (opset 17) from the live settings;
    a disabled pass is baked as a no-op via max_opacity=0."""
    progress(0.05, desc="loading torch...")
    try:
        import torch
        from cast_torch import Defringe
    except Exception as e:
        return f"⚠️ torch not available — try `uv sync` to reinstall deps.\n\n`{e}`"
    green, purple = _split(vals)
    if not g_on:
        green["max_opacity"] = 0.0
    if not pc_on:
        purple["max_opacity"] = 0.0
    # Geometry is resolution-relative; the model bakes kernels for a reference size but resamples
    # to it internally, so the export is resolution-invariant (exact at the reference, near-exact
    # elsewhere). Use the frame you're tuning on as that reference (falls back to 1080p).
    frame = current_frame(still, clip, idx)
    ref_hw = tuple(np.asarray(frame).shape[:2]) if frame is not None else (1080, 1920)
    progress(0.3, desc=f"building model for {ref_hw[1]}x{ref_hw[0]}...")
    model = Defringe(green=green, purple=purple, ref_hw=ref_hw).eval()
    dummy = torch.zeros(1, 540, 960, 3, dtype=torch.uint8)
    progress(0.55, desc="exporting ONNX (opset 17)...")
    torch.onnx.export(model, dummy, ONNX_PATH, opset_version=17,
                      input_names=["rgb"], output_names=["out"],
                      dynamic_axes={"rgb": {0: "N", 1: "H", 2: "W"},
                                    "out": {0: "N", 1: "H", 2: "W"}}, dynamo=False)
    progress(1.0, desc="done")
    mb = os.path.getsize(ONNX_PATH) / 1e6
    off = ("" if g_on else " · green OFF") + ("" if pc_on else " · purple OFF")
    return (f"✅ Exported **{ONNX_PATH}** ({mb:.2f} MB) from the current slider settings{off}. "
            f"Resolution-invariant — resamples internally to a {ref_hw[1]}×{ref_hw[0]} reference "
            f"(exact there, near-exact at other sizes); dynamic N/H/W, uint8 in/out.")


# onnxruntime execution providers, best-first, with friendly names the device widget shows.
PROVIDER_LABEL = {"CUDAExecutionProvider": "CUDA", "CoreMLExecutionProvider": "MPS",
                  "DmlExecutionProvider": "DirectML", "CPUExecutionProvider": "CPU"}
PROVIDER_ORDER = ["CUDAExecutionProvider", "CoreMLExecutionProvider", "DmlExecutionProvider",
                  "CPUExecutionProvider"]


def _available_providers():
    try:
        import onnxruntime as ort
        avail = set(ort.get_available_providers())
    except Exception:
        return []
    ordered = [p for p in PROVIDER_ORDER if p in avail]
    if "CPUExecutionProvider" not in ordered:
        ordered.append("CPUExecutionProvider")
    return ordered


def best_device_name():
    """Highest compatible device, without building a session (for the idle widget)."""
    provs = _available_providers()
    if not provs:
        return "CPU (onnxruntime not loaded)"
    return PROVIDER_LABEL.get(provs[0], provs[0])


def _make_session(ort):
    """Build a session on the highest available provider, falling back gracefully through the
    list until one actually loads. Returns (session, friendly_device_name)."""
    err = None
    for prov in _available_providers():
        try:
            provs = [prov] if prov == "CPUExecutionProvider" else [prov, "CPUExecutionProvider"]
            sess = ort.InferenceSession(ONNX_PATH, providers=provs)
            used = sess.get_providers()[0]                 # the one actually in front
            return sess, PROVIDER_LABEL.get(used, used)
        except Exception as e:
            err = e                                        # provider present but wouldn't init -> try the next
    raise err or RuntimeError("no execution provider available")


def _encoder(w, h, fps):
    # Control the RGB->YUV matrix and TAG the output BT.709, else the browser assumes 709 over
    # swscale's 601 default -> the preview colour-shifts vs the gallery stills.
    return ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{w}x{h}", "-r", f"{fps:.0f}", "-color_range", "pc", "-i", "-", "-an",
            "-vf", "scale=in_range=pc:out_range=tv:out_color_matrix=bt709,format=yuv420p,"
                   "setparams=range=tv:colorspace=bt709:color_primaries=bt709:color_trc=bt709",
            "-c:v", "libx264", "-crf", "16", "-preset", "fast", "-pix_fmt", "yuv420p", ONNX_PREVIEW]


def _run_clip(sess, clip, fps, progress):
    n, B, outs = len(clip), 4, []
    for i in progress.tqdm(range(0, n, B), desc="ONNX defringe"):
        outs.append(sess.run(None, {"rgb": np.ascontiguousarray(clip[i:i + B])})[0])
    frames = np.concatenate(outs, 0)
    h, w = frames.shape[1:3]
    proc = subprocess.Popen(_encoder(w, h, fps), stdin=subprocess.PIPE)
    proc.stdin.write(np.ascontiguousarray(frames, np.uint8).tobytes())
    proc.stdin.close(); proc.wait()
    return ONNX_PREVIEW, f"✅ {n} frames @ {fps:.0f}fps → video below ({w}×{h})."


def _run_video(sess, video_path, progress):
    """Stream the whole file through ONNX: ffmpeg-decode -> batched inference -> ffmpeg-encode,
    so a 10-minute source never has to fit in memory."""
    w, h, nb, fps = video_io.probe(video_path)
    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-i", video_path, "-pix_fmt", "rgb24",
                            "-f", "rawvideo", "-"], stdout=subprocess.PIPE)
    enc = subprocess.Popen(_encoder(w, h, fps or 25.0), stdin=subprocess.PIPE)
    fb, B, done = w * h * 3, 4, 0
    while True:
        raw = dec.stdout.read(fb * B)
        if not raw:
            break
        k = len(raw) // fb
        batch = np.frombuffer(raw[:k * fb], np.uint8).reshape(k, h, w, 3)
        enc.stdin.write(np.ascontiguousarray(sess.run(None, {"rgb": np.ascontiguousarray(batch)})[0], np.uint8).tobytes())
        done += k
        if nb:
            progress(min(done / nb, 1.0), desc=f"ONNX whole video — {done}/{nb}")
    dec.stdout.close(); dec.wait()
    enc.stdin.close(); enc.wait()
    return ONNX_PREVIEW, f"✅ whole video — {done} frames @ {fps:.0f}fps → ({w}×{h})."


def run_onnx(clip, fps, video_path, scope, progress=gr.Progress()):
    """Run the exported ONNX over the selected clip or the whole video, on the highest
    compatible device. Returns (video, status, device-widget-update)."""
    if not os.path.exists(ONNX_PATH):
        return None, "Press **Export ONNX** first.", gr.update()
    try:
        import onnxruntime as ort
    except Exception as e:
        return None, f"⚠️ onnxruntime not available — try `uv sync`.\n\n`{e}`", gr.update(value="CPU (unavailable)")
    try:
        sess, device = _make_session(ort)
    except Exception as e:
        return None, f"⚠️ couldn't start a compute device: `{e}`", gr.update()
    dev = gr.update(value=device)
    if scope == "Whole video":
        if not video_path:
            return None, "Upload a video in Tab 0 first (or choose **Selected clip**).", dev
        return (*_run_video(sess, video_path, progress), dev)
    if clip is None or len(clip) == 0:
        return None, "Extract a clip in Tab 0 first (or choose **Whole video**).", dev
    return (*_run_clip(sess, clip, fps, progress), dev)


# Capture-phase handler that intercepts Gradio's per-slider reset (↺) and applies
# window.__defaults instead of the build-time value, firing input+pointerup so the
# pipeline re-runs. Seeds the map from the hidden #defaults_blob textbox per load.
RESET_JS = """
() => {
  const seed = () => {
    try {
      const t = document.querySelector('#defaults_blob textarea, #defaults_blob input');
      if (t) window.__defaults = JSON.parse(t.value || '{}');
    } catch (e) {}
  };
  seed();
  if (!window.__resetHijack) {
    window.__resetHijack = true;
    document.addEventListener('click', (e) => {
      const btn = e.target.closest && e.target.closest('.reset-button');
      if (!btn) return;
      const block = btn.closest('[id^="def_"]');
      if (!block) return;
      seed();                                      // re-read blob: load-time seed can race empty
      const d = (window.__defaults || {})[block.id];
      if (d === undefined) return;                 // unknown -> let native reset run
      e.preventDefault(); e.stopImmediatePropagation();
      block.querySelectorAll('input').forEach(inp => {
        inp.value = d;
        inp.dispatchEvent(new Event('input', {bubbles: true}));
        inp.dispatchEvent(new Event('pointerup', {bubbles: true}));
      });
    }, true);
  }
}
"""

WHEEL_JS = Path(__file__).with_name("defringe_wheel.js").read_text()


def acc_head(title, target):
    """Header for a custom CSS collapsible whose body is `#target` (a `.acc-body` column).
    Unlike gr.Accordion it never unmounts its body, so wheel-bound sliders stay readable.
    Styled (ACC_CSS) to match gr.Accordion's header exactly."""
    return gr.HTML(f'<div class="acc-head" data-target="{target}">'
                   f'<span>{title}</span><span class="chev">▼</span></div>')


# Toggle a custom collapsible: flip `.open` on the body and its header (delegated, so it
# works for panels that mount later on tab switch). CSS in ACC_CSS hides bodies by default.
ACCORDION_JS = """
() => {
  if (window.__accWired) return;
  window.__accWired = true;
  document.addEventListener('click', (e) => {
    const h = e.target.closest('.acc-head');
    if (!h) return;
    const body = document.getElementById(h.getAttribute('data-target'));
    if (!body) return;
    h.classList.toggle('open', body.classList.toggle('open'));
  });
}
"""

ACC_CSS = """
/* .gacc mimics a gr.Accordion .block; .acc-head mimics its .label-wrap button. */
.gacc{border:1px solid var(--border-color-primary);border-radius:var(--block-radius);
  background:var(--block-background-fill);padding:var(--block-padding);gap:var(--spacing-lg,8px);}
.gacc .prose{margin:0;}
.gacc > div.block{padding:0;border:0;background:none;}   /* header gr.HTML's block: no pad, no 3px focus-border... */
.gacc .html-container{padding:0;}                         /* ...and its inner container adds no padding */
.acc-head{display:flex;justify-content:space-between;align-items:center;width:100%;
  cursor:pointer;user-select:none;color:var(--body-text-color);
  font-size:var(--block-title-text-size,14px);font-weight:var(--block-title-text-weight,400);}
.acc-head .chev{font-size:14px;transition:transform .15s ease;transform:rotate(90deg);}
.acc-head.open .chev{transform:rotate(0deg);}
.acc-body{display:none;}
.acc-body.open{display:block;}
.hidden-blob{display:none !important;}   /* mounted (so RESET_JS can read it) but invisible */
"""


def green_wheel_config():
    """Build the wheel's data-config from GREEN_REG (call after the sliders exist)."""
    gid = {e["key"]: f"def_{e['persist']}" for e in GREEN_REG}
    return json.dumps({"maxChroma": 60, "wedges": [
        {"ref": alg.RED_REF, "color": "#c0392b",
         "chroma": gid["caster_min_chroma"], "lo": gid["caster_hue_lo"], "hi": gid["caster_hue_hi"]},
        {"ref": alg.GREEN_REF, "color": "#148f77",
         "chroma": gid["fringe_min_chroma"], "lo": gid["fringe_hue_lo"], "hi": gid["fringe_hue_hi"]},
    ], "sizeReach": {"area": gid["min_area"], "reach": gid["cast_radius"]},
        "repair": {"strength": gid["max_opacity"], "spread": gid["repair_spread"], "feather": gid["feather"]}})


def purple_wheel_config():
    """Magenta-fringe cone wedge: one wedge around MAGENTA_REF whose axis is Target Hue,
    angular half-width is Hue Range, and inner radius is Minimum Chroma. The caster is a
    brightness threshold (not a hue region), so it isn't on the wheel; Excess Threshold is
    a directional excess, so it stays a slider. Centred on neutral (scene-tone offset omitted)."""
    pid = {e["key"]: f"def_{e['persist']}" for e in PURPLE_REG}
    # Same disc geometry & radial scale (60) as the green wheel, so the carved magenta slice lands
    # exactly where the full disc sat -> the wheel stays put across a tab switch.
    return json.dumps({"maxChroma": 60, "layout": "source",
                       "labels": {"left": "Casters", "right": "Shadows"}, "wedges": [
        {"ref": alg.MAGENTA_REF, "color": "#9b59b6",
         "chroma": pid["fringe_min_chroma"], "center": pid["target_hue"], "halfwidth": pid["hue_halfwidth"]},
    ], "source": {"lightness": pid["caster_min_lightness"], "area": pid["min_area"], "reach": pid["cast_radius"]},
        "excessRing": {"threshold": pid["excess_thresh"]},
        "repair": {"strength": pid["max_opacity"], "spread": pid["repair_spread"], "feather": pid["feather"]}})


with gr.Blocks(title="Defringe tuner") as demo:
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("# Defringe tuner\n"
                        "Removes green and purple fringe from videos, touching just the chroma channels.")
        with gr.Column(scale=0, min_width=330):
            with gr.Row():
                save_prof_btn = gr.DownloadButton("💾 Save Profile", size="sm", min_width=96)
                load_prof_btn = gr.UploadButton("📂 Load Profile", file_types=[".json"], type="filepath", size="sm", min_width=96)
                restart_btn = gr.Button("↻ Restart app", size="sm", min_width=96)
            save_def_stat = gr.Markdown("")
    # CSS-hidden, NOT visible=False: Gradio 6 unmounts invisible components, and RESET_JS
    # (+ the wheel's reset) seed window.__defaults by reading this element from the DOM.
    defaults_blob = gr.Textbox(elem_id="defaults_blob", elem_classes="hidden-blob")
    clip_state = gr.State(None)     # extracted video clip (frames)
    still_state = gr.State(None)    # uploaded image (when the source is a still)
    video_state = gr.State(None)    # uploaded video path (for seek/extract)
    fps_state = gr.State(25.0)      # the extracted clip's fps (for ONNX re-encode)

    with gr.Tab("0 · Source"):
        gr.Markdown("Upload an **image** to tune on directly, or a **video** to seek into and "
                    "extract a clip.")
        with gr.Row():
            with gr.Column(scale=1):
                t0_upload = gr.File(file_types=["image", "video"], type="filepath", label="source — image or video")
                t0_seek = gr.Slider(0, 1, value=0, step=1, label="seek (s)", visible=False, info="Where in the video to start the clip.")
                t0_secs = gr.Slider(1, 30, value=DEFAULT_SECS, step=1, label="seconds to grab into memory", visible=False, info="Full-res clip length decoded into RAM (longer = more memory).")
                t0_extract = gr.Button("Extract clip → memory", variant="primary", visible=False)
                t0_info = gr.Textbox(label="info", interactive=False)
                t0_frame = gr.Slider(0, 249, value=0, step=1, label="frame (within clip)", info="Which clip frame to preview and tune on.")
            with gr.Column(scale=2):
                t0_seekimg = gr.Image(label="seek-point preview — where the clip starts", height=300)
                t0_preview = gr.Image(label="current frame (feeds tabs 1 & 2)", height=300)

    with gr.Tab("1 · Green Defringe"):
        gr.Markdown("Removes green fringe cast by saturated warm (red/purple) sources — "
                    "the first pass in the chain.")
        with gr.Row():
            with gr.Column(scale=1):
                g_on = gr.Checkbox(True, label="Enable", info="Turn the green pass on/off (off = passes straight to the purple tab).")
                green_wheel = gr.HTML("<div class='defringe-wheel-mount'></div>")
                gr.Markdown("<div style='text-align:center'><sub>Drag any handle to adjust — the sliders below "
                            "mirror it.</sub></div>")
                # Caster/shadow hue+chroma are driven by the wheel above, which reads & writes
                # these sliders' DOM inputs — so they must stay mounted. A gr.Accordion unmounts
                # its body when closed (blanking the wheel), so these use a custom CSS collapsible
                # (acc_head + acc-body): collapsing just sets display:none, keeping them in the DOM.
                with gr.Column(elem_classes="gacc"):
                    acc_head("Casters (warm)", "acc-g-casters")
                    with gr.Column(elem_id="acc-g-casters", elem_classes="acc-body"):
                        S(GREEN_REG, "caster_min_chroma", 0, 50, 0.5, "Minimum Chroma", "How saturated a warm source must be to count as a fringe-caster. Lower = more casters.")
                        S(GREEN_REG, "caster_hue_lo", -90, 0, 1, "Hue Floor", "Low edge of the caster hue range; lower reaches toward purple.")
                        S(GREEN_REG, "caster_hue_hi", 0, 90, 1, "Hue Ceiling", "High edge of the caster hue range; higher reaches toward orange.")
                with gr.Column(elem_classes="gacc"):
                    acc_head("Shadows (cool)", "acc-g-shadows")
                    with gr.Column(elem_id="acc-g-shadows", elem_classes="acc-body"):
                        S(GREEN_REG, "fringe_min_chroma", 0, 30, 0.1, "Minimum Chroma", "How saturated a pixel must be to count as green fringe. Lower = catch fainter fringe.")
                        S(GREEN_REG, "fringe_hue_lo", -140, 0, 1, "Hue Floor", "Low edge of the green-fringe hue range; lower reaches toward green/yellow.")
                        S(GREEN_REG, "fringe_hue_hi", 0, 140, 1, "Hue Ceiling", "High edge of the green-fringe hue range; higher reaches toward blue/violet.")
                # Size & Reach and Repair are now driven by the wheel's blobs too, so they also
                # use the mounted custom collapsible (a gr.Accordion would unmount and blank them).
                with gr.Column(elem_classes="gacc"):
                    acc_head("Size & Reach", "acc-g-size")
                    with gr.Column(elem_id="acc-g-size", elem_classes="acc-body"):
                        S(GREEN_REG, "min_area", 0, 100, 0.5, "Minimum Area", "Ignore caster blobs smaller than this — ppm of frame area (resolution-independent).")
                        S(GREEN_REG, "cast_radius", 0, 30, 0.1, "Cast Reach", "How far from a caster to look for its green fringe — ‰ of the frame diagonal.")
                with gr.Column(elem_classes="gacc"):
                    acc_head("Repair", "acc-g-repair")
                    with gr.Column(elem_id="acc-g-repair", elem_classes="acc-body"):
                        S(GREEN_REG, "max_opacity", 0, 1, 0.05, "Maximum Strength", "Cap on correction opacity. Higher = more aggressive.")
                        S(GREEN_REG, "repair_spread", 0, 5, 0.05, "Repair Spread", "Grow the corrected region outward before feathering — ‰ of the frame diagonal.")
                        S(GREEN_REG, "feather", 0, 5, 0.05, "Feather", "Soften/blur the correction's edge — ‰ of the frame diagonal.")
                with gr.Accordion("Advanced", open=False):
                    S(GREEN_REG, "area_softness", 0, 1, 0.05, "Area Softness", "Fade marginal-size casters in/out instead of popping — for temporal stability. 0 = hard cutoff.")
                    S(GREEN_REG, "radius_softness", 0, 1, 0.05, "Reach Softness", "Fade the fringe out with distance instead of a hard edge — for temporal stability. 0 = hard cutoff.")
                    S(GREEN_REG, "full_strength_span", 1, 30, 0.5, "Full-Strength Span", "Extra chroma above Minimum Chroma that reaches full correction. Wider = gentler.")
                    S(GREEN_REG, "tone_correction_radius", 0, 25, 0.1, "Tone Correction Radius", "How far out to sample the clean colour fringe is pulled toward — ‰ of the frame diagonal.")
                    S(GREEN_REG, "tone_directionality", 0, 1, 0.05, "Tone Directionality", "Pull repair colour from the cast side, not the caster. 0 = all directions, 1 = away from caster.")
            with gr.Column(scale=2):
                t1_frame = gr.Slider(0, 249, value=0, step=1, label="frame (within clip)", visible=False)
                g_stat = gr.Textbox(label="stats", interactive=False)
                g_detect = gr.Gallery(label="casters + alpha",
                                      columns=2, object_fit="contain", preview=False)
                g_compare = gr.Gallery(label="input ↔ green-corrected (→ tab 2)",
                                       columns=2, object_fit="contain", preview=False)

    with gr.Tab("2 · Purple Defringe"):
        gr.Markdown("Removes magenta/violet fringe cast by blown highlights — "
                    "runs on top of the green pass's output.")
        with gr.Row():
            with gr.Column(scale=1):
                pc_on = gr.Checkbox(True, label="Enable", info="Turn the purple pass on/off (runs after the green pass).")
                purple_wheel = gr.HTML("<div class='defringe-wheel-mount'></div>")
                gr.Markdown("<div style='text-align:center'><sub>Drag any handle to adjust — the sliders below "
                            "mirror it.</sub></div>")
                # Every wheel-bound group must stay mounted so the wheel can read & write it, so all
                # use the custom CSS collapsible (a gr.Accordion unmounts its body when closed, blanking
                # the wheel). The source-highlight blob plays the caster: brightness = Minimum Lightness,
                # radius = Minimum Area, ring = Cast Reach.
                with gr.Column(elem_classes="gacc"):
                    acc_head("Source highlight (caster)", "acc-p-source")
                    with gr.Column(elem_id="acc-p-source", elem_classes="acc-body"):
                        S(PURPLE_REG, "caster_min_lightness", 50, 100, 1, "Minimum Lightness", "How bright a highlight must be to count as a fringe-caster.")
                        S(PURPLE_REG, "min_area", 0, 150, 0.5, "Minimum Area", "Ignore highlight blobs smaller than this — ppm of frame area (resolution-independent).")
                        S(PURPLE_REG, "cast_radius", 0, 30, 0.1, "Cast Reach", "How far from a highlight to look for its magenta fringe — ‰ of the frame diagonal.")
                with gr.Column(elem_classes="gacc"):
                    acc_head("Shadows (magenta)", "acc-p-shadows")
                    with gr.Column(elem_id="acc-p-shadows", elem_classes="acc-body"):
                        gr.Markdown("<sub>a magenta *excess* over the scene's overall lighting tone</sub>")
                        S(PURPLE_REG, "fringe_min_chroma", 0, 15, 0.1, "Minimum Chroma", "Chroma floor — pixels below this are never flagged (noise rejection).")
                        S(PURPLE_REG, "target_hue", -45, 45, 1, "Target Hue", "Shift the detected fringe colour: 0 = magenta, higher = toward red, lower = toward violet/blue.")
                        S(PURPLE_REG, "excess_thresh", 0, 20, 0.5, "Excess Threshold", "How much more magenta than the scene's overall tone a pixel must be to count as fringe. Higher = stricter.")
                        S(PURPLE_REG, "hue_halfwidth", 5, 90, 1, "Hue Range (±°)", "How wide a band of hues around Target Hue counts as fringe. 90 = essentially no limit; lower narrows it to colours nearer the target.")
                with gr.Column(elem_classes="gacc"):
                    acc_head("Repair", "acc-p-repair")
                    with gr.Column(elem_id="acc-p-repair", elem_classes="acc-body"):
                        S(PURPLE_REG, "max_opacity", 0, 1, 0.05, "Maximum Strength", "Cap on correction opacity. Higher = more aggressive.")
                        S(PURPLE_REG, "repair_spread", 0, 5, 0.05, "Repair Spread", "Grow the corrected region outward before feathering — ‰ of the frame diagonal.")
                        S(PURPLE_REG, "feather", 0, 5, 0.05, "Feather", "Soften/blur the correction's edge — ‰ of the frame diagonal.")
                with gr.Accordion("Advanced", open=False):
                    S(PURPLE_REG, "area_softness", 0, 1, 0.05, "Area Softness", "Fade marginal highlights in/out instead of popping — for temporal stability. 0 = hard cutoff.")
                    S(PURPLE_REG, "radius_softness", 0, 1, 0.05, "Reach Softness", "Fade the fringe out with distance instead of a hard edge — for temporal stability. 0 = hard cutoff.")
                    S(PURPLE_REG, "hue_softness", 0, 45, 1, "Hue Range Softness", "Feather (°) on the Hue Range edge — ramp the cutoff instead of a hard boundary, for temporal stability. 0 = hard edge.")
                    S(PURPLE_REG, "full_strength_span", 1, 30, 0.5, "Full-Strength Span", "Extra excess above the threshold that reaches full correction. Wider = gentler.")
                    S(PURPLE_REG, "tone_correction_radius", 0, 25, 0.1, "Tone Correction Radius", "How far out to sample the clean colour fringe is pulled toward — ‰ of the frame diagonal.")
                    S(PURPLE_REG, "tone_directionality", 0, 1, 0.05, "Tone Directionality", "Pull repair colour from the fringe side, not the highlight. 0 = all directions, 1 = away from highlight.")
            with gr.Column(scale=2):
                t2_frame = gr.Slider(0, 249, value=0, step=1, label="frame (within clip)", visible=False)
                pc_stat = gr.Textbox(label="stats", interactive=False)
                pc_detect = gr.Gallery(label="casters + alpha",
                                       columns=2, object_fit="contain", preview=False)
                pc_compare = gr.Gallery(label="green-corrected ↔ final",
                                        columns=2, object_fit="contain", preview=False)

    with gr.Tab("3 · Temporal Analysis"):
        gr.Markdown("Measures how much the correction flickers frame-to-frame under your current "
                    "settings. Extract a clip, then **Analyze** to spot instability before exporting.")
        with gr.Row():
            with gr.Column(scale=1):
                t3_frame = gr.Slider(0, 249, value=0, step=1, label="start frame (within clip)", visible=False)
                temporal_btn = gr.Button("Analyze flicker (6 frames)", variant="primary", interactive=False)
                temporal_stats = gr.Markdown("Extract a clip in Tab 0, then press **Analyze**.")
            with gr.Column(scale=2):
                temporal_gallery = gr.Gallery(label="corrected frames — click an image, then arrow through",
                                              columns=6, height=300, object_fit="contain", preview=True)
                temporal_map = gr.Image(label="flicker heatmap — temporal std of the correction", height=320)
                temporal_plot = gr.Image(label="flicker per frame-pair", height=260)

    with gr.Tab("4 · Run"):
        gr.Markdown("Bake your current settings into a portable ONNX model, then run it over the "
                    "selected clip or the whole video.")
        with gr.Row():
            with gr.Column(scale=1):
                onnx_export_btn = gr.Button("① Export ONNX (current settings)", variant="primary")
                onnx_export_stat = gr.Markdown("")
                onnx_scope = gr.Radio(["Selected clip", "Whole video"], value="Selected clip",
                                      label="Run over")
                onnx_run_btn = gr.Button("② Run → video", variant="primary", interactive=False)
                onnx_run_stat = gr.Markdown("Upload a video / extract a clip (Tab 0), Export, then Run.")
            with gr.Column(scale=2):
                onnx_device = gr.Textbox(label="compute device", value=best_device_name(),
                                         interactive=False, max_lines=1)
                onnx_video = gr.Video(label="ONNX output", height=420)

    # REGs are fully built now, so each wheel's slider element IDs resolve.
    green_wheel.value = f"<div class='defringe-wheel-mount' data-config='{green_wheel_config()}'></div>"
    purple_wheel.value = f"<div class='defringe-wheel-mount' data-config='{purple_wheel_config()}'></div>"

    ins = [still_state, clip_state, t0_frame, g_on, pc_on,
           *[e["comp"] for e in GREEN_REG], *[e["comp"] for e in PURPLE_REG]]
    outs = [g_compare, g_detect, g_stat,
            pc_compare, pc_detect, pc_stat]

    for c in ins:
        (c.release if isinstance(c, gr.Slider) else c.change)(run_chain, ins, outs)

    upload_outs = [still_state, clip_state, video_state, t0_info, t0_seekimg, t0_preview,
                   t0_seek, t0_secs, t0_extract, t1_frame, t2_frame, t3_frame,
                   temporal_btn, onnx_run_btn]
    t0_upload.change(on_upload, [t0_upload], upload_outs).then(run_chain, ins, outs)
    t0_seek.release(seek_preview, [video_state, t0_seek], [t0_seekimg])
    t0_extract.click(extract_clip, [video_state, t0_seek, t0_secs],
                     [still_state, clip_state, t0_info, t0_preview, t0_frame,
                      t1_frame, t2_frame, t3_frame, temporal_btn, onnx_run_btn, fps_state]) \
              .then(run_chain, ins, outs)
    t0_frame.change(pick_preview, [still_state, clip_state, t0_frame], [t0_preview])

    # Frame scrubber mirrored into tabs 1/2/3. .release is user-only, so propagating a
    # value never re-triggers — no sync loop. run_chain reads the frame from t0_frame.
    t0_frame.release(mirror_frame, [t0_frame], [t1_frame, t2_frame, t3_frame])
    t1_frame.release(mirror_frame, [t1_frame], [t0_frame, t2_frame, t3_frame]).then(run_chain, ins, outs)
    t2_frame.release(mirror_frame, [t2_frame], [t0_frame, t1_frame, t3_frame]).then(run_chain, ins, outs)
    t3_frame.release(mirror_frame, [t3_frame], [t0_frame, t1_frame, t2_frame]).then(run_chain, ins, outs)
    temporal_btn.click(temporal_analyze, ins,
                       [temporal_map, temporal_plot, temporal_stats, temporal_gallery])
    onnx_export_btn.click(export_onnx, ins, onnx_export_stat)
    onnx_run_btn.click(run_onnx, [clip_state, fps_state, video_state, onnx_scope],
                       [onnx_video, onnx_run_stat, onnx_device])

    # Seed the reset blob with the constructed values so reset is correct from first
    # load; on save, refresh window.__defaults so reset retargets without a restart.
    reg_comps = [c["comp"] for c in REG]
    defaults_blob.value = json.dumps({f"def_{c['persist']}": c["comp"].value for c in REG})
    save_prof_btn.click(export_profile, reg_comps, save_prof_btn)
    # Load: apply to sliders + persist as active profile, then retarget reset / redraw the wheels,
    # then re-run the pipeline with the loaded values.
    load_prof_btn.upload(import_profile, [load_prof_btn], reg_comps + [defaults_blob, save_def_stat]) \
                 .then(None, defaults_blob, None,
                       js="(b) => { window.__defaults = JSON.parse(b || '{}'); document.querySelectorAll('defringe-wheel').forEach(w => w.draw && w.draw()); }") \
                 .then(run_chain, ins, outs)
    restart_btn.click(restart_app)
    demo.load(run_chain, ins, outs)
    demo.load(None, None, None, js=RESET_JS)     # per-slider reset -> saved defaults
    demo.load(None, None, None, js=ACCORDION_JS) # custom collapsibles (wheel-safe)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7862, show_error=True, css=ACC_CSS,
                head=f"<script>{WHEEL_JS}</script>")
