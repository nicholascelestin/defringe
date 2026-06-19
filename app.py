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
import subprocess
from pathlib import Path

import numpy as np
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


def detect_view(base, info, caster_color):
    """Tint casters and corrected pixels over a dimmed frame (the detect image)."""
    dm = 0.45 * base
    if info["caster"].any():
        dm[info["caster"]] = 0.5 * dm[info["caster"]] + 0.5 * np.array(caster_color)
    lit = info["alpha"] > 0.05
    dm[lit] = 0.18 * dm[lit] + 0.82 * np.array(FRINGE_GREEN)
    return np.clip(dm, 0, 255).astype(np.uint8)


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
    default = (G if reg is GREEN_REG else PC)[key]
    saved = SAVED.get(persist)
    if saved and saved.get("label") == label:        # label guard: ignore stale keys
        default = saved["value"]
    comp = gr.Slider(lo, hi, value=default, step=step, label=label, info=info, elem_id=f"def_{persist}")
    entry = {"persist": persist, "label": label, "key": key, "comp": comp}
    reg.append(entry); REG.append(entry)
    return comp


def build_params(reg, vals):
    """Map a pass's slider values to its algorithm kwargs (registry order)."""
    return {e["key"]: v for e, v in zip(reg, vals)}


def restart_app():
    """Re-exec the process: a fresh server on the same port (also picks up code edits).
    The in-memory clip is lost; saved defaults persist on disk."""
    os.execv(sys.executable, [sys.executable, *sys.argv])


def save_defaults(*vals):
    """Persist current slider values and hand the client a fresh {elem_id: value}
    map (RESET_JS consumes it to retarget per-slider reset without a restart)."""
    data = {c["persist"]: {"label": c["label"], "value": v} for c, v in zip(REG, vals)}
    DEFAULTS_FILE.write_text(json.dumps(data, indent=2))
    client = {f"def_{c['persist']}": v for c, v in zip(REG, vals)}
    return json.dumps(client), f"✓ saved {len(data)} settings as default"


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
        g_detect = [(detect_view(rgb, ginfo, CASTER_RED), "casters (red) + shadows (green)"),
                    (_overlay(rgb, ginfo["alpha"]), "alpha")]
        g_stat = stat_line("green", ginfo)
    else:
        gout, g_stat = rgb, "green DISABLED (passthrough)"
        g_detect = [(rgb, "casters (red) + shadows (green)"), (rgb, "alpha")]

    if pc_on:
        pcout, pcinfo = alg.purple_cast(gout, **purple)
        pc_detect = [(detect_view(gout, pcinfo, CASTER_GOLD), "casters (gold) + fringe (green)"),
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
                hide, hide, hide, off, off)               # clip scrubbers + clip-only buttons (need a clip)
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
    progress(0.3, desc="building model from current settings...")
    model = Defringe(green=green, purple=purple).eval()
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
            f"Dynamic N/H/W, uint8 in/out. (~0.1 mean ΔRGB vs the exact numpy.)")


def run_onnx_clip(clip, fps, progress=gr.Progress()):
    """Run the exported ONNX over the whole in-memory clip; return a playable video."""
    if clip is None or len(clip) == 0:
        return None, "Extract a video clip in Tab 0 first."
    if not os.path.exists(ONNX_PATH):
        return None, "Press **Export ONNX** first."
    try:
        import onnxruntime as ort
    except Exception as e:
        return None, f"⚠️ onnxruntime not available — try `uv sync`.\n\n`{e}`"
    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    n, B, outs = len(clip), 4, []
    for i in progress.tqdm(range(0, n, B), desc="ONNX defringe"):
        outs.append(sess.run(None, {"rgb": np.ascontiguousarray(clip[i:i + B])})[0])
    frames = np.concatenate(outs, 0)
    h, w = frames.shape[1:3]
    proc = subprocess.Popen(
        # Control the RGB->YUV matrix and TAG the output BT.709, else the browser
        # assumes 709 over swscale's 601 default -> the preview colour-shifts vs the
        # gallery stills. (Same fix as the Colab notebook's COLOR_MODE="bt709".)
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{w}x{h}", "-r", f"{fps:.0f}", "-color_range", "pc", "-i", "-", "-an",
         "-vf", "scale=in_range=pc:out_range=tv:out_color_matrix=bt709,format=yuv420p,"
                "setparams=range=tv:colorspace=bt709:color_primaries=bt709:color_trc=bt709",
         "-c:v", "libx264", "-crf", "16", "-preset", "fast", "-pix_fmt", "yuv420p", ONNX_PREVIEW],
        stdin=subprocess.PIPE)
    proc.stdin.write(np.ascontiguousarray(frames, np.uint8).tobytes())
    proc.stdin.close(); proc.wait()
    return ONNX_PREVIEW, f"✅ {n} frames @ {fps:.0f}fps through ONNX → video below ({w}×{h})."


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

# Relocate each Gradio `info=` hint into a native HTML `title` tooltip and hide the
# inline text — real hover tooltips at zero layout cost, degrading to the small text
# if the script never runs.
TOOLTIP_JS = """
() => {
  const relocate = () => {
    document.querySelectorAll('.info-text').forEach(el => {
      if (el.dataset.tipDone) return;
      const txt = el.textContent.trim();
      if (!txt) return;
      const host = el.closest('.block') || el.parentElement;
      if (!host) return;
      host.setAttribute('title', txt);   // native hover tooltip on the whole control
      el.style.display = 'none';         // reclaim the inline space; label stays visible
      el.dataset.tipDone = '1';
    });
  };
  relocate();
  new MutationObserver(relocate).observe(document.body, {childList: true, subtree: true});
}
"""

with gr.Blocks(title="Defringe tuner") as demo:
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("# Defringe tuner\n"
                        "Remove green & purple colour fringing from stills and video — only chroma is "
                        "touched, lightness stays put.\n\n"
                        "**Flow:** pick a frame in **Source** → tune **Green**, then **Purple** "
                        "(each toggleable) → check **Temporal** for flicker → **ONNX export** to ship. "
                        "Hover any control for what it does.")
        with gr.Column(scale=0, min_width=300):
            with gr.Row():
                save_def_btn = gr.Button("💾 Save settings", size="sm")
                restart_btn = gr.Button("↻ Restart app", size="sm")
            save_def_stat = gr.Markdown("")
    defaults_blob = gr.Textbox(visible=False, elem_id="defaults_blob")
    clip_state = gr.State(None)     # extracted video clip (frames)
    still_state = gr.State(None)    # uploaded image (when the source is a still)
    video_state = gr.State(None)    # uploaded video path (for seek/extract)
    fps_state = gr.State(25.0)      # the extracted clip's fps (for ONNX re-encode)

    with gr.Tab("0 · Source"):
        gr.Markdown("Upload an **image** to tune on directly, or a **video** to seek into and "
                    "extract a clip. This feeds the Green and Purple tabs.")
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
                gr.Markdown("**Casters (warm)**")
                S(GREEN_REG, "caster_min_chroma", 0, 30, 0.5, "Minimum Chroma", "How saturated a warm source must be to count as a fringe-caster. Lower = more casters.")
                S(GREEN_REG, "caster_hue_lo", -90, 0, 1, "Hue Floor", "Low edge of the caster hue range; lower reaches toward purple.")
                S(GREEN_REG, "caster_hue_hi", 0, 90, 1, "Hue Ceiling", "High edge of the caster hue range; higher reaches toward orange.")
                S(GREEN_REG, "min_area", 0, 300, 1, "Minimum Area", "Ignore caster blobs smaller than this (px) to reject noise.")
                S(GREEN_REG, "cast_radius", 2, 60, 1, "Cast Reach", "How far from a caster (px) to look for its green fringe.")
                gr.Markdown("**Shadows (cool)**")
                S(GREEN_REG, "fringe_min_chroma", 0, 15, 0.1, "Minimum Chroma", "How saturated a pixel must be to count as green fringe. Lower = catch fainter fringe.")
                S(GREEN_REG, "fringe_hue_lo", -140, 0, 1, "Hue Floor", "Low edge of the green-fringe hue range; lower reaches toward green/yellow.")
                S(GREEN_REG, "fringe_hue_hi", 0, 140, 1, "Hue Ceiling", "High edge of the green-fringe hue range; higher reaches toward blue/violet.")
                gr.Markdown("**Repair**")
                S(GREEN_REG, "max_opacity", 0, 1, 0.05, "Maximum Strength", "Cap on correction opacity. Higher = more aggressive.")
                S(GREEN_REG, "repair_spread", 0, 20, 1, "Repair Spread", "Grow the corrected region outward by this many px before feathering.")
                S(GREEN_REG, "feather", 0, 5, 0.1, "Feather", "Soften/blur the correction's edge (px).")
                with gr.Accordion("Advanced", open=False):
                    S(GREEN_REG, "area_softness", 0, 1, 0.05, "Area Softness", "Fade marginal-size casters in/out instead of popping — for temporal stability. 0 = hard cutoff.")
                    S(GREEN_REG, "radius_softness", 0, 1, 0.05, "Reach Softness", "Fade the fringe out with distance instead of a hard edge — for temporal stability. 0 = hard cutoff.")
                    S(GREEN_REG, "full_strength_span", 1, 30, 0.5, "Full-Strength Span", "Extra chroma above Minimum Chroma that reaches full correction. Wider = gentler.")
                    S(GREEN_REG, "tone_correction_radius", 5, 50, 1, "Tone Correction Radius", "How far out (px) to sample the clean colour fringe is pulled toward.")
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
                gr.Markdown("**Casters (bright)**")
                S(PURPLE_REG, "caster_min_lightness", 50, 100, 1, "Minimum Lightness", "How bright a highlight must be to count as a fringe-caster.")
                S(PURPLE_REG, "min_area", 0, 300, 1, "Minimum Area", "Ignore highlight blobs smaller than this (px).")
                S(PURPLE_REG, "cast_radius", 2, 60, 1, "Cast Reach", "How far from a highlight (px) to look for its magenta fringe.")
                gr.Markdown("**Shadows (magenta)** — detected as a magenta *excess* over the scene's overall lighting tone")
                S(PURPLE_REG, "fringe_min_chroma", 0, 15, 0.1, "Minimum Chroma", "Chroma floor — pixels below this are never flagged (noise rejection).")
                S(PURPLE_REG, "target_hue", -45, 45, 1, "Target Hue", "Shift the detected fringe colour: 0 = magenta, higher = toward red, lower = toward violet/blue.")
                S(PURPLE_REG, "excess_thresh", 0, 20, 0.5, "Excess Threshold", "How much more magenta than the scene's overall tone a pixel must be to count as fringe. Higher = stricter.")
                S(PURPLE_REG, "hue_halfwidth", 5, 90, 1, "Hue Range (±°)", "How wide a band of hues around Target Hue counts as fringe. 90 = essentially no limit; lower narrows it to colours nearer the target.")
                gr.Markdown("**Repair**")
                S(PURPLE_REG, "max_opacity", 0, 1, 0.05, "Maximum Strength", "Cap on correction opacity. Higher = more aggressive.")
                S(PURPLE_REG, "repair_spread", 0, 20, 1, "Repair Spread", "Grow the corrected region outward by this many px before feathering.")
                S(PURPLE_REG, "feather", 0, 5, 0.1, "Feather", "Soften/blur the correction's edge (px).")
                with gr.Accordion("Advanced", open=False):
                    S(PURPLE_REG, "area_softness", 0, 1, 0.05, "Area Softness", "Fade marginal highlights in/out instead of popping — for temporal stability. 0 = hard cutoff.")
                    S(PURPLE_REG, "radius_softness", 0, 1, 0.05, "Reach Softness", "Fade the fringe out with distance instead of a hard edge — for temporal stability. 0 = hard cutoff.")
                    S(PURPLE_REG, "hue_softness", 0, 45, 1, "Hue Range Softness", "Feather (°) on the Hue Range edge — ramp the cutoff instead of a hard boundary, for temporal stability. 0 = hard edge.")
                    S(PURPLE_REG, "full_strength_span", 1, 30, 0.5, "Full-Strength Span", "Extra excess above the threshold that reaches full correction. Wider = gentler.")
                    S(PURPLE_REG, "tone_correction_radius", 5, 50, 1, "Tone Correction Radius", "How far out (px) to sample the clean colour fringe is pulled toward.")
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

    with gr.Tab("4 · ONNX Export"):
        gr.Markdown("Bake your current settings into a portable ONNX model, then run it over the "
                    "clip to preview. A close approximation of the exact passes.")
        with gr.Row():
            with gr.Column(scale=1):
                onnx_export_btn = gr.Button("① Export ONNX (current settings)", variant="primary")
                onnx_export_stat = gr.Markdown("")
                onnx_run_btn = gr.Button("② Run ONNX on clip → video", variant="primary", interactive=False)
                onnx_run_stat = gr.Markdown("Extract a clip (Tab 0), Export, then Run.")
            with gr.Column(scale=2):
                onnx_video = gr.Video(label="ONNX output — whole clip", height=420)

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
    onnx_run_btn.click(run_onnx_clip, [clip_state, fps_state], [onnx_video, onnx_run_stat])

    # Seed the reset blob with the constructed values so reset is correct from first
    # load; on save, refresh window.__defaults so reset retargets without a restart.
    reg_comps = [c["comp"] for c in REG]
    defaults_blob.value = json.dumps({f"def_{c['persist']}": c["comp"].value for c in REG})
    save_def_btn.click(save_defaults, reg_comps, [defaults_blob, save_def_stat]) \
                .then(None, defaults_blob, None, js="(b) => { window.__defaults = JSON.parse(b || '{}'); }")
    restart_btn.click(restart_app)
    demo.load(run_chain, ins, outs)
    demo.load(None, None, None, js=TOOLTIP_JS)   # hints -> native hover tooltips
    demo.load(None, None, None, js=RESET_JS)     # per-slider reset -> saved defaults

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7862, show_error=True)
