#!/usr/bin/env python3
import os
import sys
import json
import shutil
from pathlib import Path

import numpy as np
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt

import video_io
import onnx_runtime
import views
import parameters
import defringe_numpy as alg
from sliders import slider, GREEN_REG, PURPLE_REG, REG, split, export_profile, import_profile

DEFAULT_SECS = 10
ONNX_PATH = "cast_defringe.onnx"
ONNX_PREVIEW = "onnx_preview.mp4"
TEMPORAL_N = 6
STATIC_MOTION = 2.0     # input barely moved between frames (per-channel mean |Δ|, 0-255 levels)
TOGGLE_LEVEL = 0.1      # alpha crossing this = a pixel flipping in/out of correction

if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
    print("⚠ ffmpeg/ffprobe not on PATH — video upload needs them "
          "(brew install ffmpeg / apt install ffmpeg). Images still work.")


def gslider(name):
    return slider(parameters.GREEN_BY_NAME[name], GREEN_REG, "green")


def pslider(name):
    return slider(parameters.PURPLE_BY_NAME[name], PURPLE_REG, "purple")


def current_frame(still, clip, idx):
    if still is not None:
        return still
    if clip is not None and len(clip):
        return clip[int(np.clip(idx, 0, len(clip) - 1))]
    return None


def restart_app():
    os.execv(sys.executable, [sys.executable, *sys.argv])


def run_chain(still, clip, idx, g_on, pc_on, *vals):
    rgb = current_frame(still, clip, idx)
    if rgb is None:
        return None, None, "no frame", None, None, "no frame"
    green, purple = split(vals)

    if g_on:
        gcast = alg.green_cast(rgb, **green)
        gout = gcast.image
        g_detect = [(views.detect_view(rgb, gcast, views.CASTER_RED, green["cast_radius"]), "casters & reach"),
                    (views.overlay(rgb, gcast.alpha), "alpha")]
        g_stat = views.stat_line("green", gcast)
    else:
        gout, g_stat = rgb, "green DISABLED (passthrough)"
        g_detect = [(rgb, "casters (red) + shadows (green)"), (rgb, "alpha")]

    if pc_on:
        pcast = alg.purple_cast(gout, **purple)
        pcout = pcast.image
        pc_detect = [(views.detect_view(gout, pcast, views.CASTER_GOLD, purple["cast_radius"]), "casters & reach"),
                     (views.overlay(gout, pcast.alpha), "alpha")]
        pc_stat = views.stat_line("purple", pcast)
    else:
        pcout, pc_stat = gout, "purple DISABLED (passthrough)"
        pc_detect = [(gout, "casters (gold) + fringe (green)"), (gout, "alpha")]

    g_compare = [(rgb, "input"), (gout, "green-corrected → tab 2")]
    pc_compare = [(gout, "green-corrected"), (pcout, "final")]
    return (g_compare, g_detect, g_stat,
            pc_compare, pc_detect, pc_stat)


def on_upload(path):
    hide, off = gr.update(visible=False), gr.update(interactive=False)
    if not path:
        return (None, None, None, "", None, None,
                hide, hide, hide, hide, hide, hide, off, off)
    if video_io.is_video(path):
        w, h, nb, fps = video_io.probe(path)
        dur = nb / fps if fps else 0
        info = f"video {w}x{h} {fps:.0f}fps {dur:.0f}s — set seconds & seek, then Extract"
        return (None, None, path, info, video_io.read_frame(path, 0.0, max_w=960), None,
                gr.update(visible=True, maximum=max(0, dur), value=0),       # seek
                gr.update(visible=True), gr.update(visible=True),            # secs / extract
                hide, hide, hide, off,                    # clip scrubbers + temporal btn (need a clip)
                gr.update(interactive=True))              # onnx run (a video is enough — whole-video scope)
    img = video_io.read_image(path)
    return (img, None, None, f"image {img.shape[1]}x{img.shape[0]}", None, img,
            hide, hide, hide, hide, hide, hide, off, off)


def seek_preview(video_path, start_sec):
    if not video_path:
        return None
    return video_io.read_frame(video_path, float(start_sec), max_w=960)


def extract_clip(video_path, start_sec, secs):
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
    frame = current_frame(still, clip, idx)
    return frame if frame is not None else np.zeros((80, 80, 3), np.uint8)


def mirror_frame(v):
    return v, v, v


def _stage_metrics(corrected_frames, input_frames, alphas, n):
    corrections = [corrected_frames[i] - input_frames[i] for i in range(n)]
    static, alpha_change, per_pair, toggled = [], [], [], []
    for i in range(1, n):
        input_motion = np.abs(input_frames[i] - input_frames[i - 1]).mean(-1)
        correction_change = np.abs(corrections[i] - corrections[i - 1]).mean(-1)
        corrected = (alphas[i] > 0) | (alphas[i - 1] > 0)
        nearly_still = (input_motion < STATIC_MOTION) & corrected
        per_pair.append(float(correction_change[corrected].mean()) if corrected.any() else 0.0)
        if nearly_still.any():
            static.append(float(correction_change[nearly_still].mean()))
        if corrected.any():
            alpha_change.append(float(np.abs(alphas[i] - alphas[i - 1])[corrected].mean()))
        toggled.append(float(((alphas[i] > TOGGLE_LEVEL) != (alphas[i - 1] > TOGGLE_LEVEL)).mean()))
    coverage = np.mean([(a > 0).mean() for a in alphas])
    return dict(static=float(np.mean(static)) if static else 0.0,
                dalpha=float(np.mean(alpha_change)) if alpha_change else 0.0,
                toggle=float(np.mean(toggled)) * 100,
                cov=float(coverage) * 100,
                per_pair=per_pair)


def _pair_plot(green_per_pair, purple_per_pair, start):
    fig, ax = plt.subplots(figsize=(5.2, 2.6), dpi=100)
    xs = range(len(green_per_pair))
    labels = [f"{start+k}→{start+k+1}" for k in xs]
    ax.plot(xs, green_per_pair, "-o", color="#2ca02c", label="green")
    ax.plot(xs, purple_per_pair, "-o", color="#9467bd", label="purple")
    ax.set_xticks(list(xs)); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("mean |Δcorrection| /255", fontsize=8)
    ax.set_title("flicker per frame-pair (corrected px)", fontsize=9)
    ax.set_ylim(bottom=0); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.canvas.draw()
    plot = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return plot


def temporal_analyze(still, clip, idx, g_on, pc_on, *vals):
    if clip is None or len(clip) == 0:
        return None, None, "Extract a video clip in Tab 0 first.", None
    start = int(np.clip(idx, 0, len(clip) - 1))
    window = clip[start:start + TEMPORAL_N]
    n = len(window)
    if n < 2:
        return None, None, (f"Frame {start} is too close to the clip end "
                            f"({n} frame(s)); pick an earlier start frame."), None
    green, purple = split(vals)
    sources, green_frames, final_frames, green_alphas, purple_alphas = [], [], [], [], []
    for frame in window:
        if g_on:
            gcast = alg.green_cast(frame, **green); gout, galpha = gcast.image, gcast.alpha
        else:
            gout, galpha = frame, np.zeros(frame.shape[:2], np.float32)
        if pc_on:
            pcast = alg.purple_cast(gout, **purple); pout, palpha = pcast.image, pcast.alpha
        else:
            pout, palpha = gout, np.zeros(np.asarray(gout).shape[:2], np.float32)
        sources.append(np.asarray(frame, np.float32))
        green_frames.append(np.asarray(gout, np.float32))
        final_frames.append(np.asarray(pout, np.float32))
        green_alphas.append(galpha); purple_alphas.append(palpha)

    green_metrics = _stage_metrics(green_frames, sources, green_alphas, n)
    purple_metrics = _stage_metrics(final_frames, green_frames, purple_alphas, n)
    input_motion = float(np.mean([np.abs(sources[i] - sources[i - 1]).mean() for i in range(1, n)]))

    corrections = np.stack([final_frames[i] - sources[i] for i in range(n)], 0)
    flicker = corrections.std(0).mean(-1)
    peak = float(flicker.max())
    heatmap = (cm.turbo(np.clip(flicker / (peak + 1e-6), 0, 1))[..., :3] * 255).astype(np.uint8)

    plot = _pair_plot(green_metrics["per_pair"], purple_metrics["per_pair"], start)
    gallery = [(final_frames[k].astype(np.uint8), f"frame {start + k}") for k in range(n)]
    stats = (
        f"**Frames {start}–{start+n-1}** ({n} frames) · input motion baseline **{input_motion:.2f}**/255\n\n"
        f"| stage | static-flicker | Δalpha | toggle % | coverage % |\n"
        f"|---|---|---|---|---|\n"
        f"| green | {green_metrics['static']:.3f} | {green_metrics['dalpha']:.3f} | {green_metrics['toggle']:.2f} | {green_metrics['cov']:.2f} |\n"
        f"| purple | {purple_metrics['static']:.3f} | {purple_metrics['dalpha']:.3f} | {purple_metrics['toggle']:.2f} | {purple_metrics['cov']:.2f} |\n\n"
        f"*static-flicker* = correction jump where the input barely moved (the cleanest "
        f"flicker signal); *Δalpha* = mean per-frame alpha change on corrected pixels; "
        f"*toggle* = % of frame flipping in/out of correction per pair. "
        f"Heatmap = temporal std of the final correction (peak **{peak:.1f}**/255; "
        f"turbo: blue = stable, red = flickering)."
    )
    return heatmap, plot, stats, gallery


def export_onnx(still, clip, idx, g_on, pc_on, *vals, progress=gr.Progress()):
    progress(0.05, desc="loading torch...")
    try:
        import torch
        from defringe_torch import Defringe
    except Exception as e:
        return f"⚠️ torch not available — try `uv sync` to reinstall deps.\n\n`{e}`"
    green, purple = split(vals)
    if not g_on:
        green["max_opacity"] = 0.0          # a disabled pass bakes as a no-op
    if not pc_on:
        purple["max_opacity"] = 0.0
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


def _run_clip(sess, clip, fps, progress):
    batch, outputs = 4, []
    for i in progress.tqdm(range(0, len(clip), batch), desc="ONNX defringe"):
        outputs.append(sess.run(None, {"rgb": np.ascontiguousarray(clip[i:i + batch])})[0])
    frames = np.concatenate(outputs, 0)
    h, w = frames.shape[1:3]
    with video_io.encoder(w, h, fps, ONNX_PREVIEW) as enc:
        video_io.write_frames(enc, frames)
    return ONNX_PREVIEW, f"✅ {len(clip)} frames @ {fps:.0f}fps → video below ({w}×{h})."


def _run_video(sess, video_path, progress):
    w, h, nb, fps = video_io.probe(video_path)
    done = 0
    with video_io.encoder(w, h, fps or 25.0, ONNX_PREVIEW) as enc:
        for batch in video_io.iter_frame_batches(video_path):
            video_io.write_frames(enc, sess.run(None, {"rgb": np.ascontiguousarray(batch)})[0])
            done += len(batch)
            if nb:
                progress(min(done / nb, 1.0), desc=f"ONNX whole video — {done}/{nb}")
    return ONNX_PREVIEW, f"✅ whole video — {done} frames @ {fps:.0f}fps → ({w}×{h})."


def run_onnx(clip, fps, video_path, scope, progress=gr.Progress()):
    if not os.path.exists(ONNX_PATH):
        return None, "Press **Export ONNX** first.", gr.update()
    try:
        sess, device = onnx_runtime.make_session(ONNX_PATH)
    except Exception as e:
        return None, f"⚠️ couldn't start a compute device: `{e}`", gr.update(value="CPU (unavailable)")
    dev = gr.update(value=device)
    if scope == "Whole video":
        if not video_path:
            return None, "Upload a video in Tab 0 first (or choose **Selected clip**).", dev
        return (*_run_video(sess, video_path, progress), dev)
    if clip is None or len(clip) == 0:
        return None, "Extract a clip in Tab 0 first (or choose **Whole video**).", dev
    return (*_run_clip(sess, clip, fps, progress), dev)


ASSETS = Path(__file__).with_name("assets")
WHEEL_JS = (ASSETS / "defringe_wheel.js").read_text()
GRADIO_UI_JS = (ASSETS / "gradio_ui.js").read_text()
ACC_CSS = (ASSETS / "acc.css").read_text()


def acc_head(title, target):
    return gr.HTML(f'<div class="acc-head" data-target="{target}">'
                   f'<span>{title}</span><span class="chev">▼</span></div>')


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
    # CSS-hidden, NOT visible=False: Gradio 6 unmounts invisible components, and the JS reads
    # window.__defaults / this element from the DOM.
    defaults_blob = gr.Textbox(elem_id="defaults_blob", elem_classes="hidden-blob")
    clip_state = gr.State(None)
    still_state = gr.State(None)
    video_state = gr.State(None)
    fps_state = gr.State(25.0)

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
                # Custom CSS collapsible, not gr.Accordion: a gr.Accordion unmounts its body when
                # closed, which blanks the wheel-bound sliders. display:none keeps them in the DOM.
                with gr.Column(elem_classes="gacc"):
                    acc_head("Casters (warm)", "acc-g-casters")
                    with gr.Column(elem_id="acc-g-casters", elem_classes="acc-body"):
                        gslider("caster_min_chroma")
                        gslider("caster_hue_lo")
                        gslider("caster_hue_hi")
                with gr.Column(elem_classes="gacc"):
                    acc_head("Shadows (cool)", "acc-g-shadows")
                    with gr.Column(elem_id="acc-g-shadows", elem_classes="acc-body"):
                        gslider("fringe_min_chroma")
                        gslider("fringe_hue_lo")
                        gslider("fringe_hue_hi")
                with gr.Column(elem_classes="gacc"):
                    acc_head("Size & Reach", "acc-g-size")
                    with gr.Column(elem_id="acc-g-size", elem_classes="acc-body"):
                        gslider("min_area")
                        gslider("cast_radius")
                with gr.Column(elem_classes="gacc"):
                    acc_head("Repair", "acc-g-repair")
                    with gr.Column(elem_id="acc-g-repair", elem_classes="acc-body"):
                        gslider("max_opacity")
                        gslider("repair_spread")
                        gslider("feather")
                with gr.Accordion("Advanced", open=False):
                    for _n in ("area_softness", "radius_softness", "full_strength_span",
                               "tone_correction_radius", "tone_directionality"):
                        gslider(_n)
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
                with gr.Column(elem_classes="gacc"):
                    acc_head("Source highlight (caster)", "acc-p-source")
                    with gr.Column(elem_id="acc-p-source", elem_classes="acc-body"):
                        pslider("caster_min_lightness")
                        pslider("min_area")
                        pslider("cast_radius")
                with gr.Column(elem_classes="gacc"):
                    acc_head("Shadows (magenta)", "acc-p-shadows")
                    with gr.Column(elem_id="acc-p-shadows", elem_classes="acc-body"):
                        gr.Markdown("<sub>a magenta *excess* over the scene's overall lighting tone</sub>")
                        pslider("fringe_min_chroma")
                        pslider("target_hue")
                        pslider("excess_thresh")
                        pslider("hue_halfwidth")
                with gr.Column(elem_classes="gacc"):
                    acc_head("Repair", "acc-p-repair")
                    with gr.Column(elem_id="acc-p-repair", elem_classes="acc-body"):
                        pslider("max_opacity")
                        pslider("repair_spread")
                        pslider("feather")
                with gr.Accordion("Advanced", open=False):
                    for _n in ("area_softness", "radius_softness", "hue_softness",
                               "full_strength_span", "tone_correction_radius", "tone_directionality"):
                        pslider(_n)
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
                onnx_device = gr.Textbox(label="compute device", value=onnx_runtime.best_device_name(),
                                         interactive=False, max_lines=1)
                onnx_video = gr.Video(label="ONNX output", height=420)

    green_wheel.value = f"<div class='defringe-wheel-mount' data-config='{views.green_wheel_config(GREEN_REG)}'></div>"
    purple_wheel.value = f"<div class='defringe-wheel-mount' data-config='{views.purple_wheel_config(PURPLE_REG)}'></div>"

    ins = [still_state, clip_state, t0_frame, g_on, pc_on,
           *[e["comp"] for e in GREEN_REG], *[e["comp"] for e in PURPLE_REG]]
    outs = [g_compare, g_detect, g_stat,
            pc_compare, pc_detect, pc_stat]

    for c in ins:
        on_change = c.release if isinstance(c, gr.Slider) else c.change
        on_change(run_chain, ins, outs)

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

    # mirror_frame uses .release (user-only), so propagating a value never re-triggers — no sync loop.
    t0_frame.release(mirror_frame, [t0_frame], [t1_frame, t2_frame, t3_frame])
    t1_frame.release(mirror_frame, [t1_frame], [t0_frame, t2_frame, t3_frame]).then(run_chain, ins, outs)
    t2_frame.release(mirror_frame, [t2_frame], [t0_frame, t1_frame, t3_frame]).then(run_chain, ins, outs)
    t3_frame.release(mirror_frame, [t3_frame], [t0_frame, t1_frame, t2_frame]).then(run_chain, ins, outs)
    temporal_btn.click(temporal_analyze, ins,
                       [temporal_map, temporal_plot, temporal_stats, temporal_gallery])
    onnx_export_btn.click(export_onnx, ins, onnx_export_stat)
    onnx_run_btn.click(run_onnx, [clip_state, fps_state, video_state, onnx_scope],
                       [onnx_video, onnx_run_stat, onnx_device])

    reg_comps = [c["comp"] for c in REG]
    defaults_blob.value = json.dumps({c["elem_id"]: c["comp"].value for c in REG})
    save_prof_btn.click(export_profile, reg_comps, save_prof_btn)
    load_prof_btn.upload(import_profile, [load_prof_btn], reg_comps + [defaults_blob, save_def_stat]) \
                 .then(None, defaults_blob, None, js="(b) => applyProfileDefaults(b)") \
                 .then(run_chain, ins, outs)
    restart_btn.click(restart_app)
    demo.load(run_chain, ins, outs)
    demo.load(None, None, None, js="() => { installResetHijack(); installAccordion(); }")

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7862, show_error=True, css=ACC_CSS,
                head=f"<script>{WHEEL_JS}</script><script>{GRADIO_UI_JS}</script>")
