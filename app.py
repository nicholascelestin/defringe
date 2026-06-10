#!/usr/bin/env python3
"""Defringe tuner — tabbed Gradio app over the canonical `algorithm/` package.

Tab 0  Source   pick a still, or seek the full video and extract a 10s clip to
                memory; choose the current frame (shared by tabs 1/2/3).
Tab 1  Green    green_cast (toggle on/off) + knobs. Output feeds tab 2.
Tab 2  Purple   defringe (toggle on/off) applied to tab 1's output.
Tab 3  Temporal flicker stats over current frame + next 5 (extracted clip only).

Logic lives in algorithm/; this app only orchestrates and renders. One handler,
`run_chain`, evaluates the whole pipeline (frame -> [green] -> [purple]) so the
chain is always live with the actual knob values; every control triggers it.

Run (dedicated venv):  .venv/bin/python app.py
"""
import os
import sys
import subprocess
from pathlib import Path

import numpy as np
import gradio as gr
import matplotlib
matplotlib.use("Agg")               # headless: render plots to arrays, no display
import matplotlib.cm as cm
import matplotlib.pyplot as plt

import video_io
import algorithm as alg

STILLS = {"inside": "source/inside.webp", "horses": "source/horses.png",
          "building": "source/building.webp", "people": "source/people.webp"}
VIDEO = "source/sanguo-ep01-10min.mp4"
CLIP_SECS, PREVIEW_W = 10, 720      # full-res 10s/1080p clip would be ~1.5 GB
ONNX_PATH = "delivery/cast_defringe.onnx"   # exported model (green->purple cast)
ONNX_PREVIEW = "delivery/onnx_preview.mp4"  # last ONNX-on-clip render (gitignored)

print("preloading stills...")
STILL_IMG = {k: video_io.read_image(v) for k, v in STILLS.items()}
VW, VH, VNB, VFPS = video_io.probe(VIDEO)
VDUR = VNB / VFPS if VFPS else 0
print(f"  video {VW}x{VH} {VFPS:.0f}fps {VDUR:.0f}s")


def _overlay(rgb, alpha, thr=0.02):
    over = cm.turbo(np.clip(alpha, 0, 1))[..., :3] * 255
    return np.where((alpha > thr)[..., None], 0.2 * rgb + 0.8 * over, rgb).astype(np.uint8)


def current_frame(source, clip, idx):
    if source == "extracted clip":
        if clip is None or len(clip) == 0:
            return None
        return clip[int(np.clip(idx, 0, len(clip) - 1))]
    return STILL_IMG[source]


def run_chain(source, clip, idx,
              g_on, cast_chr, band_lo, band_hi, glo, ghi, gchr, radius,
              minar, fspan, amax, grow, feather, gtone,
              g_area_soft, g_radius_soft, g_tdir,
              pc_on, pc_bright, pc_minar, pc_area_soft, pc_radius, pc_radius_soft,
              pc_chr, pc_thue, pc_relthr, pc_fspan, pc_amax, pc_grow, pc_feather, pc_tone, pc_tdir):
    """frame -> [green] -> [purple] (casting-shadow model: bright highlight =
    caster, magenta = shadow). Returns all ten image/stat outputs."""
    rgb = current_frame(source, clip, idx)
    if rgb is None:
        blank = np.zeros((80, 80, 3), np.uint8)
        return (None, blank, blank, "no frame", None, blank, blank, "no frame")

    # --- green stage ---
    if g_on:
        gout, ginfo = alg.green_cast(
            rgb, cast_chr=cast_chr, band_lo=band_lo, band_hi=band_hi,
            green_lo=glo, green_hi=ghi, green_chr=gchr,
            cast_radius=int(radius), min_area=int(minar), full_strength_span=fspan,
            max_strength=amax, repair_spread=int(grow), feather=feather, tone_correction_radius=gtone,
            area_soft=g_area_soft, radius_soft=g_radius_soft, tone_directionality=g_tdir)
        dm = 0.45 * rgb
        if ginfo["caster"].any():
            dm[ginfo["caster"]] = 0.5 * dm[ginfo["caster"]] + 0.5 * np.array([230., 60., 50.])
        sh = ginfo["alpha"] > 0.05
        dm[sh] = 0.18 * dm[sh] + 0.82 * np.array([0., 255., 120.])
        g_detect = np.clip(dm, 0, 255).astype(np.uint8)
        g_alpha = _overlay(rgb, ginfo["alpha"])
        g_stat = f"green sel(a>0.1): {100*np.mean(ginfo['alpha']>0.1):.3f}%  a-max {ginfo['alpha'].max():.2f}"
    else:
        gout, g_detect, g_alpha, g_stat = rgb, rgb, rgb, "green DISABLED (passthrough)"

    # --- purple stage: casting-shadow model (bright highlight -> magenta fringe) ---
    if pc_on:
        pcout, pcinfo = alg.purple_cast(
            gout, bright_L=pc_bright, min_area=int(pc_minar), area_soft=pc_area_soft,
            cast_radius=int(pc_radius), radius_soft=pc_radius_soft, mag_chr=pc_chr,
            target_hue=pc_thue, full_strength_span=pc_fspan, max_strength=pc_amax,
            repair_spread=int(pc_grow), feather=pc_feather, tone_correction_radius=pc_tone,
            tone_directionality=pc_tdir, rel_thresh=pc_relthr)
        dm = 0.45 * gout
        if pcinfo["caster"].any():
            dm[pcinfo["caster"]] = 0.5 * dm[pcinfo["caster"]] + 0.5 * np.array([255., 215., 0.])
        psh = pcinfo["alpha"] > 0.05
        dm[psh] = 0.18 * dm[psh] + 0.82 * np.array([0., 255., 120.])
        pc_detect = np.clip(dm, 0, 255).astype(np.uint8)
        pc_alpha = _overlay(gout, pcinfo["alpha"])
        pc_stat = f"purple sel(a>0.1): {100*np.mean(pcinfo['alpha']>0.1):.3f}%  a-max {pcinfo['alpha'].max():.2f}"
    else:
        pcout, pc_detect, pc_alpha, pc_stat = gout, gout, gout, "purple DISABLED (passthrough)"

    g_compare = [(rgb, "input"), (gout, "green-corrected → tab 2")]
    pc_compare = [(gout, "input (green-corrected)"), (pcout, "final (green+purple)")]
    return (g_compare, g_detect, g_alpha, g_stat,       # tab 1 (green)
            pc_compare, pc_detect, pc_alpha, pc_stat)   # tab 2 (purple)


def seek_preview(start_sec):
    """Single frame at the seek point, so you can see where the clip will start."""
    return video_io.read_frame(VIDEO, float(start_sec), max_w=960)


def extract_clip(start_sec, full_res):
    max_w = None if full_res else PREVIEW_W
    frames, fps = video_io.read_clip(VIDEO, float(start_sec), CLIP_SECS, max_w=max_w)
    n = len(frames)
    info = (f"{n} frames @ {fps:.0f}fps from {start_sec:.0f}s "
            f"({frames.shape[2]}x{frames.shape[1]}, ~{frames.nbytes/1e6:.0f} MB"
            f"{' · FULL RES' if full_res else ''})")
    upd = gr.update(maximum=max(0, n - 1), value=0)
    vis = gr.update(maximum=max(0, n - 1), value=0, visible=True)   # mirrors in tabs 1/2/3
    # Switch the active source to the freshly extracted clip; otherwise the frame
    # slider keeps resolving against the still and scrubbing shows a frozen image.
    # Enable the Temporal button here directly — a programmatic source change does
    # not reliably re-fire t0_source.change, so we can't lean on frame_controls.
    return ("extracted clip", frames, info, frames[0], upd, vis, vis, vis,
            gr.update(interactive=True), gr.update(interactive=True))


def pick_preview(source, clip, idx):
    f = current_frame(source, clip, idx)
    return f if f is not None else np.zeros((80, 80, 3), np.uint8)


def frame_controls(source, clip):
    """Reveal the in-tab frame scrubbers (tabs 1/2/3) and enable the Temporal
    analyze button only for an extracted clip; size scrubbers to the clip."""
    on = source == "extracted clip" and clip is not None and len(clip) > 0
    u = gr.update(visible=True, maximum=max(0, len(clip) - 1)) if on else gr.update(visible=False)
    return u, u, u, gr.update(interactive=on), gr.update(interactive=on)


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


def temporal_analyze(source, clip, idx,
              g_on, cast_chr, band_lo, band_hi, glo, ghi, gchr, radius,
              minar, fspan, amax, grow, feather, gtone,
              g_area_soft, g_radius_soft, g_tdir,
              pc_on, pc_bright, pc_minar, pc_area_soft, pc_radius, pc_radius_soft,
              pc_chr, pc_thue, pc_relthr, pc_fspan, pc_amax, pc_grow, pc_feather, pc_tone, pc_tdir):
    """Run the live pipeline (green -> purple cast) over the current frame + next 5; report flicker."""
    if source != "extracted clip" or clip is None or len(clip) == 0:
        return None, None, "Select an **extracted clip** in Tab 0 first.", None
    i0 = int(np.clip(idx, 0, len(clip) - 1))
    sel = clip[i0:i0 + TEMPORAL_N]
    n = len(sel)
    if n < 2:
        return None, None, (f"Frame {i0} is too close to the clip end "
                            f"({n} frame(s)); pick an earlier start frame."), None
    src, greens, finals, ga, pa = [], [], [], [], []
    for fr in sel:
        if g_on:
            gout, ginfo = alg.green_cast(
                fr, cast_chr=cast_chr, band_lo=band_lo, band_hi=band_hi,
                green_lo=glo, green_hi=ghi, green_chr=gchr,
                cast_radius=int(radius), min_area=int(minar), full_strength_span=fspan,
                max_strength=amax, repair_spread=int(grow), feather=feather, tone_correction_radius=gtone,
                area_soft=g_area_soft, radius_soft=g_radius_soft, tone_directionality=g_tdir)
            galpha = ginfo["alpha"]
        else:
            gout, galpha = fr, np.zeros(fr.shape[:2], np.float32)
        if pc_on:
            pout, pinfo = alg.purple_cast(
                gout, bright_L=pc_bright, min_area=int(pc_minar), area_soft=pc_area_soft,
                cast_radius=int(pc_radius), radius_soft=pc_radius_soft, mag_chr=pc_chr,
                target_hue=pc_thue, full_strength_span=pc_fspan, max_strength=pc_amax,
                repair_spread=int(pc_grow), feather=pc_feather, tone_correction_radius=pc_tone,
                tone_directionality=pc_tdir, rel_thresh=pc_relthr)
            palpha = pinfo["alpha"]
        else:
            pout, palpha = gout, np.zeros(np.asarray(gout).shape[:2], np.float32)
        src.append(np.asarray(fr, np.float32)); greens.append(np.asarray(gout, np.float32))
        finals.append(np.asarray(pout, np.float32)); ga.append(galpha); pa.append(palpha)

    gm = _stage_metrics(greens, src, ga, n)
    pm = _stage_metrics(finals, greens, pa, n)
    d_in = float(np.mean([np.abs(src[i] - src[i-1]).mean() for i in range(1, n)]))

    # heatmap: per-pixel temporal std of the final correction (final - input),
    # which isolates algorithm-induced flicker from real scene motion.
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


def export_onnx(source, clip, idx,
                g_on, cast_chr, band_lo, band_hi, glo, ghi, gchr, radius,
                minar, fspan, amax, grow, feather, gtone,
                g_area_soft, g_radius_soft, g_tdir,
                pc_on, pc_bright, pc_minar, pc_area_soft, pc_radius, pc_radius_soft,
                pc_chr, pc_thue, pc_relthr, pc_fspan, pc_amax, pc_grow, pc_feather, pc_tone, pc_tdir,
                progress=gr.Progress()):
    """Export green->purple_cast to ONNX (opset 17) using the LIVE slider settings.
    A disabled pass (toggle off) is exported as a no-op via max_strength=0."""
    progress(0.05, desc="loading torch...")
    try:
        import torch
        dpath = str((Path(__file__).parent / "delivery").resolve())
        if dpath not in sys.path:
            sys.path.insert(0, dpath)
        from cast_torch import Defringe
    except Exception as e:
        return f"⚠️ Export extra not available — run `uv sync --extra export`.\n\n`{e}`"
    green = dict(cast_chr=cast_chr, band_lo=band_lo, band_hi=band_hi, green_lo=glo, green_hi=ghi,
                 green_chr=gchr, cast_radius=radius, min_area=minar, full_strength_span=fspan,
                 max_strength=(amax if g_on else 0.0), repair_spread=grow, feather=feather,
                 tone_correction_radius=gtone, area_soft=g_area_soft, radius_soft=g_radius_soft,
                 tone_directionality=g_tdir)
    purple = dict(bright_L=pc_bright, min_area=pc_minar, area_soft=pc_area_soft,
                  cast_radius=pc_radius, radius_soft=pc_radius_soft, mag_chr=pc_chr,
                  target_hue=pc_thue, rel_thresh=pc_relthr, full_strength_span=pc_fspan,
                  max_strength=(pc_amax if pc_on else 0.0), repair_spread=pc_grow,
                  feather=pc_feather, tone_correction_radius=pc_tone, tone_directionality=pc_tdir)
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


def run_onnx_clip(source, clip, progress=gr.Progress()):
    """Run the exported ONNX over the whole in-memory clip; return a playable video."""
    if source != "extracted clip" or clip is None or len(clip) == 0:
        return None, "Select an **extracted clip** in Tab 0 first."
    if not os.path.exists(ONNX_PATH):
        return None, "Press **Export ONNX** first."
    try:
        import onnxruntime as ort
    except Exception as e:
        return None, f"⚠️ onnxruntime not available — `uv sync --extra export`.\n\n`{e}`"
    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    n, B, outs = len(clip), 4, []
    for i in progress.tqdm(range(0, n, B), desc="ONNX defringe"):
        outs.append(sess.run(None, {"rgb": np.ascontiguousarray(clip[i:i + B])})[0])
    frames = np.concatenate(outs, 0)
    h, w = frames.shape[1:3]
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{w}x{h}", "-r", f"{VFPS:.0f}", "-i", "-", "-an", "-c:v", "libx264",
         "-crf", "16", "-preset", "fast", "-pix_fmt", "yuv420p", ONNX_PREVIEW],
        stdin=subprocess.PIPE)
    proc.stdin.write(np.ascontiguousarray(frames, np.uint8).tobytes())
    proc.stdin.close(); proc.wait()
    return ONNX_PREVIEW, f"✅ {n} frames @ {VFPS:.0f}fps through ONNX → video below ({w}×{h})."


S = lambda lo, hi, v, st, lbl, info=None: gr.Slider(lo, hi, value=v, step=st, label=lbl, info=info)
G, PC = alg.GREEN_CAST, alg.PURPLE_CAST

# Hints are written as Gradio `info=` (always renders as small text), then an
# on-load script relocates each into a native HTML `title` tooltip and hides the
# inline text — real hover tooltips with zero layout cost, degrading gracefully
# to the small text if the script ever fails to run.
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
    gr.Markdown("# Defringe tuner\nLogic in `algorithm/`. Chain: **Tab 1 green → Tab 2 purple**, "
                "each toggleable. Frame chosen in **Tab 0**.")
    clip_state = gr.State(None)

    with gr.Tab("0 · Source"):
        with gr.Row():
            with gr.Column(scale=1):
                t0_source = gr.Radio(["extracted clip", *STILLS], value="inside", label="current source")
                gr.Markdown(f"**Video** (full file {VDUR:.0f}s)")
                t0_start = gr.Slider(0, max(1, VDUR - CLIP_SECS), value=0, step=1, label=f"seek (s) — extracts {CLIP_SECS}s")
                t0_full = gr.Checkbox(False, label="full-res clip (~1.5 GB; off = 720px)")
                t0_extract = gr.Button("Extract 10s clip → memory", variant="primary")
                t0_info = gr.Textbox(label="clip info", interactive=False)
                t0_frame = gr.Slider(0, 249, value=0, step=1, label="frame (within clip)")
            with gr.Column(scale=2):
                t0_seek = gr.Image(label="seek-point preview (full video) — where the clip starts", height=300)
                t0_preview = gr.Image(label="current frame (feeds tabs 1 & 2)", height=300)

    with gr.Tab("1 · Green cast"):
        with gr.Row():
            with gr.Column(scale=1):
                g_on = gr.Checkbox(True, label="green pass ENABLED", info="Toggle the whole green-cast pass; off = passthrough to tab 2.")
                gr.Markdown("**Casters (warm)**")
                g_cast_chr = S(0, 30, G["cast_chr"], 0.5, "Minimum Chroma", "Minimum chroma (colorfulness) for a warm red/purple source to count as a caster. Lower = more casters.")
                g_band_lo = S(-90, 0, G["band_lo"], 1, "Hue Floor", "Caster hue band low edge — signed degrees from red (negative = toward purple).")
                g_band_hi = S(0, 90, G["band_hi"], 1, "Hue Ceiling", "Caster hue band high edge — signed degrees from red (positive = toward orange).")
                g_minar = S(0, 300, G["min_area"], 1, "Minimum Area", "Ignore caster blobs smaller than this many px (noise rejection).")
                g_area_soft = S(0, 1, G["area_soft"], 0.05, "Area Softness", "Softens Minimum Area — fraction of it over which a marginal-size caster fades in instead of popping. 0 = hard cutoff.")
                g_radius = S(2, 60, G["cast_radius"], 1, "Cast Reach", "How far a caster casts: search radius in px outward from a caster for its green shadow.")
                g_radius_soft = S(0, 1, G["radius_soft"], 0.05, "Reach Softness", "Softens Cast Reach — fraction of it over which the shadow fades out with distance instead of a hard edge. 0 = hard cutoff.")
                gr.Markdown("**Shadows (cool)**")
                g_gchr = S(0, 15, G["green_chr"], 0.1, "Minimum Chroma", "Minimum chroma (colorfulness) for a pixel to count as a valley-green shadow. Lower catches fainter shadows.")
                g_glo = S(-140, 0, G["green_lo"], 1, "Hue Floor", "Shadow hue band low edge — signed degrees from teal-green (negative = toward green/yellow).")
                g_ghi = S(0, 140, G["green_hi"], 1, "Hue Ceiling", "Shadow hue band high edge — signed degrees from teal-green (positive = toward blue/violet).")
                gr.Markdown("**Repair**")
                g_fspan = S(1, 30, G["full_strength_span"], 0.5, "Full-Strength Span", "Chroma above Minimum Chroma at which green correction reaches full strength. Wider = gentler (only vivid green fully corrected); narrower = more aggressive.")
                g_amax = S(0, 1, G["max_strength"], 0.05, "Maximum Strength", "Cap on correction opacity. Higher = more aggressive (this is opacity, not spatial size).")
                g_grow = S(0, 20, G["repair_spread"], 1, "Repair Spread", "Expand the corrected region outward by this many px before feathering.")
                g_feather = S(0, 5, G["feather"], 0.1, "Feather", "Alpha feather sigma (px). Softens the edge and slightly spreads it.")
                g_tone = S(5, 50, G["tone_correction_radius"], 1, "Tone Correction Radius", "Radius (px) for estimating the local tone that green chroma is pulled toward (L* kept).")
                g_tdir = S(0, 1, G["tone_directionality"], 0.05, "Tone Directionality", "Bias the repair-tone estimate away from the caster. 0 = sample tone evenly from all directions (old behavior); 1 = mostly ignore donors on the caster side, pulling repair tone from the cast (down-shadow) direction.")
            with gr.Column(scale=2):
                t1_frame = gr.Slider(0, 249, value=0, step=1, label="frame (within clip)", visible=False)
                g_stat = gr.Textbox(label="stats", interactive=False)
                g_detect = gr.Image(label="casters (red) + shadows (green)", height=360)
                g_compare = gr.Gallery(label="input ↔ green-corrected (→ tab 2) — click an image, then arrow/swipe",
                                       columns=2, object_fit="contain", preview=False)
                g_alpha = gr.Image(label="alpha", height=230)

    with gr.Tab("2 · Purple defringe"):
        with gr.Row():
            with gr.Column(scale=1):
                pc_on = gr.Checkbox(True, label="purple pass ENABLED", info="Purple defringe (casting-shadow model: bright highlight = caster, magenta = shadow) applied to tab 1's green output.")
                gr.Markdown("**Casters (bright)**")
                pc_bright = S(50, 100, PC["bright_L"], 1, "Minimum Lightness", "Min L* for a near-white blown highlight to count as a caster (the physical axial-CA source).")
                pc_minar = S(0, 300, PC["min_area"], 1, "Minimum Area", "Ignore highlight blobs smaller than this many px.")
                pc_area_soft = S(0, 1, PC["area_soft"], 0.05, "Area Softness", "Softens Minimum Area — fraction of it over which a marginal highlight fades in. 0 = hard cutoff.")
                pc_radius = S(2, 60, PC["cast_radius"], 1, "Cast Reach", "Search radius in px outward from a highlight for its magenta fringe.")
                pc_radius_soft = S(0, 1, PC["radius_soft"], 0.05, "Reach Softness", "Softens Cast Reach — fraction of it over which the fringe fades out with distance. 0 = hard cutoff.")
                gr.Markdown("**Shadows (magenta)** — detected as a magenta *excess* over the scene's overall lighting tone")
                pc_chr = S(0, 15, PC["mag_chr"], 0.1, "Minimum Chroma", "Absolute chroma floor — a pixel below this is never flagged (noise rejection), regardless of excess.")
                pc_thue = S(-45, 45, PC["target_hue"], 1, "Target Hue", "Direction the excess is measured along, as a signed offset from magenta (315°). 0 = magenta; positive shifts toward red, negative toward violet/blue. The relative-gate analog of recentering the old hue band.")
                pc_relthr = S(0, 20, PC["rel_thresh"], 0.5, "Excess Threshold", "Minimum magenta-excess-over-lighting-tone (CIELAB a*b* units) to flag a pixel as fringe. Higher = stricter; only pixels clearly more magenta than the scene's overall tone are corrected.")
                gr.Markdown("**Repair**")
                pc_fspan = S(1, 30, PC["full_strength_span"], 0.5, "Full-Strength Span", "Excess above Excess Threshold at which correction reaches full strength. Wider = gentler.")
                pc_amax = S(0, 1, PC["max_strength"], 0.05, "Maximum Strength", "Cap on correction opacity. Higher = more aggressive.")
                pc_grow = S(0, 20, PC["repair_spread"], 1, "Repair Spread", "Expand the corrected region outward by this many px before feathering.")
                pc_feather = S(0, 5, PC["feather"], 0.1, "Feather", "Alpha feather sigma (px).")
                pc_tone = S(5, 50, PC["tone_correction_radius"], 1, "Tone Correction Radius", "Radius (px) for estimating the local tone the magenta chroma is pulled toward (L* kept).")
                pc_tdir = S(0, 1, PC["tone_directionality"], 0.05, "Tone Directionality", "Bias the repair-tone estimate away from the caster (blown highlight). 0 = sample tone evenly from all directions (old behavior); 1 = mostly ignore donors on the caster side, pulling repair tone from the cast (fringe) direction.")
            with gr.Column(scale=2):
                t2_frame = gr.Slider(0, 249, value=0, step=1, label="frame (within clip)", visible=False)
                pc_stat = gr.Textbox(label="stats", interactive=False)
                pc_detect = gr.Image(label="casters (gold) + fringe (green)", height=360)
                pc_compare = gr.Gallery(label="input (green-corrected) ↔ final (green+purple) — click an image, then arrow/swipe",
                                        columns=2, object_fit="contain", preview=False)
                pc_alpha = gr.Image(label="purple alpha", height=230)

    with gr.Tab("3 · Temporal"):
        gr.Markdown("Flicker analysis on the **current frame + next 5** of the in-memory clip, "
                    "run through the **live Tab 1/2 settings**. Enabled once an *extracted clip* "
                    "is the source — no canonical temporal algorithm yet, this just measures.")
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

    with gr.Tab("ONNX export"):
        gr.Markdown("Export the **green → purple** pipeline to a portable ONNX graph using "
                    "your **current slider settings** (tabs 1 & 2), then run it over the extracted "
                    "clip. The ONNX is an ~0.1 mean-ΔRGB approximation of the exact numpy passes; "
                    "needs the `export` extra (`uv sync --extra export`).")
        with gr.Row():
            with gr.Column(scale=1):
                onnx_export_btn = gr.Button("① Export ONNX (current settings)", variant="primary")
                onnx_export_stat = gr.Markdown("")
                onnx_run_btn = gr.Button("② Run ONNX on clip → video", variant="primary", interactive=False)
                onnx_run_stat = gr.Markdown("Extract a clip (Tab 0), Export, then Run.")
            with gr.Column(scale=2):
                onnx_video = gr.Video(label="ONNX output — whole clip", height=420)

    ins = [t0_source, clip_state, t0_frame,
           g_on, g_cast_chr, g_band_lo, g_band_hi, g_glo, g_ghi, g_gchr,
           g_radius, g_minar, g_fspan, g_amax, g_grow, g_feather, g_tone,
           g_area_soft, g_radius_soft, g_tdir,
           pc_on, pc_bright, pc_minar, pc_area_soft, pc_radius, pc_radius_soft,
           pc_chr, pc_thue, pc_relthr, pc_fspan, pc_amax, pc_grow, pc_feather, pc_tone, pc_tdir]
    outs = [g_compare, g_detect, g_alpha, g_stat,
            pc_compare, pc_detect, pc_alpha, pc_stat]

    for c in ins:
        (c.release if isinstance(c, gr.Slider) else c.change)(run_chain, ins, outs)
    t0_extract.click(extract_clip, [t0_start, t0_full],
                     [t0_source, clip_state, t0_info, t0_preview, t0_frame,
                      t1_frame, t2_frame, t3_frame, temporal_btn, onnx_run_btn]) \
              .then(run_chain, ins, outs)
    t0_start.release(seek_preview, [t0_start], [t0_seek])
    for c in (t0_source, t0_frame):
        c.change(pick_preview, [t0_source, clip_state, t0_frame], [t0_preview])

    # Frame scrubber mirrored into tabs 1/2/3 (visible only for an extracted clip),
    # which also enables the Temporal analyze button. .release is user-only, so
    # propagating a value to the other sliders never re-triggers them — no sync
    # loop. run_chain reads the frame from t0_frame, so the t1/t2/t3 handlers push
    # their value there first, then re-run the chain.
    t0_source.change(frame_controls, [t0_source, clip_state],
                     [t1_frame, t2_frame, t3_frame, temporal_btn, onnx_run_btn])
    t0_frame.release(mirror_frame, [t0_frame], [t1_frame, t2_frame, t3_frame])
    t1_frame.release(mirror_frame, [t1_frame], [t0_frame, t2_frame, t3_frame]).then(run_chain, ins, outs)
    t2_frame.release(mirror_frame, [t2_frame], [t0_frame, t1_frame, t3_frame]).then(run_chain, ins, outs)
    t3_frame.release(mirror_frame, [t3_frame], [t0_frame, t1_frame, t2_frame]).then(run_chain, ins, outs)
    temporal_btn.click(temporal_analyze, ins,
                       [temporal_map, temporal_plot, temporal_stats, temporal_gallery])
    onnx_export_btn.click(export_onnx, ins, onnx_export_stat)
    onnx_run_btn.click(run_onnx_clip, [t0_source, clip_state], [onnx_video, onnx_run_stat])
    demo.load(seek_preview, [t0_start], [t0_seek])
    demo.load(run_chain, ins, outs)
    demo.load(None, None, None, js=TOOLTIP_JS)   # hints -> native hover tooltips

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7862, show_error=True)
