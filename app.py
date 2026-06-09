#!/usr/bin/env python3
"""Defringe tuner — tabbed Gradio app over the canonical `algorithm/` package.

Tab 0  Source   pick a still, or seek the full video and extract a 10s clip to
                memory; choose the current frame (shared by tabs 1 & 2).
Tab 1  Green    green_cast (toggle on/off) + knobs. Output feeds tab 2.
Tab 2  Purple   defringe (toggle on/off) applied to tab 1's output.

Logic lives in algorithm/; this app only orchestrates and renders. One handler,
`run_chain`, evaluates the whole pipeline (frame -> [green] -> [purple]) so the
chain is always live with the actual knob values; every control triggers it.

Run (dedicated venv):  .venv/bin/python app.py
"""
import numpy as np
import gradio as gr
import matplotlib.cm as cm

import video_io
import algorithm as alg

STILLS = {"inside": "source/inside.webp", "horses": "source/horses.png",
          "building": "source/building.webp", "people": "source/people.webp"}
VIDEO = "source/sanguo-ep01-10min.mp4"
CLIP_SECS, PREVIEW_W = 10, 720      # full-res 10s/1080p clip would be ~1.5 GB

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
              g_on, cast_chr, band_lo, band_hi, pstr, pdist, glo, ghi, gchr, radius,
              minar, size, chref, amax, grow, feather, gtone,
              p_on, hue, ref, spread, edge, anom, bright, tol, fref, floor, fillhi, ptone, passes):
    """frame -> [green] -> [purple]; returns all nine image/stat outputs."""
    rgb = current_frame(source, clip, idx)
    if rgb is None:
        blank = np.zeros((80, 80, 3), np.uint8)
        return (blank,) * 4 + ("no frame",) + (blank,) * 3 + ("no frame",)

    # --- green stage ---
    if g_on:
        gout, ginfo = alg.green_cast(
            rgb, cast_chr=cast_chr, band_lo=band_lo, band_hi=band_hi, purple_str=pstr,
            purple_dist=pdist, green_lo=glo, green_hi=ghi, green_chr=gchr,
            cast_radius=int(radius), min_area=int(minar), size_factor=size, ch_ref=chref,
            amax=amax, grow=int(grow), feather=feather, tone_sig=gtone)
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

    # --- purple stage (input = green output) ---
    if p_on:
        pout, pinfo = alg.defringe(
            gout, hue_power=hue, ref_sigma=ref, spread_win=int(spread), edge_pct=edge,
            anomaly_thr=anom, bright_L=bright, tol=tol, field_ref=fref, floor=floor,
            fill_L_hi=fillhi, tone_sigma=ptone, passes=int(passes))
        p_alpha = _overlay(gout, pinfo["alpha"])
        p_stat = f"purple sel(a>0.1): {100*np.mean(pinfo['alpha']>0.1):.3f}%  field-max {pinfo['field'].max():.1f}"
    else:
        pout, p_alpha, p_stat = gout, gout, "purple DISABLED (passthrough)"

    return (rgb, g_detect, gout, g_alpha, g_stat,    # tab 1
            gout, pout, p_alpha, p_stat)             # tab 2


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
    vis = gr.update(maximum=max(0, n - 1), value=0, visible=True)   # mirrors in tabs 1 & 2
    # Switch the active source to the freshly extracted clip; otherwise the frame
    # slider keeps resolving against the still and scrubbing shows a frozen image.
    return "extracted clip", frames, info, frames[0], upd, vis, vis


def pick_preview(source, clip, idx):
    f = current_frame(source, clip, idx)
    return f if f is not None else np.zeros((80, 80, 3), np.uint8)


def frame_controls(source, clip):
    """Show the in-tab frame scrubbers only for an extracted clip; size to it."""
    if source == "extracted clip" and clip is not None and len(clip) > 0:
        u = gr.update(visible=True, maximum=max(0, len(clip) - 1))
    else:
        u = gr.update(visible=False)
    return u, u


def mirror_frame(v):
    """Keep the three frame sliders (tabs 0/1/2) in lockstep."""
    return v, v


S = lambda lo, hi, v, st, lbl, info=None: gr.Slider(lo, hi, value=v, step=st, label=lbl, info=info)
G, P = alg.GREEN_CAST, alg.PURPLE_DEFAULTS

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
                gr.Markdown("**caster (red+purple)**")
                g_cast_chr = S(0, 30, G["cast_chr"], 0.5, "CAST_CHR", "Saturation a red/purple source must exceed to count as a caster. Lower = more casters.")
                g_band_lo = S(-90, 0, G["band_lo"], 1, "BAND_LO", "Caster hue band, purple edge (signed deg from red). More negative reaches further into purple.")
                g_band_hi = S(0, 90, G["band_hi"], 1, "BAND_HI", "Caster hue band, orange edge (signed deg from red). Higher includes more orange.")
                g_pstr = S(0, 1, G["purple_str"], 0.05, "PURPLE_STR", "Strength of purple casters vs red. 1 = equal, <1 = purple acts weaker.")
                g_pdist = S(10, 120, G["purple_dist"], 1, "PURPLE_DIST", "Degrees over which caster strength ramps down from red toward purple.")
                gr.Markdown("**green shadow**")
                g_glo = S(60, 180, G["green_lo"], 1, "GREEN_HUE lo", "Low edge of the valley-green hue range counted as shadow.")
                g_ghi = S(180, 300, G["green_hi"], 1, "GREEN_HUE hi", "High edge of the valley-green hue range counted as shadow.")
                g_gchr = S(0, 15, G["green_chr"], 0.1, "GREEN_CHR", "Green chroma floor. Lower catches fainter shadows (but more false positives).")
                g_radius = S(2, 60, G["cast_radius"], 1, "CAST_RADIUS", "Reach in px from a caster's edge to search for its green shadow.")
                g_minar = S(0, 300, G["min_area"], 1, "MIN_AREA", "Ignore caster blobs smaller than this many px (noise rejection).")
                g_size = S(0.5, 15, G["size_factor"], 0.1, "SIZE_FACTOR", "Reject a green region larger than this × its caster's area (a real object, not a cast).")
                gr.Markdown("**alpha + repair**")
                g_chref = S(4, 30, G["ch_ref"], 0.5, "CH_REF", "Green chroma that maps to full alpha. Lower = stronger correction on faint green.")
                g_amax = S(0, 1, G["amax"], 0.05, "AMAX (strength)", "Cap on correction opacity (strength, not size). Higher = more aggressive.")
                g_grow = S(0, 20, G["grow"], 1, "GROW (size)", "Dilate the corrected region outward by this many px before feathering.")
                g_feather = S(0, 5, G["feather"], 0.1, "FEATHER (soft)", "Alpha feather sigma (px). Softens the edge and slightly spreads it.")
                g_tone = S(5, 50, G["tone_sig"], 1, "TONE_SIG", "Radius (px) for estimating local tone that green chroma is pulled toward (L* kept).")
            with gr.Column(scale=2):
                t1_frame = gr.Slider(0, 249, value=0, step=1, label="frame (within clip)", visible=False)
                g_stat = gr.Textbox(label="stats", interactive=False)
                g_detect = gr.Image(label="casters (red) + shadows (green)", height=360)
                with gr.Row():
                    g_orig = gr.Image(label="input", height=270)
                    g_corr = gr.Image(label="green-corrected (→ tab 2)", height=270)
                g_alpha = gr.Image(label="alpha", height=230)

    with gr.Tab("2 · Purple defringe"):
        with gr.Row():
            with gr.Column(scale=1):
                p_on = gr.Checkbox(True, label="purple pass ENABLED", info="Toggle the purple-defringe pass; off = passes tab 1's output straight through.")
                gr.Markdown("**input = tab 1 output**")
                p_hue = S(1, 10, P["hue_power"], 0.5, "hue_power", "Hue selectivity of the seed. Higher = tighter to the magenta-violet target.")
                p_ref = S(1, 15, P["ref_sigma"], 0.5, "ref_sigma", "Radius (px) of the neighbour average the fringe is measured against.")
                p_spread = S(3, 21, P["spread_win"], 2, "spread_win", "Grow the anomaly seed across the fringe band (px, odd).")
                p_edge = S(70, 99, P["edge_pct"], 1, "edge_pct", "Contrast percentile that counts as a strong luminance edge. Higher = only the strongest edges.")
                p_anom = S(1, 20, P["anomaly_thr"], 0.5, "anomaly_thr", "Chroma above which a pixel is 'coloured', used to exclude blob interiors.")
                p_bright = S(60, 100, P["bright_L"], 1, "bright_L", "Min L* for the near-white 'light side' a fringe must border.")
                p_tol = S(5, 50, P["tol"], 1, "tol", "Soft reach radius (px) inward from the edge that the field fades over.")
                p_fref = S(4, 30, P["field_ref"], 0.5, "field_ref", "Field value that maps to peak correction alpha. Lower = stronger.")
                p_floor = S(0, 0.5, P["floor"], 0.01, "floor", "Minimum alpha applied wherever the field is active at all.")
                p_fillhi = S(60, 110, P["fill_L_hi"], 1, "fill_L_hi", "Dark-side cutoff: pixels brighter than this L* are skipped (don't touch the highlight).")
                p_tonesig = S(5, 50, P["tone_sigma"], 1, "tone_sigma", "Radius (px) of the surrounding-tone estimate the fringe chroma is neutralised toward.")
                p_passes = S(1, 4, P["passes"], 1, "passes", "Repair iterations (field detected once, reused). Each pass is gentler.")
            with gr.Column(scale=2):
                t2_frame = gr.Slider(0, 249, value=0, step=1, label="frame (within clip)", visible=False)
                p_stat = gr.Textbox(label="stats", interactive=False)
                with gr.Row():
                    p_in = gr.Image(label="input (green-corrected)", height=280)
                    p_out = gr.Image(label="final (green+purple)", height=280)
                p_alpha = gr.Image(label="purple alpha", height=260)

    ins = [t0_source, clip_state, t0_frame,
           g_on, g_cast_chr, g_band_lo, g_band_hi, g_pstr, g_pdist, g_glo, g_ghi, g_gchr,
           g_radius, g_minar, g_size, g_chref, g_amax, g_grow, g_feather, g_tone,
           p_on, p_hue, p_ref, p_spread, p_edge, p_anom, p_bright, p_tol, p_fref, p_floor,
           p_fillhi, p_tonesig, p_passes]
    outs = [g_orig, g_detect, g_corr, g_alpha, g_stat, p_in, p_out, p_alpha, p_stat]

    for c in ins:
        (c.release if isinstance(c, gr.Slider) else c.change)(run_chain, ins, outs)
    t0_extract.click(extract_clip, [t0_start, t0_full],
                     [t0_source, clip_state, t0_info, t0_preview, t0_frame, t1_frame, t2_frame]) \
              .then(run_chain, ins, outs)
    t0_start.release(seek_preview, [t0_start], [t0_seek])
    for c in (t0_source, t0_frame):
        c.change(pick_preview, [t0_source, clip_state, t0_frame], [t0_preview])

    # Frame scrubber mirrored into tabs 1 & 2 (visible only for an extracted clip).
    # .release is user-only, so propagating a value to the other sliders never
    # re-triggers them — no sync loop. run_chain reads the frame from t0_frame, so
    # the t1/t2 handlers push their value there first, then re-run the chain.
    t0_source.change(frame_controls, [t0_source, clip_state], [t1_frame, t2_frame])
    t0_frame.release(mirror_frame, [t0_frame], [t1_frame, t2_frame])
    t1_frame.release(mirror_frame, [t1_frame], [t0_frame, t2_frame]).then(run_chain, ins, outs)
    t2_frame.release(mirror_frame, [t2_frame], [t0_frame, t1_frame]).then(run_chain, ins, outs)
    demo.load(seek_preview, [t0_start], [t0_seek])
    demo.load(run_chain, ins, outs)
    demo.load(None, None, None, js=TOOLTIP_JS)   # hints -> native hover tooltips

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7862, show_error=True)
