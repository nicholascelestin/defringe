# Defringe

Green- and purple-fringe (chromatic aberration) removal for video and stills. Two
independent passes in CIELAB that only touch chroma — lightness (L\*) is kept and
untouched pixels pass through bit-exact.

## Algorithms

Both treat fringe as a **shadow cast by a source**: find the source ("caster"),
look for the fringe colour within its reach, then pull that chroma toward the
clean local tone. Run green first, then purple.

- **`green_cast`** — green fringe is cast by a saturated **warm** source (red/purple).
  Find warm caster blobs → flag nearby green pixels within a cast radius → feathered,
  chroma-weighted correction.
- **`purple_cast`** — magenta/violet fringe (axial CA) is cast by a **blown highlight**.
  The caster is the near-white highlight; a fringe pixel is one that's *more magenta
  than the scene's overall lighting tone* (this self-cancels warm-lit scenes, which
  a fixed hue band would over-correct). `Target Hue` aims the test; `Excess Threshold`
  sets how strict it is.

Shared repair knobs: **Tone Directionality** pulls the replacement colour from the
cast side (away from the caster, where the image is clean); the **softness** knobs
ramp decisions instead of hard-thresholding them, for temporal stability across frames.

The passes are stateless per frame, which keeps them exportable to ONNX.

## Layout

```
algorithm/          pure numpy/skimage domain logic (the source of truth)
  green_cast.py     green pass + its defaults
  purple_cast.py    purple pass + its defaults
  _common.py        Lab conversions, soft-step, shared helpers
app.py              Gradio tuner — live sliders, detect/compare/flicker views
video_io.py         ffmpeg/ffprobe wrappers (decode to RGB frames)
delivery/           shipping artefacts
  cast_torch.py     torch/ONNX port of the green→purple pipeline
  cast_defringe.onnx  exported model (uint8 RGB in/out, dynamic N/H/W)
  colab_defringe.ipynb  GPU runner: ONNX over a whole video, colour-correct encode
archive/            superseded detectors, kept for reference
source/             sample stills + clip
```

## Setup

Requires **Python ≥ 3.14**, [`uv`](https://docs.astral.sh/uv/), and **ffmpeg/ffprobe** on `PATH`.

```bash
uv sync                  # core deps (tuner + algorithm)
uv sync --extra export   # + torch/onnx, only needed to build/run the ONNX model
```

## Run

```bash
uv run python app.py     # tuner at http://127.0.0.1:7862
```

In the tuner: pick a source (Tab 0) → tune green (Tab 1) and purple (Tab 2) →
check cross-frame stability (Tab 3) → export the ONNX (ONNX tab). Slider defaults
are read straight from each pass's `DEFAULTS`, so tuned values are baked by editing
those dicts.

To process a full video on a GPU, open `delivery/colab_defringe.ipynb` in Colab,
upload the exported `cast_defringe.onnx`, and run the cells.
