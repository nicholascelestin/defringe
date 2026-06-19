# Defringe

Removes green and purple fringe from videos, touching just the chroma channels.

## Algorithms

Fringe is a **shadow cast by a source**. First, find all the sources ("casters"). Then, find all the nearby fringe ("shadows"). Then, pull that chroma towards the clean, local tone.

**Green Fringe** is cast by saturated **warm** sources (red / purple).
**Purple Fringe** is cast by bright **blown highlight** sources.

Green fringe can be cast by purple fringe. If you remove purple fringe first, the green fringe it casts will be orphaned (without a caster), and will be unable to be removed by the green fringe algorithm. So, run green first, then purple.

`defringe_algorithm.py` is the canonical numpy implementation; `cast_torch.py` is an ONNX-exportable port that approximates it (convolutions in place of scipy's `label`/EDT, no per-caster area weighting). `tests/` pins the port within tolerance of the reference so it can't silently drift.

## Layout

```
defringe_algorithm.py  the domain logic (source of truth): green & purple casts,
                       the shared cast engine, Lab/soft-step helpers — pure numpy/skimage
app.py              Gradio tuner — live sliders, detect/compare/flicker views
video_io.py         ffmpeg/ffprobe wrappers (decode to RGB frames)
cast_torch.py       torch/ONNX port — approximates the numpy reference
cast_defringe.onnx  exported model (uint8 RGB in/out, dynamic N/H/W)
colab_defringe.ipynb  GPU runner: ONNX over a whole video, colour-correct encode
tests/              numpy ↔ torch/ONNX conformance (mean/p99 tolerance)
archive/            superseded detectors, kept for reference
source/             sample stills + clip
```

## Quickstart

Needs [`uv`](https://docs.astral.sh/uv/) and **ffmpeg** (`brew install ffmpeg`, or `apt install ffmpeg`).
uv fetches a compatible Python (≥ 3.11) and the deps for you — nothing else to install.

```bash
# get uv if you don't have it:  curl -LsSf https://astral.sh/uv/install.sh | sh
uv run python app.py            # tuner at http://127.0.0.1:7862
```

Running the conformance suite (devs only) adds one extra: `uv sync --extra test`, then `pytest`.

## Process Workload

Use the app to export a tuned algorithm as ONNX, and run it against your video frames per your preference. Optionally, use the `colab_defringe` notebook to run it against a video. 
