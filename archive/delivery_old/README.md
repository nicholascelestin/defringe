# ONNX approximation of `fringe.py`

A standalone, **approximate** port of the purple-defringe algorithm to a single
ONNX graph. The original `../fringe.py` is **not modified** by anything here.

## Why "approximate"

Two ops in the original have no ONNX operator and are approximated:

| original | approximated by |
|---|---|
| `scipy.ndimage.label` (drop edge fragments < `min_edge` px) | morphological **opening** (erode→dilate) |
| `distance_transform_edt` + `exp(-d²/2tol²)` (soft reach) | `clip(2π·tol²·gauss_blur(gate, tol), 0, 1)` |
| `np.percentile(edge, 90)` | **exact**, via sort+gather with a dynamic index |

`rgb2lab`/`lab2rgb` are reimplemented to match skimage (sRGB, D65/2°).
The 2-pass repair loop is unrolled.

## Interface

| | name | dtype | shape |
|---|---|---|---|
| input | `rgb` | float32 | `[N, 3, H, W]` (NCHW, values in `[0,1]`) |
| output | `defringed` | float32 | `[N, 3, H, W]` (values in `[0,1]`) |

**Dynamic N/H/W** (opset 17). Minimum ~101px per side — the σ=25 tone blur uses a
±100px reflect-pad. fp16 is not supported on this stack: the post-hoc float16
converters produce invalid graphs on this transcribed graph (the `Cast`/`TopK`/
`Einsum` nodes), and ONNX Runtime's CPU provider has no fp16 kernels anyway.

## Measured cost (ONNX Runtime vs exact `fringe.py`, blur_ds=1)

mean ΔRGB **0.33 / 255**, p99 = **1 level**, 99.6% of pixels within 2 levels.
Max deviation ~29 at a few isolated edge pixels (the opening/EDT approximations).
ONNX Runtime output is **bit-identical to the PyTorch port** (faithful export).

## Performance (M4, 1920×1080)

| backend | time/img |
|---|---|
| scipy/numpy `fringe.py` (blur_ds=2) | ~1000 ms |
| ONNX Runtime, CPU EP | ~570 ms |
| ONNX Runtime, CoreML EP | ~2060 ms (slower!) |

CoreML is **slower** here because the transcribed graph fragments into 27
partitions — ops like `TopK`, `Einsum`, and reflect-`Pad` fall back to CPU, and
the cross-boundary copies dominate. A distilled CNN (standard conv/activation
ops only) would stay CoreML-native and accelerate properly; the op-by-op
transcription does not.

## Files

- `defringe_torch.py` — PyTorch reimplementation (the source of truth)
- `export_onnx.py` — exports `defringe.onnx` (opset 17)
- `verify_onnx.py` — runs ORT, measures deviation vs exact, writes `out/*_onnx.png`
- `eval_torch.py` — same, but the PyTorch path (`out/*_onnxport.png`)
- `providers.py` — CPU vs CoreML timing
- `ref/` — exact `fringe.py` outputs + inputs (`.npy`), the ground truth

## Run

```bash
# uses a venv with torch + onnx + onnxruntime
python onnx/export_onnx.py     # build defringe.onnx
python onnx/verify_onnx.py     # measure approximation cost
```
