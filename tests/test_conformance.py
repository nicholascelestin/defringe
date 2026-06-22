"""Conformance: the torch/ONNX port must track the numpy reference to within 8-bit rounding.

defringe_algorithm.py is canonical; cast_torch.py is its ONNX-able twin. They share the same
scale-space geometry (box-sum count for Minimum Area, square dilation + gaussian feather for
Cast Reach), so the only divergence is float32 op-by-op noise and the final uint8 rounding:
on the 1080p stills below that's ~mean 0.03 / p99 1 / max 8, and ~94% of pixels are byte-
identical. These tolerances pin that precision so any real drift fails. Spatial params are
resolution-relative; the port bakes them for ref_hw (default 1080p), matching these stills.
"""
import numpy as np
import pytest
from PIL import Image

import video_io
import defringe_algorithm as alg

STILLS = ["source/inside.webp", "source/horses.png",
          "source/building.webp", "source/people.webp"]
MEAN_TOL, P99_TOL, MAX_TOL = 0.15, 1, 16    # max allowed mean / 99th-pct / single-channel |Δ|, in 0-255 levels
INV_MEAN_TOL, INV_P99_TOL = 0.3, 4          # off-reference (resampled) match to the resolution-invariant numpy


def _numpy_reference(img):
    g, _ = alg.green_cast(img)
    out, _ = alg.purple_cast(g)
    return out


def _within_tolerance(out, ref):
    d = np.abs(out.astype(int) - ref.astype(int))
    return d.mean(), float(np.percentile(d, 99)), int(d.max())


@pytest.fixture(scope="module")
def torch_model():
    torch = pytest.importorskip("torch")
    from cast_torch import Defringe
    return torch, Defringe().eval()


@pytest.fixture(scope="module")
def onnx_session(tmp_path_factory, torch_model):
    torch, model = torch_model
    ort = pytest.importorskip("onnxruntime")
    path = tmp_path_factory.mktemp("onnx") / "defringe.onnx"
    torch.onnx.export(model, torch.zeros(1, 256, 256, 3, dtype=torch.uint8), str(path),
                      opset_version=17, input_names=["rgb"], output_names=["out"],
                      dynamic_axes={"rgb": {0: "N", 1: "H", 2: "W"},
                                    "out": {0: "N", 1: "H", 2: "W"}}, dynamo=False)
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


@pytest.mark.parametrize("path", STILLS)
def test_torch_matches_numpy(path, torch_model):
    torch, model = torch_model
    img = video_io.read_image(path)
    with torch.no_grad():
        out = model(torch.from_numpy(np.array(img[None])))[0].numpy()
    mean, p99, mx = _within_tolerance(out, _numpy_reference(img))
    assert mean <= MEAN_TOL and p99 <= P99_TOL and mx <= MAX_TOL, f"{path}: mean={mean:.3f} p99={p99:.0f} max={mx}"


@pytest.mark.parametrize("path", STILLS)
def test_onnx_matches_numpy(path, onnx_session):
    img = video_io.read_image(path)
    out = onnx_session.run(None, {"rgb": np.ascontiguousarray(img[None])})[0][0]
    mean, p99, mx = _within_tolerance(out, _numpy_reference(img))
    assert mean <= MEAN_TOL and p99 <= P99_TOL and mx <= MAX_TOL, f"{path}: mean={mean:.3f} p99={p99:.0f} max={mx}"


@pytest.mark.parametrize("path", STILLS)
def test_onnx_resolution_invariant(path, onnx_session):
    """Off the baked reference, the in-graph resample keeps the model tracking the (resolution-
    invariant) numpy reference: run at 720p and compare to numpy at 720p. (A handful of isolated
    edge pixels diverge from the delta upsample, so this pins mean/p99, not the single-pixel max.)"""
    img = video_io.read_image(path)
    small = np.asarray(Image.fromarray(img).resize((1280, 720), Image.LANCZOS))
    out = onnx_session.run(None, {"rgb": np.ascontiguousarray(small[None])})[0][0]
    mean, p99, _ = _within_tolerance(out, _numpy_reference(small))
    assert mean <= INV_MEAN_TOL and p99 <= INV_P99_TOL, f"{path}: mean={mean:.3f} p99={p99:.0f}"
