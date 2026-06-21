"""Conformance: the torch/ONNX port must track the numpy reference within tolerance.

defringe_algorithm.py is canonical; cast_torch.py is its ONNX-able twin. Since the
scale-space rewrite both share the same geometry (box-sum count for Minimum Area, square
dilation + gaussian feather for Cast Reach), so they converge tightly -- worst case ~mean
0.06 / p99 1 / max 5 on the 1080p stills below. These tests pin that so it can't silently
drift. Spatial params are resolution-relative; the port bakes them for ref_hw (default
1080p), which matches these stills.
"""
import numpy as np
import pytest

import video_io
import defringe_algorithm as alg

STILLS = ["source/inside.webp", "source/horses.png",
          "source/building.webp", "source/people.webp"]
MEAN_TOL, P99_TOL = 0.5, 2          # max allowed mean |Δ| and 99th-pct |Δ|, in 0-255 levels


def _numpy_reference(img):
    g, _ = alg.green_cast(img)
    out, _ = alg.purple_cast(g)
    return out


def _within_tolerance(out, ref):
    d = np.abs(out.astype(int) - ref.astype(int))
    return d.mean(), float(np.percentile(d, 99))


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
    mean, p99 = _within_tolerance(out, _numpy_reference(img))
    assert mean <= MEAN_TOL and p99 <= P99_TOL, f"{path}: mean={mean:.3f} p99={p99:.0f}"


@pytest.mark.parametrize("path", STILLS)
def test_onnx_matches_numpy(path, onnx_session):
    img = video_io.read_image(path)
    out = onnx_session.run(None, {"rgb": np.ascontiguousarray(img[None])})[0][0]
    mean, p99 = _within_tolerance(out, _numpy_reference(img))
    assert mean <= MEAN_TOL and p99 <= P99_TOL, f"{path}: mean={mean:.3f} p99={p99:.0f}"
