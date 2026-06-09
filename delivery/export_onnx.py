import sys, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from defringe_torch import DefringeU8

# uint8 NHWC in/out, downsampled tone blurs (canonical, ~1.5x faster than exact)
model = DefringeU8(blur_ds=2).eval()
dummy = torch.zeros(1, 1080, 1920, 3, dtype=torch.uint8)
path = "onnx/defringe.onnx"
dyn = {0: "N", 1: "H", 2: "W"}     # NHWC
torch.onnx.export(model, dummy, path, opset_version=17,
                  input_names=["rgb"], output_names=["defringed"],
                  dynamic_axes={"rgb": dyn, "defringed": dyn},
                  do_constant_folding=True)
import onnx
m = onnx.load(path); onnx.checker.check_model(m)
ops = sorted({n.op_type for n in m.graph.node})
sz = Path(path).stat().st_size/1e6
print(f"exported {path}  ({sz:.1f} MB), {len(m.graph.node)} nodes")
print("op types:", ", ".join(ops))
