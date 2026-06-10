"""Mixed-precision (fp16) variant of defringe.onnx.

Heavy ops (Conv, pooling, elementwise) -> fp16; the structurally awkward ops
(Sort/TopK, Einsum) and the small-epsilon arithmetic (Div/Sqrt/Add) stay fp32
to avoid the converter's type errors and fp16 underflow -> NaN. I/O stays fp32.
"""
import sys, numpy as np, onnx, onnxruntime as ort
from onnxconverter_common import float16
from pathlib import Path
def Path_size(p): return Path(p).stat().st_size/1e3

# fp32-retained ops: percentile (TopK), colour matmul (Einsum), and the
# epsilon-sensitive reduction/divide chain that can underflow in fp16.
BLOCK = ["TopK", "Einsum", "Div", "Sqrt", "Where", "Greater", "GreaterOrEqual"]

m = onnx.load("onnx/defringe.onnx")
m16 = float16.convert_float_to_float16(m, keep_io_types=True, op_block_list=BLOCK)
onnx.save(m16, "onnx/defringe_fp16.onnx")

# how much of the graph actually went fp16?
def count(model):
    c = {}
    for n in model.graph.node:
        c[n.op_type] = c.get(n.op_type, 0) + 1
    return c
n16 = sum(1 for n in m16.graph.node if n.op_type == "Cast")
print("blocked (kept fp32):", ", ".join(BLOCK))
print(f"fp16 model: {Path_size('onnx/defringe_fp16.onnx'):.0f} KB "
      f"(fp32 was {Path_size('onnx/defringe.onnx'):.0f} KB), Cast nodes inserted: {n16}")
