import sys, time, numpy as np, onnxruntime as ort
from pathlib import Path
from PIL import Image

stems = ["building","horses","inside","people"]
ref = Path("onnx/ref"); out = Path("onnx/out"); out.mkdir(exist_ok=True)
sess = ort.InferenceSession("onnx/defringe.onnx", providers=["CPUExecutionProvider"])

print("ONNX Runtime vs EXACT algorithm (fringe.py blur_ds=1), 0-255 scale:")
print(f"{'image':>10} {'maxΔ':>5} {'meanΔ':>7} {'p99':>4} {'%px>2':>6} {'%px>5':>6} {'ort t':>7}")
allp=[]
for s in stems:
    arr = np.load(ref/f"{s}_in.npy"); exact = np.load(ref/f"{s}.npy").astype(int)
    t=time.perf_counter(); y = sess.run(None, {"rgb": arr[None]})[0]; dt=time.perf_counter()-t
    o = y[0].astype(int)                       # uint8 NHWC in/out
    Image.fromarray(o.astype(np.uint8)).save(out/f"{s}_onnx.png")
    d = np.abs(o-exact); allp.append(d.ravel())
    print(f"{s:>10} {d.max():5d} {d.mean():7.3f} {np.percentile(d,99):4.0f} "
          f"{(d>2).mean()*100:5.1f}% {(d>5).mean()*100:5.1f}% {dt:6.2f}s")
allp=np.concatenate(allp)
print(f"\nOVERALL approximation cost: max {allp.max()}, mean {allp.mean():.3f}, "
      f"p99 {np.percentile(allp,99):.0f}, p99.9 {np.percentile(allp,99.9):.0f}, "
      f"%px>2 {(allp>2).mean()*100:.2f}%")
