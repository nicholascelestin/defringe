import sys, time, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from defringe_torch import DefringeU8

stems = ["building","horses","inside","people"]
ref = Path("onnx/ref"); out = Path("onnx/out"); out.mkdir(exist_ok=True)
model = DefringeU8().eval()

print(f"{'image':>10} {'maxΔ':>5} {'meanΔ':>7} {'%px>2':>6} {'torch t':>8}")
tot=[]
for s in stems:
    arr = np.load(ref/f"{s}_in.npy"); exact = np.load(ref/f"{s}.npy").astype(int)
    x = torch.from_numpy(arr).unsqueeze(0)        # (1,H,W,3) uint8
    with torch.no_grad():
        t=time.perf_counter(); y = model(x); dt=time.perf_counter()-t
    o = y[0].numpy().astype(int)                  # (H,W,3) uint8
    from PIL import Image; Image.fromarray(o.astype(np.uint8)).save(out/f"{s}_onnxport.png")
    d = np.abs(o - exact)
    tot.append(d)
    print(f"{s:>10} {d.max():5d} {d.mean():7.3f} {(d>2).mean()*100:5.1f}% {dt:7.2f}s")
allp = np.concatenate([t.ravel() for t in tot])
print(f"\noverall: max {allp.max()}, mean {allp.mean():.3f}, "
      f"p99 {np.percentile(allp,99):.1f}, %px>2 {(allp>2).mean()*100:.2f}%")
