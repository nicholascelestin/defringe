import time, numpy as np, onnxruntime as ort
print("available providers:", ort.get_available_providers())
arr = np.load("onnx/ref/horses_in.npy")
x = (arr.astype(np.float32).transpose(2,0,1)[None])/255.0
for prov in [["CPUExecutionProvider"], ["CoreMLExecutionProvider","CPUExecutionProvider"]]:
    try:
        s = ort.InferenceSession("onnx/defringe.onnx", providers=prov)
        for _ in range(2): s.run(None, {"rgb": x})
        t=time.perf_counter()
        for _ in range(5): s.run(None, {"rgb": x})
        print(f"{prov[0]:28} {(time.perf_counter()-t)/5*1e3:7.0f} ms/img")
    except Exception as e:
        print(f"{prov[0]:28} unavailable: {type(e).__name__}")
