#!/usr/bin/env python3
"""Defringe a video: ffmpeg decode -> ONNX (batched) -> ffmpeg encode (+audio).

Batched inference + threaded decode/encode overlap, so the GPU isn't starved by
a serial single-frame loop:

  reader thread  -> q_in  -> inference (main, batched) -> q_out -> writer thread

  python onnx/defringe_video.py IN.mp4 OUT.mp4 [--duration 10] [--batch 8] [--model M]
"""
import subprocess, time, argparse, json, threading, queue
import numpy as np, onnxruntime as ort


def probe(path):
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate,nb_frames", "-of", "json", path])
    s = json.loads(out)["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return (int(s["width"]), int(s["height"]), s["r_frame_rate"],
            float(num) / float(den), int(s.get("nb_frames") or 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--duration", type=float, default=None)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--model", default="onnx/defringe.onnx")
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--crf", default="16")
    a = ap.parse_args()

    W, H, rate, fps, nb = probe(a.input)
    fsz = W * H * 3
    span = f"first {a.duration}s" if a.duration else "full clip"
    print(f"input {W}x{H} @ {fps:.3f} fps | {span} | batch {a.batch} | model {a.model}", flush=True)

    dec = ["ffmpeg", "-v", "error"]
    if a.duration:
        dec += ["-t", str(a.duration)]
    dec += ["-i", a.input, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]

    enc = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", rate, "-i", "-"]
    if not a.no_audio:
        if a.duration:
            enc += ["-t", str(a.duration)]
        enc += ["-i", a.input, "-map", "0:v:0", "-map", "1:a:0?", "-c:a", "aac", "-shortest"]
    enc += ["-c:v", "libx264", "-crf", str(a.crf), "-pix_fmt", "yuv420p", a.output]

    dp = subprocess.Popen(dec, stdout=subprocess.PIPE)
    ep = subprocess.Popen(enc, stdin=subprocess.PIPE)
    sess = ort.InferenceSession(a.model, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    print("providers:", sess.get_providers(), flush=True)

    q_in = queue.Queue(maxsize=4)    # raw uint8 batches (B,H,W,3)
    q_out = queue.Queue(maxsize=4)   # processed frame bytes (in order)

    def reader():
        batch = []
        while True:
            buf = dp.stdout.read(fsz)
            if len(buf) < fsz:
                break
            batch.append(np.frombuffer(buf, np.uint8).reshape(H, W, 3))
            if len(batch) == a.batch:
                q_in.put(np.stack(batch, 0)); batch = []
        if batch:
            q_in.put(np.stack(batch, 0))
        q_in.put(None)

    def writer():
        while True:
            item = q_out.get()
            if item is None:
                break
            ep.stdin.write(item)

    rt = threading.Thread(target=reader, daemon=True); rt.start()
    wt = threading.Thread(target=writer, daemon=True); wt.start()

    n, t0 = 0, time.perf_counter()
    while True:
        b = q_in.get()                       # (B,H,W,3) uint8 -- raw rgb24, no CPU float work
        if b is None:
            break
        y = sess.run(None, {iname: b})[0]    # uint8 NHWC in -> uint8 NHWC out (norm/transpose on-device)
        q_out.put(y.tobytes())
        n += len(b)
        el = time.perf_counter() - t0
        print(f"  {n} frames | {el:.0f}s elapsed | {el/n*1000:.0f} ms/frame", flush=True)
    q_out.put(None)

    wt.join(); rt.join()
    ep.stdin.close(); dp.stdout.close(); ep.wait(); dp.wait()
    el = time.perf_counter() - t0
    print(f"done: {n} frames in {el:.0f}s ({el/max(n,1)*1000:.0f} ms/frame) -> {a.output}", flush=True)


if __name__ == "__main__":
    main()
