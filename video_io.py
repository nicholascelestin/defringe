import json
import os
import subprocess
from contextlib import contextmanager
from typing import Iterator, Optional, Tuple

import numpy as np
from PIL import Image

VIDEO_EXT = {".mp4", ".mkv", ".avi", ".mov", ".m4v", ".webm", ".ts", ".mpg", ".mpeg"}


def is_video(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in VIDEO_EXT


def probe(path: str) -> Tuple[int, int, int, float]:
    stream = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,nb_frames,r_frame_rate,duration",
         "-of", "json", path],
        capture_output=True, text=True, check=True).stdout)["streams"][0]
    numerator, denominator = stream["r_frame_rate"].split("/")
    fps = float(numerator) / float(denominator) if float(denominator) else 25.0
    n_frames = int(stream.get("nb_frames") or 0)
    if not n_frames and stream.get("duration"):
        n_frames = int(float(stream["duration"]) * fps)
    return int(stream["width"]), int(stream["height"]), n_frames, fps


def read_image(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"))


def read_clip(path: str, start_sec: float, dur_sec: float) -> Tuple[np.ndarray, float]:
    w, h, _, fps = probe(path)
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{start_sec}", "-t", f"{dur_sec}",
         "-i", path, "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
        capture_output=True, check=True).stdout
    count = len(raw) // (w * h * 3)
    frames = np.frombuffer(raw[: count * w * h * 3], np.uint8).reshape(count, h, w, 3)
    return frames, fps


def read_frame(path: str, t_sec: float, max_w: Optional[int] = None) -> np.ndarray:
    w, h, _, _ = probe(path)
    out_w, out_h, scale_filter = w, h, []
    if max_w and w > max_w:
        out_w = max_w
        out_h = int(round(h * max_w / w / 2)) * 2
        scale_filter = ["-vf", f"scale={out_w}:{out_h}"]
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{t_sec}", "-i", path, *scale_filter,
         "-frames:v", "1", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
        capture_output=True, check=True).stdout
    return np.frombuffer(raw[: out_w * out_h * 3], np.uint8).reshape(out_h, out_w, 3)


# ── encode ───────────────────────────────────────────────────────────────────

def encode_command(w: int, h: int, fps: float, out_path: str) -> list:
    # Force the RGB→YUV matrix and TAG the output BT.709: else a browser assumes 709 over
    # swscale's 601 default, and the preview colour-shifts vs the gallery stills.
    return ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{w}x{h}", "-r", f"{fps:.0f}", "-color_range", "pc", "-i", "-", "-an",
            "-vf", "scale=in_range=pc:out_range=tv:out_color_matrix=bt709,format=yuv420p,"
                   "setparams=range=tv:colorspace=bt709:color_primaries=bt709:color_trc=bt709",
            "-c:v", "libx264", "-crf", "16", "-preset", "fast", "-pix_fmt", "yuv420p", out_path]


@contextmanager
def encoder(w: int, h: int, fps: float, out_path: str):
    process = subprocess.Popen(encode_command(w, h, fps, out_path), stdin=subprocess.PIPE)
    try:
        yield process
    finally:
        process.stdin.close()
        process.wait()


def write_frames(enc, frames: np.ndarray) -> None:
    enc.stdin.write(np.ascontiguousarray(frames, np.uint8).tobytes())


# ── streaming decode ─────────────────────────────────────────────────────────

def iter_frame_batches(path: str, batch: int = 4) -> Iterator[np.ndarray]:
    w, h, _, _ = probe(path)
    decoder = subprocess.Popen(["ffmpeg", "-v", "error", "-i", path, "-pix_fmt", "rgb24",
                                "-f", "rawvideo", "-"], stdout=subprocess.PIPE)
    frame_bytes = w * h * 3
    try:
        while True:
            raw = decoder.stdout.read(frame_bytes * batch)
            if not raw:
                break
            count = len(raw) // frame_bytes
            yield np.frombuffer(raw[:count * frame_bytes], np.uint8).reshape(count, h, w, 3)
    finally:
        decoder.stdout.close()
        decoder.wait()
