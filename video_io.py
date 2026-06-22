"""Video / image I/O for defringe (infrastructure, not domain logic).

Thin ffmpeg/cv2 wrappers that turn files into numpy RGB frames. Kept separate from
`algorithm/` so the domain logic stays pure. RGB (not BGR) because the defringe
algorithms are skimage/rgb2lab-native.
"""
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
    """Return (width, height, nb_frames, fps) for a video."""
    st = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,nb_frames,r_frame_rate,duration",
         "-of", "json", path],
        capture_output=True, text=True, check=True).stdout)["streams"][0]
    num, den = st["r_frame_rate"].split("/")
    fps = float(num) / float(den) if float(den) else 25.0
    nb = int(st.get("nb_frames") or 0)
    if not nb and st.get("duration"):
        nb = int(float(st["duration"]) * fps)
    return int(st["width"]), int(st["height"]), nb, fps


def read_image(path: str) -> np.ndarray:
    """Read a still as RGB uint8 [H,W,3]."""
    return np.asarray(Image.open(path).convert("RGB"))


def read_clip(path: str, start_sec: float, dur_sec: float) -> Tuple[np.ndarray, float]:
    """Seek to `start_sec` and decode a `dur_sec` clip into memory as full-res RGB.

    Returns (frames[N,H,W,3] uint8, fps). `-ss` before `-i` makes the seek fast
    (keyframe-accurate).
    """
    w, h, _, fps = probe(path)
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{start_sec}", "-t", f"{dur_sec}",
         "-i", path, "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
        capture_output=True, check=True).stdout
    n = len(raw) // (w * h * 3)
    frames = np.frombuffer(raw[: n * w * h * 3], np.uint8).reshape(n, h, w, 3)
    return frames, fps


def read_frame(path: str, t_sec: float, max_w: Optional[int] = None) -> np.ndarray:
    """Decode a single RGB frame at `t_sec` (fast keyframe seek). For previewing
    where an extracted clip will start, without loading the whole clip."""
    w, h, _, _ = probe(path)
    ow, oh, vf = w, h, []
    if max_w and w > max_w:
        ow = max_w
        oh = int(round(h * max_w / w / 2)) * 2
        vf = ["-vf", f"scale={ow}:{oh}"]
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{t_sec}", "-i", path, *vf,
         "-frames:v", "1", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
        capture_output=True, check=True).stdout
    return np.frombuffer(raw[: ow * oh * 3], np.uint8).reshape(oh, ow, 3)


def sample_indices(nb: int, max_n: int) -> np.ndarray:
    """Evenly-spaced frame indices, at most max_n of them."""
    step = max(1, nb // max_n) if nb else 1
    return np.arange(0, nb, step)


# ── encode ───────────────────────────────────────────────────────────────────

def encode_command(w: int, h: int, fps: float, out_path: str) -> list:
    """ffmpeg argv: raw rgb24 on stdin → BT.709-tagged H.264 mp4 at `out_path`.
    We control the RGB→YUV matrix and TAG the output BT.709, else a browser assumes 709
    over swscale's 601 default → the preview colour-shifts vs the gallery stills."""
    return ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{w}x{h}", "-r", f"{fps:.0f}", "-color_range", "pc", "-i", "-", "-an",
            "-vf", "scale=in_range=pc:out_range=tv:out_color_matrix=bt709,format=yuv420p,"
                   "setparams=range=tv:colorspace=bt709:color_primaries=bt709:color_trc=bt709",
            "-c:v", "libx264", "-crf", "16", "-preset", "fast", "-pix_fmt", "yuv420p", out_path]


@contextmanager
def encoder(w: int, h: int, fps: float, out_path: str):
    """ffmpeg encoder as a context manager: write contiguous uint8 RGB frames via
    `write_frames`, the block exit flushes stdin and waits for the mux to finish."""
    proc = subprocess.Popen(encode_command(w, h, fps, out_path), stdin=subprocess.PIPE)
    try:
        yield proc
    finally:
        proc.stdin.close()
        proc.wait()


def write_frames(enc, frames: np.ndarray) -> None:
    """Push a batch of RGB frames into an open `encoder`."""
    enc.stdin.write(np.ascontiguousarray(frames, np.uint8).tobytes())


# ── streaming decode ─────────────────────────────────────────────────────────

def iter_frame_batches(path: str, batch: int = 4) -> Iterator[np.ndarray]:
    """Stream a whole video as uint8 RGB batches [k,H,W,3] via ffmpeg, so a long source
    never has to fit in memory. Probe separately for nb/fps if you need progress totals."""
    w, h, _, _ = probe(path)
    dec = subprocess.Popen(["ffmpeg", "-v", "error", "-i", path, "-pix_fmt", "rgb24",
                            "-f", "rawvideo", "-"], stdout=subprocess.PIPE)
    fb = w * h * 3
    try:
        while True:
            raw = dec.stdout.read(fb * batch)
            if not raw:
                break
            k = len(raw) // fb
            yield np.frombuffer(raw[:k * fb], np.uint8).reshape(k, h, w, 3)
    finally:
        dec.stdout.close()
        dec.wait()
