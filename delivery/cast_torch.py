"""Torch/ONNX port of the canonical green_cast -> purple_cast pipeline.

Option A: a SEPARATE, ONNX-able approximation of the numpy algorithm in
`algorithm/` (which is left untouched). Two ops have no ONNX equivalent and are
approximated; the per-caster area weighting is dropped:

  scipy.ndimage.label (Minimum Area)        -> morphological opening (erode/dilate)
  distance_transform_edt (Cast Reach)       -> dilate-by-reach + gaussian feather
  caster_w (per-blob area weight)           -> dropped (all kept casters equal)

Validated at ~0.1 mean ΔRGB vs the numpy pipeline (p99 <= 2 levels). Both passes
become fully convolutional: rgb<->lab, threshold, max_pool (morphology), separable
gaussian, pointwise. Params are baked from the canonical defaults at construction.

  forward(rgb): (N,H,W,3) uint8 -> (N,H,W,3) uint8   (norm/transpose baked in)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from algorithm import GREEN_CAST, PURPLE_CAST
from algorithm.green_cast import RED_REF, GREEN_REF
from algorithm.purple_cast import MAGENTA_REF

# skimage-matching colour constants (sRGB, D65/2deg)
_XYZ_FROM_RGB = np.array([[0.412453, 0.357580, 0.180423],
                          [0.212671, 0.715160, 0.072169],
                          [0.019334, 0.119193, 0.950227]], np.float64)
_RGB_FROM_XYZ = np.linalg.inv(_XYZ_FROM_RGB)
_REF_WHITE = np.array([0.95047, 1.0, 1.08883], np.float64)
_LAB_EPS, _LAB_KAPPA, _LAB_OFF = 0.008856, 7.787, 16.0 / 116.0


def _gauss1d(sigma, truncate=4.0):
    r = max(1, int(truncate * sigma + 0.5))
    xs = torch.arange(-r, r + 1, dtype=torch.float32)
    k = torch.exp(-(xs ** 2) / (2.0 * sigma * sigma))
    return (k / k.sum()).view(1, 1, 1, -1), r


class Defringe(nn.Module):
    """green_cast -> purple_cast, ONNX-able. uint8 NHWC in/out."""

    def __init__(self, green=None, purple=None):
        super().__init__()
        self.g = {**GREEN_CAST, **(green or {})}
        self.p = {**PURPLE_CAST, **(purple or {})}
        self.register_buffer("xyz_from_rgb", torch.tensor(_XYZ_FROM_RGB, dtype=torch.float32))
        self.register_buffer("rgb_from_xyz", torch.tensor(_RGB_FROM_XYZ, dtype=torch.float32))
        self.register_buffer("ref_white", torch.tensor(_REF_WHITE, dtype=torch.float32))
        for tag, P in (("g", self.g), ("p", self.p)):
            for nm, sig in (("feather", P["feather"]),
                            ("tone", P["tone_correction_radius"]),
                            ("reach", max(P["radius_soft"] * P["cast_radius"], 1e-3))):
                k, r = _gauss1d(sig)
                self.register_buffer(f"k_{tag}_{nm}", k)
                setattr(self, f"r_{tag}_{nm}", r)
            setattr(self, f"open_{tag}", max(1, int(round((P["min_area"] / np.pi) ** 0.5))))

    # ---- colour ----
    def _mm(self, x, m):
        return torch.einsum("ij,njhw->nihw", m, x)

    def rgb2lab(self, rgb):
        lin = torch.where(rgb > 0.04045, ((rgb.clamp(min=0) + 0.055) / 1.055) ** 2.4, rgb / 12.92)
        xyz = self._mm(lin, self.xyz_from_rgb) / self.ref_white.view(1, 3, 1, 1)
        f = torch.where(xyz > _LAB_EPS, xyz.clamp(min=0) ** (1.0 / 3.0), _LAB_KAPPA * xyz + _LAB_OFF)
        fx, fy, fz = f[:, 0:1], f[:, 1:2], f[:, 2:3]
        return torch.cat([116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)], 1)

    def lab2rgb(self, lab):
        L, a, b = lab[:, 0:1], lab[:, 1:2], lab[:, 2:3]
        fy = (L + 16.0) / 116.0
        fx, fz = fy + a / 500.0, fy - b / 200.0
        f = torch.cat([fx, fy, fz], 1)
        f3 = f ** 3
        xyz = torch.where(f3 > _LAB_EPS, f3, (f - _LAB_OFF) / _LAB_KAPPA) * self.ref_white.view(1, 3, 1, 1)
        lin = self._mm(xyz, self.rgb_from_xyz)
        srgb = torch.where(lin > 0.0031308, 1.055 * lin.clamp(min=0) ** (1.0 / 2.4) - 0.055, 12.92 * lin)
        return srgb.clamp(0.0, 1.0)

    # ---- morphology / blur ----
    @staticmethod
    def _dilate(x, k):
        return F.max_pool2d(x, k, stride=1, padding=k // 2)

    def _erode(self, x, k):
        return 1.0 - self._dilate(1.0 - x, k)

    def _sep(self, x, k1d, r):
        x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k1d)
        x = F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k1d.transpose(2, 3))
        return x

    def _pass(self, lab, P, tag, caster, shadow_ref, lo, hi, chr_floor):
        L, a, b = lab[:, 0:1], lab[:, 1:2], lab[:, 2:3]
        chroma = torch.sqrt(a * a + b * b + 1e-12)
        hue = (torch.atan2(b, a) * (180.0 / np.pi)) % 360.0
        ss = ((hue - shadow_ref + 180.0) % 360.0) - 180.0
        shadow = ((chroma > chr_floor) & (ss >= lo) & (ss <= hi)).float()
        ko = 2 * getattr(self, f"open_{tag}") + 1
        co = self._dilate(self._erode(caster, ko), ko)                       # ~ Minimum Area
        R = int(P["cast_radius"])
        reach = self._sep(self._dilate(co, 2 * R + 1),                       # ~ Cast Reach
                          getattr(self, f"k_{tag}_reach"), getattr(self, f"r_{tag}_reach")).clamp(0, 1)
        keepS = shadow * reach * (1.0 - co)
        abase = keepS * ((chroma - chr_floor) / max(P["full_strength_span"], 1e-3)).clamp(0, 1)
        if P["repair_spread"] > 0:
            abase = self._dilate(abase, int(2 * P["repair_spread"] + 1))
        alpha = self._sep(abase, getattr(self, f"k_{tag}_feather"),
                          getattr(self, f"r_{tag}_feather")).clamp(0, P["max_strength"])
        trust = (1.0 - alpha) + 1e-3
        kt, rt = getattr(self, f"k_{tag}_tone"), getattr(self, f"r_{tag}_tone")
        den = self._sep(trust, kt, rt) + 1e-6
        ta = self._sep(a * trust, kt, rt) / den
        tb = self._sep(b * trust, kt, rt) / den
        out = self.lab2rgb(torch.cat([L, a + alpha * (ta - a), b + alpha * (tb - b)], 1))
        keep = (alpha > 0).float()
        return out, keep

    def forward(self, rgb):                            # (N,H,W,3) uint8
        x = rgb.permute(0, 3, 1, 2).to(torch.float32) / 255.0
        lab = self.rgb2lab(x)
        a, b = lab[:, 1:2], lab[:, 2:3]
        chroma = torch.sqrt(a * a + b * b + 1e-12)
        hue = (torch.atan2(b, a) * (180.0 / np.pi)) % 360.0
        hs = ((hue - RED_REF + 180.0) % 360.0) - 180.0
        gcaster = ((chroma > self.g["cast_chr"]) & (hs >= self.g["band_lo"]) & (hs <= self.g["band_hi"])).float()
        gout, gkeep = self._pass(lab, self.g, "g", gcaster, GREEN_REF,
                                 self.g["green_lo"], self.g["green_hi"], self.g["green_chr"])
        gx = gkeep * gout + (1.0 - gkeep) * x                       # composite green
        lab2 = self.rgb2lab(gx)
        pcaster = (lab2[:, 0:1] > self.p["bright_L"]).float()
        pout, pkeep = self._pass(lab2, self.p, "p", pcaster, MAGENTA_REF,
                                 self.p["mag_lo"], self.p["mag_hi"], self.p["mag_chr"])
        px = pkeep * pout + (1.0 - pkeep) * gx                      # composite purple
        y = (px.clamp(0.0, 1.0) * 255.0).round()
        return y.permute(0, 2, 3, 1).to(torch.uint8)
