"""PyTorch reimplementation of fringe.py, built to export to ONNX.

This is a SEPARATE, APPROXIMATE port of ../fringe.py — the original is left
untouched. Three ops in the original have no ONNX operator, so they are
approximated here (all other steps are faithful):

  * scipy.ndimage.label  (drop edge fragments < min_edge px)
        -> morphological opening (erosion then dilation). Removes small/thin
           edge specks without true connected-component sizing.
  * distance_transform_edt + exp(-d^2/2 tol^2)  (soft reach from the gate)
        -> reach ~= clip(2*pi*tol^2 * gauss_blur(gate, tol), 0, 1).
           For an isolated gate pixel a normalised Gaussian of an impulse is
           exp(-d^2/2 tol^2)/(2 pi tol^2); multiplying back by 2 pi tol^2
           recovers the original falloff, and dense gates saturate at 1.
  * np.percentile(edge, 90)  -> exact, via TopK at a fixed image size.

rgb2lab / lab2rgb are reimplemented to match skimage (sRGB, D65/2deg).
The pass loop is unrolled (passes is fixed at export time).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- skimage-matching colour constants (sRGB, D65 2deg) ---
_XYZ_FROM_RGB = np.array([[0.412453, 0.357580, 0.180423],
                          [0.212671, 0.715160, 0.072169],
                          [0.019334, 0.119193, 0.950227]], np.float64)
_RGB_FROM_XYZ = np.linalg.inv(_XYZ_FROM_RGB)
_REF_WHITE = np.array([0.95047, 1.0, 1.08883], np.float64)
_LAB_EPS = 0.008856      # (6/29)^3
_LAB_KAPPA = 7.787       # (1/3)(29/6)^2
_LAB_OFF = 16.0 / 116.0

PURPLE = (0.71, -0.71)

DEFAULTS = dict(
    hue_power=5.0, ref_sigma=5.0, spread_win=7, edge_pct=90.0, min_edge=15,
    anomaly_thr=6.0, int_erode=2, bright_win=9, bright_L=90.0, tol=20.0,
    field_ref=12.0, strength=1.0, floor=0.15, floor_ramp=4.0, fill_L_hi=85.0,
    fill_L_soft=25.0, fold_mix=0.5, tone_sigma=25.0, normalize=True,
    passes=2, pass_decay=0.6,
)


def _gauss_kernel(sigma, truncate=4.0):
    r = int(truncate * sigma + 0.5)
    xs = torch.arange(-r, r + 1, dtype=torch.float32)
    k = torch.exp(-(xs ** 2) / (2.0 * sigma * sigma))
    return k / k.sum(), r


class Defringe(nn.Module):
    """Forward: rgb (N,3,H,W) in [0,1] -> defringed rgb (N,3,H,W) in [0,1]."""

    def __init__(self, **kw):
        super().__init__()
        self.p = {**DEFAULTS, **kw}
        self.register_buffer("xyz_from_rgb",
                             torch.tensor(_XYZ_FROM_RGB, dtype=torch.float32))
        self.register_buffer("rgb_from_xyz",
                             torch.tensor(_RGB_FROM_XYZ, dtype=torch.float32))
        self.register_buffer("ref_white",
                             torch.tensor(_REF_WHITE, dtype=torch.float32))
        # cache 1-D gaussian kernels used repeatedly
        for name, sig in (("ref", self.p["ref_sigma"]),
                          ("tone", self.p["tone_sigma"]),
                          ("tol", self.p["tol"])):
            k, r = _gauss_kernel(sig)
            self.register_buffer(f"k_{name}", k.view(1, 1, 1, -1))
            setattr(self, f"r_{name}", r)
        # downsampled-blur path for the large tone blurs: blur a 1/ds copy at
        # sigma/ds then upsample (the dominant cost; ~identical for smooth tone)
        self.ds = int(self.p.get("blur_ds", 2))
        if self.ds > 1:
            k, r = _gauss_kernel(self.p["tone_sigma"] / self.ds)
            self.register_buffer("k_tone_ds", k.view(1, 1, 1, -1))
            self.r_tone_ds = r
        # scharr kernels (scale is irrelevant: thresholded by percentile)
        sh = torch.tensor([[3, 10, 3], [0, 0, 0], [-3, -10, -3]],
                          dtype=torch.float32) / 16.0
        self.register_buffer("scharr_h", sh.view(1, 1, 3, 3))
        self.register_buffer("scharr_v", sh.t().contiguous().view(1, 1, 3, 3))

    # ---- colour ----
    def _mm(self, x, m):  # per-pixel 3x3 matrix multiply over channel dim
        return torch.einsum("ij,njhw->nihw", m, x)

    def rgb2lab(self, rgb):
        lin = torch.where(rgb > 0.04045,
                          ((rgb.clamp(min=0) + 0.055) / 1.055) ** 2.4,
                          rgb / 12.92)
        xyz = self._mm(lin, self.xyz_from_rgb) / self.ref_white.view(1, 3, 1, 1)
        f = torch.where(xyz > _LAB_EPS, xyz.clamp(min=0) ** (1.0 / 3.0),
                        _LAB_KAPPA * xyz + _LAB_OFF)
        fx, fy, fz = f[:, 0:1], f[:, 1:2], f[:, 2:3]
        L = 116.0 * fy - 16.0
        a = 500.0 * (fx - fy)
        b = 200.0 * (fy - fz)
        return torch.cat([L, a, b], 1)

    def lab2rgb(self, lab):
        L, a, b = lab[:, 0:1], lab[:, 1:2], lab[:, 2:3]
        fy = (L + 16.0) / 116.0
        fx = fy + a / 500.0
        fz = fy - b / 200.0
        f = torch.cat([fx, fy, fz], 1)
        f3 = f ** 3
        xyz = torch.where(f3 > _LAB_EPS, f3, (f - _LAB_OFF) / _LAB_KAPPA)
        xyz = xyz * self.ref_white.view(1, 3, 1, 1)
        lin = self._mm(xyz, self.rgb_from_xyz)
        srgb = torch.where(lin > 0.0031308,
                           1.055 * lin.clamp(min=0) ** (1.0 / 2.4) - 0.055,
                           12.92 * lin)
        return srgb.clamp(0.0, 1.0)

    # ---- separable gaussian via cached 1-D kernels ----
    def _sep(self, x, k1d, r):
        n, c, h, w = x.shape
        xb = x.reshape(n * c, 1, h, w)
        xb = F.conv2d(F.pad(xb, (r, r, 0, 0), mode="reflect"), k1d)
        xb = F.conv2d(F.pad(xb, (0, 0, r, r), mode="reflect"),
                      k1d.transpose(2, 3))
        return xb.reshape(n, c, h, w)

    def blur_ref(self, x):  return self._sep(x, self.k_ref, self.r_ref)
    def blur_tol(self, x):  return self._sep(x, self.k_tol, self.r_tol)

    def blur_tone(self, x):
        if self.ds <= 1:
            return self._sep(x, self.k_tone, self.r_tone)
        h, w = x.shape[2], x.shape[3]
        small = F.interpolate(x, scale_factor=1.0 / self.ds, mode="bilinear",
                              align_corners=False, recompute_scale_factor=False)
        small = self._sep(small, self.k_tone_ds, self.r_tone_ds)
        return F.interpolate(small, size=(h, w), mode="bilinear",
                             align_corners=False)

    # ---- morphology (same-size) ----
    @staticmethod
    def _dilate(x, k):  # max filter
        return F.max_pool2d(x, k, stride=1, padding=k // 2)

    def _erode(self, x, k):
        return 1.0 - self._dilate(1.0 - x, k)

    def _hue_weight(self, va, vb, power):
        ta, tb = PURPLE
        n = (ta * ta + tb * tb) ** 0.5
        mag = torch.sqrt(va * va + vb * vb + 1e-12)
        cos = (va * ta + vb * tb) / (mag * n + 1e-6)
        return mag * cos.clamp(0.0, 1.0) ** power

    def detect(self, lab):
        p = self.p
        L, a, b = lab[:, 0:1], lab[:, 1:2], lab[:, 2:3]
        # 1-2 anomaly / presence
        ref_a, ref_b = self.blur_ref(a), self.blur_ref(b)
        anomaly = self._hue_weight(a - ref_a, b - ref_b, p["hue_power"])
        presence = self._hue_weight(a, b, p["hue_power"])
        # 3 grow seed across band, capped by presence
        detected = torch.minimum(
            self._dilate(anomaly, p["spread_win"]), presence)
        # 4 gate: strong edge (TopK percentile) ... opened ... not interior ...
        #   ... bordering near-white
        edge = torch.sqrt(
            F.conv2d(F.pad(L, (1, 1, 1, 1), mode="reflect"), self.scharr_h) ** 2
            + F.conv2d(F.pad(L, (1, 1, 1, 1), mode="reflect"), self.scharr_v) ** 2
            + 1e-12)
        #   per-sample percentile via sort+gather with a dynamic index.
        #   Per-row (not global) so a batch of frames gets one threshold each
        #   -> batching is correct, and N=1 matches the global form exactly.
        flat = edge.reshape(edge.shape[0], -1)          # (N, H*W)
        n = flat.shape[1]
        idx = (p["edge_pct"] / 100.0 * n).to(torch.long)
        srt, _ = torch.sort(flat, dim=1)                # ascending per row
        idx_col = idx.view(1, 1).expand(flat.shape[0], 1)
        thr = srt.gather(1, idx_col).view(-1, 1, 1, 1)  # (N,1,1,1)
        strong = (edge >= thr).float()
        # label/min_edge -> morphological opening (erode then dilate)
        opened = self._dilate(self._erode(strong, 3), 3)
        interior = (anomaly > p["anomaly_thr"]).float()
        for _ in range(p["int_erode"]):
            interior = self._erode(interior, 3)
        bright_near = (self._dilate(L, p["bright_win"]) > p["bright_L"]).float()
        gate = opened * (1.0 - interior) * bright_near
        # 5 soft reach (EDT-free) on the dark side
        reach = (2.0 * np.pi * p["tol"] ** 2 * self.blur_tol(gate)).clamp(0, 1)
        darkside = ((p["fill_L_hi"] - L) / p["fill_L_soft"]).clamp(0, 1)
        return detected * reach * darkside

    def repair(self, lab, field, strength, floor):
        p = self.p
        L, a, b = lab[:, 0:1], lab[:, 1:2], lab[:, 2:3]
        ramp = (field / p["field_ref"]).clamp(0, 1)
        floor_w = (field / p["floor_ramp"]).clamp(0, 1)
        alpha = (floor * floor_w + (strength - floor) * ramp).clamp(0, 1)
        ta, tb = PURPLE
        nrm = (ta * ta + tb * tb) ** 0.5
        ua, ub = ta / nrm, tb / nrm
        rem = (a * ua + b * ub).clamp(min=0)
        ra, rb = a - rem * ua, b - rem * ub
        trust = (1.0 - alpha) + 1e-3
        den = self.blur_tone(trust) + 1e-6
        ta_ = self.blur_tone(a * trust) / den
        tb_ = self.blur_tone(b * trust) / den
        if p["normalize"]:
            ra = ra + (ta_ - self.blur_tone(ra))
            rb = rb + (tb_ - self.blur_tone(rb))
        fa = (1 - p["fold_mix"]) * ra + p["fold_mix"] * ta_
        fb = (1 - p["fold_mix"]) * rb + p["fold_mix"] * tb_
        a2 = a + alpha * (fa - a)
        b2 = b + alpha * (fb - b)
        return self.lab2rgb(torch.cat([L, a2, b2], 1))

    def forward(self, rgb):
        p = self.p
        lab = self.rgb2lab(rgb)
        field = self.detect(lab)
        out = rgb
        for i in range(max(1, p["passes"])):
            amp = p["pass_decay"] ** i
            out = self.repair(lab, field, p["strength"] * amp, p["floor"] * amp)
            if i + 1 < p["passes"]:
                lab = self.rgb2lab(out)
        return out


class DefringeU8(nn.Module):
    """uint8 NHWC in/out wrapper: the /255 normalise, NHWC<->NCHW transpose, and
    *255/round/clamp are baked into the graph and run on-device. The host side
    just hands over raw rgb24 bytes and writes raw rgb24 bytes -- no CPU float
    churn, and host<->device transfer is uint8 (4x smaller than fp32)."""

    def __init__(self, **kw):
        super().__init__()
        self.core = Defringe(**kw)

    def forward(self, rgb):                       # rgb: (N,H,W,3) uint8
        x = rgb.permute(0, 3, 1, 2).to(torch.float32) / 255.0
        y = self.core(x)                          # (N,3,H,W) float in [0,1]
        y = (y.clamp(0.0, 1.0) * 255.0).round()
        return y.permute(0, 2, 3, 1).to(torch.uint8)   # (N,H,W,3) uint8
