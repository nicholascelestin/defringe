"""Torch/ONNX port of the green_cast -> purple_cast pipeline.

Since the scale-space rewrite, defringe_algorithm.py uses the SAME geometry as this port
-- a box-sum count for Minimum Area and a square dilation + gaussian feather for Cast
Reach -- so the two now closely converge (no more label()/distance_transform_edt gap).

Spatial params are resolution-relative; ONNX bakes kernel sizes as graph constants, so they
are converted to px for a reference resolution at construction (ref_hw, default 1080p) --
re-instantiate with ref_hw=frame.shape[:2] to bake another size. Both passes are fully
convolutional: rgb<->lab, threshold, avg_pool (area), max_pool (reach), separable gaussian,
pointwise.

  forward(rgb): (N,H,W,3) uint8 -> (N,H,W,3) uint8   (norm/transpose baked in)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from defringe_algorithm import GREEN_CAST, PURPLE_CAST, RED_REF, GREEN_REF, MAGENTA_REF

# skimage-matching colour constants (sRGB, D65/2deg)
_XYZ_FROM_RGB = np.array([[0.412453, 0.357580, 0.180423],
                          [0.212671, 0.715160, 0.072169],
                          [0.019334, 0.119193, 0.950227]], np.float64)
_RGB_FROM_XYZ = np.linalg.inv(_XYZ_FROM_RGB)
_REF_WHITE = np.array([0.95047, 1.0, 1.08883], np.float64)
_LAB_EPS, _LAB_KAPPA, _LAB_OFF = 0.008856, 7.787, 16.0 / 116.0


def _gauss1d(sigma, truncate=4.0):
    if sigma <= 0:                                  # no blur (cf. numpy feather=0):
        return torch.ones(1, 1, 1, 1), 0           # 1-tap identity kernel, radius 0
    r = max(1, int(truncate * sigma + 0.5))
    xs = torch.arange(-r, r + 1, dtype=torch.float32)
    k = torch.exp(-(xs ** 2) / (2.0 * sigma * sigma))
    return (k / k.sum()).view(1, 1, 1, -1), r


class Defringe(nn.Module):
    """green_cast -> purple_cast, ONNX-able. uint8 NHWC in/out."""

    def __init__(self, green=None, purple=None, ref_hw=(1080, 1920)):
        super().__init__()
        # Spatial params arrive resolution-relative (see defringe_algorithm.cast_defringe);
        # convert to px for a reference resolution since ONNX kernel sizes are constants.
        _H, _W = ref_hw; _diag = (_W * _W + _H * _H) ** 0.5; _area = float(_H * _W)
        def _to_px(P):
            P = dict(P)
            for k in ("cast_radius", "feather", "repair_spread", "tone_correction_radius"):
                P[k] = P[k] * _diag / 1000.0           # per-mille of diagonal -> px
            P["min_area"] = P["min_area"] * _area / 1e6  # ppm of area -> px^2
            return P
        self.g = _to_px({**GREEN_CAST, **(green or {})})
        self.p = _to_px({**PURPLE_CAST, **(purple or {})})
        self.register_buffer("xyz_from_rgb", torch.tensor(_XYZ_FROM_RGB, dtype=torch.float32))
        self.register_buffer("rgb_from_xyz", torch.tensor(_RGB_FROM_XYZ, dtype=torch.float32))
        self.register_buffer("ref_white", torch.tensor(_REF_WHITE, dtype=torch.float32))
        for tag, P in (("g", self.g), ("p", self.p)):
            for nm, sig in (("feather", P["feather"]),
                            ("tone", P["tone_correction_radius"]),
                            ("reach", max(P["radius_softness"] * P["cast_radius"] * 0.4, 1e-3))):
                k, r = _gauss1d(sig)
                self.register_buffer(f"k_{tag}_{nm}", k)
                setattr(self, f"r_{tag}_{nm}", r)
            # Box-sum window for the area test (must hold ~min_area px). We test a
            # local bright-pixel COUNT, matching numpy's connected-component AREA
            # threshold, instead of a morphological opening — an opening needs a
            # solid k×k square to survive and so erased thin highlight rims, the
            # dominant casters for axial-CA purple fringe (heavy under-correction).
            setattr(self, f"areaw_{tag}", 2 * max(1, int(round(P["min_area"] ** 0.5))) + 1)

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

    def _pass(self, lab, P, tag, caster, shadow, strength):
        # `shadow` (candidate mask) and `strength` (alpha ramp, pre-clamp) are computed
        # per-detector in forward(); everything from morphology on is shared.
        L, a, b = lab[:, 0:1], lab[:, 1:2], lab[:, 2:3]
        # Minimum Area as a local bright-pixel COUNT (box-sum via avg_pool), softened
        # near the threshold for temporal stability (cf. numpy soft_step on blob area).
        w = getattr(self, f"areaw_{tag}")
        dens = F.avg_pool2d(caster, w, stride=1, padding=w // 2) * float(w * w)
        ma, aw = float(P["min_area"]), float(P["area_softness"]) * float(P["min_area"])
        if aw > 0:
            t = ((dens - ma) / (2.0 * aw) + 0.5).clamp(0, 1)
            co = caster * (t * t * (3.0 - 2.0 * t))                          # ~ Minimum Area
        else:
            co = caster * (dens >= ma).float()
        R = max(1, int(round(P["cast_radius"] * 0.8)))                       # 0.8: match numpy reach calib
        reach = self._sep(self._dilate(co, 2 * R + 1),                       # ~ Cast Reach
                          getattr(self, f"k_{tag}_reach"), getattr(self, f"r_{tag}_reach")).clamp(0, 1)
        keepS = shadow * reach * (1.0 - co)
        abase = keepS * strength.clamp(0, 1)
        sp = int(round(P["repair_spread"]))
        if sp > 0:
            abase = self._dilate(abase, 2 * sp + 1)
        alpha = self._sep(abase, getattr(self, f"k_{tag}_feather"),
                          getattr(self, f"r_{tag}_feather")).clamp(0, P["max_opacity"])
        # directional repair (cf. numpy `donor`): down-weight tone donors on the
        # caster side. `reach` approximates caster-nearness (~1 over the caster and
        # its reach, ~0 on the clean down-cast side), so multiplying it out of the
        # trust biases the tone estimate away from the source. dir=0 -> unchanged.
        trust = (1.0 - alpha) + 1e-3
        if P["tone_directionality"] > 0:
            trust = trust * (1.0 - P["tone_directionality"] * reach)
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
        gcaster = ((chroma > self.g["caster_min_chroma"]) & (hs >= self.g["caster_hue_lo"]) & (hs <= self.g["caster_hue_hi"])).float()
        # green shadow: absolute teal-green hue band; strength ramps on chroma
        gs = ((hue - GREEN_REF + 180.0) % 360.0) - 180.0
        gshadow = ((chroma > self.g["fringe_min_chroma"]) & (gs >= self.g["fringe_hue_lo"]) & (gs <= self.g["fringe_hue_hi"])).float()
        gstrength = ((chroma - self.g["fringe_min_chroma"]) / max(self.g["full_strength_span"], 1e-3)).clamp(0, 1)
        gout, gkeep = self._pass(lab, self.g, "g", gcaster, gshadow, gstrength)
        gx = gkeep * gout + (1.0 - gkeep) * x                       # composite green

        lab2 = self.rgb2lab(gx)
        L2, a2, b2 = lab2[:, 0:1], lab2[:, 1:2], lab2[:, 2:3]
        chroma2 = torch.sqrt(a2 * a2 + b2 * b2 + 1e-12)
        pcaster = (L2 > self.p["caster_min_lightness"]).float()
        # purple shadow: magenta EXCESS over the scene's global lighting tone (warm<->cool),
        # measured along target_hue; strength ramps on that excess. The reference is the
        # luminance-weighted mean of a*/b* over lit, non-clipped pixels, reduced per-frame
        # (keepdim over H,W) -> matches numpy's global (a_ref,b_ref).
        th = np.radians(MAGENTA_REF + self.p["target_hue"])
        um_a, um_b = float(np.cos(th)), float(np.sin(th))
        wmask = L2 * (L2 < self.p["caster_min_lightness"]).float()
        wsum = wmask.sum(dim=(2, 3), keepdim=True) + 1e-6
        a_ref = (wmask * a2).sum(dim=(2, 3), keepdim=True) / wsum
        b_ref = (wmask * b2).sum(dim=(2, 3), keepdim=True) / wsum
        ex_a, ex_b = a2 - a_ref, b2 - b_ref
        mexcess = ex_a * um_a + ex_b * um_b
        # angular gate: accept only excess hues within +-hue_halfwidth of the target
        # direction (mirrors purple_cast.py). soft_step smoothstep feathers the edge.
        perp = -ex_a * um_b + ex_b * um_a
        ang = torch.atan2(perp.abs(), mexcess) * (180.0 / np.pi)        # 0 = on target hue
        hw, hsoft = self.p["hue_halfwidth"] + 1e-6, self.p["hue_softness"]
        if hsoft > 0:
            t = ((-ang - (-hw)) / (2.0 * hsoft) + 0.5).clamp(0, 1)
            hue_w = t * t * (3.0 - 2.0 * t)
        else:
            hue_w = (ang < hw).float()
        pshadow = ((chroma2 > self.p["fringe_min_chroma"]) & (mexcess > self.p["excess_thresh"]) & (hue_w > 0)).float()
        pstrength = (((mexcess - self.p["excess_thresh"]) / max(self.p["full_strength_span"], 1e-3)).clamp(0, 1)) * hue_w
        pout, pkeep = self._pass(lab2, self.p, "p", pcaster, pshadow, pstrength)
        px = pkeep * pout + (1.0 - pkeep) * gx                      # composite purple
        y = (px.clamp(0.0, 1.0) * 255.0).round()
        return y.permute(0, 2, 3, 1).to(torch.uint8)
