import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import geometry
from defringe_numpy import RED_REF, GREEN_REF, MAGENTA_REF
from parameters import GREEN_DEFAULTS, PURPLE_DEFAULTS

TRUST_FLOOR = 1e-3

# sRGB / CIELAB constants (D65, 2°) matching skimage.
_XYZ_FROM_RGB = np.array([[0.412453, 0.357580, 0.180423],
                          [0.212671, 0.715160, 0.072169],
                          [0.019334, 0.119193, 0.950227]], np.float64)
_RGB_FROM_XYZ = np.linalg.inv(_XYZ_FROM_RGB)
_REF_WHITE = np.array([0.95047, 1.0, 1.08883], np.float64)
_LAB_EPS, _LAB_KAPPA, _LAB_OFF = 0.008856, 7.787, 16.0 / 116.0


class Defringe(nn.Module):

    def __init__(self, green=None, purple=None, ref_hw=(1080, 1920)):
        super().__init__()
        _H, _W = ref_hw
        green, purple = green or {}, purple or {}
        self.g = geometry.relative_to_px({**GREEN_DEFAULTS, **green}, _H, _W)
        self.p = geometry.relative_to_px({**PURPLE_DEFAULTS, **purple}, _H, _W)
        self.ref_diag = (_W * _W + _H * _H) ** 0.5
        self.register_buffer("xyz_from_rgb", torch.tensor(_XYZ_FROM_RGB, dtype=torch.float32))
        self.register_buffer("rgb_from_xyz", torch.tensor(_RGB_FROM_XYZ, dtype=torch.float32))
        self.register_buffer("ref_white", torch.tensor(_REF_WHITE, dtype=torch.float32))
        for tag, P in (("g", self.g), ("p", self.p)):
            for name, sigma in (("feather", P["feather"]),
                                ("tone", P["tone_correction_radius"]),
                                ("reach", max(P["radius_softness"] * P["cast_radius"] * geometry.REACH_FEATHER_CALIB, 1e-3))):
                kernel, radius = gaussian_1d(sigma)
                self.register_buffer(f"k_{tag}_{name}", kernel)
                setattr(self, f"r_{tag}_{name}", radius)
            setattr(self, f"areaw_{tag}", geometry.area_window(P["min_area"]))

    def forward(self, rgb):
        x = rgb.permute(0, 3, 1, 2).to(torch.float32) / 255.0
        # Resolution-invariant: kernel sizes are baked for ref_hw, so run the pipeline at the
        # reference diagonal and resize the correction *delta* back (not the frame), which keeps
        # untouched detail and is exact at ref_hw where the resizes are no-ops.
        shape = torch._shape_as_tensor(x).to(torch.float32)
        scale = self.ref_diag / torch.sqrt(shape[2] * shape[2] + shape[3] * shape[3] + 1e-12)
        ref_h = (shape[2] * scale).round().clamp(min=1.0).to(torch.int64)
        ref_w = (shape[3] * scale).round().clamp(min=1.0).to(torch.int64)
        resampled = F.interpolate(x, size=[ref_h, ref_w], mode="bilinear", align_corners=False)
        delta = self._pipeline(resampled) - resampled
        full = torch._shape_as_tensor(x)
        delta = F.interpolate(delta, size=[full[2], full[3]], mode="bilinear", align_corners=False)
        return ((x + delta).clamp(0.0, 1.0) * 255.0).round().permute(0, 2, 3, 1).to(torch.uint8)

    def _pipeline(self, x):
        lab = self.rgb2lab(x)
        a, b = lab[:, 1:2], lab[:, 2:3]
        hue, chroma = to_polar(a, b)
        g = self.g
        casters = ((chroma > g["caster_min_chroma"]) & in_band(signed_hue_offset(hue, RED_REF), g["caster_hue_lo"], g["caster_hue_hi"])).float()
        fringe = ((chroma > g["fringe_min_chroma"]) & in_band(signed_hue_offset(hue, GREEN_REF), g["fringe_hue_lo"], g["fringe_hue_hi"])).float()
        strength = ramp(chroma, g["fringe_min_chroma"], g["full_strength_span"])
        green_out, green_keep = self._pass(lab, g, "g", casters, fringe, strength)
        blended = green_keep * green_out + (1.0 - green_keep) * x

        lab = self.rgb2lab(blended)
        L, a, b = lab[:, 0:1], lab[:, 1:2], lab[:, 2:3]
        _, chroma = to_polar(a, b)
        p = self.p
        casters = (L > p["caster_min_lightness"]).float()
        scene_a, scene_b = scene_tone(L, a, b, p["caster_min_lightness"])
        excess_a, excess_b = a - scene_a, b - scene_b
        excess = (excess_a * excess_a + excess_b * excess_b).clamp_min(0).sqrt()
        off_target = angle_from_axis(excess_a, excess_b, np.radians(MAGENTA_REF + p["target_hue"]))
        within_cone = soft_within(off_target, p["hue_halfwidth"] + 1e-6, p["hue_softness"])
        fringe = ((chroma > p["fringe_min_chroma"]) & (excess > p["excess_thresh"]) & (within_cone > 0)).float()
        strength = ramp(excess, p["excess_thresh"], p["full_strength_span"]) * within_cone
        purple_out, purple_keep = self._pass(lab, p, "p", casters, fringe, strength)
        return purple_keep * purple_out + (1.0 - purple_keep) * blended

    def _pass(self, lab, P, tag, casters, fringe, strength):
        L, a, b = lab[:, 0:1], lab[:, 1:2], lab[:, 2:3]
        caster_field = self._survive_min_area(casters, P, tag)
        reach = self._cast_reach(caster_field, P, tag)
        shadow = fringe * reach * (1.0 - caster_field)
        alpha = self._feathered_alpha(shadow * strength.clamp(0, 1), P, tag)
        image = self._repair_chroma(L, a, b, alpha, reach, P, tag)
        return image, (alpha > 0).float()

    def _survive_min_area(self, casters, P, tag):
        window = getattr(self, f"areaw_{tag}")
        neighbours = F.avg_pool2d(casters, window, stride=1, padding=window // 2) * float(window * window)
        min_area = float(P["min_area"])
        softness = float(P["area_softness"]) * min_area
        if softness > 0:
            return casters * smoothstep(min_area - softness, min_area + softness, neighbours)
        return casters * (neighbours >= min_area).float()

    def _cast_reach(self, caster_field, P, tag):
        radius = max(1, int(round(P["cast_radius"] * geometry.REACH_CALIB)))
        dilated = self._dilate(caster_field, 2 * radius + 1)
        return self._blur(dilated, tag, "reach").clamp(0, 1)

    def _feathered_alpha(self, seed, P, tag):
        spread = int(round(P["repair_spread"]))
        grown = self._dilate(seed, 2 * spread + 1)
        return self._blur(grown, tag, "feather").clamp(0, P["max_opacity"])

    def _repair_chroma(self, L, a, b, alpha, reach, P, tag):
        trust = ((1.0 - alpha) + TRUST_FLOOR) * (1.0 - P["tone_directionality"] * reach)
        denom = self._blur(trust, tag, "tone")
        tone_a = safe_divide(self._blur(a * trust, tag, "tone"), denom)
        tone_b = safe_divide(self._blur(b * trust, tag, "tone"), denom)
        corrected = torch.cat([L, lerp(a, tone_a, alpha), lerp(b, tone_b, alpha)], 1)
        return self.lab2rgb(corrected)

    # ── colour ───────────────────────────────────────────────────────────────

    def rgb2lab(self, rgb):
        linear = torch.where(rgb > 0.04045, ((rgb.clamp(min=0) + 0.055) / 1.055) ** 2.4, rgb / 12.92)
        xyz = self._apply_matrix(linear, self.xyz_from_rgb) / self.ref_white.view(1, 3, 1, 1)
        f = torch.where(xyz > _LAB_EPS, xyz.clamp(min=0) ** (1.0 / 3.0), _LAB_KAPPA * xyz + _LAB_OFF)
        fx, fy, fz = f[:, 0:1], f[:, 1:2], f[:, 2:3]
        return torch.cat([116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)], 1)

    def lab2rgb(self, lab):
        L, a, b = lab[:, 0:1], lab[:, 1:2], lab[:, 2:3]
        fy = (L + 16.0) / 116.0
        fx, fz = fy + a / 500.0, fy - b / 200.0
        f = torch.cat([fx, fy, fz], 1)
        cubed = f ** 3
        xyz = torch.where(cubed > _LAB_EPS, cubed, (f - _LAB_OFF) / _LAB_KAPPA) * self.ref_white.view(1, 3, 1, 1)
        linear = self._apply_matrix(xyz, self.rgb_from_xyz)
        srgb = torch.where(linear > 0.0031308, 1.055 * linear.clamp(min=0) ** (1.0 / 2.4) - 0.055, 12.92 * linear)
        return srgb.clamp(0.0, 1.0)

    def _apply_matrix(self, x, matrix):
        return torch.einsum("ij,njhw->nihw", matrix, x)

    # ── morphology / blur ──────────────────────────────────────────────────────

    @staticmethod
    def _dilate(x, k):
        return F.max_pool2d(x, k, stride=1, padding=k // 2)

    def _blur(self, x, tag, name):
        kernel, radius = getattr(self, f"k_{tag}_{name}"), getattr(self, f"r_{tag}_{name}")
        x = F.conv2d(F.pad(x, (radius, radius, 0, 0), mode="reflect"), kernel)
        return F.conv2d(F.pad(x, (0, 0, radius, radius), mode="reflect"), kernel.transpose(2, 3))


# ── maths (torch twins of the defringe_numpy helpers) ─────────────────────────

def smoothstep(lo, hi, x):
    t = ((x - lo) / (hi - lo)).clamp(0, 1)
    return t * t * (3.0 - 2.0 * t)


def soft_within(off_target, halfwidth, softness):
    if softness > 0:
        return smoothstep(-halfwidth - softness, -halfwidth + softness, -off_target)
    return (off_target < halfwidth).float()


def lerp(start, end, t):
    return start + t * (end - start)


def ramp(value, threshold, span):
    return ((value - threshold) / max(span, 1e-3)).clamp(0, 1)


def in_band(x, lo, hi):
    return (x >= lo) & (x <= hi)


def signed_hue_offset(hue, reference):
    return ((hue - reference + 180.0) % 360.0) - 180.0


def to_polar(a, b):
    hue = (torch.atan2(b, a) * (180.0 / np.pi)) % 360.0
    chroma = torch.sqrt(a * a + b * b + 1e-12)
    return hue, chroma


def angle_from_axis(vec_a, vec_b, axis_rad):
    axis_a, axis_b = float(np.cos(axis_rad)), float(np.sin(axis_rad))
    along = vec_a * axis_a + vec_b * axis_b
    across = -vec_a * axis_b + vec_b * axis_a
    return torch.atan2(across.abs(), along) * (180.0 / np.pi)


def scene_tone(L, a, b, blown):
    lit = L * (L < blown).float()
    weight = lit.sum(dim=(2, 3), keepdim=True)
    return (safe_divide((lit * a).sum(dim=(2, 3), keepdim=True), weight),
            safe_divide((lit * b).sum(dim=(2, 3), keepdim=True), weight))


def safe_divide(numerator, denominator):
    return numerator / (denominator + 1e-6)


def gaussian_1d(sigma, truncate=4.0):
    if sigma <= 0:
        return torch.ones(1, 1, 1, 1), 0
    radius = max(1, int(truncate * sigma + 0.5))
    offsets = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel = torch.exp(-(offsets ** 2) / (2.0 * sigma * sigma))
    return (kernel / kernel.sum()).view(1, 1, 1, -1), radius
