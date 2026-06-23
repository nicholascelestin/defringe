DIAG_KEYS = ("cast_radius", "feather", "repair_spread", "tone_correction_radius")

REACH_CALIB = 0.8          # square-dilation radius = reach_px * this — calibrated to the former Euclidean reach
REACH_FEATHER_CALIB = 0.4  # reach gaussian sigma = radius_softness * reach_px * this


def relative_to_px(params, h, w):
    diag = (w * w + h * h) ** 0.5
    px = dict(params)
    for key in DIAG_KEYS:
        px[key] = params[key] * diag / 1000.0
    px["min_area"] = params["min_area"] * (h * w) / 1e6
    return px


def area_window(min_area_px):
    return 2 * max(1, int(round(min_area_px ** 0.5))) + 1
