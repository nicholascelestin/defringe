from dataclasses import dataclass


@dataclass(frozen=True)
class Param:
    name: str
    default: float
    lo: float
    hi: float
    step: float
    label: str
    help: str


GREEN_PARAMS = [
    Param("caster_min_chroma", 21.0, 0, 50, 0.5, "Minimum Chroma",
          "How saturated a warm source must be to count as a fringe-caster. Lower = more casters."),
    Param("caster_hue_lo", -54.0, -90, 0, 1, "Hue Floor",
          "Low edge of the caster hue range; lower reaches toward purple."),
    Param("caster_hue_hi", 23.0, 0, 90, 1, "Hue Ceiling",
          "High edge of the caster hue range; higher reaches toward orange."),
    Param("fringe_min_chroma", 4.5, 0, 30, 0.1, "Minimum Chroma",
          "How saturated a pixel must be to count as green fringe. Lower = catch fainter fringe."),
    Param("fringe_hue_lo", -80.0, -140, 0, 1, "Hue Floor",
          "Low edge of the green-fringe hue range; lower reaches toward green/yellow."),
    Param("fringe_hue_hi", 90.0, 0, 140, 1, "Hue Ceiling",
          "High edge of the green-fringe hue range; higher reaches toward blue/violet."),
    Param("min_area", 10.0, 0, 100, 0.5, "Minimum Area",
          "Ignore caster blobs smaller than this — ppm of frame area (resolution-independent)."),
    Param("cast_radius", 4.0, 0, 30, 0.1, "Cast Reach",
          "How far from a caster to look for its green fringe — ‰ of the frame diagonal."),
    Param("max_opacity", 0.7, 0, 1, 0.05, "Maximum Strength",
          "Cap on correction opacity. Higher = more aggressive."),
    Param("repair_spread", 0.45, 0, 5, 0.05, "Repair Spread",
          "Grow the corrected region outward before feathering — ‰ of the frame diagonal."),
    Param("feather", 0.91, 0, 5, 0.05, "Feather",
          "Soften/blur the correction's edge — ‰ of the frame diagonal."),
    Param("area_softness", 0.4, 0, 1, 0.05, "Area Softness",
          "Fade marginal-size casters in/out instead of popping — for temporal stability. 0 = hard cutoff."),
    Param("radius_softness", 0.4, 0, 1, 0.05, "Reach Softness",
          "Fade the fringe out with distance instead of a hard edge — for temporal stability. 0 = hard cutoff."),
    Param("full_strength_span", 9.5, 1, 30, 0.5, "Full-Strength Span",
          "Extra chroma above Minimum Chroma that reaches full correction. Wider = gentler."),
    Param("tone_correction_radius", 7.26, 0, 25, 0.1, "Tone Correction Radius",
          "How far out to sample the clean colour fringe is pulled toward — ‰ of the frame diagonal."),
    Param("tone_directionality", 0.5, 0, 1, 0.05, "Tone Directionality",
          "Pull repair colour from the cast side, not the caster. 0 = all directions, 1 = away from caster."),
]

PURPLE_PARAMS = [
    Param("caster_min_lightness", 88.0, 50, 100, 1, "Minimum Lightness",
          "How bright a highlight must be to count as a fringe-caster."),
    Param("min_area", 4.82, 0, 150, 0.5, "Minimum Area",
          "Ignore highlight blobs smaller than this — ppm of frame area (resolution-independent)."),
    Param("cast_radius", 7.26, 0, 30, 0.1, "Cast Reach",
          "How far from a highlight to look for its magenta fringe — ‰ of the frame diagonal."),
    Param("fringe_min_chroma", 6.0, 0, 15, 0.1, "Minimum Chroma",
          "Chroma floor — pixels below this are never flagged (noise rejection)."),
    Param("target_hue", 0.0, -45, 45, 1, "Target Hue",
          "Shift the detected fringe colour: 0 = magenta, higher = toward red, lower = toward violet/blue."),
    Param("excess_thresh", 7.5, 0, 20, 0.5, "Excess Threshold",
          "How much more magenta than the scene's overall tone a pixel must be to count as fringe. Higher = stricter."),
    Param("hue_halfwidth", 35.0, 5, 90, 1, "Hue Range (±°)",
          "How wide a band of hues around Target Hue counts as fringe. 90 = essentially no limit; lower narrows it to colours nearer the target."),
    Param("max_opacity", 1.0, 0, 1, 0.05, "Maximum Strength",
          "Cap on correction opacity. Higher = more aggressive."),
    Param("repair_spread", 2.27, 0, 5, 0.05, "Repair Spread",
          "Grow the corrected region outward before feathering — ‰ of the frame diagonal."),
    Param("feather", 0.91, 0, 5, 0.05, "Feather",
          "Soften/blur the correction's edge — ‰ of the frame diagonal."),
    Param("area_softness", 0.4, 0, 1, 0.05, "Area Softness",
          "Fade marginal highlights in/out instead of popping — for temporal stability. 0 = hard cutoff."),
    Param("radius_softness", 0.4, 0, 1, 0.05, "Reach Softness",
          "Fade the fringe out with distance instead of a hard edge — for temporal stability. 0 = hard cutoff."),
    Param("hue_softness", 10.0, 0, 45, 1, "Hue Range Softness",
          "Feather (°) on the Hue Range edge — ramp the cutoff instead of a hard boundary, for temporal stability. 0 = hard edge."),
    Param("full_strength_span", 10.0, 1, 30, 0.5, "Full-Strength Span",
          "Extra excess above the threshold that reaches full correction. Wider = gentler."),
    Param("tone_correction_radius", 7.26, 0, 25, 0.1, "Tone Correction Radius",
          "How far out to sample the clean colour fringe is pulled toward — ‰ of the frame diagonal."),
    Param("tone_directionality", 0.7, 0, 1, 0.05, "Tone Directionality",
          "Pull repair colour from the fringe side, not the highlight. 0 = all directions, 1 = away from highlight."),
]

GREEN_BY_NAME = {p.name: p for p in GREEN_PARAMS}
PURPLE_BY_NAME = {p.name: p for p in PURPLE_PARAMS}
GREEN_DEFAULTS = {p.name: p.default for p in GREEN_PARAMS}
PURPLE_DEFAULTS = {p.name: p.default for p in PURPLE_PARAMS}
