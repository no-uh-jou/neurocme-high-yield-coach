"""Geometric measurements from vessel segmentation results."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from vascular_us.segment import SegmentResult


@dataclass
class VesselMeasurements:
    """All geometric measurements for one segmented vessel cross-section."""

    # Raw pixel measurements
    area_px2: float
    perimeter_px: float

    # Physical measurements (None when pixel_spacing_mm is unknown)
    area_mm2: float | None
    perimeter_mm: float | None
    diameter_min_mm: float | None
    diameter_max_mm: float | None
    diameter_mean_mm: float | None   # equivalent circular diameter = 2√(A/π)
    radius_mm: float | None          # mean radius = diameter_mean_mm / 2

    # Shape descriptors
    circularity: float               # 4π·A/P²  (1.0 = perfect circle)
    eccentricity: float | None       # 0 = circle, 1 = line

    # Calibration used
    pixel_spacing_mm: float | None


def measure_vessel(
    result: SegmentResult,
    pixel_spacing_mm: float | None = None,
) -> VesselMeasurements:
    """Compute geometric measurements from a :class:`SegmentResult`.

    Parameters
    ----------
    result:
        Output from :func:`~vascular_us.segment.detect_vessel`.
    pixel_spacing_mm:
        Spatial calibration (mm per pixel).  If ``None``, physical
        measurements are omitted.
    """
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("opencv-python-headless required for measurements") from exc

    contour_cv = result.contour[:, ::-1].astype(np.float32).reshape(-1, 1, 2)  # (N,1,2) x,y

    area_px2 = float(cv2.contourArea(contour_cv))
    perimeter_px = float(cv2.arcLength(contour_cv, closed=True))

    circularity = (
        (4 * math.pi * area_px2) / (perimeter_px ** 2)
        if perimeter_px > 0
        else 0.0
    )
    circularity = min(1.0, circularity)

    # Fit an ellipse for diameter and eccentricity (needs ≥5 points)
    diameter_min_mm: float | None = None
    diameter_max_mm: float | None = None
    eccentricity: float | None = None

    if len(result.contour) >= 5:
        try:
            _, (minor_axis_px, major_axis_px), _ = cv2.fitEllipse(
                contour_cv.astype(np.float32)
            )
            if pixel_spacing_mm is not None:
                diameter_min_mm = minor_axis_px * pixel_spacing_mm
                diameter_max_mm = major_axis_px * pixel_spacing_mm

            a = major_axis_px / 2
            b = minor_axis_px / 2
            if a > 0:
                eccentricity = math.sqrt(max(0.0, 1 - (b / a) ** 2))
        except Exception:
            pass

    # Physical conversions
    if pixel_spacing_mm is not None:
        px_to_mm2 = pixel_spacing_mm ** 2
        area_mm2 = area_px2 * px_to_mm2
        perimeter_mm = perimeter_px * pixel_spacing_mm
        diameter_mean_mm = 2 * math.sqrt(area_mm2 / math.pi)
        radius_mm = diameter_mean_mm / 2
    else:
        area_mm2 = perimeter_mm = diameter_mean_mm = radius_mm = None

    return VesselMeasurements(
        area_px2=area_px2,
        perimeter_px=perimeter_px,
        area_mm2=area_mm2,
        perimeter_mm=perimeter_mm,
        diameter_min_mm=diameter_min_mm,
        diameter_max_mm=diameter_max_mm,
        diameter_mean_mm=diameter_mean_mm,
        radius_mm=radius_mm,
        circularity=circularity,
        eccentricity=eccentricity,
        pixel_spacing_mm=pixel_spacing_mm,
    )


def measurements_to_dict(m: VesselMeasurements) -> dict:
    """Serialise measurements to a flat dict for CSV/DataFrame export."""

    def _fmt(v: float | None, decimals: int = 3) -> float | str:
        return round(v, decimals) if v is not None else "N/A"

    return {
        "area_px2": _fmt(m.area_px2, 1),
        "area_mm2": _fmt(m.area_mm2),
        "perimeter_px": _fmt(m.perimeter_px, 1),
        "perimeter_mm": _fmt(m.perimeter_mm),
        "diameter_min_mm": _fmt(m.diameter_min_mm),
        "diameter_max_mm": _fmt(m.diameter_max_mm),
        "diameter_mean_mm": _fmt(m.diameter_mean_mm),
        "radius_mm": _fmt(m.radius_mm),
        "circularity": _fmt(m.circularity),
        "eccentricity": _fmt(m.eccentricity),
        "pixel_spacing_mm": _fmt(m.pixel_spacing_mm),
    }
