"""Doppler spectrum analysis and volumetric flow calculations.

Supports two pathways:
1. **Spectrum-image pathway** — parses a Doppler spectrum image (velocity vs time)
   to extract PSV, EDV, TAMV, PI, RI.
2. **Manual-entry pathway** — accepts PSV/EDV/TAMV values measured directly from
   the ultrasound machine display.

Volumetric flow
---------------
Q (mL/min) = TAMV (cm/s) × CSA (cm²) × cos(θ) × 60

Whole-brain blood flow (WBBF)
------------------------------
WBBF = Q_ICA_L + Q_ICA_R + Q_VA_L + Q_VA_R
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class DopplerMeasurements:
    """Velocity and derived haemodynamic indices for one Doppler waveform."""

    psv_cm_s: float                  # peak systolic velocity  (cm/s)
    edv_cm_s: float                  # end-diastolic velocity  (cm/s)
    tamv_cm_s: float                 # time-averaged mean velocity (cm/s)

    pi: float = field(init=False)    # pulsatility index = (PSV - EDV) / TAMV
    ri: float = field(init=False)    # resistivity index = (PSV - EDV) / PSV

    # Populated by calculate_flow()
    flow_ml_min: float | None = None

    def __post_init__(self) -> None:
        if self.tamv_cm_s > 0:
            self.pi = (self.psv_cm_s - self.edv_cm_s) / self.tamv_cm_s
        else:
            self.pi = float("nan")

        if self.psv_cm_s > 0:
            self.ri = (self.psv_cm_s - self.edv_cm_s) / self.psv_cm_s
        else:
            self.ri = float("nan")


def calculate_flow(
    tamv_cm_s: float,
    area_mm2: float,
    angle_deg: float = 0.0,
) -> float:
    """Compute volumetric flow (mL/min).

    Parameters
    ----------
    tamv_cm_s:
        Time-averaged mean velocity in cm/s.
    area_mm2:
        Vessel cross-sectional area in mm².
    angle_deg:
        Insonation angle in degrees.  Should be <60° for reliable Doppler.
        When 0° (beam parallel to flow), no correction is needed.

    Returns
    -------
    Flow in mL/min.
    """
    if angle_deg >= 90:
        raise ValueError("Insonation angle must be < 90 degrees")
    if angle_deg > 60:
        import warnings
        warnings.warn(
            f"Insonation angle {angle_deg}° exceeds 60° — flow measurement unreliable.",
            stacklevel=2,
        )

    area_cm2 = area_mm2 / 100.0              # mm² → cm²
    angle_rad = math.radians(angle_deg)
    corrected_tamv = tamv_cm_s / math.cos(angle_rad) if angle_deg > 0 else tamv_cm_s
    flow_ml_s = corrected_tamv * area_cm2
    return flow_ml_s * 60.0                  # cm³/s → mL/min


def calculate_whole_brain_flow(
    vessel_flows: dict[str, float | None],
) -> float:
    """Sum bilateral ICA and vertebral artery flows for whole-brain blood flow.

    Parameters
    ----------
    vessel_flows:
        Dict mapping vessel label → flow (mL/min).  Expected keys (any subset):
        ``"ICA_L"``, ``"ICA_R"``, ``"VA_L"``, ``"VA_R"``.
        ``None`` values are treated as 0 (vessel not measured).

    Returns
    -------
    WBBF in mL/min.  Normal range ~700–900 mL/min in healthy adults.
    """
    inflow_vessels = {"ICA_L", "ICA_R", "VA_L", "VA_R"}
    total = 0.0
    for key in inflow_vessels:
        val = vessel_flows.get(key)
        if val is not None:
            total += val
    return total


def extract_doppler_from_spectrum(
    spectrum_image: "np.ndarray",  # noqa: F821  (avoid importing numpy at module level)
    velocity_scale_cm_s_per_px: float,
    baseline_row: int | None = None,
) -> DopplerMeasurements:
    """Extract Doppler indices from a spectrum image (velocity-time display).

    The Doppler spectrum image is the lower half of a duplex ultrasound export.
    Velocity is encoded on the y-axis, time on the x-axis.  The bright velocity
    envelope is extracted via column-wise edge detection.

    Parameters
    ----------
    spectrum_image:
        Grayscale uint8 array of the Doppler spectrum region (H, W).
    velocity_scale_cm_s_per_px:
        Physical velocity scale: cm/s per pixel on the y-axis.
    baseline_row:
        Row index of the zero-velocity baseline (middle of spectrum display).
        Inferred automatically if ``None``.

    Returns
    -------
    :class:`DopplerMeasurements` with PSV, EDV, TAMV, PI, RI.
    """
    import numpy as np

    try:
        import cv2
    except ImportError as exc:
        raise ImportError("opencv-python-headless required for spectrum analysis") from exc

    h, w = spectrum_image.shape[:2]

    if baseline_row is None:
        # Baseline is typically the row with the most horizontal continuity
        # Use the row closest to the middle with low variance
        mid = h // 2
        search = spectrum_image[mid - h // 6 : mid + h // 6]
        row_vars = search.var(axis=1)
        baseline_row = mid - h // 6 + int(np.argmin(row_vars))

    # Extract velocity envelope: for each column, find the row of peak intensity
    # above the baseline (positive flow, i.e., towards probe)
    upper_half = spectrum_image[:baseline_row, :]
    upper_half_blurred = cv2.GaussianBlur(upper_half, (5, 5), 0)

    # Column-wise argmax gives the row of maximum brightness = peak velocity row
    envelope_rows = np.argmax(upper_half_blurred, axis=0).astype(float)

    # Convert row to velocity: higher up (smaller row index) = higher velocity
    velocities_cm_s = (baseline_row - envelope_rows) * velocity_scale_cm_s_per_px

    # Remove zero/noise columns (where the envelope row == 0, i.e. no signal)
    valid = velocities_cm_s > velocity_scale_cm_s_per_px * 2
    if valid.sum() < 3:
        raise ValueError(
            "Could not extract reliable velocity envelope from spectrum image. "
            "Check velocity_scale_cm_s_per_px and ensure this is a Doppler spectrum."
        )

    velocities_valid = velocities_cm_s[valid]

    psv = float(velocities_valid.max())
    edv = float(velocities_valid.min())
    tamv = float(velocities_valid.mean())

    return DopplerMeasurements(psv_cm_s=psv, edv_cm_s=edv, tamv_cm_s=tamv)


def doppler_to_dict(d: DopplerMeasurements) -> dict:
    """Serialise to flat dict for CSV/DataFrame export."""
    return {
        "psv_cm_s": round(d.psv_cm_s, 1),
        "edv_cm_s": round(d.edv_cm_s, 1),
        "tamv_cm_s": round(d.tamv_cm_s, 1),
        "pi": round(d.pi, 2) if not math.isnan(d.pi) else "N/A",
        "ri": round(d.ri, 2) if not math.isnan(d.ri) else "N/A",
        "flow_ml_min": round(d.flow_ml_min, 1) if d.flow_ml_min is not None else "N/A",
    }
