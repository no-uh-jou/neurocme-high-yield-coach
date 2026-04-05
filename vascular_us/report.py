"""Build result DataFrames and export to CSV."""

from __future__ import annotations

from dataclasses import asdict
from typing import Sequence

from vascular_us.measure import VesselMeasurements, measurements_to_dict
from vascular_us.doppler import DopplerMeasurements, doppler_to_dict


def build_results_dataframe(
    filename: str,
    vessel_label: str,
    frame_indices: Sequence[int],
    measurements: Sequence[VesselMeasurements],
    doppler_results: Sequence[DopplerMeasurements | None] | None = None,
) -> "pd.DataFrame":  # noqa: F821
    """Assemble per-frame measurements into a tidy DataFrame.

    Parameters
    ----------
    filename:
        Source file name (included as a column for traceability).
    vessel_label:
        Anatomical label, e.g. ``"ICA_L"``, ``"MCA_R"``.
    frame_indices:
        0-based frame indices corresponding to each measurement.
    measurements:
        Geometric measurements from :func:`~vascular_us.measure.measure_vessel`.
    doppler_results:
        Optional Doppler measurements aligned with *measurements*.
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas is required for report generation. pip install pandas") from exc

    rows = []
    n = len(measurements)
    doppler_results = doppler_results or [None] * n

    for i, (frame_idx, meas, dop) in enumerate(
        zip(frame_indices, measurements, doppler_results)
    ):
        row: dict = {
            "filename": filename,
            "vessel": vessel_label,
            "frame_index": frame_idx,
        }
        row.update(measurements_to_dict(meas))
        if dop is not None:
            row.update(doppler_to_dict(dop))
        rows.append(row)

    return pd.DataFrame(rows)


def build_wbbf_summary(vessel_data: dict[str, float | None]) -> "pd.DataFrame":  # noqa: F821
    """Build a one-row WBBF summary table.

    Parameters
    ----------
    vessel_data:
        Dict of vessel_label → flow_ml_min (or None if not measured).
        Expected keys: ICA_L, ICA_R, VA_L, VA_R.
    """
    try:
        import pandas as pd
    except ImportError as exc:
        raise ImportError("pandas required") from exc

    from vascular_us.doppler import calculate_whole_brain_flow

    wbbf = calculate_whole_brain_flow(vessel_data)
    row = {k: (round(v, 1) if v is not None else "N/A") for k, v in vessel_data.items()}
    row["WBBF_ml_min"] = round(wbbf, 1)
    return pd.DataFrame([row])


def export_csv(df: "pd.DataFrame") -> bytes:  # noqa: F821
    """Encode DataFrame as UTF-8 CSV bytes for Streamlit download."""
    return df.to_csv(index=False).encode("utf-8")
