"""Streamlit page: Vascular Ultrasound Analysis.

Workflow
--------
1. Upload B-mode / duplex ultrasound file (DICOM, MP4, AVI, PNG, JPG).
2. Optional: crop to ROI, set calibration.
3. Run vessel segmentation (edge detection).
4. View overlay and geometric measurements.
5. Enter / extract Doppler values → flow calculation.
6. Multi-vessel WBBF panel.
7. Export results as CSV.
"""

from __future__ import annotations

import io

import numpy as np
import streamlit as st


# ---------------------------------------------------------------------------
# Public entry point called from app.py
# ---------------------------------------------------------------------------

def render_vascular_page() -> None:
    st.header("Vascular Ultrasound Analysis")
    st.caption(
        "Research use only. Results are not validated for clinical decision-making."
    )

    _check_deps()

    upload_col, settings_col = st.columns([2, 1])

    with upload_col:
        uploaded = st.file_uploader(
            "Upload ultrasound file",
            type=["dcm", "mp4", "avi", "mov", "png", "jpg", "jpeg", "tif", "tiff"],
            help="DICOM (.dcm), video (.mp4, .avi), or still image (.png, .jpg, .tif)",
        )

    with settings_col:
        st.subheader("Acquisition settings")
        pixel_spacing_input = st.number_input(
            "Pixel spacing (mm/pixel)",
            min_value=0.01,
            max_value=2.0,
            value=0.10,
            step=0.01,
            format="%.3f",
            help="From DICOM metadata (auto) or measured from scale bar. "
                 "Typical range 0.05–0.30 mm/pixel.",
        )
        vessel_label = st.selectbox(
            "Vessel",
            ["ICA_L", "ICA_R", "VA_L", "VA_R", "CCA_L", "CCA_R", "MCA", "OTHER"],
        )
        despeckle_method = st.selectbox(
            "Despeckle",
            ["bilateral", "median", "nlm"],
            help="bilateral = balanced quality/speed; nlm = highest quality, slowest",
        )
        refine_contour = st.checkbox(
            "Active-contour refinement",
            value=False,
            help="Runs scikit-image snake refinement. Slower but more precise.",
        )

    if uploaded is None:
        st.info("Upload an ultrasound file to begin.")
        _render_wbbf_panel()
        return

    # ------------------------------------------------------------------
    # Load file
    # ------------------------------------------------------------------
    from vascular_us.ingest import load_file

    with st.spinner("Loading file..."):
        try:
            scan = load_file(uploaded.getvalue(), filename=uploaded.name)
        except Exception as exc:
            st.error(f"Failed to load file: {exc}")
            return

    # Auto-fill pixel spacing from DICOM if available
    if scan.pixel_spacing_mm is not None:
        pixel_spacing_mm = scan.pixel_spacing_mm
        st.success(f"DICOM pixel spacing: **{pixel_spacing_mm:.3f} mm/pixel**")
    else:
        pixel_spacing_mm = pixel_spacing_input
        if scan.modality == "DICOM":
            st.warning("Pixel spacing not found in DICOM metadata — using manual value.")

    st.write(
        f"**Loaded**: `{scan.source}` | "
        f"**Modality**: {scan.modality} | "
        f"**Frames**: {len(scan.frames)}"
    )

    # ------------------------------------------------------------------
    # Frame selection
    # ------------------------------------------------------------------
    if len(scan.frames) > 1:
        frame_idx = st.slider(
            "Select frame",
            min_value=0,
            max_value=len(scan.frames) - 1,
            value=0,
            help="Choose the frame with the clearest vessel cross-section.",
        )
    else:
        frame_idx = 0

    raw_frame = scan.frames[frame_idx]

    # ------------------------------------------------------------------
    # Duplex split detection (B-mode + Doppler spectrum)
    # ------------------------------------------------------------------
    from vascular_us.preprocess import detect_duplex_split

    split_row = detect_duplex_split(raw_frame)
    has_spectrum = split_row is not None

    if has_spectrum:
        bmode_frame = raw_frame[:split_row, :]
        spectrum_frame = raw_frame[split_row:, :]
        st.info(
            f"Duplex image detected — B-mode above row {split_row}, "
            f"Doppler spectrum below."
        )
    else:
        bmode_frame = raw_frame
        spectrum_frame = None

    # ------------------------------------------------------------------
    # ROI selection
    # ------------------------------------------------------------------
    with st.expander("ROI crop (optional)", expanded=False):
        st.write(
            "Crop to the vessel region to improve accuracy and speed. "
            "Coordinates are in pixels (col, row) from top-left."
        )
        h, w = bmode_frame.shape[:2]
        roi_col1, roi_col2 = st.columns(2)
        x1 = roi_col1.number_input("x1", 0, w - 1, 0)
        y1 = roi_col1.number_input("y1", 0, h - 1, 0)
        x2 = roi_col2.number_input("x2", 0, w, w)
        y2 = roi_col2.number_input("y2", 0, h, h)
        use_roi = st.checkbox("Apply ROI crop")
        roi = (int(x1), int(y1), int(x2), int(y2)) if use_roi else None

    # ------------------------------------------------------------------
    # Preprocessing + segmentation
    # ------------------------------------------------------------------
    analyze_btn = st.button("Run segmentation", type="primary")

    if analyze_btn:
        _run_and_display(
            bmode_frame=bmode_frame,
            spectrum_frame=spectrum_frame,
            raw_frame=raw_frame,
            roi=roi,
            pixel_spacing_mm=pixel_spacing_mm,
            vessel_label=vessel_label,
            despeckle_method=despeckle_method,
            refine=refine_contour,
            source_name=scan.source,
            frame_idx=frame_idx,
            split_row=split_row,
        )

    # ------------------------------------------------------------------
    # Always-visible WBBF panel
    # ------------------------------------------------------------------
    st.divider()
    _render_wbbf_panel()


# ---------------------------------------------------------------------------
# Core analysis logic
# ---------------------------------------------------------------------------

def _run_and_display(
    bmode_frame: np.ndarray,
    spectrum_frame: np.ndarray | None,
    raw_frame: np.ndarray,
    roi,
    pixel_spacing_mm: float,
    vessel_label: str,
    despeckle_method: str,
    refine: bool,
    source_name: str,
    frame_idx: int,
    split_row: int | None,
) -> None:
    from vascular_us.preprocess import preprocess_frame
    from vascular_us.segment import detect_vessel
    from vascular_us.measure import measure_vessel, measurements_to_dict
    from vascular_us.doppler import (
        DopplerMeasurements,
        calculate_flow,
        doppler_to_dict,
    )
    from vascular_us.report import build_results_dataframe, export_csv

    with st.spinner("Preprocessing..."):
        processed, offset = preprocess_frame(
            bmode_frame,
            roi=roi,
            despeckle_method=despeckle_method,
        )

    with st.spinner("Detecting vessel..."):
        try:
            seg = detect_vessel(processed, refine=refine)
        except ValueError as exc:
            st.error(str(exc))
            return

    with st.spinner("Measuring..."):
        meas = measure_vessel(seg, pixel_spacing_mm=pixel_spacing_mm)

    # ------------------------------------------------------------------
    # Overlay visualization
    # ------------------------------------------------------------------
    overlay = _draw_overlay(processed, seg, offset)

    img_col, metric_col = st.columns([3, 2])
    with img_col:
        st.subheader("Segmentation result")
        st.image(overlay, caption="Green = detected lumen boundary", use_container_width=True)

    with metric_col:
        st.subheader("Geometric measurements")
        _render_measurements(meas)
        st.caption(
            f"Confidence: {seg.confidence:.2f} | "
            f"Refined: {'yes' if seg.refined else 'no'}"
        )

    # ------------------------------------------------------------------
    # Doppler panel
    # ------------------------------------------------------------------
    st.subheader("Doppler / Flow")
    dop_col1, dop_col2, dop_col3 = st.columns(3)

    with dop_col1:
        psv = st.number_input(
            "Peak systolic velocity (cm/s)",
            min_value=0.0,
            max_value=500.0,
            value=80.0,
            step=1.0,
            help="Read from the ultrasound machine Doppler display",
        )
    with dop_col2:
        edv = st.number_input(
            "End-diastolic velocity (cm/s)",
            min_value=0.0,
            max_value=300.0,
            value=25.0,
            step=1.0,
        )
    with dop_col3:
        tamv = st.number_input(
            "Time-averaged mean velocity (cm/s)",
            min_value=0.0,
            max_value=300.0,
            value=40.0,
            step=1.0,
        )

    angle_deg = st.slider(
        "Insonation angle (°)",
        min_value=0,
        max_value=70,
        value=0,
        help="Angle between Doppler beam and vessel axis. Should be <60° for accuracy.",
    )
    if angle_deg > 60:
        st.warning("Insonation angle >60° — Doppler velocity measurement unreliable.")

    if tamv > 0 and meas.area_mm2 is not None:
        dop = DopplerMeasurements(psv_cm_s=psv, edv_cm_s=edv, tamv_cm_s=tamv)
        flow = calculate_flow(tamv, meas.area_mm2, angle_deg=angle_deg)
        dop.flow_ml_min = flow

        d_col1, d_col2, d_col3, d_col4 = st.columns(4)
        d_col1.metric("PI", f"{dop.pi:.2f}")
        d_col2.metric("RI", f"{dop.ri:.2f}")
        d_col3.metric("Flow (mL/min)", f"{flow:.0f}")
        d_col4.metric("Area (mm²)", f"{meas.area_mm2:.2f}")

        # Store in session state for WBBF panel
        st.session_state[f"flow_{vessel_label}"] = flow
        st.success(
            f"Flow saved for **{vessel_label}**: {flow:.0f} mL/min. "
            "Scroll down to the WBBF panel."
        )
    elif meas.area_mm2 is None:
        st.warning(
            "Pixel spacing not calibrated — area unknown, flow cannot be calculated."
        )

    # ------------------------------------------------------------------
    # Spectrum auto-extraction (if duplex image)
    # ------------------------------------------------------------------
    if spectrum_frame is not None:
        with st.expander("Auto-extract from Doppler spectrum image", expanded=False):
            st.image(spectrum_frame, caption="Doppler spectrum region", use_container_width=True)
            v_scale = st.number_input(
                "Velocity scale (cm/s per pixel)",
                min_value=0.01,
                max_value=5.0,
                value=0.5,
                step=0.05,
                help="Read the max velocity on the y-axis of the Doppler spectrum "
                     "and divide by the image height in pixels.",
            )
            if st.button("Extract velocities from spectrum"):
                _auto_extract_spectrum(spectrum_frame, v_scale)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    st.subheader("Export")
    dop_for_export = None
    if tamv > 0 and meas.area_mm2 is not None:
        dop_for_export = DopplerMeasurements(psv_cm_s=psv, edv_cm_s=edv, tamv_cm_s=tamv)
        dop_for_export.flow_ml_min = flow

    try:
        df = build_results_dataframe(
            filename=source_name,
            vessel_label=vessel_label,
            frame_indices=[frame_idx],
            measurements=[meas],
            doppler_results=[dop_for_export],
        )
        st.dataframe(df, use_container_width=True)
        csv_bytes = export_csv(df)
        st.download_button(
            label="Download CSV",
            data=csv_bytes,
            file_name=f"{source_name}_{vessel_label}_frame{frame_idx}.csv",
            mime="text/csv",
        )
    except ImportError:
        st.warning("pandas not installed — CSV export unavailable.")


# ---------------------------------------------------------------------------
# WBBF panel
# ---------------------------------------------------------------------------

def _render_wbbf_panel() -> None:
    st.subheader("Whole-Brain Blood Flow (WBBF)")
    st.write(
        "Enter or review flow values for each vessel. "
        "Values auto-populated when you run analysis above."
    )

    vessels = ["ICA_L", "ICA_R", "VA_L", "VA_R"]
    flows: dict[str, float | None] = {}

    cols = st.columns(len(vessels))
    for col, v in zip(cols, vessels):
        stored = st.session_state.get(f"flow_{v}")
        val = col.number_input(
            f"{v} (mL/min)",
            min_value=0.0,
            max_value=2000.0,
            value=float(stored) if stored else 0.0,
            step=1.0,
            key=f"wbbf_input_{v}",
        )
        flows[v] = val if val > 0 else None

    from vascular_us.doppler import calculate_whole_brain_flow

    wbbf = calculate_whole_brain_flow(flows)
    measured_count = sum(1 for v in flows.values() if v is not None)

    if measured_count > 0:
        wbbf_col, ref_col = st.columns(2)
        wbbf_col.metric(
            "WBBF (mL/min)",
            f"{wbbf:.0f}",
            help=f"Sum of {measured_count}/{len(vessels)} vessels measured.",
        )
        if measured_count < len(vessels):
            wbbf_col.caption(
                f"Partial — {len(vessels) - measured_count} vessel(s) not yet measured."
            )
        ref_col.markdown(
            "**Reference ranges**\n"
            "- Normal adult WBBF: 700–900 mL/min\n"
            "- ICA (each): 200–350 mL/min\n"
            "- VA (each): 80–200 mL/min"
        )

        try:
            from vascular_us.report import build_wbbf_summary, export_csv
            import pandas as pd
            wbbf_df = build_wbbf_summary(flows)
            st.download_button(
                "Download WBBF CSV",
                data=export_csv(wbbf_df),
                file_name="wbbf_summary.csv",
                mime="text/csv",
            )
        except ImportError:
            pass
    else:
        st.info("Analyse at least one vessel to compute WBBF.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _draw_overlay(
    frame: np.ndarray,
    seg: "SegmentResult",  # noqa: F821
    offset: tuple[int, int],
) -> np.ndarray:
    """Draw contour on a colour copy of *frame*."""
    try:
        import cv2
    except ImportError:
        return frame

    colour = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    pts = (seg.contour[:, ::-1] + np.array(offset)).astype(np.int32)  # (col, row)
    cv2.polylines(colour, [pts.reshape(-1, 1, 2)], isClosed=True, color=(0, 255, 0), thickness=2)
    cy, cx = seg.centroid
    cx = int(cx + offset[0])
    cy = int(cy + offset[1])
    cv2.drawMarker(colour, (cx, cy), color=(0, 255, 0), markerType=cv2.MARKER_CROSS, markerSize=12, thickness=2)
    return cv2.cvtColor(colour, cv2.COLOR_BGR2RGB)


def _render_measurements(meas: "VesselMeasurements") -> None:  # noqa: F821
    rows = []
    if meas.area_mm2 is not None:
        rows.append(("Area", f"{meas.area_mm2:.2f} mm²"))
    else:
        rows.append(("Area", f"{meas.area_px2:.0f} px²  (uncalibrated)"))

    if meas.diameter_mean_mm is not None:
        rows.append(("Mean diameter", f"{meas.diameter_mean_mm:.2f} mm"))
    if meas.diameter_min_mm is not None:
        rows.append(("Min diameter", f"{meas.diameter_min_mm:.2f} mm"))
    if meas.diameter_max_mm is not None:
        rows.append(("Max diameter", f"{meas.diameter_max_mm:.2f} mm"))
    if meas.perimeter_mm is not None:
        rows.append(("Perimeter", f"{meas.perimeter_mm:.2f} mm"))

    rows.append(("Circularity", f"{meas.circularity:.3f}"))
    if meas.eccentricity is not None:
        rows.append(("Eccentricity", f"{meas.eccentricity:.3f}"))

    for label, value in rows:
        c1, c2 = st.columns([1, 1])
        c1.write(f"**{label}**")
        c2.write(value)


def _auto_extract_spectrum(spectrum_frame: np.ndarray, v_scale: float) -> None:
    from vascular_us.doppler import extract_doppler_from_spectrum
    try:
        dop = extract_doppler_from_spectrum(spectrum_frame, velocity_scale_cm_s_per_px=v_scale)
        st.success(
            f"Extracted — PSV: {dop.psv_cm_s:.1f} cm/s | "
            f"EDV: {dop.edv_cm_s:.1f} cm/s | "
            f"TAMV: {dop.tamv_cm_s:.1f} cm/s | "
            f"PI: {dop.pi:.2f} | RI: {dop.ri:.2f}"
        )
        st.info("Copy these values into the Doppler fields above and re-run flow calculation.")
    except Exception as exc:
        st.error(f"Spectrum extraction failed: {exc}")


def _check_deps() -> None:
    missing = []
    for pkg in ("cv2", "PIL"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        st.error(
            "Missing required packages. Install with:\n"
            "```\npip install opencv-python-headless Pillow scikit-image pydicom pandas\n```"
        )
        st.stop()
