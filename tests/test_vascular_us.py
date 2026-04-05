"""Unit tests for the vascular_us pipeline.

These tests use synthetic images/data so no real ultrasound files are required.
All tests must pass without opencv or pydicom installed (graceful import guards).
"""

from __future__ import annotations

import math
import io

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_vessel_frame(
    h: int = 200,
    w: int = 200,
    cx: int = 100,
    cy: int = 100,
    radius: int = 30,
    noise: float = 10.0,
) -> np.ndarray:
    """Synthetic B-mode frame: dark circular lumen on grey background."""
    rng = np.random.default_rng(42)
    frame = np.full((h, w), 128, dtype=np.float32)
    # Add background speckle
    frame += rng.normal(0, noise, (h, w)).astype(np.float32)

    # Dark anechoic lumen
    yy, xx = np.ogrid[:h, :w]
    lumen_mask = (xx - cx) ** 2 + (yy - cy) ** 2 < radius ** 2
    frame[lumen_mask] = rng.uniform(0, 20, lumen_mask.sum())

    return np.clip(frame, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------

class TestLoadFile:
    def test_load_png_bytes(self):
        pytest.importorskip("PIL")
        from PIL import Image
        from vascular_us.ingest import load_file

        frame = _make_vessel_frame()
        img = Image.fromarray(frame)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        scan = load_file(buf.getvalue(), filename="test.png")
        assert scan.modality == "IMAGE"
        assert len(scan.frames) == 1
        assert scan.frames[0].shape == (200, 200)
        assert scan.frames[0].dtype == np.uint8

    def test_unsupported_extension_raises(self):
        from vascular_us.ingest import load_file
        with pytest.raises(ValueError, match="Unsupported"):
            load_file(b"dummy", filename="scan.xyz")


# ---------------------------------------------------------------------------
# preprocess
# ---------------------------------------------------------------------------

class TestPreprocess:
    def test_despeckle_bilateral(self):
        pytest.importorskip("cv2")
        from vascular_us.preprocess import despeckle

        frame = _make_vessel_frame(noise=30.0)
        result = despeckle(frame, method="bilateral")
        assert result.shape == frame.shape
        assert result.dtype == np.uint8

    def test_despeckle_median(self):
        pytest.importorskip("cv2")
        from vascular_us.preprocess import despeckle

        frame = _make_vessel_frame()
        result = despeckle(frame, method="median")
        assert result.shape == frame.shape

    def test_despeckle_unknown_raises(self):
        pytest.importorskip("cv2")
        from vascular_us.preprocess import despeckle

        with pytest.raises(ValueError, match="Unknown"):
            despeckle(_make_vessel_frame(), method="magic")

    def test_enhance_contrast(self):
        pytest.importorskip("cv2")
        from vascular_us.preprocess import enhance_contrast

        frame = _make_vessel_frame()
        result = enhance_contrast(frame)
        assert result.shape == frame.shape
        assert result.dtype == np.uint8

    def test_crop_roi(self):
        from vascular_us.preprocess import crop_roi

        frame = np.zeros((100, 100), dtype=np.uint8)
        cropped, offset = crop_roi(frame, roi=(10, 20, 60, 70))
        assert cropped.shape == (50, 50)
        assert offset == (10, 20)

    def test_preprocess_frame_no_roi(self):
        pytest.importorskip("cv2")
        from vascular_us.preprocess import preprocess_frame

        frame = _make_vessel_frame()
        processed, offset = preprocess_frame(frame)
        assert processed.shape == frame.shape
        assert offset == (0, 0)

    def test_detect_duplex_split_returns_none_on_uniform(self):
        from vascular_us.preprocess import detect_duplex_split

        uniform = np.full((200, 200), 128, dtype=np.uint8)
        result = detect_duplex_split(uniform)
        assert result is None

    def test_detect_duplex_split_finds_band(self):
        from vascular_us.preprocess import detect_duplex_split

        frame = np.full((200, 200), 128, dtype=np.uint8)
        frame[90:95, :] = 5  # thin dark band at row 90–95
        result = detect_duplex_split(frame)
        assert result is not None
        assert 80 <= result <= 100


# ---------------------------------------------------------------------------
# segment
# ---------------------------------------------------------------------------

class TestSegment:
    def test_detect_vessel_finds_circle(self):
        pytest.importorskip("cv2")
        from vascular_us.preprocess import preprocess_frame
        from vascular_us.segment import detect_vessel

        frame = _make_vessel_frame(cx=100, cy=100, radius=30)
        processed, _ = preprocess_frame(frame)
        seg = detect_vessel(processed)

        assert seg.mask.shape == processed.shape
        assert seg.area_px2 > 0
        assert len(seg.contour) > 10
        # Centroid should be roughly at (100, 100)
        cy, cx = seg.centroid
        assert 70 <= cy <= 130
        assert 70 <= cx <= 130

    def test_detect_vessel_confidence_positive(self):
        pytest.importorskip("cv2")
        from vascular_us.preprocess import preprocess_frame
        from vascular_us.segment import detect_vessel

        frame = _make_vessel_frame()
        processed, _ = preprocess_frame(frame)
        seg = detect_vessel(processed)
        assert 0.0 <= seg.confidence <= 1.0

    def test_no_vessel_raises(self):
        pytest.importorskip("cv2")
        from vascular_us.segment import detect_vessel

        # Completely uniform frame — no dark lumen
        uniform = np.full((100, 100), 180, dtype=np.uint8)
        with pytest.raises(ValueError, match="No vessel"):
            detect_vessel(uniform, min_area_px=500)


# ---------------------------------------------------------------------------
# measure
# ---------------------------------------------------------------------------

class TestMeasure:
    def _get_seg(self):
        pytest.importorskip("cv2")
        from vascular_us.preprocess import preprocess_frame
        from vascular_us.segment import detect_vessel

        frame = _make_vessel_frame(cx=100, cy=100, radius=30)
        processed, _ = preprocess_frame(frame)
        return detect_vessel(processed)

    def test_measurements_with_calibration(self):
        pytest.importorskip("cv2")
        from vascular_us.measure import measure_vessel

        seg = self._get_seg()
        meas = measure_vessel(seg, pixel_spacing_mm=0.1)

        assert meas.area_mm2 is not None
        assert meas.area_mm2 > 0
        assert meas.diameter_mean_mm is not None
        assert meas.circularity > 0

        # A 30-pixel radius circle at 0.1 mm/px ≈ 3 mm radius ≈ 28 mm²
        assert 10 < meas.area_mm2 < 50

    def test_measurements_without_calibration(self):
        pytest.importorskip("cv2")
        from vascular_us.measure import measure_vessel

        seg = self._get_seg()
        meas = measure_vessel(seg, pixel_spacing_mm=None)

        assert meas.area_mm2 is None
        assert meas.area_px2 > 0

    def test_circularity_range(self):
        pytest.importorskip("cv2")
        from vascular_us.measure import measure_vessel

        seg = self._get_seg()
        meas = measure_vessel(seg)
        assert 0.0 <= meas.circularity <= 1.0


# ---------------------------------------------------------------------------
# doppler
# ---------------------------------------------------------------------------

class TestDoppler:
    def test_doppler_indices(self):
        from vascular_us.doppler import DopplerMeasurements

        dop = DopplerMeasurements(psv_cm_s=100.0, edv_cm_s=20.0, tamv_cm_s=50.0)
        assert math.isclose(dop.pi, (100 - 20) / 50)
        assert math.isclose(dop.ri, (100 - 20) / 100)

    def test_calculate_flow_zero_angle(self):
        from vascular_us.doppler import calculate_flow

        flow = calculate_flow(tamv_cm_s=50.0, area_mm2=28.27, angle_deg=0.0)
        # 50 cm/s × 0.2827 cm² × 60 = 848 mL/min
        assert math.isclose(flow, 50.0 * 0.2827 * 60.0, rel_tol=1e-3)

    def test_calculate_flow_with_angle(self):
        from vascular_us.doppler import calculate_flow

        # Non-zero angle: corrected_tamv = tamv / cos(angle)
        flow_0 = calculate_flow(50.0, 28.27, angle_deg=0.0)
        flow_45 = calculate_flow(50.0, 28.27, angle_deg=45.0)
        # 45° correction: 1/cos(45°) ≈ 1.414 → flow_45 ≈ flow_0 * 1.414
        assert flow_45 > flow_0
        assert math.isclose(flow_45 / flow_0, 1.0 / math.cos(math.radians(45)), rel_tol=1e-3)

    def test_angle_90_raises(self):
        from vascular_us.doppler import calculate_flow

        with pytest.raises(ValueError, match="< 90"):
            calculate_flow(50.0, 28.27, angle_deg=90.0)

    def test_angle_above_60_warns(self):
        from vascular_us.doppler import calculate_flow
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            calculate_flow(50.0, 28.27, angle_deg=65.0)
            assert len(w) == 1
            assert "60°" in str(w[0].message)

    def test_whole_brain_flow(self):
        from vascular_us.doppler import calculate_whole_brain_flow

        flows = {"ICA_L": 300.0, "ICA_R": 300.0, "VA_L": 100.0, "VA_R": 100.0}
        wbbf = calculate_whole_brain_flow(flows)
        assert math.isclose(wbbf, 800.0)

    def test_whole_brain_flow_partial(self):
        from vascular_us.doppler import calculate_whole_brain_flow

        flows = {"ICA_L": 300.0, "ICA_R": None}  # missing VAs
        wbbf = calculate_whole_brain_flow(flows)
        assert math.isclose(wbbf, 300.0)


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

class TestReport:
    def _make_meas(self):
        pytest.importorskip("cv2")
        from vascular_us.preprocess import preprocess_frame
        from vascular_us.segment import detect_vessel
        from vascular_us.measure import measure_vessel

        frame = _make_vessel_frame()
        processed, _ = preprocess_frame(frame)
        seg = detect_vessel(processed)
        return measure_vessel(seg, pixel_spacing_mm=0.1)

    def test_build_results_dataframe(self):
        pytest.importorskip("cv2")
        pytest.importorskip("pandas")
        from vascular_us.report import build_results_dataframe

        meas = self._make_meas()
        df = build_results_dataframe(
            filename="test.dcm",
            vessel_label="ICA_L",
            frame_indices=[0],
            measurements=[meas],
        )
        assert len(df) == 1
        assert "area_mm2" in df.columns
        assert df["vessel"].iloc[0] == "ICA_L"

    def test_export_csv_bytes(self):
        pytest.importorskip("cv2")
        pytest.importorskip("pandas")
        from vascular_us.report import build_results_dataframe, export_csv

        meas = self._make_meas()
        df = build_results_dataframe("f.dcm", "VA_R", [0], [meas])
        csv_bytes = export_csv(df)
        assert isinstance(csv_bytes, bytes)
        assert b"area_mm2" in csv_bytes

    def test_build_wbbf_summary(self):
        pytest.importorskip("pandas")
        from vascular_us.report import build_wbbf_summary

        flows = {"ICA_L": 300.0, "ICA_R": 300.0, "VA_L": 100.0, "VA_R": 100.0}
        df = build_wbbf_summary(flows)
        assert df["WBBF_ml_min"].iloc[0] == 800.0
