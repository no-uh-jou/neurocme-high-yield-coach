"""File ingest: DICOM, video (MP4/AVI), and still images."""

from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import numpy as np


@dataclass
class LoadedScan:
    """Normalised container for any supported ultrasound input format."""

    frames: list[np.ndarray]         # list of uint8 grayscale (H, W) arrays
    pixel_spacing_mm: float | None   # mm per pixel (isotropic assumed)
    fps: float | None                # frames-per-second; None for still images
    source: str                      # original filename or label
    modality: str                    # "DICOM", "VIDEO", or "IMAGE"
    dicom_metadata: dict = field(default_factory=dict)  # raw DICOM tags if present


def load_file(
    data: Union[bytes, str, Path],
    filename: str = "",
) -> LoadedScan:
    """Load a DICOM, video, or still-image file into a :class:`LoadedScan`.

    Parameters
    ----------
    data:
        Raw bytes (from Streamlit uploader) **or** a filesystem path.
    filename:
        Original filename — used to determine format when *data* is bytes.
    """
    if isinstance(data, (str, Path)):
        path = Path(data)
        filename = filename or path.name
        raw_bytes: bytes | None = None
    else:
        path = None
        raw_bytes = data

    ext = Path(filename).suffix.lower()

    if ext == ".dcm" or _looks_like_dicom(raw_bytes, path):
        return _load_dicom(raw_bytes, path, filename)

    if ext in {".mp4", ".avi", ".mov", ".mkv"}:
        return _load_video(raw_bytes, path, filename)

    if ext in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
        return _load_image(raw_bytes, path, filename)

    raise ValueError(
        f"Unsupported file extension '{ext}'. "
        "Supported: .dcm, .mp4, .avi, .mov, .mkv, .png, .jpg, .jpeg, .tif, .tiff, .bmp"
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _looks_like_dicom(raw_bytes: bytes | None, path: Path | None) -> bool:
    """Heuristic: DICOM files start with 128 null bytes then 'DICM'."""
    try:
        if raw_bytes is not None:
            return len(raw_bytes) > 132 and raw_bytes[128:132] == b"DICM"
        if path is not None:
            with open(path, "rb") as fh:
                header = fh.read(132)
            return header[128:132] == b"DICM"
    except Exception:
        pass
    return False


def _load_dicom(
    raw_bytes: bytes | None,
    path: Path | None,
    filename: str,
) -> LoadedScan:
    try:
        import pydicom
    except ImportError as exc:
        raise ImportError(
            "pydicom is required for DICOM support. "
            "Install with: pip install pydicom"
        ) from exc

    if raw_bytes is not None:
        ds = pydicom.dcmread(io.BytesIO(raw_bytes))
    else:
        ds = pydicom.dcmread(str(path))

    pixel_array = ds.pixel_array  # shape: (H, W) or (frames, H, W)

    # Normalise to uint8 grayscale
    def _to_gray_u8(arr: np.ndarray) -> np.ndarray:
        if arr.ndim == 3 and arr.shape[2] in (3, 4):
            # RGB/RGBA → grayscale
            arr = arr[..., :3]
            arr = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2])
        mn, mx = arr.min(), arr.max()
        if mx > mn:
            arr = (arr - mn) / (mx - mn) * 255
        return arr.astype(np.uint8)

    if pixel_array.ndim == 2:
        frames = [_to_gray_u8(pixel_array)]
    elif pixel_array.ndim == 3:
        # Could be (N, H, W) for multi-frame or (H, W, C) for colour
        if pixel_array.shape[2] in (3, 4):
            frames = [_to_gray_u8(pixel_array)]
        else:
            frames = [_to_gray_u8(pixel_array[i]) for i in range(pixel_array.shape[0])]
    else:
        raise ValueError(f"Unexpected DICOM pixel_array shape: {pixel_array.shape}")

    # Extract pixel spacing (mm/pixel)
    pixel_spacing_mm: float | None = None
    for tag in ("PixelSpacing", "ImagerPixelSpacing"):
        if hasattr(ds, tag):
            vals = getattr(ds, tag)
            pixel_spacing_mm = float(vals[0])  # row spacing
            break
    # Ultrasound-specific: SequenceOfUltrasoundRegions
    if pixel_spacing_mm is None and hasattr(ds, "SequenceOfUltrasoundRegions"):
        try:
            region = ds.SequenceOfUltrasoundRegions[0]
            dx = float(region.PhysicalDeltaX)   # cm/pixel
            pixel_spacing_mm = dx * 10.0        # → mm/pixel
        except Exception:
            pass

    # Frame rate
    fps: float | None = None
    for tag in ("CineRate", "RecommendedDisplayFrameRate", "FrameTime"):
        if hasattr(ds, tag):
            val = float(getattr(ds, tag))
            fps = 1000.0 / val if tag == "FrameTime" else val
            break

    metadata = {
        "PatientID": getattr(ds, "PatientID", ""),
        "StudyDate": getattr(ds, "StudyDate", ""),
        "Modality": getattr(ds, "Modality", ""),
        "SeriesDescription": getattr(ds, "SeriesDescription", ""),
    }

    return LoadedScan(
        frames=frames,
        pixel_spacing_mm=pixel_spacing_mm,
        fps=fps,
        source=filename,
        modality="DICOM",
        dicom_metadata=metadata,
    )


def _load_video(
    raw_bytes: bytes | None,
    path: Path | None,
    filename: str,
) -> LoadedScan:
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "opencv-python-headless is required for video support. "
            "Install with: pip install opencv-python-headless"
        ) from exc

    if raw_bytes is not None:
        # Write to a temp file since VideoCapture needs a path
        import tempfile
        suffix = Path(filename).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(raw_bytes)
            tmp_path = tmp.name
        cap = cv2.VideoCapture(tmp_path)
        import os
        os.unlink(tmp_path)
    else:
        cap = cv2.VideoCapture(str(path))

    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {filename}")

    fps = cap.get(cv2.CAP_PROP_FPS) or None
    frames: list[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
    cap.release()

    if not frames:
        raise ValueError(f"No frames extracted from video: {filename}")

    return LoadedScan(
        frames=frames,
        pixel_spacing_mm=None,  # videos rarely carry spatial calibration
        fps=fps,
        source=filename,
        modality="VIDEO",
    )


def _load_image(
    raw_bytes: bytes | None,
    path: Path | None,
    filename: str,
) -> LoadedScan:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "Pillow is required for image support. "
            "Install with: pip install Pillow"
        ) from exc

    if raw_bytes is not None:
        img = Image.open(io.BytesIO(raw_bytes)).convert("L")
    else:
        img = Image.open(path).convert("L")

    frame = np.array(img, dtype=np.uint8)
    return LoadedScan(
        frames=[frame],
        pixel_spacing_mm=None,
        fps=None,
        source=filename,
        modality="IMAGE",
    )
