"""Ultrasound-specific preprocessing: despeckle, contrast enhancement, ROI crop."""

from __future__ import annotations

import numpy as np


def despeckle(frame: np.ndarray, method: str = "bilateral") -> np.ndarray:
    """Reduce speckle noise while preserving vessel edges.

    Parameters
    ----------
    method:
        ``"bilateral"``  — edge-preserving, best quality (slower).
        ``"median"``     — fast, good for heavy speckle.
        ``"nlm"``        — non-local means, highest quality (slowest).
    """
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("opencv-python-headless required for despeckle") from exc

    if method == "bilateral":
        return cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)
    if method == "median":
        return cv2.medianBlur(frame, ksize=5)
    if method == "nlm":
        return cv2.fastNlMeansDenoising(frame, h=10, templateWindowSize=7, searchWindowSize=21)
    raise ValueError(f"Unknown despeckle method: {method!r}. Choose bilateral/median/nlm")


def enhance_contrast(frame: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalisation).

    Improves visibility of vessel walls without over-amplifying noise.
    """
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("opencv-python-headless required for CLAHE") from exc

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    return clahe.apply(frame)


def crop_roi(
    frame: np.ndarray,
    roi: tuple[int, int, int, int],
) -> tuple[np.ndarray, tuple[int, int]]:
    """Crop to region-of-interest.

    Parameters
    ----------
    roi:
        ``(x1, y1, x2, y2)`` in pixel coordinates (column-first, image convention).

    Returns
    -------
    cropped:
        Cropped frame array.
    offset:
        ``(col_offset, row_offset)`` for mapping measurements back to full frame.
    """
    x1, y1, x2, y2 = roi
    x1, x2 = sorted((max(0, x1), min(frame.shape[1], x2)))
    y1, y2 = sorted((max(0, y1), min(frame.shape[0], y2)))
    return frame[y1:y2, x1:x2], (x1, y1)


def preprocess_frame(
    frame: np.ndarray,
    roi: tuple[int, int, int, int] | None = None,
    despeckle_method: str = "bilateral",
    clip_limit: float = 2.0,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Full preprocessing pipeline for one B-mode frame.

    Returns
    -------
    processed:
        Preprocessed (and optionally cropped) uint8 grayscale array.
    offset:
        ``(col_offset, row_offset)`` — (0, 0) if no ROI crop was applied.
    """
    if roi is not None:
        frame, offset = crop_roi(frame, roi)
    else:
        offset = (0, 0)

    frame = despeckle(frame, method=despeckle_method)
    frame = enhance_contrast(frame, clip_limit=clip_limit)
    return frame, offset


def detect_duplex_split(frame: np.ndarray) -> int | None:
    """Estimate the row index separating B-mode from Doppler spectrum.

    Many duplex ultrasound exports stack B-mode (upper) and Doppler spectrum
    (lower) in a single image separated by a thin black or white band.

    Returns the split row index, or ``None`` if no split is detected.
    """
    # Look for a horizontal band with very low or very high mean intensity
    # compared to its neighbours — typically the separator line.
    h = frame.shape[0]
    row_means = frame.mean(axis=1).astype(float)

    # Smooth to avoid noise
    kernel = np.ones(5) / 5
    smoothed = np.convolve(row_means, kernel, mode="same")

    # Search in the middle third of the image
    search_start = h // 4
    search_end = 3 * h // 4
    region = smoothed[search_start:search_end]

    # Find row with minimum mean (black separator band)
    local_min_idx = int(np.argmin(region)) + search_start
    local_min_val = smoothed[local_min_idx]
    global_mean = smoothed.mean()

    if local_min_val < global_mean * 0.3:
        return local_min_idx

    return None
