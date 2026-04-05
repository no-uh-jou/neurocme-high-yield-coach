"""Vessel wall segmentation and edge detection for B-mode ultrasound.

Strategy
--------
1. Anechoic (dark) lumen detection via adaptive thresholding.
2. Morphological cleanup to get clean lumen mask.
3. Contour extraction from lumen boundary (= vessel wall inner edge).
4. Optional active-contour (snake) refinement for precise wall tracing.

This approach exploits the fundamental physics of B-mode ultrasound: vessel
lumens appear anechoic (dark) against echogenic surrounding tissue.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SegmentResult:
    """Output of one vessel segmentation run."""

    mask: np.ndarray                 # binary uint8 mask (H, W), lumen = 255
    contour: np.ndarray              # (N, 2) float array of (row, col) boundary points
    centroid: tuple[float, float]    # (row, col) centroid of lumen
    bounding_box: tuple[int, int, int, int]  # (r_min, c_min, r_max, c_max)
    area_px2: float
    confidence: float = 0.0          # 0–1 quality estimate
    refined: bool = False            # True if active-contour refinement was applied
    all_contours: list[np.ndarray] = field(default_factory=list)  # for multi-vessel


def detect_vessel(
    frame: np.ndarray,
    min_area_px: int = 200,
    max_area_fraction: float = 0.5,
    refine: bool = False,
    refine_iterations: int = 100,
) -> SegmentResult:
    """Detect the primary vessel lumen in a preprocessed B-mode frame.

    Parameters
    ----------
    frame:
        Preprocessed uint8 grayscale image (H, W).
    min_area_px:
        Minimum lumen area in pixels — filters out noise blobs.
    max_area_fraction:
        Reject candidates occupying more than this fraction of total image
        area (catches background inversions).
    refine:
        If ``True``, run active-contour refinement on the initial mask.
        Slower but more accurate on clean images.
    refine_iterations:
        Active-contour iterations (ignored if ``refine=False``).

    Returns
    -------
    :class:`SegmentResult`
    """
    try:
        import cv2
    except ImportError as exc:
        raise ImportError("opencv-python-headless required for segmentation") from exc

    h, w = frame.shape[:2]
    total_px = h * w

    # ------------------------------------------------------------------
    # Step 1: Adaptive threshold to find dark (anechoic) regions
    # ------------------------------------------------------------------
    # Invert so that dark lumen becomes foreground
    inverted = cv2.bitwise_not(frame)

    # Adaptive threshold (Gaussian-weighted neighbourhood)
    thresh = cv2.adaptiveThreshold(
        inverted,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY,
        blockSize=31,
        C=-10,
    )

    # ------------------------------------------------------------------
    # Step 2: Morphological cleanup
    # ------------------------------------------------------------------
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open, iterations=1)

    # Fill holes inside contours
    filled = _fill_holes(opened)

    # ------------------------------------------------------------------
    # Step 3: Connected component analysis — pick the best lumen candidate
    # ------------------------------------------------------------------
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        filled, connectivity=8
    )

    best_label = -1
    best_score = -1.0
    all_valid: list[int] = []

    for label in range(1, num_labels):  # skip background (0)
        area = stats[label, cv2.CC_STAT_AREA]
        if area < min_area_px:
            continue
        if area > total_px * max_area_fraction:
            continue

        # Prefer roughly circular blobs near the image centre
        cx, cy = centroids[label]
        dist_to_centre = np.hypot(cx - w / 2, cy - h / 2) / max(w, h)
        circularity = _circularity_from_stats(labels == label)

        # Score: favour area, circularity; penalise distance from centre
        score = circularity * np.log1p(area) * (1.0 - dist_to_centre)
        all_valid.append(label)

        if score > best_score:
            best_score = score
            best_label = label

    if best_label == -1:
        raise ValueError(
            "No vessel lumen detected. Try adjusting the ROI or preprocessing settings."
        )

    # Build binary mask for best candidate
    mask = np.where(labels == best_label, 255, 0).astype(np.uint8)

    # ------------------------------------------------------------------
    # Step 4: Extract contour
    # ------------------------------------------------------------------
    contours_cv, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    primary_contour_cv = max(contours_cv, key=cv2.contourArea)
    contour = primary_contour_cv[:, 0, ::-1].astype(float)  # (N, 2) as (row, col)

    # Build all-vessel contours for multi-vessel display
    all_contours: list[np.ndarray] = []
    for lbl in all_valid:
        m = np.where(labels == lbl, 255, 0).astype(np.uint8)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if cs:
            c = max(cs, key=cv2.contourArea)
            all_contours.append(c[:, 0, ::-1].astype(float))

    cx, cy = centroids[best_label]
    area_px2 = float(stats[best_label, cv2.CC_STAT_AREA])

    # Bounding box in (r_min, c_min, r_max, c_max)
    x0 = stats[best_label, cv2.CC_STAT_LEFT]
    y0 = stats[best_label, cv2.CC_STAT_TOP]
    bw = stats[best_label, cv2.CC_STAT_WIDTH]
    bh = stats[best_label, cv2.CC_STAT_HEIGHT]
    bounding_box = (y0, x0, y0 + bh, x0 + bw)

    circ = _circularity_from_stats(mask > 0)
    confidence = min(1.0, circ * min(1.0, area_px2 / 1000.0))

    result = SegmentResult(
        mask=mask,
        contour=contour,
        centroid=(cy, cx),  # (row, col)
        bounding_box=bounding_box,
        area_px2=area_px2,
        confidence=confidence,
        refined=False,
        all_contours=all_contours,
    )

    # ------------------------------------------------------------------
    # Step 5 (optional): Active-contour refinement
    # ------------------------------------------------------------------
    if refine:
        result = _refine_with_active_contour(frame, result, iterations=refine_iterations)

    return result


# ---------------------------------------------------------------------------
# Active-contour refinement
# ---------------------------------------------------------------------------

def _refine_with_active_contour(
    frame: np.ndarray,
    result: SegmentResult,
    iterations: int = 100,
) -> SegmentResult:
    """Refine vessel contour using scikit-image active contours (snakes)."""
    try:
        from skimage.segmentation import active_contour
        from skimage.filters import gaussian
    except ImportError:
        # scikit-image not available — return unrefined result
        return result

    # Prepare edge image: Gaussian-smoothed normalised frame
    img_float = frame.astype(float) / 255.0
    img_smooth = gaussian(img_float, sigma=2)

    # Use existing contour as initialisation
    init_snake = result.contour  # (N, 2) as (row, col)

    # Subsample to reduce computation (active_contour is O(N))
    step = max(1, len(init_snake) // 200)
    init_snake = init_snake[::step]

    try:
        snake = active_contour(
            img_smooth,
            init_snake,
            alpha=0.01,   # tension (smoothness)
            beta=0.1,     # rigidity
            gamma=0.001,  # step size
            max_num_iter=iterations,
            boundary_condition="periodic",
        )
    except Exception:
        return result

    # Rebuild mask from refined snake using fillPoly
    try:
        import cv2
        pts = np.round(snake[:, ::-1]).astype(np.int32)  # (row,col) → (col,row) = (x,y)
        refined_mask = np.zeros_like(result.mask)
        cv2.fillPoly(refined_mask, [pts], color=255)

        # Recompute stats from refined mask
        M = cv2.moments(refined_mask)
        if M["m00"] > 0:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
        else:
            cy, cx = result.centroid

        area_px2 = float((refined_mask > 0).sum())
        cs, _ = cv2.findContours(refined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contour = max(cs, key=cv2.contourArea)[:, 0, ::-1].astype(float) if cs else snake

        return SegmentResult(
            mask=refined_mask,
            contour=contour,
            centroid=(cy, cx),
            bounding_box=result.bounding_box,
            area_px2=area_px2,
            confidence=result.confidence,
            refined=True,
            all_contours=result.all_contours,
        )
    except Exception:
        return result


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill interior holes in a binary mask using floodfill from the border."""
    try:
        import cv2
    except ImportError:
        return mask

    flood = mask.copy()
    h, w = mask.shape
    seed_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, seed_mask, (0, 0), 255)
    inverted_flood = cv2.bitwise_not(flood)
    return cv2.bitwise_or(mask, inverted_flood)


def _circularity_from_stats(binary_mask: np.ndarray) -> float:
    """Compute circularity 4π·A/P² from a binary mask."""
    try:
        import cv2
        mask_u8 = binary_mask.astype(np.uint8) * 255
        cs, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cs:
            return 0.0
        c = max(cs, key=cv2.contourArea)
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, closed=True)
        if perimeter == 0:
            return 0.0
        return min(1.0, (4 * np.pi * area) / (perimeter ** 2))
    except Exception:
        return 0.0
