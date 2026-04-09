"""Image preprocessing with quality signal extraction."""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from contextlens.config import (
    BLUR_THRESHOLD,
    BRIGHTNESS_HIGH,
    BRIGHTNESS_LOW,
    CONTRAST_LOW,
    ROTATION_ANGLE_THRESHOLD,
)
from contextlens.schemas import QualitySignals


def load_image(path: str) -> np.ndarray:
    """Load an image as a BGR numpy array (OpenCV format)."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    return img


def compute_blur_score(gray: np.ndarray) -> float:
    """Laplacian variance — higher means sharper."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_brightness(gray: np.ndarray) -> float:
    """Mean pixel intensity normalized to [0, 1]."""
    return float(np.mean(gray) / 255.0)


def compute_contrast(gray: np.ndarray) -> float:
    """Std deviation of pixel intensities normalized to [0, 1]."""
    return float(np.std(gray) / 255.0)


def estimate_rotation_angle(gray: np.ndarray) -> float:
    """Estimate dominant text line rotation angle using Hough lines.

    Returns angle in degrees. 0 means no rotation detected.
    """
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                            minLineLength=gray.shape[1] // 4, maxLineGap=10)
    if lines is None or len(lines) == 0:
        return 0.0

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1
        if abs(dx) < 1:
            continue
        angle = np.degrees(np.arctan2(dy, dx))
        # Only consider near-horizontal lines (text lines)
        if abs(angle) < 45:
            angles.append(angle)

    if not angles:
        return 0.0

    return float(np.median(angles))


def correct_rotation(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate image to correct detected skew."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    return rotated


def analyze_quality(image: np.ndarray) -> QualitySignals:
    """Analyze image quality and return QualitySignals.

    Args:
        image: BGR numpy array (OpenCV format).

    Returns:
        QualitySignals with blur, brightness, contrast, and flags.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur_score = compute_blur_score(gray)
    brightness = compute_brightness(gray)
    contrast = compute_contrast(gray)

    is_blurry = blur_score < BLUR_THRESHOLD
    rotation_angle = estimate_rotation_angle(gray)
    is_rotated = abs(rotation_angle) > ROTATION_ANGLE_THRESHOLD

    # Composite quality score
    quality = 1.0
    if is_blurry:
        quality *= 0.5
    if brightness < BRIGHTNESS_LOW or brightness > BRIGHTNESS_HIGH:
        quality *= 0.7
    if contrast < CONTRAST_LOW:
        quality *= 0.7
    if is_rotated:
        quality *= 0.8

    # Clamp
    estimated_quality = max(0.0, min(1.0, quality))

    return QualitySignals(
        blur_score=blur_score,
        brightness=brightness,
        contrast=contrast,
        estimated_quality=estimated_quality,
        is_blurry=is_blurry,
        is_rotated=is_rotated,
    )


def preprocess_image(path: str) -> tuple[np.ndarray, QualitySignals]:
    """Load image, analyze quality, attempt rotation correction if needed.

    Returns:
        Tuple of (possibly corrected image, quality signals).
    """
    image = load_image(path)
    signals = analyze_quality(image)

    if signals.is_rotated:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        angle = estimate_rotation_angle(gray)
        image = correct_rotation(image, angle)

    return image, signals
