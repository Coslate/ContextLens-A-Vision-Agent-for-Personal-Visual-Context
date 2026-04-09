"""Failure mode handlers — graceful degradation for adversarial inputs.

Each handler inspects quality signals and/or extraction results to detect a
specific failure mode, then applies appropriate confidence penalties and sets
the correct ``FailureFlag``.
"""

from __future__ import annotations

from contextlens.config import (
    BLUR_THRESHOLD,
    OCR_CONFIDENCE_LOW,
    QUALITY_PENALTY_BLURRY,
)
from contextlens.schemas import FailureFlag, QualitySignals


# ---------------------------------------------------------------------------
# Individual failure detectors
# ---------------------------------------------------------------------------

def detect_blurry(
    quality: QualitySignals | None,
) -> tuple[bool, float]:
    """Detect blurry image.

    Returns:
        (is_blurry, confidence_penalty_multiplier)
    """
    if quality is None:
        return False, 1.0
    if quality.is_blurry or quality.blur_score < BLUR_THRESHOLD:
        return True, QUALITY_PENALTY_BLURRY
    return False, 1.0


def detect_rotation(
    quality: QualitySignals | None,
) -> tuple[bool, str | None]:
    """Detect rotation issues.

    Returns:
        (is_rotated, flag_name_or_None)
        The caller decides whether correction was applied upstream; we just
        report the detection.
    """
    if quality is None:
        return False, None
    if quality.is_rotated:
        return True, FailureFlag.ROTATION_UNRESOLVED
    return False, None


def detect_partial_capture(
    field_confidence: dict[str, float],
    expected_key_fields: list[str] | None = None,
) -> bool:
    """Detect partial / cropped capture.

    A partial capture is suspected when key fields have ``None``-equivalent
    confidence (0.0) or are entirely missing from the confidence dict.
    """
    if expected_key_fields is None:
        return False

    missing = 0
    for key in expected_key_fields:
        if key not in field_confidence or field_confidence[key] == 0.0:
            missing += 1

    # If more than half of expected key fields are missing → partial
    return missing > len(expected_key_fields) / 2


def detect_low_ocr(avg_ocr_confidence: float) -> bool:
    """Detect low overall OCR confidence."""
    return avg_ocr_confidence < OCR_CONFIDENCE_LOW


# ---------------------------------------------------------------------------
# Aggregate handler
# ---------------------------------------------------------------------------

# Default expected key fields per image type
_EXPECTED_FIELDS: dict[str, list[str]] = {
    "receipt": ["merchant", "total", "items"],
    "conversation": ["participants", "action_items"],
    "document": ["structured_fields"],
    "whiteboard": ["text_blocks", "bullets"],
}


def apply_failure_handlers(
    field_confidence: dict[str, float],
    quality: QualitySignals | None = None,
    avg_ocr_confidence: float = 1.0,
    image_type: str = "unknown",
) -> tuple[dict[str, float], list[FailureFlag], bool]:
    """Run all failure handlers and return adjusted results.

    Args:
        field_confidence: Per-field confidence dict (will NOT be mutated).
        quality: Image quality signals.
        avg_ocr_confidence: Average OCR confidence across all spans.
        image_type: Predicted image type string (e.g. ``"receipt"``).

    Returns:
        Tuple of:
            - Adjusted confidence dict (new object).
            - List of FailureFlag values detected.
            - needs_clarification bool.
    """
    flags: list[FailureFlag] = []
    needs_clarification = False
    adjusted = dict(field_confidence)

    # --- Blurry ---
    is_blurry, blur_penalty = detect_blurry(quality)
    if is_blurry:
        flags.append(FailureFlag.BLURRY_IMAGE)
        adjusted = {k: v * blur_penalty for k, v in adjusted.items()}
        # Very blurry → needs clarification
        if quality is not None and quality.blur_score < BLUR_THRESHOLD * 0.5:
            needs_clarification = True

    # --- Rotation ---
    is_rotated, rot_flag = detect_rotation(quality)
    if is_rotated and rot_flag is not None:
        flags.append(FailureFlag(rot_flag))

    # --- Partial capture ---
    expected = _EXPECTED_FIELDS.get(image_type)
    if detect_partial_capture(adjusted, expected):
        flags.append(FailureFlag.PARTIAL_CAPTURE)
        needs_clarification = True

    # --- Low OCR ---
    if detect_low_ocr(avg_ocr_confidence):
        flags.append(FailureFlag.OCR_UNCERTAIN)
        # Apply global penalty for uncertain OCR
        low_ocr_penalty = 0.6
        adjusted = {k: v * low_ocr_penalty for k, v in adjusted.items()}
        needs_clarification = True

    # Clamp all values to [0, 1]
    adjusted = {k: max(0.0, min(1.0, v)) for k, v in adjusted.items()}

    return adjusted, flags, needs_clarification
