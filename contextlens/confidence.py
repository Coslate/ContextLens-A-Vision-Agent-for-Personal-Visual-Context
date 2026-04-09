"""Confidence calibration engine.

Combines OCR span confidence, image quality signals, and extraction evidence
strength into calibrated per-field confidence scores.  The goal is
*well-calibrated* scores: high confidence ↔ high accuracy.

General formula per field:
    calibrated = base_confidence × quality_penalty × evidence_multiplier
"""

from __future__ import annotations

from contextlens.config import (
    QUALITY_PENALTY_BLURRY,
    QUALITY_PENALTY_LOW_BRIGHTNESS,
    QUALITY_PENALTY_LOW_CONTRAST,
    QUALITY_PENALTY_ROTATED,
    BRIGHTNESS_LOW,
    CONTRAST_LOW,
)
from contextlens.schemas import QualitySignals


def compute_quality_penalty(quality: QualitySignals | None) -> float:
    """Return a multiplicative quality penalty in (0, 1].

    Combines penalties for blur, low brightness, low contrast, and rotation.
    If *quality* is ``None`` (unknown), returns a conservative default of 0.85.
    """
    if quality is None:
        return 0.85

    penalty = 1.0

    if quality.is_blurry:
        penalty *= QUALITY_PENALTY_BLURRY

    if quality.brightness < BRIGHTNESS_LOW:
        penalty *= QUALITY_PENALTY_LOW_BRIGHTNESS

    if quality.contrast < CONTRAST_LOW:
        penalty *= QUALITY_PENALTY_LOW_CONTRAST

    if quality.is_rotated:
        penalty *= QUALITY_PENALTY_ROTATED

    return penalty


def calibrate_field_confidence(
    raw_confidence: float,
    evidence_multiplier: float = 1.0,
    quality: QualitySignals | None = None,
) -> float:
    """Calibrate a single field's confidence score.

    Args:
        raw_confidence: Base confidence from the extractor (typically derived
            from OCR span confidence).
        evidence_multiplier: Strength of extraction evidence (1.0 = strong
            keyword match, down to ~0.3 for guessed/missing fields).
        quality: Image quality signals from the preprocessor.

    Returns:
        Calibrated confidence clamped to [0.0, 1.0].
    """
    penalty = compute_quality_penalty(quality)
    calibrated = raw_confidence * penalty * evidence_multiplier
    return max(0.0, min(1.0, calibrated))


def calibrate_confidence_dict(
    raw_confidences: dict[str, float],
    evidence: dict[str, float] | None = None,
    quality: QualitySignals | None = None,
) -> dict[str, float]:
    """Calibrate an entire confidence dict from an extractor.

    Args:
        raw_confidences: Per-field raw confidence from the extractor.
        evidence: Per-field evidence multipliers.  Missing keys default to 1.0.
        quality: Image quality signals.

    Returns:
        Dict with the same keys, values calibrated and clamped to [0, 1].
    """
    evidence = evidence or {}
    calibrated: dict[str, float] = {}
    for field, raw in raw_confidences.items():
        ev = evidence.get(field, 1.0)
        calibrated[field] = calibrate_field_confidence(raw, ev, quality)
    return calibrated
