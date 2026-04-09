"""Tests for failure mode handlers — PR6."""

import pytest

from contextlens.failure_handlers import (
    apply_failure_handlers,
    detect_blurry,
    detect_low_ocr,
    detect_partial_capture,
    detect_rotation,
)
from contextlens.schemas import FailureFlag, QualitySignals


# --- Helpers ---

def _good_quality() -> QualitySignals:
    return QualitySignals(
        blur_score=800.0,
        brightness=0.7,
        contrast=0.5,
        estimated_quality=0.9,
        is_blurry=False,
        is_rotated=False,
    )


def _blurry_quality() -> QualitySignals:
    return QualitySignals(
        blur_score=30.0,
        brightness=0.6,
        contrast=0.4,
        estimated_quality=0.3,
        is_blurry=True,
        is_rotated=False,
    )


def _very_blurry_quality() -> QualitySignals:
    """blur_score < BLUR_THRESHOLD * 0.5 → needs_clarification."""
    return QualitySignals(
        blur_score=20.0,
        brightness=0.6,
        contrast=0.4,
        estimated_quality=0.15,
        is_blurry=True,
        is_rotated=False,
    )


def _rotated_quality() -> QualitySignals:
    return QualitySignals(
        blur_score=500.0,
        brightness=0.6,
        contrast=0.4,
        estimated_quality=0.6,
        is_blurry=False,
        is_rotated=True,
    )


# =====================================================================
# detect_blurry
# =====================================================================

class TestDetectBlurry:
    def test_good_image_not_blurry(self):
        is_blurry, penalty = detect_blurry(_good_quality())
        assert is_blurry is False
        assert penalty == 1.0

    def test_blurry_image_detected(self):
        is_blurry, penalty = detect_blurry(_blurry_quality())
        assert is_blurry is True
        assert penalty < 1.0

    def test_none_quality_not_blurry(self):
        is_blurry, penalty = detect_blurry(None)
        assert is_blurry is False
        assert penalty == 1.0

    def test_low_blur_score_even_if_flag_false(self):
        q = QualitySignals(
            blur_score=50.0,  # below BLUR_THRESHOLD (100)
            brightness=0.7,
            contrast=0.5,
            estimated_quality=0.5,
            is_blurry=False,
        )
        is_blurry, penalty = detect_blurry(q)
        assert is_blurry is True


# =====================================================================
# detect_rotation
# =====================================================================

class TestDetectRotation:
    def test_good_image_not_rotated(self):
        is_rotated, flag = detect_rotation(_good_quality())
        assert is_rotated is False
        assert flag is None

    def test_rotated_image_detected(self):
        is_rotated, flag = detect_rotation(_rotated_quality())
        assert is_rotated is True
        assert flag == FailureFlag.ROTATION_UNRESOLVED

    def test_none_quality(self):
        is_rotated, flag = detect_rotation(None)
        assert is_rotated is False
        assert flag is None


# =====================================================================
# detect_partial_capture
# =====================================================================

class TestDetectPartialCapture:
    def test_all_fields_present(self):
        conf = {"merchant": 0.9, "total": 0.8, "items": 0.7}
        assert detect_partial_capture(conf, ["merchant", "total", "items"]) is False

    def test_most_fields_missing(self):
        conf = {"merchant": 0.9, "total": 0.0, "items": 0.0}
        assert detect_partial_capture(conf, ["merchant", "total", "items"]) is True

    def test_one_missing_out_of_three(self):
        conf = {"merchant": 0.9, "total": 0.8}  # items missing
        # 1 out of 3 missing → not partial (≤ half)
        assert detect_partial_capture(conf, ["merchant", "total", "items"]) is False

    def test_no_expected_fields(self):
        conf = {"a": 0.5}
        assert detect_partial_capture(conf, None) is False

    def test_empty_confidence(self):
        assert detect_partial_capture({}, ["a", "b"]) is True

    def test_all_zero_confidence(self):
        conf = {"a": 0.0, "b": 0.0}
        assert detect_partial_capture(conf, ["a", "b"]) is True


# =====================================================================
# detect_low_ocr
# =====================================================================

class TestDetectLowOCR:
    def test_good_ocr(self):
        assert detect_low_ocr(0.8) is False

    def test_low_ocr(self):
        assert detect_low_ocr(0.2) is True

    def test_threshold_boundary(self):
        # OCR_CONFIDENCE_LOW = 0.4
        assert detect_low_ocr(0.4) is False
        assert detect_low_ocr(0.39) is True


# =====================================================================
# apply_failure_handlers (aggregate)
# =====================================================================

class TestApplyFailureHandlers:
    def test_clean_image_no_flags(self):
        conf = {"merchant": 0.9, "total": 0.85, "items": 0.7}
        adjusted, flags, needs_clar = apply_failure_handlers(
            conf, _good_quality(), avg_ocr_confidence=0.9, image_type="receipt"
        )
        assert flags == []
        assert needs_clar is False
        assert adjusted == conf

    def test_blurry_adds_flag_and_penalty(self):
        conf = {"merchant": 0.9, "total": 0.8}
        adjusted, flags, _ = apply_failure_handlers(
            conf, _blurry_quality(), avg_ocr_confidence=0.8
        )
        assert FailureFlag.BLURRY_IMAGE in flags
        assert adjusted["merchant"] < conf["merchant"]
        assert adjusted["total"] < conf["total"]

    def test_very_blurry_needs_clarification(self):
        conf = {"merchant": 0.9, "total": 0.8}
        _, flags, needs_clar = apply_failure_handlers(
            conf, _very_blurry_quality(), avg_ocr_confidence=0.8
        )
        assert FailureFlag.BLURRY_IMAGE in flags
        assert needs_clar is True

    def test_rotated_adds_flag(self):
        conf = {"merchant": 0.9}
        _, flags, _ = apply_failure_handlers(
            conf, _rotated_quality(), avg_ocr_confidence=0.8
        )
        assert FailureFlag.ROTATION_UNRESOLVED in flags

    def test_partial_capture_detected_for_receipt(self):
        conf = {"merchant": 0.9, "total": 0.0, "items": 0.0}
        _, flags, needs_clar = apply_failure_handlers(
            conf, _good_quality(), avg_ocr_confidence=0.8, image_type="receipt"
        )
        assert FailureFlag.PARTIAL_CAPTURE in flags
        assert needs_clar is True

    def test_low_ocr_adds_flag_and_penalty(self):
        conf = {"a": 0.8, "b": 0.7}
        adjusted, flags, needs_clar = apply_failure_handlers(
            conf, _good_quality(), avg_ocr_confidence=0.2
        )
        assert FailureFlag.OCR_UNCERTAIN in flags
        assert adjusted["a"] < conf["a"]
        assert needs_clar is True

    def test_multiple_flags_stack(self):
        conf = {"merchant": 0.9, "total": 0.0, "items": 0.0}
        adjusted, flags, needs_clar = apply_failure_handlers(
            conf,
            _blurry_quality(),
            avg_ocr_confidence=0.2,
            image_type="receipt",
        )
        assert FailureFlag.BLURRY_IMAGE in flags
        assert FailureFlag.OCR_UNCERTAIN in flags
        assert FailureFlag.PARTIAL_CAPTURE in flags
        assert needs_clar is True
        # Penalties stacked: blur(0.5) then low_ocr(0.6)
        assert adjusted["merchant"] < 0.9 * 0.5

    def test_does_not_mutate_input(self):
        conf = {"a": 0.8, "b": 0.7}
        original = dict(conf)
        apply_failure_handlers(conf, _blurry_quality(), avg_ocr_confidence=0.2)
        assert conf == original

    def test_all_values_clamped(self):
        conf = {"a": 0.9, "b": 0.5}
        adjusted, _, _ = apply_failure_handlers(
            conf, _very_blurry_quality(), avg_ocr_confidence=0.1
        )
        for v in adjusted.values():
            assert 0.0 <= v <= 1.0

    def test_unknown_type_no_partial_flag(self):
        conf = {"a": 0.0}
        _, flags, _ = apply_failure_handlers(
            conf, _good_quality(), avg_ocr_confidence=0.8, image_type="unknown"
        )
        assert FailureFlag.PARTIAL_CAPTURE not in flags

    def test_conversation_partial_capture(self):
        conf = {"participants": 0.0, "action_items": 0.0, "key_topics": 0.5}
        _, flags, _ = apply_failure_handlers(
            conf, _good_quality(), avg_ocr_confidence=0.8, image_type="conversation"
        )
        assert FailureFlag.PARTIAL_CAPTURE in flags
