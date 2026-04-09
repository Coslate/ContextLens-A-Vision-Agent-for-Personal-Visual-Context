"""Tests for confidence calibration engine — PR6."""

import pytest

from contextlens.confidence import (
    calibrate_confidence_dict,
    calibrate_field_confidence,
    compute_quality_penalty,
)
from contextlens.schemas import QualitySignals


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


def _rotated_quality() -> QualitySignals:
    return QualitySignals(
        blur_score=500.0,
        brightness=0.6,
        contrast=0.4,
        estimated_quality=0.6,
        is_blurry=False,
        is_rotated=True,
    )


def _dark_quality() -> QualitySignals:
    return QualitySignals(
        blur_score=600.0,
        brightness=0.1,  # below BRIGHTNESS_LOW (0.2)
        contrast=0.1,    # below CONTRAST_LOW (0.15)
        estimated_quality=0.4,
        is_blurry=False,
        is_rotated=False,
    )


def _worst_quality() -> QualitySignals:
    return QualitySignals(
        blur_score=10.0,
        brightness=0.05,
        contrast=0.05,
        estimated_quality=0.1,
        is_blurry=True,
        is_rotated=True,
    )


# =====================================================================
# Quality Penalty
# =====================================================================

class TestQualityPenalty:
    def test_none_quality_returns_default(self):
        assert compute_quality_penalty(None) == pytest.approx(0.85)

    def test_good_quality_no_penalty(self):
        assert compute_quality_penalty(_good_quality()) == pytest.approx(1.0)

    def test_blurry_penalty(self):
        penalty = compute_quality_penalty(_blurry_quality())
        assert penalty == pytest.approx(0.5)

    def test_rotated_penalty(self):
        penalty = compute_quality_penalty(_rotated_quality())
        assert penalty == pytest.approx(0.7)

    def test_dark_low_contrast_penalty(self):
        penalty = compute_quality_penalty(_dark_quality())
        # 0.8 (brightness) * 0.8 (contrast) = 0.64
        assert penalty == pytest.approx(0.64)

    def test_worst_case_stacks_all_penalties(self):
        penalty = compute_quality_penalty(_worst_quality())
        # blurry(0.5) * brightness(0.8) * contrast(0.8) * rotated(0.7)
        expected = 0.5 * 0.8 * 0.8 * 0.7
        assert penalty == pytest.approx(expected)

    def test_penalty_always_positive(self):
        penalty = compute_quality_penalty(_worst_quality())
        assert penalty > 0.0


# =====================================================================
# Single Field Calibration
# =====================================================================

class TestCalibrateField:
    def test_perfect_conditions(self):
        result = calibrate_field_confidence(0.9, 1.0, _good_quality())
        assert result == pytest.approx(0.9)

    def test_blurry_reduces_confidence(self):
        good = calibrate_field_confidence(0.9, 1.0, _good_quality())
        blurry = calibrate_field_confidence(0.9, 1.0, _blurry_quality())
        assert blurry < good

    def test_weak_evidence_reduces_confidence(self):
        strong = calibrate_field_confidence(0.9, 1.0, _good_quality())
        weak = calibrate_field_confidence(0.9, 0.3, _good_quality())
        assert weak < strong

    def test_zero_raw_stays_zero(self):
        result = calibrate_field_confidence(0.0, 1.0, _good_quality())
        assert result == 0.0

    def test_clamped_to_unit(self):
        # Even with raw > 1 (shouldn't happen but be safe)
        result = calibrate_field_confidence(1.5, 1.0, _good_quality())
        assert result <= 1.0

    def test_no_quality_uses_default_penalty(self):
        result = calibrate_field_confidence(1.0, 1.0, None)
        assert result == pytest.approx(0.85)

    def test_combined_blurry_weak(self):
        result = calibrate_field_confidence(0.9, 0.3, _blurry_quality())
        # 0.9 * 0.5 * 0.3 = 0.135
        assert result == pytest.approx(0.135)


# =====================================================================
# Dict Calibration
# =====================================================================

class TestCalibrateDictConfidence:
    def test_basic_calibration(self):
        raw = {"merchant": 0.9, "total": 0.85, "items": 0.7}
        result = calibrate_confidence_dict(raw, quality=_good_quality())
        # No penalties → same values
        assert result["merchant"] == pytest.approx(0.9)
        assert result["total"] == pytest.approx(0.85)
        assert result["items"] == pytest.approx(0.7)

    def test_with_evidence(self):
        raw = {"merchant": 0.9, "total": 0.9}
        evidence = {"merchant": 1.0, "total": 0.3}
        result = calibrate_confidence_dict(raw, evidence, _good_quality())
        assert result["merchant"] > result["total"]

    def test_blurry_reduces_all(self):
        raw = {"a": 0.8, "b": 0.7}
        good = calibrate_confidence_dict(raw, quality=_good_quality())
        blurry = calibrate_confidence_dict(raw, quality=_blurry_quality())
        assert blurry["a"] < good["a"]
        assert blurry["b"] < good["b"]

    def test_missing_evidence_defaults_to_one(self):
        raw = {"field_a": 0.8}
        evidence = {}  # no keys
        result = calibrate_confidence_dict(raw, evidence, _good_quality())
        assert result["field_a"] == pytest.approx(0.8)

    def test_empty_dict(self):
        result = calibrate_confidence_dict({}, quality=_good_quality())
        assert result == {}

    def test_all_values_in_unit(self):
        raw = {"a": 0.9, "b": 0.5, "c": 0.1}
        result = calibrate_confidence_dict(raw, quality=_worst_quality())
        for v in result.values():
            assert 0.0 <= v <= 1.0

    def test_preserves_keys(self):
        raw = {"merchant": 0.9, "total": 0.8, "date": 0.7}
        result = calibrate_confidence_dict(raw, quality=_good_quality())
        assert set(result.keys()) == set(raw.keys())
