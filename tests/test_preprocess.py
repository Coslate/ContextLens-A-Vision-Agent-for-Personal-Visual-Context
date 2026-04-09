"""Tests for image preprocessing — PR2."""

import tempfile
import os

import cv2
import numpy as np
import pytest
from PIL import Image

from contextlens.preprocess import (
    analyze_quality,
    compute_blur_score,
    compute_brightness,
    compute_contrast,
    correct_rotation,
    estimate_rotation_angle,
    load_image,
    preprocess_image,
)
from contextlens.schemas import QualitySignals
from contextlens.config import BLUR_THRESHOLD


# --- Helpers ---

def make_sharp_image(width=400, height=300) -> np.ndarray:
    """Create a sharp synthetic image with text-like edges."""
    img = np.ones((height, width, 3), dtype=np.uint8) * 240
    # Draw sharp black rectangles and lines to simulate text
    for y in range(50, 250, 30):
        cv2.putText(img, "STARBUCKS $5.50", (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return img


def make_blurry_image(width=400, height=300) -> np.ndarray:
    """Create a blurry image by applying heavy Gaussian blur."""
    img = make_sharp_image(width, height)
    return cv2.GaussianBlur(img, (31, 31), 15)


def make_dark_image(width=400, height=300) -> np.ndarray:
    """Create a very dark image."""
    img = np.ones((height, width, 3), dtype=np.uint8) * 20
    cv2.putText(img, "Dark text", (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (40, 40, 40), 1)
    return img


def make_low_contrast_image(width=400, height=300) -> np.ndarray:
    """Create a low contrast image (all pixels near mid-gray)."""
    img = np.ones((height, width, 3), dtype=np.uint8) * 128
    cv2.putText(img, "Low contrast", (20, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (130, 130, 130), 1)
    return img


def make_rotated_image(angle=25, width=400, height=300) -> np.ndarray:
    """Create an image rotated by a given angle."""
    img = make_sharp_image(width, height)
    center = (width // 2, height // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (width, height),
                          borderMode=cv2.BORDER_REPLICATE)


def save_temp_image(img: np.ndarray) -> str:
    """Save image to a temp file and return the path."""
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    cv2.imwrite(path, img)
    return path


# --- Tests ---

class TestLoadImage:
    def test_load_valid_image(self):
        img = make_sharp_image()
        path = save_temp_image(img)
        try:
            loaded = load_image(path)
            assert loaded.shape == img.shape
            assert loaded.dtype == np.uint8
        finally:
            os.unlink(path)

    def test_load_nonexistent_raises(self):
        with pytest.raises(FileNotFoundError):
            load_image("/nonexistent/path/image.png")


class TestComputeBlurScore:
    def test_sharp_image_high_score(self):
        img = make_sharp_image()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        score = compute_blur_score(gray)
        assert score > BLUR_THRESHOLD

    def test_blurry_image_low_score(self):
        img = make_blurry_image()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        score = compute_blur_score(gray)
        assert score < BLUR_THRESHOLD

    def test_sharp_higher_than_blurry(self):
        sharp = make_sharp_image()
        blurry = make_blurry_image()
        sharp_score = compute_blur_score(cv2.cvtColor(sharp, cv2.COLOR_BGR2GRAY))
        blurry_score = compute_blur_score(cv2.cvtColor(blurry, cv2.COLOR_BGR2GRAY))
        assert sharp_score > blurry_score


class TestComputeBrightness:
    def test_bright_image(self):
        img = make_sharp_image()  # mostly white (240)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness = compute_brightness(gray)
        assert 0.5 < brightness <= 1.0

    def test_dark_image(self):
        img = make_dark_image()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness = compute_brightness(gray)
        assert 0.0 <= brightness < 0.2

    def test_range(self):
        gray = np.zeros((100, 100), dtype=np.uint8)
        assert compute_brightness(gray) == 0.0
        gray_white = np.ones((100, 100), dtype=np.uint8) * 255
        assert compute_brightness(gray_white) == 1.0


class TestComputeContrast:
    def test_low_contrast(self):
        img = make_low_contrast_image()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        contrast = compute_contrast(gray)
        assert contrast < 0.05

    def test_uniform_zero_contrast(self):
        gray = np.ones((100, 100), dtype=np.uint8) * 128
        assert compute_contrast(gray) == 0.0

    def test_normal_contrast(self):
        img = make_sharp_image()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        contrast = compute_contrast(gray)
        assert contrast > 0.0


class TestEstimateRotationAngle:
    def test_no_rotation(self):
        img = make_sharp_image()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        angle = estimate_rotation_angle(gray)
        assert abs(angle) < 5  # near zero

    def test_blank_image(self):
        gray = np.ones((300, 400), dtype=np.uint8) * 200
        angle = estimate_rotation_angle(gray)
        assert angle == 0.0  # no edges → no lines detected


class TestCorrectRotation:
    def test_output_same_shape(self):
        img = make_sharp_image()
        corrected = correct_rotation(img, 10.0)
        assert corrected.shape == img.shape

    def test_zero_rotation_noop(self):
        img = make_sharp_image()
        corrected = correct_rotation(img, 0.0)
        # With 0 angle, result should be very close to original
        diff = np.abs(img.astype(float) - corrected.astype(float)).mean()
        assert diff < 1.0


class TestAnalyzeQuality:
    def test_sharp_image(self):
        img = make_sharp_image()
        signals = analyze_quality(img)
        assert isinstance(signals, QualitySignals)
        assert signals.is_blurry is False
        assert signals.blur_score > BLUR_THRESHOLD
        assert signals.estimated_quality > 0.5

    def test_blurry_image(self):
        img = make_blurry_image()
        signals = analyze_quality(img)
        assert signals.is_blurry is True
        assert signals.blur_score < BLUR_THRESHOLD
        assert signals.estimated_quality < 0.6

    def test_dark_image(self):
        img = make_dark_image()
        signals = analyze_quality(img)
        assert signals.brightness < 0.2
        # Quality penalized for low brightness
        assert signals.estimated_quality < 1.0

    def test_low_contrast_image(self):
        img = make_low_contrast_image()
        signals = analyze_quality(img)
        assert signals.contrast < 0.05

    def test_quality_range(self):
        for make_fn in [make_sharp_image, make_blurry_image, make_dark_image]:
            img = make_fn()
            signals = analyze_quality(img)
            assert 0.0 <= signals.estimated_quality <= 1.0
            assert 0.0 <= signals.brightness <= 1.0
            assert 0.0 <= signals.contrast <= 1.0

    def test_blurry_quality_lower_than_sharp(self):
        sharp_signals = analyze_quality(make_sharp_image())
        blurry_signals = analyze_quality(make_blurry_image())
        assert sharp_signals.estimated_quality > blurry_signals.estimated_quality


class TestPreprocessImage:
    def test_end_to_end_sharp(self):
        img = make_sharp_image()
        path = save_temp_image(img)
        try:
            result_img, signals = preprocess_image(path)
            assert result_img.shape == img.shape
            assert isinstance(signals, QualitySignals)
            assert signals.is_blurry is False
        finally:
            os.unlink(path)

    def test_end_to_end_blurry(self):
        img = make_blurry_image()
        path = save_temp_image(img)
        try:
            result_img, signals = preprocess_image(path)
            assert signals.is_blurry is True
            assert signals.estimated_quality < 0.6
        finally:
            os.unlink(path)

    def test_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            preprocess_image("/no/such/file.png")
