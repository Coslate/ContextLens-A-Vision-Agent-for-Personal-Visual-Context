"""Tests for OCR engine — PR3."""

import os
import tempfile
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from contextlens.ocr import run_ocr, run_ocr_from_path
from contextlens.schemas import OCRResult, OCRSpan


# --- Helpers ---

def make_text_image(text_lines: list[str], width=400, height=300) -> np.ndarray:
    """Create a synthetic image with text rendered on it."""
    img = np.ones((height, width, 3), dtype=np.uint8) * 255
    y = 40
    for line in text_lines:
        cv2.putText(img, line, (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        y += 35
    return img


def make_blank_image(width=400, height=300) -> np.ndarray:
    """Create a blank white image with no text."""
    return np.ones((height, width, 3), dtype=np.uint8) * 255


def save_temp_image(img: np.ndarray) -> str:
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    cv2.imwrite(path, img)
    return path


# --- Mock-based fast tests ---

MOCK_EASYOCR_RESULTS = [
    ([[10, 10], [200, 10], [200, 30], [10, 30]], "STARBUCKS", 0.95),
    ([[10, 50], [200, 50], [200, 70], [10, 70]], "Latte $5.50", 0.88),
    ([[10, 90], [200, 90], [200, 110], [10, 110]], "TOTAL $5.50", 0.92),
]


class TestRunOCRMocked:
    """Fast tests using mocked EasyOCR reader."""

    @patch("contextlens.ocr._get_reader")
    def test_basic_ocr(self, mock_get_reader):
        reader = MagicMock()
        reader.readtext.return_value = MOCK_EASYOCR_RESULTS
        mock_get_reader.return_value = reader

        img = make_text_image(["STARBUCKS", "Latte $5.50", "TOTAL $5.50"])
        result = run_ocr(img)

        assert isinstance(result, OCRResult)
        assert len(result.spans) == 3
        assert result.spans[0].text == "STARBUCKS"
        assert result.spans[0].confidence == 0.95

    @patch("contextlens.ocr._get_reader")
    def test_raw_text(self, mock_get_reader):
        reader = MagicMock()
        reader.readtext.return_value = MOCK_EASYOCR_RESULTS
        mock_get_reader.return_value = reader

        result = run_ocr(make_text_image(["x"]))
        assert "STARBUCKS" in result.raw_text
        assert "Latte $5.50" in result.raw_text
        assert "TOTAL $5.50" in result.raw_text

    @patch("contextlens.ocr._get_reader")
    def test_avg_confidence(self, mock_get_reader):
        reader = MagicMock()
        reader.readtext.return_value = MOCK_EASYOCR_RESULTS
        mock_get_reader.return_value = reader

        result = run_ocr(make_text_image(["x"]))
        expected_avg = (0.95 + 0.88 + 0.92) / 3
        assert abs(result.avg_confidence - expected_avg) < 0.001

    @patch("contextlens.ocr._get_reader")
    def test_bbox_conversion(self, mock_get_reader):
        reader = MagicMock()
        reader.readtext.return_value = [
            ([[10, 20], [110, 20], [110, 50], [10, 50]], "Test", 0.9),
        ]
        mock_get_reader.return_value = reader

        result = run_ocr(make_text_image(["Test"]))
        bbox = result.spans[0].bbox
        assert bbox[0] == 10.0   # x
        assert bbox[1] == 20.0   # y
        assert bbox[2] == 100.0  # w
        assert bbox[3] == 30.0   # h

    @patch("contextlens.ocr._get_reader")
    def test_empty_image_returns_empty(self, mock_get_reader):
        reader = MagicMock()
        reader.readtext.return_value = []
        mock_get_reader.return_value = reader

        result = run_ocr(make_blank_image())
        assert result.spans == []
        assert result.raw_text == ""
        assert result.avg_confidence == 0.0

    @patch("contextlens.ocr._get_reader")
    def test_single_span(self, mock_get_reader):
        reader = MagicMock()
        reader.readtext.return_value = [
            ([[0, 0], [100, 0], [100, 20], [0, 20]], "Hello", 0.99),
        ]
        mock_get_reader.return_value = reader

        result = run_ocr(make_text_image(["Hello"]))
        assert len(result.spans) == 1
        assert result.avg_confidence == 0.99
        assert result.raw_text == "Hello"


class TestRunOCRFromPathMocked:
    """Test the path-based convenience function."""

    @patch("contextlens.ocr._get_reader")
    def test_from_path(self, mock_get_reader):
        reader = MagicMock()
        reader.readtext.return_value = MOCK_EASYOCR_RESULTS
        mock_get_reader.return_value = reader

        img = make_text_image(["STARBUCKS"])
        path = save_temp_image(img)
        try:
            result = run_ocr_from_path(path)
            assert isinstance(result, OCRResult)
            assert len(result.spans) == 3
        finally:
            os.unlink(path)

    def test_from_nonexistent_path_raises(self):
        with pytest.raises(FileNotFoundError):
            run_ocr_from_path("/no/such/file.png")


# --- Real EasyOCR integration test (slow, requires model download) ---

@pytest.mark.slow
class TestRunOCRReal:
    """Integration tests with real EasyOCR. Run with: pytest -m slow"""

    def test_real_ocr_on_synthetic_image(self):
        """EasyOCR should detect text on a clean synthetic image."""
        import contextlens.ocr as ocr_module
        ocr_module._reader = None  # reset singleton

        img = make_text_image(
            ["STARBUCKS", "Latte $5.50", "Muffin $3.25", "TOTAL $8.75"],
            width=500, height=250,
        )
        result = run_ocr(img)
        assert len(result.spans) > 0
        assert result.avg_confidence > 0.0
        assert result.raw_text != ""

    def test_real_ocr_blank_image(self):
        """Blank image should return empty or near-empty result."""
        result = run_ocr(make_blank_image())
        # EasyOCR may or may not find spurious text on blank
        assert isinstance(result, OCRResult)
