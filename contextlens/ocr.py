"""EasyOCR wrapper with structured output."""

from __future__ import annotations

import numpy as np

from contextlens.config import OCR_LANGUAGES
from contextlens.schemas import OCRResult, OCRSpan

# Lazy-loaded singleton to avoid repeated model loading
_reader = None


def _get_reader(languages: list[str] | None = None):
    """Get or create EasyOCR reader (singleton)."""
    global _reader
    if _reader is None:
        import easyocr
        import torch
        use_gpu = torch.cuda.is_available()
        _reader = easyocr.Reader(languages or OCR_LANGUAGES, gpu=use_gpu)
    return _reader


def run_ocr(image: np.ndarray, languages: list[str] | None = None) -> OCRResult:
    """Run EasyOCR on a BGR numpy array and return structured OCRResult.

    Args:
        image: BGR numpy array (OpenCV format).
        languages: Language codes for EasyOCR. Defaults to config.OCR_LANGUAGES.

    Returns:
        OCRResult with spans, raw_text, and avg_confidence.
    """
    reader = _get_reader(languages)
    results = reader.readtext(image)

    if not results:
        return OCRResult(spans=[], raw_text="", avg_confidence=0.0)

    spans = []
    for bbox_points, text, confidence in results:
        # bbox_points is [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        # Convert to [x, y, w, h]
        xs = [p[0] for p in bbox_points]
        ys = [p[1] for p in bbox_points]
        x = min(xs)
        y = min(ys)
        w = max(xs) - x
        h = max(ys) - y
        spans.append(OCRSpan(
            text=text,
            confidence=float(confidence),
            bbox=[float(x), float(y), float(w), float(h)],
        ))

    raw_text = "\n".join(span.text for span in spans)
    avg_confidence = sum(s.confidence for s in spans) / len(spans)

    return OCRResult(
        spans=spans,
        raw_text=raw_text,
        avg_confidence=avg_confidence,
    )


def run_ocr_from_path(path: str, languages: list[str] | None = None) -> OCRResult:
    """Convenience: load image from path, then run OCR."""
    from contextlens.preprocess import load_image
    image = load_image(path)
    return run_ocr(image, languages)
