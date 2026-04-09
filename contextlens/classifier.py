"""Rule-based image type classifier over OCR text and layout cues."""

from __future__ import annotations

import re

from contextlens.config import TYPE_CONFIDENCE_AMBIGUOUS
from contextlens.schemas import ImageType, OCRResult


# --- Indicator patterns ---

# Receipt indicators
_RECEIPT_KEYWORDS = re.compile(
    r"\b(TOTAL|SUBTOTAL|SUB\s*TOTAL|TAX|GRAND\s*TOTAL|AMOUNT\s*DUE|"
    r"CHANGE|CASH|CREDIT|DEBIT|VISA|MASTERCARD|TIP|BALANCE)\b",
    re.IGNORECASE,
)
_PRICE_PATTERN = re.compile(r"\$\s?\d+\.?\d{0,2}")

# Conversation indicators
_TIMESTAMP_PATTERN = re.compile(r"\b\d{1,2}:\d{2}(?:\s?[APap][Mm])?\b")
_SPEAKER_PATTERN = re.compile(r"^[A-Z][a-z]+\s?:", re.MULTILINE)

# Document indicators
_KEY_VALUE_PATTERN = re.compile(r"^[A-Za-z][\w\s]{0,30}:\s*.+", re.MULTILINE)
_DOCUMENT_KEYWORDS = re.compile(
    r"\b(Invoice|Form|Patient|Date of Birth|DOB|Name|Address|"
    r"Phone|Email|Account|ID|Number|Ref|Reference|Dosage|mg)\b",
    re.IGNORECASE,
)

# Whiteboard indicators
_BULLET_PATTERN = re.compile(r"^[\s]*[-*\u2022]\s+", re.MULTILINE)
_OWNER_PATTERN = re.compile(r"@\w+", re.IGNORECASE)
_HASHTAG_PATTERN = re.compile(r"#\w+")


def _score_receipt(ocr: OCRResult) -> float:
    """Score how likely the OCR text represents a receipt."""
    text = ocr.raw_text
    score = 0.0

    # Keyword matches
    keyword_hits = len(_RECEIPT_KEYWORDS.findall(text))
    score += min(keyword_hits * 0.15, 0.45)

    # Price patterns ($X.XX)
    price_hits = len(_PRICE_PATTERN.findall(text))
    score += min(price_hits * 0.1, 0.3)

    # Many short lines (receipt-like layout)
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if lines:
        avg_len = sum(len(ln) for ln in lines) / len(lines)
        if avg_len < 40:
            score += 0.15

    # Lines with prices at the end
    price_line_count = sum(
        1 for ln in lines if _PRICE_PATTERN.search(ln)
    )
    if lines and price_line_count / len(lines) > 0.3:
        score += 0.1

    return min(score, 1.0)


def _score_conversation(ocr: OCRResult) -> float:
    """Score how likely the OCR text represents a conversation screenshot."""
    text = ocr.raw_text
    score = 0.0

    # Timestamp patterns
    ts_hits = len(_TIMESTAMP_PATTERN.findall(text))
    score += min(ts_hits * 0.15, 0.3)

    # Speaker-like patterns ("Alice:", "Bob:")
    speaker_hits = len(_SPEAKER_PATTERN.findall(text))
    score += min(speaker_hits * 0.15, 0.3)

    # Short alternating messages
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if len(lines) >= 3:
        short_lines = sum(1 for ln in lines if len(ln) < 60)
        if short_lines / len(lines) > 0.5:
            score += 0.15

    # Multiple distinct speakers
    if speaker_hits >= 2:
        score += 0.15

    # Conversation action words
    action_words = re.findall(
        r"\b(let'?s|sounds good|okay|sure|meeting|call|sync|tomorrow|"
        r"send|review|schedule|prepare)\b",
        text, re.IGNORECASE,
    )
    score += min(len(action_words) * 0.05, 0.1)

    return min(score, 1.0)


def _score_document(ocr: OCRResult) -> float:
    """Score how likely the OCR text represents a document photo."""
    text = ocr.raw_text
    score = 0.0

    # Key-value patterns
    kv_hits = len(_KEY_VALUE_PATTERN.findall(text))
    score += min(kv_hits * 0.1, 0.35)

    # Document keywords
    doc_keyword_hits = len(_DOCUMENT_KEYWORDS.findall(text))
    score += min(doc_keyword_hits * 0.1, 0.35)

    # Regular block text (longer lines)
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if lines:
        avg_len = sum(len(ln) for ln in lines) / len(lines)
        if avg_len > 30:
            score += 0.15

    # Form-like structure (multiple labeled fields)
    if kv_hits >= 3:
        score += 0.15

    return min(score, 1.0)


def _score_whiteboard(ocr: OCRResult) -> float:
    """Score how likely the OCR text represents a whiteboard/handwritten note."""
    text = ocr.raw_text
    score = 0.0

    # Bullet patterns
    bullet_hits = len(_BULLET_PATTERN.findall(text))
    score += min(bullet_hits * 0.12, 0.35)

    # @ mentions (owners)
    owner_hits = len(_OWNER_PATTERN.findall(text))
    score += min(owner_hits * 0.12, 0.25)

    # Hashtags / project tags
    hashtag_hits = len(_HASHTAG_PATTERN.findall(text))
    score += min(hashtag_hits * 0.12, 0.25)

    # Sparse, scattered text (few words per line, many lines)
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if lines:
        avg_len = sum(len(ln) for ln in lines) / len(lines)
        if avg_len < 30 and len(lines) >= 3:
            score += 0.1

    # Task-like language
    task_words = re.findall(
        r"\b(TODO|task|deadline|due|action|assign|owner|sprint|standup|"
        r"backlog|priority)\b",
        text, re.IGNORECASE,
    )
    score += min(len(task_words) * 0.05, 0.15)

    return min(score, 1.0)


def classify_image(ocr: OCRResult) -> tuple[ImageType, float]:
    """Classify an image type based on OCR output.

    Args:
        ocr: OCRResult from the OCR engine.

    Returns:
        Tuple of (predicted ImageType, type_confidence).
    """
    # Handle empty OCR
    if not ocr.spans and not ocr.raw_text.strip():
        return ImageType.UNKNOWN, 0.0

    scores = {
        ImageType.RECEIPT: _score_receipt(ocr),
        ImageType.CONVERSATION: _score_conversation(ocr),
        ImageType.DOCUMENT: _score_document(ocr),
        ImageType.WHITEBOARD: _score_whiteboard(ocr),
    }

    # Pick the highest-scoring type
    best_type = max(scores, key=scores.get)  # type: ignore[arg-type]
    best_score = scores[best_type]

    # If best score is too low, classify as unknown
    if best_score < 0.1:
        return ImageType.UNKNOWN, 0.0

    # Compute confidence: higher when there's clear separation from runner-up
    sorted_scores = sorted(scores.values(), reverse=True)
    gap = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else sorted_scores[0]

    # Confidence = base score scaled by separation
    # Large gap → high confidence, small gap → ambiguous
    confidence = min(best_score * 0.6 + gap * 0.4 + 0.1, 1.0)

    # Cap confidence if it's still ambiguous
    if gap < 0.1:
        confidence = min(confidence, TYPE_CONFIDENCE_AMBIGUOUS)

    return best_type, round(confidence, 3)
