"""Tests for image type classifier — PR4."""

import pytest

from contextlens.classifier import (
    classify_image,
    _score_receipt,
    _score_conversation,
    _score_document,
    _score_whiteboard,
)
from contextlens.config import TYPE_CONFIDENCE_AMBIGUOUS
from contextlens.schemas import ImageType, OCRResult, OCRSpan


# --- Helpers ---

def make_ocr(raw_text: str, avg_confidence: float = 0.9) -> OCRResult:
    """Create a minimal OCRResult from raw text for classifier testing."""
    lines = [ln for ln in raw_text.split("\n") if ln.strip()]
    spans = [
        OCRSpan(text=ln, confidence=avg_confidence, bbox=[0, i * 30, 200, 20])
        for i, ln in enumerate(lines)
    ]
    return OCRResult(spans=spans, raw_text=raw_text, avg_confidence=avg_confidence)


# --- Receipt classification ---

class TestReceiptClassification:
    def test_clean_receipt(self):
        ocr = make_ocr(
            "STARBUCKS\n"
            "Latte    $5.50\n"
            "Muffin   $3.25\n"
            "SUBTOTAL $8.75\n"
            "TAX      $0.70\n"
            "TOTAL    $9.45\n"
            "VISA *1234\n"
            "03/15/2024"
        )
        img_type, conf = classify_image(ocr)
        assert img_type == ImageType.RECEIPT
        assert conf > TYPE_CONFIDENCE_AMBIGUOUS

    def test_receipt_keywords_boost_score(self):
        ocr = make_ocr("TOTAL $10.00\nSUBTOTAL $9.00\nTAX $1.00\nCASH")
        score = _score_receipt(ocr)
        assert score > 0.4

    def test_price_patterns_contribute(self):
        ocr = make_ocr("Item A $5.00\nItem B $3.50\nItem C $2.00")
        score = _score_receipt(ocr)
        assert score > 0.2


# --- Conversation classification ---

class TestConversationClassification:
    def test_clean_conversation(self):
        ocr = make_ocr(
            "Alice: Hey, let's meet tomorrow at 3pm\n"
            "Bob: Sounds good, I'll prepare the slides\n"
            "Alice: Great, see you then\n"
            "10:30 AM"
        )
        img_type, conf = classify_image(ocr)
        assert img_type == ImageType.CONVERSATION
        assert conf > TYPE_CONFIDENCE_AMBIGUOUS

    def test_conversation_with_timestamps(self):
        ocr = make_ocr(
            "9:00 AM\n"
            "Alice: Good morning\n"
            "10:15 AM\n"
            "Bob: Let's sync on the project\n"
            "11:30 AM\n"
            "Alice: Sure, schedule a call"
        )
        score = _score_conversation(ocr)
        assert score > 0.3

    def test_speaker_patterns(self):
        ocr = make_ocr(
            "Alice: Hello\n"
            "Bob: Hi there\n"
            "Charlie: Welcome"
        )
        score = _score_conversation(ocr)
        assert score > 0.3


# --- Document classification ---

class TestDocumentClassification:
    def test_structured_form(self):
        ocr = make_ocr(
            "Patient Name: John Doe\n"
            "Date of Birth: 01/15/1985\n"
            "Phone: 555-0123\n"
            "Address: 123 Main St\n"
            "Email: john@example.com\n"
            "Insurance ID: INS-98765"
        )
        img_type, conf = classify_image(ocr)
        assert img_type == ImageType.DOCUMENT
        assert conf > TYPE_CONFIDENCE_AMBIGUOUS

    def test_invoice_document(self):
        ocr = make_ocr(
            "Invoice Number: INV-2024-001\n"
            "Date: March 15, 2024\n"
            "Bill To: Acme Corp\n"
            "Account: ACC-12345\n"
            "Amount Due: $1,250.00\n"
            "Reference: PO-789"
        )
        img_type, conf = classify_image(ocr)
        assert img_type == ImageType.DOCUMENT
        assert conf > TYPE_CONFIDENCE_AMBIGUOUS

    def test_key_value_patterns(self):
        ocr = make_ocr(
            "Name: Jane Smith\n"
            "ID: 12345\n"
            "Department: Engineering\n"
            "Role: Senior Developer"
        )
        score = _score_document(ocr)
        assert score > 0.4


# --- Whiteboard classification ---

class TestWhiteboardClassification:
    def test_clean_whiteboard(self):
        ocr = make_ocr(
            "- Design API endpoints\n"
            "- Write unit tests @Alice\n"
            "- Deploy to staging @Bob\n"
            "- Review PR due Friday\n"
            "#ProjectAlpha #Sprint5\n"
            "TODO: update docs"
        )
        img_type, conf = classify_image(ocr)
        assert img_type == ImageType.WHITEBOARD
        assert conf > TYPE_CONFIDENCE_AMBIGUOUS

    def test_whiteboard_bullets_and_tags(self):
        ocr = make_ocr(
            "- Task 1\n"
            "- Task 2\n"
            "- Task 3\n"
            "#ProjectBeta @Charlie\n"
            "deadline: next Monday"
        )
        score = _score_whiteboard(ocr)
        assert score > 0.3

    def test_hashtags_and_owners(self):
        ocr = make_ocr("#Alpha #Beta @Alice @Bob TODO: fix bug")
        score = _score_whiteboard(ocr)
        assert score > 0.2


# --- Edge cases ---

class TestEdgeCases:
    def test_empty_ocr_returns_unknown(self):
        ocr = OCRResult(spans=[], raw_text="", avg_confidence=0.0)
        img_type, conf = classify_image(ocr)
        assert img_type == ImageType.UNKNOWN
        assert conf == 0.0

    def test_whitespace_only_returns_unknown(self):
        ocr = OCRResult(spans=[], raw_text="   \n  \n  ", avg_confidence=0.0)
        img_type, conf = classify_image(ocr)
        assert img_type == ImageType.UNKNOWN
        assert conf == 0.0

    def test_ambiguous_text_lower_confidence(self):
        """Text with no clear indicators should have low confidence."""
        ocr = make_ocr("Hello world\nSome random text\nNothing specific")
        _, conf = classify_image(ocr)
        # Should still classify something, but with low confidence
        assert conf <= 0.6

    def test_confidence_is_bounded(self):
        """Confidence should always be in [0, 1]."""
        ocr = make_ocr(
            "TOTAL $100\nSUBTOTAL $90\nTAX $10\n"
            "GRAND TOTAL $100\nCASH\nVISA\nCREDIT\n"
            "$5.00\n$10.00\n$15.00\n$20.00\n$25.00"
        )
        _, conf = classify_image(ocr)
        assert 0.0 <= conf <= 1.0

    def test_all_scores_non_negative(self):
        """All individual type scores should be non-negative."""
        ocr = make_ocr("Just some text")
        assert _score_receipt(ocr) >= 0.0
        assert _score_conversation(ocr) >= 0.0
        assert _score_document(ocr) >= 0.0
        assert _score_whiteboard(ocr) >= 0.0

    def test_classify_returns_tuple(self):
        ocr = make_ocr("TOTAL $5.00")
        result = classify_image(ocr)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], ImageType)
        assert isinstance(result[1], float)


# --- Cross-type discrimination ---

class TestCrossTypeDiscrimination:
    def test_receipt_not_classified_as_conversation(self):
        ocr = make_ocr(
            "WALMART\nMilk $3.99\nBread $2.50\nTOTAL $6.49\nCASH"
        )
        img_type, _ = classify_image(ocr)
        assert img_type != ImageType.CONVERSATION

    def test_conversation_not_classified_as_receipt(self):
        ocr = make_ocr(
            "Alice: Are you coming to the meeting?\n"
            "Bob: Yes, I'll be there at 2:00 PM\n"
            "Alice: Great, see you soon"
        )
        img_type, _ = classify_image(ocr)
        assert img_type != ImageType.RECEIPT

    def test_document_not_classified_as_whiteboard(self):
        ocr = make_ocr(
            "Patient Name: Sarah Connor\n"
            "Date of Birth: 02/28/1965\n"
            "Phone: 555-9876\n"
            "Insurance: Blue Cross #BC-45678\n"
            "Physician: Dr. Miles Dyson"
        )
        img_type, _ = classify_image(ocr)
        assert img_type != ImageType.WHITEBOARD

    def test_whiteboard_not_classified_as_document(self):
        ocr = make_ocr(
            "- Sprint planning\n"
            "- Review backlog @Alice\n"
            "- Deploy v2.0 @Bob\n"
            "#ProjectGamma\n"
            "TODO: standup notes\n"
            "priority: high"
        )
        img_type, _ = classify_image(ocr)
        assert img_type != ImageType.DOCUMENT
