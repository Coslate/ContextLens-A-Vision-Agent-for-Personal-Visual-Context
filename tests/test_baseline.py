"""Tests for the generic unified parser baseline — PR10."""

import pytest

from contextlens.baseline import (
    _flat_confidence,
    _generate_summary,
    _infer_type,
    baseline_process,
    generic_extract,
)
from contextlens.config import BASELINE_CONFIDENCE
from contextlens.schemas import (
    ConversationEntities,
    DocumentEntities,
    ImageOutput,
    ImageType,
    OCRResult,
    OCRSpan,
    QualitySignals,
    ReceiptEntities,
    WhiteboardEntities,
)


# =====================================================================
# Helper: build OCRResult quickly
# =====================================================================

def _ocr(raw_text: str, avg_confidence: float = 0.9) -> OCRResult:
    """Build a minimal OCRResult from raw text."""
    lines = [ln for ln in raw_text.split("\n") if ln.strip()]
    spans = [
        OCRSpan(text=ln, confidence=avg_confidence, bbox=[10, i * 30, 200, 20])
        for i, ln in enumerate(lines)
    ]
    return OCRResult(spans=spans, raw_text=raw_text, avg_confidence=avg_confidence)


def _quality() -> QualitySignals:
    return QualitySignals(
        blur_score=500.0, brightness=0.6, contrast=0.5,
        estimated_quality=0.8,
    )


# =====================================================================
# Sample OCR texts (type-agnostic — baseline sees all the same way)
# =====================================================================

RECEIPT_TEXT = """\
STARBUCKS
Latte  $5.50
Muffin  $3.25
TOTAL  $8.75
03/15/2024"""

CONVERSATION_TEXT = """\
Alice: Hey, can you send the report by Friday?
Bob: Sure, I'll review it tonight and send it over.
Alice: Great, let's schedule a call for Monday."""

DOCUMENT_TEXT = """\
Name: John Doe
DOB: 01/01/1990
Address: 123 Main St
Phone: 555-0100"""

WHITEBOARD_TEXT = """\
- Design API
- Write tests
- Deploy to staging
#ProjectAlpha
@Alice due Friday"""

BLURRY_RECEIPT_TEXT = """\
STARBUKS
Lat  $5.0
TOT  $5.0"""


# =====================================================================
# generic_extract — shared rules
# =====================================================================

class TestGenericExtract:
    def test_extracts_amounts_from_receipt(self):
        result = generic_extract(_ocr(RECEIPT_TEXT))
        assert len(result["amounts"]) >= 3  # 5.50, 3.25, 8.75
        assert 8.75 in result["amounts"]

    def test_total_guess_is_largest_amount(self):
        result = generic_extract(_ocr(RECEIPT_TEXT))
        assert result["total_guess"] == 8.75

    def test_extracts_dates(self):
        result = generic_extract(_ocr(RECEIPT_TEXT))
        assert len(result["dates"]) >= 1
        assert "03/15/2024" in result["dates"]

    def test_extracts_kv_from_document(self):
        result = generic_extract(_ocr(DOCUMENT_TEXT))
        assert "Name" in result["structured_fields"]
        assert result["structured_fields"]["Name"] == "John Doe"

    def test_extracts_action_items_from_conversation(self):
        result = generic_extract(_ocr(CONVERSATION_TEXT))
        assert len(result["action_items"]) >= 1
        # At least one line with an action verb
        action_text = " ".join(result["action_items"]).lower()
        assert any(v in action_text for v in ["send", "review", "schedule"])

    def test_extracts_people_from_conversation(self):
        result = generic_extract(_ocr(CONVERSATION_TEXT))
        people_lower = [p.lower() for p in result["people"]]
        assert "alice" in people_lower or "bob" in people_lower

    def test_text_blocks_collected(self):
        result = generic_extract(_ocr(WHITEBOARD_TEXT))
        assert len(result["text_blocks"]) >= 3

    def test_merchant_guess_is_first_line(self):
        result = generic_extract(_ocr(RECEIPT_TEXT))
        assert result["merchant_guess"] is not None
        assert "STARBUCKS" in result["merchant_guess"]

    def test_no_amounts_in_no_price_text(self):
        result = generic_extract(_ocr("Hello world\nNo prices here"))
        assert result["amounts"] == []
        assert result["total_guess"] is None

    def test_amount_regex_fires_on_whiteboard(self):
        """Whiteboard text with dollar signs should trigger amount regex (noise)."""
        wb_with_prices = "- Design API\n- Budget $500\n- Cost $200\n#ProjectAlpha"
        result = generic_extract(_ocr(wb_with_prices))
        assert len(result["amounts"]) >= 2  # noise from shared rules


# =====================================================================
# Type inference (output-based)
# =====================================================================

class TestInferType:
    def test_receipt_from_amounts(self):
        extracted = generic_extract(_ocr(RECEIPT_TEXT))
        assert _infer_type(extracted) == ImageType.RECEIPT

    def test_conversation_from_people_and_actions(self):
        extracted = generic_extract(_ocr(CONVERSATION_TEXT))
        assert _infer_type(extracted) == ImageType.CONVERSATION

    def test_document_from_kv(self):
        extracted = generic_extract(_ocr(DOCUMENT_TEXT))
        assert _infer_type(extracted) == ImageType.DOCUMENT

    def test_whiteboard_fallback(self):
        """Plain text without amounts, KV, or action verbs → whiteboard."""
        extracted = generic_extract(_ocr("Task 1\nTask 2\nTask 3"))
        assert _infer_type(extracted) == ImageType.WHITEBOARD

    def test_single_amount_with_merchant_is_receipt(self):
        extracted = generic_extract(_ocr("STARBUCKS\n$5.50"))
        assert _infer_type(extracted) == ImageType.RECEIPT


# =====================================================================
# Flat confidence
# =====================================================================

class TestFlatConfidence:
    def test_receipt_confidence_is_flat(self):
        extracted = generic_extract(_ocr(RECEIPT_TEXT))
        img_type = _infer_type(extracted)
        from contextlens.baseline import _ENTITY_BUILDERS
        entities = _ENTITY_BUILDERS[img_type](extracted)
        conf = _flat_confidence(entities, img_type)
        for val in conf.values():
            assert val == BASELINE_CONFIDENCE

    def test_empty_fields_get_no_confidence(self):
        entities = ReceiptEntities()  # all None/empty
        conf = _flat_confidence(entities, ImageType.RECEIPT)
        assert conf == {}


# =====================================================================
# baseline_process — integration
# =====================================================================

class TestBaselineProcess:
    def test_receipt_output_valid(self):
        out = baseline_process(_ocr(RECEIPT_TEXT), image_id="r1")
        assert isinstance(out, ImageOutput)
        assert out.image_id == "r1"
        assert out.type == ImageType.RECEIPT
        assert out.type_confidence == BASELINE_CONFIDENCE
        assert isinstance(out.extracted_entities, ReceiptEntities)

    def test_receipt_total_extracted(self):
        out = baseline_process(_ocr(RECEIPT_TEXT), image_id="r1")
        entities = out.extracted_entities
        assert entities.total == 8.75

    def test_receipt_merchant_extracted(self):
        out = baseline_process(_ocr(RECEIPT_TEXT), image_id="r1")
        entities = out.extracted_entities
        assert entities.merchant is not None
        assert "STARBUCKS" in entities.merchant

    def test_receipt_date_extracted(self):
        out = baseline_process(_ocr(RECEIPT_TEXT), image_id="r1")
        entities = out.extracted_entities
        assert entities.date == "03/15/2024"

    def test_conversation_output(self):
        out = baseline_process(_ocr(CONVERSATION_TEXT), image_id="c1")
        assert out.type == ImageType.CONVERSATION
        assert isinstance(out.extracted_entities, ConversationEntities)
        assert len(out.extracted_entities.participants) >= 1

    def test_conversation_no_calendar_hook(self):
        """Baseline never produces calendar hooks."""
        out = baseline_process(_ocr(CONVERSATION_TEXT), image_id="c1")
        assert out.calendar_hook is None

    def test_conversation_action_items(self):
        out = baseline_process(_ocr(CONVERSATION_TEXT), image_id="c1")
        assert len(out.extracted_entities.action_items) >= 1

    def test_document_output(self):
        out = baseline_process(_ocr(DOCUMENT_TEXT), image_id="d1")
        assert out.type == ImageType.DOCUMENT
        assert isinstance(out.extracted_entities, DocumentEntities)
        assert "Name" in out.extracted_entities.structured_fields

    def test_whiteboard_output(self):
        out = baseline_process(_ocr(WHITEBOARD_TEXT), image_id="w1")
        # Whiteboard or document — baseline may mis-classify since
        # it uses output-based type inference, not input routing
        assert isinstance(out, ImageOutput)
        assert len(out.raw_text) > 0

    def test_no_failure_flags(self):
        """Baseline never sets failure flags — even on blurry input."""
        blurry_quality = QualitySignals(
            blur_score=20.0, brightness=0.5, contrast=0.4,
            estimated_quality=0.3, is_blurry=True,
        )
        out = baseline_process(
            _ocr(BLURRY_RECEIPT_TEXT, avg_confidence=0.4),
            image_id="blurry",
            quality=blurry_quality,
        )
        assert out.failure_flags == []
        assert out.needs_clarification is False

    def test_no_group_id(self):
        """Baseline never assigns group_id."""
        out = baseline_process(_ocr(RECEIPT_TEXT), image_id="r1")
        assert out.group_id is None

    def test_flat_confidence_values(self):
        """All field confidence values should be exactly BASELINE_CONFIDENCE."""
        out = baseline_process(_ocr(RECEIPT_TEXT), image_id="r1")
        for val in out.field_confidence.values():
            assert val == BASELINE_CONFIDENCE

    def test_summary_non_empty(self):
        out = baseline_process(_ocr(RECEIPT_TEXT), image_id="r1")
        assert len(out.summary) > 0

    def test_quality_stored_but_not_used(self):
        """Quality signals are stored but don't affect confidence."""
        blurry_quality = QualitySignals(
            blur_score=10.0, brightness=0.3, contrast=0.2,
            estimated_quality=0.2, is_blurry=True,
        )
        out = baseline_process(
            _ocr(RECEIPT_TEXT), image_id="r1", quality=blurry_quality,
        )
        assert out.quality_signals is not None
        assert out.quality_signals.is_blurry is True
        # But confidence is still flat 0.8
        for val in out.field_confidence.values():
            assert val == BASELINE_CONFIDENCE

    def test_raw_text_preserved(self):
        out = baseline_process(_ocr(RECEIPT_TEXT), image_id="r1")
        assert "STARBUCKS" in out.raw_text
        assert "$5.50" in out.raw_text


# =====================================================================
# Baseline vs structured pipeline: key differences
# =====================================================================

class TestBaselineVsStructured:
    def test_blurry_same_confidence_as_clean(self):
        """Baseline gives same confidence to blurry and clean — unlike structured."""
        clean = baseline_process(_ocr(RECEIPT_TEXT), image_id="clean")
        blurry = baseline_process(
            _ocr(BLURRY_RECEIPT_TEXT, avg_confidence=0.4),
            image_id="blurry",
            quality=QualitySignals(
                blur_score=10.0, brightness=0.3, contrast=0.2,
                estimated_quality=0.2, is_blurry=True,
            ),
        )
        # Both get flat confidence — baseline doesn't calibrate
        clean_confs = set(clean.field_confidence.values())
        blurry_confs = set(blurry.field_confidence.values())
        assert clean_confs == {BASELINE_CONFIDENCE}
        assert blurry_confs == {BASELINE_CONFIDENCE}

    def test_no_type_specific_items_parsing(self):
        """Baseline item parsing is generic — items are just 'item', not named."""
        out = baseline_process(_ocr(RECEIPT_TEXT), image_id="r1")
        entities = out.extracted_entities
        # Baseline assigns generic "item" name (no type-specific parsing)
        for item in entities.items:
            assert item.name == "item"

    def test_whiteboard_gets_amount_noise(self):
        """Amount regex fires on whiteboard text with dollar signs — noise."""
        wb_text = "- Design API\n- Budget $500\n- Cost $200\n#ProjectAlpha"
        out = baseline_process(_ocr(wb_text), image_id="w1")
        # Baseline may misclassify whiteboard as receipt due to $ amounts
        if out.type == ImageType.RECEIPT:
            # This is the expected noise from generic extraction
            assert out.extracted_entities.total is not None

    def test_no_calendar_detection(self):
        """Even conversation mentioning a meeting gets no calendar hook."""
        meeting_text = (
            "Alice: Let's have a meeting tomorrow at 3pm\n"
            "Bob: Sounds good, I'll book the conference room"
        )
        out = baseline_process(_ocr(meeting_text), image_id="c_meeting")
        assert out.calendar_hook is None

    def test_no_linking_across_outputs(self):
        """Multiple outputs never get linked — no group_id."""
        out1 = baseline_process(_ocr(RECEIPT_TEXT), image_id="r1")
        out2 = baseline_process(
            _ocr("STARBUCKS\nEspresso  $4.00\nTOTAL  $4.00\n03/16/2024"),
            image_id="r2",
        )
        assert out1.group_id is None
        assert out2.group_id is None


# =====================================================================
# Summary generation
# =====================================================================

class TestSummary:
    def test_receipt_summary_has_merchant(self):
        out = baseline_process(_ocr(RECEIPT_TEXT), image_id="r1")
        assert "STARBUCKS" in out.summary

    def test_receipt_summary_has_total(self):
        out = baseline_process(_ocr(RECEIPT_TEXT), image_id="r1")
        assert "$8.75" in out.summary

    def test_conversation_summary_has_participant(self):
        out = baseline_process(_ocr(CONVERSATION_TEXT), image_id="c1")
        assert any(
            name in out.summary for name in ["Alice", "Bob"]
        )

    def test_document_summary_has_fields(self):
        out = baseline_process(_ocr(DOCUMENT_TEXT), image_id="d1")
        assert "field" in out.summary.lower()


# =====================================================================
# Edge cases
# =====================================================================

class TestEdgeCases:
    def test_empty_text(self):
        out = baseline_process(
            OCRResult(spans=[], raw_text="", avg_confidence=0.0),
            image_id="empty",
        )
        assert isinstance(out, ImageOutput)
        assert out.type in list(ImageType)
        assert out.failure_flags == []

    def test_single_line(self):
        out = baseline_process(_ocr("Hello World"), image_id="single")
        assert isinstance(out, ImageOutput)
        assert len(out.summary) > 0

    def test_output_serializes_to_json(self):
        out = baseline_process(_ocr(RECEIPT_TEXT), image_id="r1")
        data = out.model_dump(mode="json")
        assert isinstance(data, dict)
        assert data["image_id"] == "r1"
        assert data["type"] == "receipt"
        assert data["group_id"] is None
        assert data["calendar_hook"] is None
