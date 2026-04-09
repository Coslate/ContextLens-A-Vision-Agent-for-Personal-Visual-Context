"""Tests for Pydantic schemas — PR1."""

import json

import pytest
from pydantic import ValidationError

from contextlens.schemas import (
    Annotation,
    CalendarEventCandidate,
    CalendarHook,
    ConversationEntities,
    DocumentEntities,
    FailureFlag,
    ImageOutput,
    ImageType,
    OCRResult,
    OCRSpan,
    QualitySignals,
    ReceiptEntities,
    ReceiptItem,
    TextBlock,
    WhiteboardEntities,
    WhiteboardStructure,
)


# --- Enum Tests ---

class TestImageType:
    def test_all_types_exist(self):
        assert ImageType.RECEIPT == "receipt"
        assert ImageType.CONVERSATION == "conversation"
        assert ImageType.DOCUMENT == "document"
        assert ImageType.WHITEBOARD == "whiteboard"
        assert ImageType.UNKNOWN == "unknown"

    def test_enum_count(self):
        assert len(ImageType) == 5


class TestFailureFlag:
    def test_all_flags_exist(self):
        assert FailureFlag.BLURRY_IMAGE == "blurry_image"
        assert FailureFlag.ROTATION_CORRECTED == "rotation_corrected"
        assert FailureFlag.ROTATION_UNRESOLVED == "rotation_unresolved"
        assert FailureFlag.PARTIAL_CAPTURE == "partial_capture"
        assert FailureFlag.OCR_UNCERTAIN == "ocr_uncertain"
        assert FailureFlag.MIXED_LANGUAGE == "mixed_language"

    def test_enum_count(self):
        assert len(FailureFlag) == 6


# --- QualitySignals Tests ---

class TestQualitySignals:
    def test_valid_signals(self, sample_quality_signals):
        assert sample_quality_signals.blur_score == 850.3
        assert sample_quality_signals.is_blurry is False

    def test_boundary_values(self):
        qs = QualitySignals(
            blur_score=0.0, brightness=0.0, contrast=0.0,
            estimated_quality=0.0, is_blurry=True, is_rotated=True,
        )
        assert qs.brightness == 0.0
        assert qs.is_blurry is True

    def test_invalid_brightness_too_high(self):
        with pytest.raises(ValidationError):
            QualitySignals(
                blur_score=100, brightness=1.5, contrast=0.5,
                estimated_quality=0.5,
            )

    def test_invalid_brightness_negative(self):
        with pytest.raises(ValidationError):
            QualitySignals(
                blur_score=100, brightness=-0.1, contrast=0.5,
                estimated_quality=0.5,
            )


# --- OCR Tests ---

class TestOCRSpan:
    def test_valid_span(self):
        span = OCRSpan(text="hello", confidence=0.9, bbox=[0, 0, 100, 20])
        assert span.text == "hello"
        assert span.confidence == 0.9

    def test_invalid_confidence(self):
        with pytest.raises(ValidationError):
            OCRSpan(text="x", confidence=1.5, bbox=[0, 0, 10, 10])


class TestOCRResult:
    def test_empty_result(self):
        result = OCRResult()
        assert result.spans == []
        assert result.raw_text == ""
        assert result.avg_confidence == 0.0

    def test_with_spans(self, sample_ocr_result):
        assert len(sample_ocr_result.spans) == 4
        assert "STARBUCKS" in sample_ocr_result.raw_text
        assert sample_ocr_result.avg_confidence == 0.90


# --- Entity Tests ---

class TestReceiptEntities:
    def test_full_receipt(self, sample_receipt_entities):
        assert sample_receipt_entities.merchant == "Starbucks"
        assert len(sample_receipt_entities.items) == 1
        assert sample_receipt_entities.items[0].price == 5.50
        assert sample_receipt_entities.total == 5.50

    def test_empty_receipt(self):
        r = ReceiptEntities()
        assert r.merchant is None
        assert r.items == []
        assert r.total is None

    def test_receipt_item(self):
        item = ReceiptItem(name="Coffee", price=3.99)
        assert item.name == "Coffee"
        assert item.price == 3.99


class TestConversationEntities:
    def test_full_conversation(self, sample_conversation_entities):
        assert "Alice" in sample_conversation_entities.participants
        assert len(sample_conversation_entities.action_items) == 1
        assert len(sample_conversation_entities.referenced_events) == 1

    def test_empty_conversation(self):
        c = ConversationEntities()
        assert c.participants == []
        assert c.key_topics == []


class TestCalendarHook:
    def test_calendar_hook(self):
        hook = CalendarHook(
            mentioned=True,
            event_candidates=[
                CalendarEventCandidate(
                    title="Standup",
                    time_mention="9am",
                    participants=["Team"],
                )
            ],
        )
        assert hook.mentioned is True
        assert hook.event_candidates[0].title == "Standup"

    def test_empty_hook(self):
        hook = CalendarHook()
        assert hook.mentioned is False
        assert hook.event_candidates == []


class TestDocumentEntities:
    def test_full_document(self, sample_document_entities):
        assert sample_document_entities.document_kind == "form"
        assert sample_document_entities.structured_fields["Name"] == "John Doe"

    def test_empty_document(self):
        d = DocumentEntities()
        assert d.document_kind is None
        assert d.structured_fields == {}


class TestWhiteboardEntities:
    def test_full_whiteboard(self, sample_whiteboard_entities):
        assert len(sample_whiteboard_entities.text_blocks) == 2
        s = sample_whiteboard_entities.inferred_structure
        assert "ProjectAlpha" in s.project_tags
        assert "Alice" in s.owners

    def test_empty_whiteboard(self):
        w = WhiteboardEntities()
        assert w.text_blocks == []
        assert w.inferred_structure.bullets == []


# --- ImageOutput Tests ---

class TestImageOutput:
    def test_full_output(self, sample_image_output):
        out = sample_image_output
        assert out.image_id == "img_001"
        assert out.type == ImageType.RECEIPT
        assert out.type_confidence == 0.92
        assert out.summary != ""
        assert out.failure_flags == []
        assert out.needs_clarification is False

    def test_minimal_output(self):
        out = ImageOutput(image_id="test")
        assert out.image_id == "test"
        assert out.type == ImageType.UNKNOWN
        assert out.type_confidence == 0.0
        assert out.summary == ""

    def test_json_serialization(self, sample_image_output):
        json_str = sample_image_output.model_dump_json()
        data = json.loads(json_str)
        assert data["image_id"] == "img_001"
        assert data["type"] == "receipt"
        assert isinstance(data["field_confidence"], dict)

    def test_json_roundtrip(self, sample_image_output):
        json_str = sample_image_output.model_dump_json()
        restored = ImageOutput.model_validate_json(json_str)
        assert restored.image_id == sample_image_output.image_id
        assert restored.type == sample_image_output.type
        assert restored.type_confidence == sample_image_output.type_confidence

    def test_with_failure_flags(self):
        out = ImageOutput(
            image_id="bad_img",
            failure_flags=[FailureFlag.BLURRY_IMAGE, FailureFlag.PARTIAL_CAPTURE],
            needs_clarification=True,
        )
        assert len(out.failure_flags) == 2
        assert FailureFlag.BLURRY_IMAGE in out.failure_flags
        assert out.needs_clarification is True

    def test_with_calendar_hook(self):
        hook = CalendarHook(
            mentioned=True,
            event_candidates=[
                CalendarEventCandidate(title="Sync", time_mention="3pm")
            ],
        )
        out = ImageOutput(image_id="conv_001", calendar_hook=hook)
        assert out.calendar_hook.mentioned is True

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            ImageOutput(image_id="x", type_confidence=1.5)
        with pytest.raises(ValidationError):
            ImageOutput(image_id="x", type_confidence=-0.1)

    def test_with_each_entity_type(
        self,
        sample_receipt_entities,
        sample_conversation_entities,
        sample_document_entities,
        sample_whiteboard_entities,
    ):
        for entities, img_type in [
            (sample_receipt_entities, ImageType.RECEIPT),
            (sample_conversation_entities, ImageType.CONVERSATION),
            (sample_document_entities, ImageType.DOCUMENT),
            (sample_whiteboard_entities, ImageType.WHITEBOARD),
        ]:
            out = ImageOutput(
                image_id=f"test_{img_type.value}",
                type=img_type,
                extracted_entities=entities,
            )
            assert out.type == img_type
            json_str = out.model_dump_json()
            assert json_str  # serializes without error


# --- Annotation Tests ---

class TestAnnotation:
    def test_valid_annotation(self):
        ann = Annotation(
            image_id="img_001",
            expected_type=ImageType.RECEIPT,
            expected_entities={"merchant": "Starbucks", "total": 8.75},
            expected_group="receipts_trip",
            expected_failure_flags=[],
            notes="Clean receipt",
        )
        assert ann.image_id == "img_001"
        assert ann.expected_type == ImageType.RECEIPT
        assert ann.expected_entities["merchant"] == "Starbucks"

    def test_minimal_annotation(self):
        ann = Annotation(image_id="img_002", expected_type=ImageType.WHITEBOARD)
        assert ann.expected_entities == {}
        assert ann.expected_group is None
        assert ann.expected_failure_flags == []

    def test_annotation_with_failure_flags(self):
        ann = Annotation(
            image_id="img_003",
            expected_type=ImageType.RECEIPT,
            expected_failure_flags=[FailureFlag.BLURRY_IMAGE],
            notes="Blurry receipt",
        )
        assert FailureFlag.BLURRY_IMAGE in ann.expected_failure_flags

    def test_annotation_json_roundtrip(self):
        ann = Annotation(
            image_id="img_001",
            expected_type=ImageType.CONVERSATION,
            expected_calendar_hook=True,
            notes="Has meeting reference",
        )
        json_str = ann.model_dump_json()
        restored = Annotation.model_validate_json(json_str)
        assert restored.expected_calendar_hook is True
        assert restored.expected_type == ImageType.CONVERSATION
