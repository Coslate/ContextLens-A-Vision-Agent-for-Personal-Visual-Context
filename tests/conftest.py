"""Shared test fixtures for ContextLens."""

import pytest

from contextlens.schemas import (
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


@pytest.fixture
def sample_quality_signals():
    return QualitySignals(
        blur_score=850.3,
        brightness=0.72,
        contrast=0.65,
        estimated_quality=0.88,
        is_blurry=False,
        is_rotated=False,
    )


@pytest.fixture
def sample_ocr_result():
    return OCRResult(
        spans=[
            OCRSpan(text="STARBUCKS", confidence=0.95, bbox=[10, 10, 200, 30]),
            OCRSpan(text="Latte    $5.50", confidence=0.88, bbox=[10, 50, 200, 70]),
            OCRSpan(text="TOTAL    $5.50", confidence=0.92, bbox=[10, 90, 200, 110]),
            OCRSpan(text="03/15/2024", confidence=0.85, bbox=[10, 130, 200, 150]),
        ],
        raw_text="STARBUCKS\nLatte    $5.50\nTOTAL    $5.50\n03/15/2024",
        avg_confidence=0.90,
    )


@pytest.fixture
def sample_receipt_entities():
    return ReceiptEntities(
        merchant="Starbucks",
        items=[ReceiptItem(name="Latte", price=5.50)],
        total=5.50,
        date="03/15/2024",
        currency="USD",
    )


@pytest.fixture
def sample_conversation_entities():
    return ConversationEntities(
        participants=["Alice", "Bob"],
        key_topics=["meeting", "project update"],
        action_items=["Send report by Friday"],
        referenced_events=[
            CalendarEventCandidate(
                title="Team sync",
                time_mention="tomorrow 3pm",
                participants=["Alice", "Bob"],
            )
        ],
    )


@pytest.fixture
def sample_document_entities():
    return DocumentEntities(
        document_kind="form",
        structured_fields={"Name": "John Doe", "DOB": "01/01/1990"},
    )


@pytest.fixture
def sample_whiteboard_entities():
    return WhiteboardEntities(
        text_blocks=[
            TextBlock(text="Task 1: Design API", position="top-left"),
            TextBlock(text="Task 2: Write tests", position="center"),
        ],
        inferred_structure=WhiteboardStructure(
            bullets=["Design API", "Write tests"],
            owners=["Alice"],
            dates=["Friday"],
            tasks=["Design API", "Write tests"],
            project_tags=["ProjectAlpha"],
        ),
    )


@pytest.fixture
def sample_image_output(sample_quality_signals, sample_receipt_entities):
    return ImageOutput(
        image_id="img_001",
        type=ImageType.RECEIPT,
        type_confidence=0.92,
        extracted_entities=sample_receipt_entities,
        field_confidence={
            "merchant": 0.95,
            "items": 0.78,
            "total": 0.91,
            "date": 0.85,
            "currency": 0.70,
        },
        summary="Starbucks receipt for $5.50 on March 15, 2024: latte.",
        failure_flags=[],
        needs_clarification=False,
        quality_signals=sample_quality_signals,
        raw_text="STARBUCKS\nLatte    $5.50\nTOTAL    $5.50\n03/15/2024",
        group_id=None,
        calendar_hook=None,
    )
