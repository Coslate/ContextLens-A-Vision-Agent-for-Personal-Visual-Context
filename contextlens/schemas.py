"""Pydantic output schemas for ContextLens."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# --- Enums ---

class ImageType(str, Enum):
    RECEIPT = "receipt"
    CONVERSATION = "conversation"
    DOCUMENT = "document"
    WHITEBOARD = "whiteboard"
    UNKNOWN = "unknown"


class FailureFlag(str, Enum):
    BLURRY_IMAGE = "blurry_image"
    ROTATION_CORRECTED = "rotation_corrected"
    ROTATION_UNRESOLVED = "rotation_unresolved"
    PARTIAL_CAPTURE = "partial_capture"
    OCR_UNCERTAIN = "ocr_uncertain"
    MIXED_LANGUAGE = "mixed_language"


# --- Quality Signals ---

class QualitySignals(BaseModel):
    blur_score: float = Field(description="Laplacian variance; higher = sharper")
    brightness: float = Field(ge=0.0, le=1.0, description="Mean pixel intensity normalized")
    contrast: float = Field(ge=0.0, le=1.0, description="Std dev of pixel intensities normalized")
    estimated_quality: float = Field(ge=0.0, le=1.0, description="Overall quality score")
    is_blurry: bool = False
    is_rotated: bool = False


# --- OCR ---

class OCRSpan(BaseModel):
    text: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: list[float] = Field(description="Bounding box [x, y, w, h]")


class OCRResult(BaseModel):
    spans: list[OCRSpan] = Field(default_factory=list)
    raw_text: str = ""
    avg_confidence: float = 0.0


# --- Receipt Entities ---

class ReceiptItem(BaseModel):
    name: str
    price: float


class ReceiptEntities(BaseModel):
    merchant: Optional[str] = None
    items: list[ReceiptItem] = Field(default_factory=list)
    total: Optional[float] = None
    date: Optional[str] = None
    currency: Optional[str] = None


# --- Conversation Entities ---

class CalendarEventCandidate(BaseModel):
    title: Optional[str] = None
    time_mention: Optional[str] = None
    participants: list[str] = Field(default_factory=list)


class CalendarHook(BaseModel):
    mentioned: bool = False
    event_candidates: list[CalendarEventCandidate] = Field(default_factory=list)


class ConversationEntities(BaseModel):
    participants: list[str] = Field(default_factory=list)
    key_topics: list[str] = Field(default_factory=list)
    action_items: list[str] = Field(default_factory=list)
    referenced_events: list[CalendarEventCandidate] = Field(default_factory=list)


# --- Document Entities ---

class DocumentEntities(BaseModel):
    document_kind: Optional[str] = None  # form, invoice, label, generic
    structured_fields: dict[str, str] = Field(default_factory=dict)


# --- Whiteboard Entities ---

class TextBlock(BaseModel):
    text: str
    position: Optional[str] = None  # e.g. "top-left", "center"


class WhiteboardStructure(BaseModel):
    bullets: list[str] = Field(default_factory=list)
    owners: list[str] = Field(default_factory=list)
    dates: list[str] = Field(default_factory=list)
    tasks: list[str] = Field(default_factory=list)
    project_tags: list[str] = Field(default_factory=list)


class WhiteboardEntities(BaseModel):
    text_blocks: list[TextBlock] = Field(default_factory=list)
    inferred_structure: WhiteboardStructure = Field(default_factory=WhiteboardStructure)


# --- Main Output ---

class ImageOutput(BaseModel):
    image_id: str
    type: ImageType = ImageType.UNKNOWN
    type_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    extracted_entities: ReceiptEntities | ConversationEntities | DocumentEntities | WhiteboardEntities | dict = Field(default_factory=dict)
    field_confidence: dict[str, float] = Field(default_factory=dict)
    summary: str = ""
    failure_flags: list[FailureFlag] = Field(default_factory=list)
    needs_clarification: bool = False
    quality_signals: Optional[QualitySignals] = None
    raw_text: str = ""
    group_id: Optional[str] = None
    calendar_hook: Optional[CalendarHook] = None


# --- Annotation (Ground Truth) ---

class Annotation(BaseModel):
    image_id: str
    expected_type: ImageType
    expected_entities: dict = Field(default_factory=dict)
    expected_group: Optional[str] = None
    expected_calendar_hook: Optional[bool] = None
    expected_calendar_events: list[dict] = Field(
        default_factory=list,
        description="Expected calendar event candidates for evaluation "
        "(title, time_mention, participants).",
    )
    expected_needs_clarification: Optional[bool] = None
    expected_failure_flags: list[FailureFlag] = Field(default_factory=list)
    notes: str = ""
