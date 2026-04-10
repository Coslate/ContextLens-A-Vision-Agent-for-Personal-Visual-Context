"""Generic Unified Parser baseline — single-pass extraction for all image types.

No type routing, no confidence calibration, no failure handling, no cross-image
linking.  All images go through the same set of shared extraction rules
(amount regex, date regex, key:value, action verbs, name detection, text
blocks).  Flat 0.8 confidence for every field.

This serves as a credible baseline to compare against the structured
ContextLens pipeline.
"""

from __future__ import annotations

import re

from contextlens.config import BASELINE_CONFIDENCE
from contextlens.schemas import (
    CalendarHook,
    ConversationEntities,
    DocumentEntities,
    ImageOutput,
    ImageType,
    OCRResult,
    QualitySignals,
    ReceiptEntities,
    ReceiptItem,
    TextBlock,
    WhiteboardEntities,
    WhiteboardStructure,
)

# ---------------------------------------------------------------------------
# Shared extraction regexes (type-agnostic)
# ---------------------------------------------------------------------------

_AMOUNT_RE = re.compile(r"\$\s?(\d+\.?\d{0,2})")
_DATE_RE = re.compile(
    r"(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})"
    r"|(\w{3,9}\s+\d{1,2},?\s+\d{4})"
    r"|(\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2})"
)
_KV_RE = re.compile(r"^([A-Za-z][\w\s/]{0,40}?)\s*:\s*(.+)$", re.MULTILINE)
_ACTION_VERB_RE = re.compile(
    r"\b(send|review|schedule|prepare|update|submit|complete|finish|book|"
    r"arrange|plan|fix|deploy|test|deliver|share|create|write|draft|check|"
    r"confirm)\b",
    re.IGNORECASE,
)
_NAME_RE = re.compile(r"\b([A-Z][a-z]{2,15})(?:\s+[A-Z][a-z]{2,15})?\b")

# Common stop words to filter out of name detection
_NAME_STOP = {
    "the", "and", "for", "with", "from", "that", "this", "have", "has",
    "are", "was", "were", "been", "being", "will", "would", "could",
    "should", "may", "might", "shall", "can", "not", "but", "its",
    "total", "subtotal", "tax", "date", "price", "amount", "balance",
    "change", "cash", "credit", "debit", "item", "items", "order",
    "receipt", "invoice", "form", "name", "address", "phone", "email",
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "sunday", "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "today", "tomorrow", "yesterday", "due", "latte", "muffin",
    "design", "implement", "review", "test", "deploy", "build",
    "migrate", "refactor", "update", "fix", "create", "write", "draft",
    "prepare", "setup", "configure", "task", "action", "send", "share",
    "check", "confirm", "book", "arrange", "plan", "deliver", "submit",
    "complete", "finish", "schedule",
}


# ---------------------------------------------------------------------------
# Generic extraction (one function for ALL image types)
# ---------------------------------------------------------------------------


def generic_extract(ocr: OCRResult) -> dict:
    """Apply shared extraction rules to OCR output regardless of image type.

    Returns a flat dict with all extracted fields:
      - amounts: list of floats
      - total_guess: largest amount (or None)
      - dates: list of date strings
      - structured_fields: dict of key:value pairs
      - action_items: list of action-verb sentences
      - people: list of name-like tokens
      - text_blocks: list of all OCR text blocks
      - merchant_guess: first prominent line
    """
    text = ocr.raw_text
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]

    # 1. Amount regex
    amounts: list[float] = []
    for match in _AMOUNT_RE.finditer(text):
        try:
            amounts.append(float(match.group(1)))
        except ValueError:
            pass

    total_guess = max(amounts) if amounts else None

    # 2. Date regex
    dates: list[str] = []
    seen_dates: set[str] = set()
    for match in _DATE_RE.finditer(text):
        date_str = match.group(0).strip()
        if date_str not in seen_dates:
            seen_dates.add(date_str)
            dates.append(date_str)

    # 3. Key:Value extraction
    structured_fields: dict[str, str] = {}
    for match in _KV_RE.finditer(text):
        key = match.group(1).strip()
        val = match.group(2).strip()
        if key and val and len(key) <= 40:
            structured_fields[key] = val

    # 4. Action verb scanning
    action_items: list[str] = []
    for line in lines:
        if _ACTION_VERB_RE.search(line) and len(line) > 10:
            action_items.append(line)

    # 5. Name-like token detection
    people: list[str] = []
    seen_names: set[str] = set()
    for match in _NAME_RE.finditer(text):
        name = match.group(0).strip()
        if name.lower() not in _NAME_STOP and name not in seen_names:
            seen_names.add(name)
            people.append(name)

    # 6. Text block collection
    text_blocks = lines[:]

    # 7. First prominent line → merchant/title guess
    merchant_guess: str | None = None
    for line in lines[:3]:
        if len(line) >= 2 and not _AMOUNT_RE.fullmatch(line.strip()):
            merchant_guess = line
            break

    return {
        "amounts": amounts,
        "total_guess": total_guess,
        "dates": dates,
        "structured_fields": structured_fields,
        "action_items": action_items,
        "people": people,
        "text_blocks": text_blocks,
        "merchant_guess": merchant_guess,
    }


# ---------------------------------------------------------------------------
# Type inference from extracted fields (output-based, not input-based)
# ---------------------------------------------------------------------------


def _infer_type(extracted: dict) -> ImageType:
    """Infer image type from which fields are populated.

    This is the opposite of the structured pipeline: instead of classifying
    the image first and then extracting type-specific fields, we extract
    generically and then guess the type from what we found.
    """
    has_amounts = bool(extracted["amounts"])
    has_people = bool(extracted["people"])
    has_kv = bool(extracted["structured_fields"])
    has_actions = bool(extracted["action_items"])

    # If amounts present and more than 1 → likely a receipt
    if has_amounts and len(extracted["amounts"]) >= 2:
        return ImageType.RECEIPT
    # Single amount with a merchant-looking first line → receipt
    if has_amounts and extracted["merchant_guess"]:
        return ImageType.RECEIPT

    # If people found and action items → likely conversation
    if has_people and has_actions:
        return ImageType.CONVERSATION

    # If structured key:value pairs → likely document
    if has_kv and len(extracted["structured_fields"]) >= 2:
        return ImageType.DOCUMENT

    # Fallback: if any kv → document, otherwise whiteboard (catch-all)
    if has_kv:
        return ImageType.DOCUMENT

    return ImageType.WHITEBOARD


# ---------------------------------------------------------------------------
# Map generic extraction to typed entities
# ---------------------------------------------------------------------------


def _to_receipt_entities(extracted: dict) -> ReceiptEntities:
    """Map generic extraction to ReceiptEntities."""
    items: list[ReceiptItem] = []
    total_guess = extracted["total_guess"]
    for amt in extracted["amounts"]:
        if amt != total_guess:
            items.append(ReceiptItem(name="item", price=amt))

    return ReceiptEntities(
        merchant=extracted["merchant_guess"],
        items=items,
        total=total_guess,
        date=extracted["dates"][0] if extracted["dates"] else None,
        currency="USD" if extracted["amounts"] else None,
    )


def _to_conversation_entities(extracted: dict) -> ConversationEntities:
    """Map generic extraction to ConversationEntities."""
    return ConversationEntities(
        participants=extracted["people"],
        key_topics=[],  # no topic extraction in baseline
        action_items=extracted["action_items"],
        referenced_events=[],  # no calendar detection in baseline
    )


def _to_document_entities(extracted: dict) -> DocumentEntities:
    """Map generic extraction to DocumentEntities."""
    return DocumentEntities(
        document_kind="generic",
        structured_fields=extracted["structured_fields"],
    )


def _to_whiteboard_entities(extracted: dict) -> WhiteboardEntities:
    """Map generic extraction to WhiteboardEntities."""
    blocks = [TextBlock(text=t) for t in extracted["text_blocks"]]
    return WhiteboardEntities(
        text_blocks=blocks,
        inferred_structure=WhiteboardStructure(),  # no structure inference
    )


_ENTITY_BUILDERS = {
    ImageType.RECEIPT: _to_receipt_entities,
    ImageType.CONVERSATION: _to_conversation_entities,
    ImageType.DOCUMENT: _to_document_entities,
    ImageType.WHITEBOARD: _to_whiteboard_entities,
}


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------


def _generate_summary(img_type: ImageType, entities: object, raw_text: str) -> str:
    """Generate a simple one-sentence summary."""
    if img_type == ImageType.RECEIPT:
        ent = entities  # type: ignore[assignment]
        merchant = getattr(ent, "merchant", None) or "Unknown"
        total = getattr(ent, "total", None)
        parts = [f"{merchant} receipt"]
        if total is not None:
            parts.append(f"for ${total:.2f}")
        return " ".join(parts) + "."

    if img_type == ImageType.CONVERSATION:
        ent = entities  # type: ignore[assignment]
        people = getattr(ent, "participants", [])
        who = ", ".join(people[:3]) if people else "Unknown"
        return f"Conversation with {who}."

    if img_type == ImageType.DOCUMENT:
        ent = entities  # type: ignore[assignment]
        n = len(getattr(ent, "structured_fields", {}))
        return f"Document with {n} extracted field{'s' if n != 1 else ''}."

    if img_type == ImageType.WHITEBOARD:
        ent = entities  # type: ignore[assignment]
        n = len(getattr(ent, "text_blocks", []))
        return f"Whiteboard with {n} text block{'s' if n != 1 else ''}."

    return "Unclassified image."


# ---------------------------------------------------------------------------
# Field confidence (flat)
# ---------------------------------------------------------------------------


def _flat_confidence(entities: object, img_type: ImageType) -> dict[str, float]:
    """Assign flat BASELINE_CONFIDENCE to all non-empty fields."""
    conf: dict[str, float] = {}

    if img_type == ImageType.RECEIPT:
        for field in ("merchant", "total", "date", "currency", "items"):
            val = getattr(entities, field, None)
            if val is not None and val != [] and val != "":
                conf[field] = BASELINE_CONFIDENCE
    elif img_type == ImageType.CONVERSATION:
        for field in ("participants", "key_topics", "action_items"):
            val = getattr(entities, field, None)
            if val:
                conf[field] = BASELINE_CONFIDENCE
    elif img_type == ImageType.DOCUMENT:
        for field in ("document_kind", "structured_fields"):
            val = getattr(entities, field, None)
            if val is not None and val != {} and val != "":
                conf[field] = BASELINE_CONFIDENCE
    elif img_type == ImageType.WHITEBOARD:
        for field in ("text_blocks",):
            val = getattr(entities, field, None)
            if val:
                conf[field] = BASELINE_CONFIDENCE

    return conf


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def baseline_process(
    ocr: OCRResult,
    image_id: str = "unknown",
    quality: QualitySignals | None = None,
) -> ImageOutput:
    """Process OCR output through the generic unified parser baseline.

    This is the single entry point. No type routing, no confidence
    calibration, no failure handling.

    Args:
        ocr: OCRResult from the OCR engine.
        image_id: Identifier for the image.
        quality: Optional quality signals (stored but not used for calibration).

    Returns:
        ImageOutput with inferred type and flat confidence.
    """
    # Step 1: Generic extraction (same rules for ALL images)
    extracted = generic_extract(ocr)

    # Step 2: Infer type from output (not from input classification)
    img_type = _infer_type(extracted)

    # Step 3: Map to typed entities
    builder = _ENTITY_BUILDERS[img_type]
    entities = builder(extracted)

    # Step 4: Flat confidence
    field_confidence = _flat_confidence(entities, img_type)

    # Step 5: Summary
    summary = _generate_summary(img_type, entities, ocr.raw_text)

    return ImageOutput(
        image_id=image_id,
        type=img_type,
        type_confidence=BASELINE_CONFIDENCE,
        extracted_entities=entities,
        field_confidence=field_confidence,
        summary=summary,
        failure_flags=[],           # NO failure handling
        needs_clarification=False,  # never flags
        quality_signals=quality,    # stored but not used
        raw_text=ocr.raw_text,
        group_id=None,              # NO cross-image linking
        calendar_hook=None,         # NO calendar detection
    )
