"""End-to-end pipeline orchestration.

Chains preprocessing → OCR → classification → conditioned extraction →
confidence calibration → failure handling into a single ``process_image``
call that returns a fully populated ``ImageOutput``.
"""

from __future__ import annotations

from pathlib import Path

from contextlens.classifier import classify_image
from contextlens.confidence import calibrate_confidence_dict
from contextlens.extractors import (
    ConversationExtractor,
    DocumentExtractor,
    ReceiptExtractor,
    WhiteboardExtractor,
)
from contextlens.failure_handlers import apply_failure_handlers
from contextlens.ocr import run_ocr
from contextlens.preprocess import preprocess_image
from contextlens.schemas import (
    CalendarHook,
    ImageOutput,
    ImageType,
)

# ---------------------------------------------------------------------------
# Extractor registry
# ---------------------------------------------------------------------------

_EXTRACTORS = {
    ImageType.RECEIPT: ReceiptExtractor(),
    ImageType.CONVERSATION: ConversationExtractor(),
    ImageType.DOCUMENT: DocumentExtractor(),
    ImageType.WHITEBOARD: WhiteboardExtractor(),
}


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------

def _generate_summary(output: ImageOutput) -> str:
    """Generate a one-sentence summary from extracted entities."""
    etype = output.type
    entities = output.extracted_entities

    if etype == ImageType.RECEIPT:
        merchant = getattr(entities, "merchant", None) or "Unknown merchant"
        total = getattr(entities, "total", None)
        date = getattr(entities, "date", None)
        parts = [f"{merchant} receipt"]
        if total is not None:
            parts.append(f"for ${total:.2f}")
        if date:
            parts.append(f"on {date}")
        items = getattr(entities, "items", [])
        if items:
            names = [it.name for it in items[:3]]
            parts.append(f"({', '.join(names)})")
        return " ".join(parts) + "."

    if etype == ImageType.CONVERSATION:
        participants = getattr(entities, "participants", [])
        topics = getattr(entities, "key_topics", [])
        who = " and ".join(participants[:3]) if participants else "Unknown"
        about = ", ".join(topics[:3]) if topics else "general topics"
        return f"Conversation between {who} about {about}."

    if etype == ImageType.DOCUMENT:
        kind = getattr(entities, "document_kind", None) or "document"
        fields = getattr(entities, "structured_fields", {})
        n = len(fields)
        return f"{kind.capitalize()} with {n} extracted field{'s' if n != 1 else ''}."

    if etype == ImageType.WHITEBOARD:
        structure = getattr(entities, "inferred_structure", None)
        n_tasks = len(structure.tasks) if structure else 0
        tags = structure.project_tags if structure else []
        tag_str = ", ".join(tags[:2]) if tags else "untagged"
        return f"Whiteboard ({tag_str}): {n_tasks} task{'s' if n_tasks != 1 else ''}."

    return "Unclassified image."


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def process_image(path: str | Path, image_id: str | None = None) -> ImageOutput:
    """Run the full ContextLens pipeline on a single image.

    Args:
        path: Path to the image file.
        image_id: Optional image identifier.  Defaults to the filename stem.

    Returns:
        Fully populated ``ImageOutput``.
    """
    path = Path(path)
    if image_id is None:
        image_id = path.stem

    # 1. Preprocess — quality signals + optional rotation correction
    image_arr, quality = preprocess_image(str(path))

    # 2. OCR
    ocr_result = run_ocr(image_arr)

    # 3. Classify
    img_type, type_confidence = classify_image(ocr_result)

    # 4. Conditioned extraction (Q-Former analog)
    extractor = _EXTRACTORS.get(img_type)
    if extractor is not None:
        entities, raw_confidence, calendar_hook = extractor.extract(
            ocr_result, quality,
        )
    else:
        entities = {}
        raw_confidence = {}
        calendar_hook = None

    # 5. Confidence calibration
    calibrated = calibrate_confidence_dict(
        raw_confidence, quality=quality,
    )

    # 6. Failure handling
    adjusted, flags, needs_clarification = apply_failure_handlers(
        calibrated,
        quality=quality,
        avg_ocr_confidence=ocr_result.avg_confidence,
        image_type=img_type.value,
    )

    # 7. Assemble output
    output = ImageOutput(
        image_id=image_id,
        type=img_type,
        type_confidence=type_confidence,
        extracted_entities=entities,
        field_confidence=adjusted,
        failure_flags=flags,
        needs_clarification=needs_clarification,
        quality_signals=quality,
        raw_text=ocr_result.raw_text,
        calendar_hook=calendar_hook,
    )

    # 8. Generate summary
    output.summary = _generate_summary(output)

    return output


def process_batch(
    paths: list[str | Path],
    image_ids: list[str] | None = None,
) -> list[ImageOutput]:
    """Process multiple images through the pipeline.

    Args:
        paths: List of image file paths.
        image_ids: Optional list of identifiers (one per path).

    Returns:
        List of ``ImageOutput`` objects, one per image.
    """
    if image_ids is None:
        image_ids = [None] * len(paths)  # type: ignore[list-item]
    if len(paths) != len(image_ids):
        raise ValueError("paths and image_ids must have the same length")

    outputs: list[ImageOutput] = []
    for p, iid in zip(paths, image_ids):
        outputs.append(process_image(p, iid))
    return outputs
