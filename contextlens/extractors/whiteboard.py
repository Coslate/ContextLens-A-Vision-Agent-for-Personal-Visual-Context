"""Whiteboard / handwritten note extractor — conditioned on structure inference."""

from __future__ import annotations

import re

from contextlens.extractors.base import ConditionedExtractor
from contextlens.schemas import (
    CalendarHook,
    ImageType,
    OCRResult,
    QualitySignals,
    TextBlock,
    WhiteboardEntities,
    WhiteboardStructure,
)

# --- Patterns ---

_BULLET_RE = re.compile(r"^[\s]*[-*\u2022]\s*(.*)", re.MULTILINE)
_NUMBERED_RE = re.compile(r"^[\s]*\d+[.)]\s*(.*)", re.MULTILINE)
_OWNER_RE = re.compile(r"@(\w+)")
_HASHTAG_RE = re.compile(r"#(\w+)")
_DATE_RE = re.compile(
    r"\b((?:due|deadline|by)\s+\w+)"
    r"|(\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b)"
    r"|(\d{1,2}[/\-]\d{1,2}(?:[/\-]\d{2,4})?)"
    r"|(\b(?:today|tomorrow|next\s+\w+|this\s+\w+)\b)",
    re.IGNORECASE,
)
_TASK_RE = re.compile(
    r"\b(TODO|FIXME|HACK|task|action\s*item|assign|implement|design|"
    r"review|test|deploy|build|migrate|refactor|update|fix|create|"
    r"write|draft|prepare|setup|configure)\b",
    re.IGNORECASE,
)


class WhiteboardExtractor(ConditionedExtractor):
    """Extract text blocks and inferred structure from whiteboard images."""

    image_type = ImageType.WHITEBOARD

    def extract(
        self,
        ocr: OCRResult,
        quality: QualitySignals | None = None,
    ) -> tuple[WhiteboardEntities, dict[str, float], None]:
        text = ocr.raw_text
        avg_conf = ocr.avg_confidence if ocr.avg_confidence else 0.8

        text_blocks = self._extract_text_blocks(ocr)
        structure = self._infer_structure(text)

        # Confidence
        field_confidence: dict[str, float] = {
            "text_blocks": min(avg_conf, 0.9) if text_blocks else 0.0,
            "bullets": min(avg_conf * 0.85, 0.85) if structure.bullets else 0.0,
            "owners": min(avg_conf * 0.9, 0.9) if structure.owners else 0.0,
            "dates": min(avg_conf * 0.75, 0.75) if structure.dates else 0.0,
            "tasks": min(avg_conf * 0.8, 0.8) if structure.tasks else 0.0,
            "project_tags": min(avg_conf * 0.9, 0.9) if structure.project_tags else 0.0,
        }

        entities = WhiteboardEntities(
            text_blocks=text_blocks,
            inferred_structure=structure,
        )
        return entities, field_confidence, None

    def _extract_text_blocks(self, ocr: OCRResult) -> list[TextBlock]:
        """Convert OCR spans into positioned text blocks."""
        if not ocr.spans:
            return []

        blocks: list[TextBlock] = []
        for span in ocr.spans:
            if not span.text.strip():
                continue
            # Infer rough position from bbox
            position = self._infer_position(span.bbox)
            blocks.append(TextBlock(text=span.text.strip(), position=position))
        return blocks

    def _infer_position(self, bbox: list[float]) -> str:
        """Rough spatial position label from bounding box.

        Assumes a typical image of ~400-800px width/height.
        """
        if len(bbox) < 4:
            return "unknown"
        x, y = bbox[0], bbox[1]
        # Simple heuristic: divide into 3x3 grid
        col = "left" if x < 150 else ("center" if x < 400 else "right")
        row = "top" if y < 100 else ("middle" if y < 250 else "bottom")
        return f"{row}-{col}"

    def _infer_structure(self, text: str) -> WhiteboardStructure:
        """Infer structural elements from whiteboard text."""
        bullets = self._extract_bullets(text)
        owners = self._extract_owners(text)
        dates = self._extract_dates(text)
        tasks = self._extract_tasks(text)
        project_tags = self._extract_project_tags(text)

        return WhiteboardStructure(
            bullets=bullets,
            owners=owners,
            dates=dates,
            tasks=tasks,
            project_tags=project_tags,
        )

    def _extract_bullets(self, text: str) -> list[str]:
        """Extract bullet-point items."""
        items: list[str] = []
        for match in _BULLET_RE.finditer(text):
            content = match.group(1).strip()
            if content:
                items.append(content)
        for match in _NUMBERED_RE.finditer(text):
            content = match.group(1).strip()
            if content:
                items.append(content)
        return items

    def _extract_owners(self, text: str) -> list[str]:
        """Extract @mentions as owners."""
        seen: set[str] = set()
        owners: list[str] = []
        for match in _OWNER_RE.finditer(text):
            name = match.group(1)
            if name not in seen:
                seen.add(name)
                owners.append(name)
        return owners

    def _extract_dates(self, text: str) -> list[str]:
        """Extract date/deadline mentions."""
        dates: list[str] = []
        seen: set[str] = set()
        for match in _DATE_RE.finditer(text):
            date_str = match.group(0).strip()
            lower = date_str.lower()
            if lower not in seen:
                seen.add(lower)
                dates.append(date_str)
        return dates

    def _extract_tasks(self, text: str) -> list[str]:
        """Extract task-like lines (lines containing action keywords)."""
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        tasks: list[str] = []
        for line in lines:
            if _TASK_RE.search(line) and len(line) > 5:
                # Clean up bullet markers
                clean = re.sub(r"^[\s\-*\u2022\d.)]+\s*", "", line).strip()
                if clean and clean not in tasks:
                    tasks.append(clean)
        return tasks

    def _extract_project_tags(self, text: str) -> list[str]:
        """Extract #hashtags as project tags."""
        seen: set[str] = set()
        tags: list[str] = []
        for match in _HASHTAG_RE.finditer(text):
            tag = match.group(1)
            if tag not in seen:
                seen.add(tag)
                tags.append(tag)
        return tags
