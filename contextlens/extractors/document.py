"""Document photo extractor — conditioned on document-specific fields."""

from __future__ import annotations

import re

from contextlens.extractors.base import ConditionedExtractor
from contextlens.schemas import (
    CalendarHook,
    DocumentEntities,
    ImageType,
    OCRResult,
    QualitySignals,
)

# --- Document kind heuristics ---

_INVOICE_KEYWORDS = re.compile(
    r"\b(Invoice|Invoice\s*#|Invoice\s*Number|Bill\s*To|Ship\s*To|"
    r"Amount\s*Due|Payment\s*Due|PO\s*Number|Purchase\s*Order)\b",
    re.IGNORECASE,
)
_FORM_KEYWORDS = re.compile(
    r"\b(Patient|Date\s*of\s*Birth|DOB|Social\s*Security|SSN|"
    r"Signature|Applicant|Application|Registration|Consent)\b",
    re.IGNORECASE,
)
_MEDICATION_KEYWORDS = re.compile(
    r"\b(mg|dosage|prescription|Rx|tablet|capsule|medication|pharmacy)\b",
    re.IGNORECASE,
)
_LABEL_KEYWORDS = re.compile(
    r"\b(shipping|tracking|barcode|SKU|lot|batch|expiry|expiration)\b",
    re.IGNORECASE,
)

# Key-value extraction
_KV_PATTERN = re.compile(
    r"^([A-Za-z][\w\s/]{0,40}?)\s*:\s*(.+)$", re.MULTILINE
)
# Also match "Field  Value" with multiple spaces (common in scanned docs)
_KV_SPACED = re.compile(
    r"^([A-Za-z][\w\s]{1,25}?)\s{3,}(.{2,})$", re.MULTILINE
)

# Common field-specific patterns
_INVOICE_NUM_RE = re.compile(
    r"(?:Invoice|Inv|Ref)[\s#:]*([A-Z0-9\-]+)", re.IGNORECASE
)
_AMOUNT_RE = re.compile(r"\$\s?(\d[\d,]*\.?\d{0,2})")


class DocumentExtractor(ConditionedExtractor):
    """Extract document kind and structured key-value fields."""

    image_type = ImageType.DOCUMENT

    def extract(
        self,
        ocr: OCRResult,
        quality: QualitySignals | None = None,
    ) -> tuple[DocumentEntities, dict[str, float], None]:
        text = ocr.raw_text
        avg_conf = ocr.avg_confidence if ocr.avg_confidence else 0.8

        doc_kind, kind_conf = self._detect_kind(text)
        fields, fields_conf = self._extract_fields(text, avg_conf)

        field_confidence: dict[str, float] = {
            "document_kind": kind_conf,
            "structured_fields": fields_conf,
        }

        entities = DocumentEntities(
            document_kind=doc_kind,
            structured_fields=fields,
        )
        return entities, field_confidence, None

    def _detect_kind(self, text: str) -> tuple[str, float]:
        """Classify document sub-type."""
        scores = {
            "invoice": len(_INVOICE_KEYWORDS.findall(text)),
            "form": len(_FORM_KEYWORDS.findall(text)),
            "medication": len(_MEDICATION_KEYWORDS.findall(text)),
            "label": len(_LABEL_KEYWORDS.findall(text)),
        }
        best = max(scores, key=scores.get)  # type: ignore[arg-type]
        best_count = scores[best]

        if best_count == 0:
            return "generic", 0.4

        # Confidence scales with number of keyword hits
        conf = min(0.5 + best_count * 0.1, 0.95)
        return best, conf

    def _extract_fields(
        self, text: str, avg_conf: float
    ) -> tuple[dict[str, str], float]:
        """Extract key-value pairs from document text."""
        fields: dict[str, str] = {}

        # Primary: colon-separated patterns
        for match in _KV_PATTERN.finditer(text):
            key = match.group(1).strip()
            val = match.group(2).strip()
            if key and val and len(key) <= 40:
                fields[key] = val

        # Secondary: wide-space separated (lower priority, don't overwrite)
        for match in _KV_SPACED.finditer(text):
            key = match.group(1).strip()
            val = match.group(2).strip()
            if key and val and key not in fields and len(key) <= 40:
                fields[key] = val

        # Compute aggregate confidence
        if not fields:
            return fields, 0.0

        conf = min(avg_conf * 0.9, 0.9)
        return fields, conf
