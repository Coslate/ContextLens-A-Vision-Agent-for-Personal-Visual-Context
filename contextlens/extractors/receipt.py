"""Receipt extractor — conditioned on receipt-specific fields."""

from __future__ import annotations

import re

from contextlens.extractors.base import ConditionedExtractor
from contextlens.schemas import (
    CalendarHook,
    ImageType,
    OCRResult,
    QualitySignals,
    ReceiptEntities,
    ReceiptItem,
)

# --- Patterns ---

_PRICE_RE = re.compile(r"\$\s?(\d+\.?\d{0,2})")
_TOTAL_KEYWORDS = re.compile(
    r"\b(GRAND\s*TOTAL|TOTAL|AMOUNT\s*DUE|BALANCE\s*DUE)\b", re.IGNORECASE
)
_SUBTOTAL_KEYWORDS = re.compile(
    r"\b(SUBTOTAL|SUB\s*TOTAL)\b", re.IGNORECASE
)
_TAX_KEYWORDS = re.compile(r"\b(TAX|HST|GST|VAT)\b", re.IGNORECASE)
_TIP_KEYWORDS = re.compile(r"\b(TIP|GRATUITY)\b", re.IGNORECASE)
_EXCLUDE_KEYWORDS = re.compile(
    r"\b(SUBTOTAL|SUB\s*TOTAL|TAX|HST|GST|VAT|TIP|GRATUITY|"
    r"TOTAL|GRAND\s*TOTAL|AMOUNT\s*DUE|BALANCE\s*DUE|CHANGE|"
    r"CASH|CREDIT|DEBIT|VISA|MASTERCARD|AMEX)\b",
    re.IGNORECASE,
)
_DATE_RE = re.compile(
    r"(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})"
    r"|(\w{3,9}\s+\d{1,2},?\s+\d{4})"
    r"|(\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2})"
)
_CURRENCY_RE = re.compile(r"(\$|USD|EUR|€|£|GBP|CAD|AUD)")


class ReceiptExtractor(ConditionedExtractor):
    """Extract merchant, items, total, date, currency from receipt OCR."""

    image_type = ImageType.RECEIPT

    @staticmethod
    def _merge_span_lines(ocr: OCRResult) -> list[str]:
        """Merge horizontally-adjacent OCR spans into visual lines.

        EasyOCR often emits separate spans for text and price on the same
        row (e.g. "Latte" and "$5.50").  This groups spans by vertical
        position and joins them so downstream extraction sees full lines.
        """
        if not ocr.spans:
            return [ln.strip() for ln in ocr.raw_text.split("\n") if ln.strip()]

        # Sort spans by y-center first, then group by proximity
        sorted_spans = sorted(ocr.spans, key=lambda s: s.bbox[1] + s.bbox[3] / 2)

        # Group spans by approximate y-center (within tight threshold)
        rows: list[list[tuple[float, float, str]]] = []  # [(x, y_center, text)]
        for span in sorted_spans:
            x, y, _w, h = span.bbox
            y_center = y + h / 2
            placed = False
            for row in rows:
                # Average y-center of existing row members
                avg_yc = sum(yc for _, yc, _ in row) / len(row)
                if abs(avg_yc - y_center) < max(h * 0.35, 5):
                    row.append((x, y_center, span.text))
                    placed = True
                    break
            if not placed:
                rows.append([(x, y_center, span.text)])

        # Sort rows top-to-bottom, spans left-to-right within each row
        rows.sort(key=lambda r: min(yc for _, yc, _ in r))
        merged: list[str] = []
        for row in rows:
            row.sort(key=lambda t: t[0])
            line = "  ".join(text for _, _, text in row).strip()
            if line:
                merged.append(line)
        return merged

    def extract(
        self,
        ocr: OCRResult,
        quality: QualitySignals | None = None,
    ) -> tuple[ReceiptEntities, dict[str, float], None]:
        lines = self._merge_span_lines(ocr)
        span_conf = {s.text: s.confidence for s in ocr.spans} if ocr.spans else {}

        merchant = self._extract_merchant(lines, span_conf)
        total, total_conf = self._extract_total(lines, span_conf)
        items = self._extract_items(lines, span_conf)
        date, date_conf = self._extract_date(ocr.raw_text, span_conf)
        currency, currency_conf = self._extract_currency(ocr.raw_text)

        # Build confidence dict
        field_confidence: dict[str, float] = {}
        field_confidence["merchant"] = merchant[1] if merchant[0] else 0.0
        field_confidence["total"] = total_conf
        field_confidence["items"] = (
            sum(c for _, c in items) / len(items) if items else 0.0
        )
        field_confidence["date"] = date_conf
        field_confidence["currency"] = currency_conf

        entities = ReceiptEntities(
            merchant=merchant[0],
            items=[ReceiptItem(name=name, price=price) for (name, price), _ in items],
            total=total,
            date=date,
            currency=currency,
        )
        return entities, field_confidence, None

    def _extract_merchant(
        self, lines: list[str], span_conf: dict[str, float]
    ) -> tuple[str | None, float]:
        """First prominent line that isn't a price/keyword line → merchant."""
        for line in lines[:3]:
            # Skip lines that are mostly prices or summary keywords
            if _PRICE_RE.search(line) and _EXCLUDE_KEYWORDS.search(line):
                continue
            if _DATE_RE.search(line):
                continue
            text = line.strip()
            if len(text) >= 2:
                conf = self._best_span_conf(text, span_conf)
                return text, conf
        return None, 0.0

    def _extract_total(
        self, lines: list[str], span_conf: dict[str, float]
    ) -> tuple[float | None, float]:
        """Find TOTAL keyword + adjacent price."""
        # Search from bottom up for TOTAL (not SUBTOTAL)
        for line in reversed(lines):
            if _TOTAL_KEYWORDS.search(line) and not _SUBTOTAL_KEYWORDS.search(line):
                price_match = _PRICE_RE.search(line)
                if price_match:
                    conf = self._best_span_conf(line, span_conf)
                    return float(price_match.group(1)), conf * 0.95
        # Fallback: last line with a price
        for line in reversed(lines):
            price_match = _PRICE_RE.search(line)
            if price_match and not _TAX_KEYWORDS.search(line) and not _TIP_KEYWORDS.search(line):
                conf = self._best_span_conf(line, span_conf)
                return float(price_match.group(1)), conf * 0.6  # lower confidence
        return None, 0.0

    def _extract_items(
        self, lines: list[str], span_conf: dict[str, float]
    ) -> list[tuple[tuple[str, float], float]]:
        """Lines with text + price that aren't summary rows → items."""
        items: list[tuple[tuple[str, float], float]] = []
        for line in lines:
            if _EXCLUDE_KEYWORDS.search(line):
                continue
            price_match = _PRICE_RE.search(line)
            if price_match:
                name = line[:price_match.start()].strip()
                if not name:
                    continue
                price = float(price_match.group(1))
                conf = self._best_span_conf(line, span_conf)
                items.append(((name, price), conf))
        return items

    def _extract_date(
        self, text: str, span_conf: dict[str, float]
    ) -> tuple[str | None, float]:
        """Find date patterns in text."""
        match = _DATE_RE.search(text)
        if match:
            date_str = match.group(0).strip()
            conf = self._best_span_conf(date_str, span_conf)
            return date_str, max(conf, 0.7)
        return None, 0.0

    def _extract_currency(self, text: str) -> tuple[str | None, float]:
        """Detect currency from symbols."""
        match = _CURRENCY_RE.search(text)
        if match:
            symbol = match.group(1)
            currency_map = {
                "$": "USD", "USD": "USD", "EUR": "EUR",
                "€": "EUR", "£": "GBP", "GBP": "GBP",
                "CAD": "CAD", "AUD": "AUD",
            }
            return currency_map.get(symbol, symbol), 0.7
        return None, 0.0

    @staticmethod
    def _best_span_conf(text: str, span_conf: dict[str, float]) -> float:
        """Find the best matching span confidence for a text fragment."""
        if text in span_conf:
            return span_conf[text]
        # Partial match
        for span_text, conf in span_conf.items():
            if text in span_text or span_text in text:
                return conf
        return 0.8  # default fallback
