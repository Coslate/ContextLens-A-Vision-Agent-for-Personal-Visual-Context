"""Abstract base class for conditioned extractors (Q-Former analog).

Each extractor is conditioned on an image type, defining which fields to
extract — analogous to how Q-Former uses learned queries to attend to
specific visual regions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from contextlens.schemas import (
    CalendarHook,
    ImageType,
    OCRResult,
    QualitySignals,
)


class ConditionedExtractor(ABC):
    """Abstract interface for type-conditioned extraction.

    Subclasses implement ``extract()`` to produce type-specific entities
    and per-field confidence scores from OCR output.
    """

    image_type: ImageType

    @abstractmethod
    def extract(
        self,
        ocr: OCRResult,
        quality: QualitySignals | None = None,
    ) -> tuple[object, dict[str, float], CalendarHook | None]:
        """Extract structured entities from OCR output.

        Args:
            ocr: OCRResult from the OCR engine.
            quality: Optional quality signals from preprocessing.

        Returns:
            Tuple of:
                - Typed entities object (ReceiptEntities, etc.)
                - Dict mapping field names to raw confidence scores (0-1)
                - Optional CalendarHook (only for conversation type)
        """
        ...
