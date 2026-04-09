"""Type-conditioned extractors (Q-Former analog)."""

from contextlens.extractors.receipt import ReceiptExtractor
from contextlens.extractors.conversation import ConversationExtractor
from contextlens.extractors.document import DocumentExtractor
from contextlens.extractors.whiteboard import WhiteboardExtractor

__all__ = [
    "ReceiptExtractor",
    "ConversationExtractor",
    "DocumentExtractor",
    "WhiteboardExtractor",
]
