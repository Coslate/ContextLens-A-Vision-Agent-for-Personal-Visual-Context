"""Tests for conditioned extractors (Q-Former analog) — PR5."""

import pytest

from contextlens.extractors.receipt import ReceiptExtractor
from contextlens.extractors.conversation import ConversationExtractor
from contextlens.extractors.document import DocumentExtractor
from contextlens.extractors.whiteboard import WhiteboardExtractor
from contextlens.schemas import (
    CalendarHook,
    ConversationEntities,
    DocumentEntities,
    ImageType,
    OCRResult,
    OCRSpan,
    ReceiptEntities,
    ReceiptItem,
    WhiteboardEntities,
)


# --- Helpers ---

def make_ocr(raw_text: str, avg_confidence: float = 0.9) -> OCRResult:
    """Build an OCRResult from raw text with one span per line."""
    lines = [ln for ln in raw_text.split("\n") if ln.strip()]
    spans = [
        OCRSpan(text=ln.strip(), confidence=avg_confidence, bbox=[0, i * 30, 200, 20])
        for i, ln in enumerate(lines)
    ]
    return OCRResult(spans=spans, raw_text=raw_text, avg_confidence=avg_confidence)


# =====================================================================
# Receipt Extractor
# =====================================================================

class TestReceiptExtractor:
    def setup_method(self):
        self.extractor = ReceiptExtractor()

    def test_image_type(self):
        assert self.extractor.image_type == ImageType.RECEIPT

    def test_basic_receipt(self):
        ocr = make_ocr(
            "STARBUCKS\n"
            "Latte    $5.50\n"
            "Muffin   $3.25\n"
            "SUBTOTAL $8.75\n"
            "TAX      $0.70\n"
            "TOTAL    $9.45\n"
            "03/15/2024"
        )
        entities, conf, hook = self.extractor.extract(ocr)

        assert isinstance(entities, ReceiptEntities)
        assert entities.merchant == "STARBUCKS"
        assert entities.total == 9.45
        assert len(entities.items) == 2
        assert entities.items[0].name == "Latte"
        assert entities.items[0].price == 5.50
        assert entities.items[1].name == "Muffin"
        assert entities.items[1].price == 3.25
        assert entities.date is not None
        assert "03/15/2024" in entities.date
        assert hook is None

    def test_currency_detection(self):
        ocr = make_ocr("STORE\nItem $10.00\nTOTAL $10.00")
        entities, conf, _ = self.extractor.extract(ocr)
        assert entities.currency == "USD"
        assert conf["currency"] > 0

    def test_merchant_is_first_line(self):
        ocr = make_ocr("WALMART\nMilk $3.99\nTOTAL $3.99")
        entities, _, _ = self.extractor.extract(ocr)
        assert entities.merchant == "WALMART"

    def test_excludes_summary_lines_from_items(self):
        ocr = make_ocr(
            "SHOP\n"
            "Coffee $4.00\n"
            "SUBTOTAL $4.00\n"
            "TAX $0.32\n"
            "TOTAL $4.32"
        )
        entities, _, _ = self.extractor.extract(ocr)
        item_names = [i.name for i in entities.items]
        assert "Coffee" in item_names
        # SUBTOTAL, TAX, TOTAL should not appear as items
        for name in item_names:
            assert "SUBTOTAL" not in name.upper()
            assert "TAX" not in name.upper()
            assert "TOTAL" not in name.upper()

    def test_confidence_keys(self):
        ocr = make_ocr("STORE\nItem $5.00\nTOTAL $5.00\n01/01/2024")
        _, conf, _ = self.extractor.extract(ocr)
        assert "merchant" in conf
        assert "total" in conf
        assert "items" in conf
        assert "date" in conf
        assert "currency" in conf
        for v in conf.values():
            assert 0.0 <= v <= 1.0

    def test_empty_ocr(self):
        ocr = OCRResult(spans=[], raw_text="", avg_confidence=0.0)
        entities, conf, _ = self.extractor.extract(ocr)
        assert entities.merchant is None
        assert entities.total is None
        assert entities.items == []

    def test_no_total_keyword(self):
        """Receipt with prices but no TOTAL keyword → fallback extraction."""
        ocr = make_ocr("CAFE\nLatte $5.00\nCookie $3.00")
        entities, conf, _ = self.extractor.extract(ocr)
        # Should still find some price as total (fallback)
        assert entities.total is not None
        # Confidence should be lower for fallback
        assert conf["total"] < 0.9

    def test_date_extraction_various_formats(self):
        ocr = make_ocr("STORE\nItem $5\nTOTAL $5\n2024-03-15")
        entities, _, _ = self.extractor.extract(ocr)
        assert entities.date is not None


# =====================================================================
# Conversation Extractor
# =====================================================================

class TestConversationExtractor:
    def setup_method(self):
        self.extractor = ConversationExtractor()

    def test_image_type(self):
        assert self.extractor.image_type == ImageType.CONVERSATION

    def test_basic_conversation(self):
        ocr = make_ocr(
            "Alice: Hey, let's meet tomorrow at 3pm\n"
            "Bob: Sounds good, I'll prepare the slides\n"
            "Alice: Great, see you then"
        )
        entities, conf, hook = self.extractor.extract(ocr)

        assert isinstance(entities, ConversationEntities)
        assert "Alice" in entities.participants
        assert "Bob" in entities.participants
        assert len(entities.participants) == 2

    def test_action_items_extracted(self):
        ocr = make_ocr(
            "Alice: Can you send the report by Friday?\n"
            "Bob: Sure, I'll prepare the presentation too\n"
            "Alice: Great, also schedule a call with the team"
        )
        entities, _, _ = self.extractor.extract(ocr)
        assert len(entities.action_items) > 0
        # Should contain action verb matches
        combined = " ".join(entities.action_items).lower()
        assert any(w in combined for w in ["send", "prepare", "schedule"])

    def test_calendar_hook_with_meeting(self):
        ocr = make_ocr(
            "Alice: Let's schedule a meeting for tomorrow\n"
            "Bob: Sounds good, 3pm works for me"
        )
        _, _, hook = self.extractor.extract(ocr)
        assert hook is not None
        assert isinstance(hook, CalendarHook)
        assert hook.mentioned is True
        assert len(hook.event_candidates) >= 1
        assert hook.event_candidates[0].title == "meeting"

    def test_no_calendar_hook_without_events(self):
        ocr = make_ocr(
            "Alice: How's the weather?\n"
            "Bob: Pretty nice today\n"
            "Alice: Great, enjoy your day"
        )
        _, _, hook = self.extractor.extract(ocr)
        # No event keywords → no hook
        assert hook is None

    def test_topics_extracted(self):
        ocr = make_ocr(
            "Alice: We need to review the design before the meeting\n"
            "Bob: I'll prepare my notes and send them over"
        )
        entities, _, _ = self.extractor.extract(ocr)
        assert len(entities.key_topics) > 0

    def test_confidence_keys(self):
        ocr = make_ocr("Alice: Hello\nBob: Hi")
        _, conf, _ = self.extractor.extract(ocr)
        assert "participants" in conf
        assert "key_topics" in conf
        assert "action_items" in conf
        assert "referenced_events" in conf

    def test_empty_ocr(self):
        ocr = OCRResult(spans=[], raw_text="", avg_confidence=0.0)
        entities, conf, hook = self.extractor.extract(ocr)
        assert entities.participants == []
        assert hook is None

    def test_time_mention_in_event(self):
        ocr = make_ocr(
            "Alice: The sync is tomorrow at 2pm\n"
            "Bob: I'll be there"
        )
        _, _, hook = self.extractor.extract(ocr)
        assert hook is not None
        assert hook.event_candidates[0].time_mention is not None


# =====================================================================
# Document Extractor
# =====================================================================

class TestDocumentExtractor:
    def setup_method(self):
        self.extractor = DocumentExtractor()

    def test_image_type(self):
        assert self.extractor.image_type == ImageType.DOCUMENT

    def test_structured_form(self):
        ocr = make_ocr(
            "Patient Name: John Doe\n"
            "Date of Birth: 01/15/1985\n"
            "Phone: 555-0123\n"
            "Address: 123 Main Street\n"
            "Insurance ID: INS-98765"
        )
        entities, conf, hook = self.extractor.extract(ocr)

        assert isinstance(entities, DocumentEntities)
        assert entities.document_kind == "form"
        assert "Patient Name" in entities.structured_fields
        assert entities.structured_fields["Patient Name"] == "John Doe"
        assert "Phone" in entities.structured_fields
        assert hook is None

    def test_invoice_detection(self):
        ocr = make_ocr(
            "Invoice Number: INV-2024-001\n"
            "Date: March 15, 2024\n"
            "Bill To: Acme Corp\n"
            "Amount Due: $1,250.00"
        )
        entities, conf, _ = self.extractor.extract(ocr)
        assert entities.document_kind == "invoice"
        assert conf["document_kind"] > 0.5

    def test_generic_document(self):
        ocr = make_ocr(
            "Title: Annual Report\n"
            "Author: Jane Smith\n"
            "Year: 2024"
        )
        entities, _, _ = self.extractor.extract(ocr)
        # Should detect key-value pairs even without specific doc kind
        assert len(entities.structured_fields) >= 2

    def test_key_value_extraction(self):
        ocr = make_ocr(
            "Name: Sarah Connor\n"
            "ID: 12345\n"
            "Department: Engineering\n"
            "Role: Senior Developer"
        )
        entities, _, _ = self.extractor.extract(ocr)
        assert entities.structured_fields["Name"] == "Sarah Connor"
        assert entities.structured_fields["ID"] == "12345"
        assert entities.structured_fields["Department"] == "Engineering"

    def test_confidence_keys(self):
        ocr = make_ocr("Name: Test\nID: 123")
        _, conf, _ = self.extractor.extract(ocr)
        assert "document_kind" in conf
        assert "structured_fields" in conf

    def test_empty_ocr(self):
        ocr = OCRResult(spans=[], raw_text="", avg_confidence=0.0)
        entities, conf, _ = self.extractor.extract(ocr)
        assert entities.structured_fields == {}
        assert conf["structured_fields"] == 0.0

    def test_medication_document(self):
        ocr = make_ocr(
            "Medication: Amoxicillin 500mg\n"
            "Dosage: 1 tablet twice daily\n"
            "Prescription: Rx-45678\n"
            "Pharmacy: CVS Health"
        )
        entities, _, _ = self.extractor.extract(ocr)
        assert entities.document_kind == "medication"


# =====================================================================
# Whiteboard Extractor
# =====================================================================

class TestWhiteboardExtractor:
    def setup_method(self):
        self.extractor = WhiteboardExtractor()

    def test_image_type(self):
        assert self.extractor.image_type == ImageType.WHITEBOARD

    def test_basic_whiteboard(self):
        ocr = make_ocr(
            "- Design API endpoints @Alice\n"
            "- Write unit tests @Bob\n"
            "- Deploy to staging\n"
            "#ProjectAlpha #Sprint5\n"
            "TODO: update documentation\n"
            "due Friday"
        )
        entities, conf, hook = self.extractor.extract(ocr)

        assert isinstance(entities, WhiteboardEntities)
        assert len(entities.text_blocks) > 0
        assert "Alice" in entities.inferred_structure.owners
        assert "Bob" in entities.inferred_structure.owners
        assert "ProjectAlpha" in entities.inferred_structure.project_tags
        assert "Sprint5" in entities.inferred_structure.project_tags
        assert len(entities.inferred_structure.bullets) >= 3
        assert len(entities.inferred_structure.tasks) > 0
        assert len(entities.inferred_structure.dates) > 0
        assert hook is None

    def test_bullets_extracted(self):
        ocr = make_ocr(
            "- First item\n"
            "- Second item\n"
            "* Third item\n"
            "1) Fourth item"
        )
        entities, _, _ = self.extractor.extract(ocr)
        bullets = entities.inferred_structure.bullets
        assert len(bullets) >= 3

    def test_owners_extracted(self):
        ocr = make_ocr("Task 1 @Alice\nTask 2 @Bob\nTask 3 @Alice")
        entities, _, _ = self.extractor.extract(ocr)
        owners = entities.inferred_structure.owners
        assert "Alice" in owners
        assert "Bob" in owners
        # No duplicates
        assert len(owners) == 2

    def test_project_tags(self):
        ocr = make_ocr("#ProjectBeta #v2 #backend")
        entities, _, _ = self.extractor.extract(ocr)
        tags = entities.inferred_structure.project_tags
        assert "ProjectBeta" in tags
        assert "v2" in tags
        assert "backend" in tags

    def test_tasks_extracted(self):
        ocr = make_ocr(
            "- TODO fix login bug\n"
            "- review PR #42\n"
            "- deploy new version\n"
            "random note here"
        )
        entities, _, _ = self.extractor.extract(ocr)
        tasks = entities.inferred_structure.tasks
        assert len(tasks) >= 2

    def test_dates_extracted(self):
        ocr = make_ocr(
            "Sprint ends Friday\n"
            "Deadline: due Monday\n"
            "Launch tomorrow"
        )
        entities, _, _ = self.extractor.extract(ocr)
        dates = entities.inferred_structure.dates
        assert len(dates) >= 2

    def test_text_blocks_have_positions(self):
        ocr = make_ocr("Block A\nBlock B\nBlock C")
        entities, _, _ = self.extractor.extract(ocr)
        for block in entities.text_blocks:
            assert block.position is not None
            assert block.position != "unknown"

    def test_confidence_keys(self):
        ocr = make_ocr("- item @owner #tag\nTODO: task\ndue Friday")
        _, conf, _ = self.extractor.extract(ocr)
        assert "text_blocks" in conf
        assert "bullets" in conf
        assert "owners" in conf
        assert "tasks" in conf
        assert "project_tags" in conf
        assert "dates" in conf

    def test_empty_ocr(self):
        ocr = OCRResult(spans=[], raw_text="", avg_confidence=0.0)
        entities, conf, _ = self.extractor.extract(ocr)
        assert entities.text_blocks == []
        assert entities.inferred_structure.bullets == []
        assert entities.inferred_structure.owners == []

    def test_no_duplicates_in_tags(self):
        ocr = make_ocr("#Alpha #Alpha #Beta #Beta")
        entities, _, _ = self.extractor.extract(ocr)
        tags = entities.inferred_structure.project_tags
        assert len(tags) == len(set(tags))


# =====================================================================
# Cross-extractor interface consistency
# =====================================================================

class TestExtractorInterface:
    """All extractors should follow the same interface contract."""

    @pytest.fixture(params=[
        ReceiptExtractor,
        ConversationExtractor,
        DocumentExtractor,
        WhiteboardExtractor,
    ])
    def extractor(self, request):
        return request.param()

    def test_has_image_type(self, extractor):
        assert hasattr(extractor, "image_type")
        assert isinstance(extractor.image_type, ImageType)

    def test_extract_returns_triple(self, extractor):
        ocr = make_ocr("Some test text\nAnother line")
        result = extractor.extract(ocr)
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_extract_returns_confidence_dict(self, extractor):
        ocr = make_ocr("Some test text\nAnother line")
        _, conf, _ = extractor.extract(ocr)
        assert isinstance(conf, dict)
        for k, v in conf.items():
            assert isinstance(k, str)
            assert isinstance(v, float)
            assert 0.0 <= v <= 1.0

    def test_empty_ocr_no_crash(self, extractor):
        ocr = OCRResult(spans=[], raw_text="", avg_confidence=0.0)
        entities, conf, hook = extractor.extract(ocr)
        assert entities is not None
        assert conf is not None
