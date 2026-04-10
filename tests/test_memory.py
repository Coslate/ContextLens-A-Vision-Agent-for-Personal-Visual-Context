"""Tests for memory store and query API — PR9."""

import tempfile
from pathlib import Path

import pytest

from contextlens.memory_store import MemoryStore
from contextlens.query import query
from contextlens.schemas import (
    CalendarEventCandidate,
    CalendarHook,
    ConversationEntities,
    DocumentEntities,
    ImageOutput,
    ImageType,
    QualitySignals,
    ReceiptEntities,
    ReceiptItem,
    TextBlock,
    WhiteboardEntities,
    WhiteboardStructure,
)


# =====================================================================
# Fixtures
# =====================================================================

@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "test_memory.db"


@pytest.fixture
def store(db_path):
    s = MemoryStore(db_path)
    yield s
    s.close()


def _quality() -> QualitySignals:
    return QualitySignals(
        blur_score=500.0, brightness=0.6, contrast=0.5,
        estimated_quality=0.8,
    )


def _receipt_output(
    image_id: str = "img_001",
    merchant: str = "STARBUCKS",
    total: float = 5.50,
    date: str = "03/15/2024",
    group_id: str | None = None,
) -> ImageOutput:
    return ImageOutput(
        image_id=image_id,
        type=ImageType.RECEIPT,
        type_confidence=0.9,
        extracted_entities=ReceiptEntities(
            merchant=merchant,
            items=[ReceiptItem(name="Latte", price=total)],
            total=total,
            date=date,
            currency="USD",
        ),
        field_confidence={"merchant": 0.95, "total": 0.9, "date": 0.85},
        summary=f"{merchant} receipt for ${total:.2f} on {date}.",
        quality_signals=_quality(),
        raw_text=f"{merchant}\nLatte ${total}\nTOTAL ${total}\n{date}",
        group_id=group_id,
    )


def _conversation_output(
    image_id: str = "img_005",
    has_meeting: bool = False,
) -> ImageOutput:
    participants = ["Alice", "Bob"]
    hook = None
    refs = []
    if has_meeting:
        refs = [CalendarEventCandidate(
            title="team sync", time_mention="tomorrow 3pm",
            participants=participants,
        )]
        hook = CalendarHook(mentioned=True, event_candidates=refs)
    return ImageOutput(
        image_id=image_id,
        type=ImageType.CONVERSATION,
        type_confidence=0.85,
        extracted_entities=ConversationEntities(
            participants=participants,
            key_topics=["meeting", "project"],
            action_items=["Send report"],
            referenced_events=refs,
        ),
        field_confidence={"participants": 0.9, "topics": 0.8},
        summary="Conversation between Alice and Bob about meeting, project.",
        quality_signals=_quality(),
        raw_text="Alice: hello\nBob: hi",
        calendar_hook=hook,
    )


def _whiteboard_output(
    image_id: str = "img_011",
    project_tags: list[str] | None = None,
    group_id: str | None = None,
) -> ImageOutput:
    project_tags = project_tags or ["ProjectAlpha"]
    return ImageOutput(
        image_id=image_id,
        type=ImageType.WHITEBOARD,
        type_confidence=0.8,
        extracted_entities=WhiteboardEntities(
            text_blocks=[TextBlock(text="Task 1")],
            inferred_structure=WhiteboardStructure(
                bullets=["Task 1"],
                tasks=["Design API"],
                project_tags=project_tags,
            ),
        ),
        field_confidence={"text_blocks": 0.7},
        summary=f"Whiteboard ({', '.join(project_tags)}): 1 task.",
        quality_signals=_quality(),
        raw_text="Task 1\n#ProjectAlpha",
        group_id=group_id,
    )


def _blurry_output(image_id: str = "img_003") -> ImageOutput:
    return ImageOutput(
        image_id=image_id,
        type=ImageType.RECEIPT,
        type_confidence=0.6,
        extracted_entities=ReceiptEntities(merchant="BLURRY"),
        field_confidence={"merchant": 0.3},
        summary="Blurry receipt.",
        quality_signals=QualitySignals(
            blur_score=20.0, brightness=0.5, contrast=0.4,
            estimated_quality=0.3, is_blurry=True,
        ),
        raw_text="BLURRY",
        needs_clarification=True,
    )


# =====================================================================
# MemoryStore — basic CRUD
# =====================================================================

class TestMemoryStoreBasic:
    def test_store_and_retrieve(self, store):
        out = _receipt_output()
        store.store_output(out)
        record = store.get_image("img_001")
        assert record is not None
        assert record["image_id"] == "img_001"
        assert record["type"] == "receipt"

    def test_store_sets_processed_at(self, store):
        store.store_output(_receipt_output())
        record = store.get_image("img_001")
        assert record["processed_at"] is not None

    def test_get_nonexistent(self, store):
        assert store.get_image("nonexistent") is None

    def test_store_replaces_on_conflict(self, store):
        store.store_output(_receipt_output(total=5.50))
        store.store_output(_receipt_output(total=10.00))
        record = store.get_image("img_001")
        assert record is not None
        # Should have the latest summary
        assert "$10.00" in record["summary"]

    def test_get_all_images(self, store):
        store.store_output(_receipt_output("r1"))
        store.store_output(_receipt_output("r2"))
        all_imgs = store.get_all_images()
        assert len(all_imgs) == 2

    def test_store_batch(self, store):
        outputs = [_receipt_output("r1"), _receipt_output("r2")]
        store.store_batch(outputs)
        assert len(store.get_all_images()) == 2


# =====================================================================
# MemoryStore — type queries
# =====================================================================

class TestMemoryStoreTypeQueries:
    def test_get_by_type_receipt(self, store):
        store.store_output(_receipt_output())
        store.store_output(_conversation_output())
        receipts = store.get_images_by_type("receipt")
        assert len(receipts) == 1
        assert receipts[0]["type"] == "receipt"

    def test_get_by_type_conversation(self, store):
        store.store_output(_receipt_output())
        store.store_output(_conversation_output())
        convos = store.get_images_by_type("conversation")
        assert len(convos) == 1
        assert convos[0]["type"] == "conversation"

    def test_get_by_type_empty(self, store):
        store.store_output(_receipt_output())
        docs = store.get_images_by_type("document")
        assert docs == []


# =====================================================================
# MemoryStore — group queries
# =====================================================================

class TestMemoryStoreGroups:
    def test_get_by_group(self, store):
        store.store_output(_receipt_output("r1", group_id="trip"))
        store.store_output(_receipt_output("r2", group_id="trip"))
        store.store_output(_receipt_output("r3"))
        group = store.get_images_by_group("trip")
        assert len(group) == 2

    def test_store_group_record(self, store):
        store.store_group("trip", 2, "2 receipts from STARBUCKS.")
        record = store.get_group("trip")
        assert record is not None
        assert record["member_count"] == 2
        assert "STARBUCKS" in record["fused_summary"]

    def test_get_all_groups(self, store):
        store.store_group("trip", 2, "Trip group.")
        store.store_group("project", 3, "Project group.")
        groups = store.get_all_groups()
        assert len(groups) == 2


# =====================================================================
# MemoryStore — entities
# =====================================================================

class TestMemoryStoreEntities:
    def test_entities_stored(self, store):
        store.store_output(_receipt_output())
        entities = store.get_entities("img_001")
        assert len(entities) > 0

    def test_entity_field_names(self, store):
        store.store_output(_receipt_output())
        entities = store.get_entities("img_001")
        field_names = {e["field_name"] for e in entities}
        assert "merchant" in field_names
        assert "total" in field_names

    def test_search_entities(self, store):
        store.store_output(_whiteboard_output("w1", project_tags=["Alpha"]))
        store.store_output(_whiteboard_output("w2", project_tags=["Beta"]))
        results = store.search_entities("project_tag", "Alpha")
        assert len(results) == 1
        assert results[0]["image_id"] == "w1"

    def test_participant_entities_stored(self, store):
        store.store_output(_conversation_output())
        entities = store.get_entities("img_005")
        participants = [
            e for e in entities if e["field_name"] == "participant"
        ]
        assert len(participants) >= 2


# =====================================================================
# MemoryStore — calendar hooks
# =====================================================================

class TestMemoryStoreCalendar:
    def test_calendar_hook_stored(self, store):
        store.store_output(_conversation_output(has_meeting=True))
        hooks = store.get_calendar_hooks("img_005")
        assert len(hooks) >= 1
        assert hooks[0]["event_title"] == "team sync"

    def test_no_calendar_hook_for_receipt(self, store):
        store.store_output(_receipt_output())
        hooks = store.get_calendar_hooks("img_001")
        assert hooks == []

    def test_get_all_calendar_hooks(self, store):
        store.store_output(_conversation_output("c1", has_meeting=True))
        store.store_output(_conversation_output("c2", has_meeting=False))
        all_hooks = store.get_calendar_hooks()
        assert len(all_hooks) >= 1

    def test_images_with_calendar_hooks(self, store):
        store.store_output(_conversation_output("c1", has_meeting=True))
        store.store_output(_receipt_output("r1"))
        imgs = store.get_images_with_calendar_hooks()
        assert len(imgs) == 1
        assert imgs[0]["image_id"] == "c1"


# =====================================================================
# MemoryStore — time queries
# =====================================================================

class TestMemoryStoreTimeQueries:
    def test_get_images_since(self, store):
        store.store_output(_receipt_output())
        recent = store.get_images_since(7)
        assert len(recent) == 1  # just inserted → within 7 days

    def test_needs_clarification(self, store):
        store.store_output(_receipt_output())
        store.store_output(_blurry_output())
        flagged = store.get_needs_clarification()
        assert len(flagged) == 1
        assert flagged[0]["image_id"] == "img_003"


# =====================================================================
# MemoryStore — links
# =====================================================================

class TestMemoryStoreLinks:
    def test_store_link(self, store):
        store.store_output(_receipt_output("r1"))
        store.store_output(_receipt_output("r2"))
        store.store_link("r1", "r2", "similar", 0.85)
        # Verify via direct SQL
        cur = store.conn.cursor()
        cur.execute("SELECT * FROM links WHERE src_image_id = 'r1'")
        row = cur.fetchone()
        assert row is not None
        assert dict(row)["link_score"] == pytest.approx(0.85)


# =====================================================================
# MemoryStore — lifecycle
# =====================================================================

class TestMemoryStoreLifecycle:
    def test_clear(self, store):
        store.store_output(_receipt_output())
        store.clear()
        assert store.get_all_images() == []

    def test_context_manager(self, db_path):
        with MemoryStore(db_path) as s:
            s.store_output(_receipt_output())
            assert s.get_image("img_001") is not None
        # Connection closed, but data persists
        s2 = MemoryStore(db_path)
        assert s2.get_image("img_001") is not None
        s2.close()


# =====================================================================
# Query API — type queries
# =====================================================================

class TestQueryType:
    def test_all_receipts(self, store):
        store.store_output(_receipt_output("r1"))
        store.store_output(_conversation_output("c1"))
        results = query(store, "all receipts")
        assert len(results) == 1
        assert results[0]["type"] == "receipt"

    def test_conversations(self, store):
        store.store_output(_receipt_output("r1"))
        store.store_output(_conversation_output("c1"))
        results = query(store, "show me conversations")
        assert len(results) == 1
        assert results[0]["type"] == "conversation"

    def test_whiteboards(self, store):
        store.store_output(_whiteboard_output("w1"))
        results = query(store, "whiteboard photos")
        assert len(results) == 1

    def test_documents(self, store):
        out = ImageOutput(
            image_id="d1", type=ImageType.DOCUMENT, type_confidence=0.8,
            extracted_entities=DocumentEntities(
                document_kind="form", structured_fields={"Name": "John"},
            ),
            field_confidence={"Name": 0.9},
            summary="Form.", quality_signals=_quality(), raw_text="Name: John",
        )
        store.store_output(out)
        results = query(store, "all documents")
        assert len(results) == 1


# =====================================================================
# Query API — time queries
# =====================================================================

class TestQueryTime:
    def test_receipts_from_last_week(self, store):
        store.store_output(_receipt_output("r1"))
        results = query(store, "all receipts from last week")
        assert len(results) == 1

    def test_past_7_days(self, store):
        store.store_output(_receipt_output("r1"))
        results = query(store, "images from past 7 days")
        assert len(results) == 1

    def test_today(self, store):
        store.store_output(_receipt_output("r1"))
        results = query(store, "images from today")
        assert len(results) == 1


# =====================================================================
# Query API — group queries
# =====================================================================

class TestQueryGroup:
    def test_images_in_group(self, store):
        store.store_output(_receipt_output("r1", group_id="trip"))
        store.store_output(_receipt_output("r2", group_id="trip"))
        store.store_output(_receipt_output("r3"))
        results = query(store, "all images in group trip")
        assert len(results) == 2

    def test_group_nonexistent(self, store):
        store.store_output(_receipt_output("r1"))
        results = query(store, "images in group nonexistent")
        assert results == []


# =====================================================================
# Query API — project queries
# =====================================================================

class TestQueryProject:
    def test_whiteboard_from_project(self, store):
        store.store_output(_whiteboard_output("w1", project_tags=["Alpha"]))
        store.store_output(_whiteboard_output("w2", project_tags=["Beta"]))
        results = query(store, "whiteboard photos from project Alpha")
        assert len(results) == 1
        assert results[0]["image_id"] == "w1"


# =====================================================================
# Query API — meeting/calendar queries
# =====================================================================

class TestQueryMeeting:
    def test_conversations_mentioning_meeting(self, store):
        store.store_output(_conversation_output("c1", has_meeting=True))
        store.store_output(_receipt_output("r1"))
        results = query(store, "conversations mentioning a meeting")
        assert len(results) == 1
        assert results[0]["image_id"] == "c1"


# =====================================================================
# Query API — clarification queries
# =====================================================================

class TestQueryClarification:
    def test_images_needing_clarification(self, store):
        store.store_output(_receipt_output("r1"))
        store.store_output(_blurry_output("r2"))
        results = query(store, "images needing clarification")
        assert len(results) == 1
        assert results[0]["image_id"] == "r2"


# =====================================================================
# Query API — fallback
# =====================================================================

class TestQueryFallback:
    def test_empty_query_returns_all(self, store):
        store.store_output(_receipt_output("r1"))
        store.store_output(_conversation_output("c1"))
        results = query(store, "")
        assert len(results) == 2

    def test_unknown_query_returns_all(self, store):
        store.store_output(_receipt_output("r1"))
        results = query(store, "xyzzy foobar")
        assert len(results) == 1
