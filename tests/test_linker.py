"""Tests for cross-image context fuser (linker) — PR9."""

import numpy as np
import pytest

from contextlens.linker import (
    _cosine_similarity,
    _dates_within_range,
    _generate_group_id,
    _generate_group_summary,
    _set_overlap,
    _union_find_groups,
    compute_metadata_score,
    compute_pairwise_score,
    extract_metadata,
    link_outputs,
)
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
# Helpers — build ImageOutput quickly
# =====================================================================

def _quality() -> QualitySignals:
    return QualitySignals(
        blur_score=500.0, brightness=0.6, contrast=0.5,
        estimated_quality=0.8,
    )


def _receipt_output(
    image_id: str,
    merchant: str = "STARBUCKS",
    total: float = 5.50,
    date: str = "03/15/2024",
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
        field_confidence={"merchant": 0.9, "total": 0.9},
        summary=f"{merchant} receipt for ${total:.2f} on {date}.",
        quality_signals=_quality(),
        raw_text=f"{merchant}\nLatte ${total}\nTOTAL ${total}\n{date}",
    )


def _conversation_output(
    image_id: str,
    participants: list[str] | None = None,
    topics: list[str] | None = None,
    has_meeting: bool = False,
) -> ImageOutput:
    participants = participants or ["Alice", "Bob"]
    topics = topics or ["project update"]
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
            key_topics=topics,
            action_items=["Send report"],
            referenced_events=refs,
        ),
        field_confidence={"participants": 0.9, "topics": 0.8},
        summary=f"Conversation between {' and '.join(participants)} about {', '.join(topics)}.",
        quality_signals=_quality(),
        raw_text="Alice: hello\nBob: hi",
        calendar_hook=hook,
    )


def _whiteboard_output(
    image_id: str,
    project_tags: list[str] | None = None,
    tasks: list[str] | None = None,
) -> ImageOutput:
    project_tags = project_tags or ["ProjectAlpha"]
    tasks = tasks or ["Design API", "Write tests"]
    return ImageOutput(
        image_id=image_id,
        type=ImageType.WHITEBOARD,
        type_confidence=0.8,
        extracted_entities=WhiteboardEntities(
            text_blocks=[TextBlock(text=t) for t in tasks],
            inferred_structure=WhiteboardStructure(
                bullets=tasks,
                tasks=tasks,
                project_tags=project_tags,
            ),
        ),
        field_confidence={"text_blocks": 0.7},
        summary=f"Whiteboard ({', '.join(project_tags)}): {len(tasks)} tasks.",
        quality_signals=_quality(),
        raw_text="\n".join(tasks),
    )


def _document_output(image_id: str) -> ImageOutput:
    return ImageOutput(
        image_id=image_id,
        type=ImageType.DOCUMENT,
        type_confidence=0.85,
        extracted_entities=DocumentEntities(
            document_kind="form",
            structured_fields={"Name": "John", "Date": "03/15/2024"},
        ),
        field_confidence={"Name": 0.9},
        summary="Form with 2 extracted fields.",
        quality_signals=_quality(),
        raw_text="Name: John\nDate: 03/15/2024",
    )


# =====================================================================
# Cosine similarity
# =====================================================================

class TestCosineSimilarity:
    def test_identical_vectors(self):
        a = np.array([1.0, 0.0, 0.0])
        assert _cosine_similarity(a, a) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert _cosine_similarity(a, b) == pytest.approx(0.0)

    def test_zero_vector(self):
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 0.0])
        assert _cosine_similarity(a, b) == 0.0


# =====================================================================
# Date proximity
# =====================================================================

class TestDateProximity:
    def test_same_date(self):
        assert _dates_within_range(["03/15/2024"], ["03/15/2024"]) is True

    def test_within_range(self):
        assert _dates_within_range(["03/15/2024"], ["03/17/2024"]) is True

    def test_outside_range(self):
        assert _dates_within_range(["03/15/2024"], ["03/25/2024"]) is False

    def test_empty_lists(self):
        assert _dates_within_range([], ["03/15/2024"]) is False

    def test_unparseable_dates(self):
        assert _dates_within_range(["not-a-date"], ["03/15/2024"]) is False


# =====================================================================
# Set overlap
# =====================================================================

class TestSetOverlap:
    def test_overlap(self):
        assert _set_overlap(["Alice", "Bob"], ["Bob", "Carol"]) is True

    def test_no_overlap(self):
        assert _set_overlap(["Alice"], ["Bob"]) is False

    def test_case_insensitive(self):
        assert _set_overlap(["alice"], ["Alice"]) is True

    def test_empty(self):
        assert _set_overlap([], ["Alice"]) is False


# =====================================================================
# Metadata extraction
# =====================================================================

class TestExtractMetadata:
    def test_receipt_metadata(self):
        out = _receipt_output("r1", merchant="STARBUCKS", date="03/15/2024")
        meta = extract_metadata(out)
        assert meta["merchant"] == "STARBUCKS"
        assert "03/15/2024" in meta["dates"]

    def test_conversation_metadata(self):
        out = _conversation_output("c1", participants=["Alice", "Bob"],
                                   topics=["meeting"])
        meta = extract_metadata(out)
        assert "Alice" in meta["participants"]
        assert "meeting" in meta["topics"]

    def test_whiteboard_metadata(self):
        out = _whiteboard_output("w1", project_tags=["ProjectAlpha"])
        meta = extract_metadata(out)
        assert "ProjectAlpha" in meta["project_tags"]

    def test_document_metadata(self):
        out = _document_output("d1")
        meta = extract_metadata(out)
        assert "03/15/2024" in meta["dates"]


# =====================================================================
# Metadata score
# =====================================================================

class TestMetadataScore:
    def test_same_merchant(self):
        ma = {"merchant": "starbucks", "dates": [], "participants": [],
              "project_tags": [], "topics": []}
        mb = {"merchant": "starbucks", "dates": [], "participants": [],
              "project_tags": [], "topics": []}
        score = compute_metadata_score(ma, mb)
        assert score >= 0.3  # SAME_MERCHANT_BONUS

    def test_different_merchant(self):
        ma = {"merchant": "starbucks", "dates": [], "participants": [],
              "project_tags": [], "topics": []}
        mb = {"merchant": "uber", "dates": [], "participants": [],
              "project_tags": [], "topics": []}
        score = compute_metadata_score(ma, mb)
        assert score == 0.0

    def test_date_proximity_adds_bonus(self):
        ma = {"merchant": None, "dates": ["03/15/2024"], "participants": [],
              "project_tags": [], "topics": []}
        mb = {"merchant": None, "dates": ["03/16/2024"], "participants": [],
              "project_tags": [], "topics": []}
        score = compute_metadata_score(ma, mb)
        assert score >= 0.2  # DATE_PROXIMITY_BONUS

    def test_shared_project_tags(self):
        ma = {"merchant": None, "dates": [], "participants": [],
              "project_tags": ["ProjectAlpha"], "topics": []}
        mb = {"merchant": None, "dates": [], "participants": [],
              "project_tags": ["ProjectAlpha"], "topics": []}
        score = compute_metadata_score(ma, mb)
        assert score >= 0.4  # SHARED_PROJECT_TAGS_BONUS

    def test_multiple_bonuses_stack(self):
        ma = {"merchant": "starbucks", "dates": ["03/15/2024"],
              "participants": [], "project_tags": [], "topics": []}
        mb = {"merchant": "starbucks", "dates": ["03/16/2024"],
              "participants": [], "project_tags": [], "topics": []}
        score = compute_metadata_score(ma, mb)
        assert score >= 0.5  # merchant + date


# =====================================================================
# Union-find grouping
# =====================================================================

class TestUnionFind:
    def test_no_edges(self):
        groups = _union_find_groups(3, [])
        assert len(groups) == 3

    def test_single_edge(self):
        groups = _union_find_groups(3, [(0, 1)])
        sizes = sorted(len(g) for g in groups)
        assert sizes == [1, 2]

    def test_all_connected(self):
        groups = _union_find_groups(3, [(0, 1), (1, 2)])
        assert len(groups) == 1
        assert len(groups[0]) == 3

    def test_two_components(self):
        groups = _union_find_groups(4, [(0, 1), (2, 3)])
        assert len(groups) == 2
        sizes = sorted(len(g) for g in groups)
        assert sizes == [2, 2]


# =====================================================================
# Group ID generation
# =====================================================================

class TestGroupId:
    def test_receipt_group_id(self):
        outputs = [
            _receipt_output("r1", merchant="STARBUCKS"),
            _receipt_output("r2", merchant="STARBUCKS"),
        ]
        gid = _generate_group_id(outputs)
        assert "receipt" in gid.lower()
        assert "starbucks" in gid.lower()

    def test_whiteboard_group_id(self):
        outputs = [
            _whiteboard_output("w1", project_tags=["ProjectAlpha"]),
            _whiteboard_output("w2", project_tags=["ProjectAlpha"]),
        ]
        gid = _generate_group_id(outputs)
        assert "project" in gid.lower()
        assert "projectalpha" in gid.lower()


# =====================================================================
# Group summary generation
# =====================================================================

class TestGroupSummary:
    def test_receipt_group_summary(self):
        outputs = [
            _receipt_output("r1", merchant="STARBUCKS", date="03/15/2024"),
            _receipt_output("r2", merchant="STARBUCKS", date="03/16/2024"),
        ]
        summary = _generate_group_summary("receipts_trip", outputs)
        assert "receipts_trip" in summary
        assert "STARBUCKS" in summary
        assert "receipt" in summary.lower()

    def test_whiteboard_group_summary(self):
        outputs = [
            _whiteboard_output("w1", project_tags=["Alpha"]),
            _whiteboard_output("w2", project_tags=["Alpha"]),
        ]
        summary = _generate_group_summary("project_alpha", outputs)
        assert "Alpha" in summary


# =====================================================================
# link_outputs — integration
# =====================================================================

class TestLinkOutputs:
    def test_same_merchant_grouped(self):
        outputs = [
            _receipt_output("r1", merchant="STARBUCKS", date="03/15/2024"),
            _receipt_output("r2", merchant="STARBUCKS", date="03/16/2024"),
        ]
        linked, summaries = link_outputs(outputs)
        # Both should get the same group_id
        assert linked[0].group_id is not None
        assert linked[0].group_id == linked[1].group_id
        assert len(summaries) == 1

    def test_unrelated_not_grouped(self):
        outputs = [
            _receipt_output("r1", merchant="STARBUCKS", date="03/15/2024"),
            _conversation_output("c1", participants=["Zara", "Xander"],
                                 topics=["astronomy"]),
        ]
        linked, summaries = link_outputs(outputs)
        # Should NOT be grouped (different type, no metadata overlap)
        assert linked[0].group_id is None or linked[0].group_id != linked[1].group_id

    def test_whiteboards_same_project_grouped(self):
        outputs = [
            _whiteboard_output("w1", project_tags=["ProjectAlpha"]),
            _whiteboard_output("w2", project_tags=["ProjectAlpha"]),
        ]
        linked, summaries = link_outputs(outputs)
        assert linked[0].group_id is not None
        assert linked[0].group_id == linked[1].group_id
        assert len(summaries) == 1

    def test_single_output_no_group(self):
        outputs = [_receipt_output("r1")]
        linked, summaries = link_outputs(outputs)
        assert linked[0].group_id is None
        assert summaries == {}

    def test_empty_list(self):
        linked, summaries = link_outputs([])
        assert linked == []
        assert summaries == {}

    def test_group_summary_generated(self):
        outputs = [
            _receipt_output("r1", merchant="STARBUCKS", date="03/15/2024"),
            _receipt_output("r2", merchant="STARBUCKS", date="03/16/2024"),
        ]
        _, summaries = link_outputs(outputs)
        for gid, summary in summaries.items():
            assert len(summary) > 0
            assert "STARBUCKS" in summary

    def test_calendar_hook_propagated(self):
        outputs = [
            _conversation_output("c1", participants=["Alice", "Bob"],
                                 topics=["meeting"], has_meeting=True),
            _conversation_output("c2", participants=["Alice", "Bob"],
                                 topics=["meeting"]),
        ]
        linked, _ = link_outputs(outputs)
        # Both should have the same group
        if linked[0].group_id == linked[1].group_id and linked[0].group_id is not None:
            # Calendar hook should propagate to c2
            assert linked[1].calendar_hook is not None
            assert linked[1].calendar_hook.mentioned is True

    def test_three_receipts_same_merchant(self):
        outputs = [
            _receipt_output("r1", merchant="STARBUCKS", date="03/15/2024"),
            _receipt_output("r2", merchant="STARBUCKS", date="03/16/2024"),
            _receipt_output("r3", merchant="STARBUCKS", date="03/17/2024"),
        ]
        linked, summaries = link_outputs(outputs)
        group_ids = {o.group_id for o in linked}
        # All three should be in the same group
        assert len(group_ids) == 1
        assert None not in group_ids

    def test_mixed_types_separate(self):
        outputs = [
            _receipt_output("r1", merchant="STARBUCKS"),
            _whiteboard_output("w1", project_tags=["Beta"]),
            _document_output("d1"),
        ]
        linked, _ = link_outputs(outputs)
        # Unrelated items should mostly not be grouped
        group_ids = [o.group_id for o in linked]
        none_count = sum(1 for g in group_ids if g is None)
        assert none_count >= 2  # at most one pair could match by embedding


# =====================================================================
# Pairwise score
# =====================================================================

class TestPairwiseScore:
    def test_high_similarity_high_metadata(self):
        emb = np.ones(384)
        meta = {"merchant": "starbucks", "dates": ["03/15/2024"],
                "participants": [], "project_tags": [], "topics": []}
        score = compute_pairwise_score(emb, emb, meta, meta)
        assert score > 0.5

    def test_zero_vectors_high_metadata(self):
        emb = np.zeros(384)
        meta_a = {"merchant": "starbucks", "dates": ["03/15/2024"],
                  "participants": [], "project_tags": [], "topics": []}
        meta_b = {"merchant": "starbucks", "dates": ["03/16/2024"],
                  "participants": [], "project_tags": [], "topics": []}
        score = compute_pairwise_score(emb, emb, meta_a, meta_b)
        # Only metadata contributes
        assert score >= 0.3
