"""Tests for evaluation metrics, calibration, and comparison — PR11."""

import json
import math
import tempfile
from pathlib import Path

import pytest

from scripts.run_evaluation import (
    _ADVERSARIAL_IDS,
    _rank,
    calibration_correlation,
    collect_confidence_correctness_pairs,
    compute_aggregate_metrics,
    compute_calibration_buckets,
    evaluate_calendar_hooks,
    evaluate_conversation,
    evaluate_document,
    evaluate_extraction,
    evaluate_failure_handling,
    evaluate_linking,
    evaluate_receipt,
    evaluate_type_classification,
    evaluate_whiteboard,
    exact_match,
    expected_calibration_error,
    f1_score,
    fuzzy_match,
    generate_comparison_table,
    list_fuzzy_precision,
    list_fuzzy_recall,
    load_annotations,
    load_outputs,
    plot_calibration,
)
from contextlens.config import BASELINE_CONFIDENCE


# =====================================================================
# Helpers — build mock outputs and annotations
# =====================================================================

def _receipt_output(
    image_id: str = "img_001",
    merchant: str = "STARBUCKS",
    total: float = 12.42,
    date: str = "03/15/2024",
    items: list | None = None,
    confidence: dict | None = None,
    group_id: str | None = None,
    failure_flags: list | None = None,
    needs_clarification: bool = False,
    calendar_hook: dict | None = None,
) -> dict:
    if items is None:
        items = [
            {"name": "Latte", "price": 5.50},
            {"name": "Muffin", "price": 3.25},
            {"name": "Iced Tea", "price": 2.75},
        ]
    if confidence is None:
        confidence = {"merchant": 0.95, "total": 0.90, "date": 0.85, "currency": 0.80, "items": 0.78}
    return {
        "image_id": image_id,
        "type": "receipt",
        "type_confidence": 0.9,
        "extracted_entities": {
            "merchant": merchant,
            "items": items,
            "total": total,
            "date": date,
            "currency": "USD",
        },
        "field_confidence": confidence,
        "summary": f"{merchant} receipt" + (f" for ${total:.2f}" if total is not None else "") + ".",
        "failure_flags": failure_flags or [],
        "needs_clarification": needs_clarification,
        "raw_text": f"{merchant}\nLatte $5.50\nTOTAL {total}\n{date}",
        "group_id": group_id,
        "calendar_hook": calendar_hook,
    }


def _receipt_annotation(
    image_id: str = "img_001",
    merchant: str = "STARBUCKS",
    total: float = 12.42,
    date: str = "03/15/2024",
    group: str | None = "receipts_trip",
    failure_flags: list | None = None,
    needs_clarification: bool = False,
) -> dict:
    return {
        "image_id": image_id,
        "expected_type": "receipt",
        "expected_entities": {
            "merchant": merchant,
            "items": [
                {"name": "Latte", "price": 5.50},
                {"name": "Muffin", "price": 3.25},
                {"name": "Iced Tea", "price": 2.75},
            ],
            "total": total,
            "date": date,
            "currency": "USD",
        },
        "expected_group": group,
        "expected_calendar_hook": None,
        "expected_calendar_events": [],
        "expected_needs_clarification": needs_clarification,
        "expected_failure_flags": failure_flags or [],
    }


def _conversation_output(
    image_id: str = "img_005",
    participants: list | None = None,
    topics: list | None = None,
    action_items: list | None = None,
    calendar_hook: dict | None = None,
    confidence: dict | None = None,
) -> dict:
    participants = participants or ["Alice", "Bob"]
    topics = topics or ["project", "API"]
    action_items = action_items or ["send me the documentation"]
    if confidence is None:
        confidence = {"participants": 0.9, "key_topics": 0.8, "action_items": 0.7}
    return {
        "image_id": image_id,
        "type": "conversation",
        "type_confidence": 0.85,
        "extracted_entities": {
            "participants": participants,
            "key_topics": topics,
            "action_items": action_items,
            "referenced_events": [],
        },
        "field_confidence": confidence,
        "summary": "Conversation between Alice and Bob.",
        "failure_flags": [],
        "needs_clarification": False,
        "raw_text": "Alice: hello\nBob: hi",
        "group_id": None,
        "calendar_hook": calendar_hook,
    }


def _conversation_annotation(
    image_id: str = "img_005",
    participants: list | None = None,
    topics: list | None = None,
    action_items: list | None = None,
    calendar_hook: bool | None = None,
) -> dict:
    participants = participants or ["Alice", "Bob"]
    topics = topics or ["project", "API", "documentation", "test"]
    action_items = action_items or [
        "send me the documentation",
        "prepare it and share by EOD",
        "check the test results",
    ]
    return {
        "image_id": image_id,
        "expected_type": "conversation",
        "expected_entities": {
            "participants": participants,
            "key_topics": topics,
            "action_items": action_items,
        },
        "expected_group": None,
        "expected_calendar_hook": calendar_hook,
        "expected_calendar_events": [],
        "expected_needs_clarification": False,
        "expected_failure_flags": [],
    }


def _document_output(image_id: str = "img_008") -> dict:
    return {
        "image_id": image_id,
        "type": "document",
        "type_confidence": 0.85,
        "extracted_entities": {
            "document_kind": "form",
            "structured_fields": {
                "Patient Name": "John Doe",
                "Date of Birth": "01/15/1985",
                "Phone": "555-0123",
            },
        },
        "field_confidence": {"document_kind": 0.9, "structured_fields": 0.8},
        "summary": "Form with 3 extracted fields.",
        "failure_flags": [],
        "needs_clarification": False,
        "raw_text": "Patient Name: John Doe",
        "group_id": None,
        "calendar_hook": None,
    }


def _document_annotation(image_id: str = "img_008") -> dict:
    return {
        "image_id": image_id,
        "expected_type": "document",
        "expected_entities": {
            "document_kind": "form",
            "structured_fields": {
                "Patient Name": "John Doe",
                "Date of Birth": "01/15/1985",
                "Phone": "555-0123",
                "Address": "123 Main Street",
                "Insurance ID": "INS-98765",
                "Emergency Contact": "Jane Doe",
            },
        },
        "expected_group": None,
        "expected_calendar_hook": None,
        "expected_calendar_events": [],
        "expected_needs_clarification": False,
        "expected_failure_flags": [],
    }


def _whiteboard_output(image_id: str = "img_011", group_id: str | None = None) -> dict:
    return {
        "image_id": image_id,
        "type": "whiteboard",
        "type_confidence": 0.8,
        "extracted_entities": {
            "text_blocks": [{"text": "Design API endpoints"}, {"text": "Write unit tests"}],
            "inferred_structure": {
                "bullets": ["Design API endpoints @Alice", "Write unit tests @Bob"],
                "owners": ["Alice", "Bob"],
                "dates": ["due Friday"],
                "tasks": ["Design API endpoints", "Write unit tests"],
                "project_tags": ["ProjectAlpha", "Sprint5"],
            },
        },
        "field_confidence": {"text_blocks": 0.7, "owners": 0.8, "tasks": 0.75, "project_tags": 0.85},
        "summary": "Whiteboard (ProjectAlpha): 2 tasks.",
        "failure_flags": [],
        "needs_clarification": False,
        "raw_text": "Design API endpoints @Alice\nWrite unit tests @Bob",
        "group_id": group_id,
        "calendar_hook": None,
    }


def _whiteboard_annotation(image_id: str = "img_011", group: str | None = "project_alpha") -> dict:
    return {
        "image_id": image_id,
        "expected_type": "whiteboard",
        "expected_entities": {
            "bullets": ["Design API endpoints @Alice", "Write unit tests @Bob", "Deploy to staging @Carol"],
            "owners": ["Alice", "Bob", "Carol"],
            "tasks": ["Design API endpoints", "Write unit tests", "Deploy to staging",
                      "update documentation", "review PR #42"],
            "project_tags": ["ProjectAlpha", "Sprint5", "backend"],
            "dates": ["due Friday", "Monday"],
        },
        "expected_group": group,
        "expected_calendar_hook": None,
        "expected_calendar_events": [],
        "expected_needs_clarification": False,
        "expected_failure_flags": [],
    }


# =====================================================================
# Exact match
# =====================================================================

class TestExactMatch:
    def test_same_string(self):
        assert exact_match("STARBUCKS", "STARBUCKS") is True

    def test_case_insensitive(self):
        assert exact_match("Starbucks", "starbucks") is True

    def test_different_strings(self):
        assert exact_match("STARBUCKS", "UBER") is False

    def test_numeric_match(self):
        assert exact_match(12.42, 12.42) is True

    def test_numeric_close(self):
        assert exact_match(12.42, 12.4200001) is True

    def test_numeric_different(self):
        assert exact_match(12.42, 15.00) is False

    def test_none_none(self):
        assert exact_match(None, None) is True

    def test_none_vs_value(self):
        assert exact_match(None, "STARBUCKS") is False

    def test_value_vs_none(self):
        assert exact_match("STARBUCKS", None) is False

    def test_whitespace_trimmed(self):
        assert exact_match("  STARBUCKS  ", "STARBUCKS") is True


# =====================================================================
# Fuzzy match
# =====================================================================

class TestFuzzyMatch:
    def test_exact_strings(self):
        assert fuzzy_match("Latte", "Latte") is True

    def test_similar_strings(self):
        assert fuzzy_match("Latte", "latte") is True

    def test_close_enough(self):
        # "Iced Tea" vs "Iced tea" should pass
        assert fuzzy_match("Iced Tea", "Iced tea") is True

    def test_very_different(self):
        assert fuzzy_match("Latte", "Burger") is False

    def test_none_none(self):
        assert fuzzy_match(None, None) is True

    def test_none_vs_string(self):
        assert fuzzy_match(None, "test") is False


# =====================================================================
# List fuzzy matching
# =====================================================================

class TestListMatching:
    def test_recall_perfect(self):
        assert list_fuzzy_recall(["Alice", "Bob"], ["Alice", "Bob"]) == 1.0

    def test_recall_partial(self):
        assert list_fuzzy_recall(["Alice"], ["Alice", "Bob"]) == 0.5

    def test_recall_empty_expected(self):
        assert list_fuzzy_recall(["Alice"], []) == 1.0

    def test_recall_empty_predicted(self):
        assert list_fuzzy_recall([], ["Alice", "Bob"]) == 0.0

    def test_precision_perfect(self):
        assert list_fuzzy_precision(["Alice", "Bob"], ["Alice", "Bob"]) == 1.0

    def test_precision_extra(self):
        assert list_fuzzy_precision(["Alice", "Bob", "Carol"], ["Alice", "Bob"]) == pytest.approx(2 / 3)

    def test_precision_empty_predicted(self):
        assert list_fuzzy_precision([], []) == 1.0

    def test_precision_empty_pred_nonempty_exp(self):
        assert list_fuzzy_precision([], ["Alice"]) == 0.0

    def test_f1_perfect(self):
        assert f1_score(1.0, 1.0) == 1.0

    def test_f1_zero(self):
        assert f1_score(0.0, 0.0) == 0.0

    def test_f1_harmonic_mean(self):
        assert f1_score(0.5, 1.0) == pytest.approx(2 / 3)


# =====================================================================
# Receipt evaluation
# =====================================================================

class TestEvaluateReceipt:
    def test_perfect_extraction(self):
        out = _receipt_output()
        ann = _receipt_annotation()
        results = evaluate_receipt(out, ann)
        assert results["merchant"]["correct"] is True
        assert results["total"]["correct"] is True
        assert results["date"]["correct"] is True
        assert results["currency"]["correct"] is True

    def test_wrong_merchant(self):
        out = _receipt_output(merchant="UBER")
        ann = _receipt_annotation()
        results = evaluate_receipt(out, ann)
        assert results["merchant"]["correct"] is False

    def test_wrong_total(self):
        out = _receipt_output(total=99.99)
        ann = _receipt_annotation()
        results = evaluate_receipt(out, ann)
        assert results["total"]["correct"] is False

    def test_missing_total(self):
        out = _receipt_output(total=None)
        ann = _receipt_annotation(total=12.42)
        # total is None in output but expected is 12.42
        out["extracted_entities"]["total"] = None
        results = evaluate_receipt(out, ann)
        assert results["total"]["correct"] is False

    def test_items_f1(self):
        out = _receipt_output()
        ann = _receipt_annotation()
        results = evaluate_receipt(out, ann)
        assert results["items"]["f1"] >= 0.5
        assert results["items"]["correct"] is True

    def test_items_partial(self):
        out = _receipt_output(items=[{"name": "Latte", "price": 5.50}])
        ann = _receipt_annotation()
        results = evaluate_receipt(out, ann)
        # 1/3 recall
        assert results["items"]["recall"] == pytest.approx(1 / 3)


# =====================================================================
# Conversation evaluation
# =====================================================================

class TestEvaluateConversation:
    def test_participants_found(self):
        out = _conversation_output()
        ann = _conversation_annotation()
        results = evaluate_conversation(out, ann)
        assert results["participants"]["correct"] is True
        assert results["participants"]["f1"] == 1.0

    def test_partial_topics(self):
        out = _conversation_output(topics=["project"])
        ann = _conversation_annotation()
        results = evaluate_conversation(out, ann)
        assert results["key_topics"]["recall"] == pytest.approx(1 / 4)

    def test_action_items_recall(self):
        out = _conversation_output(action_items=["send me the documentation"])
        ann = _conversation_annotation()
        results = evaluate_conversation(out, ann)
        assert results["action_items"]["recall"] == pytest.approx(1 / 3)


# =====================================================================
# Document evaluation
# =====================================================================

class TestEvaluateDocument:
    def test_document_kind_correct(self):
        out = _document_output()
        ann = _document_annotation()
        results = evaluate_document(out, ann)
        assert results["document_kind"]["correct"] is True

    def test_structured_fields_partial(self):
        out = _document_output()
        ann = _document_annotation()
        results = evaluate_document(out, ann)
        # 3 out of 6 expected
        assert results["structured_fields"]["recall"] == pytest.approx(3 / 6)
        assert results["structured_fields"]["correct"] is True  # >= 0.3


# =====================================================================
# Whiteboard evaluation
# =====================================================================

class TestEvaluateWhiteboard:
    def test_owners_recall(self):
        out = _whiteboard_output()
        ann = _whiteboard_annotation()
        results = evaluate_whiteboard(out, ann)
        # 2 out of 3 expected owners
        assert results["owners"]["recall"] == pytest.approx(2 / 3)

    def test_tasks_recall(self):
        out = _whiteboard_output()
        ann = _whiteboard_annotation()
        results = evaluate_whiteboard(out, ann)
        # 2 out of 5 expected tasks
        assert results["tasks"]["recall"] == pytest.approx(2 / 5)

    def test_project_tags_recall(self):
        out = _whiteboard_output()
        ann = _whiteboard_annotation()
        results = evaluate_whiteboard(out, ann)
        # 2 out of 3 expected tags
        assert results["project_tags"]["recall"] == pytest.approx(2 / 3)

    def test_text_blocks_present(self):
        out = _whiteboard_output()
        ann = _whiteboard_annotation()
        results = evaluate_whiteboard(out, ann)
        assert results["text_blocks"]["correct"] is True
        assert results["text_blocks"]["count"] == 2


# =====================================================================
# evaluate_extraction dispatcher
# =====================================================================

class TestEvaluateExtraction:
    def test_receipt_dispatch(self):
        out = _receipt_output()
        ann = _receipt_annotation()
        results = evaluate_extraction(out, ann)
        assert "merchant" in results
        assert "total" in results

    def test_conversation_dispatch(self):
        out = _conversation_output()
        ann = _conversation_annotation()
        results = evaluate_extraction(out, ann)
        assert "participants" in results

    def test_document_dispatch(self):
        out = _document_output()
        ann = _document_annotation()
        results = evaluate_extraction(out, ann)
        assert "document_kind" in results

    def test_whiteboard_dispatch(self):
        out = _whiteboard_output()
        ann = _whiteboard_annotation()
        results = evaluate_extraction(out, ann)
        assert "owners" in results

    def test_unknown_type(self):
        ann = {"expected_type": "unknown"}
        results = evaluate_extraction({}, ann)
        assert results == {}


# =====================================================================
# Type classification
# =====================================================================

class TestTypeClassification:
    def test_correct(self):
        out = {"type": "receipt"}
        ann = {"expected_type": "receipt"}
        assert evaluate_type_classification(out, ann) is True

    def test_incorrect(self):
        out = {"type": "document"}
        ann = {"expected_type": "receipt"}
        assert evaluate_type_classification(out, ann) is False


# =====================================================================
# Confidence calibration
# =====================================================================

class TestConfidenceCalibration:
    def test_collect_pairs(self):
        outputs = [_receipt_output()]
        annotations = [_receipt_annotation()]
        pairs = collect_confidence_correctness_pairs(outputs, annotations)
        assert len(pairs) > 0
        for conf, correct in pairs:
            assert 0.0 <= conf <= 1.0
            assert isinstance(correct, bool)

    def test_collect_pairs_match_count(self):
        """Pairs count should match number of fields with confidence."""
        outputs = [_receipt_output()]
        annotations = [_receipt_annotation()]
        pairs = collect_confidence_correctness_pairs(outputs, annotations)
        # receipt has 5 fields with confidence (merchant, total, date, currency, items)
        assert len(pairs) == 5

    def test_calibration_buckets_structure(self):
        pairs = [(0.1, False), (0.4, True), (0.6, True), (0.8, True), (0.95, True)]
        buckets = compute_calibration_buckets(pairs)
        assert len(buckets) == 5  # default CALIBRATION_BUCKETS has 5
        for b in buckets:
            assert "bucket_low" in b
            assert "bucket_high" in b
            assert "mean_confidence" in b
            assert "actual_accuracy" in b
            assert "count" in b

    def test_perfect_calibration_buckets(self):
        """All correct at high confidence → accuracy = 1.0 in that bucket."""
        pairs = [(0.85, True), (0.88, True), (0.82, True)]
        buckets = compute_calibration_buckets(pairs)
        high_bucket = [b for b in buckets if b["bucket_low"] == 0.7][0]
        assert high_bucket["count"] == 3
        assert high_bucket["actual_accuracy"] == 1.0

    def test_all_wrong_in_bucket(self):
        pairs = [(0.85, False), (0.88, False)]
        buckets = compute_calibration_buckets(pairs)
        high_bucket = [b for b in buckets if b["bucket_low"] == 0.7][0]
        assert high_bucket["actual_accuracy"] == 0.0

    def test_empty_pairs(self):
        buckets = compute_calibration_buckets([])
        assert len(buckets) == 5
        for b in buckets:
            assert b["count"] == 0

    def test_ece_perfect(self):
        """Perfect calibration → ECE = 0."""
        buckets = [
            {"mean_confidence": 0.2, "actual_accuracy": 0.2, "count": 10},
            {"mean_confidence": 0.8, "actual_accuracy": 0.8, "count": 10},
        ]
        assert expected_calibration_error(buckets) == pytest.approx(0.0)

    def test_ece_worst_case(self):
        """All confident but all wrong → high ECE."""
        buckets = [
            {"mean_confidence": 0.9, "actual_accuracy": 0.0, "count": 10},
        ]
        assert expected_calibration_error(buckets) == pytest.approx(0.9)

    def test_ece_empty(self):
        assert expected_calibration_error([]) == 0.0

    def test_correlation_perfect(self):
        """Monotonically increasing → correlation = 1.0."""
        buckets = [
            {"mean_confidence": 0.2, "actual_accuracy": 0.1, "count": 5},
            {"mean_confidence": 0.5, "actual_accuracy": 0.5, "count": 5},
            {"mean_confidence": 0.8, "actual_accuracy": 0.9, "count": 5},
        ]
        assert calibration_correlation(buckets) == pytest.approx(1.0)

    def test_correlation_inverse(self):
        """Monotonically decreasing → correlation = -1.0."""
        buckets = [
            {"mean_confidence": 0.2, "actual_accuracy": 0.9, "count": 5},
            {"mean_confidence": 0.5, "actual_accuracy": 0.5, "count": 5},
            {"mean_confidence": 0.8, "actual_accuracy": 0.1, "count": 5},
        ]
        assert calibration_correlation(buckets) == pytest.approx(-1.0)

    def test_correlation_single_bucket(self):
        buckets = [{"mean_confidence": 0.8, "actual_accuracy": 0.8, "count": 10}]
        assert calibration_correlation(buckets) == 0.0

    def test_rank_function(self):
        assert _rank([10, 20, 30]) == [1.0, 2.0, 3.0]

    def test_rank_ties(self):
        assert _rank([10, 10, 30]) == [1.5, 1.5, 3.0]


# =====================================================================
# Linking evaluation
# =====================================================================

class TestLinkingEvaluation:
    def test_perfect_linking(self):
        outputs = [
            _receipt_output("r1", group_id="trip"),
            _receipt_output("r2", group_id="trip"),
        ]
        annotations = [
            _receipt_annotation("r1", group="trip"),
            _receipt_annotation("r2", group="trip"),
        ]
        result = evaluate_linking(outputs, annotations)
        assert result["pairwise_precision"] == 1.0
        assert result["pairwise_recall"] == 1.0
        assert result["pairwise_f1"] == 1.0

    def test_no_linking_when_expected(self):
        outputs = [
            _receipt_output("r1", group_id=None),
            _receipt_output("r2", group_id=None),
        ]
        annotations = [
            _receipt_annotation("r1", group="trip"),
            _receipt_annotation("r2", group="trip"),
        ]
        result = evaluate_linking(outputs, annotations)
        assert result["pairwise_recall"] == 0.0

    def test_no_expected_groups(self):
        outputs = [_receipt_output("r1", group_id=None)]
        annotations = [_receipt_annotation("r1", group=None)]
        result = evaluate_linking(outputs, annotations)
        assert result["pairwise_f1"] == 1.0  # nothing to link, nothing linked

    def test_false_positive_linking(self):
        outputs = [
            _receipt_output("r1", group_id="trip"),
            _conversation_output("c1"),
        ]
        outputs[1]["group_id"] = "trip"  # falsely linked
        annotations = [
            _receipt_annotation("r1", group=None),
            _conversation_annotation("c1"),
        ]
        result = evaluate_linking(outputs, annotations)
        assert result["pairwise_precision"] == 0.0

    def test_three_images_grouped(self):
        outputs = [
            _receipt_output("r1", group_id="trip"),
            _receipt_output("r2", group_id="trip"),
            _receipt_output("r3", group_id="trip"),
        ]
        annotations = [
            _receipt_annotation("r1", group="trip"),
            _receipt_annotation("r2", group="trip"),
            _receipt_annotation("r3", group="trip"),
        ]
        result = evaluate_linking(outputs, annotations)
        assert result["expected_pairs"] == 3  # C(3,2) = 3
        assert result["predicted_pairs"] == 3
        assert result["pairwise_f1"] == 1.0


# =====================================================================
# Failure handling evaluation
# =====================================================================

class TestFailureEvaluation:
    def test_correct_flags(self):
        outputs = [
            _receipt_output("img_003", failure_flags=["blurry_image"],
                            needs_clarification=True),
        ]
        annotations = [
            _receipt_annotation("img_003", failure_flags=["blurry_image"],
                                needs_clarification=True),
        ]
        result = evaluate_failure_handling(outputs, annotations)
        assert result["correct_flags"] == 1
        assert result["flag_accuracy"] == 1.0

    def test_missing_flags(self):
        outputs = [
            _receipt_output("img_003", failure_flags=[]),
        ]
        annotations = [
            _receipt_annotation("img_003", failure_flags=["blurry_image"]),
        ]
        result = evaluate_failure_handling(outputs, annotations)
        assert result["correct_flags"] == 0
        assert result["flag_accuracy"] == 0.0

    def test_clarification_accuracy(self):
        outputs = [
            _receipt_output("r1", needs_clarification=False),
            _receipt_output("r2", needs_clarification=True),
        ]
        annotations = [
            _receipt_annotation("r1", needs_clarification=False),
            _receipt_annotation("r2", needs_clarification=True),
        ]
        result = evaluate_failure_handling(outputs, annotations)
        assert result["correct_clarification"] == 2
        assert result["clarification_accuracy"] == 1.0

    def test_no_expected_flags(self):
        outputs = [_receipt_output("r1")]
        annotations = [_receipt_annotation("r1")]
        result = evaluate_failure_handling(outputs, annotations)
        assert result["flag_accuracy"] == 1.0  # no flags expected, none given


# =====================================================================
# Calendar hook evaluation
# =====================================================================

class TestCalendarEvaluation:
    def test_hook_detected(self):
        out = _conversation_output(
            "img_006",
            calendar_hook={"mentioned": True, "event_candidates": []},
        )
        ann = _conversation_annotation("img_006", calendar_hook=True)
        result = evaluate_calendar_hooks([out], [ann])
        assert result["detected_hooks"] == 1
        assert result["hook_recall"] == 1.0

    def test_hook_missed(self):
        out = _conversation_output("img_006", calendar_hook=None)
        ann = _conversation_annotation("img_006", calendar_hook=True)
        result = evaluate_calendar_hooks([out], [ann])
        assert result["detected_hooks"] == 0
        assert result["hook_recall"] == 0.0

    def test_no_hook_expected(self):
        out = _conversation_output("img_005")
        ann = _conversation_annotation("img_005", calendar_hook=None)
        result = evaluate_calendar_hooks([out], [ann])
        assert result["expected_hooks"] == 0
        assert result["hook_recall"] == 1.0


# =====================================================================
# Comparison table
# =====================================================================

class TestComparisonTable:
    def test_generates_csv(self, tmp_path):
        s_metrics = {
            "type_accuracy": 0.85,
            "extraction_accuracy_clean": 0.90,
            "extraction_accuracy_adversarial": 0.60,
            "calibration_ece": 0.08,
            "calibration_correlation": 0.95,
            "correct_flags": 5,
            "total_expected_flags": 6,
            "linking_f1": 0.80,
            "predicted_groups": 2,
            "detected_hooks": 1,
            "expected_hooks": 1,
        }
        b_metrics = {
            "type_accuracy": 0.70,
            "extraction_accuracy_clean": 0.75,
            "extraction_accuracy_adversarial": 0.40,
            "calibration_ece": 0.35,
            "calibration_correlation": 0.20,
            "correct_flags": 0,
            "total_expected_flags": 6,
            "linking_f1": 0.0,
            "predicted_groups": 0,
            "detected_hooks": 0,
            "expected_hooks": 1,
        }
        csv_path = tmp_path / "comparison.csv"
        rows = generate_comparison_table(s_metrics, b_metrics, csv_path)
        assert csv_path.exists()
        assert len(rows) == 9  # 9 comparison metrics

    def test_table_content(self, tmp_path):
        s_metrics = {"type_accuracy": 1.0, "extraction_accuracy_clean": 1.0,
                     "extraction_accuracy_adversarial": 0.5, "calibration_ece": 0.05,
                     "calibration_correlation": 0.9, "correct_flags": 3,
                     "total_expected_flags": 3, "linking_f1": 0.8,
                     "predicted_groups": 2, "detected_hooks": 1, "expected_hooks": 1}
        b_metrics = {"type_accuracy": 0.5, "extraction_accuracy_clean": 0.6,
                     "extraction_accuracy_adversarial": 0.3, "calibration_ece": 0.4,
                     "calibration_correlation": 0.1, "correct_flags": 0,
                     "total_expected_flags": 3, "linking_f1": 0.0,
                     "predicted_groups": 0, "detected_hooks": 0, "expected_hooks": 1}
        csv_path = tmp_path / "comparison.csv"
        rows = generate_comparison_table(s_metrics, b_metrics, csv_path)
        # Check first row is type classification
        assert "Type classification" in rows[0]["metric"]


# =====================================================================
# Calibration plot
# =====================================================================

class TestCalibrationPlot:
    def test_plot_creates_file(self, tmp_path):
        s_buckets = [
            {"mean_confidence": 0.2, "actual_accuracy": 0.15, "count": 3,
             "bucket_low": 0.0, "bucket_high": 0.3, "bucket_label": "[0.0, 0.3)"},
            {"mean_confidence": 0.8, "actual_accuracy": 0.85, "count": 10,
             "bucket_low": 0.7, "bucket_high": 0.9, "bucket_label": "[0.7, 0.9)"},
        ]
        b_buckets = [
            {"mean_confidence": 0.8, "actual_accuracy": 0.5, "count": 13,
             "bucket_low": 0.7, "bucket_high": 0.9, "bucket_label": "[0.7, 0.9)"},
        ]
        plot_path = tmp_path / "calibration.png"
        plot_calibration(s_buckets, b_buckets, plot_path)
        assert plot_path.exists()
        assert plot_path.stat().st_size > 0

    def test_plot_empty_buckets(self, tmp_path):
        """Should not crash on empty data."""
        plot_path = tmp_path / "calibration.png"
        plot_calibration([], [], plot_path)
        assert plot_path.exists()


# =====================================================================
# Aggregate metrics
# =====================================================================

class TestAggregateMetrics:
    def test_basic_aggregate(self):
        outputs = [_receipt_output()]
        annotations = [_receipt_annotation()]
        metrics = compute_aggregate_metrics(outputs, annotations)
        assert "type_accuracy" in metrics
        assert "extraction_accuracy_clean" in metrics
        assert "calibration_ece" in metrics
        assert "linking_f1" in metrics
        assert "flag_accuracy" in metrics
        assert "hook_recall" in metrics

    def test_type_accuracy(self):
        outputs = [_receipt_output(), _conversation_output()]
        annotations = [_receipt_annotation(), _conversation_annotation()]
        metrics = compute_aggregate_metrics(outputs, annotations)
        assert metrics["type_accuracy"] == 1.0

    def test_wrong_type_lowers_accuracy(self):
        out = _receipt_output()
        out["type"] = "document"  # wrong
        outputs = [out]
        annotations = [_receipt_annotation()]
        metrics = compute_aggregate_metrics(outputs, annotations)
        assert metrics["type_accuracy"] == 0.0

    def test_calibration_buckets_present(self):
        outputs = [_receipt_output()]
        annotations = [_receipt_annotation()]
        metrics = compute_aggregate_metrics(outputs, annotations)
        assert "calibration_buckets" in metrics
        assert len(metrics["calibration_buckets"]) == 5

    def test_clean_vs_adversarial_split(self):
        """img_001 is clean, img_003 is adversarial."""
        clean_out = _receipt_output("img_001")
        adv_out = _receipt_output("img_003")
        clean_ann = _receipt_annotation("img_001")
        adv_ann = _receipt_annotation("img_003")
        metrics = compute_aggregate_metrics([clean_out, adv_out], [clean_ann, adv_ann])
        assert metrics["clean_total"] > 0
        assert metrics["adv_total"] > 0


# =====================================================================
# Load from disk
# =====================================================================

class TestLoadFromDisk:
    def test_load_outputs(self, tmp_path):
        out = _receipt_output()
        (tmp_path / "img_001.json").write_text(json.dumps(out))
        loaded = load_outputs(tmp_path)
        assert len(loaded) == 1
        assert loaded[0]["image_id"] == "img_001"

    def test_load_annotations(self, tmp_path):
        ann = _receipt_annotation()
        (tmp_path / "img_001.json").write_text(json.dumps(ann))
        loaded = load_annotations(tmp_path)
        assert len(loaded) == 1
        assert loaded[0]["image_id"] == "img_001"

    def test_load_empty_dir(self, tmp_path):
        assert load_outputs(tmp_path) == []
        assert load_annotations(tmp_path) == []

    def test_load_multiple(self, tmp_path):
        for i in range(3):
            out = _receipt_output(f"img_{i:03d}")
            (tmp_path / f"img_{i:03d}.json").write_text(json.dumps(out))
        loaded = load_outputs(tmp_path)
        assert len(loaded) == 3


# =====================================================================
# Baseline flat confidence — calibration properties
# =====================================================================

class TestBaselineCalibration:
    def test_baseline_all_same_bucket(self):
        """Baseline with flat 0.8 puts everything in the [0.7, 0.9) bucket."""
        pairs = [(BASELINE_CONFIDENCE, True), (BASELINE_CONFIDENCE, False),
                 (BASELINE_CONFIDENCE, True), (BASELINE_CONFIDENCE, True)]
        buckets = compute_calibration_buckets(pairs)
        high_bucket = [b for b in buckets if b["bucket_low"] == 0.7][0]
        assert high_bucket["count"] == 4
        assert high_bucket["actual_accuracy"] == 0.75

    def test_baseline_zero_correlation(self):
        """All in one bucket → no correlation possible."""
        pairs = [(BASELINE_CONFIDENCE, True), (BASELINE_CONFIDENCE, False)]
        buckets = compute_calibration_buckets(pairs)
        non_empty = [b for b in buckets if b["count"] > 0]
        assert len(non_empty) == 1  # everything in one bucket
        # Can't compute meaningful correlation with 1 point
        assert calibration_correlation(buckets) == 0.0

    def test_structured_spreads_across_buckets(self):
        """Structured pipeline produces varied confidence → multiple buckets."""
        pairs = [
            (0.15, False), (0.25, False),       # low bucket
            (0.45, True),                         # medium bucket
            (0.65, True),                         # high-medium
            (0.82, True), (0.88, True),           # high
            (0.95, True),                         # very high
        ]
        buckets = compute_calibration_buckets(pairs)
        non_empty = [b for b in buckets if b["count"] > 0]
        assert len(non_empty) >= 3  # spread across multiple buckets


# =====================================================================
# Edge cases
# =====================================================================

class TestEdgeCases:
    def test_no_annotations_match(self):
        outputs = [_receipt_output("img_999")]
        annotations = [_receipt_annotation("img_001")]
        pairs = collect_confidence_correctness_pairs(outputs, annotations)
        assert pairs == []

    def test_empty_outputs(self):
        metrics = compute_aggregate_metrics([], [])
        assert metrics["type_accuracy"] == 0.0
        assert metrics["calibration_ece"] == 0.0

    def test_adversarial_ids_set(self):
        """Verify adversarial image IDs are defined."""
        assert "img_003" in _ADVERSARIAL_IDS  # blurry receipt
        assert "img_004" in _ADVERSARIAL_IDS  # cropped receipt
        assert "img_007" in _ADVERSARIAL_IDS  # cropped conversation
        assert "img_013" in _ADVERSARIAL_IDS  # messy whiteboard
        assert "img_014" in _ADVERSARIAL_IDS  # blurry whiteboard

    def test_receipt_null_total_annotation(self):
        """Cropped receipt: total is null in annotation → None matches None."""
        out = _receipt_output(total=None)
        out["extracted_entities"]["total"] = None
        ann = _receipt_annotation(total=None)
        ann["expected_entities"]["total"] = None
        results = evaluate_receipt(out, ann)
        assert results["total"]["correct"] is True  # None == None
