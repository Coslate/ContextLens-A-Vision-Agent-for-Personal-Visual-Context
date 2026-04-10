"""Evaluation metrics, calibration plot, and baseline comparison.

Usage:
    python -m scripts.run_evaluation
    python -m scripts.run_evaluation --structured-dir data/outputs/structured \
        --baseline-dir data/outputs/baseline --annotations-dir data/annotations
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from contextlens.config import CALIBRATION_BUCKETS, FUZZY_MATCH_THRESHOLD
from contextlens.schemas import Annotation, ImageType

# =====================================================================
# Field matching helpers
# =====================================================================


def exact_match(predicted: Any, expected: Any) -> bool:
    """Case-insensitive exact string/number match."""
    if predicted is None and expected is None:
        return True
    if predicted is None or expected is None:
        return False
    if isinstance(predicted, (int, float)) and isinstance(expected, (int, float)):
        return math.isclose(predicted, expected, rel_tol=1e-3)
    return str(predicted).strip().lower() == str(expected).strip().lower()


def fuzzy_match(predicted: str, expected: str, threshold: int = FUZZY_MATCH_THRESHOLD) -> bool:
    """Fuzzy string match using rapidfuzz."""
    if predicted is None or expected is None:
        return predicted is None and expected is None
    try:
        from rapidfuzz import fuzz
        score = fuzz.ratio(str(predicted).lower(), str(expected).lower())
        return score >= threshold
    except ImportError:
        return exact_match(predicted, expected)


def list_fuzzy_recall(predicted_list: list[str], expected_list: list[str],
                      threshold: int = FUZZY_MATCH_THRESHOLD) -> float:
    """Compute recall: fraction of expected items matched by predicted (fuzzy)."""
    if not expected_list:
        return 1.0
    matched = 0
    for exp in expected_list:
        for pred in predicted_list:
            if fuzzy_match(pred, exp, threshold):
                matched += 1
                break
    return matched / len(expected_list)


def list_fuzzy_precision(predicted_list: list[str], expected_list: list[str],
                         threshold: int = FUZZY_MATCH_THRESHOLD) -> float:
    """Compute precision: fraction of predicted items that match some expected."""
    if not predicted_list:
        return 1.0 if not expected_list else 0.0
    matched = 0
    for pred in predicted_list:
        for exp in expected_list:
            if fuzzy_match(pred, exp, threshold):
                matched += 1
                break
    return matched / len(predicted_list)


def f1_score(precision: float, recall: float) -> float:
    """Harmonic mean of precision and recall."""
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# =====================================================================
# Extraction accuracy — per-type field evaluation
# =====================================================================


def evaluate_receipt(output: dict, annotation: dict) -> dict[str, dict]:
    """Evaluate receipt extraction fields."""
    entities = output.get("extracted_entities", {})
    expected = annotation.get("expected_entities", {})
    results: dict[str, dict] = {}

    # merchant — exact
    results["merchant"] = {
        "correct": exact_match(entities.get("merchant"), expected.get("merchant")),
        "predicted": entities.get("merchant"),
        "expected": expected.get("merchant"),
    }

    # total — exact numeric
    results["total"] = {
        "correct": exact_match(entities.get("total"), expected.get("total")),
        "predicted": entities.get("total"),
        "expected": expected.get("total"),
    }

    # date — exact
    results["date"] = {
        "correct": exact_match(entities.get("date"), expected.get("date")),
        "predicted": entities.get("date"),
        "expected": expected.get("date"),
    }

    # currency — exact
    results["currency"] = {
        "correct": exact_match(entities.get("currency"), expected.get("currency")),
        "predicted": entities.get("currency"),
        "expected": expected.get("currency"),
    }

    # items — fuzzy list match on names
    pred_items = entities.get("items", [])
    exp_items = expected.get("items", [])
    pred_names = [it.get("name", "") if isinstance(it, dict) else getattr(it, "name", "")
                  for it in pred_items]
    exp_names = [it.get("name", "") if isinstance(it, dict) else it for it in exp_items]
    prec = list_fuzzy_precision(pred_names, exp_names)
    rec = list_fuzzy_recall(pred_names, exp_names)
    results["items"] = {
        "correct": f1_score(prec, rec) >= 0.5,
        "f1": f1_score(prec, rec),
        "precision": prec,
        "recall": rec,
        "predicted_count": len(pred_items),
        "expected_count": len(exp_items),
    }

    return results


def evaluate_conversation(output: dict, annotation: dict) -> dict[str, dict]:
    """Evaluate conversation extraction fields."""
    entities = output.get("extracted_entities", {})
    expected = annotation.get("expected_entities", {})
    results: dict[str, dict] = {}

    # participants — fuzzy list
    pred = entities.get("participants", [])
    exp = expected.get("participants", [])
    prec = list_fuzzy_precision(pred, exp)
    rec = list_fuzzy_recall(pred, exp)
    results["participants"] = {
        "correct": rec >= 0.5,
        "f1": f1_score(prec, rec),
        "precision": prec,
        "recall": rec,
    }

    # key_topics — fuzzy list
    pred = entities.get("key_topics", [])
    exp = expected.get("key_topics", [])
    rec = list_fuzzy_recall(pred, exp)
    prec = list_fuzzy_precision(pred, exp)
    results["key_topics"] = {
        "correct": rec >= 0.3,
        "f1": f1_score(prec, rec),
        "recall": rec,
    }

    # action_items — fuzzy list
    pred = entities.get("action_items", [])
    exp = expected.get("action_items", [])
    rec = list_fuzzy_recall(pred, exp)
    prec = list_fuzzy_precision(pred, exp)
    results["action_items"] = {
        "correct": rec >= 0.3,
        "f1": f1_score(prec, rec),
        "recall": rec,
    }

    return results


def evaluate_document(output: dict, annotation: dict) -> dict[str, dict]:
    """Evaluate document extraction fields."""
    entities = output.get("extracted_entities", {})
    expected = annotation.get("expected_entities", {})
    results: dict[str, dict] = {}

    # document_kind — exact
    results["document_kind"] = {
        "correct": exact_match(entities.get("document_kind"), expected.get("document_kind")),
        "predicted": entities.get("document_kind"),
        "expected": expected.get("document_kind"),
    }

    # structured_fields — fuzzy value match per key
    pred_fields = entities.get("structured_fields", {})
    exp_fields = expected.get("structured_fields", {})
    matched = 0
    total = len(exp_fields)
    for key, exp_val in exp_fields.items():
        # Try exact key first, then fuzzy key match
        pred_val = pred_fields.get(key)
        if pred_val is None:
            for pk, pv in pred_fields.items():
                if fuzzy_match(pk, key, 80):
                    pred_val = pv
                    break
        if pred_val is not None and fuzzy_match(str(pred_val), str(exp_val), 70):
            matched += 1

    recall = matched / total if total else 1.0
    results["structured_fields"] = {
        "correct": recall >= 0.3,
        "recall": recall,
        "matched": matched,
        "expected_count": total,
        "predicted_count": len(pred_fields),
    }

    return results


def evaluate_whiteboard(output: dict, annotation: dict) -> dict[str, dict]:
    """Evaluate whiteboard extraction fields."""
    entities = output.get("extracted_entities", {})
    expected = annotation.get("expected_entities", {})
    results: dict[str, dict] = {}

    # For whiteboard, entities may be nested under inferred_structure
    structure = entities.get("inferred_structure", {})
    if not structure and isinstance(entities, dict):
        structure = entities

    # owners — fuzzy list
    pred = structure.get("owners", [])
    exp = expected.get("owners", [])
    rec = list_fuzzy_recall(pred, exp)
    results["owners"] = {
        "correct": rec >= 0.3,
        "recall": rec,
    }

    # tasks — fuzzy list
    pred = structure.get("tasks", [])
    exp = expected.get("tasks", [])
    rec = list_fuzzy_recall(pred, exp)
    results["tasks"] = {
        "correct": rec >= 0.3,
        "recall": rec,
    }

    # project_tags — fuzzy list
    pred = structure.get("project_tags", [])
    exp = expected.get("project_tags", [])
    rec = list_fuzzy_recall(pred, exp)
    results["project_tags"] = {
        "correct": rec >= 0.3,
        "recall": rec,
    }

    # text_blocks — count-based
    text_blocks = entities.get("text_blocks", [])
    results["text_blocks"] = {
        "correct": len(text_blocks) > 0,
        "count": len(text_blocks),
    }

    return results


_TYPE_EVALUATORS = {
    "receipt": evaluate_receipt,
    "conversation": evaluate_conversation,
    "document": evaluate_document,
    "whiteboard": evaluate_whiteboard,
}


def evaluate_extraction(output: dict, annotation: dict) -> dict[str, dict]:
    """Evaluate extraction accuracy for a single image output vs annotation."""
    img_type = annotation.get("expected_type", "")
    evaluator = _TYPE_EVALUATORS.get(img_type)
    if evaluator is None:
        return {}
    return evaluator(output, annotation)


# =====================================================================
# Type classification accuracy
# =====================================================================


def evaluate_type_classification(output: dict, annotation: dict) -> bool:
    """Check if the predicted type matches expected type."""
    return output.get("type", "") == annotation.get("expected_type", "")


# =====================================================================
# Confidence calibration
# =====================================================================


def collect_confidence_correctness_pairs(
    outputs: list[dict],
    annotations: list[dict],
) -> list[tuple[float, bool]]:
    """Collect (confidence, is_correct) pairs across all images and fields.

    Each field extraction is matched against the annotation and paired with
    its field-level confidence score.
    """
    pairs: list[tuple[float, bool]] = []

    annotation_map = {a["image_id"]: a for a in annotations}

    for output in outputs:
        image_id = output["image_id"]
        annotation = annotation_map.get(image_id)
        if annotation is None:
            continue

        # Evaluate extraction
        field_results = evaluate_extraction(output, annotation)
        field_confidence = output.get("field_confidence", {})

        for field_name, result in field_results.items():
            conf = field_confidence.get(field_name)
            if conf is None:
                # Try to find confidence under related names
                conf = field_confidence.get(field_name + "s")  # e.g. "item" -> "items"
                if conf is None:
                    continue
            is_correct = result.get("correct", False)
            pairs.append((conf, is_correct))

    return pairs


def compute_calibration_buckets(
    pairs: list[tuple[float, bool]],
    buckets: list[tuple[float, float]] | None = None,
) -> list[dict]:
    """Compute calibration statistics per confidence bucket.

    Returns a list of dicts, each containing:
      - bucket_low, bucket_high: bucket boundaries
      - bucket_label: human-readable label
      - mean_confidence: average confidence in bucket
      - actual_accuracy: fraction of correct fields in bucket
      - count: number of pairs in bucket
    """
    if buckets is None:
        buckets = CALIBRATION_BUCKETS

    bucket_data: list[dict] = []
    for low, high in buckets:
        in_bucket = [(c, correct) for c, correct in pairs if low <= c < high]
        if not in_bucket:
            bucket_data.append({
                "bucket_low": low,
                "bucket_high": high,
                "bucket_label": f"[{low:.1f}, {high:.1f})",
                "mean_confidence": (low + high) / 2,
                "actual_accuracy": 0.0,
                "count": 0,
            })
            continue

        confs = [c for c, _ in in_bucket]
        correct_count = sum(1 for _, correct in in_bucket if correct)
        bucket_data.append({
            "bucket_low": low,
            "bucket_high": high,
            "bucket_label": f"[{low:.1f}, {high:.1f})",
            "mean_confidence": sum(confs) / len(confs),
            "actual_accuracy": correct_count / len(in_bucket),
            "count": len(in_bucket),
        })

    return bucket_data


def plot_calibration(
    structured_buckets: list[dict],
    baseline_buckets: list[dict],
    output_path: str | Path,
) -> None:
    """Generate the calibration plot (THE #1 ARTIFACT).

    Plots mean confidence vs actual accuracy for both structured pipeline
    and baseline, plus the perfect calibration diagonal.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Perfect calibration diagonal
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")

    # Structured pipeline
    s_confs = [b["mean_confidence"] for b in structured_buckets if b["count"] > 0]
    s_accs = [b["actual_accuracy"] for b in structured_buckets if b["count"] > 0]
    s_counts = [b["count"] for b in structured_buckets if b["count"] > 0]
    if s_confs:
        ax.plot(s_confs, s_accs, "bo-", markersize=8, linewidth=2,
                label="Structured Pipeline")
        for x, y, n in zip(s_confs, s_accs, s_counts):
            ax.annotate(f"n={n}", (x, y), textcoords="offset points",
                        xytext=(5, 5), fontsize=8, color="blue")

    # Baseline
    b_confs = [b["mean_confidence"] for b in baseline_buckets if b["count"] > 0]
    b_accs = [b["actual_accuracy"] for b in baseline_buckets if b["count"] > 0]
    b_counts = [b["count"] for b in baseline_buckets if b["count"] > 0]
    if b_confs:
        ax.plot(b_confs, b_accs, "rs-", markersize=8, linewidth=2,
                label="Baseline (flat 0.8)")
        for x, y, n in zip(b_confs, b_accs, b_counts):
            ax.annotate(f"n={n}", (x, y), textcoords="offset points",
                        xytext=(5, -12), fontsize=8, color="red")

    ax.set_xlabel("Mean Predicted Confidence", fontsize=12)
    ax.set_ylabel("Actual Accuracy", fontsize=12)
    ax.set_title("Confidence Calibration: Structured vs Baseline", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


# =====================================================================
# Calibration correlation (ECE & Spearman)
# =====================================================================


def expected_calibration_error(bucket_data: list[dict]) -> float:
    """Compute Expected Calibration Error (ECE).

    Weighted average of |accuracy - confidence| per bucket.
    """
    total_samples = sum(b["count"] for b in bucket_data)
    if total_samples == 0:
        return 0.0
    ece = 0.0
    for b in bucket_data:
        if b["count"] > 0:
            ece += (b["count"] / total_samples) * abs(
                b["actual_accuracy"] - b["mean_confidence"]
            )
    return ece


def calibration_correlation(bucket_data: list[dict]) -> float:
    """Compute Spearman rank correlation between confidence and accuracy.

    Returns correlation coefficient in [-1, 1]. Higher is better
    (monotonically increasing = 1.0).
    """
    non_empty = [b for b in bucket_data if b["count"] > 0]
    if len(non_empty) < 2:
        return 0.0

    confs = [b["mean_confidence"] for b in non_empty]
    accs = [b["actual_accuracy"] for b in non_empty]

    # Spearman = Pearson on ranks
    n = len(confs)
    conf_ranks = _rank(confs)
    acc_ranks = _rank(accs)

    mean_cr = sum(conf_ranks) / n
    mean_ar = sum(acc_ranks) / n

    num = sum((cr - mean_cr) * (ar - mean_ar) for cr, ar in zip(conf_ranks, acc_ranks))
    den_c = math.sqrt(sum((cr - mean_cr) ** 2 for cr in conf_ranks))
    den_a = math.sqrt(sum((ar - mean_ar) ** 2 for ar in acc_ranks))

    if den_c * den_a == 0:
        return 0.0
    return num / (den_c * den_a)


def _rank(values: list[float]) -> list[float]:
    """Compute ranks (1-based, average ties)."""
    indexed = sorted(enumerate(values), key=lambda x: x[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j + 1) / 2  # average rank for ties
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


# =====================================================================
# Linking quality evaluation
# =====================================================================


def evaluate_linking(
    outputs: list[dict],
    annotations: list[dict],
) -> dict:
    """Evaluate cross-image linking quality.

    Computes pairwise precision and recall for group assignments.
    """
    annotation_map = {a["image_id"]: a for a in annotations}

    # Build predicted pairs (same group_id)
    pred_groups: dict[str, list[str]] = defaultdict(list)
    for out in outputs:
        gid = out.get("group_id")
        if gid is not None:
            pred_groups[gid].append(out["image_id"])

    pred_pairs: set[tuple[str, str]] = set()
    for gid, members in pred_groups.items():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                pair = tuple(sorted([members[i], members[j]]))
                pred_pairs.add(pair)

    # Build expected pairs (same expected_group)
    exp_groups: dict[str, list[str]] = defaultdict(list)
    for ann in annotations:
        gid = ann.get("expected_group")
        if gid is not None:
            exp_groups[gid].append(ann["image_id"])

    exp_pairs: set[tuple[str, str]] = set()
    for gid, members in exp_groups.items():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                pair = tuple(sorted([members[i], members[j]]))
                exp_pairs.add(pair)

    # Pairwise precision and recall
    if pred_pairs:
        correct_pred = pred_pairs & exp_pairs
        precision = len(correct_pred) / len(pred_pairs)
    else:
        precision = 1.0 if not exp_pairs else 0.0

    if exp_pairs:
        correct_exp = exp_pairs & pred_pairs
        recall = len(correct_exp) / len(exp_pairs)
    else:
        recall = 1.0

    return {
        "pairwise_precision": precision,
        "pairwise_recall": recall,
        "pairwise_f1": f1_score(precision, recall),
        "predicted_pairs": len(pred_pairs),
        "expected_pairs": len(exp_pairs),
        "correct_pairs": len(pred_pairs & exp_pairs),
        "predicted_groups": len(pred_groups),
        "expected_groups": len(exp_groups),
    }


# =====================================================================
# Failure robustness evaluation
# =====================================================================


def evaluate_failure_handling(
    outputs: list[dict],
    annotations: list[dict],
) -> dict:
    """Evaluate failure flag correctness and confidence behaviour."""
    annotation_map = {a["image_id"]: a for a in annotations}

    total_expected_flags = 0
    correct_flags = 0
    total_clarification = 0
    correct_clarification = 0
    adversarial_confidence_dropped = 0
    adversarial_count = 0

    for output in outputs:
        image_id = output["image_id"]
        ann = annotation_map.get(image_id)
        if ann is None:
            continue

        exp_flags = set(ann.get("expected_failure_flags", []))
        pred_flags = set(output.get("failure_flags", []))

        # Count correct failure flags
        for flag in exp_flags:
            total_expected_flags += 1
            if flag in pred_flags:
                correct_flags += 1

        # Clarification correctness
        exp_clar = ann.get("expected_needs_clarification")
        if exp_clar is not None:
            total_clarification += 1
            if output.get("needs_clarification", False) == exp_clar:
                correct_clarification += 1

        # Confidence drop on adversarial images
        if exp_flags:
            adversarial_count += 1
            conf_values = list(output.get("field_confidence", {}).values())
            if conf_values:
                avg_conf = sum(conf_values) / len(conf_values)
                if avg_conf < 0.7:  # meaningfully reduced from default
                    adversarial_confidence_dropped += 1

    return {
        "total_expected_flags": total_expected_flags,
        "correct_flags": correct_flags,
        "flag_accuracy": correct_flags / total_expected_flags if total_expected_flags else 1.0,
        "total_clarification": total_clarification,
        "correct_clarification": correct_clarification,
        "clarification_accuracy": (
            correct_clarification / total_clarification if total_clarification else 1.0
        ),
        "adversarial_count": adversarial_count,
        "adversarial_confidence_dropped": adversarial_confidence_dropped,
    }


# =====================================================================
# Calendar hook evaluation
# =====================================================================


def evaluate_calendar_hooks(
    outputs: list[dict],
    annotations: list[dict],
) -> dict:
    """Evaluate calendar hook detection."""
    annotation_map = {a["image_id"]: a for a in annotations}

    expected_hooks = 0
    detected_hooks = 0

    for output in outputs:
        image_id = output["image_id"]
        ann = annotation_map.get(image_id)
        if ann is None:
            continue

        exp_hook = ann.get("expected_calendar_hook")
        if exp_hook is True:
            expected_hooks += 1
            hook = output.get("calendar_hook")
            if hook is not None and (
                isinstance(hook, dict) and hook.get("mentioned")
                or hasattr(hook, "mentioned") and hook.mentioned
            ):
                detected_hooks += 1

    return {
        "expected_hooks": expected_hooks,
        "detected_hooks": detected_hooks,
        "hook_recall": detected_hooks / expected_hooks if expected_hooks else 1.0,
    }


# =====================================================================
# Comparison table generation
# =====================================================================


def generate_comparison_table(
    structured_metrics: dict,
    baseline_metrics: dict,
    output_path: str | Path,
) -> list[dict]:
    """Generate structured vs baseline comparison CSV.

    Returns list of row dicts and writes to CSV.
    """
    rows = []

    def _add(metric: str, structured_val: Any, baseline_val: Any) -> None:
        rows.append({
            "metric": metric,
            "structured_pipeline": _fmt(structured_val),
            "baseline": _fmt(baseline_val),
        })

    def _fmt(v: Any) -> str:
        if isinstance(v, float):
            return f"{v:.3f}"
        return str(v)

    _add("Type classification accuracy",
         structured_metrics.get("type_accuracy", 0),
         baseline_metrics.get("type_accuracy", 0))
    _add("Extraction accuracy (clean)",
         structured_metrics.get("extraction_accuracy_clean", 0),
         baseline_metrics.get("extraction_accuracy_clean", 0))
    _add("Extraction accuracy (adversarial)",
         structured_metrics.get("extraction_accuracy_adversarial", 0),
         baseline_metrics.get("extraction_accuracy_adversarial", 0))
    _add("Calibration ECE (lower is better)",
         structured_metrics.get("calibration_ece", 0),
         baseline_metrics.get("calibration_ece", 0))
    _add("Calibration correlation",
         structured_metrics.get("calibration_correlation", 0),
         baseline_metrics.get("calibration_correlation", 0))
    _add("Failure flags correct",
         f"{structured_metrics.get('correct_flags', 0)}/{structured_metrics.get('total_expected_flags', 0)}",
         f"{baseline_metrics.get('correct_flags', 0)}/{baseline_metrics.get('total_expected_flags', 0)}")
    _add("Linking pairwise F1",
         structured_metrics.get("linking_f1", 0),
         baseline_metrics.get("linking_f1", 0))
    _add("Groups identified",
         structured_metrics.get("predicted_groups", 0),
         baseline_metrics.get("predicted_groups", 0))
    _add("Calendar hooks detected",
         f"{structured_metrics.get('detected_hooks', 0)}/{structured_metrics.get('expected_hooks', 0)}",
         f"{baseline_metrics.get('detected_hooks', 0)}/{baseline_metrics.get('expected_hooks', 0)}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "structured_pipeline", "baseline"])
        writer.writeheader()
        writer.writerows(rows)

    return rows


# =====================================================================
# Aggregate metrics for a pipeline run
# =====================================================================

# Images that are "adversarial" (have expected failure flags)
_ADVERSARIAL_IDS = {"img_002", "img_003", "img_004", "img_007", "img_010", "img_013", "img_014"}


def compute_aggregate_metrics(
    outputs: list[dict],
    annotations: list[dict],
) -> dict:
    """Compute all metrics for a set of pipeline outputs."""
    annotation_map = {a["image_id"]: a for a in annotations}

    # Type classification accuracy
    type_correct = 0
    type_total = 0
    for out in outputs:
        ann = annotation_map.get(out["image_id"])
        if ann is None:
            continue
        type_total += 1
        if evaluate_type_classification(out, ann):
            type_correct += 1

    # Extraction accuracy (split clean vs adversarial)
    clean_correct = 0
    clean_total = 0
    adv_correct = 0
    adv_total = 0

    for out in outputs:
        ann = annotation_map.get(out["image_id"])
        if ann is None:
            continue
        field_results = evaluate_extraction(out, ann)
        for field_name, result in field_results.items():
            is_correct = result.get("correct", False)
            if out["image_id"] in _ADVERSARIAL_IDS:
                adv_total += 1
                if is_correct:
                    adv_correct += 1
            else:
                clean_total += 1
                if is_correct:
                    clean_correct += 1

    # Calibration
    pairs = collect_confidence_correctness_pairs(outputs, annotations)
    cal_buckets = compute_calibration_buckets(pairs)

    # Linking
    linking = evaluate_linking(outputs, annotations)

    # Failure handling
    failure = evaluate_failure_handling(outputs, annotations)

    # Calendar hooks
    calendar = evaluate_calendar_hooks(outputs, annotations)

    return {
        "type_accuracy": type_correct / type_total if type_total else 0.0,
        "type_correct": type_correct,
        "type_total": type_total,
        "extraction_accuracy_clean": clean_correct / clean_total if clean_total else 0.0,
        "extraction_accuracy_adversarial": adv_correct / adv_total if adv_total else 0.0,
        "clean_correct": clean_correct,
        "clean_total": clean_total,
        "adv_correct": adv_correct,
        "adv_total": adv_total,
        "calibration_ece": expected_calibration_error(cal_buckets),
        "calibration_correlation": calibration_correlation(cal_buckets),
        "calibration_buckets": cal_buckets,
        "linking_f1": linking["pairwise_f1"],
        "linking_precision": linking["pairwise_precision"],
        "linking_recall": linking["pairwise_recall"],
        "predicted_groups": linking["predicted_groups"],
        "expected_groups": linking["expected_groups"],
        "predicted_pairs": linking["predicted_pairs"],
        "expected_pairs": linking["expected_pairs"],
        "correct_flags": failure["correct_flags"],
        "total_expected_flags": failure["total_expected_flags"],
        "flag_accuracy": failure["flag_accuracy"],
        "clarification_accuracy": failure["clarification_accuracy"],
        "detected_hooks": calendar["detected_hooks"],
        "expected_hooks": calendar["expected_hooks"],
        "hook_recall": calendar["hook_recall"],
    }


# =====================================================================
# Load outputs and annotations from disk
# =====================================================================


def load_outputs(output_dir: Path) -> list[dict]:
    """Load pipeline output JSONs from directory."""
    outputs = []
    for p in sorted(output_dir.glob("*.json")):
        outputs.append(json.loads(p.read_text()))
    return outputs


def load_annotations(annotations_dir: Path) -> list[dict]:
    """Load annotation JSONs from directory."""
    annotations = []
    for p in sorted(annotations_dir.glob("*.json")):
        annotations.append(json.loads(p.read_text()))
    return annotations


# =====================================================================
# Main
# =====================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ContextLens evaluation: metrics, calibration plot, comparison.",
    )
    parser.add_argument(
        "--structured-dir", default="data/outputs/structured",
        help="Directory with structured pipeline output JSONs.",
    )
    parser.add_argument(
        "--baseline-dir", default="data/outputs/baseline",
        help="Directory with baseline output JSONs.",
    )
    parser.add_argument(
        "--annotations-dir", default="data/annotations",
        help="Directory with ground truth annotation JSONs.",
    )
    parser.add_argument(
        "--results-dir", default="results",
        help="Output directory for metrics, tables, and figures.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    annotations = load_annotations(Path(args.annotations_dir))
    print(f"Loaded {len(annotations)} annotations.\n")

    # --- Structured pipeline ---
    structured_dir = Path(args.structured_dir)
    if structured_dir.exists():
        structured_outputs = load_outputs(structured_dir)
        print(f"Loaded {len(structured_outputs)} structured outputs.")
        structured_metrics = compute_aggregate_metrics(structured_outputs, annotations)
    else:
        print(f"Structured output dir not found: {structured_dir}")
        structured_outputs = []
        structured_metrics = {}

    # --- Baseline ---
    baseline_dir = Path(args.baseline_dir)
    if baseline_dir.exists():
        baseline_outputs = load_outputs(baseline_dir)
        print(f"Loaded {len(baseline_outputs)} baseline outputs.")
        baseline_metrics = compute_aggregate_metrics(baseline_outputs, annotations)
    else:
        print(f"Baseline output dir not found: {baseline_dir}")
        baseline_outputs = []
        baseline_metrics = {}

    # --- Calibration plot ---
    print("\nGenerating calibration plot...")
    s_buckets = structured_metrics.get("calibration_buckets", [])
    b_buckets = baseline_metrics.get("calibration_buckets", [])
    plot_path = results_dir / "figures" / "calibration.png"
    plot_calibration(s_buckets, b_buckets, plot_path)
    print(f"  Saved to {plot_path}")

    # --- Comparison table ---
    print("Generating comparison table...")
    table_path = results_dir / "tables" / "comparison.csv"
    rows = generate_comparison_table(structured_metrics, baseline_metrics, table_path)
    print(f"  Saved to {table_path}")

    # --- Print results ---
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    for row in rows:
        print(f"  {row['metric']:40s}  {row['structured_pipeline']:>12s}  {row['baseline']:>12s}")
    print("=" * 60)

    # --- Save metric JSONs ---
    metrics_dir = results_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    if structured_metrics:
        s_save = {k: v for k, v in structured_metrics.items() if k != "calibration_buckets"}
        (metrics_dir / "structured_metrics.json").write_text(json.dumps(s_save, indent=2))

    if baseline_metrics:
        b_save = {k: v for k, v in baseline_metrics.items() if k != "calibration_buckets"}
        (metrics_dir / "baseline_metrics.json").write_text(json.dumps(b_save, indent=2))

    print("\nDone.")


if __name__ == "__main__":
    main()
