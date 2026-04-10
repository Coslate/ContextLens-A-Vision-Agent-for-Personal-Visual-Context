"""ContextLens one-command demo.

Generates test images, runs both structured and baseline pipelines, performs
cross-image linking, stores results in the memory store, runs sample queries,
executes evaluation, and prints formatted results.

Usage:
    python -m scripts.run_demo              # full run
    python -m scripts.run_demo --dry-run    # validate without heavy processing
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TEST_IMAGES_DIR = DATA_DIR / "test_images"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
STRUCTURED_OUTPUT_DIR = DATA_DIR / "outputs" / "structured"
BASELINE_OUTPUT_DIR = DATA_DIR / "outputs" / "baseline"
RESULTS_DIR = PROJECT_ROOT / "results"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _header(title: str) -> None:
    """Print a section header."""
    width = 60
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def _subheader(title: str) -> None:
    print(f"\n--- {title} ---")


# ---------------------------------------------------------------------------
# Step 1: Generate test images
# ---------------------------------------------------------------------------

def step_generate_images() -> list[Path]:
    """Generate 14 synthetic test images if they don't already exist."""
    _header("Step 1: Generate Synthetic Test Images")

    existing = sorted(TEST_IMAGES_DIR.glob("*.png")) if TEST_IMAGES_DIR.exists() else []
    if len(existing) >= 14:
        print(f"  Found {len(existing)} images in {TEST_IMAGES_DIR} — skipping generation.")
        return existing

    print("  Generating 14 synthetic test images...")
    from scripts.generate_test_images import main as generate_main
    generate_main()

    images = sorted(TEST_IMAGES_DIR.glob("*.png"))
    print(f"  Generated {len(images)} images.")
    return images


# ---------------------------------------------------------------------------
# Step 2: Run structured pipeline
# ---------------------------------------------------------------------------

def step_structured_pipeline(image_paths: list[Path]) -> list[dict]:
    """Run the ContextLens structured pipeline on all images."""
    _header("Step 2: Structured Pipeline (Conditioned Extraction)")

    from contextlens.pipeline import process_image

    STRUCTURED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    for path in image_paths:
        image_id = path.stem
        print(f"  [{image_id}] Processing...", end=" ")
        output = process_image(str(path), image_id=image_id)
        result = output.model_dump(mode="json")
        results.append(result)

        out_path = STRUCTURED_OUTPUT_DIR / f"{image_id}.json"
        out_path.write_text(json.dumps(result, indent=2))
        print(f"type={output.type.value}  conf={output.type_confidence:.2f}  "
              f"flags={output.failure_flags}")

    print(f"\n  Saved {len(results)} outputs to {STRUCTURED_OUTPUT_DIR}")
    return results


# ---------------------------------------------------------------------------
# Step 3: Run baseline
# ---------------------------------------------------------------------------

def step_baseline(image_paths: list[Path]) -> list[dict]:
    """Run the generic unified parser baseline on all images."""
    _header("Step 3: Baseline Pipeline (Generic Unified Parser)")

    from contextlens.baseline import baseline_process
    from contextlens.ocr import run_ocr
    from contextlens.preprocess import preprocess_image

    BASELINE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    for path in image_paths:
        image_id = path.stem
        print(f"  [{image_id}] Processing...", end=" ")

        image_arr, quality = preprocess_image(str(path))
        ocr_result = run_ocr(image_arr)
        output = baseline_process(ocr_result, image_id=image_id, quality=quality)

        result = output.model_dump(mode="json")
        results.append(result)

        out_path = BASELINE_OUTPUT_DIR / f"{image_id}.json"
        out_path.write_text(json.dumps(result, indent=2))
        print(f"type={output.type.value}  conf={output.type_confidence:.2f}")

    print(f"\n  Saved {len(results)} outputs to {BASELINE_OUTPUT_DIR}")
    return results


# ---------------------------------------------------------------------------
# Step 4: Cross-image linking (BEV-JEPA analog)
# ---------------------------------------------------------------------------

def step_linking(structured_results: list[dict]) -> list[dict]:
    """Run the cross-image context fuser on structured outputs."""
    _header("Step 4: Cross-Image Context Fusion (BEV-JEPA Analog)")

    from contextlens.linker import link_outputs
    from contextlens.schemas import ImageOutput

    # Reconstruct ImageOutput objects
    outputs = [ImageOutput.model_validate(r) for r in structured_results]

    print(f"  Linking {len(outputs)} images...")
    linked_outputs, group_summaries = link_outputs(outputs)

    # Report groups
    if group_summaries:
        print(f"  Found {len(group_summaries)} group(s):")
        for gid, summary in group_summaries.items():
            members = [o.image_id for o in linked_outputs if o.group_id == gid]
            print(f"    [{gid}] {len(members)} members: {', '.join(members)}")
            print(f"      Summary: {summary}")
    else:
        print("  No groups detected.")

    # Update saved outputs with group_id
    updated_results = []
    for output in linked_outputs:
        result = output.model_dump(mode="json")
        updated_results.append(result)
        out_path = STRUCTURED_OUTPUT_DIR / f"{output.image_id}.json"
        out_path.write_text(json.dumps(result, indent=2))

    return updated_results


# ---------------------------------------------------------------------------
# Step 5: Memory store + queries
# ---------------------------------------------------------------------------

def step_memory_and_queries(structured_results: list[dict]) -> None:
    """Store outputs in SQLite and run sample queries."""
    _header("Step 5: Memory Store + Query API")

    from contextlens.memory_store import MemoryStore
    from contextlens.query import query
    from contextlens.schemas import ImageOutput

    db_path = PROJECT_ROOT / "contextlens_demo.db"
    with MemoryStore(db_path=str(db_path)) as store:
        store.clear()

        # Store all outputs
        outputs = [ImageOutput.model_validate(r) for r in structured_results]
        store.store_batch(outputs)

        # Store groups
        groups_seen: set[str] = set()
        for out in outputs:
            if out.group_id and out.group_id not in groups_seen:
                members = [o for o in outputs if o.group_id == out.group_id]
                store.store_group(
                    out.group_id,
                    len(members),
                    f"Group of {len(members)} related images.",
                )
                groups_seen.add(out.group_id)

        print(f"  Stored {len(outputs)} images in memory ({db_path.name})")

        # Sample queries
        sample_queries = [
            "all receipts",
            "whiteboard photos from project Alpha",
            "conversations mentioning a meeting",
            "images needing clarification",
            "all images from last week",
        ]

        _subheader("Sample Queries")
        for q in sample_queries:
            qr = query(store, q)
            ids = [r["image_id"] for r in qr.results]
            print(f"  Q: \"{q}\"")
            print(f"     → {len(qr)} result(s): {', '.join(ids) if ids else '(none)'}")


# ---------------------------------------------------------------------------
# Step 6: Evaluation
# ---------------------------------------------------------------------------

def step_evaluation(structured_results: list[dict], baseline_results: list[dict]) -> None:
    """Run evaluation metrics and generate plots/tables."""
    _header("Step 6: Evaluation + Calibration Plot")

    from scripts.run_evaluation import (
        compute_aggregate_metrics,
        compute_calibration_buckets,
        collect_confidence_correctness_pairs,
        expected_calibration_error,
        calibration_correlation,
        generate_comparison_table,
        load_annotations,
        plot_calibration,
    )

    annotations = load_annotations(ANNOTATIONS_DIR)
    print(f"  Loaded {len(annotations)} annotations.")

    # Compute metrics
    print("  Computing structured pipeline metrics...")
    s_metrics = compute_aggregate_metrics(structured_results, annotations)

    print("  Computing baseline metrics...")
    b_metrics = compute_aggregate_metrics(baseline_results, annotations)

    # Calibration plot
    s_pairs = collect_confidence_correctness_pairs(structured_results, annotations)
    b_pairs = collect_confidence_correctness_pairs(baseline_results, annotations)
    s_buckets = compute_calibration_buckets(s_pairs)
    b_buckets = compute_calibration_buckets(b_pairs)

    fig_dir = RESULTS_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_path = fig_dir / "calibration.png"
    plot_calibration(s_buckets, b_buckets, plot_path)
    print(f"  Calibration plot saved to {plot_path}")

    # Comparison table
    table_dir = RESULTS_DIR / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    table_path = table_dir / "comparison.csv"
    rows = generate_comparison_table(s_metrics, b_metrics, table_path)
    print(f"  Comparison table saved to {table_path}")

    # Save metric JSONs
    metrics_dir = RESULTS_DIR / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    s_save = {k: v for k, v in s_metrics.items() if k != "calibration_buckets"}
    (metrics_dir / "structured_metrics.json").write_text(json.dumps(s_save, indent=2))
    b_save = {k: v for k, v in b_metrics.items() if k != "calibration_buckets"}
    (metrics_dir / "baseline_metrics.json").write_text(json.dumps(b_save, indent=2))
    print(f"  Metrics saved to {metrics_dir}")

    # Print comparison table
    _subheader("Structured vs Baseline Comparison")
    print(f"  {'Metric':<42s} {'Structured':>12s} {'Baseline':>12s}")
    print(f"  {'-' * 42}  {'-' * 12}  {'-' * 12}")
    for row in rows:
        print(f"  {row['metric']:<42s} {row['structured_pipeline']:>12s} {row['baseline']:>12s}")

    # Print key calibration stats
    _subheader("Calibration Summary")
    print(f"  Structured ECE:          {s_metrics.get('calibration_ece', 0):.3f}")
    print(f"  Baseline ECE:            {b_metrics.get('calibration_ece', 0):.3f}")
    print(f"  Structured Correlation:  {s_metrics.get('calibration_correlation', 0):.3f}")
    print(f"  Baseline Correlation:    {b_metrics.get('calibration_correlation', 0):.3f}")


# ---------------------------------------------------------------------------
# Step 7: Summary of sample outputs
# ---------------------------------------------------------------------------

def step_show_samples(structured_results: list[dict]) -> None:
    """Print a few sample structured outputs for inspection."""
    _header("Step 7: Sample Structured Outputs")

    # Show one per type
    shown_types: set[str] = set()
    for r in structured_results:
        img_type = r.get("type", "")
        if img_type in shown_types:
            continue
        shown_types.add(img_type)

        print(f"\n  [{r['image_id']}] type={img_type}  "
              f"conf={r.get('type_confidence', 0):.2f}  "
              f"group={r.get('group_id', 'None')}")
        print(f"  Summary: {r.get('summary', '')}")
        print(f"  Flags: {r.get('failure_flags', [])}")
        print(f"  Needs clarification: {r.get('needs_clarification', False)}")

        # Show field confidence
        fc = r.get("field_confidence", {})
        if fc:
            conf_str = ", ".join(f"{k}={v:.2f}" for k, v in list(fc.items())[:5])
            print(f"  Field confidence: {conf_str}")

        if len(shown_types) >= 4:
            break


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------

def dry_run() -> None:
    """Validate imports and module availability without heavy processing."""
    _header("Dry Run: Validating Module Imports")

    modules = [
        ("contextlens.schemas", "Pydantic schemas"),
        ("contextlens.config", "Configuration"),
        ("contextlens.preprocess", "Image preprocessing"),
        ("contextlens.ocr", "OCR engine"),
        ("contextlens.classifier", "Type classifier"),
        ("contextlens.extractors", "Conditioned extractors"),
        ("contextlens.confidence", "Confidence calibrator"),
        ("contextlens.failure_handlers", "Failure handlers"),
        ("contextlens.pipeline", "Pipeline orchestration"),
        ("contextlens.baseline", "Baseline parser"),
        ("contextlens.linker", "Cross-image linker"),
        ("contextlens.memory_store", "Memory store"),
        ("contextlens.query", "Query API"),
        ("scripts.run_evaluation", "Evaluation script"),
        ("scripts.generate_test_images", "Test image generator"),
    ]

    all_ok = True
    for mod_name, desc in modules:
        try:
            __import__(mod_name)
            print(f"  [OK] {desc:30s} ({mod_name})")
        except ImportError as e:
            print(f"  [FAIL] {desc:30s} ({mod_name}) — {e}")
            all_ok = False

    # Validate annotations exist
    _subheader("Annotations")
    ann_files = sorted(ANNOTATIONS_DIR.glob("*.json")) if ANNOTATIONS_DIR.exists() else []
    print(f"  Found {len(ann_files)} annotation files in {ANNOTATIONS_DIR}")
    if len(ann_files) < 14:
        print("  WARNING: Expected 14 annotations.")

    # Validate schemas
    _subheader("Schema Validation")
    from contextlens.schemas import ImageOutput, ImageType, QualitySignals

    sample = ImageOutput(
        image_id="test",
        type=ImageType.RECEIPT,
        type_confidence=0.9,
        extracted_entities={},
        field_confidence={},
        summary="Test.",
        quality_signals=QualitySignals(
            blur_score=500.0,
            brightness=0.5,
            contrast=0.5,
            estimated_quality=0.8,
        ),
        raw_text="test",
    )
    serialized = sample.model_dump(mode="json")
    print(f"  ImageOutput serialization: OK ({len(json.dumps(serialized))} bytes)")

    # Validate output dirs are creatable
    _subheader("Output Directories")
    for d in [STRUCTURED_OUTPUT_DIR, BASELINE_OUTPUT_DIR,
              RESULTS_DIR / "figures", RESULTS_DIR / "tables", RESULTS_DIR / "metrics"]:
        d.mkdir(parents=True, exist_ok=True)
        print(f"  [OK] {d}")

    if all_ok:
        print("\n  All modules imported successfully. Ready for full run.")
        print("  Run without --dry-run to execute the full pipeline.")
    else:
        print("\n  Some modules failed to import. Fix errors before full run.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ContextLens end-to-end demo: process images, evaluate, report.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate imports and setup without running the full pipeline.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  ContextLens: A Vision Agent for Personal Visual Context")
    print("=" * 60)

    if args.dry_run:
        dry_run()
        return

    # Full pipeline
    images = step_generate_images()
    structured_results = step_structured_pipeline(images)
    baseline_results = step_baseline(images)
    structured_results = step_linking(structured_results)
    step_memory_and_queries(structured_results)
    step_evaluation(structured_results, baseline_results)
    step_show_samples(structured_results)

    _header("Done!")
    print("  All outputs saved:")
    print(f"    Structured: {STRUCTURED_OUTPUT_DIR}")
    print(f"    Baseline:   {BASELINE_OUTPUT_DIR}")
    print(f"    Results:    {RESULTS_DIR}")
    print(f"      - Calibration plot: {RESULTS_DIR / 'figures' / 'calibration.png'}")
    print(f"      - Comparison table: {RESULTS_DIR / 'tables' / 'comparison.csv'}")
    print(f"      - Metrics:          {RESULTS_DIR / 'metrics'}")
    print()


if __name__ == "__main__":
    main()
