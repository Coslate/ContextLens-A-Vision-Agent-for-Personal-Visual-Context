"""Run the generic unified parser baseline on all test images.

Usage:
    python -m scripts.run_baseline
    python -m scripts.run_baseline --images data/test_images/img_001.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from contextlens.baseline import baseline_process
from contextlens.ocr import run_ocr
from contextlens.preprocess import preprocess_image


def run_baseline_on_images(
    image_paths: list[Path],
    output_dir: Path | None = None,
) -> list[dict]:
    """Process images through the baseline pipeline and return results."""
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []

    for path in image_paths:
        image_id = path.stem
        print(f"[baseline] Processing {image_id} ...")

        # Preprocess (quality signals stored but not used by baseline)
        image_arr, quality = preprocess_image(str(path))

        # OCR
        ocr_result = run_ocr(image_arr)

        # Baseline extraction
        output = baseline_process(ocr_result, image_id=image_id, quality=quality)

        result = output.model_dump(mode="json")
        results.append(result)

        if output_dir:
            out_path = output_dir / f"{image_id}.json"
            out_path.write_text(json.dumps(result, indent=2))

        print(f"  type={output.type.value}  confidence={output.type_confidence:.2f}")
        print(f"  summary: {output.summary}")
        print()

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run generic unified parser baseline on test images.",
    )
    parser.add_argument(
        "--images",
        nargs="*",
        help="Specific image paths. Defaults to all images in data/test_images/.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/outputs/baseline",
        help="Directory to write output JSONs (default: data/outputs/baseline).",
    )
    args = parser.parse_args()

    if args.images:
        image_paths = [Path(p) for p in args.images]
    else:
        img_dir = Path("data/test_images")
        if not img_dir.exists():
            print(
                "data/test_images/ not found. Run scripts/generate_test_images.py first.",
                file=sys.stderr,
            )
            sys.exit(1)
        image_paths = sorted(img_dir.glob("*.png"))
        if not image_paths:
            print("No .png images found in data/test_images/.", file=sys.stderr)
            sys.exit(1)

    print(f"Running baseline on {len(image_paths)} image(s)...\n")
    run_baseline_on_images(image_paths, output_dir=Path(args.output_dir))
    print("Done.")


if __name__ == "__main__":
    main()
