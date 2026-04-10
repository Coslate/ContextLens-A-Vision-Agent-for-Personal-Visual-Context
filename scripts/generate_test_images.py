"""Generate 14 synthetic test images and ground-truth annotation JSONs.

Each image is rendered programmatically with Pillow so we have *perfect*
ground truth for evaluation.  Adversarial variants (blur, rotation, crop)
are produced from clean originals using OpenCV / Pillow transforms.

Image manifest
--------------
| #  | Type         | Variant              | Failure mode     |
|----|------------- |----------------------|------------------|
|  1 | receipt      | clean                | —                |
|  2 | receipt      | rotated 90°          | rotation         |
|  3 | receipt      | gaussian blur        | blurry           |
|  4 | receipt      | bottom-cropped       | partial (no tot) |
|  5 | conversation | clean chat           | —                |
|  6 | conversation | meeting reference    | calendar hook    |
|  7 | conversation | cropped (cut off)    | partial          |
|  8 | document     | structured form      | —                |
|  9 | document     | invoice              | —                |
| 10 | document     | mixed language        | mixed_language   |
| 11 | whiteboard   | clean                | —                |
| 12 | whiteboard   | same project as #11  | linking          |
| 13 | whiteboard   | messy / low OCR      | low OCR          |
| 14 | whiteboard   | blurry               | blurry           |
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
IMG_DIR = ROOT / "data" / "test_images"
ANN_DIR = ROOT / "data" / "annotations"


def _ensure_dirs() -> None:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    ANN_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Font helper
# ---------------------------------------------------------------------------

def _font(size: int = 18) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Return a monospace-ish font; fall back to default if not available."""
    for name in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeMono.ttf",
    ]:
        if os.path.exists(name):
            return ImageFont.truetype(name, size)
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def _render_lines(
    lines: list[str],
    width: int = 500,
    bg: str = "white",
    fg: str = "black",
    font_size: int = 18,
    y_start: int = 30,
    line_height: int = 28,
) -> Image.Image:
    """Render lines of text onto a new image."""
    height = y_start + line_height * len(lines) + 40
    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)
    fnt = _font(font_size)
    for i, line in enumerate(lines):
        draw.text((30, y_start + i * line_height), line, fill=fg, font=fnt)
    return img


def _apply_blur(img: Image.Image, ksize: int = 21) -> Image.Image:
    arr = np.array(img)
    blurred = cv2.GaussianBlur(arr, (ksize, ksize), 0)
    return Image.fromarray(blurred)


def _apply_rotation(img: Image.Image, angle: int = 90) -> Image.Image:
    return img.rotate(angle, expand=True, fillcolor=(255, 255, 255))


def _crop_bottom(img: Image.Image, fraction: float = 0.4) -> Image.Image:
    w, h = img.size
    return img.crop((0, 0, w, int(h * (1 - fraction))))


# ---------------------------------------------------------------------------
# Individual image generators
# ---------------------------------------------------------------------------

def _gen_receipt_clean() -> tuple[Image.Image, dict]:
    lines = [
        "STARBUCKS",
        "123 Main Street",
        "Seattle, WA 98101",
        "",
        "Latte            $5.50",
        "Muffin           $3.25",
        "Iced Tea         $2.75",
        "",
        "SUBTOTAL         $11.50",
        "TAX              $0.92",
        "TOTAL            $12.42",
        "",
        "03/15/2024  10:32 AM",
        "VISA **** 1234",
    ]
    img = _render_lines(lines, width=400)
    ann = {
        "image_id": "img_001",
        "expected_type": "receipt",
        "expected_entities": {
            "merchant": "STARBUCKS",
            "items": [
                {"name": "Latte", "price": 5.50},
                {"name": "Muffin", "price": 3.25},
                {"name": "Iced Tea", "price": 2.75},
            ],
            "total": 12.42,
            "date": "03/15/2024",
            "currency": "USD",
        },
        "expected_group": "receipts_trip",
        "expected_calendar_hook": None,
        "expected_calendar_events": [],
        "expected_needs_clarification": False,
        "expected_failure_flags": [],
        "notes": "Clean receipt, should achieve high extraction accuracy",
    }
    return img, ann


def _gen_receipt_rotated() -> tuple[Image.Image, dict]:
    img, ann = _gen_receipt_clean()
    img = _apply_rotation(img, 90)
    ann["image_id"] = "img_002"
    ann["expected_needs_clarification"] = False
    ann["expected_failure_flags"] = ["rotation_unresolved"]
    ann["notes"] = "Rotated 90 degrees; rotation correction needed"
    return img, ann


def _gen_receipt_blurry() -> tuple[Image.Image, dict]:
    img, ann = _gen_receipt_clean()
    img = _apply_blur(img, ksize=31)
    ann["image_id"] = "img_003"
    ann["expected_needs_clarification"] = True
    ann["expected_failure_flags"] = ["blurry_image"]
    ann["notes"] = "Gaussian blur applied; expect low confidence"
    return img, ann


def _gen_receipt_cropped() -> tuple[Image.Image, dict]:
    img, ann = _gen_receipt_clean()
    img = _crop_bottom(img, fraction=0.45)
    ann["image_id"] = "img_004"
    ann["expected_entities"]["total"] = None  # cropped off
    ann["expected_entities"]["date"] = None
    ann["expected_needs_clarification"] = True
    ann["expected_failure_flags"] = ["partial_capture"]
    ann["notes"] = "Bottom cropped; total and date missing"
    return img, ann


def _gen_conversation_clean() -> tuple[Image.Image, dict]:
    lines = [
        "Alice: Hey, how's the project going?",
        "Bob: Pretty well, I finished the API endpoints",
        "Alice: Great! Can you send me the documentation?",
        "Bob: Sure, I'll prepare it and share by EOD",
        "Alice: Thanks, also check the test results",
    ]
    img = _render_lines(lines, width=550)
    ann = {
        "image_id": "img_005",
        "expected_type": "conversation",
        "expected_entities": {
            "participants": ["Alice", "Bob"],
            "key_topics": ["project", "API", "documentation", "test"],
            "action_items": [
                "send me the documentation",
                "prepare it and share by EOD",
                "check the test results",
            ],
        },
        "expected_group": None,
        "expected_calendar_hook": None,
        "expected_calendar_events": [],
        "expected_needs_clarification": False,
        "expected_failure_flags": [],
        "notes": "Clean conversation with action items, no calendar event",
    }
    return img, ann


def _gen_conversation_meeting() -> tuple[Image.Image, dict]:
    lines = [
        "Alice: Let's schedule a meeting for tomorrow",
        "Bob: Sounds good, 3pm works for me",
        "Alice: I'll book the conference room",
        "Bob: Should we invite Carol to the sync?",
        "Alice: Yes, send her the invite please",
    ]
    img = _render_lines(lines, width=550)
    ann = {
        "image_id": "img_006",
        "expected_type": "conversation",
        "expected_entities": {
            "participants": ["Alice", "Bob"],
            "key_topics": ["meeting", "sync", "conference room"],
            "action_items": [
                "book the conference room",
                "send her the invite",
            ],
            "referenced_events": [
                {
                    "title": "sync",
                    "time_mention": "tomorrow 3pm",
                    "participants": ["Alice", "Bob", "Carol"],
                },
            ],
        },
        "expected_group": None,
        "expected_calendar_hook": True,
        "expected_calendar_events": [
            {
                "title": "sync",
                "time_mention": "tomorrow 3pm",
                "participants": ["Alice", "Bob", "Carol"],
            },
        ],
        "expected_needs_clarification": False,
        "expected_failure_flags": [],
        "notes": "Meeting reference triggers calendar hook",
    }
    return img, ann


def _gen_conversation_cropped() -> tuple[Image.Image, dict]:
    img, ann = _gen_conversation_clean()
    img = _crop_bottom(img, fraction=0.5)
    ann["image_id"] = "img_007"
    # Only top ~3 of 5 messages visible after 50% bottom crop.
    # Visible: "Alice: Hey...", "Bob: Pretty well...", "Alice: Great! Can you send..."
    # Cropped: "Bob: Sure, I'll prepare...", "Alice: Thanks, also check..."
    ann["expected_entities"] = {
        "participants": ["Alice", "Bob"],
        "key_topics": ["project", "API", "documentation"],
        "action_items": [
            "send me the documentation",
        ],
    }
    ann["expected_calendar_events"] = []
    ann["expected_needs_clarification"] = True
    ann["expected_failure_flags"] = ["partial_capture"]
    ann["notes"] = "Cropped conversation; bottom 50% cut, only first 3 messages visible"
    return img, ann


def _gen_document_form() -> tuple[Image.Image, dict]:
    lines = [
        "Patient Registration Form",
        "",
        "Patient Name: John Doe",
        "Date of Birth: 01/15/1985",
        "Phone: 555-0123",
        "Address: 123 Main Street",
        "Insurance ID: INS-98765",
        "Emergency Contact: Jane Doe",
    ]
    img = _render_lines(lines, width=500)
    ann = {
        "image_id": "img_008",
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
        "notes": "Structured form with key-value pairs",
    }
    return img, ann


def _gen_document_invoice() -> tuple[Image.Image, dict]:
    lines = [
        "INVOICE",
        "",
        "Invoice Number: INV-2024-001",
        "Date: March 15, 2024",
        "Bill To: Acme Corp",
        "Ship To: 456 Oak Avenue",
        "",
        "Description        Qty   Price",
        "Widget A            10   $25.00",
        "Widget B             5   $40.00",
        "",
        "Amount Due: $1,250.00",
        "Payment Due: April 15, 2024",
    ]
    img = _render_lines(lines, width=500)
    ann = {
        "image_id": "img_009",
        "expected_type": "document",
        "expected_entities": {
            "document_kind": "invoice",
            "structured_fields": {
                "Invoice Number": "INV-2024-001",
                "Date": "March 15, 2024",
                "Bill To": "Acme Corp",
                "Ship To": "456 Oak Avenue",
                "Amount Due": "$1,250.00",
                "Payment Due": "April 15, 2024",
            },
        },
        "expected_group": None,
        "expected_calendar_hook": None,
        "expected_calendar_events": [],
        "expected_needs_clarification": False,
        "expected_failure_flags": [],
        "notes": "Invoice document with structured fields",
    }
    return img, ann


def _gen_document_mixed_lang() -> tuple[Image.Image, dict]:
    lines = [
        "Medication: Amoxicillin 500mg",
        "Dosage: 1 tablet twice daily",
        "Prescription: Rx-45678",
        "Pharmacy: CVS Health",
        "",
        "Instrucciones: Tomar con agua",
        "Advertencia: No conducir",
        "Medico: Dr. Garcia",
    ]
    img = _render_lines(lines, width=500)
    ann = {
        "image_id": "img_010",
        "expected_type": "document",
        "expected_entities": {
            "document_kind": "medication",
            "structured_fields": {
                "Medication": "Amoxicillin 500mg",
                "Dosage": "1 tablet twice daily",
                "Prescription": "Rx-45678",
                "Pharmacy": "CVS Health",
                "Instrucciones": "Tomar con agua",
                "Advertencia": "No conducir",
                "Medico": "Dr. Garcia",
            },
        },
        "expected_group": None,
        "expected_calendar_hook": None,
        "expected_calendar_events": [],
        "expected_needs_clarification": False,
        "expected_failure_flags": ["mixed_language"],
        "notes": "Mixed English/Spanish medication doc",
    }
    return img, ann


def _gen_whiteboard_clean() -> tuple[Image.Image, dict]:
    lines = [
        "Sprint 5 Planning - Project Alpha",
        "",
        "- Design API endpoints @Alice",
        "- Write unit tests @Bob",
        "- Deploy to staging @Carol",
        "",
        "TODO: update documentation",
        "TODO: review PR #42",
        "",
        "#ProjectAlpha #Sprint5 #backend",
        "",
        "Deadline: due Friday",
        "Next sync: Monday 10am",
    ]
    img = _render_lines(lines, width=550)
    ann = {
        "image_id": "img_011",
        "expected_type": "whiteboard",
        "expected_entities": {
            "bullets": [
                "Design API endpoints @Alice",
                "Write unit tests @Bob",
                "Deploy to staging @Carol",
            ],
            "owners": ["Alice", "Bob", "Carol"],
            "tasks": [
                "Design API endpoints",
                "Write unit tests",
                "Deploy to staging",
                "update documentation",
                "review PR #42",
            ],
            "project_tags": ["ProjectAlpha", "Sprint5", "backend"],
            "dates": ["due Friday", "Monday"],
        },
        "expected_group": "project_alpha",
        "expected_calendar_hook": None,
        "expected_calendar_events": [],
        "expected_needs_clarification": False,
        "expected_failure_flags": [],
        "notes": "Clean whiteboard with structured elements",
    }
    return img, ann


def _gen_whiteboard_related() -> tuple[Image.Image, dict]:
    lines = [
        "Project Alpha - Retro Notes",
        "",
        "- Completed: API design phase",
        "- In progress: testing @Bob",
        "- Blocked: deployment (infra) @Carol",
        "",
        "#ProjectAlpha #Sprint5",
        "",
        "Action items:",
        "- Fix CI pipeline @Alice",
        "- Update staging config",
        "",
        "Next review: Wednesday",
    ]
    img = _render_lines(lines, width=550)
    ann = {
        "image_id": "img_012",
        "expected_type": "whiteboard",
        "expected_entities": {
            "bullets": [
                "Completed: API design phase",
                "In progress: testing @Bob",
                "Blocked: deployment (infra) @Carol",
                "Fix CI pipeline @Alice",
                "Update staging config",
            ],
            "owners": ["Bob", "Carol", "Alice"],
            "tasks": [
                "Fix CI pipeline",
                "Update staging config",
            ],
            "project_tags": ["ProjectAlpha", "Sprint5"],
            "dates": ["Wednesday"],
        },
        "expected_group": "project_alpha",
        "expected_calendar_hook": None,
        "expected_calendar_events": [],
        "expected_needs_clarification": False,
        "expected_failure_flags": [],
        "notes": "Related to img_011 via shared project tags; should be linked",
    }
    return img, ann


def _gen_whiteboard_messy() -> tuple[Image.Image, dict]:
    lines = [
        "brainstorm session",
        "  idea 1 - new login flow",
        " idea 2 redesign homepage",
        "   @Dave to prototype",
        "maybe #UXRedesign ?",
        "  check w/ stakeholders tmrw",
    ]
    img = _render_lines(lines, width=500, font_size=16)
    # Add noise to simulate messy handwriting
    arr = np.array(img)
    noise = np.random.RandomState(42).randint(0, 40, arr.shape, dtype=np.uint8)
    arr = np.clip(arr.astype(np.int16) - noise.astype(np.int16), 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    ann = {
        "image_id": "img_013",
        "expected_type": "whiteboard",
        "expected_entities": {
            "owners": ["Dave"],
            "project_tags": ["UXRedesign"],
        },
        "expected_group": None,
        "expected_calendar_hook": None,
        "expected_calendar_events": [],
        "expected_needs_clarification": True,
        "expected_failure_flags": ["ocr_uncertain"],
        "notes": "Messy handwriting with noise; low OCR confidence expected",
    }
    return img, ann


def _gen_whiteboard_blurry() -> tuple[Image.Image, dict]:
    img, ann = _gen_whiteboard_clean()
    img = _apply_blur(img, ksize=35)
    ann["image_id"] = "img_014"
    ann["expected_needs_clarification"] = True
    ann["expected_failure_flags"] = ["blurry_image"]
    ann["notes"] = "Blurry whiteboard; very low confidence, minimal extraction"
    ann["expected_group"] = None
    return img, ann


# ---------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------

_GENERATORS = [
    _gen_receipt_clean,       # img_001
    _gen_receipt_rotated,     # img_002
    _gen_receipt_blurry,      # img_003
    _gen_receipt_cropped,     # img_004
    _gen_conversation_clean,  # img_005
    _gen_conversation_meeting,# img_006
    _gen_conversation_cropped,# img_007
    _gen_document_form,       # img_008
    _gen_document_invoice,    # img_009
    _gen_document_mixed_lang, # img_010
    _gen_whiteboard_clean,    # img_011
    _gen_whiteboard_related,  # img_012
    _gen_whiteboard_messy,    # img_013
    _gen_whiteboard_blurry,   # img_014
]


def generate_all() -> list[dict]:
    """Generate all 14 test images and annotation JSONs.

    Returns list of annotation dicts (also written to disk).
    """
    _ensure_dirs()
    annotations: list[dict] = []

    for gen_fn in _GENERATORS:
        img, ann = gen_fn()
        image_id = ann["image_id"]

        # Save image
        img_path = IMG_DIR / f"{image_id}.png"
        img.save(str(img_path))

        # Save annotation
        ann_path = ANN_DIR / f"{image_id}.json"
        with open(ann_path, "w") as f:
            json.dump(ann, f, indent=2)

        annotations.append(ann)
        print(f"  {image_id}: {img.size[0]}x{img.size[1]}  → {img_path.name}")

    return annotations


if __name__ == "__main__":
    print("Generating 14 synthetic test images...")
    anns = generate_all()
    print(f"\nDone! {len(anns)} images in {IMG_DIR}")
    print(f"       {len(anns)} annotations in {ANN_DIR}")

    # Summary
    types = {}
    for a in anns:
        t = a["expected_type"]
        types[t] = types.get(t, 0) + 1
    print(f"\nDistribution: {types}")
