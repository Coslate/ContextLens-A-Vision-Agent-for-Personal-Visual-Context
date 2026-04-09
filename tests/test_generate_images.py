"""Tests for synthetic test image generation — PR7."""

import json
from pathlib import Path

import pytest
from PIL import Image

from contextlens.schemas import Annotation, FailureFlag, ImageType

ROOT = Path(__file__).resolve().parent.parent
IMG_DIR = ROOT / "data" / "test_images"
ANN_DIR = ROOT / "data" / "annotations"

IMAGE_IDS = [f"img_{i:03d}" for i in range(1, 15)]


# =====================================================================
# Generation script runs without error
# =====================================================================

class TestGenerateAll:
    def test_script_runs(self):
        """generate_all() should produce 14 images and 14 annotations."""
        from scripts.generate_test_images import generate_all

        anns = generate_all()
        assert len(anns) == 14


# =====================================================================
# Image files
# =====================================================================

class TestImageFiles:
    def test_14_images_exist(self):
        for iid in IMAGE_IDS:
            path = IMG_DIR / f"{iid}.png"
            assert path.exists(), f"Missing image: {path}"

    @pytest.mark.parametrize("image_id", IMAGE_IDS)
    def test_image_is_valid_png(self, image_id):
        path = IMG_DIR / f"{image_id}.png"
        img = Image.open(path)
        assert img.format == "PNG"
        assert img.size[0] > 0 and img.size[1] > 0

    @pytest.mark.parametrize("image_id", IMAGE_IDS)
    def test_image_is_rgb(self, image_id):
        path = IMG_DIR / f"{image_id}.png"
        img = Image.open(path)
        assert img.mode == "RGB"


# =====================================================================
# Annotation files
# =====================================================================

class TestAnnotationFiles:
    def test_14_annotations_exist(self):
        for iid in IMAGE_IDS:
            path = ANN_DIR / f"{iid}.json"
            assert path.exists(), f"Missing annotation: {path}"

    @pytest.mark.parametrize("image_id", IMAGE_IDS)
    def test_annotation_valid_json(self, image_id):
        path = ANN_DIR / f"{image_id}.json"
        with open(path) as f:
            data = json.load(f)
        assert isinstance(data, dict)
        assert data["image_id"] == image_id

    @pytest.mark.parametrize("image_id", IMAGE_IDS)
    def test_annotation_matches_schema(self, image_id):
        path = ANN_DIR / f"{image_id}.json"
        with open(path) as f:
            data = json.load(f)
        ann = Annotation(**data)
        assert ann.image_id == image_id

    @pytest.mark.parametrize("image_id", IMAGE_IDS)
    def test_annotation_has_expected_type(self, image_id):
        path = ANN_DIR / f"{image_id}.json"
        with open(path) as f:
            data = json.load(f)
        assert data["expected_type"] in [e.value for e in ImageType]


# =====================================================================
# Distribution
# =====================================================================

class TestDistribution:
    def _load_all_annotations(self) -> list[dict]:
        anns = []
        for iid in IMAGE_IDS:
            with open(ANN_DIR / f"{iid}.json") as f:
                anns.append(json.load(f))
        return anns

    def test_4_receipts(self):
        anns = self._load_all_annotations()
        count = sum(1 for a in anns if a["expected_type"] == "receipt")
        assert count == 4

    def test_3_conversations(self):
        anns = self._load_all_annotations()
        count = sum(1 for a in anns if a["expected_type"] == "conversation")
        assert count == 3

    def test_3_documents(self):
        anns = self._load_all_annotations()
        count = sum(1 for a in anns if a["expected_type"] == "document")
        assert count == 3

    def test_4_whiteboards(self):
        anns = self._load_all_annotations()
        count = sum(1 for a in anns if a["expected_type"] == "whiteboard")
        assert count == 4


# =====================================================================
# Failure flag annotations
# =====================================================================

class TestFailureAnnotations:
    def _load(self, image_id: str) -> dict:
        with open(ANN_DIR / f"{image_id}.json") as f:
            return json.load(f)

    def test_clean_receipt_no_flags(self):
        ann = self._load("img_001")
        assert ann["expected_failure_flags"] == []

    def test_rotated_receipt_flag(self):
        ann = self._load("img_002")
        assert "rotation_unresolved" in ann["expected_failure_flags"]

    def test_blurry_receipt_flag(self):
        ann = self._load("img_003")
        assert "blurry_image" in ann["expected_failure_flags"]

    def test_cropped_receipt_flag(self):
        ann = self._load("img_004")
        assert "partial_capture" in ann["expected_failure_flags"]

    def test_meeting_conversation_has_calendar_hook(self):
        ann = self._load("img_006")
        assert ann["expected_calendar_hook"] is True

    def test_mixed_language_flag(self):
        ann = self._load("img_010")
        assert "mixed_language" in ann["expected_failure_flags"]

    def test_blurry_whiteboard_flag(self):
        ann = self._load("img_014")
        assert "blurry_image" in ann["expected_failure_flags"]


# =====================================================================
# Needs-clarification annotations
# =====================================================================

class TestNeedsClarification:
    def _load(self, image_id: str) -> dict:
        with open(ANN_DIR / f"{image_id}.json") as f:
            return json.load(f)

    @pytest.mark.parametrize("image_id", [
        "img_001", "img_002", "img_005", "img_006",
        "img_008", "img_009", "img_010", "img_011", "img_012",
    ])
    def test_clean_images_no_clarification(self, image_id):
        ann = self._load(image_id)
        assert ann["expected_needs_clarification"] is False

    @pytest.mark.parametrize("image_id,reason", [
        ("img_003", "blurry_image"),
        ("img_004", "partial_capture"),
        ("img_007", "partial_capture"),
        ("img_013", "ocr_uncertain"),
        ("img_014", "blurry_image"),
    ])
    def test_adversarial_images_need_clarification(self, image_id, reason):
        ann = self._load(image_id)
        assert ann["expected_needs_clarification"] is True
        assert reason in ann["expected_failure_flags"]

    def test_all_annotations_have_field(self):
        for iid in IMAGE_IDS:
            ann = self._load(iid)
            assert "expected_needs_clarification" in ann, (
                f"{iid} missing expected_needs_clarification"
            )


# =====================================================================
# Calendar event annotations
# =====================================================================

class TestCalendarEventAnnotations:
    def _load(self, image_id: str) -> dict:
        with open(ANN_DIR / f"{image_id}.json") as f:
            return json.load(f)

    def test_meeting_conversation_has_event_details(self):
        ann = self._load("img_006")
        assert len(ann["expected_calendar_events"]) == 1
        event = ann["expected_calendar_events"][0]
        assert "title" in event
        assert "time_mention" in event
        assert "participants" in event
        assert len(event["participants"]) >= 2

    def test_meeting_conversation_entities_have_referenced_events(self):
        ann = self._load("img_006")
        assert "referenced_events" in ann["expected_entities"]
        assert len(ann["expected_entities"]["referenced_events"]) >= 1

    def test_non_meeting_images_no_calendar_events(self):
        for iid in IMAGE_IDS:
            if iid == "img_006":
                continue
            ann = self._load(iid)
            assert ann.get("expected_calendar_events", []) == [], (
                f"{iid} should not have calendar events"
            )


# =====================================================================
# Cropped conversation entity correctness
# =====================================================================

class TestCroppedConversationEntities:
    def _load(self, image_id: str) -> dict:
        with open(ANN_DIR / f"{image_id}.json") as f:
            return json.load(f)

    def test_img007_fewer_action_items_than_clean(self):
        clean = self._load("img_005")
        cropped = self._load("img_007")
        assert len(cropped["expected_entities"]["action_items"]) < len(
            clean["expected_entities"]["action_items"]
        )

    def test_img007_fewer_topics_than_clean(self):
        clean = self._load("img_005")
        cropped = self._load("img_007")
        assert len(cropped["expected_entities"]["key_topics"]) <= len(
            clean["expected_entities"]["key_topics"]
        )

    def test_img007_participants_preserved(self):
        """Participants appear in early messages so still visible after crop."""
        ann = self._load("img_007")
        assert set(ann["expected_entities"]["participants"]) == {"Alice", "Bob"}


# =====================================================================
# Linking annotations
# =====================================================================

class TestLinkingAnnotations:
    def _load(self, image_id: str) -> dict:
        with open(ANN_DIR / f"{image_id}.json") as f:
            return json.load(f)

    def test_receipts_share_group(self):
        for iid in ["img_001", "img_002", "img_003", "img_004"]:
            ann = self._load(iid)
            assert ann["expected_group"] == "receipts_trip"

    def test_whiteboards_share_group(self):
        assert self._load("img_011")["expected_group"] == "project_alpha"
        assert self._load("img_012")["expected_group"] == "project_alpha"

    def test_unrelated_no_group(self):
        ann = self._load("img_005")
        assert ann["expected_group"] is None


# =====================================================================
# Adversarial image properties
# =====================================================================

class TestAdversarialProperties:
    def test_rotated_image_dimensions_swapped(self):
        clean = Image.open(IMG_DIR / "img_001.png")
        rotated = Image.open(IMG_DIR / "img_002.png")
        # 90-degree rotation swaps width and height
        assert rotated.size[0] == clean.size[1]
        assert rotated.size[1] == clean.size[0]

    def test_blurry_image_same_size_as_clean(self):
        clean = Image.open(IMG_DIR / "img_001.png")
        blurry = Image.open(IMG_DIR / "img_003.png")
        assert blurry.size == clean.size

    def test_cropped_image_shorter(self):
        clean = Image.open(IMG_DIR / "img_001.png")
        cropped = Image.open(IMG_DIR / "img_004.png")
        assert cropped.size[0] == clean.size[0]  # same width
        assert cropped.size[1] < clean.size[1]   # shorter height

    def test_cropped_conversation_shorter(self):
        clean = Image.open(IMG_DIR / "img_005.png")
        cropped = Image.open(IMG_DIR / "img_007.png")
        assert cropped.size[1] < clean.size[1]
