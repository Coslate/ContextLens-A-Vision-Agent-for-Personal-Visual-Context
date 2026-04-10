"""Cross-image context fuser — BEV-JEPA analog.

Embeds image summaries with sentence-transformers, computes pairwise
similarity (embedding cosine + metadata overlap), groups linked images,
and generates fused group-level summaries.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta

import numpy as np

from contextlens.config import (
    DATE_PROXIMITY_BONUS,
    DATE_PROXIMITY_DAYS,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    SAME_MERCHANT_BONUS,
    SHARED_PARTICIPANTS_BONUS,
    SHARED_PROJECT_TAGS_BONUS,
    SHARED_TOPICS_BONUS,
    SIMILARITY_THRESHOLD,
    WEIGHT_EMBEDDING,
    WEIGHT_METADATA,
)
from contextlens.schemas import (
    ConversationEntities,
    DocumentEntities,
    ImageOutput,
    ImageType,
    ReceiptEntities,
    WhiteboardEntities,
)

# ---------------------------------------------------------------------------
# Lazy-loaded sentence-transformer model
# ---------------------------------------------------------------------------

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------

_DATE_RE = re.compile(
    r"(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})"
    r"|(\d{4}[/\-\.]\d{1,2}[/\-\.]\d{1,2})"
)

_DATE_FORMATS = [
    "%m/%d/%Y", "%m-%d-%Y", "%m.%d.%Y",
    "%Y/%m/%d", "%Y-%m-%d", "%Y.%m.%d",
    "%m/%d/%y", "%m-%d-%y",
]


def _parse_date(date_str: str) -> datetime | None:
    """Try multiple date formats."""
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    return None


def extract_metadata(output: ImageOutput) -> dict:
    """Extract linkable metadata from an ImageOutput.

    Returns dict with keys: merchant, dates, participants, project_tags, topics.
    """
    meta: dict = {
        "merchant": None,
        "dates": [],
        "participants": [],
        "project_tags": [],
        "topics": [],
    }

    entities = output.extracted_entities

    if isinstance(entities, ReceiptEntities):
        meta["merchant"] = entities.merchant
        if entities.date:
            meta["dates"].append(entities.date)

    elif isinstance(entities, ConversationEntities):
        meta["participants"] = list(entities.participants)
        meta["topics"] = list(entities.key_topics)

    elif isinstance(entities, DocumentEntities):
        # Extract date from structured fields if present
        for key, val in entities.structured_fields.items():
            if "date" in key.lower():
                meta["dates"].append(val)

    elif isinstance(entities, WhiteboardEntities):
        structure = entities.inferred_structure
        meta["project_tags"] = list(structure.project_tags)
        meta["topics"] = list(structure.tasks)
        meta["dates"] = list(structure.dates)

    return meta


# ---------------------------------------------------------------------------
# Pairwise scoring
# ---------------------------------------------------------------------------


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _dates_within_range(dates_a: list[str], dates_b: list[str],
                        days: int = DATE_PROXIMITY_DAYS) -> bool:
    """Check if any pair of dates from the two lists is within `days` days."""
    parsed_a = [_parse_date(d) for d in dates_a]
    parsed_b = [_parse_date(d) for d in dates_b]
    parsed_a = [d for d in parsed_a if d is not None]
    parsed_b = [d for d in parsed_b if d is not None]

    for da in parsed_a:
        for db in parsed_b:
            if abs((da - db).days) <= days:
                return True
    return False


def _set_overlap(a: list[str], b: list[str]) -> bool:
    """Check if two lists share any elements (case-insensitive)."""
    set_a = {s.lower() for s in a if s}
    set_b = {s.lower() for s in b if s}
    return bool(set_a & set_b)


def compute_metadata_score(meta_a: dict, meta_b: dict) -> float:
    """Compute metadata overlap score between two images."""
    score = 0.0

    # Same merchant
    m_a = (meta_a.get("merchant") or "").strip().lower()
    m_b = (meta_b.get("merchant") or "").strip().lower()
    if m_a and m_b and m_a == m_b:
        score += SAME_MERCHANT_BONUS

    # Date proximity
    if _dates_within_range(
        meta_a.get("dates", []), meta_b.get("dates", []),
    ):
        score += DATE_PROXIMITY_BONUS

    # Shared participants
    if _set_overlap(
        meta_a.get("participants", []), meta_b.get("participants", []),
    ):
        score += SHARED_PARTICIPANTS_BONUS

    # Shared project tags
    if _set_overlap(
        meta_a.get("project_tags", []), meta_b.get("project_tags", []),
    ):
        score += SHARED_PROJECT_TAGS_BONUS

    # Shared topics
    if _set_overlap(
        meta_a.get("topics", []), meta_b.get("topics", []),
    ):
        score += SHARED_TOPICS_BONUS

    return score


def compute_pairwise_score(
    emb_a: np.ndarray,
    emb_b: np.ndarray,
    meta_a: dict,
    meta_b: dict,
) -> float:
    """Compute combined similarity score between two images.

    Score = WEIGHT_EMBEDDING * cosine_sim + WEIGHT_METADATA * metadata_overlap.
    """
    cos_sim = _cosine_similarity(emb_a, emb_b)
    # Clamp cosine to [0, 1] — negative similarity means unrelated
    cos_sim = max(0.0, cos_sim)

    meta_score = compute_metadata_score(meta_a, meta_b)

    return WEIGHT_EMBEDDING * cos_sim + WEIGHT_METADATA * meta_score


# ---------------------------------------------------------------------------
# Grouping via union-find
# ---------------------------------------------------------------------------


def _union_find_groups(n: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    """Simple union-find to partition nodes into connected components."""
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for a, b in edges:
        union(a, b)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(i)

    return list(groups.values())


# ---------------------------------------------------------------------------
# Group summary generation
# ---------------------------------------------------------------------------


def _generate_group_summary(
    group_id: str,
    outputs: list[ImageOutput],
) -> str:
    """Generate a fused summary for a group of related images."""
    types = [o.type.value for o in outputs]
    type_counts: dict[str, int] = {}
    for t in types:
        type_counts[t] = type_counts.get(t, 0) + 1

    parts: list[str] = [f"Group '{group_id}':"]

    # Summarize by type
    for img_type, count in type_counts.items():
        parts.append(f"{count} {img_type}{'s' if count > 1 else ''}")

    # Collect key details
    merchants: set[str] = set()
    dates: set[str] = set()
    participants: set[str] = set()
    tags: set[str] = set()

    for o in outputs:
        entities = o.extracted_entities
        if isinstance(entities, ReceiptEntities):
            if entities.merchant:
                merchants.add(entities.merchant)
            if entities.date:
                dates.add(entities.date)
        elif isinstance(entities, ConversationEntities):
            participants.update(entities.participants)
        elif isinstance(entities, WhiteboardEntities):
            tags.update(entities.inferred_structure.project_tags)

    details: list[str] = []
    if merchants:
        details.append(f"from {', '.join(sorted(merchants))}")
    if dates:
        details.append(f"on {', '.join(sorted(dates))}")
    if participants:
        details.append(f"with {', '.join(sorted(participants))}")
    if tags:
        details.append(f"tagged {', '.join(sorted(tags))}")

    summary = " ".join(parts)
    if details:
        summary += " — " + ", ".join(details)
    summary += "."

    return summary


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _generate_group_id(outputs: list[ImageOutput]) -> str:
    """Generate a descriptive group_id from the group's content."""
    types = {o.type for o in outputs}
    entities_all = [o.extracted_entities for o in outputs]

    # Try to name the group from content
    if ImageType.RECEIPT in types:
        merchants = set()
        for e in entities_all:
            if isinstance(e, ReceiptEntities) and e.merchant:
                merchants.add(e.merchant.lower().replace(" ", "_"))
        if merchants:
            return f"receipts_{'_'.join(sorted(merchants)[:2])}"
        return "receipts_group"

    if ImageType.WHITEBOARD in types:
        tags = set()
        for e in entities_all:
            if isinstance(e, WhiteboardEntities):
                tags.update(
                    t.lower().replace(" ", "_")
                    for t in e.inferred_structure.project_tags
                )
        if tags:
            return f"project_{'_'.join(sorted(tags)[:2])}"
        return "whiteboard_group"

    if ImageType.CONVERSATION in types:
        return "conversation_group"

    return "group"


def link_outputs(
    outputs: list[ImageOutput],
    threshold: float = SIMILARITY_THRESHOLD,
) -> tuple[list[ImageOutput], dict[str, str]]:
    """Link related images and assign group_ids.

    Args:
        outputs: List of ImageOutput objects to link.
        threshold: Pairwise score threshold for grouping.

    Returns:
        Tuple of:
        - Updated list of ImageOutput with group_id set.
        - Dict mapping group_id → fused group summary.
    """
    n = len(outputs)
    if n < 2:
        return outputs, {}

    # Step 1: Embed summaries
    model = _get_model()
    summaries = [o.summary or o.raw_text[:200] for o in outputs]
    embeddings = model.encode(summaries, convert_to_numpy=True)

    # Step 2: Extract metadata
    metadatas = [extract_metadata(o) for o in outputs]

    # Step 3: Compute pairwise scores & find edges above threshold
    edges: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            score = compute_pairwise_score(
                embeddings[i], embeddings[j],
                metadatas[i], metadatas[j],
            )
            if score >= threshold:
                edges.append((i, j))

    # Step 4: Group via union-find
    groups = _union_find_groups(n, edges)

    # Step 5: Assign group_ids and generate summaries
    group_summaries: dict[str, str] = {}

    for group_indices in groups:
        if len(group_indices) < 2:
            # Singletons don't get a group_id
            continue

        group_outputs = [outputs[i] for i in group_indices]
        group_id = _generate_group_id(group_outputs)

        # Generate fused summary
        summary = _generate_group_summary(group_id, group_outputs)
        group_summaries[group_id] = summary

        # Assign group_id to each output
        for i in group_indices:
            outputs[i].group_id = group_id

    # Step 6: Propagate calendar hooks within groups
    for group_indices in groups:
        if len(group_indices) < 2:
            continue
        group_outputs = [outputs[i] for i in group_indices]
        has_calendar = any(
            o.calendar_hook is not None and o.calendar_hook.mentioned
            for o in group_outputs
        )
        if has_calendar:
            # Find the calendar hook to propagate
            hook = next(
                o.calendar_hook
                for o in group_outputs
                if o.calendar_hook is not None and o.calendar_hook.mentioned
            )
            for i in group_indices:
                if outputs[i].calendar_hook is None:
                    outputs[i].calendar_hook = hook

    return outputs, group_summaries
