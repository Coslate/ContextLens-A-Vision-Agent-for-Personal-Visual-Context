"""Rule-based query API for the ContextLens memory store."""

from __future__ import annotations

import re

from contextlens.memory_store import MemoryStore


# ---------------------------------------------------------------------------
# Query patterns
# ---------------------------------------------------------------------------

_TYPE_WORDS = {
    "receipt": "receipt",
    "receipts": "receipt",
    "conversation": "conversation",
    "conversations": "conversation",
    "chat": "conversation",
    "chats": "conversation",
    "document": "document",
    "documents": "document",
    "form": "document",
    "forms": "document",
    "whiteboard": "whiteboard",
    "whiteboards": "whiteboard",
    "note": "whiteboard",
    "notes": "whiteboard",
}

_TIME_PATTERNS = {
    r"last\s+week": 7,
    r"past\s+week": 7,
    r"this\s+week": 7,
    r"last\s+(\d+)\s+days?": None,  # dynamic
    r"past\s+(\d+)\s+days?": None,
    r"today": 1,
    r"yesterday": 2,
    r"last\s+month": 30,
    r"past\s+month": 30,
}


def _detect_type(query: str) -> str | None:
    """Detect image type from query text."""
    query_lower = query.lower()
    for word, img_type in _TYPE_WORDS.items():
        if word in query_lower:
            return img_type
    return None


def _detect_time_range(query: str) -> int | None:
    """Detect time range in days from query text."""
    query_lower = query.lower()
    for pattern, days in _TIME_PATTERNS.items():
        match = re.search(pattern, query_lower)
        if match:
            if days is not None:
                return days
            # Dynamic pattern (e.g. "last 5 days")
            return int(match.group(1))
    return None


def _detect_group(query: str) -> str | None:
    """Detect group reference from query text."""
    query_lower = query.lower()
    # "in group X" or "group X"
    match = re.search(r"(?:in\s+)?group\s+['\"]?(\S+)['\"]?", query_lower)
    if match:
        return match.group(1)
    return None


def _detect_project(query: str) -> str | None:
    """Detect project reference from query text."""
    query_lower = query.lower()
    match = re.search(
        r"(?:from\s+|about\s+)?project\s+['\"]?(\S+)['\"]?", query_lower,
    )
    if match:
        return match.group(1)
    return None


def _detect_meeting_mention(query: str) -> bool:
    """Check if query asks about meetings/calendar events."""
    keywords = ["meeting", "calendar", "event", "schedule", "appointment"]
    query_lower = query.lower()
    return any(kw in query_lower for kw in keywords)


def _detect_clarification(query: str) -> bool:
    """Check if query asks about images needing clarification."""
    keywords = ["clarification", "unclear", "need review", "flagged"]
    query_lower = query.lower()
    return any(kw in query_lower for kw in keywords)


def _detect_participant(query: str) -> str | None:
    """Detect participant reference from query text."""
    query_lower = query.lower()
    match = re.search(
        r"(?:with|from|by|mentioning)\s+(\w+)", query_lower,
    )
    if match:
        name = match.group(1)
        # Filter out common non-name words
        skip = {
            "project", "group", "meeting", "calendar", "last", "past",
            "the", "a", "an", "all", "receipts", "conversations",
            "documents", "whiteboards",
        }
        if name.lower() not in skip:
            return name
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def query(store: MemoryStore, query_text: str) -> list[dict]:
    """Execute a natural-language-like query against the memory store.

    Supported query patterns:
    - "all receipts from last week"
    - "whiteboard photos from project X"
    - "conversations mentioning a meeting"
    - "all images in group Y"
    - "images needing clarification"

    Args:
        store: The MemoryStore to query.
        query_text: Natural language query string.

    Returns:
        List of image records matching the query.
    """
    # Check for group query first (most specific)
    group_id = _detect_group(query_text)
    if group_id:
        return store.get_images_by_group(group_id)

    # Check for clarification query
    if _detect_clarification(query_text):
        return store.get_needs_clarification()

    # Check for meeting/calendar query
    if _detect_meeting_mention(query_text):
        results = store.get_images_with_calendar_hooks()
        # Further filter by type if specified
        img_type = _detect_type(query_text)
        if img_type:
            results = [r for r in results if r["type"] == img_type]
        return results

    # Check for project query
    project = _detect_project(query_text)
    if project:
        results = store.search_entities("project_tag", project)
        # Further filter by type if specified
        img_type = _detect_type(query_text)
        if img_type:
            results = [r for r in results if r["type"] == img_type]
        return results

    # Type + time range query
    img_type = _detect_type(query_text)
    time_days = _detect_time_range(query_text)

    if img_type and time_days:
        # Get by type, then filter by time
        by_type = store.get_images_by_type(img_type)
        since = store.get_images_since(time_days)
        since_ids = {r["image_id"] for r in since}
        return [r for r in by_type if r["image_id"] in since_ids]

    if img_type:
        return store.get_images_by_type(img_type)

    if time_days:
        return store.get_images_since(time_days)

    # Check for participant query
    participant = _detect_participant(query_text)
    if participant:
        return store.search_entities("participant", participant)

    # Fallback: return all
    return store.get_all_images()
