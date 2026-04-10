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


class QueryResult:
    """Result of a query against the memory store."""

    SUPPORTED_PATTERNS = (
        "type (receipts, conversations, documents, whiteboards), "
        "project, meeting/calendar, time range (last week, past 3 days), "
        "clarification, group, participant"
    )

    def __init__(self, results: list[dict], matched: bool):
        self.results = results
        self.matched = matched

    def __len__(self) -> int:
        return len(self.results)

    def __iter__(self):
        return iter(self.results)

    def __getitem__(self, index):
        return self.results[index]

    def __eq__(self, other):
        if isinstance(other, list):
            return self.results == other
        if isinstance(other, QueryResult):
            return self.results == other.results
        return NotImplemented

    def __bool__(self) -> bool:
        return bool(self.results)


def query(store: MemoryStore, query_text: str) -> QueryResult:
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
        QueryResult with results and a flag indicating whether a pattern matched.
    """
    # Check for group query first (most specific)
    group_id = _detect_group(query_text)
    if group_id:
        return QueryResult(store.get_images_by_group(group_id), matched=True)

    # Check for clarification query
    if _detect_clarification(query_text):
        return QueryResult(store.get_needs_clarification(), matched=True)

    # Check for meeting/calendar query
    if _detect_meeting_mention(query_text):
        results = store.get_images_with_calendar_hooks()
        # Further filter by type if specified
        img_type = _detect_type(query_text)
        if img_type:
            results = [r for r in results if r["type"] == img_type]
        return QueryResult(results, matched=True)

    # Check for project query
    project = _detect_project(query_text)
    if project:
        results = store.search_entities("project_tag", project)
        # Further filter by type if specified
        img_type = _detect_type(query_text)
        if img_type:
            results = [r for r in results if r["type"] == img_type]
        return QueryResult(results, matched=True)

    # Type + time range query
    img_type = _detect_type(query_text)
    time_days = _detect_time_range(query_text)

    if img_type and time_days:
        # Get by type, then filter by time
        by_type = store.get_images_by_type(img_type)
        since = store.get_images_since(time_days)
        since_ids = {r["image_id"] for r in since}
        return QueryResult(
            [r for r in by_type if r["image_id"] in since_ids], matched=True,
        )

    if img_type:
        return QueryResult(store.get_images_by_type(img_type), matched=True)

    if time_days:
        return QueryResult(store.get_images_since(time_days), matched=True)

    # Check for participant query
    participant = _detect_participant(query_text)
    if participant:
        return QueryResult(
            store.search_entities("participant", participant), matched=True,
        )

    # Fallback: no pattern matched — return all
    return QueryResult(store.get_all_images(), matched=False)
