"""Conversation screenshot extractor — conditioned on chat-specific fields."""

from __future__ import annotations

import re

from contextlens.extractors.base import ConditionedExtractor
from contextlens.schemas import (
    CalendarEventCandidate,
    CalendarHook,
    ConversationEntities,
    ImageType,
    OCRResult,
    QualitySignals,
)

# --- Patterns ---

_SPEAKER_RE = re.compile(r"^([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\s?:\s*(.+)", re.MULTILINE)
_TIMESTAMP_RE = re.compile(r"\b(\d{1,2}:\d{2}(?:\s?[APap][Mm])?)\b")

_ACTION_VERBS = re.compile(
    r"\b(send|review|schedule|prepare|update|submit|complete|finish|"
    r"follow up|set up|organize|create|write|draft|check|confirm|book|"
    r"arrange|plan|fix|deploy|test|deliver|share)\b",
    re.IGNORECASE,
)
_EVENT_KEYWORDS = re.compile(
    r"\b(meeting|sync|call|standup|stand-up|conference|workshop|"
    r"presentation|demo|interview|lunch|dinner|appointment|session)\b",
    re.IGNORECASE,
)
_TIME_MENTIONS = re.compile(
    r"\b(tomorrow|today|tonight|monday|tuesday|wednesday|thursday|friday|"
    r"saturday|sunday|next week|this week|morning|afternoon|evening|"
    r"\d{1,2}(?::\d{2})?\s?(?:am|pm|AM|PM)|at\s+\d{1,2})\b",
    re.IGNORECASE,
)


class ConversationExtractor(ConditionedExtractor):
    """Extract participants, topics, action items, and calendar hooks."""

    image_type = ImageType.CONVERSATION

    def extract(
        self,
        ocr: OCRResult,
        quality: QualitySignals | None = None,
    ) -> tuple[ConversationEntities, dict[str, float], CalendarHook | None]:
        text = ocr.raw_text
        span_conf = {s.text: s.confidence for s in ocr.spans} if ocr.spans else {}
        avg_conf = ocr.avg_confidence if ocr.avg_confidence else 0.8

        participants = self._extract_participants(text)
        action_items = self._extract_action_items(text)
        key_topics = self._extract_topics(text)
        events, calendar_hook = self._extract_events(text, participants)

        # Build confidence
        field_confidence: dict[str, float] = {
            "participants": min(avg_conf, 0.95) if participants else 0.0,
            "key_topics": min(avg_conf * 0.85, 0.85) if key_topics else 0.0,
            "action_items": min(avg_conf * 0.8, 0.8) if action_items else 0.0,
            "referenced_events": min(avg_conf * 0.75, 0.75) if events else 0.0,
        }

        entities = ConversationEntities(
            participants=participants,
            key_topics=key_topics,
            action_items=action_items,
            referenced_events=events,
        )
        return entities, field_confidence, calendar_hook

    def _extract_participants(self, text: str) -> list[str]:
        """Extract speaker names from 'Name: message' patterns."""
        speakers = _SPEAKER_RE.findall(text)
        # speakers is list of (name, message) tuples
        seen: set[str] = set()
        result: list[str] = []
        for name, _ in speakers:
            if name not in seen:
                seen.add(name)
                result.append(name)
        return result

    def _extract_action_items(self, text: str) -> list[str]:
        """Extract lines containing action verbs as action items."""
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        action_items: list[str] = []
        for line in lines:
            # Check for action verbs
            if _ACTION_VERBS.search(line):
                # Extract the message part if it's a speaker line
                speaker_match = _SPEAKER_RE.match(line)
                if speaker_match:
                    msg = speaker_match.group(2).strip()
                else:
                    msg = line
                # Filter out very short fragments
                if len(msg) > 10:
                    action_items.append(msg)
        return action_items

    def _extract_topics(self, text: str) -> list[str]:
        """Extract key topics via keyword frequency."""
        topics: list[str] = []

        # Check for event-related topics
        event_matches = _EVENT_KEYWORDS.findall(text)
        for match in event_matches:
            topic = match.lower()
            if topic not in topics:
                topics.append(topic)

        # Check for action-related topics
        action_matches = _ACTION_VERBS.findall(text)
        for match in action_matches:
            topic = match.lower()
            if topic not in topics:
                topics.append(topic)

        return topics

    def _extract_events(
        self, text: str, participants: list[str]
    ) -> tuple[list[CalendarEventCandidate], CalendarHook | None]:
        """Extract referenced events and build calendar hook."""
        events: list[CalendarEventCandidate] = []

        event_keywords = _EVENT_KEYWORDS.findall(text)
        time_mentions = _TIME_MENTIONS.findall(text)

        if event_keywords:
            # Build event candidates from event keyword + time mention pairs
            title = event_keywords[0].lower()
            time_mention = time_mentions[0] if time_mentions else None

            candidate = CalendarEventCandidate(
                title=title,
                time_mention=time_mention,
                participants=participants[:],
            )
            events.append(candidate)

            # Check for additional distinct events
            seen_titles = {title}
            for kw in event_keywords[1:]:
                t = kw.lower()
                if t not in seen_titles:
                    seen_titles.add(t)
                    events.append(CalendarEventCandidate(
                        title=t,
                        time_mention=None,
                        participants=participants[:],
                    ))

        calendar_hook = None
        if events:
            calendar_hook = CalendarHook(
                mentioned=True,
                event_candidates=events,
            )

        return events, calendar_hook
