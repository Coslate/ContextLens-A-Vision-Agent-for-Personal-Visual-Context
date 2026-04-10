"""SQLite memory store for ContextLens outputs."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from contextlens.config import DEFAULT_DB_PATH
from contextlens.schemas import (
    CalendarHook,
    ImageOutput,
    ImageType,
)


class MemoryStore:
    """Persistent SQLite store for processed image outputs."""

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH):
        self.db_path = str(db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self) -> None:
        cur = self.conn.cursor()
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS images (
                image_id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                type_confidence REAL,
                summary TEXT,
                group_id TEXT,
                raw_text TEXT,
                quality_score REAL,
                needs_clarification BOOLEAN,
                processed_at TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id TEXT REFERENCES images(image_id),
                field_name TEXT NOT NULL,
                field_value TEXT,
                confidence REAL
            );

            CREATE TABLE IF NOT EXISTS links (
                src_image_id TEXT REFERENCES images(image_id),
                dst_image_id TEXT REFERENCES images(image_id),
                link_type TEXT,
                link_score REAL,
                PRIMARY KEY (src_image_id, dst_image_id)
            );

            CREATE TABLE IF NOT EXISTS calendar_hooks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_id TEXT REFERENCES images(image_id),
                event_title TEXT,
                time_mention TEXT,
                participants TEXT
            );

            CREATE TABLE IF NOT EXISTS groups (
                group_id TEXT PRIMARY KEY,
                member_count INTEGER,
                fused_summary TEXT
            );
        """)
        self.conn.commit()

    # ------------------------------------------------------------------
    # Insert
    # ------------------------------------------------------------------

    def store_output(self, output: ImageOutput) -> None:
        """Store a single ImageOutput into the database."""
        cur = self.conn.cursor()

        quality_score = (
            output.quality_signals.estimated_quality
            if output.quality_signals else None
        )

        cur.execute(
            """INSERT OR REPLACE INTO images
               (image_id, type, type_confidence, summary, group_id,
                raw_text, quality_score, needs_clarification, processed_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                output.image_id,
                output.type.value,
                output.type_confidence,
                output.summary,
                output.group_id,
                output.raw_text,
                quality_score,
                output.needs_clarification,
                datetime.now().isoformat(),
            ),
        )

        # Delete old entities for this image (in case of re-insert)
        cur.execute(
            "DELETE FROM entities WHERE image_id = ?", (output.image_id,)
        )

        # Store field confidence as entities
        for field_name, confidence in output.field_confidence.items():
            # Get field value from extracted_entities
            field_value = None
            if hasattr(output.extracted_entities, field_name):
                val = getattr(output.extracted_entities, field_name)
                if val is not None:
                    field_value = str(val)
            cur.execute(
                """INSERT INTO entities
                   (image_id, field_name, field_value, confidence)
                   VALUES (?, ?, ?, ?)""",
                (output.image_id, field_name, field_value, confidence),
            )

        # Store entities from whiteboard project_tags and other metadata
        entities = output.extracted_entities
        if hasattr(entities, "inferred_structure"):
            structure = entities.inferred_structure
            for tag in getattr(structure, "project_tags", []):
                cur.execute(
                    """INSERT INTO entities
                       (image_id, field_name, field_value, confidence)
                       VALUES (?, ?, ?, ?)""",
                    (output.image_id, "project_tag", tag, 1.0),
                )
        if hasattr(entities, "participants"):
            for p in getattr(entities, "participants", []):
                cur.execute(
                    """INSERT INTO entities
                       (image_id, field_name, field_value, confidence)
                       VALUES (?, ?, ?, ?)""",
                    (output.image_id, "participant", p, 1.0),
                )

        # Store calendar hooks
        if output.calendar_hook and output.calendar_hook.mentioned:
            # Delete old hooks for this image
            cur.execute(
                "DELETE FROM calendar_hooks WHERE image_id = ?",
                (output.image_id,),
            )
            for event in output.calendar_hook.event_candidates:
                cur.execute(
                    """INSERT INTO calendar_hooks
                       (image_id, event_title, time_mention, participants)
                       VALUES (?, ?, ?, ?)""",
                    (
                        output.image_id,
                        event.title,
                        event.time_mention,
                        json.dumps(event.participants),
                    ),
                )

        self.conn.commit()

    def store_batch(self, outputs: list[ImageOutput]) -> None:
        """Store multiple ImageOutputs."""
        for output in outputs:
            self.store_output(output)

    def store_group(
        self, group_id: str, member_count: int, fused_summary: str,
    ) -> None:
        """Store a group record."""
        cur = self.conn.cursor()
        cur.execute(
            """INSERT OR REPLACE INTO groups
               (group_id, member_count, fused_summary)
               VALUES (?, ?, ?)""",
            (group_id, member_count, fused_summary),
        )
        self.conn.commit()

    def store_link(
        self,
        src_image_id: str,
        dst_image_id: str,
        link_type: str = "similar",
        link_score: float = 0.0,
    ) -> None:
        """Store a pairwise link between two images."""
        cur = self.conn.cursor()
        cur.execute(
            """INSERT OR REPLACE INTO links
               (src_image_id, dst_image_id, link_type, link_score)
               VALUES (?, ?, ?, ?)""",
            (src_image_id, dst_image_id, link_type, link_score),
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_image(self, image_id: str) -> dict | None:
        """Retrieve a single image record."""
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM images WHERE image_id = ?", (image_id,))
        row = cur.fetchone()
        return dict(row) if row else None

    def get_all_images(self) -> list[dict]:
        """Retrieve all image records."""
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM images ORDER BY processed_at DESC")
        return [dict(r) for r in cur.fetchall()]

    def get_images_by_type(self, image_type: str) -> list[dict]:
        """Get images of a specific type."""
        cur = self.conn.cursor()
        cur.execute(
            "SELECT * FROM images WHERE type = ? ORDER BY processed_at DESC",
            (image_type,),
        )
        return [dict(r) for r in cur.fetchall()]

    def get_images_by_group(self, group_id: str) -> list[dict]:
        """Get all images in a group."""
        cur = self.conn.cursor()
        cur.execute(
            "SELECT * FROM images WHERE group_id = ? ORDER BY processed_at DESC",
            (group_id,),
        )
        return [dict(r) for r in cur.fetchall()]

    def get_images_since(self, days: int) -> list[dict]:
        """Get images processed within the last N days."""
        cur = self.conn.cursor()
        cur.execute(
            """SELECT * FROM images
               WHERE processed_at > datetime('now', ?)
               ORDER BY processed_at DESC""",
            (f"-{days} days",),
        )
        return [dict(r) for r in cur.fetchall()]

    def get_entities(self, image_id: str) -> list[dict]:
        """Get all entities for an image."""
        cur = self.conn.cursor()
        cur.execute(
            "SELECT * FROM entities WHERE image_id = ?", (image_id,),
        )
        return [dict(r) for r in cur.fetchall()]

    def get_calendar_hooks(self, image_id: str | None = None) -> list[dict]:
        """Get calendar hooks, optionally filtered by image_id."""
        cur = self.conn.cursor()
        if image_id:
            cur.execute(
                "SELECT * FROM calendar_hooks WHERE image_id = ?",
                (image_id,),
            )
        else:
            cur.execute("SELECT * FROM calendar_hooks")
        return [dict(r) for r in cur.fetchall()]

    def get_images_with_calendar_hooks(self) -> list[dict]:
        """Get images that have calendar hooks."""
        cur = self.conn.cursor()
        cur.execute(
            """SELECT DISTINCT i.* FROM images i
               JOIN calendar_hooks c ON i.image_id = c.image_id
               ORDER BY i.processed_at DESC""",
        )
        return [dict(r) for r in cur.fetchall()]

    def get_group(self, group_id: str) -> dict | None:
        """Get a group record."""
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM groups WHERE group_id = ?", (group_id,))
        row = cur.fetchone()
        return dict(row) if row else None

    def get_all_groups(self) -> list[dict]:
        """Get all group records."""
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM groups")
        return [dict(r) for r in cur.fetchall()]

    def search_entities(
        self, field_name: str, field_value: str,
    ) -> list[dict]:
        """Search for images with entities matching field_name and value."""
        cur = self.conn.cursor()
        cur.execute(
            """SELECT DISTINCT i.* FROM images i
               JOIN entities e ON i.image_id = e.image_id
               WHERE e.field_name = ? AND e.field_value LIKE ?
               ORDER BY i.processed_at DESC""",
            (field_name, f"%{field_value}%"),
        )
        return [dict(r) for r in cur.fetchall()]

    def get_needs_clarification(self) -> list[dict]:
        """Get all images flagged as needing clarification."""
        cur = self.conn.cursor()
        cur.execute(
            "SELECT * FROM images WHERE needs_clarification = 1"
            " ORDER BY processed_at DESC",
        )
        return [dict(r) for r in cur.fetchall()]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def clear(self) -> None:
        """Delete all data from all tables."""
        cur = self.conn.cursor()
        for table in ("calendar_hooks", "links", "entities", "groups", "images"):
            cur.execute(f"DELETE FROM {table}")  # noqa: S608
        self.conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
