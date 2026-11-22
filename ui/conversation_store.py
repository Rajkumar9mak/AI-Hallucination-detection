"""
SQLite-backed conversation persistence for ClaimGuard AI.

Stores chat conversations and messages with metadata,
enabling history restoration across application restarts.
"""

import json
import sqlite3
import logging
import uuid
from datetime import datetime, date, timedelta, timezone
from pathlib import Path
from typing import Optional

from config.settings import DATA_DIR

logger = logging.getLogger(__name__)

DB_PATH = DATA_DIR / "conversations.db"

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL DEFAULT 'New Chat',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id              TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    role            TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
    content         TEXT NOT NULL,
    metadata_json   TEXT,
    created_at      TEXT NOT NULL,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, created_at);
CREATE INDEX IF NOT EXISTS idx_conversations_updated ON conversations(updated_at DESC);
"""


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

class ConversationStore:
    """Thread-safe SQLite conversation store with short-lived connections."""

    def __init__(self, db_path: Optional[Path] = None):
        self._db_path = db_path or DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    # -- helpers -------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        conn = self._connect()
        try:
            conn.executescript(_SCHEMA)
            conn.commit()
        finally:
            conn.close()

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

    @staticmethod
    def _new_id() -> str:
        return uuid.uuid4().hex[:16]

    # -- conversations -------------------------------------------------------

    def create_conversation(self, title: str = "New Chat") -> str:
        """Create a new conversation and return its ID."""
        conv_id = self._new_id()
        now = self._now_iso()
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (conv_id, title, now, now),
            )
            conn.commit()
        finally:
            conn.close()
        logger.debug("Created conversation %s: %s", conv_id, title)
        return conv_id

    def get_conversations(self) -> list[dict]:
        """Return all conversations ordered by most recently updated."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT id, title, created_at, updated_at FROM conversations ORDER BY updated_at DESC"
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def get_conversation(self, conv_id: str) -> Optional[dict]:
        """Return a single conversation by ID, or None."""
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?",
                (conv_id,),
            ).fetchone()
            return dict(row) if row else None
        finally:
            conn.close()

    def rename_conversation(self, conv_id: str, new_title: str) -> None:
        """Update the title of a conversation."""
        conn = self._connect()
        try:
            conn.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                (new_title, self._now_iso(), conv_id),
            )
            conn.commit()
        finally:
            conn.close()

    def delete_conversation(self, conv_id: str) -> None:
        """Delete a conversation and all its messages."""
        conn = self._connect()
        try:
            conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
            conn.commit()
        finally:
            conn.close()
        logger.debug("Deleted conversation %s", conv_id)

    def count_conversations(self) -> int:
        """Return total number of conversations."""
        conn = self._connect()
        try:
            row = conn.execute("SELECT COUNT(*) AS cnt FROM conversations").fetchone()
            return row["cnt"]
        finally:
            conn.close()

    def auto_title(self, conv_id: str) -> str:
        """
        Set the conversation title from the first user message.

        Truncates to 50 characters with ellipsis if needed.
        Returns the generated title.
        """
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT content FROM messages WHERE conversation_id = ? AND role = 'user' "
                "ORDER BY created_at ASC LIMIT 1",
                (conv_id,),
            ).fetchone()
            if not row:
                return "New Chat"
            text = row["content"].strip()
            title = text[:50] + ("..." if len(text) > 50 else "")
            conn.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
                (title, self._now_iso(), conv_id),
            )
            conn.commit()
            return title
        finally:
            conn.close()

    def get_conversations_grouped(self) -> dict[str, list[dict]]:
        """
        Return conversations grouped by date category.

        Groups: 'Today', 'Yesterday', 'Previous'.
        Each value is a list of conversation dicts ordered by updated_at DESC.
        """
        convs = self.get_conversations()
        today = date.today()
        yesterday = today - timedelta(days=1)

        groups: dict[str, list[dict]] = {
            "Today": [],
            "Yesterday": [],
            "Previous": [],
        }

        for c in convs:
            try:
                updated = datetime.fromisoformat(c["updated_at"].rstrip("Z")).date()
            except (ValueError, TypeError):
                updated = today

            if updated == today:
                groups["Today"].append(c)
            elif updated == yesterday:
                groups["Yesterday"].append(c)
            else:
                groups["Previous"].append(c)

        return groups

    # -- messages ------------------------------------------------------------

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Optional[dict] = None,
    ) -> str:
        """
        Add a message to a conversation and return the message ID.

        Args:
            conversation_id: Parent conversation ID.
            role: One of 'user', 'assistant', 'system'.
            content: Message text content.
            metadata: Optional dict (stored as JSON) — e.g. pipeline result.
        """
        msg_id = self._new_id()
        now = self._now_iso()
        meta_json = json.dumps(metadata, default=str) if metadata else None

        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO messages (id, conversation_id, role, content, metadata_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (msg_id, conversation_id, role, content, meta_json, now),
            )
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, conversation_id),
            )
            conn.commit()
        finally:
            conn.close()
        return msg_id

    def get_messages(self, conversation_id: str) -> list[dict]:
        """Return all messages for a conversation, ordered by creation time."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT id, conversation_id, role, content, metadata_json, created_at "
                "FROM messages WHERE conversation_id = ? ORDER BY created_at ASC",
                (conversation_id,),
            ).fetchall()
            result = []
            for r in rows:
                msg = dict(r)
                if msg["metadata_json"]:
                    try:
                        msg["metadata"] = json.loads(msg["metadata_json"])
                    except json.JSONDecodeError:
                        msg["metadata"] = None
                else:
                    msg["metadata"] = None
                del msg["metadata_json"]
                result.append(msg)
            return result
        finally:
            conn.close()

    def count_messages(self, conversation_id: Optional[str] = None) -> int:
        """Count messages, optionally filtered by conversation."""
        conn = self._connect()
        try:
            if conversation_id:
                row = conn.execute(
                    "SELECT COUNT(*) AS cnt FROM messages WHERE conversation_id = ?",
                    (conversation_id,),
                ).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) AS cnt FROM messages").fetchone()
            return row["cnt"]
        finally:
            conn.close()

    # -- aggregation helpers -------------------------------------------------

    def aggregate_claim_stats(self) -> dict:
        """
        Aggregate claim verification stats across all assistant messages.

        Returns dict with total_claims_verified, supported, contradicted,
        insufficient counts from stored pipeline result metadata.
        """
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT metadata_json FROM messages WHERE role = 'assistant' AND metadata_json IS NOT NULL"
            ).fetchall()
        finally:
            conn.close()

        total = supported = contradicted = insufficient = 0
        for r in rows:
            try:
                meta = json.loads(r["metadata_json"])
                summary = meta.get("verification", {}).get("summary", {})
                total += summary.get("total_claims", 0)
                supported += summary.get("supported", 0)
                contradicted += summary.get("contradicted", 0)
                insufficient += summary.get("insufficient", 0)
            except (json.JSONDecodeError, TypeError, AttributeError):
                continue

        return {
            "total_claims_verified": total,
            "supported": supported,
            "contradicted": contradicted,
            "insufficient": insufficient,
        }
