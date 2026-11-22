"""Tests for the SQLite conversation persistence layer."""

import tempfile
from pathlib import Path

import pytest

from ui.conversation_store import ConversationStore


@pytest.fixture
def store(tmp_path):
    """Create a ConversationStore backed by a temp database."""
    db = tmp_path / "test_conversations.db"
    return ConversationStore(db_path=db)


class TestConversationCRUD:
    def test_create_and_get(self, store):
        conv_id = store.create_conversation("Test Chat")
        conv = store.get_conversation(conv_id)
        assert conv is not None
        assert conv["title"] == "Test Chat"
        assert conv["id"] == conv_id

    def test_list_conversations(self, store):
        store.create_conversation("Chat 1")
        store.create_conversation("Chat 2")
        convs = store.get_conversations()
        assert len(convs) == 2

    def test_rename(self, store):
        conv_id = store.create_conversation("Old Title")
        store.rename_conversation(conv_id, "New Title")
        conv = store.get_conversation(conv_id)
        assert conv["title"] == "New Title"

    def test_delete(self, store):
        conv_id = store.create_conversation("To Delete")
        store.add_message(conv_id, "user", "Hello")
        store.delete_conversation(conv_id)
        assert store.get_conversation(conv_id) is None
        assert store.get_messages(conv_id) == []

    def test_count(self, store):
        assert store.count_conversations() == 0
        store.create_conversation("A")
        store.create_conversation("B")
        assert store.count_conversations() == 2

    def test_get_nonexistent(self, store):
        assert store.get_conversation("nonexistent") is None


class TestMessages:
    def test_add_and_get(self, store):
        conv_id = store.create_conversation()
        store.add_message(conv_id, "user", "Hello")
        store.add_message(conv_id, "assistant", "Hi there", metadata={"answer": "Hi"})
        msgs = store.get_messages(conv_id)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "Hello"
        assert msgs[1]["role"] == "assistant"
        assert msgs[1]["metadata"] == {"answer": "Hi"}

    def test_message_metadata_none(self, store):
        conv_id = store.create_conversation()
        store.add_message(conv_id, "user", "No meta")
        msgs = store.get_messages(conv_id)
        assert msgs[0]["metadata"] is None

    def test_count_messages(self, store):
        conv_id = store.create_conversation()
        assert store.count_messages(conv_id) == 0
        store.add_message(conv_id, "user", "One")
        store.add_message(conv_id, "assistant", "Two")
        assert store.count_messages(conv_id) == 2
        assert store.count_messages() == 2

    def test_message_ordering(self, store):
        conv_id = store.create_conversation()
        store.add_message(conv_id, "user", "First")
        store.add_message(conv_id, "assistant", "Second")
        store.add_message(conv_id, "user", "Third")
        msgs = store.get_messages(conv_id)
        assert [m["content"] for m in msgs] == ["First", "Second", "Third"]

    def test_cascade_delete(self, store):
        conv_id = store.create_conversation()
        store.add_message(conv_id, "user", "Msg 1")
        store.add_message(conv_id, "assistant", "Msg 2")
        assert store.count_messages(conv_id) == 2
        store.delete_conversation(conv_id)
        assert store.count_messages(conv_id) == 0


class TestAutoTitle:
    def test_auto_title_from_first_message(self, store):
        conv_id = store.create_conversation()
        store.add_message(conv_id, "user", "What is the train number?")
        title = store.auto_title(conv_id)
        assert title == "What is the train number?"
        conv = store.get_conversation(conv_id)
        assert conv["title"] == "What is the train number?"

    def test_auto_title_truncation(self, store):
        conv_id = store.create_conversation()
        long_text = "A" * 80
        store.add_message(conv_id, "user", long_text)
        title = store.auto_title(conv_id)
        assert len(title) == 53  # 50 + "..."
        assert title.endswith("...")

    def test_auto_title_no_messages(self, store):
        conv_id = store.create_conversation()
        title = store.auto_title(conv_id)
        assert title == "New Chat"


class TestGrouping:
    def test_groups_structure(self, store):
        store.create_conversation("Today Chat")
        groups = store.get_conversations_grouped()
        assert "Today" in groups
        assert "Yesterday" in groups
        assert "Previous" in groups
        # Newly created should be in Today
        assert len(groups["Today"]) == 1
        assert groups["Today"][0]["title"] == "Today Chat"


class TestAggregation:
    def test_aggregate_empty(self, store):
        stats = store.aggregate_claim_stats()
        assert stats["total_claims_verified"] == 0
        assert stats["supported"] == 0

    def test_aggregate_with_data(self, store):
        conv_id = store.create_conversation()
        meta = {
            "verification": {
                "summary": {
                    "total_claims": 5,
                    "supported": 3,
                    "contradicted": 1,
                    "insufficient": 1,
                }
            }
        }
        store.add_message(conv_id, "assistant", "Answer", metadata=meta)
        stats = store.aggregate_claim_stats()
        assert stats["total_claims_verified"] == 5
        assert stats["supported"] == 3
        assert stats["contradicted"] == 1
        assert stats["insufficient"] == 1
