"""TUI action ID resolution."""

from opendot.reversibility.ledger import LedgerEntry
from opendot.tui.app import _resolve_action_id


def _entry(action_id: str) -> LedgerEntry:
    return LedgerEntry(
        id=action_id,
        kind="write",
        detail="file.txt",
        snapshot_before="snapshot",
    )


def test_short_id_resolves_padded_action_id():
    target = _entry("000004")

    assert _resolve_action_id([target], "4") is target


def test_short_id_uses_numeric_value_not_suffix():
    first = _entry("000123")
    later = _entry("001123")

    assert _resolve_action_id([first, later], "123") is first


def test_full_padded_id_matches_exactly():
    first = _entry("000123")
    later = _entry("001123")

    assert _resolve_action_id([first, later], "001123") is later


def test_invalid_short_id_returns_no_match():
    assert _resolve_action_id([_entry("000004")], "not-an-id") is None


def test_malformed_stored_id_does_not_crash_numeric_resolution():
    malformed = _entry("not-an-id")
    target = _entry("000004")

    assert _resolve_action_id([malformed, target], "4") is target
