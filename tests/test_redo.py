"""Undo/redo cursor tests — undo must be recoverable, and honestly so.

Uses an isolated OPENDOT_HOME so the real ~/.opendot store is never touched.

Successive versions of a file deliberately differ in *length*. take_snapshot's
fast path reuses the previous snapshot's hash when (mtime, size) are both
unchanged, and two same-size writes in a test land inside one filesystem
timestamp tick — so equal-length fixtures make these tests flaky for reasons
that have nothing to do with the cursor.
"""

from pathlib import Path

import pytest

from opendot.reversibility.engine import Reversibility
from opendot.reversibility.rules import load_rules


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENDOT_HOME", str(tmp_path / "store"))
    yield


def _rev(tmp_path) -> tuple[Reversibility, Path]:
    wd = tmp_path / "ws"
    wd.mkdir()
    return Reversibility(workdir=str(wd), rules=load_rules(str(wd))), wd


def _edit(rev: Reversibility, path: Path, text: str) -> None:
    """Do a mutating action the way a tool would: snapshot, then write."""
    rev.before_action("write", str(path), reversible=True)
    path.write_text(text)


# ---- the round trip ----


def test_undo_then_redo_returns_the_pre_undo_state(tmp_path):
    rev, wd = _rev(tmp_path)
    f = wd / "a.txt"
    f.write_text("one")

    _edit(rev, f, "two-two")
    assert f.read_text() == "two-two"

    assert rev.undo_last() is not None
    assert f.read_text() == "one"

    assert rev.redo() is not None
    assert f.read_text() == "two-two"


def test_redo_restores_a_created_file(tmp_path):
    """Undo deletes a file the action created; redo must bring it back."""
    rev, wd = _rev(tmp_path)
    (wd / "keep.txt").write_text("x")

    new = wd / "new.txt"
    rev.before_action("write", str(new), reversible=True)
    new.write_text("created")

    rev.undo_last()
    assert not new.exists()

    rev.redo()
    assert new.read_text() == "created"


def test_redo_restores_a_deleted_file(tmp_path):
    rev, wd = _rev(tmp_path)
    doomed = wd / "doomed.txt"
    doomed.write_text("here")

    rev.before_action("shell", "rm doomed.txt", reversible=True)
    doomed.unlink()

    rev.undo_last()
    assert doomed.read_text() == "here"

    rev.redo()
    assert not doomed.exists()


# ---- walking multiple steps ----


def test_repeated_undo_walks_back_through_history(tmp_path):
    rev, wd = _rev(tmp_path)
    f = wd / "a.txt"
    f.write_text("one")

    _edit(rev, f, "two-two")
    _edit(rev, f, "three-three-three")
    _edit(rev, f, "four-four-four-four")

    rev.undo_last()
    assert f.read_text() == "three-three-three"
    rev.undo_last()
    assert f.read_text() == "two-two"
    rev.undo_last()
    assert f.read_text() == "one"


def test_multiple_undos_then_multiple_redos(tmp_path):
    rev, wd = _rev(tmp_path)
    f = wd / "a.txt"
    f.write_text("one")

    _edit(rev, f, "two-two")
    _edit(rev, f, "three-three-three")
    _edit(rev, f, "four-four-four-four")

    rev.undo_last()
    rev.undo_last()
    assert f.read_text() == "two-two"

    rev.redo()
    assert f.read_text() == "three-three-three"
    rev.redo()
    assert f.read_text() == "four-four-four-four"


def test_undo_past_the_first_action_stops(tmp_path):
    rev, wd = _rev(tmp_path)
    f = wd / "a.txt"
    f.write_text("one")
    _edit(rev, f, "two-two")

    assert rev.undo_last() is not None
    assert rev.undo_last() is None  # nothing left
    assert f.read_text() == "one"


def test_undo_returns_the_entry_it_actually_undid(tmp_path):
    rev, wd = _rev(tmp_path)
    f = wd / "a.txt"
    f.write_text("one")

    _edit(rev, f, "two-two")
    rev.before_action("shell", "echo hi", reversible=True)

    assert rev.undo_last().detail == "echo hi"
    assert rev.undo_last().detail == str(f)


def test_redo_returns_the_entry_it_reapplied(tmp_path):
    rev, wd = _rev(tmp_path)
    f = wd / "a.txt"
    f.write_text("one")

    _edit(rev, f, "two-two")
    rev.before_action("shell", "echo hi", reversible=True)

    rev.undo_last()
    rev.undo_last()
    assert rev.redo().detail == str(f)
    assert rev.redo().detail == "echo hi"


# ---- nothing to redo ----


def test_redo_without_undo_is_none(tmp_path):
    rev, wd = _rev(tmp_path)
    f = wd / "a.txt"
    f.write_text("one")
    _edit(rev, f, "two-two")

    assert rev.redo() is None
    assert f.read_text() == "two-two"  # and does not touch the workspace


def test_redo_on_an_empty_history_is_none(tmp_path):
    rev, _ = _rev(tmp_path)
    assert rev.redo() is None


def test_redo_past_the_head_stops(tmp_path):
    rev, wd = _rev(tmp_path)
    f = wd / "a.txt"
    f.write_text("one")
    _edit(rev, f, "two-two")

    rev.undo_last()
    assert rev.redo() is not None
    assert rev.redo() is None
    assert f.read_text() == "two-two"


# ---- a new action invalidates redo ----


def test_new_action_after_undo_clears_the_redo_stack(tmp_path):
    rev, wd = _rev(tmp_path)
    f = wd / "a.txt"
    f.write_text("one")

    _edit(rev, f, "two-two")
    rev.undo_last()
    assert f.read_text() == "one"

    # A new action from the undone state branches the history.
    _edit(rev, f, "branched-branched-branched-branched-branched")

    assert rev.redo() is None
    assert f.read_text() == "branched-branched-branched-branched-branched"


def test_undo_after_branching_undoes_the_new_action(tmp_path):
    rev, wd = _rev(tmp_path)
    f = wd / "a.txt"
    f.write_text("one")

    _edit(rev, f, "two-two")
    rev.undo_last()
    _edit(rev, f, "branched-branched-branched-branched-branched")

    rev.undo_last()
    assert f.read_text() == "one"


def test_clear_history_drops_redo(tmp_path):
    rev, wd = _rev(tmp_path)
    f = wd / "a.txt"
    f.write_text("one")

    _edit(rev, f, "two-two")
    rev.undo_last()
    rev.clear_history()

    assert rev.redo() is None


def test_clear_redo_drops_the_cursor(tmp_path):
    rev, wd = _rev(tmp_path)
    f = wd / "a.txt"
    f.write_text("one")

    _edit(rev, f, "two-two")
    rev.undo_last()
    rev.clear_redo()

    assert rev.redo() is None
    assert rev.redo_available() == 0


# ---- bookkeeping ----


def test_redo_available_counts_undone_actions(tmp_path):
    rev, wd = _rev(tmp_path)
    f = wd / "a.txt"
    f.write_text("one")

    _edit(rev, f, "two-two")
    _edit(rev, f, "three-three-three")
    assert rev.redo_available() == 0

    rev.undo_last()
    assert rev.redo_available() == 1
    rev.undo_last()
    assert rev.redo_available() == 2
    rev.redo()
    assert rev.redo_available() == 1


def test_undo_does_not_append_to_the_ledger(tmp_path):
    """The ledger is an audit trail of what opendot did, not of navigation."""
    rev, wd = _rev(tmp_path)
    f = wd / "a.txt"
    f.write_text("one")

    _edit(rev, f, "two-two")
    before = len(rev.history())

    rev.undo_last()
    rev.redo()
    assert len(rev.history()) == before


def test_a_corrupt_cursor_file_does_not_break_undo(tmp_path):
    from opendot.reversibility import redo as redo_mod

    rev, wd = _rev(tmp_path)
    f = wd / "a.txt"
    f.write_text("one")
    _edit(rev, f, "two-two")

    redo_mod._path(rev.project_id).write_text("{not json", encoding="utf-8")

    assert rev.undo_last() is not None
    assert f.read_text() == "one"


def test_lockfile_warning_still_reported_through_redo(tmp_path):
    """redo goes through restore_to, so it must report lockfile changes too."""
    rev, wd = _rev(tmp_path)
    (wd / "package-lock.json").write_text('{"version": 1}')

    rev.before_action("shell", "npm install foo", reversible=True)
    (wd / "package-lock.json").write_text('{"version": 2, "note": "upgraded"}')

    rev.undo_last()
    assert any("package-lock.json" in p for p in rev.last_changed_lockfiles)

    rev.redo()
    assert any("package-lock.json" in p for p in rev.last_changed_lockfiles)


# ---- no-snapshot actions (issue #136) ----


def test_trailing_no_snapshot_action_does_not_block_undo_and_redo(tmp_path):
    """Issue #136: A trailing no-snapshot action (snapshot_before="") must be
    skipped by undo_last() so earlier reversible actions can be undone and redone."""
    rev, wd = _rev(tmp_path)
    f = wd / "a.txt"
    f.write_text("one")

    _edit(rev, f, "two-two")
    rev.before_action("shell", "shred secret.txt", snapshot=False)

    assert rev.redo_available() == 0

    undone = rev.undo_last()
    assert undone is not None
    assert undone.detail == str(f)
    assert f.read_text() == "one"
    assert rev.redo_available() == 2

    assert rev.undo_last() is None

    redone = rev.redo()
    assert redone is not None
    assert redone.detail == str(f)
    assert f.read_text() == "two-two"
    assert rev.redo_available() == 0
    assert rev.redo() is None


def test_interleaved_no_snapshot_action_undo_and_redo(tmp_path):
    """A no-snapshot action between reversible actions is skipped during both
    undo and redo while maintaining correct order and cursor bookkeeping."""
    rev, wd = _rev(tmp_path)
    f = wd / "a.txt"
    f.write_text("one")

    _edit(rev, f, "two-two")
    rev.before_action("shell", "shred secret.txt", snapshot=False)
    _edit(rev, f, "three-three-three")

    assert rev.redo_available() == 0

    # Undo C
    undone_c = rev.undo_last()
    assert undone_c is not None
    assert undone_c.detail == str(f)
    assert f.read_text() == "two-two"
    assert rev.redo_available() == 1

    # Undo A (skipping no-snapshot B)
    undone_a = rev.undo_last()
    assert undone_a is not None
    assert undone_a.detail == str(f)
    assert f.read_text() == "one"
    assert rev.redo_available() == 3

    assert rev.undo_last() is None

    # Redo A (lands at snapshot before C, after B)
    redone_a = rev.redo()
    assert redone_a is not None
    assert redone_a.detail == str(f)
    assert f.read_text() == "two-two"
    assert rev.redo_available() == 1

    # Redo C (lands at head)
    redone_c = rev.redo()
    assert redone_c is not None
    assert redone_c.detail == str(f)
    assert f.read_text() == "three-three-three"
    assert rev.redo_available() == 0
    assert rev.redo() is None


def test_multiple_trailing_no_snapshot_actions(tmp_path):
    """Multiple trailing no-snapshot actions are all skipped when undoing."""
    rev, wd = _rev(tmp_path)
    f = wd / "a.txt"
    f.write_text("one")

    _edit(rev, f, "two-two")
    rev.before_action("shell", "shred secret1.txt", snapshot=False)
    rev.before_action("shell", "shred secret2.txt", snapshot=False)

    undone = rev.undo_last()
    assert undone is not None
    assert undone.detail == str(f)
    assert f.read_text() == "one"
    assert rev.redo_available() == 3

    redone = rev.redo()
    assert redone is not None
    assert redone.detail == str(f)
    assert f.read_text() == "two-two"
    assert rev.redo_available() == 0


def test_all_no_snapshot_actions_undo_returns_none(tmp_path):
    """When history contains only no-snapshot actions, undo returns None cleanly."""
    rev, wd = _rev(tmp_path)
    rev.before_action("shell", "shred secret1.txt", snapshot=False)
    rev.before_action("shell", "shred secret2.txt", snapshot=False)

    assert rev.undo_last() is None
    assert rev.redo_available() == 0
    assert rev.redo() is None
