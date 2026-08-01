"""Tests for the coding tools (grep, glob, edit) and that edit is undoable."""

import pytest

from opendot.reversibility.engine import Reversibility
from opendot.reversibility.snapshots import IgnoreRules
from opendot.tools.local import Toolbox


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENDOT_HOME", str(tmp_path / "store"))
    yield


def _tb(tmp_path):
    wd = tmp_path / "ws"
    (wd / "src").mkdir(parents=True)
    (wd / "src" / "a.py").write_text("x = 1  # TODO fix\ny = 2\n")
    (wd / "src" / "b.py").write_text("z = 3\n")
    (wd / "README.md").write_text("hello\n")
    rev = Reversibility(workdir=str(wd), rules=IgnoreRules())
    return Toolbox(str(wd), reversibility=rev, confirm=lambda p: True), wd, rev


def test_grep_finds_matches(tmp_path):
    tb, wd, _ = _tb(tmp_path)
    out = tb.call("grep", {"pattern": "TODO"})
    assert "a.py:1:" in out
    assert "TODO" in out


def test_grep_no_match(tmp_path):
    tb, wd, _ = _tb(tmp_path)
    assert tb.call("grep", {"pattern": "nonexistent_xyz"}) == "no matches"


def test_glob(tmp_path):
    tb, wd, _ = _tb(tmp_path)
    out = tb.call("glob", {"pattern": "**/*.py"})
    assert "src/a.py" in out and "src/b.py" in out
    assert "README.md" not in out

def test_list_files_shows_file_sizes(tmp_path):
    tb, _, _ = _tb(tmp_path)
    out = tb.call("list_files", {})
    assert "src/" in out
    assert "README.md (" in out
    assert "B)" in out


def test_edit_replaces_and_reports(tmp_path):
    tb, wd, _ = _tb(tmp_path)
    out = tb.call("edit", {"path": "src/b.py", "find": "z = 3", "replace": "z = 99"})
    assert "1 replacement" in out
    assert "-z = 3" in out and "+z = 99" in out  # diff is shown
    assert (wd / "src" / "b.py").read_text() == "z = 99\n"


def test_edit_missing_find_is_error_no_change(tmp_path):
    tb, wd, _ = _tb(tmp_path)
    before = (wd / "src" / "b.py").read_text()
    out = tb.call("edit", {"path": "src/b.py", "find": "not there", "replace": "x"})
    assert "not present" in out
    assert (wd / "src" / "b.py").read_text() == before  # unchanged


def test_edit_is_undoable(tmp_path):
    tb, wd, rev = _tb(tmp_path)
    tb.call("edit", {"path": "src/a.py", "find": "x = 1", "replace": "x = 42"})
    assert "x = 42" in (wd / "src" / "a.py").read_text()

    rev.undo_last()  # walk back the edit
    assert (wd / "src" / "a.py").read_text() == "x = 1  # TODO fix\ny = 2\n"


def test_new_tools_are_in_specs(tmp_path):
    tb, _, _ = _tb(tmp_path)
    names = {s["function"]["name"] for s in tb.specs()}
    assert {
        "grep",
        "glob",
        "edit",
        "run_shell",
        "read_file",
        "write_file",
        "list_files",
        "web_fetch",
    } <= names


def test_web_fetch_returns_markdown(tmp_path, monkeypatch):
    """web_fetch delegates to PyScrappy and returns its markdown, without hitting
    the network (PyScrappy's scrape is stubbed)."""
    import pyscrappy

    class _Result:
        def to_markdown(self):
            return "# Example\n\nHello world."

    monkeypatch.setattr(pyscrappy, "scrape", lambda url: _Result())
    tb, _, _ = _tb(tmp_path)
    out = tb.call("web_fetch", {"url": "https://example.com"})
    assert "# Example" in out and "Hello world." in out


def test_web_fetch_rejects_non_http_schemes(tmp_path):
    """Only http(s) — never file:// (would exfiltrate local files) or others."""
    tb, _, _ = _tb(tmp_path)
    for bad in ("file:///etc/passwd", "ftp://x/y", "data:text/plain,hi"):
        out = tb.call("web_fetch", {"url": bad})
        assert out.startswith("error: only http")


def test_web_fetch_is_read_only():
    from opendot.tools.local import Toolbox

    assert "web_fetch" in Toolbox.READ_ONLY  # runs without a snapshot, like read_file


def test_write_outside_workspace_is_irreversible_and_confirmed(tmp_path):
    """A write outside the working dir isn't covered by the snapshot, so it must
    be confirmed first and recorded as NOT reversible (undo that doesn't lie)."""
    tb, wd, rev = _tb(tmp_path)
    outside = tmp_path / "outside.txt"  # sibling of ws/, not under it

    out = tb.call("write_file", {"path": str(outside), "content": "hi"})
    assert "created" in out or "updated" in out
    assert outside.read_text() == "hi"

    last = rev.history()[-1]
    assert last.reversible is False  # ledger tells the truth
    assert "not undoable" in last.note


def test_declining_outside_write_skips_it(tmp_path):
    from opendot.reversibility.engine import Reversibility
    from opendot.reversibility.snapshots import IgnoreRules
    from opendot.tools.local import Toolbox

    wd = tmp_path / "ws"
    wd.mkdir()
    rev = Reversibility(workdir=str(wd), rules=IgnoreRules())
    tb = Toolbox(str(wd), reversibility=rev, confirm=lambda p: False)  # decline

    outside = tmp_path / "outside.txt"
    out = tb.call("write_file", {"path": str(outside), "content": "hi"})
    assert out.startswith("skipped")
    assert not outside.exists()  # nothing written


def test_write_inside_workspace_stays_reversible(tmp_path):
    tb, wd, rev = _tb(tmp_path)
    tb.call("write_file", {"path": "new.txt", "content": "inside"})
    assert rev.history()[-1].reversible is True  # in-workspace = undoable
