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


def test_grep_outside_workspace_returns_matches(tmp_path):
    # Regression: an absolute path outside workdir used to raise ValueError from
    # Path.relative_to; it must now return matches (shown with the absolute path).
    tb, wd, _ = _tb(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "c.py"
    target.write_text("marker_outside = 1\n")
    out = tb.call("grep", {"pattern": "marker_outside", "path": str(outside)})
    assert "marker_outside" in out
    assert str(target) in out


def test_grep_ignore_case(tmp_path):
    tb, wd, _ = _tb(tmp_path)

    # ignore_case=True matches the lowercase pattern against the file's "TODO"
    out = tb.call("grep", {"pattern": "todo", "ignore_case": True})
    assert "TODO" in out

    # default is case-sensitive: "todo" must NOT match "TODO"
    assert tb.call("grep", {"pattern": "todo"}) == "no matches"


def test_grep_schema_exposes_ignore_case(tmp_path):
    tb, _, _ = _tb(tmp_path)
    specs = {s["function"]["name"]: s["function"]["parameters"] for s in tb.specs()}
    assert "ignore_case" in specs["grep"]["properties"]


def test_grep_schema_exposes_context(tmp_path):
    tb, _, _ = _tb(tmp_path)
    specs = {s["function"]["name"]: s["function"]["parameters"] for s in tb.specs()}
    assert "context" in specs["grep"]["properties"]


def test_grep_context_lines_returns_neighbors(tmp_path):
    tb, wd, _ = _tb(tmp_path)
    target = wd / "src" / "a.py"
    target.write_text("line 1\nline 2\nTODO fix\nline 4\nline 5\n")
    out = tb.call("grep", {"pattern": "TODO", "context": 2})
    assert "src/a.py:3:TODO fix" in out
    assert "src/a.py:1-line 1" in out
    assert "src/a.py:2-line 2" in out
    assert "src/a.py:4-line 4" in out
    assert "src/a.py:5-line 5" in out


def test_grep_context_zero_is_default_and_no_context_markers(tmp_path):
    tb, wd, _ = _tb(tmp_path)
    target = wd / "src" / "a.py"
    target.write_text("line 1\nline 2\nTODO fix\nline 4\n")
    out = tb.call("grep", {"pattern": "TODO"})
    assert "src/a.py:3:TODO fix" in out
    assert "src/a.py:3-" not in out
    assert "src/a.py:2-" not in out


def test_grep_context_respects_max_matches(tmp_path):
    tb, wd, _ = _tb(tmp_path)
    target = wd / "src" / "a.py"
    target.write_text(
        "A\nTODO one\nB\n"
        "C\nTODO two\nD\n"
        "E\nTODO three\nF\n"
    )
    out = tb.call("grep", {"pattern": "TODO", "context": 1, "max_matches": 2})
    # Two matches plus their immediate neighbors, then a cap marker.
    assert out.count(":TODO") == 2
    assert out.count("TODO one") == 1
    assert out.count("TODO two") == 1
    assert "TODO three" not in out
    assert "capped at 2" in out


def test_glob(tmp_path):
    tb, wd, _ = _tb(tmp_path)
    out = tb.call("glob", {"pattern": "**/*.py"})
    assert "src/a.py" in out and "src/b.py" in out
    assert "README.md" not in out


def test_list_files_shows_sizes(tmp_path):
    tb, wd, _ = _tb(tmp_path)
    out = tb.call("list_files", {"path": "."})
    # files show a human-readable size; README.md is "hello\n" = 6 bytes
    assert "README.md (6 B)" in out
    # directories are marked with a trailing slash and carry no size
    assert "src/" in out
    assert "src/ (" not in out


def test_list_files_survives_unreadable_entry(tmp_path):
    """A broken symlink (stat() raises) must not crash the whole listing — the
    entry is shown without a size instead."""
    import os

    tb, wd, _ = _tb(tmp_path)
    os.symlink(wd / "does-not-exist", wd / "dangling")
    out = tb.call("list_files", {"path": "."})
    assert "dangling" in out  # listed, no crash
    assert "README.md (6 B)" in out  # readable files still get sizes


def test_list_files_ignores_same_dirs_as_grep_and_glob(tmp_path):
    """list_files must hide the same noise dirs that grep/glob skip (issue #58).

    Previously list_files used its own ad-hoc filter and surfaced .venv and
    .opendot, which grep/glob (via the shared _IGNORE set) hide."""
    tb, wd, _ = _tb(tmp_path)
    for name in (".venv", "venv", ".opendot", "node_modules", "__pycache__"):
        (wd / name).mkdir()
    (wd / "keep").mkdir()

    out = tb.call("list_files", {"path": "."})

    for name in (".venv", "venv", ".opendot", "node_modules", "__pycache__"):
        assert f"{name}/" not in out
    assert "keep/" in out  # non-ignored dirs are still listed


def test_edit_replaces_and_reports(tmp_path):
    tb, wd, _ = _tb(tmp_path)
    out = tb.call("edit", {"path": "src/b.py", "find": "z = 3", "replace": "z = 99"})
    assert "1 replacement" in out
    assert "-z = 3" in out and "+z = 99" in out  # diff is shown
    assert (wd / "src" / "b.py").read_text() == "z = 99\n"


def test_edit_positive_count_limits_replacements(tmp_path):
    tb, wd, _ = _tb(tmp_path)
    target = wd / "src" / "b.py"
    target.write_text("z = 3\nz = 3\nz = 3\n")

    out = tb.call("edit", {"path": "src/b.py", "find": "z = 3", "replace": "z = 99", "count": 1})

    assert "1 replacement" in out
    assert target.read_text() == "z = 99\nz = 3\nz = 3\n"


def test_edit_negative_count_replaces_all_and_reports_actual_count(tmp_path):
    tb, wd, _ = _tb(tmp_path)
    target = wd / "src" / "b.py"
    target.write_text("z = 3\nz = 3\nz = 3\n")

    out = tb.call("edit", {"path": "src/b.py", "find": "z = 3", "replace": "z = 99", "count": -1})

    assert "3 replacement" in out
    assert target.read_text() == "z = 99\nz = 99\nz = 99\n"


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


def test_write_file_noop_does_not_snapshot_or_report_updated(tmp_path):
    """Writing identical content must short-circuit: no ledger entry, no 'updated'."""
    tb, wd, rev = _tb(tmp_path)
    tb.call("write_file", {"path": "new.txt", "content": "hello"})
    history_len_after_first = len(rev.history())
    out = tb.call("write_file", {"path": "new.txt", "content": "hello"})
    assert "no change" in out
    assert "updated" not in out
    assert len(rev.history()) == history_len_after_first

    # An existing EMPTY file re-written as "" is also a no-op (regression: a
    # truthiness check on old content would wrongly snapshot this).
    tb.call("write_file", {"path": "empty.txt", "content": ""})
    history_len_after_empty = len(rev.history())
    out_empty = tb.call("write_file", {"path": "empty.txt", "content": ""})
    assert "no change" in out_empty
    assert len(rev.history()) == history_len_after_empty


def test_read_file_directory_returns_clear_error(tmp_path):
    """read_file on a directory should suggest the right tool instead of a low-level
    IsADirectoryError."""
    tb, wd, _ = _tb(tmp_path)
    out = tb.call("read_file", {"path": "src"})
    assert "is a directory" in out
    assert "list_files" in out
    assert "IsADirectoryError" not in out


def test_read_file_line_range_returns_numbered_slice(tmp_path):
    """A 1-based inclusive start/end pair returns the requested lines with line numbers."""
    tb, wd, _ = _tb(tmp_path)
    out = tb.call("read_file", {"path": "src/a.py", "start": 1, "end": 2})
    assert "1: x = 1  # TODO fix" in out
    assert "2: y = 2" in out
    # A whole-file read (both bounds omitted) is unchanged: raw contents, no
    # line numbers. Only a requested range slices and numbers lines.
    whole = tb.call("read_file", {"path": "src/a.py"})
    assert whole == "x = 1  # TODO fix\ny = 2\n"


def test_read_file_start_only(tmp_path):
    tb, wd, _ = _tb(tmp_path)
    out = tb.call("read_file", {"path": "src/a.py", "start": 2})
    assert out == "2: y = 2"


def test_read_file_end_only(tmp_path):
    tb, wd, _ = _tb(tmp_path)
    out = tb.call("read_file", {"path": "src/a.py", "end": 1})
    assert out == "1: x = 1  # TODO fix"


def test_read_file_invalid_range_returns_clear_error(tmp_path):
    tb, wd, _ = _tb(tmp_path)
    assert "start cannot be greater than end" in tb.call(
        "read_file", {"path": "src/a.py", "start": 2, "end": 1}
    )
    assert "start must be a positive" in tb.call("read_file", {"path": "src/a.py", "start": 0})
    assert "end must be a positive" in tb.call("read_file", {"path": "src/a.py", "end": 0})
    assert "past the end of the file" in tb.call("read_file", {"path": "src/a.py", "start": 10})


def test_read_file_schema_exposes_start_and_end(tmp_path):
    tb, _, _ = _tb(tmp_path)
    specs = {s["function"]["name"]: s["function"]["parameters"] for s in tb.specs()}
    assert "start" in specs["read_file"]["properties"]
    assert "end" in specs["read_file"]["properties"]


def test_max_output_default_when_unset(monkeypatch):
    """The tool-output cap defaults to 30000 when OPENDOT_MAX_TOOL_OUTPUT is unset."""
    from opendot.tools.local import _max_output

    monkeypatch.delenv("OPENDOT_MAX_TOOL_OUTPUT", raising=False)
    assert _max_output() == 30_000


def test_max_output_honors_valid_override(monkeypatch):
    from opendot.tools.local import _max_output

    monkeypatch.setenv("OPENDOT_MAX_TOOL_OUTPUT", "500")
    assert _max_output() == 500


def test_max_output_falls_back_on_invalid_or_nonpositive(monkeypatch):
    """Non-integer and non-positive values keep the 30000 default."""
    from opendot.tools.local import _max_output

    for bad in ("not-a-number", "0", "-5"):
        monkeypatch.setenv("OPENDOT_MAX_TOOL_OUTPUT", bad)
        assert _max_output() == 30_000, f"{bad!r} should fall back to the default"


def test_truncate_respects_env_override(tmp_path, monkeypatch):
    """_truncate uses the configured cap, so a lower override truncates sooner."""
    from opendot.tools.local import _truncate

    monkeypatch.setenv("OPENDOT_MAX_TOOL_OUTPUT", "10")
    out = _truncate("x" * 50)
    assert out.startswith("x" * 10)
    assert "truncated" in out


def test_run_shell_uses_default_timeout_when_env_unset(tmp_path):
    """Without OPENDOT_SHELL_TIMEOUT, a command that finishes quickly succeeds."""
    tb, wd, _ = _tb(tmp_path)
    out = tb.call("run_shell", {"command": "echo hello"})
    assert "hello" in out
    assert "[exit 0]" in out


def test_run_shell_uses_opendot_shell_timeout_env_var(tmp_path, monkeypatch):
    """OPENDOT_SHELL_TIMEOUT overrides the default timeout."""
    monkeypatch.setenv("OPENDOT_SHELL_TIMEOUT", "1")
    tb, wd, _ = _tb(tmp_path)
    out = tb.call("run_shell", {"command": "sleep 2"})
    assert "timed out after 1s" in out


def test_run_shell_invalid_env_var_falls_back_to_default(tmp_path, monkeypatch):
    """A non-numeric OPENDOT_SHELL_TIMEOUT is ignored, falling back to 120s."""
    monkeypatch.setenv("OPENDOT_SHELL_TIMEOUT", "not-a-number")
    tb, wd, _ = _tb(tmp_path)
    out = tb.call("run_shell", {"command": "sleep 0.1"})
    assert "[exit 0]" in out
    assert "timed out" not in out


def test_run_shell_per_call_timeout_overrides_env_var(tmp_path, monkeypatch):
    """An explicit timeout argument still takes precedence over the env default."""
    monkeypatch.setenv("OPENDOT_SHELL_TIMEOUT", "600")
    tb, wd, _ = _tb(tmp_path)
    out = tb.call("run_shell", {"command": "sleep 0.1", "timeout": 1})
    assert "[exit 0]" in out
    assert "timed out" not in out


def test_run_shell_non_positive_env_var_falls_back(tmp_path, monkeypatch):
    """Zero or negative OPENDOT_SHELL_TIMEOUT is treated as invalid."""
    monkeypatch.setenv("OPENDOT_SHELL_TIMEOUT", "0")
    tb, wd, _ = _tb(tmp_path)
    out = tb.call("run_shell", {"command": "sleep 0.1"})
    assert "[exit 0]" in out
    assert "timed out" not in out


def test_run_shell_zero_timeout_argument_uses_default(tmp_path):
    """An explicit timeout=0 should not time out instantly; it should fall back to
    the default so the model gets a useful command result."""
    tb, wd, _ = _tb(tmp_path)
    out = tb.call("run_shell", {"command": "sleep 0.1", "timeout": 0})
    assert "[exit 0]" in out
    assert "timed out" not in out


def test_run_shell_negative_timeout_argument_uses_default(tmp_path):
    """An explicit negative timeout should also fall back to the default."""
    tb, wd, _ = _tb(tmp_path)
    out = tb.call("run_shell", {"command": "sleep 0.1", "timeout": -1})
    assert "[exit 0]" in out
    assert "timed out" not in out


def test_run_shell_positive_timeout_argument_still_enforced(tmp_path):
    """A positive per-call timeout still works normally."""
    tb, wd, _ = _tb(tmp_path)
    out = tb.call("run_shell", {"command": "sleep 2", "timeout": 1})
    assert "timed out after 1s" in out
