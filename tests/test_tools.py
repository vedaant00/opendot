"""Tests for the coding tools (grep, glob, edit) and that edit is undoable."""

import os
from pathlib import Path

import pytest

from opendot.reversibility.engine import Reversibility
from opendot.reversibility.snapshots import IgnoreRules
from opendot.tools.local import Toolbox, _same_path


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
    target.write_text("A\nTODO one\nB\nC\nTODO two\nD\nE\nTODO three\nF\n")
    out = tb.call("grep", {"pattern": "TODO", "context": 1, "max_matches": 2})
    # Two matches plus their immediate neighbors, then a cap marker.
    assert out.count(":TODO") == 2
    assert out.count("TODO one") == 1
    assert out.count("TODO two") == 1
    assert "TODO three" not in out
    assert "capped at 2" in out


def test_grep_max_matches_is_global_across_files(tmp_path):
    # The cap counts matches across ALL searched files, not per file. Five files
    # with one match each and max_matches=2 must yield 2 total: a per-file counter
    # would never trip (1 < 2 in every file) and leak all 5 through.
    tb, wd, _ = _tb(tmp_path)
    grepdir = wd / "src" / "many"
    grepdir.mkdir()
    for n in range(5):
        (grepdir / f"f{n}.py").write_text(f"TODO marker {n}\n")
    out = tb.call("grep", {"pattern": "TODO marker", "path": "src/many", "max_matches": 2})
    assert out.count(":TODO marker") == 2
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


def test_list_files_recursive_lists_nested_entries(tmp_path):
    # recursive=True walks subdirectories and shows paths relative to the base,
    # so a nested file appears with its dir prefix (issue #85).
    tb, wd, _ = _tb(tmp_path)
    out = tb.call("list_files", {"path": ".", "recursive": True})
    assert "src/" in out
    assert "src/a.py" in out
    assert "src/b.py" in out
    assert "README.md" in out


def test_list_files_default_is_top_level_only(tmp_path):
    # Default (recursive=False) is unchanged: only the top level, so nested files
    # are not listed with a dir prefix.
    tb, wd, _ = _tb(tmp_path)
    out = tb.call("list_files", {"path": "."})
    assert "src/" in out
    assert "src/a.py" not in out  # not descended into
    assert "src/b.py" not in out


def test_list_files_recursive_skips_ignored_dirs(tmp_path):
    # The recursive walk skips the same _IGNORE dirs as grep/glob, in both modes.
    tb, wd, _ = _tb(tmp_path)
    for name in (".venv", ".opendot", "node_modules", "__pycache__"):
        (wd / name).mkdir()
        (wd / name / "junk.py").write_text("x = 1\n")

    out = tb.call("list_files", {"path": ".", "recursive": True})

    for name in (".venv", ".opendot", "node_modules", "__pycache__"):
        assert name not in out  # neither the dir nor its nested junk.py
    assert "src/a.py" in out  # non-ignored nested files still appear


def test_list_files_schema_exposes_recursive(tmp_path):
    tb, _, _ = _tb(tmp_path)
    specs = {s["function"]["name"]: s["function"]["parameters"] for s in tb.specs()}
    assert "recursive" in specs["list_files"]["properties"]


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
        "move",
    } <= names


def test_move_renames_file_and_reports_summary(tmp_path):
    tb, wd, _ = _tb(tmp_path)
    out = tb.call("move", {"src": "src/b.py", "dst": "src/renamed.py"})
    assert "src/b.py -> src/renamed.py" in out
    assert not (wd / "src" / "b.py").exists()
    assert (wd / "src" / "renamed.py").read_text() == "z = 3\n"


def test_move_creates_parent_dirs_of_dst(tmp_path):
    tb, wd, _ = _tb(tmp_path)
    tb.call("move", {"src": "src/b.py", "dst": "new_dir/nested/b.py"})
    assert (wd / "new_dir" / "nested" / "b.py").read_text() == "z = 3\n"


def test_move_is_undoable(tmp_path):
    tb, wd, rev = _tb(tmp_path)
    tb.call("move", {"src": "src/b.py", "dst": "src/renamed.py"})
    assert (wd / "src" / "renamed.py").exists()

    rev.undo_last()

    assert not (wd / "src" / "renamed.py").exists()
    assert (wd / "src" / "b.py").read_text() == "z = 3\n"


def test_move_missing_src_is_clear_error(tmp_path):
    tb, wd, _ = _tb(tmp_path)
    out = tb.call("move", {"src": "src/does_not_exist.py", "dst": "src/renamed.py"})
    assert "error" in out
    assert "not found" in out
    assert not (wd / "src" / "renamed.py").exists()


def test_move_existing_dst_is_clear_error_without_overwrite(tmp_path):
    tb, wd, _ = _tb(tmp_path)
    out = tb.call("move", {"src": "src/b.py", "dst": "src/a.py"})
    assert "error" in out
    assert "already exists" in out
    # nothing changed
    assert (wd / "src" / "b.py").exists()
    assert (wd / "src" / "a.py").read_text() == "x = 1  # TODO fix\ny = 2\n"


def test_move_existing_dst_with_overwrite_replaces_it(tmp_path):
    tb, wd, _ = _tb(tmp_path)
    out = tb.call("move", {"src": "src/b.py", "dst": "src/a.py", "overwrite": True})
    assert "src/b.py -> src/a.py" in out
    assert not (wd / "src" / "b.py").exists()
    assert (wd / "src" / "a.py").read_text() == "z = 3\n"


def test_move_overwrite_rejects_directory_destination(tmp_path):
    """overwrite=true must not silently move src INTO an existing directory dst
    (shutil.move's default behavior) — that contradicts the documented
    'replace it' semantics of overwrite."""
    tb, wd, _ = _tb(tmp_path)
    (wd / "otherdir").mkdir()

    out = tb.call("move", {"src": "src/b.py", "dst": "otherdir", "overwrite": True})

    assert "error" in out
    assert "directory" in out
    assert (wd / "src" / "b.py").exists()  # unmoved
    assert not (wd / "otherdir" / "b.py").exists()  # not moved into the dir either


def test_move_same_src_and_dst_with_overwrite_is_a_no_op(tmp_path):
    """A no-op move (src == dst) with overwrite=true must not delete the
    source: unlinking dst before the move would otherwise wipe it out since
    src and dst are the same file."""
    tb, wd, _ = _tb(tmp_path)
    out = tb.call("move", {"src": "src/b.py", "dst": "src/b.py", "overwrite": True})
    assert "error" not in out
    assert (wd / "src" / "b.py").read_text() == "z = 3\n"


def test_move_symlink_aliasing_is_not_treated_as_same_path(tmp_path):
    """Two distinct symlinks that happen to resolve to the same target must not
    be treated as a same-path no-op — Path.resolve() dereferences symlinks,
    which would wrongly conflate two different paths as identical."""
    tb, wd, _ = _tb(tmp_path)
    target = wd / "src" / "b.py"
    link_a = wd / "link_a.py"
    link_b = wd / "link_b.py"
    link_a.symlink_to(target)
    link_b.symlink_to(target)

    out = tb.call("move", {"src": "link_a.py", "dst": "link_b.py"})

    assert "no change" not in out
    assert "already exists" in out


def test_same_path_distinguishes_different_paths():
    assert _same_path(Path("/ws/src/a.py"), Path("/ws/src/b.py")) is False


def test_same_path_is_case_insensitive_via_normcase(monkeypatch):
    """The same-path comparison must fold case via os.path.normcase, so paths
    differing only by case are recognized as aliases on case-insensitive
    filesystems (Windows, default macOS). normcase is a no-op on this
    (case-sensitive) test host, so monkeypatch it to simulate that platform's
    behavior — this keeps the test deterministic and portable."""
    monkeypatch.setattr(os.path, "normcase", str.lower)
    assert _same_path(Path("/ws/src/B.py"), Path("/ws/src/b.py")) is True


def test_move_symlink_reports_its_own_path_not_target(tmp_path):
    """The returned 'src -> dst' summary (and the confirm/ledger text) must
    show the symlink's own path, not the path it resolves to — self._rel()
    dereferences symlinks, which would misreport a symlink move as if the
    real target file moved."""
    tb, wd, _ = _tb(tmp_path)
    link = wd / "link.py"
    link.symlink_to(wd / "src" / "b.py")

    out = tb.call("move", {"src": "link.py", "dst": "renamed_link.py"})

    assert "link.py -> renamed_link.py" in out
    assert "src/b.py" not in out


def test_move_dst_parent_is_a_file_returns_clean_error(tmp_path):
    """If dst's parent path component is actually an existing file (not a
    directory), mkdir(parents=True) raises — this must surface as the same
    kind of clean error string the rest of move() returns, matching how
    write_file/edit wrap their mkdir+mutate in one try/except."""
    tb, wd, _ = _tb(tmp_path)
    blocker = wd / "blocker.txt"
    blocker.write_text("im a file, not a dir\n")

    out = tb.call("move", {"src": "src/b.py", "dst": "blocker.txt/nested/b.py"})

    assert out.startswith("error moving")
    assert (wd / "src" / "b.py").exists()  # unmoved


def test_move_outside_workspace_is_irreversible_and_confirmed(tmp_path):
    """A move whose destination escapes the workspace isn't covered by the
    snapshot, so it must be confirmed first and recorded as NOT reversible."""
    tb, wd, rev = _tb(tmp_path)
    outside = tmp_path / "outside.py"

    out = tb.call("move", {"src": "src/b.py", "dst": str(outside)})
    assert "src/b.py ->" in out
    assert outside.read_text() == "z = 3\n"

    last = rev.history()[-1]
    assert last.reversible is False
    assert "not undoable" in last.note


def test_declining_outside_move_skips_it(tmp_path):
    wd = tmp_path / "ws"
    (wd / "src").mkdir(parents=True)
    (wd / "src" / "b.py").write_text("z = 3\n")
    rev = Reversibility(workdir=str(wd), rules=IgnoreRules())
    tb = Toolbox(str(wd), reversibility=rev, confirm=lambda p: False)  # decline

    outside = tmp_path / "outside.py"
    out = tb.call("move", {"src": "src/b.py", "dst": str(outside)})
    assert out.startswith("skipped")
    assert not outside.exists()
    assert (wd / "src" / "b.py").exists()  # unchanged


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


def test_write_file_does_not_follow_symlink_out_of_workspace(tmp_path):
    """In-workspace writes use open-time containment (O_NOFOLLOW), so if the
    target path is a symlink pointing outside — including one swapped in after a
    path check (TOCTOU) — the write does not follow it. The symlink is replaced
    by a real in-workspace file instead. (#130)"""
    tb, wd, _ = _tb(tmp_path)
    outside = tmp_path / "outside_secret"
    outside.write_text("ORIGINAL", encoding="utf-8")
    target = wd / "notes.txt"
    os.symlink(outside, target)

    tb._safe_write_within_workspace(target, "NEW")

    assert outside.read_text() == "ORIGINAL"  # never written through the symlink
    assert not target.is_symlink()  # replaced by a real file
    assert target.read_text() == "NEW"


def test_edit_does_not_follow_in_workspace_symlink(tmp_path):
    """`edit` is a mutating write, so it must use the same open-time containment as
    write_file: an in-workspace symlink is replaced by a real file rather than
    written through. It previously used a bare Path.write_text, so editing
    `link.txt` silently rewrote whatever the link pointed at."""
    tb, wd, _ = _tb(tmp_path)
    (wd / "real.txt").write_text("hello", encoding="utf-8")
    os.symlink(wd / "real.txt", wd / "link.txt")

    tb.call("edit", {"path": "link.txt", "find": "hello", "replace": "bye"})

    assert (wd / "real.txt").read_text() == "hello"  # never written through
    assert not (wd / "link.txt").is_symlink()  # replaced by a real file
    assert (wd / "link.txt").read_text() == "bye"


def test_edit_reports_the_named_path_not_a_symlink_target(tmp_path):
    """The report must name the path the caller asked for, not a symlink's target
    (_rel resolved, so editing `link.txt` reported `real.txt`)."""
    tb, wd, _ = _tb(tmp_path)
    (wd / "real.txt").write_text("hello", encoding="utf-8")
    os.symlink(wd / "real.txt", wd / "link.txt")

    out = tb.call("edit", {"path": "link.txt", "find": "hello", "replace": "bye"})

    assert "link.txt" in out.splitlines()[0]
    assert "real.txt" not in out.splitlines()[0]


def test_write_file_normal_in_workspace_still_works(tmp_path):
    tb, wd, _ = _tb(tmp_path)
    out = tb.call("write_file", {"path": "sub/dir/file.txt", "content": "hi"})
    assert "created" in out
    assert (wd / "sub" / "dir" / "file.txt").read_text() == "hi"


def test_openat2_tier_used_when_supported(tmp_path):
    """On a Linux kernel with openat2, the strongest containment tier does a
    normal write successfully (feature-detected; skipped elsewhere)."""
    from opendot.tools.local import _openat2_supported, _write_via_openat2

    if not _openat2_supported():
        pytest.skip("openat2 not available on this platform")
    wd = tmp_path / "ws"
    wd.mkdir()
    p = wd / "file.txt"
    assert _write_via_openat2(str(wd), p, b"hello") is True
    assert p.read_text() == "hello"


def test_openat2_refuses_symlink_escape_when_supported(tmp_path):
    """openat2 with RESOLVE_NO_SYMLINKS returns False for a symlinked target
    (escape refused), so the caller falls back and neutralises it."""
    from opendot.tools.local import _openat2_supported, _write_via_openat2

    if not _openat2_supported():
        pytest.skip("openat2 not available on this platform")
    wd = tmp_path / "ws"
    wd.mkdir()
    outside = tmp_path / "outside"
    outside.write_text("ORIGINAL", encoding="utf-8")
    link = wd / "link.txt"
    os.symlink(outside, link)
    assert _write_via_openat2(str(wd), link, b"NEW") is False
    assert outside.read_text() == "ORIGINAL"


def test_write_refuses_intermediate_symlink_escape(tmp_path):
    """An intermediate path component that is a symlink pointing outside the
    workspace must not be followed — the write stays inside, the symlinked dir is
    replaced by a real one. Closes the intermediate-symlink race on every tier
    (not just Linux openat2). (#130)"""
    tb, wd, _ = _tb(tmp_path)
    outside_dir = tmp_path / "attacker_dir"
    outside_dir.mkdir()
    (wd / "sub").symlink_to(outside_dir)  # wd/sub -> outside

    tb._safe_write_within_workspace(wd / "sub" / "data.txt", "NEW")

    assert not (outside_dir / "data.txt").exists()  # never escaped
    assert not (wd / "sub").is_symlink()  # symlinked component replaced by real dir
    assert (wd / "sub" / "data.txt").read_text() == "NEW"


def test_plain_verified_refuses_intermediate_symlink(tmp_path):
    """The dir_fd-less fallback (Windows path) also refuses an intermediate
    symlink escape, not just the POSIX fd-walk."""
    tb, wd, _ = _tb(tmp_path)
    outside_dir = tmp_path / "attacker_dir2"
    outside_dir.mkdir()
    (wd / "sub").symlink_to(outside_dir)

    tb._write_plain_verified(wd / "sub" / "data.txt", b"NEW")

    assert not (outside_dir / "data.txt").exists()
    assert (wd / "sub" / "data.txt").read_text() == "NEW"


def test_write_outside_workspace_via_plain_path_raises(tmp_path):
    """The dir_fd-less fallback raises on a refusal (path not under workspace)
    instead of silently returning, so write_file can't report a false success."""
    tb, wd, _ = _tb(tmp_path)
    with pytest.raises(OSError):
        tb._write_plain_verified(tmp_path / "outside.txt", b"x")


def test_grep_context_windows_do_not_duplicate_lines(tmp_path):
    # Regression (#165): overlapping context windows re-emitted shared lines.
    tb, wd, _ = _tb(tmp_path)
    (wd / "f.txt").write_text("aaa\nbbb\naaa\nccc\naaa\n")
    out = tb.call("grep", {"pattern": "aaa", "path": str(wd / "f.txt"), "context": 2})
    body_lines = [ln for ln in out.splitlines() if ln.startswith("f.txt:")]
    assert (
        len(body_lines) == 5
    )  # the file has 5 lines; 3 overlapping windows must still emit each once
    import re

    assert [re.match(r"f\.txt:(\d+)", ln).group(1) for ln in body_lines] == [
        "1",
        "2",
        "3",
        "4",
        "5",
    ]
    # A line already printed as context that later matches must be upgraded to
    # the colon marker, not left as a context dash (Copilot review on the PR).
    assert body_lines[0] == "f.txt:1:aaa"  # first hit emitted directly
    assert body_lines[1] == "f.txt:2-bbb"  # pure context stays dashed
    assert body_lines[2] == "f.txt:3:aaa"  # line 3 was line 1's context, then matched
    assert body_lines[4] == "f.txt:5:aaa"  # same upgrade for the last hit


def test_grep_max_matches_zero_returns_no_matches(tmp_path):
    # Regression (#165): cap was checked after appending, so max_matches=0
    # returned one match labelled "(capped at 0)".
    tb, wd, _ = _tb(tmp_path)
    assert tb.call("grep", {"pattern": "TODO", "max_matches": 0}) == "no matches"


def test_grep_max_matches_still_includes_first(tmp_path):
    tb, wd, _ = _tb(tmp_path)
    (wd / "f.txt").write_text("hit one\nhit two\n")
    out = tb.call("grep", {"pattern": "hit", "path": str(wd / "f.txt"), "max_matches": 1})
    assert "hit one" in out
    assert "hit two" not in out
    assert "(capped at 1)" in out
