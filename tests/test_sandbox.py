"""Tests for the --sandbox container isolation scaffold (#130 Phase 1).

The live in-container run is validated on a docker/podman host; these cover the
host-side plumbing — runtime detection/fail-closed, the run command, and
copy-in/diff-out commit-back — with the runtime mocked.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from opendot import sandbox
from opendot.reversibility.snapshots import IgnoreRules

# -- runtime detection / fail-closed --


def test_require_runtime_fails_closed_when_none(monkeypatch):
    monkeypatch.setattr(sandbox, "detect_runtime", lambda: None)
    with pytest.raises(sandbox.SandboxError, match="container runtime"):
        sandbox.require_runtime()


def test_require_runtime_returns_detected(monkeypatch):
    monkeypatch.setattr(sandbox, "detect_runtime", lambda: "podman")
    assert sandbox.require_runtime() == "podman"


def test_detect_runtime_prefers_podman(monkeypatch):
    monkeypatch.setattr(sandbox.shutil, "which", lambda name: "/usr/bin/" + name)
    assert sandbox.detect_runtime() == "podman"  # first in preference order


# -- run command shape --


def test_build_run_command_network_off_and_scoped_env():
    argv = sandbox.build_run_command(
        "docker",
        "img",
        Path("/tmp/sbx"),
        "do it",
        "gpt-5.1",
        network=False,
        env_keys=["OPENAI_API_KEY"],
    )
    assert argv[:2] == ["docker", "run"]
    assert "--network" in argv and "none" in argv  # isolated by default
    assert "-e" in argv and "OPENAI_API_KEY" in argv  # only the named key
    # runs opendot one-shot inside the container, auto-approve (already isolated)
    assert argv[-7:] == ["img", "opendot", "-p", "do it", "--model", "gpt-5.1", "--yes"]


def test_build_run_command_network_on_when_allowed():
    argv = sandbox.build_run_command(
        "podman", "img", Path("/tmp/sbx"), "x", "m", network=True, env_keys=[]
    )
    assert "--network" not in argv  # network allowed -> no isolation flag


# -- copy-in / commit-back --


def _fake_runner_editing(edits):
    """A runner that applies `edits(sandbox_dir)` and returns rc=0."""

    def run(argv):
        sbx = Path(argv[argv.index("-v") + 1].split(":")[0])
        edits(sbx)

        class P:
            returncode = 0

        return P()

    return run


def test_run_sandboxed_commits_changes_back(tmp_path, monkeypatch):
    monkeypatch.setattr(sandbox, "detect_runtime", lambda: "docker")
    wd = tmp_path / "ws"
    wd.mkdir()
    (wd / "keep.txt").write_text("original")
    (wd / "gone.txt").write_text("bye")

    def edits(sbx: Path):
        (sbx / "keep.txt").write_text("EDITED")  # modify
        (sbx / "new.txt").write_text("created")  # create
        (sbx / "gone.txt").unlink()  # delete

    res = sandbox.run_sandboxed(
        str(wd), "go", "gpt-5.1", image="img", runner=_fake_runner_editing(edits)
    )
    assert res["returncode"] == 0
    assert set(res["changed"]) == {"keep.txt", "new.txt", "gone.txt"}
    assert (wd / "keep.txt").read_text() == "EDITED"
    assert (wd / "new.txt").read_text() == "created"
    assert not (wd / "gone.txt").exists()


def test_run_sandboxed_no_commit_on_container_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(sandbox, "detect_runtime", lambda: "docker")
    wd = tmp_path / "ws"
    wd.mkdir()
    (wd / "keep.txt").write_text("original")

    def run(argv):
        sbx = Path(argv[argv.index("-v") + 1].split(":")[0])
        (sbx / "keep.txt").write_text("half-done")  # container messed with it then failed

        class P:
            returncode = 1

        return P()

    res = sandbox.run_sandboxed(str(wd), "x", "gpt-5.1", image="img", runner=run)
    assert res["returncode"] == 1
    assert res["changed"] == []
    assert (wd / "keep.txt").read_text() == "original"  # workspace untouched on failure


def test_run_sandboxed_fails_closed_without_runtime(tmp_path, monkeypatch):
    monkeypatch.setattr(sandbox, "detect_runtime", lambda: None)
    with pytest.raises(sandbox.SandboxError):
        sandbox.run_sandboxed(str(tmp_path), "x", "m", image="img", runner=lambda a: None)


def test_run_sandboxed_fails_closed_when_runner_returns_no_process(tmp_path, monkeypatch):
    # A runner that returns None (or anything without .returncode) means the
    # container never ran; that must fail closed, not commit back as rc=0.
    monkeypatch.setattr(sandbox, "detect_runtime", lambda: "docker")
    wd = tmp_path / "ws"
    wd.mkdir()
    (wd / "keep.txt").write_text("original")
    with pytest.raises(sandbox.SandboxError, match="returncode"):
        sandbox.run_sandboxed(str(wd), "x", "m", image="img", runner=lambda a: None)
    assert (wd / "keep.txt").read_text() == "original"  # nothing committed


def test_run_sandboxed_wraps_staging_error_as_sandbox_error(tmp_path, monkeypatch):
    # A copy-in failure surfaces as SandboxError (the CLI's fail-closed handler),
    # not a raw OSError that bypasses it.
    monkeypatch.setattr(sandbox, "detect_runtime", lambda: "docker")

    def boom(*a, **k):
        raise OSError("unreadable")

    monkeypatch.setattr(sandbox, "_copy_workspace", boom)
    wd = tmp_path / "ws"
    wd.mkdir()
    with pytest.raises(sandbox.SandboxError, match="stage workspace"):
        sandbox.run_sandboxed(str(wd), "x", "m", image="img", runner=lambda a: None)


def test_commit_back_reports_unreadable_sandbox_file(tmp_path, monkeypatch):
    # A sandbox file whose hash can't be read is reported in `failed`, not skipped
    # as "unchanged" via a None == None comparison.
    wd = tmp_path / "ws"
    wd.mkdir()
    sbx = tmp_path / "sbx"
    sbx.mkdir()
    (sbx / "bad.txt").write_text("x")

    real_hash = sandbox._hash

    def maybe_none(path):
        return None if path.name == "bad.txt" else real_hash(path)

    monkeypatch.setattr(sandbox, "_hash", maybe_none)
    changed, failed = sandbox.commit_back(sbx, wd, IgnoreRules())
    assert failed == ["bad.txt"]
    assert "bad.txt" not in changed
    assert not (wd / "bad.txt").exists()


def test_commit_back_does_not_follow_symlink_out_of_workspace(tmp_path):
    # A symlinked path component in the real workspace must not let commit-back
    # write through it to a target outside the workspace (containment escape).
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "victim.txt").write_text("original-outside")

    wd = tmp_path / "ws"
    wd.mkdir()
    (wd / "linkdir").symlink_to(outside, target_is_directory=True)

    sbx = tmp_path / "sbx"
    (sbx / "linkdir").mkdir(parents=True)
    (sbx / "linkdir" / "victim.txt").write_text("HACKED")

    changed, failed = sandbox.commit_back(sbx, wd, IgnoreRules())
    # The outside file is untouched; the write landed inside the workspace instead.
    assert (outside / "victim.txt").read_text() == "original-outside"
    assert not (wd / "linkdir").is_symlink()
    assert (wd / "linkdir" / "victim.txt").read_text() == "HACKED"
    assert "linkdir/victim.txt" in changed


def test_commit_back_refuses_when_symlink_cannot_be_neutralized(tmp_path, monkeypatch):
    # If a symlink path component can't be unlinked, commit-back must refuse
    # (-> failed) BEFORE any mkdir traverses it and creates dirs in the target.
    outside = tmp_path / "outside"
    outside.mkdir()

    wd = tmp_path / "ws"
    wd.mkdir()
    (wd / "linkdir").symlink_to(outside, target_is_directory=True)

    sbx = tmp_path / "sbx"
    (sbx / "linkdir").mkdir(parents=True)
    (sbx / "linkdir" / "deep" / "x.txt").parent.mkdir(parents=True)
    (sbx / "linkdir" / "deep" / "x.txt").write_text("nope")

    import pathlib

    real_unlink = pathlib.Path.unlink

    def refuse_link(self, *a, **k):
        if self.is_symlink():
            raise OSError("cannot unlink symlink")
        return real_unlink(self, *a, **k)

    monkeypatch.setattr(pathlib.Path, "unlink", refuse_link)
    changed, failed = sandbox.commit_back(sbx, wd, IgnoreRules())
    assert "linkdir/deep/x.txt" in failed
    # Nothing was created in the symlink target outside the workspace.
    assert not (outside / "deep").exists()


def test_commit_back_refuses_dir_to_file_replacement(tmp_path):
    # The sandbox has a file where the real workspace has a directory. rmtree-ing
    # that dir could wipe ignored subtrees (.venv, node_modules), which commit_back
    # promises never to touch — so it must refuse and report the path as failed.
    wd = tmp_path / "ws"
    (wd / "thing" / ".venv").mkdir(parents=True)
    (wd / "thing" / ".venv" / "keep").write_text("precious")

    sbx = tmp_path / "sbx"
    sbx.mkdir()
    (sbx / "thing").write_text("now a file")  # dir -> file in the sandbox

    changed, failed = sandbox.commit_back(sbx, wd, IgnoreRules())
    assert "thing" in failed
    assert "thing" not in changed
    assert (wd / "thing").is_dir()  # directory (and its contents) left intact
    assert (wd / "thing" / ".venv" / "keep").read_text() == "precious"


def test_commit_back_leaves_ignored_trees_untouched(tmp_path):
    # A change under an ignored tree in the sandbox must not be copied back.
    wd = tmp_path / "ws"
    wd.mkdir()
    sbx = tmp_path / "sbx"
    (sbx / ".git").mkdir(parents=True)
    (sbx / ".git" / "config").write_text("sneaky")
    (sbx / "real.txt").write_text("ok")

    changed, failed = sandbox.commit_back(sbx, wd, IgnoreRules())
    assert "real.txt" in changed
    assert failed == []
    assert not (wd / ".git").exists()  # ignored tree not committed back


def test_commit_back_reports_failed_deletion(tmp_path, monkeypatch):
    # A delete that can't be performed is reported in `failed`, not silently dropped.
    wd = tmp_path / "ws"
    wd.mkdir()
    (wd / "locked.txt").write_text("x")  # present in real, absent in sandbox -> delete
    sbx = tmp_path / "sbx"
    sbx.mkdir()

    import pathlib

    real_unlink = pathlib.Path.unlink

    def boom(self, *a, **k):
        if self.name == "locked.txt":
            raise OSError("permission denied")
        return real_unlink(self, *a, **k)

    monkeypatch.setattr(pathlib.Path, "unlink", boom)
    changed, failed = sandbox.commit_back(sbx, wd, IgnoreRules())
    assert failed == ["locked.txt"]
    assert "locked.txt" not in changed


def test_sandbox_without_prompt_fails_closed(monkeypatch, capsys):
    # --sandbox with no one-shot prompt must error, not run un-isolated.
    import sys as _sys

    from opendot import cli

    monkeypatch.setattr(_sys, "argv", ["opendot", "--sandbox", "--repl"])

    class _Tty:
        def isatty(self):
            return True

    monkeypatch.setattr(_sys, "stdin", _Tty())
    with pytest.raises(SystemExit) as ei:
        cli.main()
    assert ei.value.code == 2


def test_cli_sandbox_propagates_container_exit_code(monkeypatch):
    # A non-zero container exit must become the process exit code so CI/scripting
    # sees the failure, not a silent success.
    import sys as _sys

    from opendot import cli
    from opendot import sandbox as sbx

    monkeypatch.setattr(_sys, "argv", ["opendot", "-p", "do it", "--model", "gpt-5.1", "--sandbox"])
    monkeypatch.setattr(
        sbx,
        "run_sandboxed",
        lambda *a, **k: {"runtime": "docker", "changed": [], "failed": [], "returncode": 3},
    )
    with pytest.raises(SystemExit) as ei:
        cli.main()
    assert ei.value.code == 3


def test_cli_sandbox_partial_commit_back_exits_nonzero(monkeypatch):
    # Container exited 0 but some files could not be committed back -> the
    # workspace is indeterminate, so the CLI must exit non-zero for CI/scripting.
    import sys as _sys

    from opendot import cli
    from opendot import sandbox as sbx

    monkeypatch.setattr(_sys, "argv", ["opendot", "-p", "do it", "--model", "gpt-5.1", "--sandbox"])
    monkeypatch.setattr(
        sbx,
        "run_sandboxed",
        lambda *a, **k: {
            "runtime": "docker",
            "changed": ["a.txt"],
            "failed": ["b.txt"],
            "returncode": 0,
        },
    )
    with pytest.raises(SystemExit) as ei:
        cli.main()
    assert ei.value.code == 1


def test_sandbox_forwards_policy_flags():
    # Regression (#163): --deny/--usd/--tokens/--api-base were silently dropped
    # on the sandbox path; the container ran with a bare --yes.
    from pathlib import Path

    from opendot.sandbox import build_run_command

    argv = build_run_command(
        "docker",
        "img",
        Path("/tmp/sb"),
        "do the thing",
        "m",
        network=False,
        env_keys=["OPENAI_API_KEY"],
        deny=["rm -rf*"],
        usd=0.5,
        tokens=1234,
        api_base="http://localhost:8080",
    )
    tail = " ".join(argv)
    assert "--deny rm -rf*" in tail
    assert "--usd 0.5" in tail
    assert "--tokens 1234" in tail
    assert "--api-base http://localhost:8080" in tail


def test_sandbox_omits_policy_flags_when_unset():
    from pathlib import Path

    from opendot.sandbox import build_run_command

    argv = build_run_command(
        "docker",
        "img",
        Path("/tmp/sb"),
        "do the thing",
        "m",
        network=False,
        env_keys=[],
    )
    tail = " ".join(argv)
    assert "--deny" not in tail
    assert "--usd" not in tail
    assert "--tokens" not in tail
    assert "--api-base" not in tail


def test_sandbox_warns_when_api_base_set_without_network(monkeypatch, capsys):
    """--api-base names a server the container must reach, but the default
    --network none leaves it no network at all. Warn rather than fail obscurely."""
    import sys as _sys

    from opendot import cli
    from opendot import sandbox as sbx

    monkeypatch.setattr(
        _sys,
        "argv",
        ["opendot", "-p", "x", "--model", "m", "--sandbox", "--api-base", "http://localhost:8080"],
    )
    monkeypatch.setattr(
        sbx,
        "run_sandboxed",
        lambda *a, **k: {"runtime": "docker", "changed": [], "failed": [], "returncode": 0},
    )
    with pytest.raises(SystemExit):
        cli.main()
    assert "no network" in capsys.readouterr().out


def test_sandbox_warns_when_api_base_is_loopback_with_network(monkeypatch, capsys):
    """With --sandbox-net the container has a network, but loopback still means
    the container itself, not the host."""
    import sys as _sys

    from opendot import cli
    from opendot import sandbox as sbx

    monkeypatch.setattr(
        _sys,
        "argv",
        [
            "opendot",
            "-p",
            "x",
            "--model",
            "m",
            "--sandbox",
            "--sandbox-net",
            "--api-base",
            "http://localhost:8080",
        ],
    )
    monkeypatch.setattr(
        sbx,
        "run_sandboxed",
        lambda *a, **k: {"runtime": "docker", "changed": [], "failed": [], "returncode": 0},
    )
    with pytest.raises(SystemExit):
        cli.main()
    assert "loopback" in capsys.readouterr().out
