"""Container sandbox for unattended runs (``--sandbox``).

The classifier decides *when to ask*; it is not a security boundary (see #130).
For an unattended run there is no human to ask, so ``--sandbox`` moves the
guarantee to a **kernel-enforced** boundary: run the agent inside a container
against a *copy* of the workspace, and commit only the resulting workspace diff
back to the real tree on success. Whatever the agent does — opaque subprocess,
symlink race, ``rm -rf`` — is contained by the container, not by parsing shell.

Design (Phase 1 of #130):

- **Opt-in, runtime-gated, fails closed.** ``--sandbox`` needs docker or podman;
  if neither is present it errors, it never silently falls back to running
  directly (that would imply an isolation the run doesn't have).
- **Copy-in / diff-out.** The workspace is copied to a temp dir, that copy is
  mounted into the container, and after the run the copy is reconciled back to
  the real workspace through the reversibility engine — so the commit-back is
  itself snapshotted and undoable.
- **Only for one-shot runs.** An interactive TUI/REPL inside a headless
  container isn't useful; ``--sandbox`` applies to ``opendot -p`` (and CI).

NOTE: the *live in-container execution* is validated on a docker/podman host; the
host-side plumbing here (runtime detection, gating, copy-in, commit-back) is
unit-tested with the runtime mocked.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from opendot.reversibility.rules import load_rules
from opendot.reversibility.snapshots import IgnoreRules, _iter_files

# Container runtimes we support, in preference order. podman first: rootless by
# default, so the contained process can't trivially act as host root.
_RUNTIMES = ("podman", "docker")


class SandboxError(Exception):
    """Raised when a sandbox run can't be set up (no runtime, copy failure, …).

    ``--sandbox`` fails closed: this is surfaced to the user, never swallowed
    into a silent direct run."""


def detect_runtime() -> str | None:
    """Return the first available container runtime on PATH, or None."""
    for rt in _RUNTIMES:
        if shutil.which(rt):
            return rt
    return None


def require_runtime() -> str:
    """Return an available runtime, or raise SandboxError (fail closed)."""
    rt = detect_runtime()
    if rt is None:
        raise SandboxError(
            "--sandbox needs a container runtime, but neither podman nor docker "
            "was found on PATH. Install one, or drop --sandbox to run directly "
            "(no kernel isolation)."
        )
    return rt


def _copy_workspace(workdir: Path, dest: Path, rules: IgnoreRules) -> None:
    """Copy the (non-ignored) workspace tree into ``dest``.

    Ignored trees (.git, node_modules, venvs, …) are skipped per the ignore rules,
    so the sandbox works on the same ignored-path view the reversibility engine
    reconciles. (Unlike snapshotting, this does not skip very large files — the
    whole non-ignored tree is copied.)"""
    for f in _iter_files(workdir, rules):
        rel = f.relative_to(workdir)
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, target)


def _hash(path: Path) -> str | None:
    import hashlib

    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()
    except OSError:
        return None


def _prepare_dest(dst: Path, real_root: Path) -> None:
    """Make ``dst`` safe to write during commit-back, or raise SandboxError.

    The sandbox is untrusted, so a path it produces must not let commit-back
    write outside the workspace. A symlink anywhere in ``dst``'s existing path
    would make ``shutil.copy2`` (and any ``mkdir``) follow it and touch the link
    target — possibly outside ``real_root``.

    Everything here is check-first: no directory is created and no file is copied
    until the whole path is proven contained. Walking top-down from ``real_root``:

    - each existing component that resolves outside ``real_root`` is refused;
    - a symlink component is removed so nothing downstream follows it, and if it
      can't be removed we refuse rather than fall through to a ``mkdir`` that
      would traverse it and create dirs in the link target;
    - only once the path is clean and contained do we create parents.
    """
    # dst must be lexically under real_root to begin with; a crafted rel path with
    # ".." components would fail this before we touch the filesystem.
    if not dst.is_relative_to(real_root):
        raise SandboxError(f"commit-back destination escapes workspace: {dst}")

    # Walk the components strictly between real_root and dst (real_root is the
    # trusted base and is never a symlink we'd remove), top-down, ending at dst.
    rel_parts = dst.relative_to(real_root).parts
    for i in range(1, len(rel_parts) + 1):
        comp = real_root.joinpath(*rel_parts[:i])
        if comp.is_symlink():
            # Remove the link (never follow it). Do NOT swallow the error: a
            # symlink we can't unlink must abort before any mkdir traverses it and
            # creates directories inside the link target.
            try:
                comp.unlink()
            except OSError as exc:
                raise SandboxError(
                    f"commit-back can't neutralize symlink on path: {comp} ({exc})"
                ) from exc
        elif comp == dst and comp.is_dir():
            # A real directory where the sandbox now has a file. We must NOT rmtree
            # it: that tree can contain ignored subtrees (.venv, node_modules, …)
            # that commit_back promises never to touch, and _iter_files only skips
            # them during enumeration, not during a recursive delete. Refuse the
            # replacement and let the caller report it as failed.
            raise SandboxError(
                f"commit-back destination is a directory (refusing to delete): {dst}"
            )
    # Path is now symlink-free and contained; safe to create the parent tree.
    dst.parent.mkdir(parents=True, exist_ok=True)


def commit_back(
    sandbox_dir: Path, workdir: Path, rules: IgnoreRules
) -> tuple[list[str], list[str]]:
    """Reconcile the post-run sandbox copy back into the real workspace.

    Applies files that were created or changed in the sandbox, and removes
    (non-ignored) files the sandbox deleted, so the real workspace ends up
    matching the sandbox's non-ignored view. Ignored trees are never touched.

    Returns ``(changed, failed)``: relative paths successfully reconciled, and
    paths that could NOT be reconciled (a copy or delete that raised) — the
    caller surfaces ``failed`` so a partial commit-back is never reported as a
    clean one. The caller snapshots the real workspace *before* calling this, so
    the whole commit-back is itself undoable (subject to the snapshot size limit).
    """
    changed: list[str] = []
    failed: list[str] = []
    real_root = workdir.resolve()

    sandbox_files = {
        f.relative_to(sandbox_dir).as_posix(): f for f in _iter_files(sandbox_dir, rules)
    }
    real_files = {f.relative_to(workdir).as_posix(): f for f in _iter_files(workdir, rules)}

    # 1. Create / update files present in the sandbox (new, or content changed).
    for rel, src in sandbox_files.items():
        dst = workdir / rel
        src_h = _hash(src)
        if src_h is None:
            # Can't read the sandbox file to compare or copy it — report it rather
            # than let a None==None hash comparison skip it as "unchanged".
            failed.append(rel)
            continue
        dst_h = _hash(dst) if rel in real_files else None
        if rel not in real_files or src_h != dst_h:
            try:
                _prepare_dest(dst, real_root)
                shutil.copy2(src, dst)
                changed.append(rel)
            except (OSError, SandboxError):
                failed.append(rel)

    # 2. Remove files the sandbox deleted (present in real, absent in sandbox).
    for rel, dst in real_files.items():
        if rel not in sandbox_files:
            try:
                dst.unlink()
                changed.append(rel)
            except OSError:
                failed.append(rel)

    return sorted(set(changed)), sorted(set(failed))


def build_run_command(
    runtime: str,
    image: str,
    sandbox_dir: Path,
    prompt: str,
    model: str,
    *,
    network: bool,
    env_keys: list[str],
    deny: list[str] | None = None,
    usd: float | None = None,
    tokens: int | None = None,
    api_base: str | None = None,
) -> list[str]:
    """Build the ``docker/podman run`` argv that runs opendot one-shot inside the
    container against the mounted sandbox workspace.

    - the sandbox copy is mounted at /work and used as the working dir
    - network is off by default (``--network none``) unless explicitly allowed
    - only the named API-key env vars are forwarded (``-e KEY``), nothing else
    - runs ``opendot -p <prompt> --yes`` inside (already isolated, so auto-approve)
    """
    argv = [runtime, "run", "--rm", "-v", f"{sandbox_dir}:/work", "-w", "/work"]
    if not network:
        argv += ["--network", "none"]
    for key in env_keys:
        argv += ["-e", key]  # forward the value from the host env by name
    argv += [image, "opendot", "-p", prompt, "--model", model, "--yes"]
    # Policy and budget flags must survive the trip into the container: a
    # hard-blocked pattern must not become auto-approved just because the run
    # is sandboxed.
    for pat in deny or []:
        argv += ["--deny", pat]
    if usd is not None:
        argv += ["--usd", str(usd)]
    if tokens is not None:
        argv += ["--tokens", str(tokens)]
    if api_base:
        argv += ["--api-base", api_base]
    return argv


def run_sandboxed(
    workdir: str,
    prompt: str,
    model: str,
    *,
    image: str,
    network: bool = False,
    env_keys: list[str] | None = None,
    deny: list[str] | None = None,
    usd: float | None = None,
    tokens: int | None = None,
    api_base: str | None = None,
    runner=subprocess.run,
) -> dict:
    """Run one opendot turn inside a container against a copy of ``workdir``, then
    commit the resulting diff back to the real workspace (snapshotted first).

    Returns ``{"runtime", "changed", "failed", "returncode"}``. Raises SandboxError
    if no runtime is available, the workspace can't be staged, or the runner returns
    no usable process. ``runner`` is injected for testing (defaults to subprocess.run).
    """
    import tempfile

    runtime = require_runtime()
    wd = Path(workdir).resolve()
    rules = load_rules(str(wd))
    env_keys = env_keys or []

    staging = Path(tempfile.mkdtemp(prefix="opendot-sbx-"))
    sandbox_dir = staging / "work"
    sandbox_dir.mkdir()
    try:
        try:
            _copy_workspace(wd, sandbox_dir, rules)
        except OSError as exc:
            raise SandboxError(f"failed to stage workspace copy: {exc}") from exc

        argv = build_run_command(
            runtime,
            image,
            sandbox_dir,
            prompt,
            model,
            network=network,
            env_keys=env_keys,
            deny=deny,
            usd=usd,
            tokens=tokens,
            api_base=api_base,
        )
        proc = runner(argv)
        # A runner that returns nothing usable means the container never ran; treat
        # that as a failure rather than a returncode-0 "success" that commits back.
        if proc is None or not hasattr(proc, "returncode"):
            raise SandboxError("container runner returned no process/returncode")
        returncode = int(proc.returncode)

        changed: list[str] = []
        failed: list[str] = []
        if returncode == 0:
            # Snapshot the real workspace first so the commit-back is undoable,
            # then reconcile the sandbox result into it.
            from opendot.reversibility.engine import Reversibility

            rev = Reversibility(workdir=str(wd), rules=rules)
            rev.before_action(
                "write", "sandbox commit", reversible=True, note="sandbox --sandbox run"
            )
            changed, failed = commit_back(sandbox_dir, wd, rules)
        return {
            "runtime": runtime,
            "changed": changed,
            "failed": failed,
            "returncode": returncode,
        }
    finally:
        shutil.rmtree(staging, ignore_errors=True)
