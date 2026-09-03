"""Local tools the agent can call — they act on the user's REAL filesystem and
shell, rooted at the working directory. No sandbox (opendot's safety model is
reversibility, not isolation — the snapshot/undo engine lands in M2).

Each tool is a `Tool` (name, JSON-schema spec, and a Python callable). The specs
are the OpenAI/LiteLLM function-calling format. Callables return a string that
becomes the tool result fed back to the model.
"""

from __future__ import annotations

import difflib
import errno
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

# --- Linux openat2 containment (strongest tier of _safe_write_within_workspace) --
#
# openat2(2) opens a path relative to a directory fd with resolution flags the
# kernel enforces atomically. RESOLVE_BENEATH forbids escaping the dir_fd (via
# ..  or an absolute path) and RESOLVE_NO_SYMLINKS forbids following any symlink
# in the path — closing the TOCTOU races a userspace path check cannot. It's
# Linux-only and not in the stdlib, so it's bound via ctypes and feature-detected
# at runtime; every other platform uses the openat/O_NOFOLLOW + verify fallback.

_RESOLVE_NO_SYMLINKS = 0x04
_RESOLVE_BENEATH = 0x08
# openat2 syscall number is stable on the common 64-bit arches (x86-64, arm64).
_SYS_openat2 = (
    {"x86_64": 437, "aarch64": 437, "arm64": 437}.get(os.uname().machine)
    if (sys.platform.startswith("linux") and hasattr(os, "uname"))
    else None
)


_openat2_probe: bool | None = None


def _openat2_supported() -> bool:
    """True only if the running kernel actually has openat2 (>= 5.6), not just if
    the syscall number is known — kernels < 5.6 return ENOSYS. Probed once (with a
    deliberately bad fd so the probe never touches the filesystem) and cached."""
    global _openat2_probe
    if _SYS_openat2 is None:
        return False
    if _openat2_probe is not None:
        return _openat2_probe
    import ctypes

    try:
        libc = ctypes.CDLL(None, use_errno=True)

        class _How(ctypes.Structure):
            _fields_ = [
                ("flags", ctypes.c_uint64),
                ("mode", ctypes.c_uint64),
                ("resolve", ctypes.c_uint64),
            ]

        how = _How(flags=os.O_RDONLY, mode=0, resolve=0)
        # Bad dir_fd (-1) => the syscall returns EBADF if openat2 exists, ENOSYS
        # if the kernel doesn't implement it. Either way nothing is opened.
        libc.syscall(
            ctypes.c_long(_SYS_openat2),
            ctypes.c_int(-1),
            ctypes.c_char_p(b"."),
            ctypes.byref(how),
            ctypes.c_size_t(ctypes.sizeof(how)),
        )
        _openat2_probe = ctypes.get_errno() != errno.ENOSYS
    except Exception:  # noqa: BLE001 - any failure => treat as unsupported
        _openat2_probe = False
    return _openat2_probe


def _write_via_openat2(workdir, p: Path, data: bytes) -> bool:
    """Write ``data`` to ``p`` using Linux openat2 with RESOLVE_BENEATH |
    RESOLVE_NO_SYMLINKS, anchored at ``workdir``. Returns True on success, False
    if openat2 isn't available/usable (caller then uses the portable fallback).

    A path that escapes the workspace (symlink or ..) makes openat2 fail, so the
    write simply doesn't happen through it — the caller's fallback then replaces
    the offending entry with a real in-workspace file.
    """
    if not _openat2_supported():
        return False
    import ctypes

    # Lexical relative path — NOT p.resolve(), which would dereference a symlinked
    # component in userspace before openat2 sees it, defeating RESOLVE_NO_SYMLINKS.
    # openat2 enforces containment itself and refuses symlinks atomically.
    wd = Path(workdir).resolve()
    cand = p if p.is_absolute() else wd / p
    cand = Path(os.path.normpath(str(cand)))
    try:
        rel = str(cand.relative_to(wd))
    except ValueError:
        return False  # not lexically under the workspace; let the caller handle it
    if rel == "." or rel.startswith(".."):
        return False

    class _OpenHow(ctypes.Structure):
        _fields_ = [
            ("flags", ctypes.c_uint64),
            ("mode", ctypes.c_uint64),
            ("resolve", ctypes.c_uint64),
        ]

    libc = ctypes.CDLL(None, use_errno=True)
    try:
        dir_fd = os.open(str(Path(workdir).resolve()), os.O_RDONLY)
    except OSError:
        return False
    try:
        how = _OpenHow(
            flags=os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
            mode=0o644,
            resolve=_RESOLVE_BENEATH | _RESOLVE_NO_SYMLINKS,
        )
        fd = libc.syscall(
            ctypes.c_long(_SYS_openat2),
            ctypes.c_int(dir_fd),
            ctypes.c_char_p(rel.encode()),
            ctypes.byref(how),
            ctypes.c_size_t(ctypes.sizeof(how)),
        )
        if fd < 0:
            # ELOOP/EXDEV/etc: escape refused, or ENOSYS on a kernel < 5.6.
            return False
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        return True
    except (OSError, ValueError):
        return False
    finally:
        os.close(dir_fd)


def _max_output() -> int:
    """Cap tool output so one huge file can't blow the context.

    Override with OPENDOT_MAX_TOOL_OUTPUT (characters). Invalid values keep the
    default of 30_000.
    """
    raw = os.environ.get("OPENDOT_MAX_TOOL_OUTPUT", "30000")
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return 30_000
    return value if value > 0 else 30_000


def _shell_timeout() -> int:
    """Default shell timeout read from ``OPENDOT_SHELL_TIMEOUT``.

    Falls back to 120s when the variable is unset or not a positive integer.
    """
    raw = os.environ.get("OPENDOT_SHELL_TIMEOUT", "120")
    try:
        value = int(raw)
    except ValueError:
        return 120
    return value if value > 0 else 120


def _unified_diff(old: str, new: str, path: str, max_lines: int = 200) -> str:
    """A unified diff of a file change, for the model and for colored rendering.

    opendot shows what it changed (not just 'wrote N chars') — that transparency
    is the point. The TUI colors +/- lines; the model reads the same text.
    """
    diff = list(
        difflib.unified_diff(
            old.splitlines(),
            new.splitlines(),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            lineterm="",
        )
    )
    if not diff:
        return "(no change)"
    if len(diff) > max_lines:
        diff = diff[:max_lines] + [f"... (+{len(diff) - max_lines} more diff lines)"]
    return "\n".join(diff)


def _same_path(a: Path, b: Path) -> bool:
    """True if `a` and `b` denote the same filesystem path, compared lexically
    (normpath + normcase) — NOT via Path.resolve(), which would dereference
    symlinks and wrongly conflate two distinct paths pointing at the same
    target. normcase folds case on case-insensitive filesystems (Windows,
    default macOS) and is a no-op elsewhere.
    """

    def _norm(p: Path) -> str:
        return os.path.normcase(os.path.normpath(str(p)))

    return _norm(a) == _norm(b)


def _truncate(text: str) -> str:
    max_output = _max_output()
    if len(text) > max_output:
        return text[:max_output] + f"\n... [truncated, {len(text)} chars total]"
    return text


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema for the args
    run: Callable[..., str]

    def spec(self) -> dict[str, Any]:
        """OpenAI/LiteLLM tool-definition format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class Toolbox:
    """The set of tools available in a run, rooted at ``workdir``.

    If a ``Reversibility`` engine is given, every mutating tool (write_file,
    run_shell) snapshots the workspace and logs the action *before* executing,
    so it can be undone. A ``confirm`` callback (returns True to proceed) gates
    actions the classifier marks irreversible.
    """

    #: tools that only observe — never mutate the filesystem or run commands.
    READ_ONLY = {
        "list_files",
        "read_file",
        "grep",
        "glob",
        "read_xlsx",
        "read_pptx",
        "read_docx",
        "web_fetch",
    }

    def __init__(
        self,
        workdir: str,
        reversibility=None,
        confirm=None,
        read_only: bool = False,
        mcp_manager=None,
    ) -> None:
        self.workdir = Path(workdir).resolve()
        self.rev = reversibility
        self._confirm = confirm or (lambda prompt: True)
        self.read_only = read_only
        self.mcp = mcp_manager
        tools = self._build()
        # Office tools (.xlsx/.pptx) only when the optional deps are installed.
        from opendot.tools.office import build_office_tools

        tools += build_office_tools(self)
        if read_only:
            tools = [t for t in tools if t.name in self.READ_ONLY]
        self._tools = {t.name: t for t in tools}
        # External MCP tools (never in read-only explorer boxes).
        self._mcp_tools = {}
        if self.mcp is not None and not read_only:
            self._mcp_tools = {t.qualified: t for t in self.mcp.tools}

    def forget_mcp_server(self, server: str) -> None:
        """Drop a removed MCP server's cached tools so the agent stops offering
        (and can't call) them for the rest of this session."""
        self._mcp_tools = {q: t for q, t in self._mcp_tools.items() if t.server != server}

    _IGNORE = {".git", "node_modules", "__pycache__", ".venv", "venv", ".opendot"}

    # -- resolution / safety helpers --
    def _resolve(self, path: str) -> Path:
        """Resolve a (possibly relative) path against workdir."""
        p = Path(path)
        if not p.is_absolute():
            p = self.workdir / p
        return p

    def _rel(self, p: Path) -> str:
        """Path relative to workdir for display, falling back to absolute."""
        try:
            return str(p.resolve().relative_to(self.workdir))
        except ValueError:
            return str(p)

    def _rel_no_resolve(self, p: Path) -> str:
        """Like _rel, but purely lexical (normpath, no Path.resolve()). Used
        where the path itself — e.g. a symlink — is what the user asked to
        act on, not whatever it points at; resolving would misreport it."""
        norm = Path(os.path.normpath(str(p)))
        wd = Path(os.path.normpath(str(self.workdir)))
        try:
            return str(norm.relative_to(wd))
        except ValueError:
            return str(norm)

    def _is_outside_workspace(self, p: Path) -> bool:
        """True if ``p`` is not under the working directory. Writes/edits to such
        paths can't be undone (the snapshot only covers the workspace), so they're
        recorded as irreversible and confirmed first, like escaping shell commands."""
        try:
            p.resolve().relative_to(self.workdir)
            return False
        except ValueError:
            return True

    def _safe_write_within_workspace(self, p: Path, content: str) -> None:
        """Write ``content`` to ``p`` (inside the workspace) with the strongest
        open-time containment the OS offers, so a symlinked / swapped path
        component can't redirect the write outside the snapshot boundary (the
        TOCTOU cases a text-level path check can't close).

        Behaviour is consistent across platforms: if any path component is a
        symlink or would escape the workspace, the write does not follow it — the
        offending entry is removed and a real in-workspace file is created in its
        place. The *mechanism* differs by what the platform provides:

        - Linux with ``openat2`` -> ``RESOLVE_BENEATH | RESOLVE_NO_SYMLINKS``:
          the kernel refuses any symlink or ``..`` escape atomically (strongest).
        - Any POSIX with ``dir_fd`` (macOS/BSD, older Linux) -> a component-by-
          component ``openat`` walk with ``O_NOFOLLOW``, so *no* intermediate
          symlink is ever followed either — the same containment as openat2,
          built from openat primitives (race-free per component).
        - Windows / no ``dir_fd`` -> rebuild each path component as a real dir
          (replacing an escaping symlink), then write.
        """
        data = content.encode("utf-8")

        # Tier 1: Linux openat2 with RESOLVE_BENEATH (atomic, race-free).
        if _write_via_openat2(self.workdir, p, data):
            return

        # Tier 2: component-by-component openat + O_NOFOLLOW walk. Closes the
        # intermediate-symlink race too, on any platform with dir_fd support.
        if os.open in getattr(os, "supports_dir_fd", set()) and hasattr(os, "O_NOFOLLOW"):
            if self._write_via_fd_walk(p, data):
                return

        # Tier 3: no dir_fd (e.g. Windows). Neutralise an escaping symlink, write,
        # then verify the bytes landed inside the workspace.
        self._write_plain_verified(p, data)

    def _rel_parts(self, p: Path) -> list[str] | None:
        """Lexical path components of ``p`` relative to the workspace, or None if
        ``p`` is not lexically under it or contains a ``..`` segment.

        Deliberately does NOT call ``resolve()`` — resolving would follow a
        symlinked component and defeat the fd-walk, whose whole job is to refuse
        exactly those. The walk enforces containment; this only supplies the
        literal component names to walk.
        """
        # Make the candidate absolute against the (already-resolved) workdir and
        # normalise lexically — NOT resolve(), which would follow a symlinked
        # component and defeat the fd-walk. The fd-walk enforces containment; this
        # just supplies the literal component names, anchored at self.workdir.
        cand = Path(p)
        if not cand.is_absolute():
            cand = self.workdir / cand
        cand = Path(os.path.normpath(str(cand)))
        try:
            rel = cand.relative_to(self.workdir)
        except ValueError:
            return None
        parts = list(rel.parts)
        if not parts or any(part == ".." for part in parts):
            return None
        return parts

    def _write_via_fd_walk(self, p: Path, data: bytes) -> bool:
        """Open every path component under the workspace with openat+O_NOFOLLOW,
        so no symlink at *any* level is followed. Creates missing intermediate
        dirs as real dirs. Returns True on success, False to fall through.

        Anchoring each open to the previous component's fd (not a re-resolved
        string) is what makes this race-free per component: a swap after we've
        opened a dir fd can't change what that fd points at.
        """
        parts = self._rel_parts(p)
        if not parts:
            return False
        nofollow = os.O_NOFOLLOW
        wd_fd = None
        cur_fd = None
        try:
            wd_fd = os.open(str(self.workdir), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            cur_fd = wd_fd
            # Walk/create intermediate directories, each O_NOFOLLOW-checked.
            for comp in parts[:-1]:
                try:
                    nxt = os.open(
                        comp, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | nofollow, dir_fd=cur_fd
                    )
                except FileNotFoundError:
                    os.mkdir(comp, 0o755, dir_fd=cur_fd)
                    nxt = os.open(
                        comp, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | nofollow, dir_fd=cur_fd
                    )
                except OSError as exc:
                    # ELOOP => an intermediate component is a symlink: refuse.
                    if getattr(exc, "errno", None) in (errno.ELOOP, errno.ENOTDIR):
                        return False
                    raise
                if cur_fd != wd_fd:
                    os.close(cur_fd)
                cur_fd = nxt
            # Open (or replace) the final component as a real file.
            name = parts[-1]
            flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC | nofollow
            try:
                fd = os.open(name, flags, 0o644, dir_fd=cur_fd)
            except OSError as exc:
                if getattr(exc, "errno", None) == errno.ELOOP:
                    os.unlink(name, dir_fd=cur_fd)  # final component was a symlink
                    fd = os.open(name, flags, 0o644, dir_fd=cur_fd)
                else:
                    return False
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            return True
        except OSError:
            return False
        finally:
            if cur_fd is not None and cur_fd != wd_fd:
                os.close(cur_fd)
            if wd_fd is not None:
                os.close(wd_fd)

    def _write_plain_verified(self, p: Path, data: bytes) -> None:
        """Last-resort write for platforms without dir_fd (e.g. Windows).

        Rebuilds every path component from the workspace root down as a REAL
        directory — replacing any symlinked component (which could redirect the
        write outside) with a real dir — then writes the file. This mirrors the
        fd-walk's containment lexically, so an intermediate-symlink escape is
        neutralised here too, not just on POSIX.
        """
        parts = self._rel_parts(p)
        if parts is None:
            # Not lexically under the workspace: refuse (raise so write_file
            # surfaces an error rather than falsely reporting success).
            raise OSError(f"refused: {p} is not inside the workspace")
        cur = self.workdir
        for comp in parts[:-1]:
            cur = cur / comp
            if cur.is_symlink():
                cur.unlink()  # replace an escaping symlinked dir with a real one
            if not cur.exists():
                cur.mkdir(mode=0o755)
            elif not cur.is_dir():
                raise OSError(f"refused: {cur} is not a directory")
        target = cur / parts[-1]
        if target.is_symlink():
            target.unlink()  # never write through a symlinked final component
        target.write_bytes(data)
        # Belt-and-suspenders: confirm the bytes actually landed inside the
        # workspace. The component rebuild above should guarantee it, but if a
        # concurrent swap slipped through, refuse loudly rather than report a
        # false success (write_file turns this into an error).
        try:
            target.resolve().relative_to(self.workdir)
        except ValueError:
            raise OSError(f"refused: {target} escaped the workspace") from None

    def _is_ignored(self, p: Path) -> bool:
        return any(part in self._IGNORE for part in p.parts)

    def _walk_files(self, base: Path):
        """Yield files under base, skipping ignored dirs (for grep)."""
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in self._IGNORE]
            for fn in filenames:
                yield Path(dirpath) / fn

    # -- the tools --
    def _build(self) -> list[Tool]:
        def _format_size(num_bytes: int) -> str:
            if num_bytes < 1024:
                return f"{num_bytes} B"
            if num_bytes < 1024 * 1024:
                return f"{num_bytes / 1024:.1f} KB"
            if num_bytes < 1024 * 1024 * 1024:
                return f"{num_bytes / (1024 * 1024):.1f} MB"
            return f"{num_bytes / (1024 * 1024 * 1024):.1f} GB"

        def _entry_line(e: Path, name: str) -> str:
            """Format one listing line: dirs get a trailing '/', files get a size.
            A broken symlink or unreadable file shouldn't crash the listing — just
            show the name without a size."""
            if e.is_dir():
                return f"{name}/"
            try:
                return f"{name} ({_format_size(e.stat().st_size)})"
            except OSError:
                return name

        # Cap a recursive listing so an enormous tree can't blow the context. The
        # per-char _truncate still applies on top; this bounds the entry count.
        _MAX_RECURSIVE_ENTRIES = 1000

        def list_files(path: str = ".", recursive: bool = False) -> str:
            base = self._resolve(path)
            if not base.exists():
                return f"error: path does not exist: {base}"
            if base.is_file():
                return str(base)

            if not recursive:
                entries = [
                    _entry_line(e, e.name)
                    for e in sorted(base.iterdir())
                    if e.name not in self._IGNORE
                ]
                return _truncate("\n".join(entries) or "(empty)")

            # Recursive: os.walk with the same _IGNORE pruning as _walk_files, so it
            # skips .git/node_modules/.venv/.opendot exactly like grep/glob. Paths are
            # shown relative to `base` so the tree stays readable.
            entries = []
            truncated = False
            for dirpath, dirnames, filenames in os.walk(base):
                dirnames[:] = sorted(d for d in dirnames if d not in self._IGNORE)
                here = Path(dirpath)
                for name in dirnames:
                    rel = str((here / name).relative_to(base))
                    entries.append(f"{rel}/")
                for fn in sorted(filenames):
                    p = here / fn
                    rel = str(p.relative_to(base))
                    entries.append(_entry_line(p, rel))
                    if len(entries) >= _MAX_RECURSIVE_ENTRIES:
                        truncated = True
                        break
                if truncated:
                    break
            out = "\n".join(entries) or "(empty)"
            if truncated:
                out += f"\n... (listing capped at {_MAX_RECURSIVE_ENTRIES} entries)"
            return _truncate(out)

        def read_file(
            path: str,
            start: int | None = None,
            end: int | None = None,
        ) -> str:
            p = self._resolve(path)
            if not p.exists():
                return f"error: file not found: {p}"
            if p.is_dir():
                return f"error: {p} is a directory (use list_files to see its contents)"
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except Exception as exc:  # noqa: BLE001
                return f"error reading {p}: {exc}"

            # No range requested: return the raw file contents unchanged (the
            # original behavior). Only a requested range slices + numbers lines.
            if start is None and end is None:
                return _truncate(text)

            if start is not None and start < 1:
                return "error: start must be a positive 1-based line number"
            if end is not None and end < 1:
                return "error: end must be a positive 1-based line number"
            if start is not None and end is not None and start > end:
                return "error: start cannot be greater than end"
            lines = text.splitlines()
            file_len = len(lines)
            first = start if start is not None else 1
            last = end if end is not None else file_len
            if first > file_len:
                return f"error: start ({first}) is past the end of the file ({file_len} lines)"
            if last > file_len:
                return f"error: end ({last}) is past the end of the file ({file_len} lines)"
            sliced = lines[first - 1 : last]
            numbered = [f"{first + i}: {line}" for i, line in enumerate(sliced)]
            return _truncate("\n".join(numbered))

        def write_file(path: str, content: str) -> str:
            p = self._resolve(path)
            old = ""
            read_ok = False  # did we actually read the existing file's content?
            if p.exists():
                try:
                    old = p.read_text(encoding="utf-8", errors="replace")
                    read_ok = True
                except OSError:
                    old = ""
            # No-op: the file exists, we read it, and the content is identical (an
            # existing empty file re-written as "" counts too). Skip the snapshot
            # and rewrite so the undo ledger isn't polluted with a phantom change.
            if read_ok and old == content:
                rel = self._rel(p)
                return f"no change to {rel} (content identical)\n"
            # A write outside the working dir isn't covered by the snapshot, so it
            # can't be undone. Confirm first and record it honestly as irreversible,
            # exactly like an escaping shell command. Never claim a lying undo.
            outside = self._is_outside_workspace(p)
            if outside and not self._confirm(
                f"This writes outside the workspace and can't be undone:\n  {p}\nWrite it?"
            ):
                return "skipped: user declined a write outside the workspace"
            if self.rev is not None:
                self.rev.before_action(
                    "write",
                    str(p),
                    reversible=not outside,
                    note="outside the workspace — not undoable" if outside else "",
                )
            try:
                if outside:
                    # Already confirmed + recorded irreversible above; a plain
                    # write is fine (containment doesn't apply outside the workspace).
                    p.parent.mkdir(parents=True, exist_ok=True)
                    p.write_text(content, encoding="utf-8")
                else:
                    # In-workspace: open-time containment so a symlinked target
                    # can't redirect the write outside the snapshot boundary.
                    self._safe_write_within_workspace(p, content)
            except Exception as exc:  # noqa: BLE001
                return f"error writing {p}: {exc}"
            rel = self._rel(p)
            verb = "created" if not old else "updated"
            return f"{verb} {rel}\n" + _unified_diff(old, content, rel)

        def run_shell(command: str, timeout: int | None = None) -> str:
            if timeout is None or timeout <= 0:
                timeout = _shell_timeout()
            # Escape hatch: prefixing a command with OPENDOT_NO_SNAPSHOT=1 runs it
            # WITHOUT snapshotting first. Use it for things you *want* gone and
            # don't want a recoverable copy of (e.g. `shred secrets.txt`) or huge
            # throwaway files not worth capturing. The action is still logged, but
            # marked not-reversible since no snapshot backs it.
            no_snapshot = False
            stripped = command.lstrip()
            for prefix in ("OPENDOT_NO_SNAPSHOT=1 ", "OPENDOT_NO_SNAPSHOT=true "):
                if stripped.startswith(prefix):
                    no_snapshot = True
                    command = stripped[len(prefix) :]
                    break

            # Classify the command: safe/contained vs. irreversible/escaping.
            from opendot.reversibility.classifier import classify

            verdict = classify(command, str(self.workdir))
            if not verdict.reversible:
                ok = self._confirm(
                    f"This command may not be undoable ({verdict.reason}):\n  {command}\nRun it?"
                )
                if not ok:
                    return "skipped: user declined an irreversible command"

            # Snapshot before running (records reversibility from the verdict),
            # unless the user opted out for this one command.
            if self.rev is not None:
                if no_snapshot:
                    self.rev.before_action(
                        "shell",
                        command,
                        snapshot=False,
                        note="snapshot skipped (OPENDOT_NO_SNAPSHOT) — not undoable",
                    )
                else:
                    self.rev.before_action(
                        "shell",
                        command,
                        reversible=verdict.reversible,
                        note=verdict.reason,
                    )
            try:
                proc = subprocess.run(
                    command,
                    shell=True,
                    cwd=str(self.workdir),
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                out = proc.stdout or ""
                err = proc.stderr or ""
                body = out
                if err:
                    body += ("\n[stderr]\n" + err) if body else ("[stderr]\n" + err)
                body += f"\n[exit {proc.returncode}]"
                return _truncate(body)
            except subprocess.TimeoutExpired:
                return f"error: command timed out after {timeout}s"
            except Exception as exc:  # noqa: BLE001
                return f"error running command: {exc}"

        def grep(
            pattern: str,
            path: str = ".",
            max_matches: int = 100,
            ignore_case: bool = False,
            context: int = 0,
        ) -> str:
            """Search file contents for a regex, returning path:line:text matches.

            When ``context`` > 0, also include that many lines before and after each
            match. Context lines use ``path:line-text`` (dash) while matched lines
            keep ``path:line:text`` (colon).
            """
            import re

            base = self._resolve(path)
            if context < 0:
                return "error: context must be a non-negative integer"
            try:
                rx = re.compile(
                    pattern,
                    re.IGNORECASE if ignore_case else 0,
                )
            except re.error as exc:
                return f"error: bad pattern: {exc}"
            hits: list[str] = []
            match_count = 0  # counted across all files, so max_matches is a global cap
            roots = [base] if base.is_file() else self._walk_files(base)
            for f in roots:
                try:
                    lines = f.read_text(encoding="utf-8", errors="ignore").splitlines()
                except OSError:
                    continue
                for i, line in enumerate(lines, 1):
                    if not rx.search(line):
                        continue
                    match_count += 1
                    rel = self._rel(f)
                    first = max(1, i - context)
                    last = min(len(lines), i + context)
                    for j in range(first, last + 1):
                        text = lines[j - 1].strip()[:200]
                        marker = ":" if j == i else "-"
                        hits.append(f"{rel}:{j}{marker}{text}")
                    if match_count >= max_matches:
                        return _truncate("\n".join(hits) + f"\n... (capped at {max_matches})")
            return _truncate("\n".join(hits)) if hits else "no matches"

        def glob(pattern: str) -> str:
            """Find files matching a glob pattern (e.g. '**/*.py')."""
            base = self.workdir
            matches = [
                str(p.relative_to(base))
                for p in base.glob(pattern)
                if p.is_file() and not self._is_ignored(p)
            ]
            return _truncate("\n".join(sorted(matches))) if matches else "no files match"

        def edit(path: str, find: str, replace: str, count: int = 0) -> str:
            """Replace occurrences of `find` with `replace` in a file (surgical edit).

            count=0 replaces all occurrences. Snapshotted, so it's undoable.
            """
            p = self._resolve(path)
            if not p.exists():
                return f"error: file not found: {p}"
            try:
                text = p.read_text(encoding="utf-8")
            except Exception as exc:  # noqa: BLE001
                return f"error reading {p}: {exc}"
            n = text.count(find)
            if n == 0:
                return f"error: `find` string not present in {p} (no change made)"
            # An edit outside the working dir isn't covered by the snapshot, so it
            # can't be undone: confirm first and record it as irreversible.
            outside = self._is_outside_workspace(p)
            if outside and not self._confirm(
                f"This edits a file outside the workspace and can't be undone:\n  {p}\nEdit it?"
            ):
                return "skipped: user declined an edit outside the workspace"
            if self.rev is not None:
                self.rev.before_action(
                    "write",
                    str(p),
                    reversible=not outside,
                    note="outside the workspace — not undoable" if outside else "",
                )
            new = text.replace(find, replace, count if count > 0 else -1)
            p.write_text(new, encoding="utf-8")
            replaced = n if count <= 0 else min(n, count)
            rel = self._rel(p)
            return f"edited {rel} ({replaced} replacement(s))\n" + _unified_diff(text, new, rel)

        def move(src: str, dst: str, overwrite: bool = False) -> str:
            """Move/rename a file or directory (surgical, undoable).

            Errors on a missing src or an existing dst unless overwrite=True —
            mirrors the confirm-first-then-record pattern edit/write_file use for
            paths outside the workspace.
            """
            p_src = self._resolve(src)
            p_dst = self._resolve(dst)
            if not p_src.exists():
                return f"error: file not found: {p_src}"
            if _same_path(p_src, p_dst):
                # Same path either way — nothing to do, and critically must NOT
                # fall into the overwrite=True unlink-then-move path below, which
                # would delete src (since dst IS src) before the move can run.
                return f"no change: {src} and {dst} are the same path\n"
            if p_dst.exists():
                if not overwrite:
                    return (
                        f"error: destination already exists: {p_dst} "
                        "(pass overwrite=true to replace it)"
                    )
                if p_dst.is_dir():
                    # shutil.move()'s default behavior for a directory dst is to
                    # move src INTO it, not replace it — that contradicts what
                    # overwrite=true promises, so reject it explicitly.
                    return f"error: cannot overwrite a directory: {p_dst}"
            # A move touching a path outside the working dir isn't covered by the
            # snapshot on that side, so it can't be fully undone. Confirm first and
            # record it honestly as irreversible, exactly like an escaping write/edit.
            outside = self._is_outside_workspace(p_src) or self._is_outside_workspace(p_dst)
            # _rel_no_resolve, not _rel: a symlink's own path is what was asked
            # to move, not whatever it points at — resolving would misreport
            # the confirm prompt, ledger entry, and return value alike.
            rel_src, rel_dst = self._rel_no_resolve(p_src), self._rel_no_resolve(p_dst)
            if outside and not self._confirm(
                f"This moves a file outside the workspace and can't be undone:\n"
                f"  {rel_src} -> {rel_dst}\nMove it?"
            ):
                return "skipped: user declined a move outside the workspace"
            if self.rev is not None:
                self.rev.before_action(
                    "write",
                    f"{rel_src} -> {rel_dst}",
                    reversible=not outside,
                    note="outside the workspace — not undoable" if outside else "",
                )
            try:
                p_dst.parent.mkdir(parents=True, exist_ok=True)
                # Remove an existing dst explicitly rather than relying on
                # os.rename's implicit overwrite (shutil.move's fallback) — that
                # isn't reliable across platforms (os.rename raises on Windows if
                # dst exists).
                if overwrite and p_dst.exists():
                    p_dst.unlink()
                shutil.move(str(p_src), str(p_dst))
            except Exception as exc:  # noqa: BLE001
                return f"error moving {p_src} to {p_dst}: {exc}"
            return f"{rel_src} -> {rel_dst}"

        def web_fetch(url: str) -> str:
            """Fetch an http(s) URL and return its content as LLM-ready markdown
            (structured text, headings, links, tables) via PyScrappy. Read-only:
            it never mutates the workspace, so it runs without a snapshot, like
            read_file."""
            if not url.lower().startswith(("http://", "https://")):
                return "error: only http:// and https:// URLs are supported"
            import pyscrappy

            try:
                result = pyscrappy.scrape(url)
            except Exception as exc:  # noqa: BLE001 - surface any fetch failure to the model
                return f"error fetching {url}: {type(exc).__name__}: {exc}"
            return _truncate(result.to_markdown()) or "(empty response)"

        return [
            Tool(
                "list_files",
                "List files and directories at a path (relative to the working dir). "
                "Set recursive=true to walk subdirectories in one call.",
                {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to list; defaults to '.'.",
                        },
                        "recursive": {
                            "type": "boolean",
                            "description": (
                                "Walk subdirectories (skipping .git/node_modules/.venv/etc), "
                                "showing paths relative to the listed dir. Default false."
                            ),
                        },
                    },
                },
                list_files,
            ),
            Tool(
                "read_file",
                "Read a text file's contents, optionally by line range (1-based, inclusive).",
                {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "start": {
                            "type": "integer",
                            "description": "First 1-based line to include (defaults to 1).",
                        },
                        "end": {
                            "type": "integer",
                            "description": "Last 1-based line to include (defaults to EOF).",
                        },
                    },
                    "required": ["path"],
                },
                read_file,
            ),
            Tool(
                "write_file",
                "Create or overwrite a file with the given content.",
                {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                    },
                    "required": ["path", "content"],
                },
                write_file,
            ),
            Tool(
                "run_shell",
                "Run a shell command in the working directory (npm, git, build, cp, etc.; "
                "prefer the move tool over `mv`).",
                {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "timeout": {
                            "type": "integer",
                            "description": "Seconds before timeout. Must be a positive integer; values <= 0 fall back to the default (120s, or OPENDOT_SHELL_TIMEOUT if set).",
                        },
                    },
                    "required": ["command"],
                },
                run_shell,
            ),
            Tool(
                "grep",
                "Search file contents for a regular expression. Returns path:line:text matches; with context>0, surrounding lines are included as path:line-text.",
                {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string"},
                        "path": {
                            "type": "string",
                            "description": "Dir or file to search (default '.').",
                        },
                        "max_matches": {"type": "integer"},
                        "ignore_case": {
                            "type": "boolean",
                            "description": "Case-insensitive search (default false).",
                        },
                        "context": {
                            "type": "integer",
                            "description": "Lines of context before and after each match (default 0).",
                        },
                    },
                    "required": ["pattern"],
                },
                grep,
            ),
            Tool(
                "glob",
                "Find files matching a glob pattern, e.g. '**/*.py' or 'src/*.ts'.",
                {
                    "type": "object",
                    "properties": {"pattern": {"type": "string"}},
                    "required": ["pattern"],
                },
                glob,
            ),
            Tool(
                "edit",
                "Make a targeted find-and-replace edit in a file (surgical, undoable). Prefer this over rewriting whole files.",
                {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "find": {
                            "type": "string",
                            "description": "Exact text to find.",
                        },
                        "replace": {"type": "string"},
                        "count": {
                            "type": "integer",
                            "description": "Max replacements; 0 = all (default).",
                        },
                    },
                    "required": ["path", "find", "replace"],
                },
                edit,
            ),
            Tool(
                "move",
                "Move or rename a file or directory (surgical, undoable). Errors if "
                "dst already exists unless overwrite=true.",
                {
                    "type": "object",
                    "properties": {
                        "src": {"type": "string"},
                        "dst": {"type": "string"},
                        "overwrite": {
                            "type": "boolean",
                            "description": "Replace dst if it already exists (default false).",
                        },
                    },
                    "required": ["src", "dst"],
                },
                move,
            ),
            Tool(
                "web_fetch",
                "Fetch an http(s) URL and return its content as markdown (text, headings, links, tables). Read-only.",
                {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "http:// or https:// URL to fetch.",
                        },
                    },
                    "required": ["url"],
                },
                web_fetch,
            ),
        ]

    def specs(self) -> list[dict[str, Any]]:
        specs = [t.spec() for t in self._tools.values()]
        # External MCP tools, namespaced mcp__server__tool.
        for mt in self._mcp_tools.values():
            specs.append(
                {
                    "type": "function",
                    "function": {
                        "name": mt.qualified,
                        "description": f"[MCP:{mt.server}] {mt.description}",
                        "parameters": mt.input_schema,
                    },
                }
            )
        # External Composio tools — the Tool Router keeps this a small, bounded
        # meta-tool set, so we don't need a total-count cap here.
        if not self.read_only:
            from opendot.tools import composio as composio_tools

            specs.extend(composio_tools.build_tool_specs())
        return specs

    def call(self, name: str, args: dict[str, Any]) -> str:
        # Composio tools are external + opaque like MCP: confirm + mark irreversible.
        from opendot.tools import composio as composio_tools

        if composio_tools.is_composio_tool(name):
            reason = "external Composio tool — opendot cannot undo it"
            if not self._confirm(f"Run {name}?  {reason}"):
                return "skipped: user declined an external Composio tool call"
            if self.rev is not None:
                self.rev.before_action("shell", name, reversible=False, note=reason)
            return composio_tools.execute_tool(name, args)

        # MCP tools are external + opaque: opendot can't undo them, so gate every
        # call through confirm and record it as irreversible in the ledger.
        mt = self._mcp_tools.get(name)
        if mt is not None:
            reason = f"external MCP tool ({mt.server}) — opendot cannot undo it"
            if not self._confirm(f"Run {mt.server}.{mt.name}?  {reason}"):
                return "skipped: user declined an external MCP tool call"
            if self.rev is not None:
                self.rev.before_action("shell", name, reversible=False, note=reason)
            return self.mcp.call_tool(mt.server, mt.name, args)

        tool = self._tools.get(name)
        if tool is None:
            return f"error: unknown tool {name!r}"
        try:
            return tool.run(**args)
        except TypeError as exc:
            return f"error: bad arguments for {name}: {exc}"
        except Exception as exc:  # noqa: BLE001
            return f"error in {name}: {exc}"
