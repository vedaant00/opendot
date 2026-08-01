"""Local tools the agent can call — they act on the user's REAL filesystem and
shell, rooted at the working directory. No sandbox (opendot's safety model is
reversibility, not isolation — the snapshot/undo engine lands in M2).

Each tool is a `Tool` (name, JSON-schema spec, and a Python callable). The specs
are the OpenAI/LiteLLM function-calling format. Callables return a string that
becomes the tool result fed back to the model.
"""

from __future__ import annotations

import difflib
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

_MAX_OUTPUT = 30_000  # cap tool output so one huge file can't blow the context


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


def _truncate(text: str) -> str:
    if len(text) > _MAX_OUTPUT:
        return text[:_MAX_OUTPUT] + f"\n... [truncated, {len(text)} chars total]"
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

    def _is_outside_workspace(self, p: Path) -> bool:
        """True if ``p`` is not under the working directory. Writes/edits to such
        paths can't be undone (the snapshot only covers the workspace), so they're
        recorded as irreversible and confirmed first, like escaping shell commands."""
        try:
            p.resolve().relative_to(self.workdir)
            return False
        except ValueError:
            return True

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
        def _format_size(size: int) -> str:
            if size < 1024:
                return f"{size} B"
            if size < 1024 * 1024:
                return f"{size / 1024:.1f} KB"
            if size < 1024 * 1024 * 1024:
                return f"{size / (1024 * 1024):.1f} MB"
            return f"{size / (1024 * 1024 * 1024):.1f} GB"
            
        def list_files(path: str = ".") -> str:
            base = self._resolve(path)
            if not base.exists():
                return f"error: path does not exist: {base}"
            if base.is_file():
                return str(base)
            entries = []
            for e in sorted(base.iterdir()):
                if e.name.startswith(".git") or e.name in {
                    "node_modules",
                    "__pycache__",
                }:
                    continue
                if e.is_dir():
                    entries.append(f"{e.name}/")
                else:
                    size = _format_size(e.stat().st_size)
                    entries.append(f"{e.name} ({size})")
                     
            return _truncate("\n".join(entries) or "(empty)")

        def read_file(path: str) -> str:
            p = self._resolve(path)
            if not p.exists():
                return f"error: file not found: {p}"
            try:
                return _truncate(p.read_text(encoding="utf-8", errors="replace"))
            except Exception as exc:  # noqa: BLE001
                return f"error reading {p}: {exc}"

        def write_file(path: str, content: str) -> str:
            p = self._resolve(path)
            old = ""
            if p.exists():
                try:
                    old = p.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    old = ""
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
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(content, encoding="utf-8")
            except Exception as exc:  # noqa: BLE001
                return f"error writing {p}: {exc}"
            rel = self._rel(p)
            verb = "created" if not old else "updated"
            return f"{verb} {rel}\n" + _unified_diff(old, content, rel)

        def run_shell(command: str, timeout: int = 120) -> str:
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

        def grep(pattern: str, path: str = ".", max_matches: int = 100) -> str:
            """Search file contents for a regex, returning path:line:text matches."""
            import re

            base = self._resolve(path)
            try:
                rx = re.compile(pattern)
            except re.error as exc:
                return f"error: bad pattern: {exc}"
            hits: list[str] = []
            roots = [base] if base.is_file() else self._walk_files(base)
            for f in roots:
                try:
                    for i, line in enumerate(
                        f.read_text(encoding="utf-8", errors="ignore").splitlines(), 1
                    ):
                        if rx.search(line):
                            rel = f.relative_to(self.workdir)
                            hits.append(f"{rel}:{i}:{line.strip()[:200]}")
                            if len(hits) >= max_matches:
                                return _truncate(
                                    "\n".join(hits) + f"\n... (capped at {max_matches})"
                                )
                except OSError:
                    continue
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
            replaced = n if count == 0 else min(n, count)
            rel = self._rel(p)
            return f"edited {rel} ({replaced} replacement(s))\n" + _unified_diff(text, new, rel)

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
                "List files and directories at a path (relative to the working dir).",
                {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to list; defaults to '.'.",
                        }
                    },
                },
                list_files,
            ),
            Tool(
                "read_file",
                "Read a text file's contents.",
                {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
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
                "Run a shell command in the working directory (npm, git, build, mv, cp, etc.).",
                {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "timeout": {
                            "type": "integer",
                            "description": "Seconds before timeout (default 120).",
                        },
                    },
                    "required": ["command"],
                },
                run_shell,
            ),
            Tool(
                "grep",
                "Search file contents for a regular expression. Returns path:line:text matches.",
                {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string"},
                        "path": {
                            "type": "string",
                            "description": "Dir or file to search (default '.').",
                        },
                        "max_matches": {"type": "integer"},
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
