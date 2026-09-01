"""MCP client manager — connects opendot to external MCP servers.

MCP's client API is async and connection-oriented (a session must stay open for
its lifetime, via nested async context managers). opendot's tool layer is sync.
So this manager runs a dedicated **background asyncio loop on its own thread**,
opens every server session there and keeps them alive, and exposes plain sync
methods (`list_tools`, `call_tool`) that bridge onto that loop. All the async /
context-manager complexity is contained here.

Config lives at ``~/.opendot/mcp.json`` (Claude-Desktop-style):

    {
      "mcpServers": {
        "pyscrappy": { "command": "pyscrappy-mcp" },
        "github":    { "command": "npx", "args": ["-y", "@modelcontextprotocol/server-github"],
                       "env": {"GITHUB_TOKEN": "..."} },
        "remote":    { "url": "https://example.com/mcp" },         // http/sse
        "supabase":  { "url": "https://mcp.supabase.com/mcp?project_ref=abc",
                       "headers": {"Authorization": "Bearer <PAT>"} },  // static token
        "linear":    { "url": "https://mcp.linear.app/mcp", "auth": "oauth" }  // browser OAuth
      }
    }

Tokens for ``"auth": "oauth"`` servers live outside this file (see
``opendot.mcp.oauth``); the config only records that the server uses OAuth.

MCP tools are OPAQUE — opendot can't know if a call is reversible. So callers
treat every MCP tool as irreversible (confirm + mark ✗ in the ledger).
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import threading
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from opendot.reversibility.snapshots import store_root


def _config_path() -> Path:
    return store_root() / "mcp.json"


def load_mcp_config() -> dict[str, dict]:
    """Read ~/.opendot/mcp.json → {server_name: server_spec}. Empty if absent."""
    path = _config_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}
    return data.get("mcpServers", data) or {}


def save_mcp_config(servers: dict[str, dict]) -> None:
    """Write the full server map back to ~/.opendot/mcp.json."""
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # A static config can carry Authorization headers, so it must never exist
    # group/world-readable — not even briefly. Create a unique temp file that is
    # 0600 *from the first byte* (mkstemp uses O_EXCL + mode 0600, so no collision
    # and no world-readable window), then atomically rename it in. os.replace
    # preserves the temp file's mode, so no follow-up chmod is needed (and none is
    # wanted — an unguarded chmod can fail on permission-mapped mounts the creation
    # mode already handles).
    payload = json.dumps({"mcpServers": servers}, indent=2).encode("utf-8")
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    replaced = False
    try:
        # If fdopen itself fails, the raw fd would leak — close it and re-raise.
        try:
            f = os.fdopen(fd, "wb")
        except OSError:
            os.close(fd)
            raise
        with f:
            f.write(payload)
        os.replace(tmp, path)
        replaced = True
    finally:
        # Only clean up on failure. After a successful replace the temp name is
        # freed, and a blind unlink could race and delete an unrelated file that
        # took that name — so key the cleanup on `replaced`, not tmp.exists().
        if not replaced:
            tmp.unlink(missing_ok=True)


def add_mcp_server(name: str, spec: dict) -> None:
    """Add/replace one server in the config."""
    servers = load_mcp_config()
    servers[name] = spec
    save_mcp_config(servers)


def remove_mcp_server(name: str) -> bool:
    """Remove a server; returns True if it existed."""
    servers = load_mcp_config()
    if name not in servers:
        return False
    del servers[name]
    save_mcp_config(servers)
    # Forget any cached OAuth tokens/registration for this name (a no-op for
    # servers that never used OAuth).
    try:
        from opendot.mcp.oauth import clear_tokens

        clear_tokens(name)
    except Exception:  # noqa: BLE001
        pass
    return True


@dataclass
class AuthorizeResult:
    ok: bool
    tool_count: int = 0
    error: str | None = None


def authorize_oauth_server(name: str, spec: dict) -> AuthorizeResult:
    """Connect one OAuth server now, driving the browser flow, to prove it works.

    Called at add-time (from the TUI) so the user authorizes in-browser immediately
    and tokens get cached — no restart needed to reach a working state. Spins up a
    throwaway single-server manager, connects (which opens the browser on first
    auth), and reports the tool count or the failure. Blocking; run off the UI thread.
    """
    mgr = MCPManager({name: spec})
    try:
        # The OAuth browser dance waits on a human, so allow well past start's
        # default settle window (matches the provider's 300s flow timeout).
        mgr.start(settle_timeout=300.0)
        if name in mgr.connected:
            return AuthorizeResult(
                ok=True, tool_count=sum(1 for t in mgr.tools if t.server == name)
            )
        return AuthorizeResult(ok=False, error=mgr.errors.get(name, "connection failed"))
    finally:
        mgr.shutdown()


@dataclass
class MCPTool:
    server: str
    name: str  # bare tool name on the server
    description: str
    input_schema: dict[str, Any]

    @property
    def qualified(self) -> str:
        # namespaced so two servers can't collide; also signals it's an MCP tool
        return f"mcp__{self.server}__{self.name}"


class MCPManager:
    """Owns MCP server connections on a background loop; sync-facing API."""

    def __init__(self, config: dict[str, dict] | None = None) -> None:
        self.config = config if config is not None else load_mcp_config()
        self.tools: list[MCPTool] = []
        self.connected: list[str] = []
        self.errors: dict[str, str] = {}
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._sessions: dict[str, Any] = {}
        self._stack = None  # AsyncExitStack holding all open connections
        self._callbacks: list[Any] = []  # loopback OAuth callback servers to close

    # -- lifecycle --
    def start(self, settle_timeout: float = 30.0) -> None:
        """Spin up the background loop and connect all configured servers.

        ``settle_timeout`` bounds how long ``start`` blocks waiting for connections
        to settle. The default (30s) suits normal launch; the OAuth authorize flow
        passes a larger value because it waits on a human clicking through a browser
        consent screen.
        """
        if not self.config or self._thread is not None:
            return
        ready = threading.Event()

        def run() -> None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.create_task(self._connect_all(ready))
            self._loop.run_forever()

        self._thread = threading.Thread(target=run, daemon=True, name="opendot-mcp")
        self._thread.start()
        ready.wait(timeout=settle_timeout)  # let connections settle (best-effort)

    async def _connect_all(self, ready: threading.Event) -> None:
        from contextlib import AsyncExitStack

        from mcp import ClientSession

        self._stack = AsyncExitStack()
        for name, spec in self.config.items():
            try:
                read, write = await self._open_transport({**spec, "_name": name})
                session = await self._stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                self._sessions[name] = session
                self.connected.append(name)
                resp = await session.list_tools()
                for t in resp.tools:
                    self.tools.append(
                        MCPTool(
                            server=name,
                            name=t.name,
                            description=t.description or "",
                            input_schema=t.inputSchema or {"type": "object", "properties": {}},
                        )
                    )
            except Exception as exc:  # noqa: BLE001 - a bad server must not kill the rest
                self.errors[name] = f"{type(exc).__name__}: {exc}"
        # OAuth callback servers have done their job once every server is connected.
        # shutdown() also sweeps any survivors, in case it races this coroutine
        # (e.g. start() timed out and the app exits before we get here).
        self._close_callbacks()
        ready.set()

    def _close_callbacks(self) -> None:
        """Close and forget any open loopback OAuth callback servers (idempotent)."""
        while self._callbacks:
            cb = self._callbacks.pop()
            try:
                cb.close()
            except Exception:  # noqa: BLE001
                pass

    async def _open_transport(self, spec: dict):
        """Return (read, write) streams for a stdio or http/sse server."""
        if spec.get("url"):
            url = spec["url"]
            headers = spec.get("headers") or None  # e.g. {"Authorization": "Bearer ..."}
            auth = None
            if spec.get("auth") == "oauth":
                # Browser-OAuth server: attach an OAuthClientProvider (an httpx.Auth)
                # that drives the authorize/refresh flow. Static headers are ignored
                # for OAuth servers — the provider supplies Authorization itself.
                from opendot.mcp.oauth import build_oauth_provider

                auth, callback = build_oauth_provider(url, spec["_name"])
                self._callbacks.append(callback)
                headers = None
            if url.rstrip("/").endswith("/sse"):
                from mcp.client.sse import sse_client

                return (
                    await self._stack.enter_async_context(
                        sse_client(url, headers=headers, auth=auth)
                    )
                )[:2]
            from mcp.client.streamable_http import streamablehttp_client

            return (
                await self._stack.enter_async_context(
                    streamablehttp_client(url, headers=headers, auth=auth)
                )
            )[:2]
        # stdio
        from mcp import StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(
            command=spec["command"],
            args=spec.get("args", []),
            env=spec.get("env") or None,
        )
        return await self._stack.enter_async_context(stdio_client(params))

    # -- sync bridge --
    def _run(self, coro, timeout: float = 120.0):
        if self._loop is None:
            raise RuntimeError("MCP manager not started")
        fut: Future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result(timeout=timeout)

    def forget_server(self, name: str) -> None:
        """Drop a server from the in-memory view (connected list, errors, tools)
        so it stops appearing and its tools stop being offered this session.

        The underlying session/subprocess is left for the normal shutdown path;
        this just makes an in-session removal immediately consistent instead of
        lingering until the next launch.
        """
        if name in self.connected:
            self.connected.remove(name)
        self.errors.pop(name, None)
        self._sessions.pop(name, None)
        self.tools = [t for t in self.tools if t.server != name]

    def call_tool(self, server: str, name: str, args: dict[str, Any]) -> str:
        """Call an MCP tool synchronously; return a text result."""
        session = self._sessions.get(server)
        if session is None:
            return f"error: MCP server {server!r} not connected"

        async def _call():
            res = await session.call_tool(name, args or {})
            # prefer structured content, else concatenate text blocks
            structured = getattr(res, "structuredContent", None)
            if structured:
                return json.dumps(structured, default=str)
            parts = []
            for block in getattr(res, "content", []) or []:
                parts.append(getattr(block, "text", "") or str(block))
            return "\n".join(parts) or "(no content)"

        try:
            return self._run(_call())
        except Exception as exc:  # noqa: BLE001
            return f"error calling {server}.{name}: {exc}"

    def shutdown(self) -> None:
        # Close any loopback OAuth callback servers first — they may still be open
        # if start() timed out mid-authorization and we never reached _connect_all's
        # own cleanup. Idempotent, so double-closing after a clean connect is fine.
        self._close_callbacks()
        if self._loop and self._loop.is_running():

            async def _close():
                if self._stack:
                    await self._stack.aclose()

            try:
                self._run(_close(), timeout=10)
            except Exception:  # noqa: BLE001
                pass
            self._loop.call_soon_threadsafe(self._loop.stop)
