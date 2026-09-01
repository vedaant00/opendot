"""Tests for MCP client integration (config, tool exposure, irreversible gating)."""

import json
import os

import pytest

from opendot.reversibility.engine import Reversibility
from opendot.reversibility.snapshots import IgnoreRules
from opendot.tools.local import Toolbox


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENDOT_HOME", str(tmp_path / "store"))
    yield


def test_load_mcp_config(tmp_path, monkeypatch):
    from opendot.mcp.manager import load_mcp_config

    store = tmp_path / "store"
    store.mkdir(parents=True)
    (store / "mcp.json").write_text(json.dumps({"mcpServers": {"github": {"command": "x"}}}))
    assert "github" in load_mcp_config()


def test_missing_config_is_empty(tmp_path):
    from opendot.mcp.manager import load_mcp_config

    assert load_mcp_config() == {}


def test_save_mcp_config_is_owner_only(tmp_path):
    from opendot.mcp.manager import load_mcp_config, save_mcp_config

    store = tmp_path / "store"
    store.mkdir(parents=True)
    path = store / "mcp.json"
    path.write_text(json.dumps({"mcpServers": {}}), encoding="utf-8")
    path.chmod(0o644)

    save_mcp_config(
        {
            "supabase": {
                "url": "https://mcp.supabase.com/mcp",
                "headers": {"Authorization": "Bearer tok123"},
            }
        }
    )

    # POSIX permission bits are meaningless on Windows; the mode check is the
    # security guarantee, so gate just that assertion, not the whole test.
    if os.name == "posix":
        assert (path.stat().st_mode & 0o777) == 0o600
    assert load_mcp_config()["supabase"]["headers"]["Authorization"] == "Bearer tok123"
    assert not any(p.name.endswith(".tmp") for p in store.iterdir())


class _FakeMCP:
    """Stand-in MCPManager: exposes one tool, records calls."""

    def __init__(self):
        from opendot.mcp.manager import MCPTool

        self.tools = [
            MCPTool(
                server="stub",
                name="greet",
                description="say hi",
                input_schema={
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                },
            )
        ]
        self.connected = ["stub"]
        self.errors = {}
        self.calls = []

    def call_tool(self, server, name, args):
        self.calls.append((server, name, args))
        return "hi!"


def _tb(tmp_path, confirm):
    rev = Reversibility(workdir=str(tmp_path), rules=IgnoreRules())
    return Toolbox(str(tmp_path), reversibility=rev, confirm=confirm, mcp_manager=_FakeMCP()), rev


def test_mcp_tool_appears_in_specs(tmp_path):
    tb, _ = _tb(tmp_path, lambda p: True)
    names = {s["function"]["name"] for s in tb.specs()}
    assert "mcp__stub__greet" in names


def test_toolbox_forget_mcp_server_drops_tools(tmp_path):
    """After a server is removed, its tools stop appearing in specs and can't be
    called this session (in-session removal, not just config)."""
    tb, _ = _tb(tmp_path, lambda p: True)
    assert "mcp__stub__greet" in {s["function"]["name"] for s in tb.specs()}

    tb.forget_mcp_server("stub")
    assert "mcp__stub__greet" not in {s["function"]["name"] for s in tb.specs()}
    assert "unknown tool" in tb.call("mcp__stub__greet", {"name": "x"}).lower()


def test_manager_forget_server_purges_in_memory_state():
    """forget_server drops the server from connected/errors/tools/_sessions so it
    stops showing and its tools stop being offered until next launch."""
    from opendot.mcp.manager import MCPManager, MCPTool

    mgr = MCPManager({})
    mgr.connected = ["stub", "other"]
    mgr.errors = {"stub": "boom"}
    mgr._sessions = {"stub": object()}
    mgr.tools = [MCPTool("stub", "greet", "", {}), MCPTool("other", "ping", "", {})]

    mgr.forget_server("stub")

    assert mgr.connected == ["other"]
    assert "stub" not in mgr.errors
    assert "stub" not in mgr._sessions
    assert [t.server for t in mgr.tools] == ["other"]


def test_mcp_call_declined_by_confirm(tmp_path):
    tb, rev = _tb(tmp_path, lambda p: False)
    out = tb.call("mcp__stub__greet", {"name": "x"})
    assert "declined" in out
    assert tb.mcp.calls == []  # never actually called
    assert rev.history() == []  # nothing logged


def test_mcp_call_approved_runs_and_logs_irreversible(tmp_path):
    tb, rev = _tb(tmp_path, lambda p: True)
    out = tb.call("mcp__stub__greet", {"name": "world"})
    assert out == "hi!"
    assert tb.mcp.calls == [("stub", "greet", {"name": "world"})]
    entry = rev.history()[-1]
    assert entry.reversible is False  # MCP calls are marked irreversible
    assert "MCP" in entry.note


def test_mcp_hidden_from_readonly_explorer(tmp_path):
    rev = Reversibility(workdir=str(tmp_path), rules=IgnoreRules())
    tb = Toolbox(str(tmp_path), reversibility=rev, read_only=True, mcp_manager=_FakeMCP())
    names = {s["function"]["name"] for s in tb.specs()}
    assert "mcp__stub__greet" not in names  # explorers can't call external tools


def test_config_add_list_remove(tmp_path):
    from opendot.mcp.manager import add_mcp_server, load_mcp_config, remove_mcp_server

    add_mcp_server("stdio1", {"command": "some-cmd", "args": ["-y", "pkg"], "env": {"K": "v"}})
    add_mcp_server("remote1", {"url": "https://example.com/mcp"})

    cfg = load_mcp_config()
    assert cfg["stdio1"]["command"] == "some-cmd"
    assert cfg["stdio1"]["args"] == ["-y", "pkg"]  # command flags preserved
    assert cfg["stdio1"]["env"] == {"K": "v"}
    assert cfg["remote1"]["url"] == "https://example.com/mcp"

    assert remove_mcp_server("stdio1") is True
    assert "stdio1" not in load_mcp_config()
    assert remove_mcp_server("nope") is False


def test_mcp_add_cli_parses_flags_vs_command(tmp_path, monkeypatch):
    """`opendot mcp add NAME --env K=V -- cmd -y pkg` must keep -y in the command,
    not treat it as an opendot flag (the bug that used to launch the TUI)."""
    import sys

    from opendot import cli
    from opendot.mcp.manager import load_mcp_config

    monkeypatch.setattr(
        sys,
        "argv",
        ["opendot", "mcp", "add", "s", "--env", "K=v", "--", "cmd", "-y", "pkg"],
    )
    cli.main()
    cfg = load_mcp_config()
    assert cfg["s"] == {"command": "cmd", "args": ["-y", "pkg"], "env": {"K": "v"}}


def test_mcp_add_remote_with_header(tmp_path, monkeypatch):
    """`opendot mcp add NAME --url <u> --header 'Authorization=Bearer x'` stores
    the header so authenticated remote servers (e.g. Supabase PAT) can connect."""
    import sys

    from opendot import cli
    from opendot.mcp.manager import load_mcp_config

    monkeypatch.setenv("OPENDOT_HOME", str(tmp_path / "store"))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "opendot",
            "mcp",
            "add",
            "supabase",
            "--url",
            "https://mcp.supabase.com/mcp?project_ref=abc",
            "--header",
            "Authorization=Bearer tok123",
        ],
    )
    cli.main()
    cfg = load_mcp_config()
    assert cfg["supabase"] == {
        "url": "https://mcp.supabase.com/mcp?project_ref=abc",
        "headers": {"Authorization": "Bearer tok123"},
    }


def test_mcp_test_cli_reports_connected_tools(tmp_path, monkeypatch, capsys):
    import sys

    from opendot import cli, mcp
    from opendot.mcp.manager import MCPTool, add_mcp_server

    add_mcp_server("demo", {"command": "ignored"})

    class FakeManager:
        def __init__(self, config):
            assert config == {"demo": {"command": "ignored"}}
            self.connected = ["demo"]
            self.errors = {}
            self.tools = [MCPTool("demo", "ping", "", {})]

        def start(self):
            pass

        def shutdown(self):
            pass

    monkeypatch.setattr(mcp, "MCPManager", FakeManager)
    monkeypatch.setattr(sys, "argv", ["opendot", "mcp", "test", "demo"])
    cli.main()

    output = capsys.readouterr().out
    assert "connected" in output
    assert "1 tool: ping" in output


def test_mcp_test_cli_reports_connection_error(tmp_path, monkeypatch, capsys):
    import sys

    from opendot import cli, mcp
    from opendot.mcp.manager import add_mcp_server

    add_mcp_server("broken", {"command": "ignored"})

    class FakeManager:
        def __init__(self, config):
            self.connected = []
            self.errors = {"broken": "ConnectionError: refused"}
            self.tools = []

        def start(self):
            pass

        def shutdown(self):
            pass

    monkeypatch.setattr(mcp, "MCPManager", FakeManager)
    monkeypatch.setattr(sys, "argv", ["opendot", "mcp", "test", "broken"])
    cli.main()

    output = capsys.readouterr().out
    assert "connection failed" in output
    assert "ConnectionError: refused" in output
