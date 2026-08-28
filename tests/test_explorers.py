"""Tests for parallel read-only explorer subagents."""

import asyncio

import pytest

from opendot.tools.local import Toolbox


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENDOT_HOME", str(tmp_path / "store"))
    yield


def test_readonly_toolbox_excludes_mutating_tools(tmp_path):
    tb = Toolbox(str(tmp_path), read_only=True)
    names = {s["function"]["name"] for s in tb.specs()}
    # read-only: observers only
    assert {"grep", "glob", "read_file", "list_files"} <= names
    # no mutators
    assert not ({"write_file", "edit", "run_shell"} & names)


def test_readonly_toolbox_refuses_write_even_if_called(tmp_path):
    # Even if something tries to call write_file on a read-only box, it's not there.
    tb = Toolbox(str(tmp_path), read_only=True)
    out = tb.call("write_file", {"path": "x.txt", "content": "no"})
    assert "unknown tool" in out
    assert not (tmp_path / "x.txt").exists()


@pytest.mark.asyncio
async def test_explorers_run_parallel_readonly_and_return_findings(tmp_path, monkeypatch):
    """Explorers run concurrently, never write, and their findings come back."""
    from opendot.agent import explorers
    from opendot.agent.events import Event

    order: list[str] = []

    # Replace the real Agent with a fake read-only agent that "explores".
    class FakeAgent:
        def __init__(self, config=None, confirm=None, read_only=False):
            self.config = config
            from opendot.tools.local import Toolbox

            self.toolbox = Toolbox(config.workdir, read_only=read_only)

        async def run(self, task):
            order.append(f"start:{task}")
            yield Event("tool_start", tool="grep", args={"pattern": task})
            await asyncio.sleep(0.05)  # let lanes interleave
            yield Event("text", text=f"found stuff for {task}")
            yield Event("final")
            order.append(f"end:{task}")

    monkeypatch.setattr("opendot.agent.loop.Agent", FakeAgent)

    events = []
    async for ev in explorers.run_explorers(
        ["task A", "task B", "task C"], model="fake", workdir=str(tmp_path)
    ):
        events.append(ev)

    types = [e.type for e in events]
    assert types.count("explorer_start") == 3
    assert types.count("explorer_done") == 3
    # merged findings returned as a tool_end
    merged = [e for e in events if e.type == "tool_end" and e.tool == "spawn_explorers"]
    assert merged and "task A" in merged[0].result and "task C" in merged[0].result

    # parallelism: all three started before all three ended (interleaved),
    # i.e. not strictly start:A,end:A,start:B,end:B,...
    starts = [o for o in order if o.startswith("start")]
    assert len(starts) == 3
    # at least one 'start' happens after the first 'start' but before its 'end'
    assert order[0].startswith("start") and order[1].startswith("start")

    # nothing was written anywhere (read-only)
    assert not any(tmp_path.iterdir()) or all(p.name == "store" for p in tmp_path.iterdir())
