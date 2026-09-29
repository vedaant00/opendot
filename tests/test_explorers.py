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


def _explorer_task_names():
    return sorted(
        t.get_coro().__qualname__
        for t in asyncio.all_tasks()
        if t is not asyncio.current_task()
        and not t.done()
        and t.get_coro() is not None
        and "run_explorers" in t.get_coro().__qualname__
    )


@pytest.mark.asyncio
async def test_explorer_constructor_failure_completes_with_finding(tmp_path, monkeypatch):
    """A lane whose Agent(...) raises must not deadlock the fan-out (issue #166,
    bug A): the failure surfaces as that lane's finding and iteration ends."""
    from opendot.agent import explorers

    class BoomAgent:
        def __init__(self, *a, **k):
            raise RuntimeError("boom during construction")

    monkeypatch.setattr("opendot.agent.loop.Agent", BoomAgent)

    async def collect():
        out = []
        async for ev in explorers.run_explorers(["a", "b"], model="fake", workdir=str(tmp_path)):
            out.append(ev)
        return out

    events = await asyncio.wait_for(collect(), timeout=10)

    dones = [e for e in events if e.type == "explorer_done"]
    assert len(dones) == 2
    assert all("(explorer failed: boom during construction)" in e.text for e in dones)
    merged = [e for e in events if e.type == "tool_end" and e.tool == "spawn_explorers"]
    assert merged and "boom during construction" in merged[0].result


@pytest.mark.asyncio
async def test_explorer_run_failure_midstream_reports_partial_findings(tmp_path, monkeypatch):
    """A lane whose run() raises mid-stream keeps its partial text plus the
    failure note, and the fan-out still completes."""
    from opendot.agent import explorers
    from opendot.agent.events import Event

    class FailRunAgent:
        def __init__(self, *a, **k):
            pass

        async def run(self, task):
            yield Event("text", text="partial-")
            raise RuntimeError("mid-stream boom")

    monkeypatch.setattr("opendot.agent.loop.Agent", FailRunAgent)

    events = []
    async for ev in explorers.run_explorers(["a"], model="fake", workdir=str(tmp_path)):
        events.append(ev)

    dones = [e for e in events if e.type == "explorer_done"]
    assert len(dones) == 1
    assert "partial-" in dones[0].text
    assert "(explorer failed: mid-stream boom)" in dones[0].text


@pytest.mark.asyncio
async def test_explorer_early_close_leaves_no_running_lanes(tmp_path, monkeypatch):
    """Stopping iteration early and calling aclose() must cancel the runner and
    all lanes instead of leaking them (issue #166, bug B)."""
    from opendot.agent import explorers
    from opendot.agent.events import Event

    class SlowAgent:
        def __init__(self, *a, **k):
            pass

        async def run(self, task):
            await asyncio.sleep(30)
            yield Event("text", text="too late")

    monkeypatch.setattr("opendot.agent.loop.Agent", SlowAgent)

    agen = explorers.run_explorers(["a", "b"], model="fake", workdir=str(tmp_path))
    async for _ev in agen:
        break  # consumer stops early, e.g. user pressed Esc
    await agen.aclose()
    await asyncio.sleep(0.2)

    assert _explorer_task_names() == []


@pytest.mark.asyncio
async def test_explorer_cancellation_propagates_and_stops_lanes(tmp_path, monkeypatch):
    """Cancelling the consumer must surface CancelledError (not an ordinary
    error event) and still leave no running lanes behind."""
    from opendot.agent import explorers
    from opendot.agent.events import Event

    started = asyncio.Event()

    class SlowAgent:
        def __init__(self, *a, **k):
            pass

        async def run(self, task):
            started.set()
            await asyncio.sleep(30)
            yield Event("text", text="too late")

    monkeypatch.setattr("opendot.agent.loop.Agent", SlowAgent)

    async def consume_forever():
        async for _ev in explorers.run_explorers(["a", "b"], model="fake", workdir=str(tmp_path)):
            pass

    consumer = asyncio.create_task(consume_forever())
    await asyncio.wait_for(started.wait(), timeout=10)
    consumer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await consumer
    await asyncio.sleep(0.2)

    assert _explorer_task_names() == []
