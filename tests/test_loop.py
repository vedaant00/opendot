"""Agent streaming-loop tests.

Covers the streaming path's choice-less chunk bug: some providers emit a final
usage-only chunk that has no `choices` attribute at all. Direct attribute access
would raise `AttributeError` and be swallowed by the broad `run()` exception
handler as a "model call failed" failure.

Also covers retrying transient provider errors (rate-limit/5xx/timeout) with
backoff instead of aborting the whole turn on the first blip.
"""

import asyncio
import types

import litellm as real_litellm

from opendot.agent.config import AgentConfig
from opendot.agent.events import Event
from opendot.agent.loop import Agent, _Assembled


class _Usage:
    prompt_tokens = 1
    completion_tokens = 1
    total_tokens = 2


class _Choice:
    def __init__(self, content: str | None = None):
        self.delta = types.SimpleNamespace()
        self.delta.content = content
        self.delta.reasoning_content = None
        self.delta.tool_calls = None


def _chunk_no_choices(with_usage: bool = False):
    """A chunk that lacks the `choices` attribute entirely (e.g. usage-only)."""
    c = types.SimpleNamespace()
    c.usage = _Usage() if with_usage else None
    return c


def _chunk_with_choices(content: str | None = None, with_usage: bool = False):
    c = types.SimpleNamespace()
    c.choices = [_Choice(content)]
    c.usage = _Usage() if with_usage else None
    return c


def _bare_agent(max_retries: int = 3):
    a = Agent.__new__(Agent)
    a.config = AgentConfig(model="gpt-4o", workdir="/tmp", max_retries=max_retries)
    a.usage = types.SimpleNamespace(add_response=lambda *a, **k: None, total_tokens=0, cost_usd=0.0)
    a.messages = [{"role": "system", "content": "sys"}]
    a.toolbox = types.SimpleNamespace(_confirm=None, specs=lambda: [], call=lambda *a, **k: "")
    a.explorers_enabled = False
    return a


class _FlakyLiteLLM(types.SimpleNamespace):
    """Fake litellm module: `acompletion` raises for the first `fail_times`
    calls, then succeeds. Reuses the real exception classes so isinstance
    checks in the retry helper behave exactly as with the real litellm.
    """

    RateLimitError = real_litellm.RateLimitError
    APIConnectionError = real_litellm.APIConnectionError
    Timeout = real_litellm.Timeout
    InternalServerError = real_litellm.InternalServerError
    ServiceUnavailableError = real_litellm.ServiceUnavailableError
    AuthenticationError = real_litellm.AuthenticationError
    BadRequestError = real_litellm.BadRequestError

    def __init__(self, fail_times, make_exc, **kw):
        super().__init__(**kw)
        self.fail_times = fail_times
        self.make_exc = make_exc
        self.calls = 0

    async def acompletion(self, **kw):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self.make_exc()

        async def gen():
            yield _chunk_with_choices("done", with_usage=True)

        return gen()


def _rate_limit_error():
    return real_litellm.RateLimitError(
        message="rate limited", llm_provider="openai", model="gpt-4o"
    )


def _auth_error():
    return real_litellm.AuthenticationError(
        message="bad key", llm_provider="openai", model="gpt-4o"
    )


def test_stream_turn_skips_choiceless_chunk():
    """A chunk with no `choices` attribute is skipped; the turn completes."""

    class _FakeLiteLLM(types.SimpleNamespace):
        async def acompletion(self, **kw):
            async def gen():
                yield _chunk_no_choices(with_usage=True)  # final usage-only chunk
                yield _chunk_with_choices("hello")  # actual content chunk

            return gen()

    a = _bare_agent()

    async def run():
        events = []
        async for ev in a._stream_turn(_FakeLiteLLM(), []):
            events.append(ev)
        return events

    events = asyncio.run(run())
    assert any(isinstance(ev, _Assembled) for ev in events)
    text_events = [ev for ev in events if isinstance(ev, Event) and ev.type == "text"]
    assert len(text_events) == 1
    assert text_events[0].text == "hello"


def test_stream_turn_choiceless_chunk_not_fatal():
    """A choice-less chunk yields no error event from the streaming path."""

    class _FakeLiteLLM(types.SimpleNamespace):
        async def acompletion(self, **kw):
            async def gen():
                yield _chunk_no_choices(with_usage=True)
                yield _chunk_with_choices("ok")

            return gen()

    a = _bare_agent()

    async def run():
        events = []
        async for ev in a._stream_turn(_FakeLiteLLM(), []):
            events.append(ev)
        return events

    events = asyncio.run(run())
    assert not any(isinstance(ev, Event) and ev.type == "error" for ev in events)


def _run_agent(a, litellm, monkeypatch, user_message="hi", sleep_log=None):
    import sys

    from opendot.agent import loop as loop_module

    monkeypatch.setitem(sys.modules, "litellm", litellm)
    # Don't actually wait through backoff sleeps in tests. `asyncio` is a
    # single shared module object, so capture the real sleep before patching
    # it — otherwise the replacement would call itself recursively.
    real_sleep = loop_module.asyncio.sleep

    async def fake_sleep(seconds, *a, **k):
        if sleep_log is not None:
            sleep_log.append(seconds)
        await real_sleep(0)

    monkeypatch.setattr(loop_module.asyncio, "sleep", fake_sleep)

    async def go():
        events = []
        async for ev in a.run(user_message):
            events.append(ev)
        return events

    return asyncio.run(go())


def test_run_survives_transient_failures_then_succeeds(monkeypatch):
    """Two rate-limit blips followed by a success: no error event, turn completes."""
    a = _bare_agent(max_retries=3)
    fake = _FlakyLiteLLM(fail_times=2, make_exc=_rate_limit_error)

    events = _run_agent(a, fake, monkeypatch)

    assert fake.calls == 3
    assert not any(ev.type == "error" for ev in events)
    assert any(ev.type == "final" for ev in events)
    text_events = [ev for ev in events if ev.type == "text"]
    assert text_events and text_events[-1].text == "done"


def test_run_does_not_retry_fatal_errors(monkeypatch):
    """An authentication error fails immediately, with no retry attempts."""
    a = _bare_agent(max_retries=3)
    fake = _FlakyLiteLLM(fail_times=99, make_exc=_auth_error)

    events = _run_agent(a, fake, monkeypatch)

    assert fake.calls == 1
    assert len(events) == 1
    assert events[0].type == "error"
    assert "bad key" in events[0].text


def test_run_exhausts_retries_then_yields_terminal_error(monkeypatch):
    """More transient failures than max_retries allows: the turn ultimately fails."""
    a = _bare_agent(max_retries=2)
    fake = _FlakyLiteLLM(fail_times=99, make_exc=_rate_limit_error)

    events = _run_agent(a, fake, monkeypatch)

    # 1 initial attempt + 2 retries = 3 calls total.
    assert fake.calls == 3
    assert len(events) == 1
    assert events[0].type == "error"
    assert "rate limited" in events[0].text


def test_run_zero_max_retries_fails_on_first_transient_error(monkeypatch):
    """max_retries=0 means no retrying at all — the first blip is terminal."""
    a = _bare_agent(max_retries=0)
    fake = _FlakyLiteLLM(fail_times=99, make_exc=_rate_limit_error)

    events = _run_agent(a, fake, monkeypatch)

    assert fake.calls == 1
    assert len(events) == 1
    assert events[0].type == "error"


class _DropsAfterUsageChunk(types.SimpleNamespace):
    """Fake litellm: first call's stream yields a usage-only chunk then drops
    with a transient error; second call succeeds. Models a provider that
    reports usage before the connection is lost mid-turn.
    """

    RateLimitError = real_litellm.RateLimitError
    APIConnectionError = real_litellm.APIConnectionError
    Timeout = real_litellm.Timeout
    InternalServerError = real_litellm.InternalServerError
    ServiceUnavailableError = real_litellm.ServiceUnavailableError
    AuthenticationError = real_litellm.AuthenticationError
    BadRequestError = real_litellm.BadRequestError

    def __init__(self, **kw):
        super().__init__(**kw)
        self.calls = 0

    async def acompletion(self, **kw):
        self.calls += 1
        if self.calls == 1:

            async def gen_fail():
                yield _chunk_no_choices(with_usage=True)
                raise real_litellm.Timeout(message="dropped", model="gpt-4o", llm_provider="openai")

            return gen_fail()

        async def gen_ok():
            yield _chunk_with_choices("done", with_usage=True)

        return gen_ok()


def test_retry_does_not_double_count_usage(monkeypatch):
    """A transient error after usage arrives but before the stream finishes
    must not record that usage — the retried, successful attempt records it
    once, not the aborted attempt too."""
    a = _bare_agent(max_retries=1)
    usage_calls = []
    a.usage.add_response = lambda *a2, **k: usage_calls.append(1)
    fake = _DropsAfterUsageChunk()

    events = _run_agent(a, fake, monkeypatch)

    assert fake.calls == 2
    assert not any(ev.type == "error" for ev in events)
    assert usage_calls == [1]


def test_backoff_delay_is_capped(monkeypatch):
    """Exponential backoff must not grow unbounded as retries increase."""
    from opendot.agent import loop as loop_module

    a = _bare_agent(max_retries=10)
    fake = _FlakyLiteLLM(fail_times=99, make_exc=_rate_limit_error)
    sleep_log: list[float] = []

    _run_agent(a, fake, monkeypatch, sleep_log=sleep_log)

    assert fake.calls == 11  # 1 initial attempt + 10 retries
    assert len(sleep_log) == 10
    assert max(sleep_log) <= loop_module._MAX_BACKOFF_SECONDS
    # Naive 2**attempt would reach 512s by the 10th retry; confirm it's capped.
    assert 512 not in sleep_log


# --- budget enforcement (#123) ---


def _agent_with_real_usage(max_usd=None, max_tokens=None):
    from opendot.agent.usage import Usage

    a = Agent.__new__(Agent)
    a.config = AgentConfig(
        model="gpt-4o", workdir="/tmp", max_retries=0, max_usd=max_usd, max_tokens=max_tokens
    )
    a.usage = Usage()
    a.messages = [{"role": "system", "content": "sys"}]
    a.toolbox = types.SimpleNamespace(_confirm=None, specs=lambda: [], call=lambda *a, **k: "")
    a.explorers_enabled = False
    return a


class _BudgetLiteLLM(types.SimpleNamespace):
    """Fake litellm whose every non-stream call reports 1000 tokens at a fixed
    per-token cost, so a couple of turns cross a small budget."""

    RateLimitError = real_litellm.RateLimitError
    APIConnectionError = real_litellm.APIConnectionError
    Timeout = real_litellm.Timeout
    InternalServerError = real_litellm.InternalServerError
    ServiceUnavailableError = real_litellm.ServiceUnavailableError
    AuthenticationError = real_litellm.AuthenticationError
    BadRequestError = real_litellm.BadRequestError

    async def acompletion(self, **kw):
        # Streaming shape (run() tries streaming first): a tool-call delta chunk,
        # then a usage-only final chunk. The tool call keeps the loop going so,
        # absent a cap, it would spend on the next turn too.
        async def gen():
            tc = types.SimpleNamespace(
                index=0,
                id="c1",
                function=types.SimpleNamespace(name="noop", arguments="{}"),
            )
            delta = types.SimpleNamespace(content=None, reasoning_content=None, tool_calls=[tc])
            yield types.SimpleNamespace(choices=[types.SimpleNamespace(delta=delta)], usage=None)
            yield types.SimpleNamespace(
                choices=[],
                usage=types.SimpleNamespace(
                    prompt_tokens=800, completion_tokens=200, total_tokens=1000
                ),
            )

        return gen()

    def cost_per_token(self, model, prompt_tokens, completion_tokens):
        return (0.01 * prompt_tokens, 0.01 * completion_tokens)  # $10 per 1k tokens here


def test_run_stops_when_token_budget_exceeded(monkeypatch):
    a = _agent_with_real_usage(max_tokens=1500)  # one 1000-tok call is fine, two isn't
    monkeypatch.setitem(__import__("sys").modules, "litellm", _BudgetLiteLLM())
    a.toolbox.call = lambda *a, **k: "ok"

    async def run():
        return [ev async for ev in a.run("go")]

    events = asyncio.run(run())
    assert any(ev.type == "error" and "token limit exceeded" in ev.text for ev in events)


def test_run_stops_when_usd_budget_exceeded(monkeypatch):
    a = _agent_with_real_usage(max_usd=0.01)  # first call already blows a 1-cent cap
    monkeypatch.setitem(__import__("sys").modules, "litellm", _BudgetLiteLLM())
    a.toolbox.call = lambda *a, **k: "ok"

    async def run():
        return [ev async for ev in a.run("go")]

    events = asyncio.run(run())
    assert any(ev.type == "error" and "budget exceeded" in ev.text for ev in events)


def test_no_budget_means_unlimited():
    from opendot.agent.config import AgentConfig as AC

    c = AC(model="m", workdir="/tmp")
    assert c.max_usd is None and c.max_tokens is None


def test_session_summary_counts_only_this_session(tmp_path):
    # A ledger entry written before the agent starts must not be counted; only
    # actions taken during this session appear, split by reversibility.
    from opendot.reversibility.ledger import LedgerEntry
    from opendot.reversibility.snapshots import project_id_for

    workdir = str(tmp_path)
    pid = project_id_for(workdir)

    def _log(id_, reversible):
        from opendot.reversibility import ledger

        ledger.append(
            pid,
            LedgerEntry(
                id=id_, kind="write", detail="f", snapshot_before="s", reversible=reversible
            ),
        )

    _log("000001", True)  # a prior-session action

    a = Agent(AgentConfig(model="m", workdir=workdir))
    assert a.session_summary()["actions"] == 0  # baseline excludes the prior action

    _log("000002", True)
    _log("000003", False)
    a.usage.cost_usd = 0.05
    a.usage.total_tokens = 2000

    s = a.session_summary()
    assert s["actions"] == 2
    assert s["reversible"] == 1
    assert s["irreversible"] == 1
    assert s["cost_usd"] == 0.05
    assert s["total_tokens"] == 2000


# -- issue #139: explorer subagents must not be able to spawn explorers --


def _run_and_capture_tools(a: Agent, monkeypatch) -> list[dict]:
    """Run one turn with `_dispatch_turn` faked out (no real model call), and
    return the `tools` list `run()` actually handed to the dispatch path —
    the same list a live model call would have seen."""
    captured: dict[str, list[dict]] = {}

    async def fake_dispatch_turn(litellm, tools):
        captured["tools"] = tools
        yield _Assembled({"role": "assistant", "content": "done"}, [])

    monkeypatch.setattr(a, "_dispatch_turn", fake_dispatch_turn)

    async def run():
        async for _ in a.run("hi"):
            pass

    asyncio.run(run())
    return captured["tools"]


def test_normal_agent_toolbox_is_not_read_only(tmp_path):
    a = Agent(AgentConfig(model="m", workdir=str(tmp_path)))
    assert a.toolbox.read_only is False


def test_normal_agent_has_explorers_enabled(tmp_path):
    a = Agent(AgentConfig(model="m", workdir=str(tmp_path)))
    assert a.explorers_enabled is True


def test_read_only_agent_toolbox_is_read_only(tmp_path):
    a = Agent(AgentConfig(model="m", workdir=str(tmp_path)), read_only=True)
    assert a.toolbox.read_only is True


def test_read_only_agent_has_explorers_disabled(tmp_path):
    a = Agent(AgentConfig(model="m", workdir=str(tmp_path)), read_only=True)
    assert a.explorers_enabled is False


def test_read_only_agent_run_does_not_pass_spawn_explorers(tmp_path, monkeypatch):
    a = Agent(AgentConfig(model="m", workdir=str(tmp_path)), read_only=True)
    tools = _run_and_capture_tools(a, monkeypatch)
    assert not any(t["function"]["name"] == "spawn_explorers" for t in tools)


def test_normal_agent_run_passes_spawn_explorers(tmp_path, monkeypatch):
    a = Agent(AgentConfig(model="m", workdir=str(tmp_path)))
    tools = _run_and_capture_tools(a, monkeypatch)
    assert any(t["function"]["name"] == "spawn_explorers" for t in tools)
