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
import pytest

from opendot.agent.config import AgentConfig
from opendot.agent.events import Event
from opendot.agent.loop import Agent, _Assembled, _looks_like_tool_call, _StreamUnsupported


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


# --- issue #140: plain-text tool-call detection and non-streaming fallback ---


def test_looks_like_tool_call_helper():
    tools = [
        {"type": "function", "function": {"name": "read_file", "parameters": {}}},
        {"type": "function", "function": {"name": "run_shell", "parameters": {}}},
    ]

    # Common <tool_call> formats
    assert _looks_like_tool_call(
        '<tool_call>{"name":"read_file","arguments":{"path":"foo.py"}}</tool_call>',
        tools,
    )
    assert _looks_like_tool_call(
        '\n<tool_call>\n{\n  "name": "read_file",\n  "arguments": {"path": "foo.py"}\n}\n</tool_call>\n',
        tools,
    )
    assert _looks_like_tool_call(
        'Prefix text before tag\n<tool_call>{"name":"read_file","arguments":{"path":"foo.py"}}</tool_call>',
        tools,
    )

    # Bare JSON formats (including wrapped in markdown code fence)
    assert _looks_like_tool_call(
        '{"name": "read_file", "arguments": {"path": "foo.py"}}',
        tools,
    )
    assert _looks_like_tool_call(
        '```json\n{"name": "read_file", "arguments": {"path": "foo.py"}}\n```',
        tools,
    )

    # Negative cases: must NOT match
    # Malformed JSON in <tool_call> and bare text
    assert not _looks_like_tool_call(
        '<tool_call>{"name":"read_file","arguments":{</tool_call>',
        tools,
    )
    assert not _looks_like_tool_call(
        '{"name":"read_file","arguments":',
        tools,
    )
    # JSON list of tool calls is not recognized
    assert not _looks_like_tool_call(
        '[{"name": "read_file", "arguments": {"path": "foo.py"}}]',
        tools,
    )
    # Tool not in supplied tools
    assert not _looks_like_tool_call(
        '<tool_call>{"name":"unknown_tool","arguments":{}}</tool_call>',
        tools,
    )
    assert not _looks_like_tool_call(
        '{"name": "unknown_tool", "arguments": {}}',
        tools,
    )
    # Ordinary text
    assert not _looks_like_tool_call("Hello! How can I help you today?", tools)
    # Ordinary JSON response
    assert not _looks_like_tool_call('{"status": "ok", "items": [1, 2, 3]}', tools)
    assert not _looks_like_tool_call('{"name": "Alice", "role": "admin"}', tools)
    # JSON with matching name but not a tool call shape (no arguments)
    assert not _looks_like_tool_call('{"name": "read_file", "description": "reads a file"}', tools)
    # Empty inputs
    assert not _looks_like_tool_call("", tools)
    assert not _looks_like_tool_call('{"name": "read_file", "arguments": {}}', [])
    assert not _looks_like_tool_call('{"name": "read_file", "arguments": {}}', None)


class _BaseFakeLiteLLM(types.SimpleNamespace):
    RateLimitError = real_litellm.RateLimitError
    APIConnectionError = real_litellm.APIConnectionError
    Timeout = real_litellm.Timeout
    InternalServerError = real_litellm.InternalServerError
    ServiceUnavailableError = real_litellm.ServiceUnavailableError
    AuthenticationError = real_litellm.AuthenticationError
    BadRequestError = real_litellm.BadRequestError


def test_stream_turn_raises_stream_unsupported_on_plain_text_tool_call():
    """_stream_turn directly raises _StreamUnsupported when plain text tool call is assembled."""

    class _PlainTextLiteLLM(_BaseFakeLiteLLM):
        async def acompletion(self, **kw):
            async def gen():
                yield _chunk_with_choices(
                    '<tool_call>{"name":"read_file","arguments":{"path":"a.py"}}</tool_call>'
                )

            return gen()

    a = _bare_agent()
    tools = [{"type": "function", "function": {"name": "read_file", "parameters": {}}}]

    async def run():
        events = []
        async for ev in a._stream_turn(_PlainTextLiteLLM(), tools):
            events.append(ev)
        return events

    with pytest.raises(_StreamUnsupported):
        asyncio.run(run())


def test_stream_tool_call_plain_text_falls_back_to_nonstream_and_executes(monkeypatch):
    """Fake streaming LiteLLM returns a tool call as plain text, then the non-streaming
    call returns a structured tool call. Verifies:
      - streaming path detects it,
      - fallback occurs,
      - non-streaming is called,
      - the tool actually executes,
      - the tool call is not silently dropped.
    """
    tools = [{"type": "function", "function": {"name": "my_tool", "parameters": {}}}]
    a = _bare_agent()
    a.toolbox.specs = lambda: tools
    tool_calls_executed = []
    a.toolbox.call = lambda name, args: tool_calls_executed.append((name, args)) or "success_output"

    class _FallbackLiteLLM(_BaseFakeLiteLLM):
        def __init__(self):
            super().__init__()
            self.stream_calls = 0
            self.nonstream_calls = 0

        async def acompletion(self, **kw):
            if kw.get("stream"):
                self.stream_calls += 1
                if self.stream_calls == 1:
                    # Turn 1 streaming: returns tool call as plain text
                    async def gen_plain_text():
                        yield _chunk_with_choices(
                            '<tool_call>{"name":"my_tool","arguments":{"file":"test.txt"}}</tool_call>'
                        )

                    return gen_plain_text()
                else:
                    # Turn 2 streaming: returns final answer
                    async def gen_final():
                        yield _chunk_with_choices("Done processing")

                    return gen_final()
            else:
                # Fallback to non-streaming
                self.nonstream_calls += 1
                tc = types.SimpleNamespace(
                    id="call_fallback_1",
                    function=types.SimpleNamespace(name="my_tool", arguments='{"file":"test.txt"}'),
                )
                msg = types.SimpleNamespace(content="", tool_calls=[tc])
                resp = types.SimpleNamespace(
                    choices=[types.SimpleNamespace(message=msg)],
                    usage=_Usage(),
                )
                return resp

    fake = _FallbackLiteLLM()
    events = _run_agent(a, fake, monkeypatch)

    # 1. streaming path detects it and attempts turn 1 stream
    assert fake.stream_calls >= 1
    # 2. fallback occurs and non-streaming is called
    assert fake.nonstream_calls == 1
    # 3. the tool actually executes with correct args
    assert tool_calls_executed == [("my_tool", {"file": "test.txt"})]
    # 4. the tool call is not silently dropped: tool_start and tool_end events exist
    assert any(ev.type == "tool_start" and ev.tool == "my_tool" for ev in events)
    assert any(
        ev.type == "tool_end" and ev.tool == "my_tool" and ev.result == "success_output"
        for ev in events
    )
    assert any(ev.type == "final" for ev in events)
    assert not any(ev.type == "error" for ev in events)


def test_bare_json_tool_call_stream_falls_back_and_executes(monkeypatch):
    """A bare JSON tool call payload during streaming also triggers fallback and executes."""
    tools = [{"type": "function", "function": {"name": "my_tool", "parameters": {}}}]
    a = _bare_agent()
    a.toolbox.specs = lambda: tools
    tool_calls_executed = []
    a.toolbox.call = lambda name, args: tool_calls_executed.append((name, args)) or "ok"

    class _BareJsonFallbackLiteLLM(_BaseFakeLiteLLM):
        def __init__(self):
            super().__init__()
            self.stream_calls = 0
            self.nonstream_calls = 0

        async def acompletion(self, **kw):
            if kw.get("stream"):
                self.stream_calls += 1
                if self.stream_calls == 1:

                    async def gen_bare_json():
                        yield _chunk_with_choices('{"name":"my_tool","arguments":{"count":42}}')

                    return gen_bare_json()
                else:

                    async def gen_final():
                        yield _chunk_with_choices("Finished")

                    return gen_final()
            else:
                self.nonstream_calls += 1
                tc = types.SimpleNamespace(
                    id="call_fallback_2",
                    function=types.SimpleNamespace(name="my_tool", arguments='{"count":42}'),
                )
                msg = types.SimpleNamespace(content="", tool_calls=[tc])
                return types.SimpleNamespace(
                    choices=[types.SimpleNamespace(message=msg)],
                    usage=_Usage(),
                )

    fake = _BareJsonFallbackLiteLLM()
    events = _run_agent(a, fake, monkeypatch)

    assert fake.nonstream_calls == 1
    assert tool_calls_executed == [("my_tool", {"count": 42})]
    assert any(ev.type == "tool_start" and ev.tool == "my_tool" for ev in events)
    assert any(ev.type == "final" for ev in events)


def test_normal_structured_streaming_tool_calls_no_fallback(monkeypatch):
    """Verify normal structured streaming tool_calls across chunk fragments
    still works and does not trigger fallback."""
    tools = [{"type": "function", "function": {"name": "my_tool", "parameters": {}}}]
    a = _bare_agent()
    a.toolbox.specs = lambda: tools
    tool_calls_executed = []
    a.toolbox.call = lambda name, args: tool_calls_executed.append((name, args)) or "ok"

    class _StructuredStreamLiteLLM(_BaseFakeLiteLLM):
        def __init__(self):
            super().__init__()
            self.stream_calls = 0
            self.nonstream_calls = 0

        async def acompletion(self, **kw):
            if kw.get("stream"):
                self.stream_calls += 1
                if self.stream_calls == 1:

                    async def gen_tc():
                        # Chunk 1: delivers tool name and opening argument fragment
                        tc1 = types.SimpleNamespace(
                            index=0,
                            id="call_normal_stream",
                            function=types.SimpleNamespace(name="my_tool", arguments='{"key":'),
                        )
                        delta1 = types.SimpleNamespace(
                            content=None, reasoning_content=None, tool_calls=[tc1]
                        )
                        yield types.SimpleNamespace(
                            choices=[types.SimpleNamespace(delta=delta1)], usage=None
                        )
                        # Chunk 2: delivers closing argument fragment without repeated name
                        tc2 = types.SimpleNamespace(
                            index=0,
                            id=None,
                            function=types.SimpleNamespace(name=None, arguments='"value"}'),
                        )
                        delta2 = types.SimpleNamespace(
                            content=None, reasoning_content=None, tool_calls=[tc2]
                        )
                        yield types.SimpleNamespace(
                            choices=[types.SimpleNamespace(delta=delta2)], usage=None
                        )

                    return gen_tc()
                else:

                    async def gen_done():
                        yield _chunk_with_choices("Completed")

                    return gen_done()
            else:
                self.nonstream_calls += 1
                raise AssertionError(
                    "Non-streaming should NOT have been called for structured stream tool call"
                )

    fake = _StructuredStreamLiteLLM()
    events = _run_agent(a, fake, monkeypatch)

    assert fake.stream_calls == 2
    assert fake.nonstream_calls == 0  # No fallback!
    assert tool_calls_executed == [("my_tool", {"key": "value"})]
    assert any(
        ev.type == "tool_start" and ev.tool == "my_tool" and ev.args == {"key": "value"}
        for ev in events
    )
    assert any(ev.type == "final" for ev in events)


def test_ordinary_streamed_text_does_not_trigger_fallback(monkeypatch):
    """Verify ordinary streamed text (including normal JSON answers) does not trigger fallback."""
    tools = [{"type": "function", "function": {"name": "my_tool", "parameters": {}}}]
    a = _bare_agent()
    a.toolbox.specs = lambda: tools

    class _TextStreamLiteLLM(_BaseFakeLiteLLM):
        def __init__(self, text: str):
            super().__init__()
            self.text = text
            self.stream_calls = 0
            self.nonstream_calls = 0

        async def acompletion(self, **kw):
            if kw.get("stream"):
                self.stream_calls += 1

                async def gen():
                    yield _chunk_with_choices(self.text)

                return gen()
            else:
                self.nonstream_calls += 1
                raise AssertionError("Non-streaming fallback should NOT have been triggered")

    # Test ordinary plain text
    fake1 = _TextStreamLiteLLM("Here is the answer to your question.")
    events1 = _run_agent(a, fake1, monkeypatch)
    assert fake1.stream_calls == 1
    assert fake1.nonstream_calls == 0
    assert any(ev.type == "final" for ev in events1)

    # Test ordinary JSON response (not a tool call)
    fake2 = _TextStreamLiteLLM('{"status": "ok", "items": ["a", "b"]}')
    events2 = _run_agent(a, fake2, monkeypatch)
    assert fake2.stream_calls == 1
    assert fake2.nonstream_calls == 0
    assert any(ev.type == "final" for ev in events2)

    # Test JSON response for a tool not in tools
    fake3 = _TextStreamLiteLLM('{"name": "other_tool", "arguments": {"x": 1}}')
    events3 = _run_agent(a, fake3, monkeypatch)
    assert fake3.stream_calls == 1
    assert fake3.nonstream_calls == 0
    assert any(ev.type == "final" for ev in events3)


def test_malformed_streamed_tool_call_does_not_trigger_fallback(monkeypatch):
    """When a stream emits a malformed <tool_call> payload for a known tool,
    _StreamUnsupported is not raised through dispatch, non-streaming fallback is
    not called, and the response completes normally."""
    tools = [{"type": "function", "function": {"name": "my_tool", "parameters": {}}}]
    a = _bare_agent()
    a.toolbox.specs = lambda: tools

    class _MalformedStreamLiteLLM(_BaseFakeLiteLLM):
        def __init__(self):
            super().__init__()
            self.stream_calls = 0
            self.nonstream_calls = 0

        async def acompletion(self, **kw):
            if kw.get("stream"):
                self.stream_calls += 1

                async def gen():
                    yield _chunk_with_choices(
                        '<tool_call>{"name":"my_tool","arguments":{</tool_call>'
                    )

                return gen()
            else:
                self.nonstream_calls += 1
                raise AssertionError("Non-streaming fallback should NOT have been triggered")

    fake = _MalformedStreamLiteLLM()
    events = _run_agent(a, fake, monkeypatch)

    assert fake.stream_calls == 1
    assert fake.nonstream_calls == 0
    assert any(ev.type == "final" for ev in events)


def test_usage_not_recorded_for_failed_streaming_attempt(monkeypatch):
    """Verify usage is not recorded for the failed streaming attempt before fallback.
    Only the successful non-streaming call's usage is recorded."""
    tools = [{"type": "function", "function": {"name": "my_tool", "parameters": {}}}]
    a = _bare_agent()
    a.toolbox.specs = lambda: tools
    a.toolbox.call = lambda name, args: "ok"

    usage_recorded = []
    a.usage.add_response = lambda resp, *args, **kw: usage_recorded.append(resp)

    class _UsageTrackingLiteLLM(_BaseFakeLiteLLM):
        def __init__(self):
            super().__init__()
            self.stream_calls = 0
            self.nonstream_calls = 0

        async def acompletion(self, **kw):
            if kw.get("stream"):
                self.stream_calls += 1
                if self.stream_calls == 1:
                    # Turn 1: streaming yields usage chunk AND plain-text tool call
                    async def gen_tool_with_usage():
                        yield _chunk_no_choices(with_usage=True)
                        yield _chunk_with_choices(
                            '<tool_call>{"name":"my_tool","arguments":{}}</tool_call>'
                        )

                    return gen_tool_with_usage()
                else:
                    # Turn 2: final stream call
                    async def gen_final():
                        yield _chunk_with_choices("Done", with_usage=True)

                    return gen_final()
            else:
                # Non-streaming fallback for Turn 1
                self.nonstream_calls += 1
                tc = types.SimpleNamespace(
                    id="call_fallback",
                    function=types.SimpleNamespace(name="my_tool", arguments="{}"),
                )
                msg = types.SimpleNamespace(content="", tool_calls=[tc])
                resp = types.SimpleNamespace(
                    choices=[types.SimpleNamespace(message=msg)],
                    usage=_Usage(),
                )
                return resp

    fake = _UsageTrackingLiteLLM()
    _run_agent(a, fake, monkeypatch)

    assert fake.nonstream_calls == 1
    # Usage was recorded for:
    # 1. The non-streaming fallback of Turn 1
    # 2. The successful streaming of Turn 2
    # The failed streaming attempt of Turn 1 was NOT recorded!
    assert len(usage_recorded) == 2
