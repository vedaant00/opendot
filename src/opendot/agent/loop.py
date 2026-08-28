"""The agent loop — a model-agnostic ReAct loop over LiteLLM.

`Agent.run(user_message)` is an async generator of `Event`s. It keeps a running
conversation (`self.messages`) across turns so the CLI REPL is just repeated
`run(...)` calls. Tool calls are executed locally via the `Toolbox` and fed back
to the model until it produces a final answer (no more tool calls).
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator

from opendot.agent.config import AgentConfig
from opendot.agent.events import Event
from opendot.agent.prompt import DEFAULT_SYSTEM_PROMPT
from opendot.tools.local import Toolbox

_SPAWN_EXPLORERS_SPEC = {
    "type": "function",
    "function": {
        "name": "spawn_explorers",
        "description": (
            "Investigate several INDEPENDENT questions about the project in "
            "parallel. Each task runs as a read-only subagent (grep/glob/read "
            "only — it cannot change anything) and returns findings. Use this to "
            "understand a codebase fast by splitting distinct questions across "
            "lanes. For anything that changes files, do it yourself."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "tasks": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "2-6 independent, self-contained investigation tasks.",
                }
            },
            "required": ["tasks"],
        },
    },
}


class _StreamUnsupported(Exception):
    """Raised when a provider's stream can't yield structured tool calls."""


# Cap on the exponential-backoff sleep between retries so a high max_retries
# can't turn into multi-minute waits before a turn finally gives up.
_MAX_BACKOFF_SECONDS = 30


@dataclass
class _Assembled:
    """Sentinel yielded at the end of a turn carrying the assembled result."""

    assistant_msg: dict[str, Any]
    tool_calls: list[dict[str, Any]]


def _assistant_msg(content: str, calls: list[dict[str, Any]]) -> dict[str, Any]:
    msg: dict[str, Any] = {"role": "assistant", "content": content or None}
    if calls:
        msg["tool_calls"] = [
            {
                "id": c["id"],
                "type": "function",
                "function": {"name": c["name"], "arguments": c["args"] or "{}"},
            }
            for c in calls
        ]
    return msg


class Agent:
    def __init__(
        self,
        config: AgentConfig | None = None,
        confirm=None,
        mcp_manager=None,
        read_only: bool = False,
    ) -> None:
        self.config = config or AgentConfig()
        self.mcp = mcp_manager

        # The reversibility engine snapshots + logs before every mutating action.
        # Snapshot include/exclude rules come from OPENDOT.md (or defaults).
        from opendot.reversibility.engine import Reversibility
        from opendot.reversibility.rules import load_rules

        # Record which model + sampling params drove each action, for a fully
        # auditable ledger. Only include params that are actually set, so entries
        # stay tidy and the default case adds nothing.
        params = {
            k: v
            for k, v in (
                ("temperature", self.config.temperature),
                ("api_base", self.config.api_base),
            )
            if v is not None
        }
        self.reversibility = Reversibility(
            workdir=self.config.workdir,
            rules=load_rules(self.config.workdir),
            model=self.config.model,
            params=params,
        )
        # Explorer toolboxes are read-only and previously used reversibility=None.
        # Preserve that behavior while constructing the correct toolbox up front.
        self.toolbox = Toolbox(
            self.config.workdir,
            reversibility=None if read_only else self.reversibility,
            confirm=confirm,
            read_only=read_only,
            mcp_manager=mcp_manager,
        )
        system = self.config.system_prompt or DEFAULT_SYSTEM_PROMPT
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": system}]

        from opendot.agent.usage import Usage

        self.usage = Usage()  # running token/cost totals for the session

        # Mark where the ledger stands at session start, so the end-of-session
        # summary can report only the actions taken during *this* session (the
        # ledger itself is append-only and spans every past session).
        self._session_start_actions = len(self.reversibility.history())

        # The main agent can fan out read-only explorer subagents; explorers
        # themselves cannot (no recursive spawning). Read-only toolboxes = explorer.
        self.explorers_enabled = not getattr(self.toolbox, "read_only", False)

    def reset(self) -> None:
        """Clear the conversation (the /clear context-reset barrier), keeping system."""
        self.messages = self.messages[:1]

    def session_summary(self) -> dict[str, Any]:
        """A snapshot of what this session spent and did, for an end-of-run card.

        Counts only actions taken since the session started (the ledger persists
        across sessions), and splits them by whether they're reversible.
        """
        actions = self.reversibility.history()[self._session_start_actions :]
        reversible = sum(1 for a in actions if a.reversible)
        return {
            "actions": len(actions),
            "reversible": reversible,
            "irreversible": len(actions) - reversible,
            "cost_usd": self.usage.cost_usd,
            "total_tokens": self.usage.total_tokens,
            "calls": len(self.usage.calls),
        }

    def _session_path(self) -> Path:
        from opendot.reversibility.snapshots import project_id_for, store_root

        project_id = project_id_for(self.config.workdir)
        return store_root() / "sessions" / f"{project_id}.json"

    def save_session(self) -> Path:
        """Persist this project's conversation and return the session path."""
        path = self._session_path()
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        payload = {
            "version": 1,
            "workdir": str(Path(self.config.workdir).resolve()),
            "model": self.config.model,
            "messages": self.messages,
        }
        # Create the temp file 0600 from the start (not write-then-chmod), so the
        # conversation content is never briefly readable at the umask default.
        temp = path.with_suffix(".tmp")
        data = json.dumps(payload, ensure_ascii=False, indent=2)
        fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
        temp.replace(path)
        return path

    def load_session(self) -> bool:
        """Resume this project's saved conversation, or leave this agent fresh."""
        try:
            payload = json.loads(self._session_path().read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return False

        messages = payload.get("messages") if isinstance(payload, dict) else None
        model = payload.get("model") if isinstance(payload, dict) else None
        saved_workdir = payload.get("workdir") if isinstance(payload, dict) else None
        version = payload.get("version") if isinstance(payload, dict) else None
        if (
            version != 1
            or not isinstance(messages, list)
            or not messages
            or not all(isinstance(message, dict) for message in messages)
            or messages[0].get("role") != "system"
            or not isinstance(model, str)
            or not model
            or not isinstance(saved_workdir, str)
            or Path(saved_workdir).resolve() != Path(self.config.workdir).resolve()
        ):
            return False

        # Keep today's system/project prompt rather than reviving a stale copy.
        self.messages = self.messages[:1] + messages[1:]
        self.config.model = model
        self.reversibility.model = model
        return True

    def compact(self, keep_recent: int = 6) -> int:
        """Trim old turns to fight context bloat, keeping the system prompt and
        the most recent messages. Returns how many messages were dropped.

        v1 is a simple truncation (honest and predictable). A summarizing
        compaction can replace this later without changing the interface.
        """
        if len(self.messages) <= keep_recent + 1:
            return 0
        system, tail = self.messages[:1], self.messages[-keep_recent:]
        dropped = len(self.messages) - len(system) - len(tail)
        self.messages = system + tail
        return dropped

    async def run(self, user_message: str) -> AsyncIterator[Event]:
        """Run one user turn to completion, yielding events."""
        import litellm

        # Keep litellm/provider chatter out of the user's terminal.
        litellm.suppress_debug_info = True
        litellm.set_verbose = False

        self.messages.append({"role": "user", "content": user_message})
        tools = self.toolbox.specs()
        if self.explorers_enabled:
            tools = tools + [_SPAWN_EXPLORERS_SPEC]

        for _ in range(self.config.max_steps):
            # Stream the turn: reasoning ("thinking") and answer ("text") arrive
            # live, and tool-call deltas are reassembled. Streaming is what makes
            # the UX feel alive. If streaming fails for a provider, fall back to
            # a single non-streaming call (some local models emit tool calls as
            # text in streaming mode). Transient provider errors (rate-limit/5xx/
            # timeout) are retried with backoff before any output is produced.
            gen = None
            async for ev in self._dispatch_turn(litellm, tools):
                if isinstance(ev, _Assembled):
                    gen = ev
                else:
                    yield ev

            if gen is None:  # a terminal "error" event was already yielded
                return

            self.messages.append(gen.assistant_msg)

            if not gen.tool_calls:
                yield Event("final")
                return

            for i, c in enumerate(gen.tool_calls):
                name, call_id, raw_args = c["name"], c["id"], c["args"]
                try:
                    args = json.loads(raw_args or "{}")
                except json.JSONDecodeError:
                    args = {}

                # spawn_explorers is handled by the runtime (parallel read-only
                # subagents), not the toolbox — it streams lane-tagged events.
                if name == "spawn_explorers" and self.explorers_enabled:
                    from opendot.agent.explorers import run_explorers

                    result = "(no findings)"
                    async for ev in run_explorers(
                        args.get("tasks", []),
                        model=self.config.model,
                        workdir=self.config.workdir,
                        api_base=self.config.api_base,
                        temperature=self.config.temperature,
                    ):
                        if ev.type == "tool_end" and ev.tool == "spawn_explorers":
                            result = ev.result  # merged findings
                        else:
                            yield ev
                    self.messages.append(
                        {"role": "tool", "tool_call_id": call_id, "content": result}
                    )
                    continue

                yield Event("tool_start", tool=name, args=args)
                # Run the (synchronous) tool off the event loop. This is what lets
                # a confirm-callback safely block for a UI prompt (e.g. the TUI
                # modal) without freezing the loop.
                result = await asyncio.to_thread(self.toolbox.call, name, args)
                yield Event("tool_end", tool=name, result=result)
                self.messages.append({"role": "tool", "tool_call_id": call_id, "content": result})

        yield Event("error", text=f"stopped: hit max_steps ({self.config.max_steps})")

    @staticmethod
    def _transient_errors(litellm) -> tuple[type[BaseException], ...]:
        """Provider errors worth retrying: rate-limit, 5xx, timeout, connection.

        Deliberately excludes fatal errors (auth, bad request) — those should
        fail fast rather than burn through the retry budget.
        """
        return (
            litellm.RateLimitError,
            litellm.APIConnectionError,
            litellm.Timeout,
            litellm.InternalServerError,
            litellm.ServiceUnavailableError,
        )

    async def _dispatch_turn(self, litellm, tools):
        """Run one model turn (streaming, falling back to non-streaming),
        retrying transient provider errors with exponential backoff.

        Yields Events, then a final _Assembled sentinel on success, or a
        terminal "error" Event if the turn ultimately fails. A retry is only
        attempted while no output has been produced yet for this attempt —
        once content has streamed to the user, a blip is surfaced as-is
        rather than silently redone.
        """
        transient = self._transient_errors(litellm)
        attempt = 0
        while True:
            produced_output = False
            try:
                async for ev in self._stream_turn(litellm, tools):
                    if not isinstance(ev, _Assembled):
                        produced_output = True
                    yield ev
                return
            except _StreamUnsupported:
                try:
                    async for ev in self._nonstream_turn(litellm, tools):
                        if not isinstance(ev, _Assembled):
                            produced_output = True
                        yield ev
                    return
                except Exception as exc:  # noqa: BLE001
                    should_retry = (
                        not produced_output
                        and attempt < self.config.max_retries
                        and isinstance(exc, transient)
                    )
                    if not should_retry:
                        yield Event("error", text=f"model call failed: {exc}")
                        return
            except Exception as exc:  # noqa: BLE001
                should_retry = (
                    not produced_output
                    and attempt < self.config.max_retries
                    and isinstance(exc, transient)
                )
                if not should_retry:
                    yield Event("error", text=f"model call failed: {exc}")
                    return

            await asyncio.sleep(min(2**attempt, _MAX_BACKOFF_SECONDS))
            attempt += 1

    async def _stream_turn(self, litellm, tools):
        """Stream one model turn. Yields Events, then a final _Assembled sentinel.

        Raises _StreamUnsupported if the stream looks like a provider that emits
        tool calls as plain text (so the caller can fall back to non-streaming).
        """
        started = time.monotonic()
        stream = await litellm.acompletion(
            model=self.config.model,
            messages=self.messages,
            tools=tools,
            temperature=self.config.temperature,
            stream=True,
            stream_options={"include_usage": True},
            api_base=self.config.api_base,  # None => provider default
        )
        content_parts: list[str] = []
        tool_calls: dict[int, dict[str, Any]] = {}
        # Usage may arrive on its own final chunk (no choices) or attached to a
        # content chunk — capture whichever arrives first, but don't record it
        # until the stream fully completes. Recording it as soon as it's seen
        # would double-count on a retry: if a transient error drops the
        # connection after the usage chunk but before the stream finishes, the
        # aborted attempt must not have already billed usage that the retried,
        # successful attempt will bill again.
        usage_chunk = None

        async for chunk in stream:
            if usage_chunk is None and getattr(chunk, "usage", None):
                usage_chunk = chunk
            choices = getattr(chunk, "choices", None)
            if not choices:
                continue
            delta = choices[0].delta
            # reasoning models expose reasoning separately
            reasoning = getattr(delta, "reasoning_content", None)
            if reasoning:
                yield Event("thinking", text=reasoning)
            if getattr(delta, "content", None):
                content_parts.append(delta.content)
                yield Event("text", text=delta.content)
            for tc in getattr(delta, "tool_calls", None) or []:
                slot = tool_calls.setdefault(tc.index, {"id": "", "name": "", "args": ""})
                if tc.id:
                    slot["id"] = tc.id
                if tc.function and tc.function.name:
                    slot["name"] = tc.function.name
                if tc.function and tc.function.arguments:
                    slot["args"] += tc.function.arguments

        if usage_chunk is not None:
            self.usage.add_response(
                usage_chunk, litellm, model=self.config.model, latency_s=time.monotonic() - started
            )

        # Enforce spend/token budget set via --usd/--tokens or OPENDOT_MAX_USD/MAX_TOKENS.
        if self.config.max_usd is not None and self.usage.cost_usd > self.config.max_usd:
            yield Event(
                "error",
                text=f"budget exceeded: ${self.usage.cost_usd:.4f} > ${self.config.max_usd:.2f}",
            )
            return
        if self.config.max_tokens is not None and self.usage.total_tokens > self.config.max_tokens:
            yield Event(
                "error",
                text=f"token limit exceeded: {self.usage.total_tokens} > {self.config.max_tokens}",
            )
            return

        calls = [
            {"id": c["id"] or f"call_{i}", "name": c["name"], "args": c["args"]}
            for i, c in sorted(tool_calls.items())
            if c["name"]
        ]
        yield _Assembled(_assistant_msg("".join(content_parts), calls), calls)

    async def _nonstream_turn(self, litellm, tools):
        """Non-streaming fallback (reliable tool calls; no live tokens)."""
        started = time.monotonic()
        resp = await litellm.acompletion(
            model=self.config.model,
            messages=self.messages,
            tools=tools,
            temperature=self.config.temperature,
            stream=False,
            api_base=self.config.api_base,  # None => provider default
        )
        self.usage.add_response(
            resp, litellm, model=self.config.model, latency_s=time.monotonic() - started
        )

        # Enforce spend/token budget.
        if self.config.max_usd is not None and self.usage.cost_usd > self.config.max_usd:
            yield Event(
                "error",
                text=f"budget exceeded: ${self.usage.cost_usd:.4f} > ${self.config.max_usd:.2f}",
            )
            return
        if self.config.max_tokens is not None and self.usage.total_tokens > self.config.max_tokens:
            yield Event(
                "error",
                text=f"token limit exceeded: {self.usage.total_tokens} > {self.config.max_tokens}",
            )
            return

        msg = resp.choices[0].message
        raw = getattr(msg, "tool_calls", None) or []
        calls = [
            {
                "id": tc.id or f"call_{i}",
                "name": tc.function.name,
                "args": tc.function.arguments or "{}",
            }
            for i, tc in enumerate(raw)
        ]
        if msg.content:
            yield Event("text", text=msg.content)
        yield _Assembled(_assistant_msg(msg.content or "", calls), calls)
