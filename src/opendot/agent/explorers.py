"""Parallel read-only explorer subagents.

The main agent can call `spawn_explorers([task, task, ...])` to fan out several
independent *read-only* investigations at once — each a full reasoning agent
restricted to grep/glob/read/list (NO write, edit, or shell). They run
concurrently and each returns a short findings summary, which is fed back to the
main agent.

Why read-only: opendot's guarantee is a clean, sequential, undoable action
ledger. Parallel *writers* would race the filesystem and make "undo the last
action" ambiguous. Explorers never mutate anything, so they parallelize the safe
part (understanding the project) without touching that guarantee.

Events are lane-tagged (lane = subagent index) so the TUI can show parallel
"Explore Task" lanes like opencode.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator

from opendot.agent.events import Event

MAX_EXPLORERS = 6

_EXPLORER_SYSTEM = """\
You are a read-only explorer subagent. Your ONLY job is to investigate and \
report — you cannot change anything. You have grep, glob, read_file, and \
list_files. Use them to answer the task, then give a concise findings summary \
(a few bullet points with the key files/lines). Do not attempt to write or run \
anything; those tools are not available to you.
"""


async def run_explorers(
    tasks: list[str],
    *,
    model: str,
    workdir: str,
    api_base: str | None = None,
    temperature: float | None = None,
) -> AsyncIterator[Event]:
    """Run each task as a concurrent read-only subagent, yielding lane-tagged
    events, and finally a merged findings summary the caller can use."""
    from opendot.agent.config import AgentConfig
    from opendot.agent.loop import Agent

    tasks = [t for t in tasks if t.strip()][:MAX_EXPLORERS]
    if not tasks:
        yield Event("error", text="spawn_explorers: no tasks given")
        return

    # A queue lets concurrent lanes interleave their events into one stream.
    q: asyncio.Queue = asyncio.Queue()
    findings: dict[int, str] = {}

    async def one(lane: int, task: str) -> None:
        await q.put(Event("explorer_start", text=task, lane=lane))
        agent = Agent(
            AgentConfig(
                model=model,
                workdir=workdir,
                system_prompt=_EXPLORER_SYSTEM,
                api_base=api_base,
                temperature=temperature,
            ),
            read_only=True,
        )
        answer_parts: list[str] = []
        try:
            async for ev in agent.run(task):
                if ev.type == "tool_start":
                    await q.put(Event("explorer_step", text=f"{ev.tool}", lane=lane))
                elif ev.type == "text":
                    answer_parts.append(ev.text)
                elif ev.type == "error":
                    await q.put(Event("explorer_step", text=f"error: {ev.text}", lane=lane))
        except Exception as exc:  # noqa: BLE001
            answer_parts.append(f"(explorer failed: {exc})")
        summary = "".join(answer_parts).strip() or "(no findings)"
        findings[lane] = summary
        await q.put(Event("explorer_done", text=summary, lane=lane))

    async def run_all() -> None:
        await asyncio.gather(*(one(i, t) for i, t in enumerate(tasks)))
        await q.put(None)  # sentinel

    runner = asyncio.create_task(run_all())
    while True:
        ev = await q.get()
        if ev is None:
            break
        yield ev
    await runner

    # Attach the merged findings as the tool's textual result via a final event.
    merged = "\n\n".join(
        f"[explorer {i + 1}] {tasks[i]}\n{findings.get(i, '(no findings)')}"
        for i in range(len(tasks))
    )
    yield Event("tool_end", tool="spawn_explorers", result=merged)
