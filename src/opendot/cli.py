"""opendot CLI — the interactive chat REPL (default) plus a one-shot mode.

    opendot                      # open an interactive chat session
    opendot -p "list my files"   # run one task and exit (scripting/CI)
    opendot --model ollama/qwen2.5

This is a thin layer over the core Agent. The richer TUI (panels/live boxes) is
a later milestone; v1 uses a clean Rich-rendered REPL.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from opendot import __version__
from opendot.agent.config import AgentConfig
from opendot.agent.config import _max_tokens as _env_max_tokens
from opendot.agent.config import _max_usd as _env_max_usd
from opendot.agent.loop import Agent
from opendot.agent.prompt import DEFAULT_SYSTEM_PROMPT

console = Console()

SLASH_HELP = """\
[bold]Commands[/bold]
  /help     show this help
  /log      show the auditable history of actions taken
  /trace    per-model-call cost and timing for this session
  /diff     preview what /undo <id> would change, without touching disk
  /undo     revert the last action ( /undo <id> to restore to a point )
  /redo     re-apply the action /undo just reverted
  /clear    reset the conversation
  /save     save this project's conversation
  /resume   reload this project's saved conversation
  /compact  trim old conversation turns to free up context
  /model    show the current model ( /model <id> to switch )
  /provider connect a provider ( /provider <ENV_VAR> <api-key> )
  exit      quit (also: /exit, /quit, Ctrl-D)
"""


def _load_project_context(workdir: str) -> str | None:
    """Append an OPENDOT.md from the working dir to the system prompt, if present."""
    p = Path(workdir) / "OPENDOT.md"
    if p.exists():
        try:
            return p.read_text(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            return None
    return None


def _confirm(prompt: str) -> bool:
    """Ask the user to approve an irreversible action (interactive only)."""
    console.print(f"\n[bold yellow]⚠ {prompt}[/bold yellow]")
    try:
        ans = input("  [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    return ans in {"y", "yes"}


def _build_policy(args, workdir: str):
    """Merge the project's OPENDOT.md policy with the CLI --yes/--allow/--deny."""
    from opendot.agent.permissions import Policy, load_policy

    cli_policy = Policy(
        allow=list(getattr(args, "allow", []) or []),
        deny=list(getattr(args, "deny", []) or []),
        auto_approve=bool(getattr(args, "yes", False)),
    )
    return load_policy(workdir).merged_with(cli_policy)


def _forwarded_env_keys(model: str) -> list[str]:
    """API-key env vars to forward into a --sandbox container: just the one the
    model needs (if set), so the containerized run can reach its provider without
    leaking every secret in the host env."""
    from opendot.providers import env_var_for

    var = env_var_for(model)
    return [var] if var and os.environ.get(var) else []


def _make_confirm(args, workdir: str, interactive: bool):
    """Build the confirm callback: policy-gated, falling back to the interactive
    prompt (or an auto-decline in one-shot mode) for anything left to ask."""
    policy = _build_policy(args, workdir)
    ask = _confirm if interactive else (lambda _p: False)
    return policy.make_confirm(ask)


def _warn_if_missing_key(model: str) -> None:
    """Print a friendly hint if the model's expected API key isn't in the env.

    A warning only — we still let the call proceed, because LiteLLM may find the
    key another way and we'd rather not falsely block a working setup.
    """
    from opendot.providers import env_var_for

    env_var = env_var_for(model)
    if env_var and not os.environ.get(env_var):
        console.print(
            f"[yellow]![/yellow] No [bold]{env_var}[/bold] found in your environment "
            f"for model [cyan]{model}[/cyan].\n"
            f"  Set it (e.g. [dim]export {env_var}=...[/dim]), or pick another model "
            f"with [dim]--model[/dim] / [dim]$OPENDOT_MODEL[/dim].\n"
            f"  See the model table: https://github.com/vedaant00/opendot#any-model\n"
        )


def _build_agent(
    model: str,
    workdir: str,
    confirm=None,
    api_base: str | None = None,
    max_usd: float | None = None,
    max_tokens: int | None = None,
) -> Agent:
    # With a custom api_base (local OpenAI-compatible server like llama.cpp), no
    # provider key is needed — so skip the auto-switch. Otherwise: if the chosen
    # model's key isn't set but another provider's is, switch to that provider.
    if not api_base:
        from opendot.providers import env_var_for, model_for_available_key

        var = env_var_for(model)
        if var and not os.environ.get(var):
            alt = model_for_available_key()
            if alt and alt != model:
                console.print(
                    f"[dim]no {var}; using [cyan]{alt}[/cyan] "
                    f"(found its key in your environment)[/dim]"
                )
                model = alt

    system = DEFAULT_SYSTEM_PROMPT
    ctx = _load_project_context(workdir)
    if ctx:
        system += "\n\n# Project context (OPENDOT.md)\n" + ctx

    # Connect to any configured MCP servers (~/.opendot/mcp.json).
    mcp_manager = None
    try:
        from opendot.mcp import MCPManager, load_mcp_config

        cfg = load_mcp_config()
        if cfg:
            mcp_manager = MCPManager(cfg)
            mcp_manager.start()
    except Exception:  # noqa: BLE001 - MCP is optional; never block startup
        mcp_manager = None

    # The CLI flag wins when given, but passing None here would override
    # AgentConfig's env-var default_factory (an explicit None means the factory
    # never runs), silently dropping OPENDOT_MAX_USD / OPENDOT_MAX_TOKENS. Fall
    # back to the env var explicitly so the documented precedence holds.
    if max_usd is None:
        max_usd = _env_max_usd()
    if max_tokens is None:
        max_tokens = _env_max_tokens()

    return Agent(
        AgentConfig(
            model=model,
            workdir=workdir,
            system_prompt=system,
            api_base=api_base,
            max_usd=max_usd,
            max_tokens=max_tokens,
        ),
        confirm=confirm,
        mcp_manager=mcp_manager,
    )


def _cmd_log(workdir: str, clear: bool = False) -> None:
    """`opendot log` — show the auditable action history (or --clear it)."""
    from opendot.reversibility.engine import Reversibility
    from opendot.reversibility.rules import load_rules

    rev = Reversibility(workdir=workdir, rules=load_rules(workdir))
    if clear:
        n = rev.clear_history()
        console.print(
            f"[green]cleared[/green] {n} ledger entr{'y' if n == 1 else 'ies'} "
            "(undo history discarded)"
        )
        return
    entries = rev.history()
    if not entries:
        console.print("[dim]no actions recorded yet[/dim]")
        return

    # Timeline: actions oldest→newest, with a cursor marking where the workspace
    # currently sits. `undone` trailing actions have been reverted (undo) and are
    # redoable; they render dimmed below the cursor.
    undone = rev.redo_available()
    cursor_after = len(entries) - undone  # actions [0:cursor_after] are applied

    console.print("[bold]opendot action history[/bold] (most recent last)\n")
    for i, e in enumerate(entries):
        is_undone = i >= cursor_after
        detail = e.detail if len(e.detail) < 70 else e.detail[:67] + "..."
        if is_undone:
            console.print(f"  [dim]{e.id}  ↶ undone  {e.kind}  {detail}[/dim]", highlight=False)
        else:
            mark = "[green]↺[/green]" if e.reversible else "[red]✗ irreversible[/red]"
            console.print(f"  {e.id}  {mark}  [cyan]{e.kind}[/cyan]  {detail}")
            if e.note:
                console.print(f"        [dim]{e.note}[/dim]")
            if e.model:
                params = "".join(f" {k}={v}" for k, v in e.params.items())
                console.print(f"        [dim]model: {e.model}{params}[/dim]")
        # Draw the cursor right after the last applied action.
        if i == cursor_after - 1:
            console.print("  [bold cyan]▸ you are here[/bold cyan]")

    if cursor_after == 0:
        console.print("  [bold cyan]▸ you are here[/bold cyan] [dim](everything undone)[/dim]")

    console.print("\n[dim]opendot undo           revert the last applied action[/dim]")
    if undone:
        console.print(
            f"[dim]opendot redo           re-apply the next of {undone} undone action(s)[/dim]"
        )
    console.print("[dim]opendot undo <id>      restore the workspace to before that action[/dim]")


def _cmd_undo(workdir: str, snap_id: str | None) -> None:
    """`opendot undo [id]` — restore the workspace."""
    from opendot.reversibility.engine import Reversibility
    from opendot.reversibility.rules import load_rules

    rev = Reversibility(workdir=workdir, rules=load_rules(workdir))
    entries = rev.history()
    if not entries:
        console.print("[dim]nothing to undo[/dim]")
        return
    if snap_id:
        target = next((e for e in entries if e.id == snap_id), None)
        if target is None:
            console.print(f"[red]no action with id {snap_id}[/red]  (see: opendot log)")
            return
        if not target.snapshot_before:
            console.print(f"[yellow]action {snap_id} has no snapshot to undo[/yellow]")
            return
        changed_locks = rev.restore_to(target.snapshot_before)
        # An explicit jump leaves the undo/redo walk, so the cursor no longer
        # describes where the workspace is.
        rev.clear_redo()
        console.print(
            f"[green]restored[/green] workspace to before action {snap_id} ({target.kind}: {target.detail[:50]})"
        )
        _note_lockfiles(console, changed_locks)
    else:
        undone = rev.undo_last()
        if undone is None:
            console.print(
                "[yellow]nothing to undo[/yellow] (or the last action wasn't snapshotted)"
            )
            return
        console.print(f"[green]undid[/green] last action ({undone.kind}: {undone.detail[:50]})")
        console.print("[dim]run `opendot redo` to put it back[/dim]")
        _note_lockfiles(console, rev.last_changed_lockfiles)


def _cmd_redo(workdir: str) -> None:
    """`opendot redo` — re-apply the action the last undo reverted."""
    from opendot.reversibility.engine import Reversibility
    from opendot.reversibility.rules import load_rules

    rev = Reversibility(workdir=workdir, rules=load_rules(workdir))
    redone = rev.redo()
    if redone is None:
        console.print("[dim]nothing to redo[/dim]")
        return
    console.print(f"[green]redid[/green] action ({redone.kind}: {redone.detail[:50]})")
    _note_lockfiles(console, rev.last_changed_lockfiles)


def _cmd_diff(workdir: str, snap_id: str) -> None:
    """`opendot diff <id>` — show what `opendot undo <id>` would change."""
    from opendot.reversibility.engine import Reversibility
    from opendot.reversibility.rules import load_rules

    rev = Reversibility(workdir=workdir, rules=load_rules(workdir))
    entries = rev.history()
    if not entries:
        console.print("[dim]no actions recorded yet[/dim]")
        return
    target = next((e for e in entries if e.id == snap_id), None)
    if target is None:
        console.print(f"[red]no action with id {snap_id}[/red]  (see: opendot log)")
        return
    if not target.snapshot_before:
        console.print(f"[yellow]action {snap_id} has no snapshot to diff[/yellow]")
        return

    delta = rev.diff_to(target.snapshot_before)
    console.print(f"[bold]diff for {snap_id}[/bold] ({target.kind}: {target.detail[:50]})")

    if not (delta["added"] or delta["removed"] or delta["modified"]):
        console.print("[dim]workspace already matches the snapshot[/dim]")
        return

    for path in delta["added"]:
        console.print(f"  [green]+[/green] {path}  (would be created)")
    for path in delta["removed"]:
        console.print(f"  [red]-[/red] {path}  (would be deleted)")
    for item in delta["modified"]:
        path = item["path"]
        console.print(f"  [yellow]~[/yellow] {path}  (content differs)")
        diff_text = item.get("unified_diff")
        if diff_text:
            # markup=False so diff content containing [..] isn't parsed as Rich markup.
            console.print(diff_text, highlight=False, markup=False)


def _note_lockfiles(console, changed: list[str]) -> None:
    """`opendot undo` is a non-interactive command with no model in the loop, so
    it states the fact plainly: a lockfile was rolled back, meaning the installed
    environment no longer matches until the package manager is re-run. (In the
    interactive TUI the agent narrates this in its own words instead.)"""
    if not changed:
        return
    names = ", ".join(changed)
    console.print(
        f"[bold yellow]note:[/bold yellow] lockfile(s) rolled back ([bold]{names}[/bold]) — "
        f"installed packages are unchanged; re-run your package manager to match."
    )


def _cmd_mcp(args) -> None:
    """`opendot mcp add|list|remove|test` — manage external MCP servers."""
    from opendot.mcp import (
        MCPManager,
        add_mcp_server,
        load_mcp_config,
        remove_mcp_server,
    )

    cmd = getattr(args, "mcp_command", None)

    if cmd == "list" or cmd is None:
        servers = load_mcp_config()
        if not servers:
            console.print("[dim]no MCP servers configured.[/dim]")
            console.print(
                "[dim]add one:  opendot mcp add <name> -- <command> [args...]   (or --url <url>)[/dim]"
            )
            return
        console.print("[bold]MCP servers[/bold] (from ~/.opendot/mcp.json)\n")
        for name, spec in servers.items():
            if spec.get("url"):
                target = spec["url"]
            else:
                target = " ".join([spec.get("command", "")] + spec.get("args", []))
            console.print(f"  [cyan]{name}[/cyan]  {target}")
        return

    if cmd == "add":
        env = {}
        for pair in args.env:
            if "=" in pair:
                k, v = pair.split("=", 1)
                env[k] = v
        if args.url:
            spec = {"url": args.url}
            if getattr(args, "oauth", False):
                spec["auth"] = "oauth"  # browser-OAuth; opendot obtains tokens itself
            headers = {}
            for pair in getattr(args, "header", []):
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    headers[k.strip()] = v.strip()
            if headers and "auth" not in spec:
                spec["headers"] = headers  # e.g. Authorization=Bearer <token>
        else:
            cmd = list(getattr(args, "post_dashdash", []))
            if not cmd:
                console.print(
                    "[red]provide a launch command after `--`, or use --url for a remote server[/red]"
                )
                return
            spec = {"command": cmd[0]}
            if len(cmd) > 1:
                spec["args"] = cmd[1:]
        if env:
            spec["env"] = env
        add_mcp_server(args.name, spec)
        if spec.get("auth") == "oauth":
            from opendot.mcp import authorize_oauth_server

            console.print(f"opening your browser to authorize [cyan]{args.name}[/cyan]…")
            res = authorize_oauth_server(args.name, spec)
            if res.ok:
                console.print(
                    f"[green]✓ authorized[/green] [cyan]{args.name}[/cyan] — "
                    f"{res.tool_count} tools. They load next time you run opendot."
                )
            else:
                console.print(
                    f"[red]✗ authorization failed[/red] — {res.error}. "
                    f"The server is saved; retry with `opendot mcp test {args.name}`."
                )
            return
        console.print(
            f"[green]added[/green] MCP server [cyan]{args.name}[/cyan]. It will connect next time you run opendot."
        )
        return

    if cmd == "test":
        servers = load_mcp_config()
        spec = servers.get(args.name)
        if spec is None:
            console.print(f"[red]no MCP server named {args.name!r}.[/red]")
            return
        manager = MCPManager({args.name: spec})
        try:
            manager.start()
            if args.name in manager.connected:
                names = [tool.name for tool in manager.tools if tool.server == args.name]
                suffix = f": {', '.join(names)}" if names else ""
                tool_label = "tool" if len(names) == 1 else "tools"
                console.print(f"[green]✓ connected[/green] — {len(names)} {tool_label}{suffix}")
            else:
                error = manager.errors.get(args.name, "connection did not complete")
                console.print(f"[red]✗ connection failed[/red] — {error}")
        finally:
            manager.shutdown()
        return

    if cmd == "remove":
        if remove_mcp_server(args.name):
            console.print(f"[green]removed[/green] MCP server [cyan]{args.name}[/cyan].")
        else:
            console.print(f"[dim]no MCP server named {args.name!r}.[/dim]")
        return


def _print_session_summary(agent: Agent, elapsed_s: float) -> None:
    """End-of-session card: time, spend, and how much of what the agent did is
    reversible. Skipped when the session did nothing (no actions, no spend)."""
    s = agent.session_summary()
    if not s["actions"] and not s["cost_usd"] and not s["total_tokens"]:
        return

    took = f"{elapsed_s:.0f}s" if elapsed_s >= 1 else f"{elapsed_s * 1000:.0f}ms"
    tokens = (
        f"{s['total_tokens'] / 1000:.1f}k" if s["total_tokens"] >= 1000 else str(s["total_tokens"])
    )
    line1 = f"{took}  ·  ${s['cost_usd']:.4f}  ·  {tokens} tokens  ·  {s['calls']} call(s)"

    if s["actions"] == 0:
        line2 = "no files or commands touched"
    elif s["irreversible"] == 0:
        line2 = f"{s['actions']} action(s) · [green]all reversible[/green]"
    else:
        line2 = (
            f"{s['actions']} action(s) · {s['reversible']} reversible · "
            f"[yellow]{s['irreversible']} not undoable[/yellow]"
        )

    console.print(
        Panel.fit(f"{line1}\n{line2}", title="session", border_style="dim", title_align="left")
    )


async def _run_turn(agent: Agent, message: str) -> None:
    """Run one turn, streaming events to the console live.

    Reasoning ("thinking") streams dimmed; the answer streams in normal weight;
    tool activity prints as it happens. Text is streamed token-by-token.
    """
    mode = None  # track what we're currently printing: "thinking" | "text" | None

    def _switch(new: str, label: str = "") -> None:
        nonlocal mode
        if mode != new:
            if mode is not None:
                console.print()  # newline between phases
            if label:
                console.print(label)
            mode = new

    async for ev in agent.run(message):
        if ev.type == "thinking":
            _switch("thinking", "[dim italic]thinking…[/dim italic]")
            console.print(f"[dim]{ev.text}[/dim]", end="", markup=False, soft_wrap=True)
        elif ev.type == "text":
            _switch("text")
            console.print(ev.text, end="", markup=False, soft_wrap=True)
        elif ev.type == "tool_start":
            _switch("tool")
            mode = None  # tool output isn't a text phase
            arg_preview = ", ".join(f"{k}={v!r}"[:60] for k, v in ev.args.items())
            console.print(f"[cyan]▸[/cyan] [bold]{ev.tool}[/bold][dim]({arg_preview})[/dim]")
        elif ev.type == "tool_end":
            first = (ev.result.strip().splitlines() or ["(done)"])[0]
            console.print(f"  [dim]{first[:100]}[/dim]")
        elif ev.type == "final":
            if mode is not None:
                console.print()
        elif ev.type == "error":
            if mode is not None:
                console.print()
            console.print(f"[bold red]error:[/bold red] {ev.text}")
            mode = None


def _interactive(agent: Agent) -> None:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import InMemoryHistory

    console.print(
        Panel.fit(
            f"[bold]opendot[/bold] v{__version__}  ·  model: [cyan]{agent.config.model}[/cyan]\n"
            f"working in [dim]{agent.config.workdir}[/dim]\n"
            "Type a message. /help for commands, exit to quit.",
            border_style="cyan",
        )
    )
    session: PromptSession = PromptSession(history=InMemoryHistory())
    started = time.monotonic()

    while True:
        try:
            text = session.prompt("\nopendot › ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]bye[/dim]")
            _print_session_summary(agent, time.monotonic() - started)
            return

        if not text:
            continue
        low = text.lower()
        if low in {"exit", "/exit", "/quit", "quit"}:
            console.print("[dim]bye[/dim]")
            _print_session_summary(agent, time.monotonic() - started)
            return
        if low == "/help":
            console.print(SLASH_HELP)
            continue
        if low == "/clear":
            agent.reset()
            console.print("[dim]context cleared[/dim]")
            continue
        if low == "/save":
            try:
                agent.save_session()
                console.print("[dim]session saved[/dim]")
            except OSError as exc:
                console.print(f"[red]could not save session:[/red] {exc}")
            continue
        if low == "/resume":
            if agent.load_session():
                console.print(
                    f"[dim]session resumed: {len(agent.messages) - 1} message(s), "
                    f"model {agent.config.model}[/dim]"
                )
            else:
                console.print("[dim]no valid saved session for this project[/dim]")
            continue
        if low == "/compact":
            dropped = agent.compact()
            console.print(f"[dim]compacted: dropped {dropped} old message(s)[/dim]")
            continue
        if low.startswith("/model"):
            parts = text.split(maxsplit=1)
            if len(parts) > 1:
                agent.config.model = parts[1].strip()
                console.print(f"model → [cyan]{agent.config.model}[/cyan]")
                # A local api_base needs no provider key, so skip the hint here
                # too (mirrors the startup guard in main()).
                if not agent.config.api_base:
                    _warn_if_missing_key(agent.config.model)
            else:
                console.print(
                    f"model: [cyan]{agent.config.model}[/cyan]  ([dim]/model <id> to change[/dim])"
                )
            continue
        if low.startswith("/provider") or low.startswith("/connect"):
            from opendot.providers import register_key

            parts = text.split()
            if len(parts) == 3:
                register_key(parts[1], parts[2])
                console.print(
                    f"[green]✓[/green] set {parts[1]} for this session — "
                    f"persist with [dim]export {parts[1]}=…[/dim]"
                )
            else:
                console.print(
                    "usage: [dim]/provider <ENV_VAR> <api-key>[/dim]  "
                    "(the TUI has an interactive picker)"
                )
            continue
        if low == "/log":
            _cmd_log(agent.config.workdir)
            continue
        if low == "/trace":
            for line in agent.usage.trace_lines():
                console.print(f"[dim]{line}[/dim]")
            continue
        if low.startswith("/diff"):
            parts = text.split(maxsplit=1)
            if len(parts) > 1:
                _cmd_diff(agent.config.workdir, parts[1].strip())
            else:
                console.print("[dim]usage: /diff <id>  (see /log for ids)[/dim]")
            continue
        if low.startswith("/redo"):  # before /undo: neither prefixes the other,
            _cmd_redo(agent.config.workdir)  # but keep them adjacent and explicit
            continue
        if low.startswith("/undo"):
            parts = text.split(maxsplit=1)
            _cmd_undo(agent.config.workdir, parts[1].strip() if len(parts) > 1 else None)
            continue
        if low.startswith("/"):
            console.print(f"[dim]unknown command {text!r} — /help for the list[/dim]")
            continue

        try:
            asyncio.run(_run_turn(agent, text))
        except KeyboardInterrupt:
            console.print("\n[dim]interrupted[/dim]")


def main() -> None:
    # Split off a launch command after `--` (for `opendot mcp add NAME -- cmd ...`)
    # before argparse, so the command's own flags aren't parsed by opendot.
    post_dashdash: list[str] = []
    argv = sys.argv[1:]
    if "--" in argv:
        i = argv.index("--")
        argv, post_dashdash = argv[:i], argv[i + 1 :]

    parser = argparse.ArgumentParser(
        prog="opendot",
        description="An interactive terminal AI agent you can fully undo.",
    )
    parser.add_argument("-p", "--prompt", help="Run a single task and exit (one-shot mode).")
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENDOT_MODEL", "gpt-5.1"),
        help="Model id (any LiteLLM model, e.g. gpt-5.1, claude-opus-4-5, ollama/qwen3).",
    )
    parser.add_argument(
        "--api-base",
        default=os.environ.get("OPENAI_API_BASE"),
        help="Base URL of an OpenAI-compatible server "
        "(llama.cpp/llama-server, vLLM, LM Studio). "
        "Use with --model openai/<name>.",
    )
    parser.add_argument(
        "-C", "--dir", default=os.getcwd(), help="Working directory (default: cwd)."
    )
    parser.add_argument(
        "--repl",
        action="store_true",
        help="Use the plain REPL instead of the full-screen TUI.",
    )
    parser.add_argument(
        "--usd",
        default=None,
        type=float,
        metavar="DOLLARS",
        help="Hard spend cap for this agent (stops after exceeding). "
        "Also controlled by OPENDOT_MAX_USD.",
    )
    parser.add_argument(
        "--tokens",
        default=None,
        type=int,
        metavar="N",
        help="Hard token cap for this agent (stops after exceeding). "
        "Also controlled by OPENDOT_MAX_TOKENS.",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Auto-approve actions that would otherwise prompt for confirmation "
        "(for unattended / CI runs). Reversibility still snapshots everything; "
        "--deny patterns still block.",
    )
    parser.add_argument(
        "--allow",
        action="append",
        default=[],
        metavar="PATTERN",
        help="Auto-approve actions whose confirm prompt contains PATTERN "
        "(repeatable, e.g. --allow 'pytest').",
    )
    parser.add_argument(
        "--deny",
        action="append",
        default=[],
        metavar="PATTERN",
        help="Always refuse actions whose confirm prompt contains PATTERN, even "
        "with --yes (repeatable, e.g. --deny 'git push').",
    )
    parser.add_argument(
        "--sandbox",
        action="store_true",
        help="Run a one-shot (-p) turn inside a container against a copy of the "
        "workspace, committing only the resulting diff back on success. Needs "
        "docker or podman; errors if neither is present (never falls back to a "
        "direct run). Kernel-enforced isolation for unattended runs.",
    )
    parser.add_argument(
        "--sandbox-image",
        default="python:3.12-slim",
        help="Container image for --sandbox (must have opendot installed, or "
        "install it in an entrypoint). Default: python:3.12-slim.",
    )
    parser.add_argument(
        "--sandbox-net",
        action="store_true",
        help="Allow network access inside the --sandbox container (off by default).",
    )
    parser.add_argument("--version", action="version", version=f"opendot {__version__}")

    sub = parser.add_subparsers(dest="command")
    p_log = sub.add_parser("log", help="Show the auditable history of actions opendot took.")
    p_log.add_argument(
        "--clear",
        action="store_true",
        help="Wipe this project's action ledger (discards undo history).",
    )
    p_undo = sub.add_parser("undo", help="Restore the workspace (last action, or to a given id).")
    p_undo.add_argument("id", nargs="?", help="Action id from `opendot log` (default: last).")
    sub.add_parser("redo", help="Re-apply the action `opendot undo` just reverted.")
    p_diff = sub.add_parser(
        "diff", help="Show what `opendot undo <id>` would change, without touching disk."
    )
    p_diff.add_argument("id", help="Action id from `opendot log`.")
    sub.add_parser("resume", help="Resume this project's saved conversation.")

    # opendot mcp add/list/remove
    p_mcp = sub.add_parser("mcp", help="Manage MCP servers opendot connects to.")
    mcp_sub = p_mcp.add_subparsers(dest="mcp_command")
    p_add = mcp_sub.add_parser(
        "add",
        help="Add an MCP server.",
        epilog="For a stdio server, put its launch command after `--`. "
        "For a remote server, pass --url instead.",
    )
    p_add.add_argument("name", help="A short name for the server.")
    p_add.add_argument("--url", help="A remote MCP server URL (http/sse) instead of a command.")
    p_add.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VAL",
        help="Environment variable for the server (repeatable).",
    )
    p_add.add_argument(
        "--header",
        action="append",
        default=[],
        metavar="KEY=VAL",
        help="HTTP header for a remote (--url) server, e.g. "
        "'Authorization=Bearer <token>' (repeatable).",
    )
    p_add.add_argument(
        "--oauth",
        action="store_true",
        help="Authorize a remote (--url) server via browser OAuth instead of a static "
        "token. Opens your browser to authorize when you add it.",
    )
    # The launch command (after `--`) is captured from argv in main(), not here,
    # so its own flags (e.g. -y) aren't parsed by argparse.
    mcp_sub.add_parser("list", help="List configured MCP servers.")
    p_rm = mcp_sub.add_parser("remove", help="Remove an MCP server.")
    p_rm.add_argument("name", help="Server name to remove.")
    p_test = mcp_sub.add_parser("test", help="Connect to one server and list its tools.")
    p_test.add_argument("name", help="Configured server name to test.")

    args = parser.parse_args(argv)
    args.post_dashdash = post_dashdash
    workdir = os.path.abspath(args.dir)

    # Reversibility subcommands (no model call needed).
    if args.command == "log":
        _cmd_log(workdir, clear=getattr(args, "clear", False))
        return
    if args.command == "undo":
        _cmd_undo(workdir, args.id)
        return
    if args.command == "redo":
        _cmd_redo(workdir)
        return
    if args.command == "diff":
        _cmd_diff(workdir, args.id)
        return
    if args.command == "mcp":
        _cmd_mcp(args)
        return

    # Everything below this point calls a model — hint if the key looks missing.
    if args.command != "resume" and not args.api_base:
        _warn_if_missing_key(args.model)

    # One-shot: -p flag, or piped stdin.
    oneshot = args.prompt
    if oneshot is None and not sys.stdin.isatty():
        oneshot = sys.stdin.read().strip() or None

    # --sandbox only makes sense for a one-shot run (a headless-container TUI/REPL
    # isn't useful). Fail closed rather than silently running un-isolated — a user
    # who asked for the sandbox must not get a direct run thinking they're isolated.
    if getattr(args, "sandbox", False) and not oneshot:
        console.print(
            "[bold red]sandbox:[/bold red] --sandbox requires a one-shot prompt "
            "(-p '...'); it can't isolate an interactive session. Run with -p, or "
            "drop --sandbox."
        )
        raise SystemExit(2)

    if oneshot and getattr(args, "sandbox", False):
        # Kernel-isolated run: execute the turn inside a container against a copy
        # of the workspace, then commit the diff back. Fails closed (errors) if no
        # container runtime — never silently runs directly.
        from opendot import sandbox

        # --api-base names a server the *container* must reach. With networking
        # off it can reach nothing, and a loopback address means the container
        # itself rather than the host, so warn instead of failing obscurely.
        _sbx_api_base = getattr(args, "api_base", None)
        if _sbx_api_base:
            if not args.sandbox_net:
                console.print(
                    "[yellow]sandbox: --api-base is set but the container has no network "
                    "(--network none). Add --sandbox-net, or drop --api-base.[/yellow]"
                )
            elif any(h in _sbx_api_base for h in ("//localhost", "//127.0.0.1", "//[::1]")):
                console.print(
                    "[yellow]sandbox: --api-base points at loopback, which inside the "
                    "container is the container itself, not your host. Use an address the "
                    "container can reach (e.g. host.docker.internal).[/yellow]"
                )

        try:
            result = sandbox.run_sandboxed(
                workdir,
                oneshot,
                args.model,
                image=args.sandbox_image,
                network=args.sandbox_net,
                env_keys=_forwarded_env_keys(args.model),
                # Forward policy/budget flags: a hard-blocked --deny pattern must
                # not become auto-approved just because the run is sandboxed.
                deny=list(getattr(args, "deny", []) or []),
                usd=getattr(args, "usd", None),
                tokens=getattr(args, "tokens", None),
                api_base=getattr(args, "api_base", None),
            )
        except sandbox.SandboxError as exc:
            console.print(f"[bold red]sandbox:[/bold red] {exc}")
            raise SystemExit(2) from exc
        n = len(result["changed"])
        console.print(
            f"[dim]sandbox ({result['runtime']}) exited {result['returncode']}; "
            f"committed {n} changed file(s) back to the workspace[/dim]"
        )
        failed = result.get("failed") or []
        if failed:
            console.print(
                f"[yellow]sandbox: {len(failed)} file(s) could not be committed back "
                f"(check permissions): {', '.join(failed[:5])}"
                f"{'…' if len(failed) > 5 else ''}[/yellow]"
            )
        # Propagate the container's exit status so CI/scripting sees a failed run.
        # A clean container exit with a partial commit-back leaves the workspace in
        # an indeterminate state, so surface that as a failure too (exit 1).
        exit_code = int(result["returncode"])
        if exit_code == 0 and failed:
            exit_code = 1
        raise SystemExit(exit_code)

    if oneshot:
        # Non-interactive: can't prompt, so decline irreversible commands unless
        # a policy (--yes / --allow / OPENDOT.md) approves them.
        agent = _build_agent(
            args.model,
            workdir,
            confirm=_make_confirm(args, workdir, interactive=False),
            api_base=args.api_base,
            max_usd=args.usd,
            max_tokens=args.tokens,
        )
        if args.command == "resume":
            agent.load_session()
            if not args.api_base:
                _warn_if_missing_key(agent.config.model)
        started = time.monotonic()
        asyncio.run(_run_turn(agent, oneshot))
        _print_session_summary(agent, time.monotonic() - started)
    elif args.repl:
        agent = _build_agent(
            args.model,
            workdir,
            confirm=_make_confirm(args, workdir, interactive=True),
            api_base=args.api_base,
            max_usd=args.usd,
            max_tokens=args.tokens,
        )
        if args.command == "resume":
            if agent.load_session():
                console.print(
                    f"[dim]session resumed: {len(agent.messages) - 1} message(s), "
                    f"model {agent.config.model}[/dim]"
                )
            else:
                console.print("[dim]no valid saved session for this project; starting fresh[/dim]")
            if not args.api_base:
                _warn_if_missing_key(agent.config.model)
        _interactive(agent)
    else:
        # Default: the full-screen TUI. It installs its own confirm callback
        # (a blocking modal) on the agent's toolbox, so irreversible commands are
        # confirmed in-app. The placeholder here is replaced in OpendotTUI.__init__.
        from opendot.tui import run_tui

        agent = _build_agent(
            args.model,
            workdir,
            confirm=lambda _p: False,
            api_base=args.api_base,
            max_usd=args.usd,
            max_tokens=args.tokens,
        )
        if args.command == "resume":
            agent.load_session()
            # load_session may have changed the model; warn early like the other
            # paths so a missing key surfaces now, not mid-turn.
            if not args.api_base:
                _warn_if_missing_key(agent.config.model)
        run_tui(agent, policy=_build_policy(args, workdir))


if __name__ == "__main__":
    main()
