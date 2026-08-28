"""opendot's full-screen TUI (Textual) — the App itself.

Layout:
  ┌───────────────────────────────┬──────────────────┐
  │ transcript (streamed thinking, │  sidebar:        │
  │ tool activity, answers)        │   model          │
  │                                │   context/cost   │
  │                                │   ACTION LEDGER   │  <- the differentiator
  ├───────────────────────────────┴──────────────────┤
  │ > input                                            │
  └────────────────────────────────────────────────────┘

The sidebar leads with the reversibility ledger — every action, marked undoable
or irreversible — which is opendot's reason to exist. Ctrl+Z / the /undo
command walk it back live.
"""

from __future__ import annotations

from rich.markdown import Markdown
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Footer, Header, Input, Static

from opendot.agent.loop import Agent
from opendot.reversibility.ledger import LedgerEntry
from opendot.tui.commands import SLASH_COMMANDS
from opendot.tui.helpers import _LOGO_PATH, _render_tool_result, _row_bar
from opendot.tui.modals import ApiKeyModal, ConfirmModal, McpAddModal, SearchListModal
from opendot.tui.sidebar import Sidebar


def _resolve_action_id(entries: list[LedgerEntry], snap_id: str) -> LedgerEntry | None:
    """Resolve a displayed action ID without relying on a fixed-width suffix."""
    target = next((entry for entry in entries if entry.id == snap_id), None)
    if target is not None:
        return target

    try:
        numeric_id = int(snap_id)
    except (TypeError, ValueError):
        return None

    for entry in entries:
        try:
            if int(entry.id) == numeric_id:
                return entry
        except (TypeError, ValueError):
            continue
    return None


class OpendotTUI(App):
    ENABLE_COMMAND_PALETTE = True  # ctrl+p

    CSS = """
    Screen { layout: vertical; layers: base overlay; }
    #body { height: 1fr; }                     /* main column + sidebar fill the middle */
    #main { width: 3fr; height: 1fr; }         /* left column: transcript + input + mode */
    #transcript { height: 1fr; padding: 0 1 1 1; } /* pad bottom so the last message isn't glued to the input */
    #sidebar { width: 34; height: 1fr; padding: 1; border-left: solid $panel; }

    /* Welcome layout: small centered logo with the input right below it,
       the whole group centered in the screen (opencode-style). #main's
       `align: center middle` centers the children — no auto margins needed. */
    /* The logo wrapper is a full-width Center; hidden in the normal layout. */
    #welcome-wrap { display: none; }
    #welcome { width: auto; height: 6; }
    Screen.-welcome #transcript { display: none; }
    Screen.-welcome #sidebar { display: none; }
    /* Center the welcome group vertically. Safe to use middle-align because the
       command popup is a floating overlay (layer: overlay) — opening it doesn't
       change this column's height, so the logo never shifts. */
    Screen.-welcome #main { width: 1fr; height: 1fr; align: center middle; }
    /* All three welcome children share width:60% so #main's align centers them
       as one column. The logo image inside is centered by its Center wrapper. */
    Screen.-welcome #welcome-wrap {
        display: block; width: 60%; max-width: 90; height: auto; margin-bottom: 1;
    }
    Screen.-welcome #input { width: 70%; max-width: 100; margin: 0; }
    Screen.-welcome #modeline { width: 70%; max-width: 100; margin: 0; }

    /* Input: taller box, bounded by the left column so it never crosses the
       sidebar (opencode-style). Mode line sits just under it. */
    #input { height: 5; border: round $accent; margin: 0 1; padding: 0 1; }
    #modeline { height: 1; margin: 0 1; color: $text-muted; }

    /* Slash-command autocomplete popup — a FLOATING overlay on its own layer,
       so it renders on top of the logo/transcript without pushing them down.
       It's absolutely positioned each time it opens, just above the input
       (see _position_popup), matching the input's x and width. */
    #cmdpopup { layer: overlay; height: auto; max-height: 10; padding: 0;
                border: round $panel; background: $surface; }
    #cmdpopup > .option-list--option-highlighted { background: $accent; color: $text; }

    /* Message blocks — understated, opencode-style. The answer is the only
       block with real visual weight; everything else recedes. */
    .msg { margin: 1 0 0 0; }
    .user   { color: $text; border-left: solid $accent; padding: 0 1; }
    .think  { color: $text-muted; margin: 0 0 0 2; }
    .answer { color: $text; border-left: solid #2dd4bf; padding: 0 1; }
    .tool   { color: $text-muted; margin: 1 0 0 2; }
    .toolout{ color: $text-muted; margin: 0 0 0 4; }
    .err    { color: $error; text-style: bold; border-left: solid $error; padding: 0 1; }
    .sys    { color: $text-muted; text-style: italic; }
    """

    BINDINGS = [
        Binding("ctrl+z", "undo", "Undo last action"),
        # ctrl+shift+z is the editor convention but many terminals don't
        # distinguish it from ctrl+z; ctrl+y is the one that reliably arrives.
        Binding("ctrl+y", "redo", "Redo last undo"),
        Binding("ctrl+l", "log", "Show ledger note"),
        Binding("escape", "interrupt", "Interrupt", show=False),
        Binding("ctrl+c", "quit", "Quit"),
    ]

    def __init__(self, agent: Agent, policy=None) -> None:
        super().__init__()
        self.agent = agent
        self._turn_worker = None
        self._busy = False
        self._policy = policy
        # Give the agent a confirm callback that shows a blocking modal. It's
        # invoked from a worker thread (tool runs via asyncio.to_thread), so
        # call_from_thread is the correct, non-deadlocking bridge to the UI.
        # A permission policy (--yes / --allow / --deny / OPENDOT.md) can
        # auto-approve or hard-deny before the modal is ever shown.
        if policy is not None:
            agent.toolbox._confirm = policy.make_confirm(self._confirm_from_thread)
        else:
            agent.toolbox._confirm = self._confirm_from_thread

    def _confirm_from_thread(self, prompt: str) -> bool:
        try:
            return bool(self.call_from_thread(self.push_screen_wait, ConfirmModal(prompt)))
        except Exception:  # noqa: BLE001 - if anything goes wrong, fail safe (decline)
            return False

    def compose(self) -> ComposeResult:
        from textual.containers import Center
        from textual.widgets import OptionList

        yield Header(show_clock=False)
        with Horizontal(id="body"):
            # Left column: transcript fills, input + mode line pinned at its bottom.
            with Vertical(id="main"):
                # Welcome logo — the real PNG, shown until the first message.
                # textual-image auto-picks TGP/Sixel on capable terminals and
                # falls back to Unicode blocks elsewhere. If it can't load at
                # all, we degrade to a text wordmark. Wrapped in Center so the
                # auto-width image is reliably horizontally centred.
                with Center(id="welcome-wrap"):
                    yield self._welcome_widget()
                yield VerticalScroll(id="transcript")
                yield Input(placeholder='Ask opendot…   "fix broken tests"', id="input")
                yield Static(self._mode_line(), id="modeline")
            yield Sidebar(self.agent)
        # Slash-command autocomplete — a floating overlay (own layer) so it
        # renders ON TOP of the transcript/logo without pushing anything down.
        popup = OptionList(id="cmdpopup")
        popup.display = False
        popup.can_focus = False  # keep typing focus in the input
        yield popup
        yield Footer()

    def _welcome_widget(self):
        """The welcome logo. Real PNG via textual-image where supported, with a
        text wordmark fallback if the package/image can't load.

        The logo PNG is transparent; terminal image protocols flatten alpha onto
        a solid colour. So we pre-composite it onto the TUI's own background
        colour (#121212) — then the image's edges blend into the screen instead
        of showing a black box."""
        try:
            from textual_image.widget import Image

            if _LOGO_PATH.exists():
                return Image(self._logo_on_theme_bg(), id="welcome")
        except Exception:  # noqa: BLE001 - any failure → text fallback
            pass
        return Static(Text("opendot", style="bold"), id="welcome")

    def _logo_on_theme_bg(self):
        """Load the transparent logo, trim its transparent margins (so the
        wordmark centres correctly), and flatten it onto the theme background so
        there's no visible box. Returns a PIL image (or the path on failure).

        ANSI themes report their background as an ANSI sentinel (no real RGB —
        the terminal owns it), so we can't pick a matching colour; in that case
        we leave the PNG transparent and let the image protocol composite it."""
        try:
            from PIL import Image as PILImage

            logo = PILImage.open(_LOGO_PATH).convert("RGBA")
            bbox = logo.getbbox()  # tight box around non-transparent pixels
            if bbox:
                logo = logo.crop(bbox)

            bg = self.screen.styles.background
            # ANSI-defaulted background: real colour unknown → don't composite.
            if bg is None or getattr(bg, "ansi", None) is not None:
                return logo
            canvas = PILImage.new("RGBA", logo.size, (bg.r, bg.g, bg.b, 255))
            canvas.alpha_composite(logo)
            return canvas.convert("RGB")
        except Exception:  # noqa: BLE001
            return str(_LOGO_PATH)

    def watch_theme(self, theme_name: str) -> None:
        """Re-composite the welcome logo when the theme changes, so its
        background keeps matching the (possibly light) screen. Only matters
        while the welcome screen is still up — it's gone after the first message.
        Recomposites in place (Image.image is settable) to avoid a widget swap."""

        def _rebuild():
            try:
                if not self.screen.has_class("-welcome"):
                    return
                w = self.query_one("#welcome")
                if hasattr(w, "image"):  # textual-image widget; text fallback has none
                    w.image = self._logo_on_theme_bg()
            except Exception:  # noqa: BLE001 - never let a theme change crash the UI
                pass

        # Defer: when watch_theme fires, screen.styles.background still holds the
        # OLD theme colour. Recompute after the refresh applies the new theme.
        self.call_after_refresh(_rebuild)

    def _mode_line(self):
        return Text.assemble(
            ("  esc ", "bold cyan"),
            ("interrupt", "dim"),
            ("  ·  ", "dim"),
            ("ctrl+p ", "bold cyan"),
            ("commands", "dim"),
        )

    def _dismiss_welcome(self) -> None:
        """Hide the welcome logo and reveal the normal transcript+sidebar layout.
        Called once, on the first user message."""
        if self.screen.has_class("-welcome"):
            self.screen.remove_class("-welcome")

    def on_mount(self) -> None:
        self.title = "opendot"
        self.sub_title = self.agent.config.workdir
        # Drop the ANSI themes — their background is the terminal's (unknown to
        # us), so the welcome logo can't be composited to match them.
        for _t in ("ansi-dark", "ansi-light"):
            try:
                self.unregister_theme(_t)
            except Exception:  # noqa: BLE001
                pass
        # Start on the welcome screen (logo only); first message reveals the rest.
        self.screen.add_class("-welcome")
        self.query_one("#input", Input).focus()
        if len(getattr(self.agent, "messages", [])) > 1:
            self._dismiss_welcome()
            self._write(
                f"session resumed: {len(self.agent.messages) - 1} message(s), "
                f"model {self.agent.config.model}",
                "sys",
            )

    # -- transcript helpers --
    def _write(self, renderable, cls: str = "") -> None:
        w = Static(renderable, classes=f"msg {cls}".strip())
        self.query_one("#transcript", VerticalScroll).mount(w)
        self.call_after_refresh(self._scroll_end)

    def _clear_transcript(self) -> None:
        """Wipe the on-screen transcript (like clearing a terminal). Does not
        touch the conversation/context — that's what /clear adds."""
        for w in self.query("#transcript > .msg"):
            w.remove()

    def _scroll_end(self) -> None:
        self.query_one("#transcript", VerticalScroll).scroll_end(animate=False)

    def _refresh_sidebar(self) -> None:
        self.query_one(Sidebar).refresh()

    # -- slash-command autocomplete --
    def _popup(self):
        from textual.widgets import OptionList

        return self.query_one("#cmdpopup", OptionList)

    @property
    def _popup_open(self) -> bool:
        return self._popup().display

    def _matches(self, text: str) -> list[tuple[str, str]]:
        """Commands matching the current input. Active only while the line is a
        single '/word' with no space yet (i.e. still choosing a command)."""
        if not text.startswith("/") or " " in text:
            return []
        q = text[1:].lower()
        return [(n, d) for n, d in SLASH_COMMANDS if n[1:].lower().startswith(q)]

    def _sync_popup(self, text: str) -> None:
        from textual.widgets.option_list import Option

        matches = self._matches(text)
        popup = self._popup()
        if not matches:
            popup.display = False
            return
        popup.display = True
        popup.clear_options()
        # Use the expanding-grid row so the description right-aligns to the popup's
        # ACTUAL width at render time — no manual padding that can overshoot the
        # width and wrap the description onto a second line.
        for name, desc in matches:
            popup.add_option(
                Option(_row_bar(name, desc, right_style="dim", left_style="bold"), id=name)
            )
        popup.highlighted = 0
        # Float it just above the input (after layout settles so heights are known).
        self.call_after_refresh(self._position_popup)

    def _position_popup(self) -> None:
        """Place the floating popup directly above the input, matching its x and
        width. Runs after refresh so the input region and popup height are known."""
        try:
            popup = self._popup()
            if not popup.display:
                return
            inp = self.query_one("#input", Input)
            ir = inp.region
            # Match the input's outer box. The popup's round border adds 2 cells
            # of width, so set content width to the input width minus the border.
            popup.styles.width = max(10, ir.width - 2)
            popup.styles.height = "auto"
            # Sit its bottom flush with the top of the input.
            top = max(0, ir.y - popup.outer_size.height)
            popup.styles.offset = (ir.x, top)
        except Exception:  # noqa: BLE001
            pass

    def _highlighted_command(self) -> str | None:
        popup = self._popup()
        if popup.highlighted is None:
            return None
        return self._popup().get_option_at_index(popup.highlighted).id

    def _accept_popup(self, *, run: bool) -> None:
        """Pick the highlighted command. If ``run``, execute it immediately
        (Enter); otherwise just complete it into the input so the user can add
        an argument (Tab), e.g. `/undo 4` or `/model gpt-5.1`."""
        name = self._highlighted_command()
        if name is None:
            return
        inp = self.query_one("#input", Input)
        self._popup().display = False
        if run:
            inp.value = ""
            self._slash(name)
        else:
            inp.value = name + " "
            inp.cursor_position = len(inp.value)

    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "input":
            self._sync_popup(event.value)

    def on_key(self, event) -> None:
        """Drive the autocomplete popup from the keyboard while it's open."""
        if not self._popup_open:
            return
        popup = self._popup()
        if event.key == "down":
            popup.action_cursor_down()
            event.stop()
            event.prevent_default()
        elif event.key == "up":
            popup.action_cursor_up()
            event.stop()
            event.prevent_default()
        elif event.key == "tab":
            self._accept_popup(run=False)
            event.stop()
            event.prevent_default()
        elif event.key == "escape":
            popup.display = False
            event.stop()
            event.prevent_default()

    # -- input handling --
    async def on_input_submitted(self, event: Input.Submitted) -> None:
        # Only the main prompt submits chat. A submit from any other Input
        # (e.g. a modal's password/search field) must never become a message —
        # modals also .stop() it, this is defence in depth so secrets can't leak.
        if event.input.id != "input":
            return
        # If the popup is open, Enter runs the highlighted command immediately.
        if self._popup_open:
            self._accept_popup(run=True)
            return
        text = event.value.strip()
        self.query_one("#input", Input).value = ""
        if not text or self._busy:
            return

        if text.lower() in {"exit", "quit", "/exit", "/quit"}:
            self.exit()
            return
        # Bare `clear`/`cls` = clear the screen (transcript), like a terminal —
        # not a shell command and not a conversation reset. `/clear` does both.
        if text.lower() in {"clear", "cls"}:
            self._clear_transcript()
            return
        if text.startswith("/"):
            self._slash(text)
            return

        # First real message leaves the welcome screen for the full layout.
        self._dismiss_welcome()

        # Header: system username + local time, then the message.
        import datetime
        import getpass

        try:
            who = getpass.getuser()
        except Exception:  # noqa: BLE001
            who = "you"
        now = datetime.datetime.now().strftime("%H:%M")
        header = Text.assemble((who, "dim"), (f"  {now}", "dim"))
        self._write(Text.assemble(header, "\n", (text, "")), "user")
        # Use the message as the sidebar's task title (first ~40 chars).
        sb = self.query_one(Sidebar)
        sb.task_title = text[:40] + ("…" if len(text) > 40 else "")
        sb.refresh()

        # If the model's API key isn't set, guide the user instead of letting the
        # turn fail with a raw provider auth error.
        if self._missing_key_hint():
            return

        self._busy = True
        self._turn_worker = self.run_worker(self._run_turn(text), exclusive=True)

    def _missing_key_hint(self) -> bool:
        """If the current model needs an API key that isn't set, write a friendly
        hint to the transcript and return True (so the caller skips the turn)."""
        import os

        if getattr(self.agent.config, "api_base", None):
            return False

        from opendot.providers import env_var_for

        var = env_var_for(self.agent.config.model)
        if not var or os.environ.get(var):
            return False
        self._write(
            Text.assemble(
                ("no API key for ", "yellow"),
                (self.agent.config.model, "bold yellow"),
                ("\n", ""),
                (f"Set {var}, or run ", "dim"),
                ("/provider", "bold"),
                (" to paste one. Keys: github.com/vedaant00/opendot#any-model", "dim"),
            ),
            "err",
        )
        return True

    def _slash(self, text: str) -> None:
        cmd, _, rest = text[1:].partition(" ")
        cmd = cmd.lower()
        a = self.agent
        if cmd == "help":
            self._write(
                "commands: /log /trace /diff <id> /undo [id] /redo /clear /save /resume "
                "/compact /model /provider /mcp /composio /help",
                "sys",
            )
        elif cmd == "clear":
            a.reset()
            self._clear_transcript()
            self._write("cleared — screen and conversation reset", "sys")
        elif cmd == "save":
            try:
                a.save_session()
                self._write("session saved", "sys")
            except OSError as exc:
                self._write(f"could not save session: {exc}", "err")
        elif cmd == "resume":
            if a.load_session():
                self._write(
                    f"session resumed: {len(a.messages) - 1} message(s), model {a.config.model}",
                    "sys",
                )
                self._refresh_sidebar()
            else:
                self._write("no valid saved session for this project", "sys")
        elif cmd == "compact":
            n = a.compact()
            self._write(f"compacted: dropped {n} old message(s)", "sys")
        elif cmd == "model":
            # A bare arg sets the model directly; otherwise open the picker.
            if rest.strip():
                self._set_model(rest.strip())
            else:
                self.run_worker(self._pick_model(), exclusive=False)
        elif cmd in ("provider", "connect"):
            self.run_worker(self._connect_provider(), exclusive=False)
        elif cmd == "mcp":
            self.run_worker(self._manage_mcp(), exclusive=False)
        elif cmd == "composio":
            self.run_worker(self._manage_composio(), exclusive=False)
        elif cmd == "log":
            if rest.strip().lower() == "clear":
                self.run_worker(self._clear_log(), exclusive=False)
            else:
                self.action_log()
        elif cmd == "trace":
            self._write("\n".join(self.agent.usage.trace_lines()), "sys")
        elif cmd == "diff":
            self._do_diff(rest.strip() or None)
        elif cmd == "undo":
            self._do_undo(rest.strip() or None)
        elif cmd == "redo":
            self._do_redo()
        else:
            self._write(f"unknown command: /{cmd}", "sys")

    # -- model / provider pickers --
    def _set_model(self, model: str) -> None:
        """Switch the live model and refresh the sidebar (which shows it)."""
        self.agent.config.model = model
        self._write(f"model → {model}", "sys")
        self._refresh_sidebar()
        # Nudge if the key for this model isn't set yet.
        import os

        from opendot.providers import env_var_for

        var = env_var_for(model)
        if var and not os.environ.get(var):
            self._write(f"note: {var} is not set — use /provider to connect.", "sys")

    async def _pick_model(self) -> None:
        import asyncio

        from opendot import catalog

        # Text chat models from LiteLLM's registry, grouped by provider.
        entries = await asyncio.to_thread(catalog.list_models)
        if not entries:
            self._write("model list unavailable; use  /model <id>  to set one directly.", "sys")
            return
        # name == model in the catalog, so show the model string once.
        items = [(e["model"], e["model"], e["provider"]) for e in entries]
        chosen = await self.push_screen_wait(SearchListModal("Select model", items))
        if chosen:
            self._set_model(chosen)

    async def _connect_provider(self) -> None:
        import asyncio

        from opendot.providers import connectable_providers, register_key

        # LiteLLM-routable providers that have text models (from the catalog).
        pairs = await asyncio.to_thread(connectable_providers)
        items = [(var, name, "Providers") for name, var in pairs]
        var = await self.push_screen_wait(SearchListModal("Connect a provider", items))
        if not var:
            return
        name = next((n for n, v in pairs if v == var), var)
        key = await self.push_screen_wait(ApiKeyModal(name, var))
        if not key:
            return
        register_key(var, key)
        self._write(f"✓ {name} connected for this session.", "sys")
        self._write(f"to persist, add to your shell:  export {var}=…", "sys")
        self._refresh_sidebar()

    async def _manage_mcp(self) -> None:
        import asyncio

        from opendot.mcp import (
            add_mcp_server,
            authorize_oauth_server,
            load_mcp_config,
            remove_mcp_server,
        )

        servers = load_mcp_config()
        mgr = getattr(self.agent, "mcp", None)
        connected = set(mgr.connected) if mgr else set()
        errors = dict(mgr.errors) if mgr else {}

        # Build the list: each server with a status glyph, then an Add entry.
        items: list[tuple[str, str, str]] = []
        for name, spec in servers.items():
            target = spec.get("url") or " ".join([spec.get("command", "")] + spec.get("args", []))
            if name in connected:
                n = sum(1 for mt in mgr.tools if mt.server == name)
                status = f"✓ {n} tools"
            elif name in errors:
                status = "✗ failed"
            else:
                status = "· not connected"
            items.append((f"server:{name}", f"{name}   {target[:40]}   {status}", "Servers"))
        items.append(("__add__", "➕ Add a server…", ""))

        chosen = await self.push_screen_wait(SearchListModal("MCP servers", items))

        if not chosen:
            return

        if chosen != "__add__":
            name = chosen.removeprefix("server:")

            actions = [
                ("remove", "🗑 Remove server", "Actions"),
                ("keep", "Keep", "Actions"),
            ]

            action = await self.push_screen_wait(SearchListModal(f"MCP server: {name}", actions))

            if action == "remove":
                if remove_mcp_server(name):
                    # Drop it from the live session too, so it disappears from the
                    # sidebar and its tools stop being callable without a restart.
                    if mgr is not None:
                        mgr.forget_server(name)
                    if self.agent.toolbox is not None:
                        self.agent.toolbox.forget_mcp_server(name)
                    self._write(f"✓ removed MCP server '{name}'.", "sys")
                    self._refresh_sidebar()
                    await self._manage_mcp()
                else:
                    self._write(Text(f"No MCP server named '{name}'."), "err")

            return

        result = await self.push_screen_wait(McpAddModal())
        if not result:
            return

        name, spec = result["name"], result["spec"]
        add_mcp_server(name, spec)

        if spec.get("auth") == "oauth":
            # Authorize in the browser now so the server is usable without a restart.
            self._write(f"→ opening your browser to authorize '{name}'…", "sys")
            res = await asyncio.to_thread(authorize_oauth_server, name, spec)
            if res.ok:
                self._write(
                    f"✓ '{name}' authorized — {res.tool_count} tools. "
                    f"They load on next launch (restart opendot).",
                    "sys",
                )
            else:
                self._write(
                    f"✗ couldn't authorize '{name}': {res.error}. "
                    f"The server is saved; retry from /mcp.",
                    "sys",
                )
            return

        self._write(
            f"✓ added MCP server '{name}' — it connects on next launch (restart opendot).",
            "sys",
        )

    async def _enter_composio_key(self, cx) -> bool:
        """Prompt for a Composio API key, validate it, and store it only if it
        works. Returns True on a saved, working key. Can be called both on first
        setup and to replace a bad/expired key."""
        import asyncio

        key = await self.push_screen_wait(ApiKeyModal("Composio", "COMPOSIO_API_KEY"))
        if not key:
            return False
        self._write("checking key…", "sys")
        if not await asyncio.to_thread(cx.verify_api_key, key):
            self._write("that key didn't work — check it and run /composio again.", "sys")
            return False
        cx.set_api_key(key)
        return True

    async def _manage_composio(self) -> None:
        import asyncio
        import webbrowser

        from opendot.tools import composio as cx

        if not cx.composio_available():
            # composio is a core dep, so this only trips on a broken install.
            self._write("Composio isn't available — try reinstalling opendot.", "sys")
            return

        # First run (or no key yet): ask for the Composio API key.
        if not cx.is_configured():
            if not await self._enter_composio_key(cx):
                return
            self._write("✓ Composio connected. Run /composio again to browse apps.", "sys")
            self._refresh_sidebar()
            return

        # Configured: list apps (marking connected/enabled ones), let the user pick.
        self._write("loading Composio apps…", "sys")
        apps = await asyncio.to_thread(cx.list_apps)
        if not apps:
            # Most likely a bad/expired key (it was stored without validation in
            # older versions). Offer to re-enter it instead of dead-ending.
            self._write(
                "couldn't load Composio apps — your key may be invalid or expired.",
                "sys",
            )
            if await self._enter_composio_key(cx):
                apps = await asyncio.to_thread(cx.list_apps)
            if not apps:
                return
        connected = await asyncio.to_thread(cx.list_connected)
        enabled = set(cx.enabled_apps())

        # Sort enabled apps to the top so they're easy to find/disable.
        apps.sort(key=lambda a: (a["slug"] not in enabled, a["name"].lower()))
        items: list[tuple[str, str, str, str]] = []
        for app in apps:
            slug = app["slug"]
            status = (
                "✓ enabled" if slug in enabled else ("· connected" if slug in connected else "")
            )
            label = f"{app['name']}   {slug}"
            items.append((slug, label, "Apps", status))

        slug = await self.push_screen_wait(SearchListModal("Composio apps", items))
        if not slug:
            return

        # Already enabled → offer to disable it (remove from opendot's tool set).
        if slug in enabled:
            choice = await self.push_screen_wait(
                SearchListModal(
                    f"{slug} — enabled",
                    [
                        ("disable", f"Disconnect {slug}", "Manage"),
                        ("keep", "Keep it enabled", "Manage"),
                    ],
                )
            )
            if choice == "disable":
                revoked = await asyncio.to_thread(cx.disable_app, slug)
                note = "connection revoked" if revoked else "removed locally"
                self._write(
                    f"✓ {slug} disconnected ({note}). Its tools stop loading on "
                    f"next launch (restart opendot).",
                    "sys",
                )
                self._refresh_sidebar()
            return

        # Connect (OAuth → browser + wait, or direct connector → immediate).
        self._write(f"connecting {slug}…", "sys")
        res = await asyncio.to_thread(cx.begin_connect, slug)
        if res.error:
            self._write(f"couldn't connect {slug}: {res.error}", "sys")
            return
        if res.needs_auth and res.redirect_url:
            self._write(f"→ opening your browser to authorize {slug}…", "sys")
            webbrowser.open(res.redirect_url)
            self._write(f"  if it didn't open: {res.redirect_url}", "sys")
            try:
                await asyncio.to_thread(res.request.wait_for_connection, 180)
            except Exception as exc:  # noqa: BLE001
                self._write(f"authorization didn't complete: {exc}", "sys")
                return
        cx.add_enabled_app(slug)
        self._write(
            f"✓ {slug} connected and enabled. Its tools load on next launch (restart opendot).",
            "sys",
        )
        self._refresh_sidebar()

    async def _run_turn(self, message: str) -> None:
        mode = None
        buf: list[str] = []

        def flush_answer():
            if buf:
                from rich.console import Group

                self._write(
                    Group(Text("opendot", style="bold green"), Markdown("".join(buf))),
                    "answer",
                )
                buf.clear()

        try:
            async for ev in self.agent.run(message):
                if ev.type == "thinking":
                    if mode != "think":
                        flush_answer()
                        mode = "think"
                        self._write(Text("Thought", style="dim bold"), "think")
                    self._write(Text(ev.text.rstrip(), style="dim italic"), "think")
                elif ev.type == "text":
                    mode = "answer"
                    buf.append(ev.text)
                elif ev.type == "tool_start":
                    flush_answer()
                    mode = None
                    args = "  ".join(str(v)[:50] for v in ev.args.values())
                    line = Text("→ ", style="dim").append(ev.tool, style="bold")
                    if args:
                        line.append("  " + args, style="dim")
                    self._write(line, "tool")
                elif ev.type == "tool_end":
                    self._write(_render_tool_result(ev.tool, ev.result), "toolout")
                    self._refresh_sidebar()
                elif ev.type == "explorer_start":
                    flush_answer()
                    mode = None
                    self._write(
                        Text(f"⇉ explorer {ev.lane + 1}: {ev.text}", style="bold magenta"),
                        "tool",
                    )
                elif ev.type == "explorer_step":
                    self._write(
                        Text(f"    [{ev.lane + 1}] {ev.text}", style="magenta"),
                        "toolout",
                    )
                elif ev.type == "explorer_done":
                    first = (ev.text.strip().splitlines() or ["(done)"])[0]
                    self._write(
                        Text(f"    [{ev.lane + 1}] ✓ {first[:100]}", style="dim magenta"),
                        "toolout",
                    )
                elif ev.type == "error":
                    flush_answer()
                    mode = None
                    self._write(Text(ev.text), "err")
            flush_answer()
        finally:
            self._busy = False
            self._refresh_sidebar()

    # -- actions --
    def action_interrupt(self) -> None:
        """Esc: cancel the in-flight turn."""
        if self._busy and self._turn_worker is not None:
            self._turn_worker.cancel()
            self._busy = False
            self._write("interrupted", "sys")

    def action_undo(self) -> None:
        if not self._busy:
            self._do_undo(None)

    def action_redo(self) -> None:
        if not self._busy:
            self._do_redo()

    def _do_diff(self, snap_id: str | None) -> None:
        """Preview what `/undo <id>` would change, without touching the disk."""
        if not snap_id:
            self._write("usage: /diff <id>  (see /log for ids)", "sys")
            return
        rev = self.agent.reversibility
        entries = rev.history()
        target = _resolve_action_id(entries, snap_id)
        if not target:
            self._write(f"no action {snap_id} (see /log)", "sys")
            return
        if not target.snapshot_before:
            self._write(f"action {target.id} has no snapshot to diff", "sys")
            return
        delta = rev.diff_to(target.snapshot_before)
        if not (delta["added"] or delta["removed"] or delta["modified"]):
            self._write("workspace already matches the snapshot", "sys")
            return
        lines = [f"diff for {target.id} ({target.kind}):"]
        lines += [f"  + {p}  (would be created)" for p in delta["added"]]
        lines += [f"  - {p}  (would be deleted)" for p in delta["removed"]]
        lines += [f"  ~ {item['path']}  (content differs)" for item in delta["modified"]]
        self._write("\n".join(lines), "sys")

    def _do_undo(self, snap_id: str | None) -> None:
        rev = self.agent.reversibility
        entries = rev.history()
        if not entries:
            self._write("nothing to undo", "sys")
            return
        if snap_id:
            target = _resolve_action_id(entries, snap_id)
            if not target:
                self._write(f"no action {snap_id} (see /log)", "sys")
                return
            if not target.snapshot_before:
                self._write(f"action {target.id} has no snapshot to undo", "sys")
                return
            changed_locks = rev.restore_to(target.snapshot_before)
            # An explicit jump leaves the undo/redo walk; the cursor no longer
            # describes where the workspace is.
            rev.clear_redo()
            what = f"action {target.id} ({target.kind}: {target.detail.rsplit('/', 1)[-1]})"
        else:
            undone = rev.undo_last()
            if undone is None:
                self._write("nothing to undo (or last action wasn't snapshotted)", "sys")
                return
            changed_locks = rev.last_changed_lockfiles
            what = f"the last action ({undone.kind}: {undone.detail.rsplit('/', 1)[-1]})"

        self._write(f"↺ reverted {what}", "sys")  # immediate, deterministic ack
        self._refresh_sidebar()
        # Then the agent narrates the undo (and, per its system prompt, loudly
        # warns about environment drift when a lockfile changed). Detection is
        # done in code above; only the phrasing/help is left to the model.
        if self._busy:
            return  # a turn is already running; don't stack another
        note = f"[undo] Reverted {what}."
        if changed_locks:
            note += f" Changed lockfile(s): {', '.join(changed_locks)}."
        self._busy = True
        self._turn_worker = self.run_worker(self._run_turn(note), exclusive=True)

    def _do_redo(self) -> None:
        """Re-apply what /undo just reverted. Mirrors _do_undo's ack + narration."""
        rev = self.agent.reversibility
        redone = rev.redo()
        if redone is None:
            self._write("nothing to redo", "sys")
            return
        changed_locks = rev.last_changed_lockfiles
        what = f"the last undone action ({redone.kind}: {redone.detail.rsplit('/', 1)[-1]})"

        self._write(f"↻ reapplied {what}", "sys")  # immediate, deterministic ack
        self._refresh_sidebar()
        if self._busy:
            return  # a turn is already running; don't stack another
        note = f"[redo] Reapplied {what}."
        if changed_locks:
            note += f" Changed lockfile(s): {', '.join(changed_locks)}."
        self._busy = True
        self._turn_worker = self.run_worker(self._run_turn(note), exclusive=True)

    def action_log(self) -> None:
        history = self.agent.reversibility.history()
        if not history:
            self._write("no actions recorded", "sys")
            return
        t = Text()
        t.append("action history\n", style="bold")
        for e in history:
            mark = "↺" if e.reversible else "✗ irreversible"
            t.append(f"  {e.id}  {mark}  {e.kind}  {e.detail}\n", style="dim")
        self._write(t, "")

    async def _clear_log(self) -> None:
        """`/log clear` — wipe this project's action ledger after confirming
        (it discards the undo history, so gate it behind the confirm modal)."""
        rev = self.agent.reversibility
        if not rev.history():
            self._write("no actions to clear", "sys")
            return
        ok = await self.push_screen_wait(
            ConfirmModal(
                "Clear the action ledger for this project?\n"
                "This discards the undo history — past actions can no longer be undone."
            )
        )
        if not ok:
            return
        n = rev.clear_history()
        self._write(f"cleared {n} ledger entr{'y' if n == 1 else 'ies'}.", "sys")
        self._refresh_sidebar()


def run_tui(agent: Agent, policy=None) -> None:
    OpendotTUI(agent, policy=policy).run()
