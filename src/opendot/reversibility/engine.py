"""The reversibility engine — ties snapshots + ledger together.

This is the moat. Before any mutating action (a file write or a shell command),
``before_action`` snapshots the workspace and records a ledger entry tying that
action to its pre-state. ``undo`` / ``restore_to`` walk the workspace back to a
recorded point, exactly (byte-for-byte), and honestly (never touching ignored
paths, never claiming to reverse what it can't).

The classifier (which decides whether a shell command is reversible / needs
confirmation) lives in ``classifier.py`` and is consulted by the caller; this
module just records the ``reversible`` flag it's given.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from opendot.reversibility import ledger, redo, snapshots
from opendot.reversibility.ledger import ActionKind, LedgerEntry
from opendot.reversibility.snapshots import IgnoreRules, project_id_for


@dataclass
class Reversibility:
    workdir: str
    rules: IgnoreRules
    enabled: bool = True
    # The model + sampling params driving the session, stamped into every ledger
    # entry for a fully auditable trail. Set by the agent loop; empty for direct
    # CLI actions with no model behind them.
    model: str = ""
    params: dict = field(default_factory=dict)
    # Lockfiles changed by the most recent undo (see undo_last). Lets the UI warn
    # that the environment isn't restored until the package manager is re-run.
    last_changed_lockfiles: list[str] = field(default_factory=list)

    @property
    def project_id(self) -> str:
        return project_id_for(self.workdir)

    def before_action(
        self,
        kind: ActionKind,
        detail: str,
        *,
        reversible: bool = True,
        note: str = "",
        timestamp: str = "",
        snapshot: bool = True,
    ) -> str:
        """Snapshot the workspace and log the action about to happen.

        When ``snapshot`` is False, the action is still logged (for the audit
        trail) but no snapshot is taken — so nothing is captured into the store.
        Used for the OPENDOT_NO_SNAPSHOT escape hatch, where the whole point is to
        NOT keep a recoverable copy (e.g. shredding a secret). Such an action is
        inherently not reversible.

        Returns the snapshot id (== ledger entry id). No-op returns "" if disabled.
        """
        if not self.enabled:
            return ""
        # A new action branches the history: whatever we had undone is no longer
        # reachable, so the redo stack goes (standard editor semantics).
        redo.clear(self.project_id)
        if not snapshot:
            # No snapshot is taken, so derive a sortable id from the shared
            # action-id counter (snapshots + ledger both feed it), keeping ids
            # unique and monotonic even when snapshot and no-snapshot actions mix.
            entry_id = f"{snapshots.max_action_id(self.project_id) + 1:06d}"
            ledger.append(
                self.project_id,
                LedgerEntry(
                    id=entry_id,
                    kind=kind,
                    detail=detail,
                    snapshot_before="",  # no snapshot exists to restore
                    reversible=False,
                    note=note,
                    timestamp=timestamp,
                    model=self.model,
                    params=dict(self.params),
                ),
            )
            return entry_id
        snap = snapshots.take_snapshot(self.workdir, self.rules)
        # If files were too large to snapshot, this action can't be fully undone —
        # record that honestly in the ledger note.
        if snap.skipped_large:
            n = len(snap.skipped_large)
            big_note = f"{n} file(s) too large to snapshot — changes to them can't be undone"
            note = f"{note}; {big_note}" if note else big_note
        ledger.append(
            self.project_id,
            LedgerEntry(
                id=snap.id,
                kind=kind,
                detail=detail,
                snapshot_before=snap.id,
                reversible=reversible,
                note=note,
                timestamp=timestamp,
                model=self.model,
                params=dict(self.params),
            ),
        )
        return snap.id

    # -- history / undo --
    def history(self) -> list[LedgerEntry]:
        return ledger.read_all(self.project_id)

    def clear_history(self) -> int:
        """Clear this project's action ledger (discards undo history). Returns
        the number of entries removed."""
        redo.clear(self.project_id)
        return ledger.clear(self.project_id)

    def clear_redo(self) -> None:
        """Drop the redo stack. For callers that move the workspace outside the
        undo/redo walk (e.g. `opendot undo <id>` jumping to an arbitrary point),
        after which the cursor no longer describes where we are."""
        redo.clear(self.project_id)

    def redo_available(self) -> int:
        """How many actions could currently be re-applied."""
        return redo.read(self.project_id, len(self.history())).undone

    def restore_to(self, snapshot_id: str) -> list[str]:
        """Restore the workspace to the state captured in ``snapshot_id``.

        Returns any dependency lockfiles the restore changed — the caller should
        warn that the environment isn't restored until the package manager is
        re-run (restoring a lockfile rolls back declared versions, not installed
        packages).
        """
        snap = snapshots.load_snapshot(self.project_id, snapshot_id)
        return snapshots.restore_snapshot(snap, self.rules)

    def diff_to(self, snapshot_id: str) -> dict:
        """Dry-run comparison: what would change if we restored ``snapshot_id``?

        Returns a dict with ``added``, ``removed``, and ``modified`` keys describing
        the delta between the snapshot and the current workspace. This is read-only
        and never mutates the workspace.
        """
        snap = snapshots.load_snapshot(self.project_id, snapshot_id)
        return snapshots.diff_snapshot(snap, self.rules)

    def undo_last(self) -> LedgerEntry | None:
        """Revert the most recent not-yet-undone action (restore its before-snapshot).

        Repeated calls walk further back through the history rather than
        restoring the same snapshot again. Returns the entry that was undone, or
        None if there's nothing left to undo. Any dependency lockfiles the
        restore changed are left in ``last_changed_lockfiles`` for the caller to
        warn about.
        """
        self.last_changed_lockfiles = []
        entries = self.history()
        if not entries:
            return None
        state = redo.read(self.project_id, len(entries))
        if state.undone >= len(entries):
            return None
        target_idx = None
        for i in range(len(entries) - 1 - state.undone, -1, -1):
            if entries[i].snapshot_before:
                target_idx = i
                break
        if target_idx is None:
            return None
        target = entries[target_idx]

        if state.undone == 0:
            # The state we're leaving is the head, and no later action snapshotted
            # it. Capture it now or redo has nothing to come back to. Every other
            # "after" state is some other action's snapshot_before.
            state.head_snapshot = snapshots.take_snapshot(self.workdir, self.rules).id

        self.last_changed_lockfiles = self.restore_to(target.snapshot_before)
        state.undone = len(entries) - target_idx
        redo.write(self.project_id, state)
        return target

    def redo(self) -> LedgerEntry | None:
        """Re-apply the most recently undone action.

        Returns the entry that was re-applied, or None if there's nothing to
        redo. Like ``undo_last``, any lockfiles the restore changed are left in
        ``last_changed_lockfiles``.
        """
        self.last_changed_lockfiles = []
        entries = self.history()
        state = redo.read(self.project_id, len(entries))
        if not state.can_redo:
            return None

        target_index = len(entries) - state.undone
        target = entries[target_index]

        after = ""
        next_undone = 0
        for i in range(target_index + 1, len(entries)):
            if entries[i].snapshot_before:
                after = entries[i].snapshot_before
                next_undone = len(entries) - i
                break
        if not after:
            after = state.head_snapshot
            next_undone = 0

        if not after:
            return None

        self.last_changed_lockfiles = self.restore_to(after)
        state.undone = next_undone
        redo.write(self.project_id, state)
        return target
