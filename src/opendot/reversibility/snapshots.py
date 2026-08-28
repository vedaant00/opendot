"""Content-addressed snapshot store — the foundation of opendot's reversibility.

A snapshot captures the state of a workspace (a directory subtree) at a point in
time. Files are stored *once* by content hash under ``~/.opendot/objects/``, so
repeated snapshots are cheap: only changed files add new objects. A snapshot
itself is a small manifest (path -> hash) under
``~/.opendot/snapshots/<project>/<id>.json``.

Restoring a snapshot makes the workspace byte-for-byte match the manifest:
files are rewritten from their stored objects, and files that exist now but were
not in the snapshot are removed. This is the "walk it back" guarantee — and it
must be exact, so it is covered by round-trip tests.

Ignore rules keep snapshots small and relevant. Defaults skip VCS/dep/build
dirs; callers may override in both directions (see ``IgnoreRules``).
"""

from __future__ import annotations

import difflib
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Store location
# ---------------------------------------------------------------------------


def store_root() -> Path:
    """Root of the global opendot store (override with OPENDOT_HOME, for tests)."""
    root = Path(os.environ.get("OPENDOT_HOME", Path.home() / ".opendot"))
    return root


def _objects_dir() -> Path:
    d = store_root() / "objects"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _snapshots_dir(project_id: str) -> Path:
    d = store_root() / "snapshots" / project_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def project_id_for(workdir: str | Path) -> str:
    """Stable per-workspace id (hash of the absolute path)."""
    p = str(Path(workdir).resolve())
    return hashlib.sha256(p.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Ignore rules
# ---------------------------------------------------------------------------

_DEFAULT_IGNORE_DIRS = {
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    ".mypy_cache",
    ".pytest_cache",
    ".opendot",
    ".DS_Store",
}


@dataclass
class IgnoreRules:
    """What to skip when snapshotting.

    ``skip_dirs`` are directory *names* skipped anywhere in the tree. ``force_include``
    are directory names to snapshot even if they're in the default skip set — this
    is how OPENDOT.md's "snapshot venv/node_modules" override works. ``extra_skip``
    adds user-specified skips. ``force_include`` wins over skips.
    """

    force_include: set[str] = field(default_factory=set)
    extra_skip: set[str] = field(default_factory=set)

    def skipped(self, name: str) -> bool:
        if name in self.force_include:
            return False
        return name in (_DEFAULT_IGNORE_DIRS | self.extra_skip)


# ---------------------------------------------------------------------------
# Hashing / object storage
# ---------------------------------------------------------------------------

# Chunk size for streaming a file through the hash / into the store, so a large
# file is never loaded whole into memory.
_CHUNK = 1024 * 1024  # 1 MB


def _hash_file(path: Path) -> str:
    """SHA-256 of a file, read in chunks (never loads the whole file into RAM)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(_CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def _clone_or_copy(src: Path, dst: Path) -> None:
    """Copy src -> dst, preferring a copy-on-write clone (reflink) so identical
    blocks are shared until one side is written. Safe because store objects are
    immutable (never edited in place), and CoW means a later edit to the source
    file allocates new blocks rather than corrupting the stored object.

    Falls back to a plain streamed copy on filesystems without reflink support.
    """
    try:
        # Python 3.14+ / platforms with clonefile (APFS) or FICLONE (Btrfs/XFS)
        # honor copy-on-write here; elsewhere it's a normal copy.
        import shutil

        shutil.copyfile(src, dst)  # uses OS CoW fast-copy when available
    except OSError:
        with open(src, "rb") as fin, open(dst, "wb") as fout:
            for block in iter(lambda: fin.read(_CHUNK), b""):
                fout.write(block)


def _write_object_from_path(path: Path) -> str:
    """Store a file by content hash without loading it into memory; return the
    hash. Streams the hash, then clones/copies the bytes into the store."""
    h = _hash_file(path)
    obj = _objects_dir() / h
    if not obj.exists():
        tmp = obj.with_suffix(".tmp")
        _clone_or_copy(path, tmp)
        tmp.replace(obj)  # atomic
    return h


def _read_object(h: str) -> bytes:
    return (_objects_dir() / h).read_bytes()


# ---------------------------------------------------------------------------
# Snapshot / restore
# ---------------------------------------------------------------------------

# Files larger than this are NOT snapshotted (skipped), so a giant file can't
# make every snapshot slow and bloat the store. An action touching such a file
# is therefore not fully undoable — take_snapshot reports the skipped paths so
# the caller can say so.
_MAX_FILE_BYTES = 25 * 1024 * 1024  # 25 MB


@dataclass
class FileEntry:
    """One captured file: content hash, POSIX mode bits, and (mtime, size,
    ctime_ns) so a later snapshot can skip re-hashing an unchanged file. ctime_ns
    is part of the change signature so a same-size rewrite within one mtime tick
    is still detected."""

    h: str
    mode: int | None = None  # st_mode & 0o777, or None if unknown (old snapshots)
    mtime: float | None = None
    size: int | None = None
    ctime_ns: int | None = None  # change-time in nanoseconds, used to detect rewrites


@dataclass
class Snapshot:
    id: str
    project_id: str
    workdir: str
    files: dict[str, FileEntry]  # relative posix path -> FileEntry
    skipped_large: list[str] = field(default_factory=list)  # paths too big to capture


def _iter_files(workdir: Path, rules: IgnoreRules):
    """Yield files under workdir, honoring ignore rules (dir-name based)."""
    for dirpath, dirnames, filenames in os.walk(workdir):
        # prune skipped dirs in-place so os.walk doesn't descend
        dirnames[:] = [d for d in dirnames if not rules.skipped(d)]
        for fn in filenames:
            if rules.skipped(fn):
                continue
            full = Path(dirpath) / fn
            if full.is_symlink() or not full.is_file():
                continue
            yield full


def take_snapshot(workdir: str | Path, rules: IgnoreRules | None = None) -> Snapshot:
    """Capture the workspace state. Cheap when little changed (content-addressed).

    Files over ``_MAX_FILE_BYTES`` are skipped (recorded in ``skipped_large``) so
    one huge file can't make snapshots slow/bloated — those are not undoable.
    """
    rules = rules or IgnoreRules()
    wd = Path(workdir).resolve()
    pid = project_id_for(wd)

    # Previous snapshot's entries, keyed by path — used to skip re-hashing files
    # whose (mtime, size) are unchanged. This is what makes snapshots of large,
    # mostly-static workspaces near-instant.
    prev = _latest_snapshot_files(pid)

    files: dict[str, FileEntry] = {}
    skipped_large: list[str] = []
    for f in _iter_files(wd, rules):
        rel = f.relative_to(wd).as_posix()
        try:
            st = f.stat()
        except OSError:
            continue
        mode = st.st_mode & 0o777
        # Fast path: unchanged since the last snapshot -> reuse its hash, no read.
        # We compare (mtime, size, ctime_ns). ctime advances on every write even
        # when mtime is pinned to the same value, so including it prevents a
        # same-size rewrite inside one mtime tick from reusing a stale hash.
        old = prev.get(rel)
        if (
            old is not None
            and old.h
            and old.mtime == st.st_mtime
            and old.size == st.st_size
            and old.ctime_ns is not None
            and old.ctime_ns == st.st_ctime_ns
        ):
            files[rel] = FileEntry(
                h=old.h,
                mode=mode,
                mtime=st.st_mtime,
                size=st.st_size,
                ctime_ns=st.st_ctime_ns,
            )
            continue
        if st.st_size > _MAX_FILE_BYTES:
            skipped_large.append(rel)
            continue
        try:
            h = _write_object_from_path(f)  # streams: never loads the file whole
        except OSError:
            continue  # unreadable file: skip rather than fail the whole snapshot
        files[rel] = FileEntry(
            h=h, mode=mode, mtime=st.st_mtime, size=st.st_size, ctime_ns=st.st_ctime_ns
        )

    snap_id = _next_snapshot_id(pid)
    snap = Snapshot(
        id=snap_id,
        project_id=pid,
        workdir=str(wd),
        files=files,
        skipped_large=skipped_large,
    )
    _write_manifest(snap)

    # Bound disk growth: drop this project's oldest snapshots beyond the
    # retention limit, then delete any objects no longer referenced anywhere.
    if prune_project_snapshots(pid):
        gc_objects()
    return snap


def _latest_snapshot_files(project_id: str) -> dict[str, FileEntry]:
    """The file map of the most recent snapshot for this project, for the
    unchanged-file fast path. Empty if none / unreadable."""
    ids = list_snapshots(project_id)
    if not ids:
        return {}
    try:
        return load_snapshot(project_id, ids[-1]).files
    except Exception:  # noqa: BLE001
        return {}


# How many snapshots to keep per project before older ones are pruned. Bounds
# disk growth on long, edit-heavy sessions. Content is deduped across snapshots,
# so this is generous.
MAX_SNAPSHOTS_PER_PROJECT = 50


def _referenced_hashes() -> set[str]:
    """Every content hash referenced by ANY surviving snapshot, across ALL
    projects. Objects are shared globally (deduped), so GC must consider every
    project's manifests before deleting an object."""
    refs: set[str] = set()
    snaps_root = store_root() / "snapshots"
    if not snaps_root.exists():
        return refs
    for proj_dir in snaps_root.iterdir():
        if not proj_dir.is_dir():
            continue
        for manifest in proj_dir.glob("*.json"):
            try:
                d = json.loads(manifest.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                # Unreadable manifest → be conservative, keep its objects by
                # skipping (we simply can't enumerate them, so we won't GC them
                # blindly — but we also can't add refs; safest is to abort GC).
                raise
            for v in d.get("files", {}).values():
                h = v if isinstance(v, str) else v.get("h")
                if h:
                    refs.add(h)
    return refs


def prune_project_snapshots(project_id: str, keep: int | None = None) -> int:
    """Delete this project's oldest snapshot manifests beyond ``keep`` (default:
    MAX_SNAPSHOTS_PER_PROJECT, read at call time). Returns how many manifests
    were removed. Objects are freed separately by gc_objects."""
    if keep is None:
        keep = MAX_SNAPSHOTS_PER_PROJECT
    ids = list_snapshots(project_id)
    if len(ids) <= keep:
        return 0
    to_drop = ids[: len(ids) - keep]  # ids are zero-padded, sorted oldest→newest
    d = _snapshots_dir(project_id)
    removed = 0
    for sid in to_drop:
        try:
            (d / f"{sid}.json").unlink()
            removed += 1
        except OSError:
            pass
    return removed


def gc_objects() -> int:
    """Delete stored objects not referenced by any surviving snapshot (across
    all projects). Returns the number of objects removed. Aborts safely (returns
    0) if any manifest is unreadable, so we never delete a still-referenced blob."""
    try:
        refs = _referenced_hashes()
    except Exception:  # noqa: BLE001 - unreadable manifest: don't risk deleting live data
        return 0
    objects = _objects_dir()
    removed = 0
    for obj in objects.iterdir():
        if obj.suffix == ".tmp":
            continue
        if obj.name not in refs:
            try:
                obj.unlink()
                removed += 1
            except OSError:
                pass
    return removed


def max_action_id(project_id: str) -> int:
    """Highest action id seen for this project, across BOTH snapshot manifests and
    ledger entries. Snapshot and ledger ids share one monotonic sequence, but a
    no-snapshot action (OPENDOT_NO_SNAPSHOT) advances only the ledger — so the id
    counter must consider both to stay unique. 0 if none.
    """
    d = _snapshots_dir(project_id)
    ids = [int(p.stem) for p in d.glob("*.json") if p.stem.isdigit()]
    # The ledger lives at <store>/ledger/<project>.jsonl; read ids without
    # importing the ledger module (that would be a circular import).
    ledger_file = store_root() / "ledger" / f"{project_id}.jsonl"
    if ledger_file.exists():
        for line in ledger_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                lid = json.loads(line).get("id", "")
                if isinstance(lid, str) and lid.isdigit():
                    ids.append(int(lid))
            except Exception:  # noqa: BLE001 - a malformed line shouldn't break id gen
                continue
    return max(ids) if ids else 0


def _next_snapshot_id(project_id: str) -> str:
    """Monotonic, sortable id. Counter persisted per project (no wall-clock dep)."""
    return f"{max_action_id(project_id) + 1:06d}"


def _write_manifest(snap: Snapshot) -> None:
    path = _snapshots_dir(snap.project_id) / f"{snap.id}.json"
    path.write_text(
        json.dumps(
            {
                "id": snap.id,
                "project_id": snap.project_id,
                "workdir": snap.workdir,
                "files": {
                    rel: {"h": e.h, "m": e.mode, "t": e.mtime, "s": e.size, "c": e.ctime_ns}
                    for rel, e in snap.files.items()
                },
                "skipped_large": snap.skipped_large,
            },
            indent=0,
        ),
        encoding="utf-8",
    )


def load_snapshot(project_id: str, snap_id: str) -> Snapshot:
    path = _snapshots_dir(project_id) / f"{snap_id}.json"
    d = json.loads(path.read_text(encoding="utf-8"))
    files: dict[str, FileEntry] = {}
    for rel, v in d["files"].items():
        # Back-compat: old manifests stored a bare hash string (no metadata).
        if isinstance(v, str):
            files[rel] = FileEntry(h=v, mode=None)
        else:
            files[rel] = FileEntry(
                h=v["h"],
                mode=v.get("m"),
                mtime=v.get("t"),
                size=v.get("s"),
                ctime_ns=v.get("c"),
            )
    return Snapshot(
        id=d["id"],
        project_id=d["project_id"],
        workdir=d["workdir"],
        files=files,
        skipped_large=d.get("skipped_large", []),
    )


def list_snapshots(project_id: str) -> list[str]:
    d = _snapshots_dir(project_id)
    return sorted(p.stem for p in d.glob("*.json") if p.stem.isdigit())


# Dependency lockfiles. Restoring one rolls back which versions are *declared*,
# but NOT the installed environment (node_modules, site-packages, global caches,
# postinstall side effects). So when undo changes a lockfile we warn loudly: the
# user must re-run their package manager to actually match it. (Ledger principle:
# never let "files restored" be mistaken for "environment restored".)
_LOCKFILES = {
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "npm-shrinkwrap.json",
    "uv.lock",
    "poetry.lock",
    "Pipfile.lock",
    "requirements.txt",
    "Cargo.lock",
    "Gemfile.lock",
    "composer.lock",
    "go.sum",
}


def _ensure_real_dirs(base: Path, leaf: Path) -> None:
    """Make every path component from ``base`` (exclusive) down to ``leaf``
    (inclusive) a real directory. If a component is a symlink or a non-directory,
    remove it first. This stops restore from writing through a symlinked parent
    and escaping the workspace. ``base`` itself is never modified.
    """
    # Components from base down to leaf, outermost first.
    parts: list[Path] = []
    cur = leaf
    while cur != base and base in cur.parents:
        parts.append(cur)
        cur = cur.parent
    for d in reversed(parts):
        if d.is_symlink():
            d.unlink()
        elif d.exists() and not d.is_dir():
            d.unlink()
        d.mkdir(exist_ok=True)


def restore_snapshot(snap: Snapshot, rules: IgnoreRules | None = None) -> list[str]:
    """Make the workspace byte-for-byte match the snapshot.

    Rewrites/creates every file in the manifest; removes files that exist now but
    were not captured (respecting ignore rules — we never touch skipped paths).

    Returns the relative paths of any dependency lockfiles whose content the
    restore changed — the caller should warn that the *environment* isn't
    restored until the package manager is re-run.
    """
    rules = rules or IgnoreRules()
    wd = Path(snap.workdir).resolve()
    changed_lockfiles: list[str] = []

    # 1. Restore all files from the manifest (content + permission mode).
    for rel, entry in snap.files.items():
        # Defense in depth: manifest paths are written by opendot from a walk
        # *inside* the workspace, so they're always safe relative paths. But if a
        # snapshot file were tampered with to contain an absolute path or a ".."
        # segment, `wd / rel` could resolve outside the workspace. Skip any such
        # entry rather than write outside (and rather than abort the whole
        # restore — the remaining, valid files should still be restored).
        rel_path = Path(rel)
        if rel_path.is_absolute() or ".." in rel_path.parts:
            continue
        target = wd / rel
        # Before overwriting, note if this is a lockfile whose content differs —
        # that means the restore rolled dependency versions back.
        if Path(rel).name in _LOCKFILES:
            try:
                if not target.exists() or _hash_file(target) != entry.h:
                    changed_lockfiles.append(rel)
            except OSError:
                changed_lockfiles.append(rel)
        # Rebuild the path to the file as REAL directories. If any parent
        # component was replaced by a symlink (or a non-dir), writing through it
        # could land the file OUTSIDE the workspace — the one thing restore must
        # never do. So walk wd -> target.parent and force each level to a real dir.
        _ensure_real_dirs(wd, target.parent)
        # If something now occupies the file's own path (it was a file at snapshot
        # time but became a symlink or a directory), remove it first so write_bytes
        # recreates the real file rather than writing through a symlink or raising.
        if target.is_symlink():
            target.unlink()
        elif target.is_dir():
            import shutil

            shutil.rmtree(target)
        target.write_bytes(_read_object(entry.h))
        if entry.mode is not None:
            try:
                target.chmod(entry.mode)
            except OSError:
                pass  # best-effort: FS may not support the mode

    # 2. Remove files present now but absent from the snapshot (only non-skipped).
    manifest_paths = set(snap.files)
    for f in _iter_files(wd, rules):
        rel = f.relative_to(wd).as_posix()
        if rel not in manifest_paths:
            if f.name in _LOCKFILES:  # a lockfile the install created → undo removes it
                changed_lockfiles.append(rel)
            try:
                f.unlink()
            except OSError:
                pass

    # 3. Prune now-empty directories (best effort, bottom-up).
    for dirpath, dirnames, filenames in os.walk(wd, topdown=False):
        p = Path(dirpath)
        if p == wd:
            continue
        if any(rules.skipped(part) for part in p.relative_to(wd).parts):
            continue
        try:
            if not any(p.iterdir()):
                p.rmdir()
        except OSError:
            pass

    # De-dupe (a lockfile could be both changed and matched) and return.
    return sorted(set(changed_lockfiles))


def _is_text(data: bytes) -> bool:
    """Best-effort text detection for diff purposes."""
    if b"\x00" in data:
        return False
    try:
        data.decode("utf-8")
        return True
    except UnicodeDecodeError:
        return False


def diff_snapshot(
    snap: Snapshot,
    rules: IgnoreRules | None = None,
    *,
    include_text_diff: bool = True,
) -> dict:
    """Compare a snapshot against the current workspace.

    Returns a dry-run delta describing what ``restore_snapshot(snap)`` would do:

    - ``added`` — files in the snapshot but missing now (a restore would recreate them)
    - ``removed`` — files present now but not in the snapshot (a restore would delete them)
    - ``modified`` — files whose content differs. Each entry is a dict with
      ``path`` and, for text files when ``include_text_diff`` is True, a
      ``unified_diff`` string.

    Ignored paths are excluded from the live walk, matching the restore behavior.
    """
    rules = rules or IgnoreRules()
    wd = Path(snap.workdir).resolve()

    current_files: dict[str, Path] = {}
    for f in _iter_files(wd, rules):
        rel = f.relative_to(wd).as_posix()
        rel_path = Path(rel)
        if rel_path.is_absolute() or ".." in rel_path.parts:
            continue
        current_files[rel] = f

    snap_files = set(snap.files)
    current_paths = set(current_files)

    added = sorted(snap_files - current_paths)
    removed = sorted(current_paths - snap_files)
    modified: list[dict] = []

    for rel in sorted(snap_files & current_paths):
        entry = snap.files[rel]
        try:
            current_hash = _hash_file(current_files[rel])
        except OSError:
            # Can't read the file to hash it, but restore would still overwrite it.
            # Report it as modified (without a text diff) rather than hide it.
            modified.append({"path": rel, "unified_diff": None})
            continue
        if current_hash != entry.h:
            diff_entry: dict = {"path": rel}
            if include_text_diff:
                try:
                    old_bytes = _read_object(entry.h)
                    new_bytes = current_files[rel].read_bytes()
                except OSError:
                    # A read failed: report the change but skip the (now unreliable)
                    # text diff rather than emit a misleading empty one.
                    diff_entry["unified_diff"] = None
                    modified.append(diff_entry)
                    continue
                if _is_text(old_bytes) and _is_text(new_bytes):
                    old_text = old_bytes.decode("utf-8")
                    new_text = new_bytes.decode("utf-8")
                    diff_lines = list(
                        difflib.unified_diff(
                            old_text.splitlines(keepends=True),
                            new_text.splitlines(keepends=True),
                            fromfile=f"snapshot/{rel}",
                            tofile=f"workspace/{rel}",
                        )
                    )
                    diff_entry["unified_diff"] = "".join(diff_lines)
                else:
                    diff_entry["unified_diff"] = None
            modified.append(diff_entry)

    return {"added": added, "removed": removed, "modified": modified}
