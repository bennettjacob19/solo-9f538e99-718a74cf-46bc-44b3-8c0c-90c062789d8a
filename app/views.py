"""Consistent snapshot views over the archive for the web UI.

Opening the overview page pins a *snapshot view*: one extra SQLite connection
to the very same database file, holding an open read transaction.  In WAL
mode that read transaction sees exactly the committed state from the moment
the first screen was read — nothing is copied, writers keep committing new
archives and same-id re-archives undisturbed, and the reader never blocks
them.

Paging (Older/Newer), returning from a trace detail, and the detail itself
all read inside the view, so the list, its order, the total count and the
spans of every trace stay exactly as the first screen rendered them — even
if a trace id is re-archived while the user is browsing.

Views are bounded: each lives for a fixed TTL after the first screen, at most
``max_views`` exist at the same time, and expired views are closed
automatically (both lazily on access and by a janitor thread).  Views are
kept in memory only, so a restart invalidates every view link.
"""

from __future__ import annotations

import logging
import secrets
import sqlite3
import threading
import time
from typing import Callable, Optional

from .storage import get_trace_on, list_traces_on

log = logging.getLogger("tailsampler.views")

DEFAULT_VIEW_TTL_SECONDS = 300.0
DEFAULT_MAX_VIEWS = 8


class ViewLimitError(Exception):
    """Raised when the maximum number of concurrent views is already open."""


class ViewClosedError(Exception):
    """Raised when a view is used after its snapshot was released."""


class TraceView:
    """One pinned read snapshot of the archive.

    The connection holds a deferred read transaction; the first SELECT issued
    at creation fixes the WAL read mark, so every later query on this
    connection observes the same committed state.
    """

    def __init__(
        self,
        view_id: str,
        conn: sqlite3.Connection,
        clock: Callable[[], float],
        created_at: float,
        ttl_seconds: float,
    ) -> None:
        self.view_id = view_id
        self._conn = conn
        self._clock = clock
        self.created_at = created_at
        self.expires_at = created_at + ttl_seconds
        self._lock = threading.Lock()
        self._closed = False

    def expires_in(self) -> float:
        return max(0.0, self.expires_at - self._clock())

    def is_expired(self, now: float) -> bool:
        return now >= self.expires_at

    def list_traces(self, limit: int, offset: int) -> tuple[list[dict], int]:
        with self._lock:
            self._ensure_open()
            return list_traces_on(self._conn, limit, offset)

    def get_trace(self, trace_id: str) -> Optional[dict]:
        with self._lock:
            self._ensure_open()
            return get_trace_on(self._conn, trace_id)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._conn.rollback()  # release the pinned read snapshot
                self._conn.close()
            except sqlite3.Error:  # pragma: no cover - defensive
                log.exception("error while closing view %s", self.view_id)

    def _ensure_open(self) -> None:
        if self._closed:
            raise ViewClosedError(f"view {self.view_id} is closed")


class ViewManager:
    """Registry of open snapshot views with TTL, cap and automatic reaping."""

    def __init__(
        self,
        db_path: str,
        ttl_seconds: float = DEFAULT_VIEW_TTL_SECONDS,
        max_views: int = DEFAULT_MAX_VIEWS,
        clock: Optional[Callable[[], float]] = None,
        reap_interval_seconds: float = 1.0,
        start_reaper: bool = True,
    ) -> None:
        if ttl_seconds <= 0 or max_views <= 0:
            raise ValueError("view ttl and max_views must be positive")
        self._db_path = db_path
        self._ttl = float(ttl_seconds)
        self._max_views = int(max_views)
        self._clock: Callable[[], float] = clock or time.monotonic
        self._reap_interval = max(0.1, float(reap_interval_seconds))

        self._lock = threading.Lock()
        self._views: dict[str, TraceView] = {}
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        if start_reaper:
            self.start()

    # ------------------------------------------------------------------ life
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="trace-view-reaper", daemon=True
        )
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=timeout)
            self._thread = None

    def _run(self) -> None:
        while not self._stop.wait(self._reap_interval):
            try:
                self._reap()
            except Exception:  # pragma: no cover - defensive
                log.exception("view reaper failed")

    # ------------------------------------------------------------------ views
    def create(self) -> TraceView:
        """Pin a new snapshot view.  Raises ViewLimitError when at the cap."""

        with self._lock:
            self._reap_locked()
            if len(self._views) >= self._max_views:
                raise ViewLimitError(
                    f"{self._max_views} views already open; retry later"
                )
            view_id = secrets.token_hex(8)
            conn = self._open_snapshot_conn()
            view = TraceView(view_id, conn, self._clock, self._clock(), self._ttl)
            self._views[view_id] = view
            log.info(
                "view %s opened (ttl=%.0fs, %d open)",
                view_id,
                self._ttl,
                len(self._views),
            )
            return view

    def get(self, view_id: str) -> Optional[TraceView]:
        """Return the live view for ``view_id`` or None if gone/expired."""

        with self._lock:
            self._reap_locked()
            return self._views.get(view_id)

    def end(self, view_id: str) -> bool:
        """Close a view early (the "view latest" action).  Idempotent."""

        with self._lock:
            view = self._views.pop(view_id, None)
        if view is None:
            return False
        view.close()
        log.info("view %s ended early", view_id)
        return True

    def close_all(self) -> None:
        """Release every view (service shutdown)."""

        self.stop()
        with self._lock:
            views = list(self._views.values())
            self._views.clear()
        for view in views:
            view.close()

    def open_count(self) -> int:
        with self._lock:
            self._reap_locked()
            return len(self._views)

    # ------------------------------------------------------------------ intern
    def _open_snapshot_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self._db_path,
            timeout=10.0,
            check_same_thread=False,
            isolation_level=None,  # explicit transaction control
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 10000")
        conn.execute("BEGIN")
        # Pin the WAL read mark: from here on this transaction observes exactly
        # the committed state of this moment, no matter what gets archived next.
        conn.execute("SELECT COUNT(*) FROM archived_traces").fetchone()
        conn.execute("PRAGMA query_only = ON")
        return conn

    def _reap(self) -> None:
        with self._lock:
            self._reap_locked()

    def _reap_locked(self) -> None:
        now = self._clock()
        expired = [
            view_id
            for view_id, view in self._views.items()
            if view.is_expired(now)
        ]
        for view_id in expired:
            self._views.pop(view_id).close()
            log.info("view %s expired and was released", view_id)
