"""FastAPI tail-sampling collector.

Run with ``uvicorn app.main:app``.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Callable, Optional

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates

from .otlp import OtlpParseError, export_response, parse_trace_request
from .sampler import (
    DEFAULT_CAPACITY,
    DEFAULT_MAX_SPANS,
    DEFAULT_WINDOW_SECONDS,
    TraceSampler,
)
from .storage import ArchiveStore
from .views import (
    DEFAULT_MAX_VIEWS,
    DEFAULT_VIEW_TTL_SECONDS,
    TraceView,
    ViewClosedError,
    ViewLimitError,
    ViewManager,
)

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("tailsampler")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "templates")
DEFAULT_DB_PATH = os.path.join(BASE_DIR, "..", "data", "traces.db")
MAX_BODY_BYTES = int(os.environ.get("MAX_BODY_BYTES", str(8 * 1024 * 1024)))

# ---------------------------------------------------------------------------
# Template helpers
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def _pretty_json(value) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def _format_duration(ns) -> str:
    if ns is None:
        return "—"
    try:
        ns = int(ns)
    except (TypeError, ValueError):
        return "—"
    if ns < 1_000:
        return f"{ns} ns"
    if ns < 1_000_000:
        return f"{ns / 1_000:.1f} µs"
    if ns < 1_000_000_000:
        return f"{ns / 1_000_000:.2f} ms"
    return f"{ns / 1_000_000_000:.3f} s"


def _format_time(ns, fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    if ns is None:
        return "—"
    try:
        stamp = datetime.datetime.fromtimestamp(
            int(ns) / 1_000_000_000, tz=datetime.timezone.utc
        )
    except (TypeError, ValueError, OSError, OverflowError):
        return "—"
    # Only add sub-second precision when the format asks for it.
    return stamp.strftime(fmt) + (" UTC" if "%f" not in fmt else "")


def _env_float(name: str, default: float) -> float:
    try:
        value = float(os.environ.get(name, ""))
    except ValueError:
        return default
    return value if value > 0 else default


def _env_int(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, ""))
    except ValueError:
        return default
    return value if value > 0 else default


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(
    db_path: Optional[str] = None,
    *,
    window_seconds: Optional[float] = None,
    capacity: Optional[int] = None,
    max_spans_per_trace: Optional[int] = None,
    view_ttl_seconds: Optional[float] = None,
    max_views: Optional[int] = None,
    start_sampler_thread: bool = True,
    clock: Optional[Callable[[], float]] = None,
) -> FastAPI:
    db_path = db_path or os.environ.get("DB_PATH", DEFAULT_DB_PATH)
    store = ArchiveStore(db_path)
    sampler = TraceSampler(
        store,
        window_seconds=window_seconds
        if window_seconds is not None
        else _env_float("SAMPLE_WINDOW_SECONDS", DEFAULT_WINDOW_SECONDS),
        capacity=capacity
        if capacity is not None
        else _env_int("MAX_ACTIVE_TRACES", DEFAULT_CAPACITY),
        max_spans_per_trace=max_spans_per_trace
        if max_spans_per_trace is not None
        else _env_int("MAX_SPANS_PER_TRACE", DEFAULT_MAX_SPANS),
        clock=clock,
    )
    views = ViewManager(
        db_path,
        ttl_seconds=view_ttl_seconds
        if view_ttl_seconds is not None
        else _env_float("VIEW_TTL_SECONDS", DEFAULT_VIEW_TTL_SECONDS),
        max_views=max_views
        if max_views is not None
        else _env_int("MAX_VIEWS", DEFAULT_MAX_VIEWS),
        clock=clock,
        start_reaper=start_sampler_thread,
    )
    templates = Jinja2Templates(directory=TEMPLATE_DIR)
    templates.env.filters["pretty_json"] = _pretty_json
    templates.env.filters["duration"] = _format_duration
    templates.env.filters["utc_time"] = _format_time

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        if start_sampler_thread:
            sampler.start()
        log.info(
            "tail sampler ready db=%s window=%.2fs capacity=%d max_spans=%d",
            db_path,
            sampler.window_seconds,
            sampler.capacity,
            sampler.max_spans_per_trace,
        )
        try:
            yield
        finally:
            sampler.stop()
            views.close_all()
            store.close()

    app = FastAPI(title="OTLP Tail Sampler", lifespan=lifespan)
    app.state.store = store
    app.state.sampler = sampler
    app.state.views = views

    def render(request: Request, name: str, context: dict, status_code: int = 200):
        return templates.TemplateResponse(
            request, name, context, status_code=status_code
        )

    def page_context(request: Request, **extra) -> dict:
        context = {"status": sampler.snapshot(), "request": request}
        context.update(extra)
        return context

    def flush_due_writes() -> None:
        """Run due archive writes inline so a page reflects just-finished work."""
        sampler.tick()

    def render_index(
        request: Request, view: TraceView, limit: int, offset: int
    ) -> HTMLResponse:
        """Render the overview from one pinned snapshot view."""

        traces, total = view.list_traces(limit, offset)
        return render(
            request,
            "index.html",
            page_context(
                request,
                traces=traces,
                total=total,
                limit=limit,
                offset=offset,
                view_id=view.view_id,
                view_expires_in=int(view.expires_in()),
            ),
        )

    def view_gone(request: Request) -> HTMLResponse:
        """The view expired, was ended, or never existed (e.g. after restart)."""
        return render(
            request, "view_gone.html", page_context(request), status_code=410
        )

    def overview(request: Request) -> HTMLResponse:
        """Open a fresh snapshot view and render its first screen."""

        try:
            view = views.create()
        except ViewLimitError:
            return render(
                request, "view_full.html", page_context(request), status_code=503
            )
        try:
            return render_index(request, view, limit=50, offset=0)
        except ViewClosedError:
            return view_gone(request)

    # -- OTLP ingest --------------------------------------------------------
    @app.post("/v1/traces")
    async def ingest_traces(request: Request) -> Response:
        content_type = request.headers.get("content-type", "").lower()
        if "protobuf" in content_type:
            return JSONResponse(
                status_code=415,
                content=export_response(
                    0,
                    "only OTLP/HTTP JSON is supported; "
                    "send Content-Type: application/json",
                ),
            )
        body = await request.body()
        if len(body) > MAX_BODY_BYTES:
            return JSONResponse(
                status_code=413,
                content=export_response(0, "request body too large"),
            )
        try:
            spans, invalid = parse_trace_request(body)
        except OtlpParseError as exc:
            # Malformed export: nothing was accepted, all spans implied lost.
            return JSONResponse(
                status_code=400, content=export_response(0, str(exc))
            )

        result = sampler.ingest(spans, invalid=invalid)
        if result.rejected:
            detail = ", ".join(
                f"{reason}={count}" for reason, count in sorted(result.reasons.items())
            )
            message = f"rejected {result.rejected} span(s): {detail}"
            log.info("export partial success: %s", message)
            return JSONResponse(
                status_code=200,
                content=export_response(
                    result.rejected, message, accepted=result.accepted
                ),
            )
        return JSONResponse(status_code=200, content=export_response(0))

    # -- HTML ---------------------------------------------------------------
    @app.get("/", response_class=HTMLResponse)
    async def index(
        request: Request,
        limit: int = Query(50, ge=1, le=200),
        offset: int = 0,
        view: Optional[str] = None,
    ):
        offset = max(0, offset)
        if view:
            # Paging or returning from a detail: keep reading the first
            # screen's snapshot instead of the live archive.
            current = views.get(view)
            if current is None:
                return view_gone(request)
            try:
                return render_index(request, current, limit, offset)
            except ViewClosedError:
                return view_gone(request)
        # First screen: pin a fresh snapshot of the committed archive.
        try:
            current = views.create()
        except ViewLimitError:
            return render(
                request, "view_full.html", page_context(request), status_code=503
            )
        try:
            return render_index(request, current, limit, offset)
        except ViewClosedError:
            return view_gone(request)

    @app.get("/traces/{trace_id}", response_class=HTMLResponse)
    async def trace_detail(
        request: Request,
        trace_id: str,
        view: Optional[str] = None,
        limit: int = Query(50, ge=1, le=200),
        offset: int = 0,
    ):
        current = None
        if view:
            current = views.get(view)
            if current is None:
                return view_gone(request)
        try:
            # Inside a view the detail shows the trace exactly as the first
            # screen archived it; a bare link reads the live archive.
            trace = (
                current.get_trace(trace_id)
                if current is not None
                else store.get_trace(trace_id)
            )
        except ViewClosedError:
            return view_gone(request)
        if trace is None:
            return render(
                request,
                "missing.html",
                page_context(request, trace_id=trace_id, view_id=view),
                status_code=404,
            )
        return render(
            request,
            "trace.html",
            page_context(
                request,
                trace=trace,
                view_id=view,
                limit=limit,
                offset=max(0, offset),
                view_expires_in=int(current.expires_in()) if current else None,
            ),
        )

    @app.post("/views/{view_id}/end")
    async def end_view(view_id: str):
        # "View latest": end the current view, then re-read a fresh first
        # screen.  Every other open view stays valid.
        views.end(view_id)
        return RedirectResponse(url="/", status_code=303)

    @app.post("/pause", response_class=HTMLResponse)
    async def page_pause(request: Request):
        sampler.pause()
        return overview(request)

    @app.post("/resume", response_class=HTMLResponse)
    async def page_resume(request: Request):
        sampler.resume()
        flush_due_writes()
        return overview(request)

    # -- JSON API -----------------------------------------------------------
    @app.get("/api/status")
    async def api_status() -> dict:
        snapshot = sampler.snapshot()
        snapshot["archivedTraces"] = store.count()
        return snapshot

    @app.get("/api/traces")
    async def api_traces(limit: int = 100, offset: int = 0) -> dict:
        traces, total = store.list_traces(limit=limit, offset=offset)
        return {"total": total, "traces": traces}

    @app.get("/api/traces/{trace_id}")
    async def api_trace(trace_id: str):
        trace = store.get_trace(trace_id)
        if trace is None:
            return JSONResponse(status_code=404, content={"error": "not archived"})
        return trace

    @app.post("/api/archive/pause")
    async def api_pause() -> dict:
        sampler.pause()
        return {"paused": True}

    @app.post("/api/archive/resume")
    async def api_resume() -> dict:
        sampler.resume()
        flush_due_writes()
        return {"paused": False}

    @app.get("/healthz")
    async def healthz() -> dict:
        return {"status": "ok"}

    return app


app = create_app()
