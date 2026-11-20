"""Snapshot-view behaviour: stable paging, pinned details, TTL, cap, restart."""

from __future__ import annotations

import json
import os
import re
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.main import create_app  # noqa: E402

VIEW_RE = re.compile(r"view=([0-9a-f]{16})")


def tid(n: int) -> str:
    return f"{n:032x}"


def sid(n: int) -> str:
    return f"{n:016x}"


def otlp_payload(spans) -> dict:
    groups: dict[tuple, list] = {}
    for item in spans:
        trace_id = item["trace"]
        key = (item.get("service", "checkout"), item.get("scope", "test"))
        span = {
            "traceId": trace_id,
            "spanId": item["span"],
            "name": item.get("name", "op"),
            "kind": item.get("kind", 2),
            "startTimeUnixNano": str(item.get("start", 1_700_000_000_000_000_000)),
            "endTimeUnixNano": str(item.get("end", 1_700_000_000_500_000_000)),
            "attributes": item.get(
                "attributes",
                [{"key": "http.route", "value": {"stringValue": "/checkout"}}],
            ),
            "status": item.get(
                "status",
                {"code": 2, "message": "payment declined"}
                if item.get("error")
                else {"code": 1},
            ),
        }
        if item.get("parent"):
            span["parentSpanId"] = item["parent"]
        groups.setdefault(key, []).append(span)

    return {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [
                        {"key": "service.name", "value": {"stringValue": service}}
                    ]
                },
                "scopeSpans": [{"scope": {"name": scope}, "spans": group}],
            }
            for (service, scope), group in groups.items()
        ]
    }


class Clock:
    def __init__(self) -> None:
        self.now = 5000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture()
def clock():
    return Clock()


def make_client(tmp_path, clock, **overrides):
    params = dict(
        window_seconds=2.0,
        capacity=16,
        max_spans_per_trace=64,
        start_sampler_thread=False,
        clock=clock,
    )
    params.update(overrides)
    app = create_app(str(tmp_path / "traces.db"), **params)
    return app


@pytest.fixture()
def client(tmp_path, clock):
    app = make_client(tmp_path, clock)
    with TestClient(app) as test_client:
        test_client.app_state = app.state
        yield test_client


def post(client, spans):
    return client.post(
        "/v1/traces",
        content=json.dumps(otlp_payload(spans)),
        headers={"content-type": "application/json"},
    )


def close_window(client, clock):
    clock.advance(2.0)
    client.app_state.sampler.tick()


def archive_trace(client, clock, n: int, name: str = "op") -> str:
    trace = tid(n)
    post(client, [{"trace": trace, "span": sid(1), "name": name, "error": True}])
    close_window(client, clock)
    return trace


def first_view_id(page_text: str) -> str:
    match = VIEW_RE.search(page_text)
    assert match, "expected a view id in the page links"
    return match.group(1)


def test_paging_is_stable_while_new_traces_archive(client, clock):
    """First screen pins the list: later archives never shift Older/Newer."""

    traces = [archive_trace(client, clock, n, name=f"op-{n}") for n in range(1, 4)]

    first = client.get("/?limit=2")
    assert first.status_code == 200
    view = first_view_id(first.text)
    assert "of 3" in first.text
    assert traces[2] in first.text  # newest first
    assert traces[1] in first.text
    assert traces[0] not in first.text

    # Two more traces commit while the user is browsing.
    new_traces = [archive_trace(client, clock, n, name=f"op-{n}") for n in (4, 5)]

    older = client.get(f"/?view={view}&limit=2&offset=2")
    assert older.status_code == 200
    assert "of 3" in older.text  # total frozen at the first screen
    assert traces[0] in older.text
    for trace in traces[1:] + new_traces:
        assert trace not in older.text

    newer = client.get(f"/?view={view}&limit=2&offset=0")
    assert "of 3" in newer.text
    assert traces[2] in newer.text
    assert traces[1] in newer.text

    # The live JSON API still reports current data.
    assert client.get("/api/traces").json()["total"] == 5


def test_detail_inside_view_shows_the_version_from_the_first_screen(
    tmp_path, clock
):
    # Give the view a TTL longer than the closed-id retention (300s) so the
    # same trace id can be re-archived while the view is alive.
    app = make_client(tmp_path, clock, view_ttl_seconds=3600.0)
    with TestClient(app) as client:
        client.app_state = app.state
        trace = tid(42)
        post(client, [{"trace": trace, "span": sid(1), "name": "v1-span", "error": True}])
        close_window(client, clock)

        first = client.get("/")
        view = first_view_id(first.text)

        # Same trace id, new version: wait out the closed-id retention, then
        # archive a replacement with different spans.
        clock.advance(301.0)
        post(client, [{"trace": trace, "span": sid(2), "name": "v2-span", "error": True}])
        close_window(client, clock)

        # The view keeps showing the version its first screen listed.
        in_view = client.get(f"/traces/{trace}?view={view}")
        assert in_view.status_code == 200
        assert "v1-span" in in_view.text
        assert "v2-span" not in in_view.text

        # Bare links and the JSON API follow the live archive.
        live = client.get(f"/traces/{trace}")
        assert "v2-span" in live.text
        api = client.get(f"/api/traces/{trace}").json()
        assert [s["name"] for s in api["spans"]] == ["v2-span"]


def test_view_links_expire_and_prompt_reopening(client, clock):
    archive_trace(client, clock, 7, name="op-7")
    view = first_view_id(client.get("/").text)

    clock.advance(301.0)  # past the 5 minute view TTL

    page = client.get(f"/?view={view}")
    assert page.status_code == 410
    assert "Reopen the overview" in page.text

    detail = client.get(f"/traces/{tid(7)}?view={view}")
    assert detail.status_code == 410
    assert "Reopen the overview" in detail.text

    # Resources were released: a fresh first screen works immediately.
    assert client.get("/").status_code == 200


def test_view_cap_prompts_retry_later(tmp_path, clock):
    app = make_client(tmp_path, clock, max_views=2)
    with TestClient(app) as client:
        client.app_state = app.state
        archive_trace(client, clock, 1, name="op-1")
        assert client.get("/").status_code == 200
        second = client.get("/")
        assert second.status_code == 200
        view2 = first_view_id(second.text)

        full = client.get("/")
        assert full.status_code == 503
        assert "retry" in full.text.lower()

        # Ending one view frees a slot again (don't follow the redirect: the
        # redirect target itself opens a fresh view).
        client.post(f"/views/{view2}/end", follow_redirects=False)
        assert client.get("/").status_code == 200


def test_view_latest_ends_only_the_current_view(client, clock):
    archive_trace(client, clock, 8, name="op-8")
    view_a = first_view_id(client.get("/").text)
    view_b = first_view_id(client.get("/").text)

    response = client.post(f"/views/{view_a}/end", follow_redirects=False)
    assert response.status_code == 303
    assert response.headers["location"] == "/"

    # The ended view is gone; the other view keeps working.
    assert client.get(f"/?view={view_a}").status_code == 410
    assert client.get(f"/traces/{tid(8)}?view={view_a}").status_code == 410
    still_there = client.get(f"/?view={view_b}")
    assert still_there.status_code == 200
    assert tid(8) in still_there.text

    # The redirect target re-reads: a brand new view on a fresh first screen.
    fresh = client.get("/")
    assert fresh.status_code == 200
    assert first_view_id(fresh.text) not in (view_a, view_b)


def test_restart_invalidates_views_but_keeps_the_archive(tmp_path, clock):
    db_file = str(tmp_path / "traces.db")
    app1 = create_app(
        db_file,
        window_seconds=2.0,
        capacity=16,
        max_spans_per_trace=64,
        start_sampler_thread=False,
        clock=clock,
    )
    with TestClient(app1) as client1:
        client1.app_state = app1.state
        archive_trace(client1, clock, 9, name="op-9")
        view = first_view_id(client1.get("/").text)

    app2 = create_app(
        db_file,
        window_seconds=2.0,
        capacity=16,
        max_spans_per_trace=64,
        start_sampler_thread=False,
        clock=clock,
    )
    with TestClient(app2) as client2:
        client2.app_state = app2.state
        gone = client2.get(f"/?view={view}")
        assert gone.status_code == 410
        assert "Reopen the overview" in gone.text

        # Same SQLite volume: the archived trace is still there.
        fresh = client2.get("/")
        assert fresh.status_code == 200
        assert tid(9) in fresh.text


def test_view_does_not_block_or_copy_the_archive(client, clock, tmp_path):
    archive_trace(client, clock, 10, name="op-10")
    view = first_view_id(client.get("/").text)

    # New archives and same-id re-archives commit normally while a view sits
    # open, and no second database file appears on disk.
    archive_trace(client, clock, 11, name="op-11")
    assert client.app_state.store.count() == 2
    db_files = {p.name for p in tmp_path.iterdir()}
    assert db_files <= {"traces.db", "traces.db-wal", "traces.db-shm"}

    page = client.get(f"/?view={view}")
    assert "of 1" in page.text  # the view is frozen
    assert client.get("/api/traces").json()["total"] == 2
