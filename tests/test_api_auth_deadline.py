"""Optional central authentication lifetime can only shorten the default session."""
import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from faceauth.api import app as api


@pytest.mark.parametrize("invalid", [True, False, 0, -1, 10001, 1.5, "1000", float("nan"), float("inf")])
def test_auth_timeout_is_strict_bounded_milliseconds(invalid):
    with pytest.raises(ValidationError):
        api.SessionIn(purpose="auth", device_id="fixture", action_id="fixture", timeout_ms=invalid)


@pytest.mark.parametrize("purpose,timeout,seconds", [("auth", 1, .001), ("auth", 4321, 4.321),
    ("auth", 10000, 10), ("auth", None, 30), ("enroll", None, 180)])
def test_session_timeout_preserves_defaults_and_uses_current_database_clock(monkeypatch, purpose, timeout, seconds):
    statements = []

    @asynccontextmanager
    async def connection():
        yield None

    async def device(*args):
        return {"id": "fixture"}

    async def one(conn, query, params):
        statements.append((query, params))
        return {"n": 0} if query.startswith("SELECT count") else {"expires_at": datetime.now(timezone.utc)}

    monkeypatch.setattr(api.C, "pool", SimpleNamespace(connection=connection))
    monkeypatch.setattr(api, "_device", device)
    monkeypatch.setattr(api, "_one", one)
    body = api.SessionIn(purpose=purpose, device_id="fixture", action_id="fixture", timeout_ms=timeout)
    result = asyncio.run(api.create_session(body, None))
    assert set(result) == {"session_id", "challenge", "expires_at"}
    query, params = statements[-1]
    assert "clock_timestamp() + %s * interval '1 second'" in query and params[-1] == seconds


def test_auth_timeout_cannot_change_enrollment_lifetime():
    body = api.SessionIn(purpose="enroll", device_id="fixture", action_id="fixture", timeout_ms=1000)
    with pytest.raises(HTTPException) as exc:
        asyncio.run(api.create_session(body, None))
    assert exc.value.status_code == 422 and exc.value.detail == "timeout_requires_auth"


def test_consumption_checks_wall_clock_after_waiting_for_subject_lock(monkeypatch):
    queries = []

    async def one(conn, query, params):
        queries.append(query)
        # An expired row is not consumed, so the caller cannot issue an approval.
        return None

    monkeypatch.setattr(api, "_one", one)
    assert asyncio.run(api._consume(None, "fixture", "approved", "ok")) is False
    assert "expires_at > clock_timestamp()" in queries[0]
    assert "consumed_at = clock_timestamp()" in queries[0]
