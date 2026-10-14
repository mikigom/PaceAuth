"""Login-free LAN console boundaries; all identities and frames are synthetic."""
import threading
import time

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from faceauth.common import token
from faceauth.terminal.web_auth import CONSOLE_COOKIE, COOKIE, CaptureCancelled
from test_account_management_web import synthetic_photo, PHOTO_CONSENT
from test_web_auth import CONSENT, CREDENTIALS, HEADERS, finish, setup


@pytest.fixture
def console(setup):
    client, service, backend = setup
    service.open_console = True
    yield client, service, backend


def connect(client):
    response = client.get("/api/status")
    assert response.status_code == 200
    auth = response.json()["auth"]
    assert auth["console_mode"] and auth["console_ready"] and auth["management_allowed"]
    assert auth["account"]["authenticated"] is False
    return response


def create(client, account_id="bobby"):
    body = {"display_name": "봄"}
    if account_id is not None:
        body["account_id"] = account_id
    response = client.post("/api/auth/accounts", json=body, headers=HEADERS)
    assert response.status_code == 201, response.text
    return response.json()["account"]


def test_status_bootstraps_private_operator_cookie_without_creating_an_account(console):
    client, service, _ = console
    response = connect(client)
    cookie = response.headers["set-cookie"].lower()
    assert "httponly" in cookie and "samesite=strict" in cookie and "max-age=7200" in cookie
    secret = client.cookies.get(CONSOLE_COOKIE)
    assert service.console_session(secret)
    assert response.json()["auth"]["console_session_id"] == service.accounts.digest(secret)[:16]
    assert "이 페이지를 사용하는 사람이 조회·삭제" in response.json()["auth"]["consent"]["text"]
    assert client.cookies.get(COOKIE) is None
    assert service.accounts.list_accounts() == []
    assert "set-cookie" not in connect(client).headers
    assert client.cookies.get(CONSOLE_COOKIE) == secret
    with service.accounts.connect() as db:
        assert db.execute("SELECT count(*) FROM sessions").fetchone()[0] == 0


def test_status_bootstrap_rejects_cross_site_and_cross_origin(console):
    client, service, _ = console
    for headers in ({"Origin": "https://foreign.invalid"}, {"Sec-Fetch-Site": "cross-site"}):
        response = client.get("/api/status", headers=headers)
        assert response.status_code == 403
        assert client.cookies.get(CONSOLE_COOKIE) is None
    assert not service.console_session(None)


def test_management_requires_issued_console_session_and_ignores_old_login(console):
    client, service, _ = console
    service.accounts.create("alice", CREDENTIALS["password"])
    service.accounts.bootstrap_admin("alice")
    client.cookies.set(COOKIE, service.accounts.issue("alice"))
    assert client.get("/api/auth/accounts").status_code == 401
    client.cookies.set(CONSOLE_COOKIE, "forged")
    assert client.get("/api/auth/accounts").status_code == 401
    client.cookies.clear()
    connect(client)
    assert client.get("/api/auth/accounts").status_code == 200
    assert service.accounts.get("alice")["is_admin"]


def test_password_free_account_create_generates_id_and_never_returns_credentials(console):
    client, service, _ = console
    connect(client)
    secret = client.cookies.get(CONSOLE_COOKIE)
    created = create(client, None)
    assert created["account_id"].startswith("person-")
    assert created["display_name"] == "봄" and created["enabled"] and not created["is_admin"]
    assert not {"password", "password_hash", "salt", "session", "token"}.intersection(created)
    assert client.cookies.get(CONSOLE_COOKIE) == secret
    assert len(service.accounts.list_accounts()) == 1
    assert service.accounts.account(secret) is None


def test_console_ignores_supplied_password_and_rejects_nameless_accounts(console):
    client, service, _ = console
    connect(client)
    response = client.post("/api/auth/accounts", json=CREDENTIALS, headers=HEADERS)
    assert response.status_code == 422
    response = client.post("/api/auth/accounts", json={**CREDENTIALS, "display_name": "봄"}, headers=HEADERS)
    assert response.status_code == 201
    with pytest.raises(HTTPException) as error:
        service.accounts.login("alice", CREDENTIALS["password"])
    assert error.value.status_code == 401


@pytest.mark.parametrize("headers", [{}, {**HEADERS, "Origin": "https://foreign.invalid"},
                                    {**HEADERS, "Sec-Fetch-Site": "cross-site"}])
def test_console_mutations_keep_origin_and_custom_header_guards(console, headers):
    client, service, _ = console
    connect(client)
    create(client)
    operations = [
        ("/accounts", {"account_id": "charlie", "display_name": "다른 이름"}),
        ("/accounts/bobby/profile", {"display_name": "다른 이름"}),
        ("/accounts/bobby/status", {"enabled": False}),
        ("/accounts/bobby/delete", {"confirm_account_id": "bobby"}),
        ("/accounts/bobby/photo/delete", {}),
        ("/accounts/bobby/enroll", CONSENT),
        ("/accounts/bobby/authenticate", CONSENT),
    ]
    for path, body in operations:
        assert client.post("/api/auth" + path, json=body, headers=headers).status_code == 403
    assert service.accounts.get("bobby")["enabled"]
    assert service.accounts.get("charlie") is None


def test_console_can_manage_and_remove_even_the_only_legacy_admin(console):
    client, service, backend = console
    service.accounts.create("alice", CREDENTIALS["password"])
    service.accounts.bootstrap_admin("alice")
    old_login = service.accounts.issue("alice")
    connect(client)
    assert client.post("/api/auth/accounts/alice/profile", json={"display_name": "변경"}, headers=HEADERS).status_code == 200
    result = client.post("/api/auth/accounts/alice/status", json={"enabled": False}, headers=HEADERS)
    assert result.status_code == 200 and result.json()["account"]["enabled"] is False
    assert service.accounts.account(old_login) is None and not backend.account_enabled["alice"]
    assert client.post("/api/auth/accounts/alice/status", json={"enabled": True}, headers=HEADERS).status_code == 200
    assert client.post("/api/auth/accounts/alice/delete", json={"confirm_account_id": "alice"}, headers=HEADERS).status_code == 200
    assert service.accounts.list_accounts() == [] and backend.account_faces["alice"] == "revoked"
    assert client.get("/api/auth/accounts").status_code == 200


def test_console_photo_read_delete_uses_same_operator_cookie(console):
    client, service, _ = console
    connect(client)
    create(client)
    service.accounts.store_photo("bobby", synthetic_photo(), **PHOTO_CONSENT)
    response = client.get("/api/auth/accounts/bobby/photo")
    assert response.status_code == 200 and response.headers["content-type"] == "image/jpeg"
    assert response.headers["cache-control"].startswith("no-store")
    assert client.post("/api/auth/accounts/bobby/photo/delete", json={}, headers=HEADERS).status_code == 200
    assert service.accounts.photo("bobby") is None


def test_console_enrollment_asserts_selected_account_as_admin_without_user_actor(console):
    client, service, backend = console
    connect(client)
    create(client)
    response = client.post("/api/auth/accounts/bobby/enroll", json=CONSENT, headers=HEADERS)
    assert response.status_code == 202
    outcome = finish(client, response.json())
    assert outcome["state"] == "succeeded", outcome
    claims = token.verify(backend.key.public_key(), backend.enrollment["assertion"])
    assert claims["acct"] == "bobby" and claims["method"] == "admin"
    op = service.operations[outcome["id"]]
    assert op["actor"] is None and op["console"] is True
    assert service.accounts.photo("bobby") is None


@pytest.mark.parametrize("matched", [True, False])
def test_console_authentication_still_checks_approval_target(console, matched):
    client, _, backend = console
    connect(client)
    create(client)
    backend.account = "bobby" if matched else "someone-else"
    response = client.post("/api/auth/accounts/bobby/authenticate", json=CONSENT, headers=HEADERS)
    assert response.status_code == 202
    outcome = finish(client, response.json())
    assert outcome["state"] == ("succeeded" if matched else "error"), outcome
    if not matched:
        assert outcome["result"]["reason"] == "approval_scope_mismatch"


def test_console_target_is_explicit_active_and_consented(console):
    client, _, _ = console
    connect(client)
    create(client)
    for path in ("/enroll", "/authenticate"):
        assert client.post("/api/auth" + path, json=CONSENT, headers=HEADERS).status_code == 400
    for action in ("enroll", "authenticate"):
        assert client.post("/api/auth/accounts/absent/" + action, json=CONSENT, headers=HEADERS).status_code == 403
        assert client.post("/api/auth/accounts/bobby/" + action,
                           json={"consent": {"version": "v1", "agreed": False}}, headers=HEADERS).status_code == 400
    client.post("/api/auth/accounts/bobby/status", json={"enabled": False}, headers=HEADERS)
    for action in ("enroll", "authenticate"):
        assert client.post("/api/auth/accounts/bobby/" + action, json=CONSENT, headers=HEADERS).status_code == 403


def test_console_operations_are_isolated_between_browsers(console):
    client, service, backend = console
    connect(client)
    create(client)
    backend.block = threading.Event()
    request_body = {**CONSENT, "client_request_id": "separate-browser-001"}
    response = client.post("/api/auth/accounts/bobby/authenticate", json=request_body, headers=HEADERS)
    assert response.status_code == 202
    operation_id = response.json()["id"]
    second = TestClient(client.app)
    try:
        connect(second)
        assert client.cookies.get(CONSOLE_COOKIE) != second.cookies.get(CONSOLE_COOKIE)
        assert second.get("/api/status").json()["auth"]["operation_id"] is None
        assert second.get("/api/auth/operation", params={"id": operation_id}).status_code == 404
        assert second.post("/api/auth/cancel", json={"id": operation_id}, headers=HEADERS).status_code == 404
        assert second.post("/api/auth/accounts/bobby/authenticate", json=request_body, headers=HEADERS).status_code == 409
        retry = client.post("/api/auth/accounts/bobby/authenticate", json=request_body, headers=HEADERS)
        assert retry.status_code == 202 and retry.json()["id"] == operation_id
        assert client.post("/api/auth/cancel", json={"id": operation_id}, headers=HEADERS).status_code == 200
    finally:
        backend.block.set()
        second.close()
    assert finish(client, response.json())["state"] == "cancelled"


def test_console_session_tampering_expiry_and_target_disable_cancel_capture(console, monkeypatch):
    client, service, _ = console
    connect(client)
    create(client)
    secret = client.cookies.get(CONSOLE_COOKIE)
    assert not service.console_session(secret[:-1] + ("a" if secret[-1] != "a" else "b"))
    op = {"console": True, "actor": None, "account": "bobby", "cancel": threading.Event()}
    service._check(op, secret)
    service.accounts.set_enabled("bobby", False)
    with pytest.raises(CaptureCancelled):
        service._check(op, secret)
    service.accounts.set_enabled("bobby", True)
    now = time.time()
    monkeypatch.setattr("faceauth.terminal.web_auth.time.time", lambda: now + 7201)
    assert not service.console_session(secret)
    with pytest.raises(CaptureCancelled):
        service._check(op, secret)
    assert client.get("/api/auth/accounts").status_code == 401
    connect(client)
    assert client.cookies.get(CONSOLE_COOKIE) != secret


def test_console_restart_changes_operator_identity_without_changing_accounts(console):
    client, service, _ = console
    before = connect(client).json()["auth"]["console_session_id"]
    created = create(client)
    service._console_key = b"synthetic-restarted-process-key-32"
    assert client.get("/api/auth/accounts").status_code == 401
    after = connect(client).json()["auth"]["console_session_id"]
    assert before != after
    assert service.accounts.get(created["account_id"]) == created


def test_console_observation_requires_selected_target(console):
    client, _, _ = console
    connect(client)
    create(client)
    body = {**CONSENT, "kind": "live", "split": "calibration", "subject_id": "synthetic-001"}
    assert client.post("/api/auth/observe", json=body, headers=HEADERS).status_code == 400
    response = client.post("/api/auth/observe", json={**body, "target_account_id": "bobby"}, headers=HEADERS)
    assert response.status_code == 202
    assert finish(client, response.json())["state"] == "succeeded"


def test_console_has_no_account_login_signup_or_logout_flow(console):
    client, service, _ = console
    connect(client)
    for path in ("login", "signup"):
        assert client.post("/api/auth/" + path, json=CREDENTIALS, headers=HEADERS).status_code == 409
    assert client.post("/api/auth/logout", headers=HEADERS).status_code == 409
    assert service.accounts.list_accounts() == []
