"""Account-free 1:N operations still require a scoped, signed face approval."""
import json
import threading
import time

import pytest

from faceauth.common import token
from faceauth.terminal.web_auth import CONSOLE_COOKIE
from test_open_console import connect, console, create
from test_web_auth import CONSENT, HEADERS, finish, setup, signup

START = {"start": True}


def begin(client, **fields):
    response = client.post("/api/auth/identify", json={**START, **fields}, headers=HEADERS)
    assert response.status_code == 202, response.text
    return response.json()


def assert_no_identity(outcome):
    assert "matched_account_id" not in (outcome.get("result") or {})
    assert "matched_display_name" not in (outcome.get("result") or {})
    assert "approval_token" not in json.dumps(outcome)


@pytest.mark.parametrize("account_id", ["alice", "bobby"])
def test_identification_returns_whichever_registered_person_was_approved(console, account_id):
    client, service, backend = console
    connect(client)
    create(client, "alice")
    create(client, "bobby")
    service.accounts.rename("alice", "첫 번째 사람")
    service.accounts.rename("bobby", "두 번째 사람")
    backend.account = account_id
    operation = begin(client)
    assert operation["kind"] == "identify"
    assert operation["target_account_id"] is None and operation["target_display_name"] is None
    assert_no_identity(operation)
    outcome = finish(client, operation)
    assert outcome["state"] == "succeeded", outcome
    assert outcome["result"] == {"decision": "approved", "matched_account_id": account_id,
                                  "matched_display_name": service.accounts.get(account_id)["display_name"]}
    assert "approval_token" not in json.dumps(outcome)
    assert backend.submitted == 1
    # The start button represents an operator request, not a participant's
    # biometric-consent assertion or a new photo/face enrollment.
    internal = service.operations[outcome["id"]]
    assert "consent_at" not in internal
    assert "consent" not in internal and "photo_consent" not in internal
    assert service.accounts.photo(account_id) is None
    assert not hasattr(backend, "enrollment")


@pytest.mark.parametrize("decision", ["rejected", "step_up"])
def test_rejection_never_exposes_unverified_identity_even_when_backend_supplies_one(console, decision):
    client, _, backend = console
    connect(client)
    backend.authenticate = lambda *a, **k: {
        "decision": decision, "reason": "no_match", "matched_account_id": "injected",
        "matched_display_name": "Not a verified identity", "approval_token": "unverified"}
    outcome = finish(client, begin(client))
    assert outcome["state"] == "rejected" and outcome["result"]["decision"] == decision
    assert_no_identity(outcome)


@pytest.mark.parametrize("change", ["dev", "act", "sid", "unsigned", "no_face", "unknown", "disabled", "malformed_account"])
def test_identification_rejects_unscoped_or_ineligible_approval(console, change):
    client, service, backend = console
    connect(client)
    create(client, "alice")
    if change == "disabled":
        service.accounts.set_enabled("alice", False)

    def approve(session, *args, **kwargs):
        now = int(time.time())
        claims = {"dev": "dev-001", "act": "pilot-check", "sid": session["session_id"],
                  "acct": "alice", "amr": ["face"], "iat": now, "exp": now + 60}
        if change in ("dev", "act", "sid"):
            claims[change] = "wrong-scope"
        if change == "no_face":
            claims["amr"] = ["password"]
        if change == "unknown":
            claims["acct"] = "missing-person"
        if change == "malformed_account":
            claims["acct"] = ["alice"]
        signed = "not-a-signed-token" if change == "unsigned" else token.sign(backend.key, claims)
        return {"decision": "approved", "approval_token": signed}

    backend.authenticate = approve
    outcome = finish(client, begin(client))
    assert outcome["state"] == "error", outcome
    assert_no_identity(outcome)


def test_identification_retains_pad_mode_and_refuses_a_changed_mode(console):
    client, service, backend = console
    connect(client)
    create(client, "alice")
    service.backend.update(pad_enabled=False, pad_mode="disabled_pilot")
    original = backend.authenticate
    backend.authenticate = lambda *a, **k: {**original(*a, **k), "pad_enabled": False, "pad_mode": "disabled_pilot"}
    outcome = finish(client, begin(client))
    assert outcome["state"] == "succeeded"
    assert outcome["pad_enabled"] is False and outcome["result"]["pad_mode"] == "disabled_pilot"
    backend.authenticate = lambda *a, **k: {**original(*a, **k), "pad_enabled": True, "pad_mode": "enabled"}
    outcome = finish(client, begin(client))
    assert outcome["state"] == "error" and outcome["result"]["reason"] == "approval_scope_mismatch"
    assert_no_identity(outcome)


def test_identification_requires_console_session_and_request_origin(console):
    client, _, backend = console
    assert client.post("/api/auth/identify", json=START, headers=HEADERS).status_code == 401
    connect(client)
    assert client.post("/api/auth/identify", json=START).status_code == 403
    assert client.post("/api/auth/identify", json=START,
                       headers={**HEADERS, "Origin": "https://foreign.invalid"}).status_code == 403
    assert backend.created == backend.submitted == 0


@pytest.mark.parametrize("body", [{}, {"start": False}, {"start": 1}, {"start": 0},
                                 {"start": "true"}, {"start": None}, CONSENT,
                                 {**START, **CONSENT}, {**START, "photo_consent": True}])
def test_identification_requires_an_explicit_true_start_without_consent_claims(console, body):
    client, service, backend = console
    connect(client)
    assert client.post("/api/auth/identify", json=body, headers=HEADERS).status_code == 422
    assert service.active is None and not service.operations
    assert backend.created == backend.submitted == 0


@pytest.mark.parametrize("action", ["enroll", "authenticate", "observe"])
@pytest.mark.parametrize("consent", [None, {"version": "v1", "agreed": False},
                                    {"version": "outdated", "agreed": True}])
def test_operator_start_does_not_replace_consent_for_other_capture_flows(console, action, consent):
    client, service, backend = console
    connect(client)
    create(client, "alice")
    body = {"start": True}
    path = "/api/auth/accounts/alice/" + action
    if action == "observe":
        path = "/api/auth/observe"
        body.update(target_account_id="alice", kind="live", split="calibration", subject_id="subject-001")
    if consent is not None:
        body["consent"] = consent
    assert client.post(path, json=body, headers=HEADERS).status_code == (422 if consent is None else 400)
    assert service.active is None and not service.operations
    assert backend.created == backend.submitted == 0


@pytest.mark.parametrize("field", ["target_account_id", "account_id"])
def test_identification_rejects_accidentally_selected_account_fields(console, field):
    client, _, backend = console
    connect(client)
    assert client.post("/api/auth/identify", json={**START, field: "alice"}, headers=HEADERS).status_code == 422
    assert backend.created == backend.submitted == 0


def test_person_disabled_while_final_approval_is_returning_cannot_be_published(console):
    client, service, backend = console
    connect(client)
    create(client, "alice")
    approval_ready, allow_return = threading.Event(), threading.Event()
    original = backend.authenticate

    def delayed_approval(*args, **kwargs):
        out = original(*args, **kwargs)
        approval_ready.set()
        assert allow_return.wait(3)
        return out

    backend.authenticate = delayed_approval
    operation = begin(client)
    try:
        assert approval_ready.wait(3)
        response = client.post("/api/auth/accounts/alice/status", json={"enabled": False}, headers=HEADERS)
        assert response.status_code == 409  # Normal management already excludes an active capture.
        # Simulate a separate administrative state change after the API decision.
        with service.management_lock:
            service.accounts.set_enabled("alice", False)
    finally:
        allow_return.set()
    outcome = finish(client, operation)
    assert outcome["state"] == "error" and outcome["result"]["reason"] == "approval_scope_mismatch"
    assert_no_identity(outcome)


def test_password_login_mode_keeps_targeted_authentication_boundary(setup):
    client, _, _ = setup
    signup(client)
    assert client.post("/api/auth/identify", json=START, headers=HEADERS).status_code == 403


def test_identification_retries_reuse_operation_and_account_free_capture_can_cancel(console):
    client, service, backend = console
    connect(client)
    backend.block = threading.Event()
    try:
        op = begin(client, client_request_id="identify-request-001")
        retry = begin(client, client_request_id="identify-request-001")
        assert retry["id"] == op["id"]
        assert client.post("/api/auth/identify", json={**START, "client_request_id": "identify-request-002"},
                           headers=HEADERS).status_code == 409
        assert client.post("/api/auth/cancel", json={"id": op["id"]}, headers=HEADERS).status_code == 200
    finally:
        backend.block.set()
    outcome = finish(client, op)
    assert outcome["state"] == "cancelled" and backend.submitted == 0
    assert_no_identity(outcome)


def test_operator_expiry_cancels_account_free_capture(console, monkeypatch):
    from faceauth.terminal.web_auth import CaptureCancelled

    client, service, _ = console
    connect(client)
    secret = client.cookies.get(CONSOLE_COOKIE)
    op = {"kind": "identify", "console": True, "actor": None, "account": None, "cancel": threading.Event()}
    service._check(op, secret)
    now = time.time()
    monkeypatch.setattr("faceauth.terminal.web_auth.time.time", lambda: now + 7201)
    with pytest.raises(CaptureCancelled):
        service._check(op, secret)


@pytest.mark.parametrize("timeout_ms", [0, -1, 10001, True, False, 1.5, "1000", None])
def test_identification_rejects_invalid_run_budgets(console, timeout_ms):
    client, service, backend = console
    connect(client)
    response = client.post("/api/auth/identify", json={**START, "timeout_ms": timeout_ms}, headers=HEADERS)
    assert response.status_code == 422
    assert service.active is None and backend.created == backend.submitted == 0


@pytest.mark.parametrize("timeout_ms", [None, 2500])
def test_identification_passes_only_remaining_budget_to_central_session(console, timeout_ms):
    client, service, backend = console
    connect(client)
    create(client, "alice")
    started = time.monotonic()
    operation = begin(client, **({"timeout_ms": timeout_ms} if timeout_ms else {}))
    outcome = finish(client, operation)
    assert outcome["state"] == "succeeded"
    budget = timeout_ms or 10000
    assert 0 < backend.session_timeout_ms <= budget
    assert started < service.operations[operation["id"]]["_deadline"] <= time.monotonic() + budget / 1000
    assert backend.deadline_monotonic == service.operations[operation["id"]]["_deadline"]
    assert "_deadline" not in outcome and "deadline_monotonic" not in outcome


def test_identification_expiring_during_capture_never_submits(console):
    client, service, backend = console
    connect(client)
    reached, allow_return = threading.Event(), threading.Event()
    original = backend.capture_guide

    def delayed_guide(*args, **kwargs):
        reached.set()
        assert allow_return.wait(3)
        return original(*args, **kwargs)

    backend.capture_guide = delayed_guide
    operation = begin(client)
    try:
        assert reached.wait(3)
        service.operations[operation["id"]]["_deadline"] = time.monotonic() - 0.001
    finally:
        allow_return.set()
    outcome = finish(client, operation)
    assert outcome["state"] == "expired" and outcome["result"] == {"reason": "identification_expired"}
    assert backend.submitted == 0
    assert_no_identity(outcome)


@pytest.mark.parametrize("stop", ["expired", "cancelled"])
def test_late_final_approval_cannot_publish_identity_after_run_ends(console, stop):
    client, service, backend = console
    connect(client)
    create(client, "alice")
    reached, allow_return = threading.Event(), threading.Event()
    original = backend.authenticate

    def delayed_approval(*args, **kwargs):
        out = original(*args, **kwargs)
        reached.set()
        assert allow_return.wait(3)
        return out

    backend.authenticate = delayed_approval
    operation = begin(client)
    try:
        assert reached.wait(3)
        if stop == "expired":
            service.operations[operation["id"]]["_deadline"] = time.monotonic() - 0.001
        else:
            response = client.post("/api/auth/cancel", json={"id": operation["id"]}, headers=HEADERS)
            assert response.status_code == 200
            # The in-flight worker still owns the camera until its response returns.
            assert service.active == operation["id"]
    finally:
        allow_return.set()
    outcome = finish(client, operation)
    assert outcome["state"] == stop
    assert_no_identity(outcome)
    assert (outcome.get("result") or {}).get("decision") != "approved"
    assert backend.submitted == 1


def test_camera_wait_is_capped_to_remaining_identification_budget(console, monkeypatch):
    from faceauth.terminal.web_auth import IdentificationExpired

    client, service, _ = console
    connect(client)
    secret = client.cookies.get(CONSOLE_COOKIE)
    op = {"kind": "identify", "console": True, "actor": None, "account": None,
          "cancel": threading.Event(), "_deadline": time.monotonic() + 0.05}
    waits = []

    def no_frame(after, timeout):
        waits.append(timeout)
        op["_deadline"] = time.monotonic() - 0.001
        return None

    monkeypatch.setattr(service.camera, "capture_after", no_frame)
    with pytest.raises(IdentificationExpired):
        list(service._frames(op, secret, {}, 15))
    assert len(waits) == 1 and 0 < waits[0] <= 0.05
