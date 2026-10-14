"""Account lifecycle with real PostgreSQL/mTLS and synthetic model/search/KMS fixtures.

Run on the isolated apitest stack; never operates on real people or production accounts.
"""
import concurrent.futures
import time

import httpx
import pytest

from tests import test_api_flow as f
from tests import test_api_live_flow as live

pytestmark = f.pytestmark
stack, dev = f.stack, f.dev


def set_enabled(account, enabled):
    response = f.admin("POST", f"/v1/admin/accounts/{account}/status", json={"enabled": enabled})
    assert response.status_code == 200, response.text
    return response.json()


def summary(*accounts):
    response = f.admin("POST", "/v1/admin/accounts/summary", json={"account_ids": list(accounts)})
    assert response.status_code == 200, response.text
    return response.json()["accounts"]


def delete_face(account):
    return f.admin("DELETE", f"/v1/admin/accounts/{account}/face")


def test_management_requires_admin_and_bounds_input(dev):
    account = f.uid("access")
    for path, method, body in [("/v1/admin/accounts/summary", "POST", {"account_ids": [account]}),
                               (f"/v1/admin/accounts/{account}/status", "POST", {"enabled": False}),
                               (f"/v1/admin/accounts/{account}/face", "DELETE", None)]:
        assert dev.http.request(method, path, json=body).status_code == 401
    assert f.admin("POST", "/v1/admin/accounts/summary", json={"account_ids": ["x"] * 501}).status_code == 422
    assert f.admin("POST", f"/v1/admin/accounts/{account}/status", json={"enabled": "false"}).status_code == 422
    assert summary(account)[0]["face_status"] == "none"


def test_disable_hides_stale_index_from_live_auth_and_preserves_face(dev):
    ident = f.uid("disabled")
    response, account, _ = f.enroll(dev, ident)
    sid = response.json()["subject_id"]
    assert summary(account)[0]["face_status"] == "active"
    try:
        f.fakes(search_frozen=True)
        disabled = set_enabled(account, False)
        assert disabled["face_status"] == "disabled" and disabled["templates_changed"] == 4
        assert f.q1("SELECT count(*) FROM templates WHERE subject_id=%s AND NOT active "
                    "AND ciphertext IS NOT NULL", sid)[0] == 4
        assert f.auth(dev, ident)[0].json()["decision"] == "rejected"
        faces = live.live(dev, dev.session("auth", "live-demo"), [live.person(ident)]).json()["faces"]
        assert faces[0]["status"] == "unknown" and "account_id" not in faces[0]
        fallback = dev.session()
        result = dev.post("/v1/fallback", json={"session_id": fallback["session_id"],
                          "assertion": f.assertion(account, fallback["challenge"])})
        assert result.json()["decision"] == "rejected" and f.reason_of(fallback) == "account_disabled"
    finally:
        f.fakes(search_frozen=False)
    enabled = set_enabled(account, True)
    assert enabled["face_status"] == "active" and enabled["templates_changed"] == 4
    f.wait_index()
    assert f.auth(dev, ident)[0].json()["decision"] == "approved"


def test_disabled_account_without_face_cannot_enroll_or_fallback(dev):
    account = f.uid("no-face")
    assert set_enabled(account, False)["face_status"] == "none"
    response, _, session = f.enroll(dev, f.uid("face"), acct=account)
    assert response.json()["status"] == "rejected" and f.reason_of(session) == "account_disabled"
    session = dev.session()
    response = dev.post("/v1/fallback", json={"session_id": session["session_id"],
                        "assertion": f.assertion(account, session["challenge"])})
    assert response.json()["decision"] == "rejected"
    assert f.q1("SELECT count(*) FROM subjects WHERE account_id=%s", account)[0] == 0


def test_disable_blocks_previously_issued_stepup(dev):
    ident = f.uid("stepup")
    _, account, _ = f.enroll(dev, ident)
    result, _ = f.auth(dev, ident, action=f.HIGH_RISK)
    stepup = result.json()["step_up_id"]
    set_enabled(account, False)
    response = dev.post(f"/v1/stepups/{stepup}/complete", json={"assertion": f.assertion(account, stepup, "otp")})
    assert response.json()["decision"] == "rejected" and "approval_token" not in response.json()


def test_enable_only_current_templates_never_revoked_or_pending(dev):
    ident = f.uid("versions")
    response, account, _ = f.enroll(dev, ident)
    sid = response.json()["subject_id"]
    assert f.enroll(dev, ident, acct=account, re_enroll=True)[0].json()["template_version"] == 2
    set_enabled(account, False)
    set_enabled(account, True)
    assert f.qa("SELECT DISTINCT template_version FROM templates WHERE subject_id=%s AND active", sid) == [(2,)]
    assert delete_face(account).status_code == 200
    set_enabled(account, True)
    assert summary(account)[0]["face_status"] == "revoked"
    assert f.q1("SELECT count(*) FROM templates WHERE subject_id=%s AND active", sid)[0] == 0

    other = f.uid("pending")
    f.enroll(dev, other)
    response, pending, _ = f.enroll(dev, other)
    psid = response.json()["subject_id"]
    assert response.json()["status"] == "pending_review"
    set_enabled(pending, False)
    set_enabled(pending, True)
    assert summary(pending)[0]["face_status"] == "pending_review"
    assert f.q1("SELECT count(*) FROM templates WHERE subject_id=%s AND active", psid)[0] == 0
    rid = f.q1("SELECT id FROM enrollment_reviews WHERE subject_id=%s AND status='open'", psid)[0]
    response = f.admin("POST", f"/v1/admin/reviews/{rid}", json={"decision": "approve", "by": "test"})
    assert response.status_code == 200 and response.json()["subject_status"] == "active"
    assert f.q1("SELECT count(*) FROM templates WHERE subject_id=%s AND active", psid)[0] == 4


def test_delete_all_versions_including_disabled_is_idempotent(dev):
    ident = f.uid("delete-all")
    first, account, _ = f.enroll(dev, ident)
    assert delete_face(account).status_code == 200
    second, _, _ = f.enroll(dev, ident, acct=account)
    set_enabled(account, False)
    result = delete_face(account)
    assert result.status_code == 200, result.text
    assert result.json()["subjects_deleted"] == 2 and result.json()["key_destroyed"] is True
    assert result.json()["templates_deleted"] == 4 and result.json()["index_removed"] is True
    assert summary(account)[0]["enabled"] is False
    assert f.q1("SELECT count(*) FROM templates t JOIN subjects s ON s.id=t.subject_id "
                "WHERE s.account_id=%s AND (t.active OR t.ciphertext IS NOT NULL)", account)[0] == 0
    for sid in [first.json()["subject_id"], second.json()["subject_id"]]:
        assert httpx.get(f"{f.FAKES}/_test/kms/{sid}").json() == {"exists": False}
    assert delete_face(account).json()["templates_deleted"] == 0
    assert delete_face(f.uid("never-enrolled")).json()["key_destroyed"] is True


def test_review_approval_does_not_reenable_disabled_account(dev):
    ident = f.uid("review-disabled")
    f.enroll(dev, ident)
    response, account, _ = f.enroll(dev, ident)
    sid = response.json()["subject_id"]
    set_enabled(account, False)
    rid = f.q1("SELECT id FROM enrollment_reviews WHERE subject_id=%s AND status='open'", sid)[0]
    result = f.admin("POST", f"/v1/admin/reviews/{rid}", json={"decision": "approve", "by": "test"})
    assert result.status_code == 200 and result.json()["subject_status"] == "active"
    assert summary(account)[0]["face_status"] == "disabled"
    assert f.q1("SELECT count(*) FROM templates WHERE subject_id=%s AND active", sid)[0] == 0
    set_enabled(account, True)
    assert f.q1("SELECT count(*) FROM templates WHERE subject_id=%s AND active", sid)[0] == 4


def test_delete_kms_failure_is_not_success_and_retry_shreds_key(dev):
    ident = f.uid("kms-retry")
    response, account, _ = f.enroll(dev, ident)
    sid = response.json()["subject_id"]
    try:
        f.fakes(kms_fail=True)
        assert delete_face(account).status_code == 503
        assert summary(account)[0]["face_status"] == "revoked"
        assert f.auth(dev, ident)[0].json()["decision"] == "rejected"
    finally:
        f.fakes(kms_fail=False)
    assert delete_face(account).json()["key_destroyed"] is True
    assert httpx.get(f"{f.FAKES}/_test/kms/{sid}").json() == {"exists": False}


def _await_account_waiter():
    deadline = time.monotonic() + 10
    while not f.q1("SELECT count(*) FROM pg_locks WHERE locktype='advisory' AND classid=7003 AND NOT granted")[0]:
        assert time.monotonic() < deadline, "request did not reach its account commit lock"
        time.sleep(0.02)


@pytest.mark.parametrize("enabled", [False, True], ids=["disable", "face-reset"])
def test_lifecycle_change_rejects_already_processing_enrollment(dev, enabled):
    account, ident = f.uid("race-enroll"), f.uid("race-face")
    with concurrent.futures.ThreadPoolExecutor() as executor:
        with f.db() as conn, conn.transaction():
            conn.execute("SELECT pg_advisory_xact_lock(7003, hashtext(%s))", (account,))
            pending = executor.submit(f.enroll, dev, ident, account)
            _await_account_waiter()
            # Same locked write as lifecycle routes, while enrollment waits at its final commit.
            conn.execute("INSERT INTO account_controls (account_id, enabled, enrollment_epoch) VALUES (%s,%s,1)",
                         (account, enabled))
        response, _, session = pending.result(timeout=15)
    assert response.json()["status"] == "rejected"
    assert f.reason_of(session) == ("account_changed" if enabled else "account_disabled")
    assert f.q1("SELECT count(*) FROM subjects WHERE account_id=%s", account)[0] == 0


def test_disable_serializes_with_inflight_auth_commit(dev):
    ident = f.uid("race-auth")
    _, account, _ = f.enroll(dev, ident)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        with f.db() as conn, conn.transaction():
            conn.execute("SELECT pg_advisory_xact_lock(7003, hashtext(%s))", (account,))
            pending = executor.submit(f.auth, dev, ident)
            _await_account_waiter()
            conn.execute("INSERT INTO account_controls (account_id, enabled, enrollment_epoch) VALUES (%s,false,1)",
                         (account,))
        response, session = pending.result(timeout=15)
    assert response.json()["decision"] == "rejected" and "approval_token" not in response.json()
    assert f.q1("SELECT count(*) FROM approvals WHERE session_id=%s", session["session_id"])[0] == 0


def test_lifecycle_audit_chain_remains_valid():
    result = f.admin("GET", "/v1/admin/audit/verify")
    assert result.status_code == 200 and result.json()["ok"]
