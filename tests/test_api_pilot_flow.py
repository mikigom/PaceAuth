"""Opt-in API integration tests, run only against an isolated PAD-disabled contract-fake stack."""
import os

import pytest

from tests import test_api_flow as f

pytestmark = pytest.mark.skipif(os.environ.get("FA_TEST_PILOT_DISABLE_PAD") != "1" or not f._up(),
                                reason="requires isolated PAD-disabled apitest stack")
stack, dev = f.stack, f.dev


@pytest.fixture(autouse=True)
def unvalidated_pad():
    f.fakes(pad_ready=False, pad_enabled=False)
    yield
    f.fakes(pad_ready=True, pad_enabled=False)


def test_pilot_enroll_auth_skip_pad_and_mark_response_token_audit(dev):
    ident = f.uid("pilot")
    frames = [f.frame(ident, f.POSES[d], live=False) for d in f.POSES for _ in range(2)]
    r, acct, enrollment = f.enroll(dev, ident, frames=frames)
    assert r.json()["status"] == "active", r.text
    assert r.json()["pad_enabled"] is False and r.json()["pad_mode"] == "disabled_pilot"
    state = dev.http.get("/v1/terminal/readiness", params={"device_id": dev.id}).json()
    assert state["ready"] is True and state["pad_validation"]["ready"] is False
    r, session = f.auth(dev, ident, live=False)
    assert r.json()["decision"] == "approved", r.text
    assert r.json()["pad_enabled"] is False and r.json()["pad_mode"] == "disabled_pilot"
    claims = f.check_approval(r.json()["approval_token"], dev, "pay", acct, ["face"], session)
    assert claims["pad_enabled"] is False and claims["pad_mode"] == "disabled_pilot"
    for item in (enrollment, session):
        detail = f.q1("SELECT detail FROM audit_log WHERE session_id=%s ORDER BY id DESC LIMIT 1", item["session_id"])[0]
        assert detail["pad_enabled"] is False and detail["pad_mode"] == "disabled_pilot"


def test_pilot_preserves_identity_quality_coverage_and_packet_integrity(dev):
    ident = f.uid("pilot-checks")
    r, acct, _ = f.enroll(dev, ident)
    assert r.json()["status"] == "active", r.text
    r, session = f.auth(dev, f.uid("other"))
    assert r.json()["decision"] == "rejected" and r.json()["reason"] == "not_recognized"
    r, session = f.auth(dev, ident, faces=2)
    assert r.json()["decision"] == "approved" and f.reason_of(session) == "ok"
    r, _, session = f.enroll(dev, f.uid("coverage"), frames=f.enroll_frames(ident, dirs=("C",)))
    assert r.json()["status"] == "rejected" and f.reason_of(session) == "coverage"
    other = f.uid("inconsistent")
    frames = f.enroll_frames(other)
    frames[0] = f.frame(ident)
    r, _, session = f.enroll(dev, other, frames=frames)
    assert r.json()["status"] == "rejected" and f.reason_of(session) == "same_person"
    session = dev.session()
    packet = bytearray(dev.packet(session, [f.frame(ident)] * 3))
    packet[-1] ^= 1
    r = dev.post("/v1/authentications", content=bytes(packet))
    assert r.json()["decision"] == "rejected" and f.reason_of(session) == "hash_mismatch"


def test_pilot_mode_disagreement_fails_before_decision(dev):
    try:
        f.fakes(pad_enabled=True)
        state = dev.http.get("/v1/terminal/readiness", params={"device_id": dev.id}).json()
        assert not state["ready"] and not state["capture_ready"]
        assert state["blockers"][0]["code"] == "pad_mode_mismatch"
        r, session = f.auth(dev, "mismatch")
        assert r.status_code == 503
        assert f.q1("SELECT consumed_at FROM sessions WHERE id=%s", session["session_id"])[0] is None
    finally:
        f.fakes(pad_enabled=False)


def test_stepup_rejects_an_originating_face_factor_with_a_different_pad_mode(dev):
    ident = f.uid("pilot-risk")
    _, acct, _ = f.enroll(dev, ident)
    r, session = f.auth(dev, ident, action=f.HIGH_RISK)
    assert r.json()["decision"] == "step_up"
    stepup = r.json()["step_up_id"]
    # Isolated test DB fixture: model a face factor recorded while PAD was enabled.
    with f.db() as conn:
        conn.execute("INSERT INTO audit_log (actor,event,session_id,device_id,detail) VALUES "
                     "('test-fixture','auth_decision',%s,%s,%s::jsonb)",
                     (session["session_id"], dev.id,
                      '{"decision":"step_up","pad_enabled":true,"pad_mode":"enabled"}'))
    r = dev.post(f"/v1/stepups/{stepup}/complete", json={"assertion": f.assertion(acct, stepup, "otp")})
    assert r.json()["decision"] == "rejected" and "approval_token" not in r.json()
    assert f.q1("SELECT completed_at FROM stepups WHERE id=%s", stepup)[0] is None


def test_pilot_live_recognition_uses_unvalidated_pad_free_labels_without_authentication(dev):
    from tests.test_api_live_flow import live, person

    ident = f.uid("pilot-live")
    _, account, _ = f.enroll(dev, ident)
    session = dev.session("auth", "live-demo")
    response = live(dev, session, [person(ident), person(f.uid("pilot-live-unknown"))])
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["pad_enabled"] is False and body["pad_mode"] == "disabled_demo"
    assert [x["status"] for x in body["faces"]] == ["known", "unknown"]
    assert body["faces"][0]["account_id"] == account and "approval_token" not in body
    assert f.q1("SELECT consumed_at FROM sessions WHERE id=%s", session["session_id"])[0] is None
