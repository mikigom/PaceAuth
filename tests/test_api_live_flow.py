"""Multi-person demo API through an isolated real mTLS gateway and contract-fake model/search services."""
import httpx
import pytest

from tests import test_api_flow as f

pytestmark = f.pytestmark
stack, dev = f.stack, f.dev


def live_frame(people):
    return f.frame("unused", people=people)


def live(dev, session, people, **extra):
    packet = dev.packet(session, [live_frame(people)], extra={"capture_purpose": "auth", **extra})
    return dev.http.post("/v1/live/recognize", content=packet, headers={"Content-Type": "application/msgpack"})


def person(ident, box=None, **kw):
    return {"id": ident, "bbox": box or [100, 80, 350, 390], "noise": 0, **kw}


def test_live_identifies_each_face_independently_and_never_issues_approval(dev):
    a, b = f.uid("live-a"), f.uid("live-b")
    _, acct_a, _ = f.enroll(dev, a)
    _, acct_b, _ = f.enroll(dev, b)
    before = (f.q1("SELECT count(*) FROM templates")[0], f.q1("SELECT count(*) FROM approvals")[0])
    session = dev.session("auth", "live-demo")
    people = [person(a), person(b, [370, 120, 600, 440]), person(f.uid("unknown"), [0, 0, 80, 80]),
              person(a, [0, 120, 70, 210], reasons=["blur"])]
    for _ in range(2):
        response = live(dev, session, people)
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["demo"] is True and body["pad_enabled"] is False and body["pad_mode"] == "disabled_demo"
        assert body["image_size"] == [640, 480] and body["detected_faces"] == 4 and not body["truncated"]
        by_box = {tuple(x["bbox"]): x for x in body["faces"]}
        assert by_box[tuple(people[0]["bbox"])]["account_id"] == acct_a
        assert by_box[tuple(people[1]["bbox"])]["account_id"] == acct_b
        assert by_box[tuple(people[2]["bbox"])]["status"] == "unknown"
        assert by_box[tuple(people[3]["bbox"])]["status"] == "uncertain"
        assert all("account_id" not in x for x in body["faces"] if x["status"] != "known")
        assert not set(k for k, _ in f._walk(body)) & {"emb", "score", "scores", "top1", "top2", "approval_token",
                                                    "subject_id", "rgb", "depth", "ir", "kps"}
    assert f.q1("SELECT consumed_at FROM sessions WHERE id=%s", session["session_id"])[0] is None
    assert before == (f.q1("SELECT count(*) FROM templates")[0], f.q1("SELECT count(*) FROM approvals")[0])


def test_live_no_face_and_ambiguous_are_distinct(dev):
    session = dev.session("auth", "live-demo")
    out = live(dev, session, [])
    assert out.status_code == 200 and out.json()["faces"] == []
    a, b = f.uid("amb-a"), f.uid("amb-b")
    f.enroll(dev, a)
    f.enroll(dev, b)
    out = live(dev, session, [person({a: 1, b: 1})])
    face = out.json()["faces"][0]
    assert face["status"] == "uncertain" and face["reason"] == "ambiguous" and "account_id" not in face


def test_live_requires_bound_mtls_signed_dedicated_session(dev):
    session = dev.session("auth", "live-demo")
    other = f.new_device()
    assert live(other, session, [person("x")]).status_code == 409
    assert live(dev, dev.session(), [person("x")]).status_code == 409
    assert live(dev, dev.session("enroll", "live-demo"), [person("x")], capture_purpose="enroll").status_code == 409
    packet = bytearray(dev.packet(session, [live_frame([person("x")])], extra={"capture_purpose": "auth"}))
    packet[-1] ^= 1
    assert dev.http.post("/v1/live/recognize", content=bytes(packet)).status_code == 400
    # Direct API traffic cannot forge the gateway's certificate headers.
    packet = dev.packet(session, [live_frame([person("x")])], extra={"capture_purpose": "auth"})
    assert httpx.post(f.API + "/v1/live/recognize", content=packet).status_code == 401
    assert dev.http.post("/v1/authentications", content=dev.packet(session, [f.frame("x")] * 3)).status_code == 409
    assert dev.http.post("/v1/fallback", json={"session_id": session["session_id"],
        "assertion": f.assertion("account", session["challenge"], "password")}).status_code == 409
    assert f.q1("SELECT consumed_at FROM sessions WHERE id=%s", session["session_id"])[0] is None


def test_live_hides_inactive_and_nonconsenting_subjects_even_if_index_is_stale(dev):
    a, b = f.uid("revoked-live"), f.uid("consent-live")
    ra, _, _ = f.enroll(dev, a)
    rb, _, _ = f.enroll(dev, b)
    with f.db() as conn:
        conn.execute("UPDATE subjects SET status='revoked' WHERE id=%s", (ra.json()["subject_id"],))
        conn.execute("UPDATE subjects SET consent_version='obsolete' WHERE id=%s", (rb.json()["subject_id"],))
    body = live(dev, dev.session("auth", "live-demo"), [person(a), person(b)]).json()
    assert all(x["status"] == "unknown" and "account_id" not in x for x in body["faces"])


def test_live_engine_failures_and_mode_mismatch_fail_closed(dev):
    session = dev.session("auth", "live-demo")
    try:
        f.fakes(engine_fail=True)
        assert live(dev, session, []).status_code == 503
        f.fakes(engine_fail=False, pad_enabled=False)
        assert live(dev, session, []).status_code == 503
    finally:
        f.fakes(engine_fail=False, pad_enabled=True)


def test_live_skips_pad_validation_but_reusable_session_still_expires(dev):
    session = dev.session("auth", "live-demo")
    try:
        f.fakes(pad_ready=False)
        assert live(dev, session, []).status_code == 200
        with f.db() as conn:
            conn.execute("UPDATE sessions SET expires_at=now()-interval '1 second' WHERE id=%s",
                         (session["session_id"],))
        assert live(dev, session, []).status_code == 409
        assert f.q1("SELECT consumed_at FROM sessions WHERE id=%s", session["session_id"])[0] is None
    finally:
        f.fakes(pad_ready=True)


def test_auth_rejects_identity_switch_when_largest_person_changes(dev):
    a, b = f.uid("switch-a"), f.uid("switch-b")
    f.enroll(dev, a)
    f.enroll(dev, b)
    big, small = [100, 50, 400, 400], [450, 100, 550, 220]
    frames = [live_frame([person(a, big), person(b, small)]),
              live_frame([person(a, small), person(b, big)]),
              live_frame([person(a, big), person(b, small)])]
    session = dev.session()
    response = dev.post("/v1/authentications", content=dev.packet(session, frames))
    assert response.json()["decision"] == "rejected" and response.json()["reason"] == "retry"
    assert f.reason_of(session) == "same_person" and "approval_token" not in response.json()


def test_auth_largest_quality_failure_does_not_fall_back_to_smaller_match(dev):
    ident = f.uid("largest-invalid")
    f.enroll(dev, ident)
    people = [person(ident, [100, 50, 500, 460], reasons=["blur"]), person(ident, [0, 0, 80, 80])]
    session = dev.session()
    response = dev.post("/v1/authentications", content=dev.packet(session, [live_frame(people)] * 3))
    assert response.json()["decision"] == "rejected" and f.reason_of(session) == "quality:blur"


def test_live_face_count_cap_retains_total_and_truncation(dev):
    response = live(dev, dev.session("auth", "live-demo"), [person(f.uid("cap")) for _ in range(20)])
    body = response.json()
    assert len(body["faces"]) == 16 and body["detected_faces"] == 20 and body["truncated"] is True
