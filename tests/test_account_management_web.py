"""Management authorization and camera flows use synthetic frames and an API fake."""
import threading
import time

import cv2
import numpy as np
import pytest

from faceauth.common import token
from faceauth.terminal.live_recognition import LiveRecognition
from faceauth.terminal.web_auth import COOKIE
from test_web_auth import CONSENT, HEADERS, finish, setup, signup


TARGET = {"account_id": "bobby", "password": "synthetic-target-password", "display_name": "봄"}
PHOTO_CONSENT = {"consented_at": "2026-10-14T01:00:00+00:00"}


def manager(fixture):
    client, service, backend = fixture
    signup(client)
    service.accounts.bootstrap_admin("alice")
    return client, service, backend


def create_target(client):
    response = client.post("/api/auth/accounts", json=TARGET, headers=HEADERS)
    assert response.status_code == 201, response.text
    return response.json()["account"]


def synthetic_photo():
    ok, photo = cv2.imencode(".jpg", np.zeros((48, 64, 3), np.uint8))
    assert ok
    return photo.tobytes()


def enroll_target(client, **extra):
    response = client.post("/api/auth/accounts/bobby/enroll", json={**CONSENT, **extra}, headers=HEADERS)
    assert response.status_code == 202, response.text
    return finish(client, response.json())


@pytest.mark.parametrize("authenticated,expected", [(False, 401), (True, 403)])
def test_all_management_routes_require_an_enabled_administrator(setup, authenticated, expected):
    client, service, _ = setup
    if authenticated:
        signup(client)
    service.accounts.create("bobby", TARGET["password"])
    endpoints = [
        ("GET", "/api/auth/accounts", None),
        ("GET", "/api/auth/accounts/bobby/photo", None),
        ("POST", "/api/auth/accounts", {**TARGET, "account_id": "charlie"}),
        ("POST", "/api/auth/accounts/bobby/profile", {"display_name": "변경 이름"}),
        ("POST", "/api/auth/accounts/bobby/status", {"enabled": False}),
        ("POST", "/api/auth/accounts/bobby/delete", {"confirm_account_id": "bobby"}),
        ("POST", "/api/auth/accounts/bobby/photo/delete", {}),
        ("POST", "/api/auth/accounts/bobby/enroll", CONSENT),
    ]
    for method, url, body in endpoints:
        response = client.request(method, url, json=body, headers=HEADERS)
        assert response.status_code == expected, (url, response.text)
    assert service.accounts.get("bobby")["enabled"]
    assert service.accounts.get("charlie") is None


def test_managed_creation_does_not_replace_manager_session_or_grant_admin(setup):
    client, service, backend = manager(setup)
    before = client.cookies.get(COOKIE)
    created = create_target(client)
    assert client.cookies.get(COOKIE) == before
    assert client.get("/api/status").json()["auth"]["account"]["account_id"] == "alice"
    assert created["display_name"] == "봄" and not created["is_admin"] and created["enabled"]
    assert backend.account_enabled["bobby"] is True
    assert service.accounts.account(service.accounts.login("bobby", TARGET["password"])) == "bobby"
    listing = client.get("/api/auth/accounts").json()
    assert listing["management_ready"] is True
    assert {row["account_id"] for row in listing["accounts"]} == {"alice", "bobby"}
    assert not {"password", "password_hash", "salt", "ciphertext"}.intersection(listing["accounts"][0])


def test_management_writes_reject_cross_origin_and_no_action_header(setup):
    client, _, _ = manager(setup)
    create_target(client)
    response = client.post("/api/auth/accounts/bobby/status", json={"enabled": False})
    assert response.status_code == 403
    response = client.post("/api/auth/accounts/bobby/profile", json={"display_name": "외부"},
                           headers={**HEADERS, "Origin": "https://outside.invalid"})
    assert response.status_code == 403


def test_rename_disable_and_reenable_refresh_labels_and_revoke_sessions(setup):
    client, service, backend = manager(setup)
    create_target(client)
    old_secret = service.accounts.login("bobby", TARGET["password"])
    response = client.post("/api/auth/accounts/bobby/profile", json={"display_name": "새 이름"}, headers=HEADERS)
    assert response.status_code == 200
    assert response.json()["account"]["display_name"] == "새 이름"
    assert service.accounts.names(["bobby"])["bobby"] == "새 이름"
    response = client.post("/api/auth/accounts/bobby/status", json={"enabled": False}, headers=HEADERS)
    assert response.status_code == 200
    assert response.json()["account"]["enabled"] is False and backend.account_enabled["bobby"] is False
    assert service.accounts.account(old_secret) is None and service.accounts.names(["bobby"]) == {}
    assert client.post("/api/auth/accounts/bobby/enroll", json=CONSENT, headers=HEADERS).status_code == 403
    assert client.post("/api/auth/accounts/bobby/status", json={"enabled": True}, headers=HEADERS).status_code == 200
    assert backend.account_enabled["bobby"] is True and service.accounts.account(old_secret) is None
    assert service.accounts.account(service.accounts.login("bobby", TARGET["password"])) == "bobby"


def test_current_manager_cannot_disable_or_delete_self(setup):
    client, service, backend = manager(setup)
    assert client.post("/api/auth/accounts/alice/status", json={"enabled": False}, headers=HEADERS).status_code == 409
    assert client.post("/api/auth/accounts/alice/delete", json={"confirm_account_id": "alice"},
                       headers=HEADERS).status_code == 409
    assert service.accounts.get("alice")["enabled"] and backend.account_enabled["alice"]


def test_delete_requires_correct_target_and_complete_backend_erasure(setup, monkeypatch):
    client, service, backend = manager(setup)
    create_target(client)
    secret = service.accounts.login("bobby", TARGET["password"])
    service.accounts.store_photo("bobby", synthetic_photo(), **PHOTO_CONSENT)
    saved = service.accounts.photo("bobby")
    assert client.post("/api/auth/accounts/bobby/delete", json={"confirm_account_id": "alice"},
                       headers=HEADERS).status_code == 400
    assert service.accounts.get("bobby")["enabled"]
    complete_delete = backend.delete_account_face
    monkeypatch.setattr(backend, "delete_account_face", lambda _: {
        "account_id": "bobby", "revoked": True, "key_destroyed": True, "index_removed": False})
    response = client.post("/api/auth/accounts/bobby/delete", json={"confirm_account_id": "bobby"}, headers=HEADERS)
    assert response.status_code == 503
    assert service.accounts.get("bobby")["enabled"] is False
    assert service.accounts.photo("bobby") == saved and service.accounts.account(secret) is None
    assert backend.account_enabled["bobby"] is False
    monkeypatch.setattr(backend, "delete_account_face", complete_delete)
    response = client.post("/api/auth/accounts/bobby/delete", json={"confirm_account_id": "bobby"}, headers=HEADERS)
    assert response.status_code == 200 and response.json()["deleted"] is True
    assert service.accounts.get("bobby") is None and backend.account_faces["bobby"] == "revoked"
    assert client.get("/api/auth/accounts/bobby/photo").status_code == 404
    recreated = create_target(client)
    assert recreated["enabled"] and not recreated["has_photo"]
    assert backend.account_enabled["bobby"] is True


def test_backend_management_outage_does_not_invent_ready_or_create_local_account(setup, monkeypatch):
    client, service, backend = manager(setup)

    def unavailable(*_):
        raise RuntimeError("synthetic backend unavailable")

    monkeypatch.setattr(backend, "account_summary", unavailable)
    listing = client.get("/api/auth/accounts")
    assert listing.status_code == 200
    assert listing.json()["management_ready"] is False
    assert listing.json()["accounts"][0]["face_status"] == "unknown"
    assert client.post("/api/auth/accounts", json=TARGET, headers=HEADERS).status_code == 503
    assert service.accounts.get("bobby") is None


@pytest.mark.parametrize("reply", [
    {"accounts": []},
    {"accounts": [{"account_id": "someone-else", "enabled": True, "face_status": "none"}]},
    {"accounts": [{"account_id": ["alice"], "enabled": True, "face_status": "none"}]},
    {"accounts": [{"account_id": "alice", "enabled": 1, "face_status": "none"}]},
])
def test_malformed_summary_fails_closed_for_listing_and_creation(setup, monkeypatch, reply):
    client, service, backend = manager(setup)
    monkeypatch.setattr(backend, "account_summary", lambda _: reply)
    response = client.get("/api/auth/accounts")
    assert response.status_code == 200 and response.json()["management_ready"] is False
    assert client.post("/api/auth/accounts", json=TARGET, headers=HEADERS).status_code == 503
    assert service.accounts.get("bobby") is None


@pytest.mark.parametrize("reply", [
    {"account_id": "alice", "enabled": False},
    {"account_id": "bobby", "enabled": True},
    {"account_id": "bobby", "enabled": 0},
])
def test_status_response_must_confirm_exact_target_and_boolean_before_local_change(setup, monkeypatch, reply):
    client, service, backend = manager(setup)
    create_target(client)
    secret = service.accounts.login("bobby", TARGET["password"])
    monkeypatch.setattr(backend, "account_status", lambda *_: reply)
    response = client.post("/api/auth/accounts/bobby/status", json={"enabled": False}, headers=HEADERS)
    assert response.status_code == 503
    assert service.accounts.get("bobby")["enabled"] and service.accounts.account(secret) == "bobby"


@pytest.mark.parametrize("reply", [
    {"account_id": "alice", "revoked": True, "key_destroyed": True, "index_removed": True},
    {"account_id": "bobby", "revoked": "true", "key_destroyed": True, "index_removed": True},
    {"account_id": "bobby", "revoked": True, "key_destroyed": True},
])
def test_malformed_deletion_confirmation_keeps_local_identity_and_photo_for_retry(setup, monkeypatch, reply):
    client, service, backend = manager(setup)
    create_target(client)
    service.accounts.store_photo("bobby", synthetic_photo(), **PHOTO_CONSENT)
    monkeypatch.setattr(backend, "delete_account_face", lambda _: reply)
    response = client.post("/api/auth/accounts/bobby/delete", json={"confirm_account_id": "bobby"}, headers=HEADERS)
    assert response.status_code == 503
    assert service.accounts.get("bobby")["enabled"] is False
    assert service.accounts.photo("bobby") is not None
    assert service.accounts.get("alice")["enabled"]


def test_orphaned_central_identity_cannot_be_claimed_by_new_password(setup):
    client, service, backend = manager(setup)
    backend.account_faces["bobby"] = "active"
    response = client.post("/api/auth/accounts", json=TARGET, headers=HEADERS)
    assert response.status_code == 409 and service.accounts.get("bobby") is None


def test_managed_enrollment_asserts_target_and_keeps_actor_identity_and_photo_default_off(setup):
    client, service, backend = manager(setup)
    create_target(client)
    original_secret = client.cookies.get(COOKIE)
    outcome = enroll_target(client)
    assert outcome["state"] == "succeeded", outcome
    assert outcome["target_account_id"] == "bobby" and outcome["target_display_name"] == "봄"
    claims = token.verify(backend.key.public_key(), backend.enrollment["assertion"])
    assert claims["acct"] == "bobby" and claims["nonce"] == "unpredictable-fixture"
    assert claims["method"] == "admin"
    assert backend.account_faces == {"bobby": "active"}
    assert client.cookies.get(COOKIE) == original_secret
    assert service.accounts.account(original_secret) == "alice"
    assert service.accounts.photo("bobby") is None
    assert service.operations[outcome["id"]]["actor"] == "alice"


def test_consented_success_saves_one_thumbnail_and_reenroll_without_consent_removes_it(setup):
    client, service, backend = manager(setup)
    create_target(client)
    outcome = enroll_target(client, photo_consent=True)
    assert outcome["state"] == "succeeded" and outcome["result"]["photo_saved"] is True
    meta = service.accounts.get("bobby")
    assert meta["has_photo"] and meta["photo_consent_version"] == "registration-photo-v1"
    assert meta["photo_consented_at"] == service.operations[outcome["id"]]["consent_at"]
    response = client.get("/api/auth/accounts/bobby/photo")
    assert response.status_code == 200 and response.headers["content-type"] == "image/jpeg"
    assert "no-store" in response.headers["cache-control"]
    assert cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR).shape == (48, 64, 3)
    assert service.accounts.photo("alice") is None
    outcome = enroll_target(client, re_enroll=True)
    assert outcome["state"] == "succeeded" and backend.enrollment["re_enroll"] is True
    assert service.accounts.photo("bobby") is None


def test_successful_reenroll_with_photo_save_failure_removes_stale_photo_and_stays_successful(setup, monkeypatch):
    client, service, backend = manager(setup)
    create_target(client)
    backend.account_faces["bobby"] = "active"
    service.accounts.store_photo("bobby", synthetic_photo(), **PHOTO_CONSENT)
    assert service.accounts.get("bobby")["has_photo"]

    def cannot_save(*args, **kwargs):
        raise RuntimeError("synthetic image storage failure")

    monkeypatch.setattr(service.accounts, "store_photo", cannot_save)
    outcome = enroll_target(client, photo_consent=True, re_enroll=True)
    assert outcome["state"] == "succeeded", outcome
    assert outcome["result"]["status"] == "active" and outcome["result"]["photo_saved"] is False
    assert backend.account_faces["bobby"] == "active" and backend.enrollment["re_enroll"] is True
    assert service.accounts.photo("bobby") is None
    assert not service.accounts.get("bobby")["has_photo"]


@pytest.mark.parametrize("status", ["rejected", "pending_review"])
def test_rejected_or_pending_enrollment_never_stores_photo(setup, status):
    client, service, backend = manager(setup)
    create_target(client)
    backend.enrollment_status = status
    outcome = enroll_target(client, photo_consent=True)
    assert outcome["state"] == "rejected"
    assert service.accounts.photo("bobby") is None


def test_photo_delete_does_not_revoke_enrolled_face(setup):
    client, service, backend = manager(setup)
    create_target(client)
    backend.account_faces["bobby"] = "active"
    service.accounts.store_photo("bobby", synthetic_photo(), **PHOTO_CONSENT)
    response = client.post("/api/auth/accounts/bobby/photo/delete", headers=HEADERS)
    assert response.status_code == 200 and response.json()["account"]["has_photo"] is False
    assert backend.account_faces["bobby"] == "active" and service.accounts.get("bobby")["enabled"]


def test_management_mutations_block_while_camera_is_capturing(setup):
    client, service, backend = manager(setup)
    create_target(client)
    backend.block = threading.Event()
    response = client.post("/api/auth/accounts/bobby/enroll", json={**CONSENT, "photo_consent": True}, headers=HEADERS)
    assert response.status_code == 202
    op = response.json()
    try:
        for path, body in (("status", {"enabled": False}), ("delete", {"confirm_account_id": "bobby"}),
                           ("photo/delete", {})):
            assert client.post(f"/api/auth/accounts/bobby/{path}", json=body, headers=HEADERS).status_code == 409
        assert client.post("/api/auth/cancel", json={"id": op["id"]}, headers=HEADERS).status_code == 200
    finally:
        backend.block.set()
    assert finish(client, op)["state"] == "cancelled"
    assert service.accounts.photo("bobby") is None and backend.submitted == 0


def test_target_disabled_during_capture_cancels_before_enrollment_or_photo_commit(setup):
    client, service, backend = manager(setup)
    create_target(client)
    backend.block = threading.Event()
    response = client.post("/api/auth/accounts/bobby/enroll", json={**CONSENT, "photo_consent": True}, headers=HEADERS)
    assert response.status_code == 202
    service.accounts.set_enabled("bobby", False)  # Simulate another trusted local deployment process.
    backend.block.set()
    assert finish(client, response.json())["state"] == "cancelled"
    assert backend.submitted == 0 and service.accounts.photo("bobby") is None


def test_cached_live_label_immediately_reflects_rename_disable_and_delete(setup):
    client, service, _ = manager(setup)
    create_target(client)
    worker = LiveRecognition(service.camera, service)
    packet = {"demo": True, "calib_version": "test-calib", "image_size": [64, 48], "faces": [
        {"bbox": [10, 5, 40, 40], "status": "known", "account_id": "bobby", "reason": "ok"}],
        "pad_enabled": False, "pad_mode": "disabled_demo", "truncated": False}
    frame = {"depth": np.zeros((48, 64), np.uint16), "calib_version": "test-calib", "captured_at_ms": 123}
    for sequence in (1, 2):
        worker._publish(packet, (sequence, time.monotonic(), frame), time.monotonic())
    assert worker.snapshot()["faces"][0]["display_name"] == "봄"
    service.accounts.rename("bobby", "새 이름")
    assert worker.snapshot()["faces"][0]["display_name"] == "새 이름"
    service.accounts.set_enabled("bobby", False)
    assert worker.snapshot()["faces"][0]["status"] == "unknown"
    service.accounts.delete("bobby")
    face = worker.snapshot()["faces"][0]
    assert face["display_name"] == "미등록" and face["reason"] == "account_unavailable"
