"""
FastAPI 프론트엔드 단위 테스트.
"""

import pytest
from fastapi.testclient import TestClient

from src.frontend.app import app, push_state, _state


@pytest.fixture(autouse=True)
def reset_state():
    _state.update({
        "screen": "waiting",
        "conversation": "idle",
        "mic": "active",
        "ai_text": "",
        "user_name": None,
        "is_new_user": True,
        "speaker_verified": None,
        "failed_verifications": 0,
        "cart_items": [],
        "cart_total": 0,
        "payment_method": None,
        "transaction_id": None,
        "voice_duration": 0.0,
    })


client = TestClient(app)


def test_index_returns_html():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "음성 AI 캐셔" in resp.text


def test_all_screens_in_html():
    resp = client.get("/")
    for screen in ["waiting", "ordering", "checkout", "payment_processing",
                   "voice_save_prompt", "register", "complete", "locked"]:
        assert f"screen-{screen}" in resp.text


def test_push_state_updates():
    push_state({"screen": "ordering", "cart_total": 6500})
    assert _state["screen"] == "ordering"
    assert _state["cart_total"] == 6500


def test_action_checkout_endpoint():
    resp = client.post("/action/checkout")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


def test_action_payment_endpoint():
    resp = client.post("/action/payment", json={"method": "app_card"})
    assert resp.status_code == 200


def test_action_save_voice_endpoint():
    resp = client.post("/action/save_voice", json={"save": True})
    assert resp.status_code == 200


def test_action_register_endpoint():
    resp = client.post("/action/register", json={"name": "홍길동", "phone": "01012345678"})
    assert resp.status_code == 200


def test_action_retry_endpoint():
    resp = client.post("/action/retry_verification")
    assert resp.status_code == 200


def test_action_add_menu_endpoint():
    resp = client.post("/action/add_menu", json={"name": "치즈버거"})
    assert resp.status_code == 200
