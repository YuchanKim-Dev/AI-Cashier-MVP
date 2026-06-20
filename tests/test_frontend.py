"""
FastAPI 프론트엔드 단위 테스트.

상태 업데이트 함수와 HTTP 엔드포인트를 검증한다.
"""

import pytest
from fastapi.testclient import TestClient

from src.frontend.app import app, update_state, append_ai_text, clear_ai_text, _state


@pytest.fixture(autouse=True)
def reset_state():
    """각 테스트 전에 상태 초기화."""
    _state.update({
        "mic": "disconnected",
        "conversation": "idle",
        "ai_text": "",
        "speaker_verified": None,
    })


client = TestClient(app)


def test_index_returns_html():
    resp = client.get("/")
    assert resp.status_code == 200
    assert "음성 AI 캐셔" in resp.text


def test_update_state_changes_state():
    update_state({"mic": "active"})
    assert _state["mic"] == "active"


def test_append_ai_text_accumulates():
    append_ai_text("안녕")
    append_ai_text("하세요")
    assert _state["ai_text"] == "안녕하세요"


def test_clear_ai_text():
    _state["ai_text"] = "이전 텍스트"
    clear_ai_text()
    assert _state["ai_text"] == ""
