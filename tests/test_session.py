"""
SessionState 단위 테스트.
"""

import time
import pytest
from src.orchestrator.session import SessionState


def test_initial_state():
    s = SessionState()
    assert s.screen == "waiting"
    assert s.is_new_user is True
    assert s.voice_duration == 0.0


def test_voice_duration_tracking():
    s = SessionState()
    t0 = time.time()
    s.on_speech_start(t0)
    s.on_speech_end(t0 + 4.0)
    assert s.voice_duration == pytest.approx(4.0, abs=0.01)


def test_voice_duration_accumulates():
    s = SessionState()
    t0 = time.time()
    s.on_speech_start(t0)
    s.on_speech_end(t0 + 1.5)
    s.on_speech_start(t0 + 3.0)
    s.on_speech_end(t0 + 5.0)
    assert s.voice_duration == pytest.approx(3.5, abs=0.01)


def test_enough_voice_threshold():
    s = SessionState()
    t0 = time.time()
    s.on_speech_start(t0)
    s.on_speech_end(t0 + 2.9)
    assert s.enough_voice is False

    s.on_speech_start(t0 + 4.0)
    s.on_speech_end(t0 + 4.2)
    assert s.enough_voice is True


def test_to_dict_serializable():
    s = SessionState()
    d = s.to_dict()
    import json
    json.dumps(d)  # 직렬화 가능한지 확인


def test_reset():
    s = SessionState()
    s.screen = "complete"
    s.user_name = "홍길동"
    s.voice_duration = 5.0
    s.reset()
    assert s.screen == "waiting"
    assert s.user_name is None
    assert s.voice_duration == 0.0
