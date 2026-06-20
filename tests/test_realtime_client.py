"""
RealtimeClient 단위 테스트.

WebSocket을 모킹해 실제 OpenAI 연결 없이 이벤트 핸들링을 검증한다.
핵심: 각 이벤트 타입이 올바른 콜백을 호출하는가.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.realtime.client import RealtimeClient


@pytest.fixture
def client():
    """콜백 스파이가 연결된 클라이언트."""
    c = RealtimeClient(
        api_key="test-key",
        on_audio_delta=MagicMock(),
        on_text_delta=MagicMock(),
        on_function_call=MagicMock(),
        on_session_ready=MagicMock(),
        on_status_update=MagicMock(),
    )
    # WebSocket 연결은 모킹
    c._ws = AsyncMock()
    c._connected = True
    return c


class TestEventHandling:
    @pytest.mark.asyncio
    async def test_session_created_calls_on_ready(self, client):
        await client._handle_event({"type": "session.created"})
        client.on_session_ready.assert_called_once()

    @pytest.mark.asyncio
    async def test_audio_delta_calls_on_audio(self, client):
        await client._handle_event({"type": "response.audio.delta", "delta": "base64data"})
        client.on_audio_delta.assert_called_once_with("base64data")

    @pytest.mark.asyncio
    async def test_empty_audio_delta_not_called(self, client):
        """델타가 빈 문자열이면 콜백을 호출하지 않는다."""
        await client._handle_event({"type": "response.audio.delta", "delta": ""})
        client.on_audio_delta.assert_not_called()

    @pytest.mark.asyncio
    async def test_text_delta_calls_on_text(self, client):
        await client._handle_event({"type": "response.audio_transcript.delta", "delta": "안녕"})
        client.on_text_delta.assert_called_once_with("안녕")

    @pytest.mark.asyncio
    async def test_speech_started_status(self, client):
        await client._handle_event({"type": "input_audio_buffer.speech_started"})
        client.on_status_update.assert_called_with("listening")

    @pytest.mark.asyncio
    async def test_speech_stopped_status(self, client):
        await client._handle_event({"type": "input_audio_buffer.speech_stopped"})
        client.on_status_update.assert_called_with("processing")

    @pytest.mark.asyncio
    async def test_response_done_status(self, client):
        await client._handle_event({"type": "response.done"})
        client.on_status_update.assert_called_with("idle")

    @pytest.mark.asyncio
    async def test_function_call_done(self, client):
        event = {"type": "response.function_call_arguments.done", "name": "add_to_cart"}
        await client._handle_event(event)
        client.on_function_call.assert_called_once_with(event)


class TestSendAudioChunk:
    @pytest.mark.asyncio
    async def test_sends_base64_encoded(self, client):
        """PCM bytes가 base64로 인코딩돼 WebSocket으로 전송되는지 확인."""
        import base64
        pcm = b"\x01\x02\x03\x04"
        await client.send_audio_chunk(pcm)

        call_args = client._ws.send.call_args[0][0]
        payload = json.loads(call_args)
        assert payload["type"] == "input_audio_buffer.append"
        assert payload["audio"] == base64.b64encode(pcm).decode()

    @pytest.mark.asyncio
    async def test_no_send_when_disconnected(self, client):
        client._connected = False
        await client.send_audio_chunk(b"\x00\x01")
        client._ws.send.assert_not_called()
