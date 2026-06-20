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
        on_function_call=AsyncMock(),   # on_function_call은 async
        on_session_ready=MagicMock(),
        on_status_update=MagicMock(),   # (status, timestamp) 서명
    )
    c._ws = AsyncMock()
    c._connected = True
    return c


class TestEventHandling:
    @pytest.mark.asyncio
    async def test_session_created_calls_on_ready(self, client):
        await client._handle_event({"type": "session.created"})
        client.on_session_ready.assert_called_once()

    @pytest.mark.asyncio
    async def test_audio_delta_calls_on_audio_ga(self, client):
        """GA 이벤트명 response.output_audio.delta"""
        await client._handle_event({"type": "response.output_audio.delta", "delta": "base64data"})
        client.on_audio_delta.assert_called_once_with("base64data")

    @pytest.mark.asyncio
    async def test_audio_delta_calls_on_audio_beta_compat(self, client):
        """Beta 호환 이벤트명 response.audio.delta 도 동작해야 함"""
        await client._handle_event({"type": "response.audio.delta", "delta": "base64data"})
        client.on_audio_delta.assert_called_once_with("base64data")

    @pytest.mark.asyncio
    async def test_empty_audio_delta_not_called(self, client):
        await client._handle_event({"type": "response.output_audio.delta", "delta": ""})
        client.on_audio_delta.assert_not_called()

    @pytest.mark.asyncio
    async def test_text_delta_calls_on_text_ga(self, client):
        """GA 이벤트명 response.output_text.delta"""
        await client._handle_event({"type": "response.output_text.delta", "delta": "안녕"})
        client.on_text_delta.assert_called_once_with("안녕")

    @pytest.mark.asyncio
    async def test_text_delta_calls_on_text_beta_compat(self, client):
        """Beta 호환 이벤트명도 동작해야 함"""
        await client._handle_event({"type": "response.audio_transcript.delta", "delta": "안녕"})
        client.on_text_delta.assert_called_once_with("안녕")

    @pytest.mark.asyncio
    async def test_speech_started_status(self, client):
        await client._handle_event({"type": "input_audio_buffer.speech_started"})
        # on_status_update는 (status, timestamp) 두 인수로 호출된다
        args = client.on_status_update.call_args[0]
        assert args[0] == "listening"

    @pytest.mark.asyncio
    async def test_speech_stopped_status(self, client):
        await client._handle_event({"type": "input_audio_buffer.speech_stopped"})
        args = client.on_status_update.call_args[0]
        assert args[0] == "processing"

    @pytest.mark.asyncio
    async def test_response_done_status(self, client):
        await client._handle_event({"type": "response.done"})
        args = client.on_status_update.call_args[0]
        assert args[0] == "idle"

    @pytest.mark.asyncio
    async def test_function_call_done(self, client):
        event = {
            "type": "response.function_call_arguments.done",
            "call_id": "call_1",
            "name": "add_to_cart",
            "arguments": '{"item_name":"치즈버거"}',
        }
        await client._handle_event(event)
        client.on_function_call.assert_called_once_with("call_1", "add_to_cart", '{"item_name":"치즈버거"}')


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
