"""
OpenAI Realtime API WebSocket 클라이언트.

전체 흐름에서 이 모듈의 위치:
  마이크 PCM → [이 모듈] → Realtime API → 음성 응답(base64 PCM)

Realtime API GA 엔드포인트:
  wss://api.openai.com/v1/realtime?model=gpt-realtime-2
  헤더: Authorization: Bearer <key>, OpenAI-Beta: realtime=v1

연결 후 session.update로 캐셔 시스템 프롬프트, 음성, 언어 설정.
마이크 PCM은 input_audio_buffer.append (base64 인코딩)로 흘려보낸다.
모델이 말하는 동안 response.audio.delta 이벤트로 오디오 청크가 온다.
"""

import asyncio
import base64
import json
import os
from typing import Callable, Optional

import websockets
from websockets.legacy.client import WebSocketClientProtocol


REALTIME_URL = "wss://api.openai.com/v1/realtime"
DEFAULT_MODEL = "gpt-realtime-2"


class RealtimeClient:
    """
    Realtime API WebSocket 연결을 관리하고 이벤트를 라우팅하는 클래스.

    1단계에서는 캐셔 로직 없이 잡담 모드로 동작.
    2단계에서 function calling이 추가된다.
    """

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        voice: str = "alloy",
        on_audio_delta: Optional[Callable[[str], None]] = None,
        on_text_delta: Optional[Callable[[str], None]] = None,
        on_function_call: Optional[Callable[[dict], None]] = None,
        on_session_ready: Optional[Callable[[], None]] = None,
        on_status_update: Optional[Callable[[str], None]] = None,
    ):
        self.api_key = api_key
        self.model = model
        self.voice = voice
        # 이벤트 콜백 — orchestrator가 연결해 각 이벤트에 반응
        self.on_audio_delta = on_audio_delta
        self.on_text_delta = on_text_delta
        self.on_function_call = on_function_call
        self.on_session_ready = on_session_ready
        self.on_status_update = on_status_update
        self._ws: Optional[WebSocketClientProtocol] = None
        self._connected = False

    async def connect(self):
        """WebSocket 연결 + session.update로 초기 설정."""
        url = f"{REALTIME_URL}?model={self.model}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            # GA 전환 시점까지 Beta 헤더 필요 (서버가 있으면 수락, 없어도 무시)
            "OpenAI-Beta": "realtime=v1",
        }
        self._ws = await websockets.connect(url, extra_headers=headers)
        self._connected = True
        print(f"[RealtimeClient] 연결 성공: {url}")

        # session.update: 캐셔 시스템 프롬프트, 음성, VAD 설정
        await self._send_session_update()

    async def _send_session_update(self):
        """
        세션 초기 설정.
        - input_audio_format: pcm16 (마이크에서 받는 포맷과 일치)
        - output_audio_format: pcm16 (24kHz, 재생 모듈과 일치)
        - turn_detection: server_vad — 서버가 발화 끝을 감지해 자동으로 응답 시작
        """
        system_prompt = (
            "당신은 친절한 음성 AI 캐셔입니다. "
            "손님이 말하면 간결하고 자연스럽게 응답하세요. "
            "현재는 메뉴 안내 없이 잡담 모드입니다. "
            "한국어로 대화하세요."
        )
        event = {
            "type": "session.update",
            "session": {
                "modalities": ["audio", "text"],
                "instructions": system_prompt,
                "voice": self.voice,
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                # server_vad: 서버가 발화 끝을 자동 감지 → 사용자가 말을 멈추면 응답 시작
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 500,
                },
                "input_audio_transcription": {"model": "whisper-1"},
            },
        }
        await self._send(event)

    async def send_audio_chunk(self, pcm_bytes: bytes):
        """
        마이크 PCM 청크를 Realtime API로 전송.
        Realtime은 input_audio_buffer.append로 base64 PCM을 받는다.
        여기서 마이크 청크를 인코딩해 흘려보낸다 (1초 응답 경로의 핵심).
        """
        if not self._connected:
            return
        b64 = base64.b64encode(pcm_bytes).decode("utf-8")
        event = {"type": "input_audio_buffer.append", "audio": b64}
        await self._send(event)

    async def _send(self, event: dict):
        if self._ws:
            await self._ws.send(json.dumps(event))

    async def listen(self):
        """
        WebSocket 이벤트 수신 루프.
        이벤트 타입별로 등록된 콜백을 호출한다.
        연결이 끊어지면 루프 종료.
        """
        async for message in self._ws:
            event = json.loads(message)
            await self._handle_event(event)

    async def _handle_event(self, event: dict):
        event_type = event.get("type", "")

        if event_type == "session.created":
            print("[RealtimeClient] 세션 생성됨")
            if self.on_session_ready:
                self.on_session_ready()

        elif event_type == "session.updated":
            print("[RealtimeClient] 세션 설정 완료")

        elif event_type == "response.audio.delta":
            # 모델이 생성한 오디오 청크 — 즉시 재생 큐로 넘긴다
            delta = event.get("delta", "")
            if delta and self.on_audio_delta:
                self.on_audio_delta(delta)

        elif event_type == "response.audio_transcript.delta":
            # 모델 발화 텍스트 (화면 표시용)
            delta = event.get("delta", "")
            if delta and self.on_text_delta:
                self.on_text_delta(delta)

        elif event_type == "response.function_call_arguments.done":
            # function calling 완료 이벤트 (2단계에서 캐셔 로직과 연결)
            if self.on_function_call:
                self.on_function_call(event)

        elif event_type == "input_audio_buffer.speech_started":
            if self.on_status_update:
                self.on_status_update("listening")

        elif event_type == "input_audio_buffer.speech_stopped":
            if self.on_status_update:
                self.on_status_update("processing")

        elif event_type == "response.done":
            if self.on_status_update:
                self.on_status_update("idle")

        elif event_type == "error":
            print(f"[RealtimeClient] 오류: {event.get('error', {})}")

    async def close(self):
        self._connected = False
        if self._ws:
            await self._ws.close()
            print("[RealtimeClient] 연결 종료")
