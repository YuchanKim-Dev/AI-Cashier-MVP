"""
OpenAI Realtime API WebSocket 클라이언트 (2단계 — function calling 포함).

전체 흐름에서 이 모듈의 위치:
  마이크 PCM → [이 모듈] → Realtime API → 음성/텍스트/function call 이벤트

변경 (2단계):
  - websockets.legacy → websockets (최신 API, 경고 제거)
  - session.update에 tools 포함
  - function_call_arguments.done 이벤트 처리
  - send_function_result() 메서드 추가
"""

import asyncio
import base64
import json
import os
from typing import Callable, Optional, Awaitable

import websockets

from src.tools.handlers import TOOLS


REALTIME_URL = "wss://api.openai.com/v1/realtime"
DEFAULT_MODEL = "gpt-realtime-2"


class RealtimeClient:
    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        voice: str = "alloy",
        on_audio_delta: Optional[Callable[[str], None]] = None,
        on_text_delta: Optional[Callable[[str], None]] = None,
        on_response_done: Optional[Callable[[], None]] = None,
        on_function_call: Optional[Callable[[str, str, str], Awaitable[None]]] = None,
        on_session_ready: Optional[Callable[[], None]] = None,
        on_status_update: Optional[Callable[[str, float], None]] = None,
    ):
        self.api_key = api_key
        self.model = model
        self.voice = voice
        self.on_audio_delta = on_audio_delta
        self.on_text_delta = on_text_delta
        self.on_response_done = on_response_done
        # (call_id, name, arguments_json) → None
        self.on_function_call = on_function_call
        self.on_session_ready = on_session_ready
        # (status, timestamp) → None : timestamp는 발화 시간 추적용
        self.on_status_update = on_status_update
        self._ws = None
        self._connected = False

    async def connect(self):
        url = f"{REALTIME_URL}?model={self.model}"
        self._ws = await websockets.connect(
            url,
            additional_headers={
                "Authorization": f"Bearer {self.api_key}",
                "OpenAI-Beta": "realtime=v1",
            },
        )
        self._connected = True
        print(f"[RealtimeClient] 연결 성공: {url}")
        await self._send_session_update()

    async def _send_session_update(self):
        """
        세션 설정 — 캐셔 시스템 프롬프트 + tools 등록.
        tools를 여기서 정의해야 모델이 function calling을 사용할 수 있다.
        """
        system_prompt = (
            "당신은 친절하고 빠른 음성 AI 캐셔입니다. 한국어로 짧고 명확하게 응답하세요.\n"
            "- 손님이 메뉴를 말하면 바로 add_to_cart를 호출하세요.\n"
            "- 메뉴를 물어보면 recommend_menu를 호출하세요.\n"
            "- '결제', '주문할게요', '그게 다야' 등의 말이 나오면 checkout을 호출하세요.\n"
            "- 장바구니가 비어있으면 checkout을 호출하지 마세요.\n"
            "- 응답은 2문장 이내로 짧게. 불필요한 인사말 반복 금지.\n"
            "- 가격은 항상 '원' 단위로 말하세요."
        )
        event = {
            "type": "session.update",
            "session": {
                "modalities": ["audio", "text"],
                "instructions": system_prompt,
                "voice": self.voice,
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 600,
                },
                "input_audio_transcription": {"model": "whisper-1"},
                "tools": TOOLS,
                "tool_choice": "auto",
            },
        }
        await self._send(event)

    async def send_audio_chunk(self, pcm_bytes: bytes):
        """마이크 PCM → base64 → Realtime input_audio_buffer.append (1초 응답 경로 핵심)."""
        if not self._connected:
            return
        b64 = base64.b64encode(pcm_bytes).decode("utf-8")
        await self._send({"type": "input_audio_buffer.append", "audio": b64})

    async def send_function_result(self, call_id: str, output: str):
        """
        function call 결과를 Realtime API로 전송.
        conversation.item.create(function_call_output) 후 response.create로 다음 응답 유도.
        """
        await self._send({
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": output,
            },
        })
        await self._send({"type": "response.create"})

    async def _send(self, event: dict):
        if self._ws:
            await self._ws.send(json.dumps(event))

    async def listen(self):
        """WebSocket 이벤트 수신 루프."""
        async for message in self._ws:
            event = json.loads(message)
            await self._handle_event(event)

    async def _handle_event(self, event: dict):
        t = event.get("type", "")

        if t == "session.created":
            print("[RealtimeClient] 세션 생성됨")
            if self.on_session_ready:
                self.on_session_ready()

        elif t == "session.updated":
            print("[RealtimeClient] 세션 설정 완료 (tools 등록됨)")

        elif t == "response.audio.delta":
            delta = event.get("delta", "")
            if delta and self.on_audio_delta:
                self.on_audio_delta(delta)

        elif t == "response.audio_transcript.delta":
            delta = event.get("delta", "")
            if delta and self.on_text_delta:
                self.on_text_delta(delta)

        elif t == "response.function_call_arguments.done":
            # 모델이 function call을 완성 → 실행 후 결과 전송
            call_id = event.get("call_id", "")
            name = event.get("name", "")
            arguments = event.get("arguments", "{}")
            print(f"[RealtimeClient] function call: {name}({arguments})")
            if self.on_function_call:
                await self.on_function_call(call_id, name, arguments)

        elif t == "response.done":
            if self.on_response_done:
                self.on_response_done()
            if self.on_status_update:
                import time
                self.on_status_update("idle", time.time())

        elif t == "input_audio_buffer.speech_started":
            import time
            if self.on_status_update:
                self.on_status_update("listening", time.time())

        elif t == "input_audio_buffer.speech_stopped":
            import time
            if self.on_status_update:
                self.on_status_update("processing", time.time())

        elif t == "response.audio.done":
            if self.on_status_update:
                import time
                self.on_status_update("speaking_done", time.time())

        elif t == "error":
            print(f"[RealtimeClient] 오류: {event.get('error', {})}")

    async def close(self):
        self._connected = False
        if self._ws:
            await self._ws.close()
            print("[RealtimeClient] 연결 종료")
