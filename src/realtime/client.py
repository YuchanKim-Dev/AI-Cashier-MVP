"""
OpenAI Realtime API WebSocket 클라이언트 (GA — gpt-realtime-2).

GA API 주요 변경사항 (Beta 2026-05 폐기):
  - OpenAI-Beta 헤더 제거
  - session.type = "realtime" 필수
  - audio 포맷: string → object  { type: "audio/pcm", rate: N }
  - voice/turn_detection → session.audio.input|output 하위로 이동
  - 이벤트명 변경:
      response.audio.delta       → response.output_audio.delta
      response.audio_transcript.delta → response.output_text.delta
      response.audio.done        → response.output_audio.done
"""

import base64
import json
import ssl
import time
from typing import Callable, Optional, Awaitable

import certifi
import websockets

from src.tools.handlers import TOOLS


REALTIME_URL = "wss://api.openai.com/v1/realtime"
DEFAULT_MODEL = "gpt-realtime-2"

# GA PCM 포맷 오브젝트 (string "pcm16" 아님)
_FMT_INPUT  = {"type": "audio/pcm", "rate": 24000}   # GA 최소 24kHz (마이크도 24kHz 캡처)
_FMT_OUTPUT = {"type": "audio/pcm", "rate": 24000}   # 스피커: 24kHz


class RealtimeClient:
    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        voice: str = "alloy",
        on_audio_delta: Optional[Callable[[str], None]] = None,
        on_text_delta: Optional[Callable[[str], None]] = None,
        on_user_text: Optional[Callable[[str], None]] = None,
        on_user_text_delta: Optional[Callable[[str], None]] = None,
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
        self.on_user_text = on_user_text
        self.on_user_text_delta = on_user_text_delta
        self.on_response_done = on_response_done
        self.on_function_call = on_function_call   # async (call_id, name, args_json)
        self.on_session_ready = on_session_ready
        self.on_status_update = on_status_update   # (status, timestamp)
        self._ws = None
        self._connected = False
        self._queued_fn_outputs: list[tuple[str, str]] = []   # (call_id, output_json)
        self._response_active = False   # 현재 응답 생성 중 여부

    async def connect(self):
        url = f"{REALTIME_URL}?model={self.model}"
        ssl_ctx = ssl.create_default_context(cafile=certifi.where())
        self._ws = await websockets.connect(
            url,
            additional_headers={"Authorization": f"Bearer {self.api_key}"},
            ssl=ssl_ctx,
        )
        self._connected = True
        print(f"[RealtimeClient] 연결 성공: {url}")
        await self._send_session_update()

    async def _send_session_update(self):
        system_prompt = (
            "You are a voice AI cashier. Your ONLY job: help customers order food.\n"
            "LANGUAGE: detect the customer's first utterance language and NEVER switch. Korean→Korean only. English→English only.\n"
            "RULES:\n"
            "- Call add_to_cart immediately when a menu item is mentioned. No confirmation.\n"
            "- Call remove_from_cart immediately when removal is requested.\n"
            "- Call recommend_menu only when customer asks what to order.\n"
            "- Call checkout when customer is done ordering. Never if cart is empty.\n"
            "- On checkout screen: call select_payment('app_card') or select_payment('physical_card') based on what customer says.\n"
            "- If customer asks anything unrelated to food ordering, say only: '주문을 도와드릴게요.' (Korean) or 'I can only help with your order.' (English)\n"
            "- MAX 1 short sentence per response. No filler. No repeated greetings."
        )
        event = {
            "type": "session.update",
            "session": {
                "type": "realtime",
                "instructions": system_prompt,
                "output_modalities": ["audio"],
                "audio": {
                    "input": {
                        "format": _FMT_INPUT,
                        "turn_detection": {
                            "type": "server_vad",
                            "threshold": 0.5,
                            "prefix_padding_ms": 200,
                            "silence_duration_ms": 500,
                        },
                        "transcription": {"model": "whisper-1"},
                    },
                    "output": {
                        "format": _FMT_OUTPUT,
                        "voice": self.voice,
                    },
                },
                "tools": TOOLS,
                "tool_choice": "auto",
            },
        }
        await self._send(event)

    async def send_audio_chunk(self, pcm_bytes: bytes):
        """마이크 PCM → base64 → input_audio_buffer.append."""
        if not self._connected:
            return
        b64 = base64.b64encode(pcm_bytes).decode("utf-8")
        await self._send({"type": "input_audio_buffer.append", "audio": b64})

    async def send_function_result(self, call_id: str, output: str):
        """function call 결과를 큐에 저장. response.create는 response.done 후 한번만 보낸다."""
        await self._send({
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": output,
            },
        })
        self._queued_fn_outputs.append(call_id)

    async def _send(self, event: dict):
        if self._ws:
            await self._ws.send(json.dumps(event))

    async def listen(self):
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
            print("[RealtimeClient] 세션 설정 완료")

        # ── 오디오 델타 (GA: response.output_audio.delta, Beta 호환: response.audio.delta)
        elif t in ("response.output_audio.delta", "response.audio.delta"):
            delta = event.get("delta", "")
            if delta and self.on_audio_delta:
                self.on_audio_delta(delta)

        # ── 텍스트 델타 (GA: response.output_text.delta, Beta: response.audio_transcript.delta)
        elif t in ("response.output_text.delta", "response.audio_transcript.delta"):
            delta = event.get("delta", "")
            if delta and self.on_text_delta:
                self.on_text_delta(delta)

        # ── function call
        elif t == "response.function_call_arguments.done":
            call_id  = event.get("call_id", "")
            name     = event.get("name", "")
            arguments = event.get("arguments", "{}")
            print(f"[RealtimeClient] function call: {name}({arguments})")
            if self.on_function_call:
                await self.on_function_call(call_id, name, arguments)

        # ── 응답 시작
        elif t == "response.created":
            self._response_active = True

        # ── 응답 완료
        elif t == "response.done":
            self._response_active = False
            if self._queued_fn_outputs:
                # function call 결과들이 쌓여 있으면 → 한번만 response.create
                self._queued_fn_outputs.clear()
                await self._send({"type": "response.create"})
            else:
                if self.on_response_done:
                    self.on_response_done()
                if self.on_status_update:
                    self.on_status_update("idle", time.time())

        # ── 발화 감지 (VAD)
        elif t == "input_audio_buffer.speech_started":
            if self.on_status_update:
                self.on_status_update("listening", time.time())

        elif t == "input_audio_buffer.speech_stopped":
            if self.on_status_update:
                self.on_status_update("processing", time.time())

        # ── 사용자 발화 전사 델타 (실시간)
        elif t == "conversation.item.input_audio_transcription.delta":
            delta = event.get("delta", "")
            if delta and self.on_user_text_delta:
                self.on_user_text_delta(delta)

        # ── 사용자 발화 전사 완료
        elif t == "conversation.item.input_audio_transcription.completed":
            transcript = event.get("transcript", "").strip()
            if transcript and self.on_user_text:
                self.on_user_text(transcript)

        # ── 오디오 출력 완료 (GA/Beta 호환)
        elif t in ("response.output_audio.done", "response.audio.done"):
            if self.on_status_update:
                self.on_status_update("speaking_done", time.time())

        elif t == "error":
            err = event.get("error", {})
            code = err.get("code", "")
            print(f"[RealtimeClient] 오류: {err}")
            if code == "unknown_parameter":
                print(f"[RealtimeClient] 미지원 파라미터 무시: {err.get('param')}")

    async def update_instructions(self, name: str):
        """화자 인식 후 AI에게 사용자 이름 알림."""
        system_prompt = (
            "You are a voice AI cashier. Your ONLY job: help customers order food.\n"
            "LANGUAGE: detect the customer's first utterance language and NEVER switch. Korean→Korean only. English→English only.\n"
            "RULES:\n"
            "- Call add_to_cart immediately when a menu item is mentioned. No confirmation.\n"
            "- Call remove_from_cart immediately when removal is requested.\n"
            "- Call recommend_menu only when customer asks what to order.\n"
            "- Call checkout when customer is done ordering. Never if cart is empty.\n"
            "- On checkout screen: call select_payment('app_card') or select_payment('physical_card') based on what customer says.\n"
            "- If customer asks anything unrelated to food ordering, say only: '주문을 도와드릴게요.' (Korean) or 'I can only help with your order.' (English)\n"
            "- MAX 1 short sentence per response. No filler. No repeated greetings.\n"
            f"- This customer is '{name}', recognized by voice. Greet by name once, then take the order."
        )
        await self._send({
            "type": "session.update",
            "session": {"type": "realtime", "instructions": system_prompt},
        })

    async def greet_returning_user(self, name: str):
        """화자 인식 완료 — 주문 흐름 방해 없이 지시사항만 조용히 업데이트."""
        await self.update_instructions(name)

    async def close(self):
        self._connected = False
        if self._ws:
            await self._ws.close()
            print("[RealtimeClient] 연결 종료")
