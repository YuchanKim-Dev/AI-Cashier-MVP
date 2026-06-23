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

import asyncio
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
        on_ai_transcript_done: Optional[Callable[[str], None]] = None,
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
        self.on_ai_transcript_done = on_ai_transcript_done  # AI 발화 완료 (전체 텍스트)
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
        self._transcript_sent = False   # 이번 응답에서 transcript 콜백 이미 호출됨
        self._lang = "ko"              # 현재 언어 ("ko" | "en")

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
        system_prompt = self._make_prompt("ko")
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
                            "threshold": 0.8,
                            "prefix_padding_ms": 300,
                            "silence_duration_ms": 700,
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

        # ── 텍스트 델타 (GA 실제 이벤트명 확인됨)
        elif t in ("response.output_audio_transcript.delta",
                   "response.output_text.delta",
                   "response.audio_transcript.delta"):
            delta = event.get("delta", "")
            if delta and self.on_text_delta:
                self.on_text_delta(delta)

        # ── AI 발화 전사 완료 (GA 실제 이벤트명 확인됨)
        elif t in ("response.output_audio_transcript.done",
                   "response.audio_transcript.done",
                   "response.output_text.done"):
            transcript = (event.get("transcript") or event.get("text") or "").strip()
            print(f"[RealtimeClient] AI 전사 완료: {transcript[:60]!r}")
            if transcript and self.on_ai_transcript_done and not self._transcript_sent:
                self._transcript_sent = True
                self.on_ai_transcript_done(transcript)

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
            self._transcript_sent = False

        # ── 응답 완료
        elif t == "response.done":
            self._response_active = False
            # response.done 안에 output items의 transcript가 포함됨 → fallback (중복 방지)
            if not self._transcript_sent and self.on_ai_transcript_done:
                response_obj = event.get("response", {})
                for item in response_obj.get("output", []):
                    for part in item.get("content", []):
                        transcript = (part.get("transcript") or part.get("text") or "").strip()
                        if transcript:
                            print(f"[RealtimeClient] response.done fallback transcript: {transcript[:60]!r}")
                            self._transcript_sent = True
                            self.on_ai_transcript_done(transcript)
                            break
                    if self._transcript_sent:
                        break
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
            print(f"[RealtimeClient] 사용자 전사 완료: {transcript[:60]!r}")
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

        elif not t.startswith("rate_limit") and t not in (
            "session.created", "session.updated",
            "response.created", "response.done",
            "response.output_item.added", "response.output_item.done",
            "response.content_part.added", "response.content_part.done",
            "response.function_call_arguments.delta",
            "input_audio_buffer.committed", "input_audio_buffer.cleared",
            "conversation.item.created", "conversation.item.truncated",
            "conversation.item.added", "conversation.item.done",
        ):
            print(f"[RealtimeClient] 미처리 이벤트: {t}")

    def _make_prompt(self, lang: str = "ko", user_name: str = None, is_new_user: bool = True) -> str:
        """언어 및 사용자 이름에 맞는 시스템 프롬프트 생성."""
        if lang == "en":
            prompt = (
                "You are 'Kay', a cheerful and professional AI cashier at a burger restaurant. "
                "Talk exactly like a real service-industry employee — warm, upbeat, slightly formal but natural. "
                "Think of a friendly Starbucks barista tone. No robotic phrases. Respond ONLY in English.\n"
                "Rules:\n"
                "- Call add_to_cart immediately when customer mentions any menu item (no confirmation)\n"
                "- Call remove_from_cart when customer wants to remove something\n"
                "- Call recommend_menu only when asked for suggestions\n"
                "- Call checkout when order is complete (never if cart is empty)\n"
                "- At payment screen: call select_payment when customer mentions 'app card' or 'card'\n"
                "- Keep answers SHORT and NATURAL. Max 1–2 sentences.\n"
                "- Example tone: 'Of course! One cheeseburger, coming right up! Anything else for you today?'\n"
                "- Do NOT repeat greetings."
            )
            if is_new_user and not user_name:
                prompt += "\n- Casually mention once at checkout that they can register name+phone for personalized service next time."
        else:
            prompt = (
                "너는 햄버거 가게 카운터 직원 '케이'야. 진짜 서비스직 직원처럼 밝고 친절하게 말해.\n"
                "말투: 카페 직원처럼 따뜻하고 활기차게. 존댓말 사용. 딱딱하거나 로봇 같은 말투 절대 금지. 한국어만.\n"
                "예시 말투:\n"
                "  - '네, 치즈버거 바로 담아드릴게요~!'\n"
                "  - '맛있는 선택이세요! 감자튀김도 추가해 드릴까요?'\n"
                "  - '주문 완료됐어요! 맛있게 드세요 :)'\n"
                "  - '앗, 바로 빼드릴게요~'\n"
                "규칙:\n"
                "- 메뉴 이름 나오면 바로 add_to_cart 호출. 확인 질문 없이.\n"
                "- 취소하면 바로 remove_from_cart 호출.\n"
                "- 추천은 고객이 물어볼 때만 recommend_menu 호출.\n"
                "- 주문 끝나면 checkout 호출. 장바구니 비면 절대 금지.\n"
                "- 결제 화면: 앱카드/현장카드 말하면 select_payment 호출.\n"
                "- 답은 1~2문장. 간결하고 명랑하게.\n"
                "- 인사말 반복 금지."
            )
            if is_new_user and not user_name:
                prompt += (
                    "\n- 결제 전에 한 번만: '이름이랑 전화번호 등록하시면 다음 방문 때 목소리로 바로 주문하실 수 있어요! 앱에서 등록 가능하세요~' 라고 알려줘."
                )
        if user_name:
            if lang == "en":
                prompt += f"\n- This customer is {user_name}. Greet them warmly by name once."
            else:
                prompt += f"\n- 이 고객은 '{user_name}'님이야. 처음 한 번만 이름 불러서 반갑게 맞이해."
        return prompt

    async def update_instructions(self, name: str):
        """화자 인식 후 AI에게 사용자 이름 알림 (등록된 사용자 — 등록 안내 생략)."""
        await self._send({
            "type": "session.update",
            "session": {"type": "realtime", "instructions": self._make_prompt(self._lang, name, is_new_user=False)},
        })

    async def set_language(self, lang: str):
        """사용자 첫 발화 언어에 따라 AI 응답 언어 전환."""
        self._lang = lang
        await self._send({
            "type": "session.update",
            "session": {"type": "realtime", "instructions": self._make_prompt(lang)},
        })
        print(f"[RealtimeClient] 언어 전환: {lang}")

    async def send_initial_greeting(self):
        """세션 시작 즉시 AI가 인사말을 발화하도록 트리거."""
        if self._lang == "en":
            inst = (
                "Greet the customer warmly and naturally in English. "
                "One short sentence. Example: 'Hey there! Welcome, what can I get for you today?'"
            )
        else:
            inst = (
                "고객에게 진짜 직원처럼 자연스럽고 활기차게 한국어로 짧게 인사해. "
                "한 문장만. 예: '어서오세요! 뭐 드릴까요?'"
            )
        await self._send({
            "type": "response.create",
            "response": {"instructions": inst}
        })

    async def greet_returning_user(self, name: str):
        """화자 인식 완료 — 지시사항 업데이트 후 이름으로 인사.
        AI가 이미 응답 중이면 인사는 생략하고 지시만 업데이트한다."""
        await self.update_instructions(name)
        if self._response_active:
            # 이미 응답 중 — 인사를 끊으면 어색하므로 다음 응답부터 이름 포함
            return
        if self._lang == "en":
            inst = f"Welcome back, {name}! Greet them warmly by name in one short sentence."
        else:
            inst = f"'{name}'님 반갑다고 짧게 이름 불러서 인사해. 한 문장만."
        await self._send({
            "type": "response.create",
            "response": {"instructions": inst}
        })

    async def cancel_response(self):
        """진행 중인 응답 취소 + 오디오 버퍼 초기화."""
        if self._response_active:
            await self._send({"type": "response.cancel"})
        await self._send({"type": "input_audio_buffer.clear"})

    async def send_alert(self, instructions: str, max_wait: float = 2.0):
        """응답이 끝날 때까지 기다린 후 경고 메시지 전송.
        _response_active 플래그와 서버 상태 사이의 race condition을 방지하기 위해
        False가 된 후에도 추가 대기 후 cancel을 한 번 더 호출한다."""
        await self.cancel_response()
        waited = 0.0
        while waited < max_wait:
            await asyncio.sleep(0.1)
            waited += 0.1
            if self._response_active:
                await self.cancel_response()  # 응답 중이면 재취소
            else:
                break  # False가 됐어도 아래서 추가 대기
        # race condition 방지: False여도 0.4초 더 기다리고 cancel 재시도
        await asyncio.sleep(0.4)
        await self.cancel_response()
        await asyncio.sleep(0.1)
        await self._send({
            "type": "response.create",
            "response": {"instructions": instructions}
        })

    async def close(self):
        self._connected = False
        if self._ws:
            await self._ws.close()
            print("[RealtimeClient] 연결 종료")
