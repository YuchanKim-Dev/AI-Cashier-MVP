"""
오케스트레이터 — 2단계 진입점.

전체 파이프라인:
  1. FastAPI 키오스크 서버 (별도 스레드)
  2. Realtime API 연결 + session 설정 (tools 포함)
  3. 마이크 캡처 → Realtime 스트리밍
  4. 이벤트 처리:
     - 오디오 델타 → 스피커 재생
     - 텍스트 델타 → 화면 업데이트
     - function call → 장바구니/결제 처리
     - 발화 시간 → voice_duration 누적
  5. 버튼 액션 처리:
     - checkout → 결제 화면
     - payment → 결제 처리 (mock)
     - save_voice / register → 등록 흐름
     - retry_verification → 잠금 해제 시도
"""

import asyncio
import os
import ssl
import time
import threading

import certifi
import uvicorn
from dotenv import load_dotenv

# 맥 Python에서 SSL 인증서를 못 찾는 문제 해결
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

from src.audio.capture import AsyncMicrophoneCapture
from src.audio.playback import AudioPlayback
from src.frontend.app import app as fastapi_app, push_state, set_context
from src.orchestrator.session import SessionState
from src.realtime.client import RealtimeClient
from src.tools.cart import CartManager
from src.tools.handlers import FunctionCallHandler
from src.tools.payment import payment_gateway
from src.tools.user_store import save_user, get_first_user

load_dotenv()

# ─── 전역 세션 상태 ────────────────────────────────────────────────────────────
session = SessionState()
cart = CartManager()
action_queue: asyncio.Queue = asyncio.Queue()


def _sync_and_push(updates: dict = None):
    """session 상태를 frontend로 SSE 전송. asyncio 루프에서 호출."""
    if updates:
        for k, v in updates.items():
            setattr(session, k, v)
    push_state(session.to_dict())


# ─── FastAPI 서버 (별도 스레드) ────────────────────────────────────────────────
def start_frontend_server():
    config = uvicorn.Config(fastapi_app, host="0.0.0.0", port=8000, log_level="warning")
    uvicorn.Server(config).run()


# ─── 결제 처리 ─────────────────────────────────────────────────────────────────
async def _do_payment():
    """실제 mock 결제 처리 → 결과 화면 전환."""
    _sync_and_push({"screen": "payment_processing"})
    await asyncio.sleep(0.1)
    result = await payment_gateway.process(amount=cart.total, method=session.payment_method or "physical_card")
    if result["success"]:
        session.transaction_id = result["transaction_id"]
        if session.is_new_user and session.enough_voice:
            _sync_and_push({"screen": "voice_save_prompt"})
        else:
            _sync_and_push({"screen": "complete"})
    else:
        _sync_and_push({"screen": "checkout"})
        print(f"[Orchestrator] 결제 실패: {result.get('error')}")


async def process_payment(method: str):
    """결제 처리. physical_card면 카드 삽입 대기 화면 먼저."""
    _sync_and_push({"payment_method": method})
    if method == "physical_card":
        _sync_and_push({"screen": "card_insert"})
        await asyncio.sleep(3)   # 카드 꽂는 mock 대기
    elif method == "app_card":
        _sync_and_push({"screen": "app_payment"})
        return   # 실제 결제는 사용자가 결제수단 선택 후 별도 액션으로 처리
    await _do_payment()


# ─── 메인 비동기 파이프라인 ────────────────────────────────────────────────────
async def run():
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        print("[Orchestrator] OPENAI_API_KEY가 없습니다. .env 파일을 확인하세요.")
        return

    _pending_ai: str = ""    # user transcript 오기 전에 완성된 AI 텍스트
    _pending_user: str = ""  # AI 응답 오기 전에 도착한 user transcript

    voice = os.getenv("OPENAI_REALTIME_VOICE", "alloy")
    model = os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime-2")
    loop  = asyncio.get_event_loop()

    # frontend 액션 큐 + 이벤트 루프 주입
    set_context(action_queue, loop)

    # 재생 모듈 — 텍스트 출력 모드에서는 브라우저 TTS 사용, 서버 재생 비활성
    # playback = AudioPlayback()
    # playback.start()

    # function call 핸들러 (checkout 시 결제 화면 전환 콜백)
    async def on_checkout_fn():
        action_queue.put_nowait({"type": "checkout"})

    fn_handler = FunctionCallHandler(cart=cart, session=session, on_checkout=on_checkout_fn)

    # ── Realtime 콜백들 ──

    def on_audio_delta(b64: str):
        pass  # 텍스트 모드에서 미사용 (브라우저 TTS가 재생)

    def on_text_delta(delta: str):
        session.ai_text += delta
        push_state(session.to_dict())

    def on_user_text(text: str):
        nonlocal _pending_ai, _pending_user
        session.user_text = text
        if _pending_ai:
            # AI 응답이 이미 완성됐으면 순서대로 추가
            session.conversation_log.append({"role": "user", "text": text})
            session.conversation_log.append({"role": "ai",   "text": _pending_ai})
            _pending_ai = ""
        else:
            # AI 응답 아직 안 옴 — 대기
            _pending_user = text
        push_state(session.to_dict())

    def on_response_done():
        nonlocal _pending_ai, _pending_user
        ai_text = session.ai_text.strip()
        session.ai_text = ""
        session.conversation = "idle"
        if ai_text:
            if _pending_user:
                # user transcript가 먼저 와있으면 순서대로 추가
                session.conversation_log.append({"role": "user", "text": _pending_user})
                session.conversation_log.append({"role": "ai",   "text": ai_text})
                _pending_user = ""
            else:
                # user transcript 아직 안 옴 — 대기
                _pending_ai = ai_text
        push_state(session.to_dict())

    def on_session_ready():
        print("[Orchestrator] Realtime 준비 완료. 말씀하세요!")
        # 파일 DB에서 마지막 등록 사용자 로드 (Phase 2 mock 인식)
        known_user = get_first_user()
        greeting = "안녕하세요! 주문을 도와드릴게요. 무엇을 드시겠어요?"
        if known_user:
            session.user_name = known_user["name"]
            session.is_new_user = False
            session.speaker_verified = True
            greeting = f"어서오세요, {known_user['name']}님! 오늘도 주문 도와드릴게요."
            print(f"[Orchestrator] 등록 사용자 로드: {known_user['name']}")
        session.conversation_log = [{"role": "ai", "text": greeting}]
        _sync_and_push({"mic": "active", "screen": "waiting"})

    def on_status_update(status: str, ts: float):
        if status == "listening":
            session.ai_text = ""           # 새 발화 시작 시 텍스트 초기화
            session.on_speech_start(ts)
            session.conversation = "listening"
            if session.screen == "waiting":
                session.screen = "ordering"
        elif status == "processing":
            session.on_speech_end(ts)
            session.conversation = "processing"
        elif status in ("idle", "speaking_done"):
            session.conversation = "idle"
        push_state(session.to_dict())

    async def on_function_call(call_id: str, name: str, arguments: str):
        result_json = await fn_handler.handle(call_id, name, arguments)
        await client.send_function_result(call_id, result_json)
        push_state(session.to_dict())

    # ── 버튼 액션 처리 루프 (run() 내부 — _pending_ai/_pending_user 접근) ──
    async def action_handler():
        nonlocal _pending_ai, _pending_user
        while True:
            action = await action_queue.get()
            atype = action.get("type")

            if atype == "reset":
                _pending_ai = ""
                _pending_user = ""
                session.__init__()   # SessionState 초기화
                cart.clear()
                session.conversation_log = []
                push_state(session.to_dict())

            elif atype == "start":
                # 시작하기 버튼 → 주문 화면으로 전환
                _sync_and_push({"screen": "ordering"})

            elif atype == "checkout":
                if cart.is_empty:
                    print("[Orchestrator] 장바구니가 비어있어 결제 불가")
                    continue
                _sync_and_push({"screen": "checkout"})

            elif atype == "payment":
                await process_payment(action.get("method", "physical_card"))

            elif atype == "app_payment_confirm":
                await _do_payment()

            elif atype == "save_voice":
                if action.get("save"):
                    _sync_and_push({"screen": "register"})
                else:
                    # 저장 거부 → 바로 완료 화면
                    _sync_and_push({"screen": "complete"})

            elif atype == "register":
                name  = action.get("name", "").strip()
                phone = action.get("phone", "").strip()
                if name and phone:
                    save_user(name, phone)   # JSON 파일에 영구 저장
                    print(f"[Orchestrator] 등록 완료: {name} / {phone}")
                    _sync_and_push({"user_name": name, "is_new_user": False, "speaker_verified": True, "screen": "complete"})

            elif atype == "add_menu":
                # 화면 메뉴 카드 클릭 → 장바구니 추가 (음성 없이)
                result = cart.add_item(action.get("name", ""))
                if result.get("success"):
                    _sync_and_push({
                        "cart_items": result["cart"]["items"],
                        "cart_total": result["cart"]["total"],
                        "screen": "ordering",
                    })

            elif atype == "retry_verification":
                # 화자인증 재시도 — 잠금 해제 후 주문 화면 복귀
                session.failed_verifications = 0
                _sync_and_push({"screen": "ordering", "speaker_verified": None})

    # ── Realtime 연결 ──
    client = RealtimeClient(
        api_key=api_key,
        model=model,
        voice=voice,
        on_audio_delta=on_audio_delta,
        on_text_delta=on_text_delta,
        on_user_text=on_user_text,
        on_response_done=on_response_done,
        on_function_call=on_function_call,
        on_session_ready=on_session_ready,
        on_status_update=on_status_update,
    )

    try:
        await client.connect()
    except Exception as e:
        print(f"[Orchestrator] Realtime 연결 실패: {e}")
        playback.stop()
        return

    # 마이크 캡처
    mic = AsyncMicrophoneCapture()
    mic.start()
    print("[Orchestrator] 마이크 시작. http://localhost:8000 에서 화면을 확인하세요.")

    _MUTE_SCREENS = {"payment_processing", "register", "complete",
                      "voice_save_prompt", "card_insert", "app_payment"}

    async def mic_to_realtime():
        async for chunk in mic:
            if session.screen in _MUTE_SCREENS:
                continue   # 주문 완료 / 결제 중에는 AI가 응답 안 해도 됨
            await client.send_audio_chunk(chunk)

    try:
        await asyncio.gather(
            mic_to_realtime(),
            client.listen(),
            action_handler(),
        )
    except KeyboardInterrupt:
        pass
    finally:
        mic.stop()
        await client.close()
        print("[Orchestrator] 종료")


def main():
    server_thread = threading.Thread(target=start_frontend_server, daemon=True)
    server_thread.start()
    print("[Orchestrator] 키오스크 화면: http://localhost:8000")

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\n[Orchestrator] 종료합니다.")


if __name__ == "__main__":
    main()
