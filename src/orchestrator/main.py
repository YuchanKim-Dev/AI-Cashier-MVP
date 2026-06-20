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
import time
import threading

import uvicorn
from dotenv import load_dotenv

from src.audio.capture import AsyncMicrophoneCapture
from src.audio.playback import AudioPlayback
from src.frontend.app import app as fastapi_app, push_state, set_context
from src.orchestrator.session import SessionState
from src.realtime.client import RealtimeClient
from src.tools.cart import CartManager
from src.tools.handlers import FunctionCallHandler
from src.tools.payment import payment_gateway

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
async def process_payment(method: str):
    """결제 방법 선택 → mock 결제 처리 → 화면 전환."""
    _sync_and_push({"screen": "payment_processing", "payment_method": method})
    await asyncio.sleep(0.1)  # SSE 전송 시간 확보

    result = await payment_gateway.process(
        amount=cart.total,
        method=method,
    )

    if result["success"]:
        session.transaction_id = result["transaction_id"]
        # 신규 사용자이고 목소리가 3초 이상 쌓였으면 저장 질문
        if session.is_new_user and session.enough_voice:
            _sync_and_push({"screen": "voice_save_prompt"})
        else:
            _sync_and_push({"screen": "complete"})
    else:
        _sync_and_push({"screen": "checkout"})
        print(f"[Orchestrator] 결제 실패: {result.get('error')}")


# ─── 버튼 액션 처리 루프 ───────────────────────────────────────────────────────
async def action_handler():
    """
    frontend에서 버튼을 누르면 action_queue로 액션이 들어온다.
    각 액션을 처리해 세션 상태를 업데이트한다.
    """
    while True:
        action = await action_queue.get()
        atype = action.get("type")

        if atype == "start":
            # 시작하기 버튼 → 주문 화면으로 전환
            _sync_and_push({"screen": "ordering"})

        elif atype == "checkout":
            if cart.is_empty:
                print("[Orchestrator] 장바구니가 비어있어 결제 불가")
                continue
            _sync_and_push({"screen": "checkout"})

        elif atype == "payment":
            await process_payment(action.get("method", "physical_card"))

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
                # 3단계에서 실제 DB 저장으로 교체. 지금은 세션에만 기록.
                print(f"[Orchestrator] 등록 요청: {name} / {phone} (3단계에서 DB 저장)")
                _sync_and_push({"user_name": name, "is_new_user": False, "screen": "complete"})

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


# ─── 메인 비동기 파이프라인 ────────────────────────────────────────────────────
async def run():
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        print("[Orchestrator] OPENAI_API_KEY가 없습니다. .env 파일을 확인하세요.")
        return

    voice = os.getenv("OPENAI_REALTIME_VOICE", "alloy")
    model = os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime-2")
    loop  = asyncio.get_event_loop()

    # frontend 액션 큐 + 이벤트 루프 주입
    set_context(action_queue, loop)

    # 재생 모듈
    playback = AudioPlayback()
    playback.start()

    # function call 핸들러 (checkout 시 결제 화면 전환 콜백)
    async def on_checkout_fn():
        action_queue.put_nowait({"type": "checkout"})

    fn_handler = FunctionCallHandler(cart=cart, session=session, on_checkout=on_checkout_fn)

    # ── Realtime 콜백들 ──

    def on_audio_delta(b64: str):
        playback.play_base64(b64)
        if session.screen != "locked":
            session.conversation = "speaking"
            push_state(session.to_dict())

    def on_text_delta(delta: str):
        session.ai_text += delta
        push_state(session.to_dict())

    def on_response_done():
        session.conversation = "idle"
        push_state(session.to_dict())

    def on_session_ready():
        print("[Orchestrator] Realtime 준비 완료. 말씀하세요!")
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

    # ── Realtime 연결 ──
    client = RealtimeClient(
        api_key=api_key,
        model=model,
        voice=voice,
        on_audio_delta=on_audio_delta,
        on_text_delta=on_text_delta,
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

    async def mic_to_realtime():
        async for chunk in mic:
            # 잠금 상태이거나 결제/등록 화면이면 마이크 전송 중단
            if session.screen in ("payment_processing", "register", "complete"):
                continue
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
        playback.stop()
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
