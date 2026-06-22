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

import certifi
import uvicorn
from dotenv import load_dotenv

# 맥 Python에서 SSL 인증서를 못 찾는 문제 해결
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

from src.frontend.app import (
    app as fastapi_app,
    get_new_session_queue,
    get_session,
    remove_session,
    push_session_state,
)
from src.orchestrator.session import SessionState
from src.realtime.client import RealtimeClient
from src.audio.speaker_verify import preload_model, extract_embedding, find_user
from src.tools.cart import CartManager
from src.tools.handlers import FunctionCallHandler
from src.tools.payment import payment_gateway
from src.tools.user_store import (
    save_user, get_all_users, get_first_user,
    find_user_by_phone, update_user_preferences,
)

load_dotenv()


async def _start_frontend_server():
    config = uvicorn.Config(fastapi_app, host="0.0.0.0", port=8000, log_level="warning")
    server = uvicorn.Server(config)
    await server.serve()


# ─── 세션 워커 ─────────────────────────────────────────────────────────────────
async def run_session(session_id: str):
    """접속한 브라우저마다 독립 실행되는 파이프라인."""
    sid = session_id[:8]  # 로그용 축약 ID
    sess_data = get_session(session_id)
    if not sess_data:
        return

    audio_queue  = sess_data["audio_queue"]
    action_queue = sess_data["action_queue"]
    session      = SessionState()
    cart         = CartManager()

    def _push(updates: dict = None):
        if updates:
            for k, v in updates.items():
                setattr(session, k, v)
        push_session_state(session_id, session.to_dict())

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        print(f"[{sid}] OPENAI_API_KEY 없음")
        return

    _pending_ai: str = ""
    _pending_user: str = ""
    _voice_buffer: bytearray = bytearray()
    _MAX_BUFFER   = 24000 * 2 * 6
    _is_listening = False
    _verification_done = False
    voice = os.getenv("OPENAI_REALTIME_VOICE", "alloy")
    model = os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime-2")
    loop  = asyncio.get_event_loop()

    async def on_checkout_fn():
        await action_queue.put({"type": "checkout"})

    async def on_select_payment_fn(method: str):
        await action_queue.put({"type": "payment", "method": method})

    fn_handler = FunctionCallHandler(
        cart=cart,
        session=session,
        on_checkout=on_checkout_fn,
        on_select_payment=on_select_payment_fn,
    )

    def on_audio_delta(b64: str):
        pass

    def on_text_delta(delta: str):
        session.ai_text += delta
        push_session_state(session_id, session.to_dict())

    def on_user_text_delta(delta: str):
        session.user_text = (session.user_text or "") + delta
        push_session_state(session_id, session.to_dict())

    def on_user_text(text: str):
        nonlocal _pending_ai, _pending_user
        session.user_text = text
        if _pending_ai:
            session.conversation_log.append({"role": "user", "text": text})
            session.conversation_log.append({"role": "ai",   "text": _pending_ai})
            _pending_ai = ""
        else:
            _pending_user = text
        push_session_state(session_id, session.to_dict())

    def on_response_done():
        nonlocal _pending_ai, _pending_user
        ai_text = session.ai_text.strip()
        session.ai_text = ""
        session.conversation = "idle"
        if ai_text:
            if _pending_user:
                session.conversation_log.append({"role": "user", "text": _pending_user})
                session.conversation_log.append({"role": "ai",   "text": ai_text})
                _pending_user = ""
            else:
                _pending_ai = ai_text
        push_session_state(session_id, session.to_dict())

    def on_session_ready():
        nonlocal _verification_done, _voice_buffer, _is_listening
        print(f"[{sid}] Realtime 준비 완료.")
        has_registered = bool(get_all_users())
        if has_registered:
            greeting = "안녕하세요! 목소리를 확인하는 중입니다. 말씀해 주세요."
        else:
            greeting = "안녕하세요! 주문을 도와드릴게요. 무엇을 드시겠어요?"
        _verification_done = False
        _voice_buffer = bytearray()
        _is_listening  = False
        session.conversation_log = [{"role": "ai", "text": greeting}]
        _push({"mic": "active", "screen": "waiting"})

    async def _run_speaker_verify():
        nonlocal _verification_done
        if _verification_done:
            return
        _verification_done = True
        audio_data = bytes(_voice_buffer)
        if len(audio_data) < 48_000:
            print(f"[{sid}] 오디오 불충분 — 화자인식 건너뜀")
            return
        try:
            all_users = get_all_users()
            users_with_embedding = [u for u in all_users if u.get("embedding")]
            if not users_with_embedding:
                # 등록된 목소리 없음 → 불일치가 아니라 그냥 미확인 상태 유지
                print(f"[{sid}] 등록된 목소리 없음 — 화자인식 건너뜀")
                session.speaker_verified = None
                push_session_state(session_id, session.to_dict())
                return
            emb = await extract_embedding(audio_data, sample_rate=24000)
            if emb is None:
                return
            match = find_user(emb, users_with_embedding)
            if match:
                session.user_name        = match["name"]
                session.is_new_user      = False
                session.speaker_verified = True
                print(f"[{sid}] 인식됨: {match['name']}")
                await client.greet_returning_user(match["name"])
                push_session_state(session_id, session.to_dict())
            else:
                print(f"[{sid}] 유사도 낮음 — 불일치")
                session.speaker_verified = False
                push_session_state(session_id, session.to_dict())
        except Exception as e:
            print(f"[{sid}] 화자인식 오류: {e}")

    def on_status_update(status: str, ts: float):
        nonlocal _is_listening
        if status == "listening":
            _is_listening = True
            session.ai_text = ""
            session.user_text = ""  # 새 발화 시작 — 실시간 자막 초기화
            session.on_speech_start(ts)
            session.conversation = "listening"
            if session.screen == "waiting":
                session.screen = "ordering"
        elif status == "processing":
            _is_listening = False
            session.on_speech_end(ts)
            session.conversation = "processing"
            if not _verification_done and len(_voice_buffer) >= 48_000:
                loop.create_task(_run_speaker_verify())
        elif status in ("idle", "speaking_done"):
            _is_listening = False
            session.conversation = "idle"
        push_session_state(session_id, session.to_dict())

    async def on_function_call(call_id: str, name: str, arguments: str):
        result_json = await fn_handler.handle(call_id, name, arguments)
        await client.send_function_result(call_id, result_json)
        push_session_state(session_id, session.to_dict())

    async def _save_preferences():
        """결제 완료 후 대화 기반 취향 요약을 백그라운드에서 저장."""
        if not session.user_name or len(session.conversation_log) < 3:
            return
        try:
            from openai import AsyncOpenAI
            oai = AsyncOpenAI(api_key=api_key)
            convo = "\n".join(
                f"{'고객' if m['role']=='user' else 'AI'}: {m['text']}"
                for m in session.conversation_log[:20]
            )
            resp = await oai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content":
                     "AI 캐셔 대화를 분석해 고객 취향을 한 문장으로 요약하세요. "
                     "주문한 메뉴, 선호도, 특이사항을 포함하세요. "
                     "예: '불고기버거+감자튀김 조합 선호, 음료는 콜라 선택함'"},
                    {"role": "user", "content": convo},
                ],
                max_tokens=80,
            )
            summary = resp.choices[0].message.content.strip()
            update_user_preferences(session.user_name, summary)
            print(f"[{sid}] 취향 저장: {summary}")
        except Exception as e:
            print(f"[{sid}] 취향 저장 실패: {e}")

    async def _do_payment():
        _push({"screen": "payment_processing"})
        await asyncio.sleep(0.1)
        result = await payment_gateway.process(
            amount=cart.total, method=session.payment_method or "physical_card"
        )
        if result["success"]:
            session.transaction_id = result["transaction_id"]
            _push({"screen": "complete"})
            # 취향 요약 백그라운드 저장
            loop.create_task(_save_preferences())
        else:
            _push({"screen": "checkout"})
            print(f"[{sid}] 결제 실패: {result.get('error')}")

    async def process_payment(method: str):
        _push({"payment_method": method})
        if method == "physical_card":
            _push({"screen": "card_insert"})
            await asyncio.sleep(3)
        elif method == "app_card":
            _push({"screen": "app_payment"})
            return
        await _do_payment()

    async def action_handler():
        nonlocal _pending_ai, _pending_user
        while True:
            action = await action_queue.get()
            atype = action.get("type")

            if atype == "reset":
                _pending_ai = ""
                _pending_user = ""
                session.__init__()
                cart.clear()
                session.conversation_log = []
                push_session_state(session_id, session.to_dict())

            elif atype == "start":
                _push({"screen": "ordering"})

            elif atype == "checkout":
                if cart.is_empty:
                    continue
                _push({"screen": "checkout"})

            elif atype == "payment":
                await process_payment(action.get("method", "physical_card"))

            elif atype == "app_payment_confirm":
                await _do_payment()

            elif atype == "save_voice":
                if action.get("save"):
                    _push({"screen": "register"})
                else:
                    _push({"screen": "complete"})

            elif atype == "register":
                name  = action.get("name", "").strip()
                phone = action.get("phone", "").strip()
                if name and phone:
                    emb = None
                    if len(_voice_buffer) >= 48_000:
                        try:
                            emb = await extract_embedding(bytes(_voice_buffer))
                        except Exception as e:
                            print(f"[{sid}] 임베딩 추출 실패: {e}")
                    save_user(name, phone, embedding=emb)
                    print(f"[{sid}] 등록 완료: {name} / {phone}")
                    _push({"user_name": name, "is_new_user": False, "speaker_verified": True, "screen": "complete"})

            elif atype == "add_menu":
                result = cart.add_item(action.get("name", ""))
                if result.get("success"):
                    _push({
                        "cart_items": result["cart"]["items"],
                        "cart_total": result["cart"]["total"],
                        "screen": "ordering",
                    })

            elif atype == "identify":
                name  = action.get("name", "").strip()
                phone = action.get("phone", "").strip()
                if name and phone:
                    # 현재 세션에서 녹음된 목소리 임베딩 추출
                    emb = None
                    if len(_voice_buffer) >= 48_000:
                        try:
                            emb = await extract_embedding(bytes(_voice_buffer), sample_rate=24000)
                        except Exception as e:
                            print(f"[{sid}] identify 임베딩 추출 실패: {e}")
                    existing = find_user_by_phone(phone)
                    if existing:
                        save_user(name, phone, embedding=emb or existing.get("embedding"))
                        session.is_new_user = False
                        print(f"[{sid}] 기존 사용자 확인 + 목소리 업데이트: {name}")
                    else:
                        save_user(name, phone, embedding=emb)
                        session.is_new_user = False
                        print(f"[{sid}] 신규 등록 (목소리{'저장됨' if emb else ' 없음'}): {name}")
                    session.user_name  = name
                    session.user_phone = phone
                    _push({"user_name": name, "user_phone": phone, "is_new_user": session.is_new_user})

            elif atype == "retry_verification":
                session.failed_verifications = 0
                _push({"screen": "ordering", "speaker_verified": None})

    client = RealtimeClient(
        api_key=api_key,
        model=model,
        voice=voice,
        on_audio_delta=on_audio_delta,
        on_text_delta=on_text_delta,
        on_user_text=on_user_text,
        on_user_text_delta=on_user_text_delta,
        on_response_done=on_response_done,
        on_function_call=on_function_call,
        on_session_ready=on_session_ready,
        on_status_update=on_status_update,
    )

    try:
        await client.connect()
    except Exception as e:
        print(f"[{sid}] Realtime 연결 실패: {e}")
        remove_session(session_id)
        return

    print(f"[{sid}] 세션 시작. 브라우저 마이크 대기 중.")

    # checkout 화면에서도 마이크 유지 — 결제 수단 음성 선택 가능
    _MUTE_SCREENS = {"payment_processing", "register", "complete",
                      "voice_save_prompt", "card_insert", "app_payment"}

    async def mic_to_realtime():
        while True:
            chunk = await audio_queue.get()
            if session.screen in _MUTE_SCREENS:
                continue
            if not _verification_done and len(_voice_buffer) < _MAX_BUFFER:
                _voice_buffer.extend(chunk)
            await client.send_audio_chunk(chunk)

    try:
        await asyncio.gather(
            mic_to_realtime(),
            client.listen(),
            action_handler(),
        )
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        await client.close()
        remove_session(session_id)
        print(f"[{sid}] 세션 종료")


# ─── 세션 디스패처 ──────────────────────────────────────────────────────────────
async def run():
    new_session_q = get_new_session_queue()
    print("[Orchestrator] 준비 완료. http://localhost:8000 에서 접속하세요.")
    while True:
        session_id = await new_session_q.get()
        print(f"[Orchestrator] 새 세션: {session_id[:8]}")
        asyncio.create_task(run_session(session_id))


def main():
    print("[Orchestrator] 키오스크 화면: http://localhost:8000")

    async def _run_all():
        asyncio.create_task(preload_model())
        # uvicorn과 오케스트레이터를 같은 이벤트 루프에서 실행 — 큐 공유 안전
        await asyncio.gather(
            _start_frontend_server(),
            run(),
        )

    try:
        asyncio.run(_run_all())
    except KeyboardInterrupt:
        print("\n[Orchestrator] 종료합니다.")


if __name__ == "__main__":
    main()
