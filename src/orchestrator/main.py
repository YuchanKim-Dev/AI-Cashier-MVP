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
import base64
import os
import re as _re
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
    push_audio_out,
)
from src.orchestrator.session import SessionState
from src.realtime.client import RealtimeClient
from src.audio.speaker_verify import preload_model, extract_embedding, find_user, cosine_sim
from src.tools.cart import CartManager
from src.tools.handlers import FunctionCallHandler
from src.tools.payment import payment_gateway
from src.tools.user_store import (
    save_user, get_all_users, get_first_user,
    find_user_by_phone, update_user_preferences,
)

load_dotenv()


async def _start_frontend_server(server_holder: list):
    config = uvicorn.Config(fastapi_app, host="0.0.0.0", port=8000, log_level="critical")
    server = uvicorn.Server(config)
    server_holder.append(server)
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

    _voice_buffer: bytearray = bytearray()
    _MAX_BUFFER   = 24000 * 2 * 6
    _is_listening = False
    _ai_speaking = False          # AI 오디오 재생 중 → 마이크 뮤트
    _ai_audio_bytes = 0           # 현재 응답에서 전송한 오디오 바이트 수
    _mic_reenable_scheduled = False  # 이중 스케줄 방지
    _detected_language: str | None = None  # "ko" | "en" — 첫 발화로 결정
    _check_buffer = bytearray()   # 발화별 화자 확인용 버퍼
    _verified_embedding: list | None = None  # 인증된 사용자의 임베딩
    _utterance_count = 0          # 이번 세션 발화 횟수
    voice = os.getenv("OPENAI_REALTIME_VOICE", "coral")
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
        nonlocal _ai_speaking, _ai_audio_bytes, _mic_reenable_scheduled
        _ai_speaking = True
        _mic_reenable_scheduled = False  # 새 응답 시작 시 플래그 초기화
        pcm = base64.b64decode(b64)
        _ai_audio_bytes += len(pcm)
        push_audio_out(session_id, pcm)

    def on_text_delta(delta: str):
        session.ai_text += delta
        push_session_state(session_id, session.to_dict())

    def on_ai_transcript_done(transcript: str):
        """AI 발화 전사 완료 — 즉시 로그에 추가."""
        # 너무 짧은 전사(중단된 응답) 무시
        if len(transcript) < 4:
            return
        session.ai_text = transcript
        session.conversation_log.append({"role": "ai", "text": transcript})
        print(f"[{sid}] AI 로그 추가: {transcript[:60]!r}")
        push_session_state(session_id, session.to_dict())

    def on_user_text_delta(delta: str):
        session.user_text = (session.user_text or "") + delta
        push_session_state(session_id, session.to_dict())

    def _is_noise(text: str) -> bool:
        """에코/배경소음 필터.
        - 한국어 감지 → 정상 입력
        - 영어이고 _detected_language != 'ko' → 실제 영어 발화로 허용
        - 그 외 → 에코/소음으로 무시
        """
        t = text.strip()
        if len(t) < 2:
            return True
        has_korean = any('가' <= c <= '힣' or 'ㄱ' <= c <= 'ㅎ' for c in t)
        if has_korean:
            return False
        # 한글 없음 — 영어 여부 판단
        eng_words = _re.findall(r'[a-zA-Z]{2,}', t)
        is_substantial = len(eng_words) >= 2 or (len(eng_words) == 1 and len(eng_words[0]) >= 5 and len(t) >= 8)
        if is_substantial and _detected_language != "ko":
            return False  # 영어 모드 or 아직 미결정 → 실제 영어 입력으로 처리
        print(f"[{sid}] 비한국어 전사 무시 (에코/소음): {t[:40]!r}")
        return True

    def on_user_text(text: str):
        nonlocal _detected_language, _utterance_count
        if _is_noise(text):
            return
        has_korean = any('가' <= c <= '힣' or 'ㄱ' <= c <= 'ㅎ' for c in text)
        # 첫 발화로 언어 결정
        if _detected_language is None:
            _detected_language = "ko" if has_korean else "en"
            if _detected_language == "en":
                print(f"[{sid}] 영어 감지 → 영어 모드로 전환")
                loop.create_task(client.set_language("en"))
        # 한국어 모드인데 비한국어가 오면 에코/소음
        elif _detected_language == "ko" and not has_korean:
            print(f"[{sid}] 한국어 모드 영어 무시: {text[:40]!r}")
            return
        _utterance_count += 1
        session.user_text = text
        session.conversation_log.append({"role": "user", "text": text})
        print(f"[{sid}] 사용자 로그 추가: {text[:60]!r}")
        push_session_state(session_id, session.to_dict())
        # 매 발화마다 화자 확인 (등록된 사람이 있을 때)
        audio_snap = bytes(_check_buffer)
        loop.create_task(_verify_speaker(audio_snap))

    def on_response_done():
        # 로그 처리는 on_ai_transcript_done에서 이미 완료
        # 만약 transcript_done 이벤트가 안 왔다면 여기서 fallback 처리
        session.conversation = "idle"
        leftover = session.ai_text.strip()
        session.ai_text = ""
        if leftover:
            # transcript_done 없이 response.done이 먼저 온 경우 fallback
            already = any(
                m["role"] == "ai" and m["text"] == leftover
                for m in session.conversation_log[-3:]
            )
            if not already:
                session.conversation_log.append({"role": "ai", "text": leftover})
                print(f"[{sid}] AI 로그 fallback: {leftover[:60]!r}")
        push_session_state(session_id, session.to_dict())

    def on_session_ready():
        nonlocal _voice_buffer, _is_listening
        print(f"[{sid}] Realtime 준비 완료.")
        _voice_buffer = bytearray()
        _is_listening  = False
        session.conversation_log = []
        _push({"mic": "active", "screen": "waiting"})

    async def _verify_speaker(audio: bytes):
        """매 발화마다 호출. 처음엔 등록 유저 매칭, 이후엔 동일 화자 확인."""
        nonlocal _verified_embedding
        # 차단된 상태(False)에서 복구 시도는 낮은 조건(1초), 정상 확인은 높은 조건(2초)
        if _verified_embedding is not None:
            min_bytes = 48_000 if session.speaker_verified is False else 96_000
        else:
            min_bytes = 48_000
        if len(audio) < min_bytes:
            return
        try:
            all_users = get_all_users()
            users_with_embedding = [u for u in all_users if u.get("embedding")]
            if not users_with_embedding:
                return  # 등록된 목소리 없음 — 화자인식 불필요
            emb = await extract_embedding(audio, sample_rate=24000)
            if emb is None:
                return
            if _verified_embedding is not None:
                # 인식된 사람이 계속 말하는지 확인 (다른 사람 끼어들었는지)
                sim = cosine_sim(emb, _verified_embedding)
                print(f"[{sid}] 화자 연속 확인: 유사도 {sim:.3f}")
                if sim < 0.15:
                    print(f"[{sid}] 다른 목소리 감지!")
                    session.speaker_verified = False  # 결제 차단
                    push_session_state(session_id, session.to_dict())
                    # 진행 중인 AI 응답을 끊고 바로 경고
                    await client.cancel_response()
                    await asyncio.sleep(0.15)
                    await client._send({
                        "type": "response.create",
                        "response": {
                            "instructions": (
                                "지금 다른 분 목소리가 감지됐어요. "
                                f"처음에 주문 시작하신 {session.user_name}님이 직접 말씀해 주시겠어요?"
                            )
                        }
                    })
                else:
                    # 원래 사람으로 확인됨 — 차단 해제
                    if session.speaker_verified is False:
                        print(f"[{sid}] 화자 복구됨: {session.user_name}")
                        session.speaker_verified = True
                        push_session_state(session_id, session.to_dict())
            else:
                # 아직 인식 전 — 등록 유저 중 매칭 시도
                match = find_user(emb, users_with_embedding)
                if match:
                    _verified_embedding = emb
                    session.user_name        = match["name"]
                    session.is_new_user      = False
                    session.speaker_verified = True
                    print(f"[{sid}] 인식됨: {match['name']}")
                    await client.greet_returning_user(match["name"])
                else:
                    print(f"[{sid}] 유사도 낮음 — 불일치")
                    session.speaker_verified = False
                push_session_state(session_id, session.to_dict())
        except Exception as e:
            print(f"[{sid}] 화자 확인 오류: {e}")

    def on_status_update(status: str, ts: float):
        nonlocal _is_listening, _check_buffer
        if status == "listening":
            _is_listening = True
            _check_buffer = bytearray()  # 이번 발화 시작 시 초기화
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
        elif status == "speaking_done":
            # 오디오 전송 완료 → 실제 재생 시간만큼 마이크 뮤트 유지
            nonlocal _mic_reenable_scheduled
            _is_listening = False
            session.conversation = "idle"
            if not _mic_reenable_scheduled:
                _mic_reenable_scheduled = True
                audio_duration = _ai_audio_bytes / (2 * 24000) / 1.35  # 1.35배속 반영
                mute_delay = max(audio_duration + 0.5, 0.8)
                print(f"[{sid}] 마이크 재활성 대기: {mute_delay:.2f}s (오디오 {audio_duration:.2f}s)")
                async def _reenable_mic(delay=mute_delay):
                    nonlocal _ai_speaking, _ai_audio_bytes, _mic_reenable_scheduled
                    await asyncio.sleep(delay)
                    _ai_audio_bytes = 0
                    _ai_speaking = False
                    _mic_reenable_scheduled = False
                    print(f"[{sid}] 마이크 재활성")
                loop.create_task(_reenable_mic())
        elif status == "idle":
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
        nonlocal _is_listening, _ai_speaking, _ai_audio_bytes, _mic_reenable_scheduled
        nonlocal _voice_buffer, _detected_language
        nonlocal _verified_embedding, _utterance_count
        while True:
            action = await action_queue.get()
            atype = action.get("type")

            if atype == "reset":
                _is_listening = False
                _ai_speaking = False
                _ai_audio_bytes = 0
                _mic_reenable_scheduled = False
                _check_buffer.clear()
                session.__init__()
                cart.clear()
                # 진행 중인 Realtime 응답 취소
                loop.create_task(client.cancel_response())
                push_session_state(session_id, session.to_dict())

            elif atype == "start":
                # 매 대화마다 상태 초기화
                _voice_buffer = bytearray()
                _check_buffer.clear()
                _detected_language = None
                _verified_embedding = None
                _utterance_count = 0
                _ai_speaking = False
                _ai_audio_bytes = 0
                _mic_reenable_scheduled = False
                session.speaker_verified = None
                session.conversation_log = []
                _push({"screen": "ordering"})
                loop.create_task(client.send_initial_greeting())

            elif atype == "checkout":
                if cart.is_empty:
                    continue
                _push({"screen": "checkout"})

            elif atype == "payment":
                # 결제 직전 화자 강제 재확인 (짧은 발화라도 최소 0.5초 이상이면 체크)
                if _verified_embedding is not None:
                    snap = bytes(_check_buffer)
                    if len(snap) >= 24_000:
                        try:
                            emb = await extract_embedding(snap, sample_rate=24000)
                            if emb is not None:
                                sim = cosine_sim(emb, _verified_embedding)
                                print(f"[{sid}] 결제 화자 재확인: 유사도 {sim:.3f}")
                                if sim < 0.15:
                                    session.speaker_verified = False
                                    push_session_state(session_id, session.to_dict())
                        except Exception as e:
                            print(f"[{sid}] 결제 화자 확인 오류: {e}")
                # 등록 사용자가 있는데 다른 목소리가 감지된 상태면 결제 거부
                if _verified_embedding is not None and session.speaker_verified is False:
                    print(f"[{sid}] 결제 차단 — 화자 불일치")
                    await client.cancel_response()
                    await asyncio.sleep(0.15)
                    await client._send({
                        "type": "response.create",
                        "response": {
                            "instructions": (
                                f"결제는 주문하신 {session.user_name}님만 진행할 수 있어요. "
                                f"{session.user_name}님이 직접 말씀해 주시면 바로 도와드릴게요."
                            )
                        }
                    })
                else:
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
        on_ai_transcript_done=on_ai_transcript_done,
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
            if not _ai_speaking:
                # 침묵 제외 — VAD가 발화 감지할 때만 버퍼에 수집 (임베딩 품질 향상)
                if _is_listening:
                    if len(_voice_buffer) < _MAX_BUFFER:
                        _voice_buffer.extend(chunk)
                    if len(_check_buffer) < _MAX_BUFFER:
                        _check_buffer.extend(chunk)
            if _ai_speaking:
                continue  # Realtime API 전송만 막음 — echo loop 방지
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
        import signal
        import sys
        loop = asyncio.get_event_loop()
        stop_event = asyncio.Event()
        server_holder: list = []

        def _handle_sigint():
            if stop_event.is_set():
                # 두 번째 Ctrl+C → 즉시 강제 종료
                print("\n[Orchestrator] 강제 종료합니다.")
                os._exit(0)
            stop_event.set()
            print("\n[Orchestrator] 종료 중... (한 번 더 누르면 강제 종료)")

        loop.add_signal_handler(signal.SIGINT,  _handle_sigint)
        loop.add_signal_handler(signal.SIGTERM, _handle_sigint)

        asyncio.create_task(preload_model())
        server_task = asyncio.create_task(_start_frontend_server(server_holder))
        run_task    = asyncio.create_task(run())

        await stop_event.wait()

        # uvicorn graceful 종료 (should_exit → 에러 없이 종료)
        if server_holder:
            server_holder[0].should_exit = True
        run_task.cancel()
        await asyncio.gather(server_task, run_task, return_exceptions=True)

    try:
        asyncio.run(_run_all())
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        print("\n[Orchestrator] 종료합니다.")


if __name__ == "__main__":
    main()
