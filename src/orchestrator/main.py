"""
오케스트레이터 — 1단계 진입점.

전체 흐름:
  1. FastAPI 서버를 백그라운드에서 시작 (키오스크 화면)
  2. Realtime API에 연결
  3. 마이크 캡처 시작 → PCM 청크를 Realtime으로 전송
  4. Realtime 응답(오디오 델타) → 스피커 재생 + 화면 업데이트

1단계 제약: 화자인증 없음, 캐셔 function calling 없음, 잡담 모드.
"""

import asyncio
import os
import threading

import uvicorn
from dotenv import load_dotenv

from src.audio.capture import AsyncMicrophoneCapture
from src.audio.playback import AudioPlayback
from src.frontend.app import app as fastapi_app
from src.frontend.app import update_state, append_ai_text, clear_ai_text
from src.realtime.client import RealtimeClient


load_dotenv()


def start_frontend_server():
    """FastAPI 키오스크 서버를 별도 스레드에서 실행 (asyncio 루프와 분리)."""
    config = uvicorn.Config(
        fastapi_app,
        host="0.0.0.0",
        port=8000,
        log_level="warning",  # orchestrator 로그가 묻히지 않도록
    )
    server = uvicorn.Server(config)
    server.run()


async def run():
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        print("[Orchestrator] OPENAI_API_KEY가 .env에 없습니다. .env 파일을 확인하세요.")
        return

    voice = os.getenv("OPENAI_REALTIME_VOICE", "alloy")
    model = os.getenv("OPENAI_REALTIME_MODEL", "gpt-realtime-2")

    # 재생 모듈 시작
    playback = AudioPlayback()
    playback.start()

    # Realtime 클라이언트 콜백 연결
    def on_audio_delta(b64: str):
        # 오디오 델타: 재생 큐에 넣고 상태를 speaking으로
        playback.play_base64(b64)
        update_state({"conversation": "speaking"})

    def on_text_delta(delta: str):
        # AI 발화 텍스트 실시간 누적 → 화면에 표시
        append_ai_text(delta)

    def on_session_ready():
        print("[Orchestrator] Realtime 세션 준비 완료. 말씀하세요!")
        update_state({"mic": "active", "conversation": "idle"})

    def on_status_update(status: str):
        # listening / processing / idle 상태 변경
        if status == "listening":
            clear_ai_text()
        update_state({"conversation": status})

    client = RealtimeClient(
        api_key=api_key,
        model=model,
        voice=voice,
        on_audio_delta=on_audio_delta,
        on_text_delta=on_text_delta,
        on_session_ready=on_session_ready,
        on_status_update=on_status_update,
    )

    # Realtime API 연결
    try:
        await client.connect()
    except Exception as e:
        print(f"[Orchestrator] Realtime 연결 실패: {e}")
        playback.stop()
        return

    # 마이크 캡처 시작
    mic = AsyncMicrophoneCapture()
    mic.start()
    update_state({"mic": "active"})
    print("[Orchestrator] 마이크 시작. 브라우저에서 http://localhost:8000 을 열어 화면을 확인하세요.")

    async def mic_to_realtime():
        """마이크 청크를 읽어 Realtime API로 전송하는 루프."""
        async for chunk in mic:
            await client.send_audio_chunk(chunk)

    # 마이크 전송과 Realtime 이벤트 수신을 동시에 실행
    try:
        await asyncio.gather(
            mic_to_realtime(),
            client.listen(),
        )
    except KeyboardInterrupt:
        pass
    finally:
        mic.stop()
        await client.close()
        playback.stop()
        print("[Orchestrator] 종료")


def main():
    # 키오스크 화면 서버를 백그라운드 스레드에서 먼저 시작
    server_thread = threading.Thread(target=start_frontend_server, daemon=True)
    server_thread.start()
    print("[Orchestrator] 키오스크 화면: http://localhost:8000")

    # asyncio 이벤트 루프에서 메인 파이프라인 실행
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        print("\n[Orchestrator] Ctrl+C 감지. 종료합니다.")


if __name__ == "__main__":
    main()
