"""
음성 재생 모듈.

전체 흐름에서 이 모듈의 위치:
  Realtime API → base64 PCM 청크 수신 → [이 모듈] → 스피커 출력

Realtime API는 응답 오디오를 24kHz PCM16 base64로 청크 스트리밍한다.
청크를 수신할 때마다 즉시 재생해 지연을 최소화한다.
"""

import asyncio
import base64
import queue
import threading
from typing import Optional

import numpy as np
import sounddevice as sd


# Realtime API 응답 오디오 포맷: 24kHz mono int16
PLAYBACK_SAMPLE_RATE = 24000
CHANNELS = 1
DTYPE = "int16"


class AudioPlayback:
    """
    PCM bytes를 실시간으로 스피커에 출력하는 클래스.

    백그라운드 스레드에서 sounddevice OutputStream을 열고,
    외부에서 play(pcm_bytes)를 호출하면 큐를 통해 비동기적으로 재생한다.
    메인 스레드(asyncio 루프)가 블로킹되지 않도록 분리.
    """

    def __init__(
        self,
        sample_rate: int = PLAYBACK_SAMPLE_RATE,
        device: Optional[int] = None,
    ):
        self.sample_rate = sample_rate
        self.device = device
        self._queue: queue.Queue[Optional[np.ndarray]] = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._playback_loop, daemon=True)
        self._thread.start()
        print(f"[AudioPlayback] 재생 준비: {self.sample_rate}Hz")

    def stop(self):
        # None을 넣어 재생 스레드의 루프를 종료 신호로 사용
        self._running = False
        self._queue.put(None)
        if self._thread:
            self._thread.join(timeout=2)

    def play(self, pcm_bytes: bytes):
        """PCM bytes(int16)를 재생 큐에 넣는다. asyncio 스레드에서 호출 가능."""
        audio = np.frombuffer(pcm_bytes, dtype=np.int16)
        self._queue.put(audio)

    def play_base64(self, b64_audio: str):
        """Realtime API의 base64 인코딩 PCM을 디코딩해 재생 큐에 넣는다."""
        pcm_bytes = base64.b64decode(b64_audio)
        self.play(pcm_bytes)

    def _playback_loop(self):
        # OutputStream은 콜백 방식 대신 write() 방식으로 사용 — 청크 단위 제어에 유리.
        with sd.OutputStream(
            samplerate=self.sample_rate,
            channels=CHANNELS,
            dtype=DTYPE,
            device=self.device,
        ) as stream:
            while self._running:
                audio = self._queue.get()
                if audio is None:
                    break
                stream.write(audio)
