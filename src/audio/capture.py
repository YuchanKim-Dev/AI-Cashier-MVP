"""
마이크 PCM 캡처 모듈.

전체 흐름에서 이 모듈의 위치:
  마이크 → [이 모듈] → PCM 청크를 두 곳에 분기
    - Realtime API 클라이언트 (1초 응답 경로)
    - 화자인증 모듈 (비동기 백그라운드, 3단계에서 연결)

sounddevice가 PortAudio를 통해 마이크에서 raw PCM(int16, 16kHz, mono)을
읽고 asyncio.Queue에 청크를 넣는다. 소비자가 큐에서 꺼내 각자 처리한다.
"""

import asyncio
import queue
import threading
from typing import Callable, Optional

import numpy as np
import sounddevice as sd


# Realtime API가 요구하는 포맷: 16kHz mono int16
SAMPLE_RATE = 24000   # GA Realtime API 최소 요구치 24kHz (3단계 화자인증은 다운샘플링 예정)
CHANNELS = 1
DTYPE = "int16"
# 한 청크당 프레임 수 — 100ms. 너무 작으면 오버헤드, 너무 크면 지연.
CHUNK_FRAMES = int(SAMPLE_RATE * 0.1)


class MicrophoneCapture:
    """
    마이크에서 raw PCM을 캡처해 콜백으로 전달하는 클래스.

    사용 패턴:
        capture = MicrophoneCapture(on_chunk=my_callback)
        capture.start()
        ...
        capture.stop()

    on_chunk: bytes → None 형태의 콜백. sounddevice 내부 스레드에서 호출되므로
    콜백 안에서 asyncio 코루틴을 직접 await 하면 안 된다.
    asyncio와 연동하려면 loop.call_soon_threadsafe 또는 Queue를 사용한다.
    """

    def __init__(
        self,
        on_chunk: Callable[[bytes], None],
        sample_rate: int = SAMPLE_RATE,
        chunk_frames: int = CHUNK_FRAMES,
        device: Optional[int] = None,
    ):
        self.on_chunk = on_chunk
        self.sample_rate = sample_rate
        self.chunk_frames = chunk_frames
        self.device = device
        self._stream: Optional[sd.RawInputStream] = None

    def _sd_callback(self, indata: bytes, frames: int, time, status):
        # sounddevice가 매 chunk_frames마다 호출. indata는 bytes (int16 raw PCM).
        # status가 있으면 드라이버 레벨 경고(overflow 등) — 무시하지 않고 출력.
        if status:
            print(f"[MicCapture] sounddevice 상태 경고: {status}")
        self.on_chunk(bytes(indata))

    def start(self):
        # RawInputStream: numpy 변환 없이 bytes로 직접 받아 복사 비용 최소화.
        self._stream = sd.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=self.chunk_frames,
            device=self.device,
            channels=CHANNELS,
            dtype=DTYPE,
            callback=self._sd_callback,
        )
        self._stream.start()
        print(f"[MicCapture] 마이크 캡처 시작: {self.sample_rate}Hz, {self.chunk_frames}frames/chunk")

    def stop(self):
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
            print("[MicCapture] 마이크 캡처 중지")


class AsyncMicrophoneCapture:
    """
    asyncio 환경에서 마이크 PCM을 사용하기 위한 래퍼.

    MicrophoneCapture의 콜백을 asyncio.Queue에 연결한다.
    소비자는 `async for chunk in capture` 또는 `await capture.queue.get()`으로 청크를 읽는다.
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        chunk_frames: int = CHUNK_FRAMES,
        device: Optional[int] = None,
        maxsize: int = 100,
    ):
        self.queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=maxsize)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._capture: Optional[MicrophoneCapture] = None
        self.sample_rate = sample_rate
        self.chunk_frames = chunk_frames
        self.device = device

    def _on_chunk(self, chunk: bytes):
        # sounddevice 스레드 → asyncio 큐 안전하게 넣기.
        # call_soon_threadsafe: 다른 스레드에서 이벤트 루프에 콜백을 예약하는 유일하게 안전한 방법.
        if self._loop:
            self._loop.call_soon_threadsafe(self._put_nowait, chunk)

    def _put_nowait(self, chunk: bytes):
        try:
            self.queue.put_nowait(chunk)
        except asyncio.QueueFull:
            # 소비자가 느리면 드롭. 실시간 스트리밍이라 오래된 청크보다 최신이 더 중요.
            pass

    def start(self):
        self._loop = asyncio.get_event_loop()
        self._capture = MicrophoneCapture(
            on_chunk=self._on_chunk,
            sample_rate=self.sample_rate,
            chunk_frames=self.chunk_frames,
            device=self.device,
        )
        self._capture.start()

    def stop(self):
        if self._capture:
            self._capture.stop()

    def __aiter__(self):
        return self

    async def __anext__(self) -> bytes:
        return await self.queue.get()
