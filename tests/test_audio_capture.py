"""
MicrophoneCapture / AsyncMicrophoneCapture 단위 테스트.

sounddevice를 모킹해 실제 마이크 없이 테스트한다.
핵심 검증: 콜백이 호출되면 큐에 정확히 청크가 들어가는가.
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from src.audio.capture import MicrophoneCapture, AsyncMicrophoneCapture, CHUNK_FRAMES


class TestMicrophoneCapture:
    def test_callback_called_on_chunk(self):
        """_sd_callback이 호출되면 on_chunk로 bytes가 전달되는지 확인."""
        received = []
        capture = MicrophoneCapture(on_chunk=lambda b: received.append(b))

        fake_pcm = b"\x00\x01" * CHUNK_FRAMES
        capture._sd_callback(fake_pcm, CHUNK_FRAMES, None, None)

        assert len(received) == 1
        assert received[0] == fake_pcm

    def test_status_warning_printed(self, capsys):
        """sounddevice 상태 경고가 있으면 출력하는지 확인."""
        capture = MicrophoneCapture(on_chunk=lambda b: None)
        capture._sd_callback(b"\x00", 1, None, "InputOverflow")
        captured = capsys.readouterr()
        assert "InputOverflow" in captured.out

    @patch("src.audio.capture.sd.RawInputStream")
    def test_start_stop(self, mock_stream_cls):
        """start/stop이 RawInputStream을 올바르게 열고 닫는지 확인."""
        mock_stream = MagicMock()
        mock_stream_cls.return_value = mock_stream

        capture = MicrophoneCapture(on_chunk=lambda b: None)
        capture.start()
        mock_stream.start.assert_called_once()

        capture.stop()
        mock_stream.stop.assert_called_once()
        mock_stream.close.assert_called_once()


class TestAsyncMicrophoneCapture:
    @pytest.mark.asyncio
    async def test_chunk_reaches_queue(self):
        """마이크 콜백 → asyncio 큐 전달 경로 테스트."""
        with patch("src.audio.capture.sd.RawInputStream"):
            capture = AsyncMicrophoneCapture()
            capture._loop = asyncio.get_event_loop()

            fake_chunk = b"\x01\x02" * 10
            # 직접 콜백 호출 (sounddevice 스레드 흉내)
            capture._on_chunk(fake_chunk)

            # 이벤트 루프가 put_nowait를 실행할 시간을 준다
            await asyncio.sleep(0)

            assert not capture.queue.empty()
            received = capture.queue.get_nowait()
            assert received == fake_chunk

    @pytest.mark.asyncio
    async def test_queue_full_drops_chunk(self):
        """큐가 꽉 찼을 때 청크가 드롭되고 예외가 발생하지 않는지 확인."""
        with patch("src.audio.capture.sd.RawInputStream"):
            capture = AsyncMicrophoneCapture(maxsize=1)
            capture._loop = asyncio.get_event_loop()

            # 큐를 먼저 채우고
            capture._put_nowait(b"first")
            # 추가 청크는 드롭돼야 함 (QueueFull 예외 없이)
            capture._put_nowait(b"dropped")

            assert capture.queue.qsize() == 1
