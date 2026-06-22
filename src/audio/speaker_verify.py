"""
화자 인식 — SpeechBrain ECAPA-TDNN.

24kHz PCM 오디오 → 임베딩 → 코사인 유사도 비교.
모델은 서버 시작 시 백그라운드에서 미리 로드한다.
"""

import asyncio
import numpy as np
from concurrent.futures import ThreadPoolExecutor

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="speaker")
_model = None


def _load_model_sync():
    global _model
    if _model is not None:
        return _model
    try:
        from speechbrain.inference.speaker import EncoderClassifier
        _model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="pretrained_models/ecapa",
            run_opts={"device": "cpu"},
        )
        print("[SpeakerVerify] ECAPA-TDNN 모델 로드 완료")
    except Exception as e:
        print(f"[SpeakerVerify] 모델 로드 실패 (화자인식 비활성): {e}")
        _model = None
    return _model


def _extract_sync(audio_bytes: bytes, sample_rate: int = 24000) -> list | None:
    import torch
    model = _load_model_sync()
    if model is None:
        return None

    audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    waveform = torch.tensor(audio).unsqueeze(0)

    if sample_rate != 16000:
        import torchaudio
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)

    with torch.no_grad():
        emb = model.encode_batch(waveform)
    return emb.squeeze().tolist()


async def preload_model():
    """서버 시작 시 백그라운드에서 모델 미리 로드."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(_executor, _load_model_sync)


async def extract_embedding(audio_bytes: bytes, sample_rate: int = 24000) -> list | None:
    """비동기 임베딩 추출 (스레드풀에서 실행)."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, _extract_sync, audio_bytes, sample_rate)


def cosine_sim(a: list, b: list) -> float:
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 1e-8 else 0.0


def find_user(embedding: list, users: list, threshold: float = 0.55) -> dict | None:
    """등록된 사용자 중 코사인 유사도 임계값 이상인 사람 반환."""
    best, best_score = None, threshold
    for u in users:
        stored = u.get("embedding")
        if not stored:
            continue
        score = cosine_sim(embedding, stored)
        if score > best_score:
            best_score = score
            best = u
    return best
