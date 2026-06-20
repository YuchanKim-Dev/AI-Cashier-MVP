"""
키오스크 프론트엔드 — FastAPI 서버.

전체 흐름에서 이 모듈의 위치:
  orchestrator → [이 모듈] → 브라우저 화면

브라우저가 http://localhost:8000을 열면 키오스크 화면을 보여준다.
SSE(Server-Sent Events)로 상태 변경을 실시간으로 브라우저에 밀어넣는다.
orchestrator가 update_state()를 호출하면 SSE 이벤트가 전송된다.

1단계에서 보여주는 정보:
  - 마이크 상태 (연결/캡처 중)
  - 대화 상태 (대기/듣는 중/처리 중/말하는 중)
  - AI 발화 텍스트 (실시간)
  - 화자인증 상태 (3단계까지는 미구현 표시)
"""

import asyncio
import json
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse

app = FastAPI(title="Voice AI Cashier")

# 현재 상태를 전역으로 관리 (1단계는 단일 세션이므로 단순하게)
_state = {
    "mic": "disconnected",       # disconnected | active
    "conversation": "idle",      # idle | listening | processing | speaking
    "ai_text": "",               # AI가 현재 말하는 텍스트 (누적)
    "speaker_verified": None,    # None | True | False (3단계에서 사용)
}

# SSE 구독자 큐 목록 — 새 이벤트가 생기면 모두에게 전송
_sse_queues: list[asyncio.Queue] = []


def update_state(updates: dict):
    """
    orchestrator에서 호출. 상태를 업데이트하고 SSE로 브라우저에 전파.
    asyncio 이벤트 루프에서 호출되어야 한다.
    """
    _state.update(updates)
    data = json.dumps(_state)
    for q in _sse_queues:
        try:
            q.put_nowait(data)
        except asyncio.QueueFull:
            pass


def append_ai_text(delta: str):
    """AI 발화 텍스트 누적 업데이트."""
    _state["ai_text"] += delta
    update_state({})


def clear_ai_text():
    """새 응답 시작 시 텍스트 초기화."""
    update_state({"ai_text": ""})


@app.get("/", response_class=HTMLResponse)
async def index():
    """키오스크 메인 화면."""
    return HTMLResponse(content=_get_html())


@app.get("/events")
async def sse_events():
    """SSE 엔드포인트 — 브라우저가 연결하면 상태 변경을 실시간으로 받는다."""
    q: asyncio.Queue = asyncio.Queue(maxsize=50)
    _sse_queues.append(q)

    async def generate() -> AsyncGenerator[str, None]:
        # 연결 즉시 현재 상태를 한 번 전송
        yield f"data: {json.dumps(_state)}\n\n"
        try:
            while True:
                data = await q.get()
                yield f"data: {data}\n\n"
        finally:
            _sse_queues.remove(q)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def _get_html() -> str:
    return """<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>음성 AI 캐셔</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Apple SD Gothic Neo', 'Noto Sans KR', sans-serif;
      background: #0f172a;
      color: #f1f5f9;
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      padding: 24px;
    }
    h1 { font-size: 2rem; font-weight: 700; margin-bottom: 32px; letter-spacing: -0.5px; }

    .status-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      width: 100%;
      max-width: 640px;
      margin-bottom: 24px;
    }
    .status-card {
      background: #1e293b;
      border-radius: 12px;
      padding: 20px;
      border: 1px solid #334155;
    }
    .status-label {
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 1px;
      color: #94a3b8;
      margin-bottom: 8px;
    }
    .status-value {
      font-size: 1.1rem;
      font-weight: 600;
    }
    .status-dot {
      display: inline-block;
      width: 10px; height: 10px;
      border-radius: 50%;
      margin-right: 8px;
    }

    /* 상태별 색상 */
    .dot-active { background: #22c55e; box-shadow: 0 0 8px #22c55e; }
    .dot-listening { background: #3b82f6; box-shadow: 0 0 8px #3b82f6; animation: pulse 1s infinite; }
    .dot-processing { background: #f59e0b; box-shadow: 0 0 8px #f59e0b; }
    .dot-speaking { background: #8b5cf6; box-shadow: 0 0 8px #8b5cf6; animation: pulse 0.7s infinite; }
    .dot-idle { background: #64748b; }
    .dot-disconnected { background: #ef4444; }
    .dot-pending { background: #64748b; }

    @keyframes pulse {
      0%, 100% { opacity: 1; }
      50% { opacity: 0.4; }
    }

    /* AI 발화 텍스트 박스 */
    .ai-text-box {
      width: 100%;
      max-width: 640px;
      background: #1e293b;
      border: 1px solid #334155;
      border-radius: 12px;
      padding: 24px;
      min-height: 120px;
      font-size: 1.2rem;
      line-height: 1.7;
      color: #e2e8f0;
    }
    .ai-text-label {
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 1px;
      color: #94a3b8;
      margin-bottom: 12px;
    }
    .ai-text-content { color: #a78bfa; }

    /* 안내 메시지 */
    .hint {
      margin-top: 20px;
      font-size: 0.85rem;
      color: #64748b;
      text-align: center;
    }
  </style>
</head>
<body>
  <h1>음성 AI 캐셔</h1>

  <div class="status-grid">
    <div class="status-card">
      <div class="status-label">마이크</div>
      <div class="status-value" id="mic-status">
        <span class="status-dot dot-disconnected" id="mic-dot"></span>
        <span id="mic-text">연결 중...</span>
      </div>
    </div>
    <div class="status-card">
      <div class="status-label">대화 상태</div>
      <div class="status-value" id="conv-status">
        <span class="status-dot dot-idle" id="conv-dot"></span>
        <span id="conv-text">대기 중</span>
      </div>
    </div>
    <div class="status-card">
      <div class="status-label">화자인증</div>
      <div class="status-value">
        <span class="status-dot dot-pending" id="spk-dot"></span>
        <span id="spk-text">3단계에서 구현</span>
      </div>
    </div>
    <div class="status-card">
      <div class="status-label">연결</div>
      <div class="status-value">
        <span class="status-dot dot-idle" id="conn-dot"></span>
        <span id="conn-text">SSE 연결 중</span>
      </div>
    </div>
  </div>

  <div class="ai-text-box">
    <div class="ai-text-label">AI 응답</div>
    <div class="ai-text-content" id="ai-text">말씀하세요...</div>
  </div>

  <div class="hint">마이크에 대고 말하면 AI가 응답합니다.</div>

  <script>
    const micDot = document.getElementById('mic-dot');
    const micText = document.getElementById('mic-text');
    const convDot = document.getElementById('conv-dot');
    const convText = document.getElementById('conv-text');
    const connDot = document.getElementById('conn-dot');
    const connText = document.getElementById('conn-text');
    const aiText = document.getElementById('ai-text');

    const MIC_LABELS = { active: '캡처 중', disconnected: '연결 안 됨' };
    const CONV_LABELS = {
      idle: '대기 중', listening: '듣는 중', processing: '처리 중', speaking: '말하는 중'
    };

    function applyState(state) {
      // 마이크
      micDot.className = 'status-dot dot-' + state.mic;
      micText.textContent = MIC_LABELS[state.mic] || state.mic;

      // 대화
      convDot.className = 'status-dot dot-' + state.conversation;
      convText.textContent = CONV_LABELS[state.conversation] || state.conversation;

      // AI 텍스트
      if (state.ai_text) {
        aiText.textContent = state.ai_text;
      }
    }

    // SSE 연결
    const es = new EventSource('/events');
    es.onopen = () => {
      connDot.className = 'status-dot dot-active';
      connText.textContent = 'SSE 연결됨';
    };
    es.onmessage = (e) => {
      applyState(JSON.parse(e.data));
    };
    es.onerror = () => {
      connDot.className = 'status-dot dot-disconnected';
      connText.textContent = 'SSE 끊김 — 재연결 중';
    };
  </script>
</body>
</html>"""
