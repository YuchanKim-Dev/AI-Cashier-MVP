"""
키오스크 프론트엔드 — FastAPI 서버 (2단계 전체 UI).

화면(screen) 종류:
  waiting             대기 화면
  ordering            주문 중 (장바구니 + AI 대화)
  checkout            결제 화면 (주문 요약 + 결제 방법 선택)
  payment_processing  결제 처리 중
  voice_save_prompt   목소리 저장 질문 (신규 + 3초 이상)
  register            이름 + 전화번호 입력
  complete            주문 완료
  locked              화자인증 실패 잠금

SSE로 상태를 실시간 전달. 버튼 액션은 POST /action으로 수신.
orchestrator가 set_context()로 액션 큐와 이벤트 루프를 주입한다.
"""

import asyncio
import json
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse

app = FastAPI(title="Voice AI Cashier Kiosk")

# orchestrator가 주입하는 값들
_state: dict = {
    "screen": "waiting",
    "conversation": "idle",
    "mic": "active",
    "ai_text": "",
    "user_text": "",
    "conversation_log": [],
    "user_name": None,
    "is_new_user": True,
    "speaker_verified": None,
    "failed_verifications": 0,
    "cart_items": [],
    "cart_total": 0,
    "payment_method": None,
    "transaction_id": None,
    "voice_duration": 0.0,
}
_sse_queues: list[asyncio.Queue] = []
_action_queue: Optional[asyncio.Queue] = None
_main_loop: Optional[asyncio.AbstractEventLoop] = None


def set_context(action_queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
    """orchestrator에서 호출. 액션 큐와 이벤트 루프를 주입."""
    global _action_queue, _main_loop
    _action_queue = action_queue
    _main_loop = loop


def push_state(updates: dict):
    """
    상태 업데이트 + SSE 브로드캐스트.
    asyncio 이벤트 루프에서 호출해야 한다.
    """
    _state.update(updates)
    data = json.dumps(_state, ensure_ascii=False)
    for q in _sse_queues:
        try:
            q.put_nowait(data)
        except asyncio.QueueFull:
            pass


def _enqueue_action(action: dict):
    """FastAPI 핸들러(별도 스레드)에서 orchestrator 이벤트 루프로 액션 전달."""
    if _main_loop and _action_queue:
        _main_loop.call_soon_threadsafe(_action_queue.put_nowait, action)


# ─── HTTP 엔드포인트 ────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(content=_build_html())


@app.get("/events")
async def sse():
    """SSE 스트림 — 브라우저가 연결해 상태를 실시간으로 받는다."""
    q: asyncio.Queue = asyncio.Queue(maxsize=50)
    _sse_queues.append(q)

    async def generate() -> AsyncGenerator[str, None]:
        yield f"data: {json.dumps(_state, ensure_ascii=False)}\n\n"
        try:
            while True:
                data = await q.get()
                yield f"data: {data}\n\n"
        finally:
            if q in _sse_queues:
                _sse_queues.remove(q)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/action/checkout")
async def action_checkout():
    """주문하기 버튼 — 결제 화면으로 전환."""
    _enqueue_action({"type": "checkout"})
    return {"ok": True}


@app.post("/action/payment")
async def action_payment(request: Request):
    """결제 방법 선택 버튼."""
    body = await request.json()
    _enqueue_action({"type": "payment", "method": body.get("method", "physical_card")})
    return {"ok": True}


@app.post("/action/save_voice")
async def action_save_voice(request: Request):
    """목소리 저장 Y/N 버튼."""
    body = await request.json()
    _enqueue_action({"type": "save_voice", "save": body.get("save", False)})
    return {"ok": True}


@app.post("/action/register")
async def action_register(request: Request):
    """이름 + 전화번호 등록 폼 제출."""
    body = await request.json()
    _enqueue_action({
        "type": "register",
        "name": body.get("name", "").strip(),
        "phone": body.get("phone", "").strip(),
    })
    return {"ok": True}


@app.post("/action/retry_verification")
async def action_retry():
    """화자인증 실패 후 재시도 버튼."""
    _enqueue_action({"type": "retry_verification"})
    return {"ok": True}


@app.post("/action/add_menu")
async def action_add_menu(request: Request):
    """화면 메뉴 카드 클릭으로 장바구니 추가."""
    body = await request.json()
    _enqueue_action({"type": "add_menu", "name": body.get("name", "")})
    return {"ok": True}


@app.post("/action/start")
async def action_start():
    """시작하기 버튼 — 대기 화면 → 주문 화면."""
    _enqueue_action({"type": "start"})
    return {"ok": True}


@app.post("/action/reset")
async def action_reset():
    """처음으로 — 세션 전체 초기화."""
    _enqueue_action({"type": "reset"})
    return {"ok": True}


@app.post("/action/app_payment_confirm")
async def action_app_payment_confirm(request: Request):
    """앱카드 결제수단 선택 후 결제 실행."""
    body = await request.json()
    _enqueue_action({"type": "app_payment_confirm", "method": body.get("method", "신용카드")})
    return {"ok": True}


@app.get("/app", response_class=HTMLResponse)
async def app_demo():
    """앱 등록 가상 화면 — 별도 탭으로 열림."""
    return HTMLResponse(content=_build_app_html())


# ─── HTML ──────────────────────────────────────────────────────────────────────

def _build_html() -> str:
    return r"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>음성 AI 캐셔</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:    #f7f3ee;   /* 따뜻한 크림 */
    --bg2:   #ffffff;   /* 흰 카드 */
    --bg3:   #e8ddd3;   /* 따뜻한 경계선 */
    --accent:  #d4531a; /* 매장 오렌지-레드 */
    --accent2: #f59e0b; /* 앰버 */
    --green: #16a34a;
    --yellow: #d97706;
    --red:   #dc2626;
    --blue:  #2563eb;
    --text:  #1c1917;   /* 따뜻한 거의-검정 */
    --muted: #78716c;   /* 따뜻한 회색 */
    --radius: 16px;
  }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'Apple SD Gothic Neo', 'Noto Sans KR', sans-serif;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
  }

  /* ── 상단 상태바 ── */
  #statusbar {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 12px 24px;
    background: var(--accent);   /* 포인트 오렌지-레드 헤더 */
    border-bottom: none;
    flex-shrink: 0;
    box-shadow: 0 2px 8px rgba(212,83,26,.25);
  }
  #statusbar .logo {
    font-size: 1.15rem;
    font-weight: 800;
    color: #fff;
    margin-right: auto;
    letter-spacing: -0.5px;
  }
  .chip {
    display: flex;
    align-items: center;
    gap: 6px;
    background: rgba(255,255,255,.18);
    border: 1px solid rgba(255,255,255,.35);
    border-radius: 999px;
    padding: 5px 12px;
    font-size: 0.78rem;
    color: #fff;
  }
  .dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
  }
  .dot-active    { background: #86efac; }
  .dot-listening { background: #93c5fd; animation: blink 1s infinite; }
  .dot-processing{ background: #fde68a; }
  .dot-speaking  { background: #fff;   animation: blink .7s infinite; }
  .dot-idle      { background: rgba(255,255,255,.4); }
  .dot-ok        { background: #86efac; }
  .dot-fail      { background: #fca5a5; }
  .dot-pending   { background: rgba(255,255,255,.3); }

  @keyframes blink { 0%,100%{opacity:1} 50%{opacity:.3} }

  /* ── 메인 레이아웃 ── */
  #main {
    display: flex;
    flex: 1;
    overflow: hidden;
  }

  /* ── 왼쪽: 콘텐츠 영역 ── */
  #content {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow-y: auto;
    padding: 32px;
  }

  /* ── 오른쪽: 장바구니 패널 ── */
  #cart-panel {
    width: 300px;
    background: var(--bg2);
    border-left: 1px solid var(--bg3);
    display: flex;
    flex-direction: column;
    flex-shrink: 0;
  }
  #cart-panel.hidden { display: none; }
  #cart-header {
    padding: 20px;
    border-bottom: 1px solid var(--bg3);
    font-weight: 700;
    font-size: 1rem;
  }
  #cart-items {
    flex: 1;
    overflow-y: auto;
    padding: 12px;
  }
  .cart-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 10px 8px;
    border-bottom: 1px solid var(--bg3);
    font-size: 0.9rem;
  }
  .cart-item-name { font-weight: 500; }
  .cart-item-qty  { color: var(--muted); font-size: 0.8rem; margin-top: 2px; }
  .cart-item-price{ color: var(--accent); font-weight: 700; }
  #cart-footer {
    padding: 16px 20px;
    border-top: 1px solid var(--bg3);
  }
  #cart-total {
    display: flex;
    justify-content: space-between;
    font-size: 1.1rem;
    font-weight: 700;
    margin-bottom: 12px;
  }
  .cart-empty {
    text-align: center;
    color: var(--muted);
    font-size: 0.85rem;
    padding: 40px 0;
  }

  /* ── 화면별 스타일 ── */
  .screen { display: none; flex-direction: column; align-items: center; justify-content: center; flex: 1; }
  .screen.active { display: flex; }

  /* 대기 화면 */
  #screen-waiting { text-align: center; }
  .waiting-icon { font-size: 5rem; margin-bottom: 24px; animation: float 3s ease-in-out infinite; }
  @keyframes float { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-12px)} }
  .waiting-title { font-size: 2.4rem; font-weight: 800; margin-bottom: 12px; }
  .waiting-hint  { color: var(--muted); font-size: 1rem; }

  /* 주문 화면 */
  #screen-ordering { width: 100%; align-items: flex-start; justify-content: flex-start; gap: 16px; }

  /* ── 대화 패널 ── */
  #conv-panel {
    width: 100%;
    background: var(--bg2);
    border: 1px solid var(--bg3);
    border-radius: var(--radius);
    display: flex;
    flex-direction: column;
    overflow: hidden;
    flex-shrink: 0;
  }
  #conv-header {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 12px 16px;
    border-bottom: 1px solid var(--bg3);
    font-size: 0.8rem;
    color: var(--muted);
    background: var(--bg);
  }
  .conv-avatar {
    width: 30px; height: 30px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem; flex-shrink: 0;
  }
  .conv-avatar.ai   { background: var(--accent); }
  .conv-avatar.user { background: #92400e; }
  #conv-header .ai-name { font-weight: 600; color: var(--accent); }

  /* 메시지 목록 */
  #chat-log {
    overflow-y: auto;
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 14px;
    max-height: 280px;
    scroll-behavior: smooth;
  }
  .msg-row {
    display: flex;
    gap: 10px;
    align-items: flex-end;
  }
  .msg-row.user { flex-direction: row-reverse; }

  .msg-av {
    width: 34px; height: 34px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem; flex-shrink: 0; align-self: flex-start;
  }
  .msg-av.ai   { background: var(--accent); }
  .msg-av.user { background: #92400e; }

  .msg-body { display: flex; flex-direction: column; max-width: 76%; }
  .msg-row.user .msg-body { align-items: flex-end; }
  .msg-name { font-size: 0.68rem; color: var(--muted); margin-bottom: 4px; }
  .msg-bubble {
    padding: 11px 15px;
    border-radius: 18px;
    font-size: 0.95rem;
    line-height: 1.6;
    word-break: break-word;
  }
  .msg-bubble.ai   { background: #fff3ee; color: #7c2d12; border-bottom-left-radius: 4px; border-left: 3px solid var(--accent); }
  .msg-bubble.user { background: #fef3c7; color: #78350f; border-bottom-right-radius: 4px; border-right: 3px solid var(--accent2); }

  /* 하단 입력/스트리밍 영역 */
  #conv-footer {
    border-top: 1px solid var(--bg3);
    padding: 12px 16px;
    min-height: 56px;
    display: flex;
    align-items: center;
    gap: 12px;
  }
  #conv-status-icon {
    width: 32px; height: 32px; border-radius: 50%; background: var(--accent);
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem; flex-shrink: 0;
  }
  #conv-stream-text {
    flex: 1; font-size: 0.95rem; color: var(--accent); line-height: 1.5;
  }
  /* 타이핑 점 */
  .typing-dots { display: flex; gap: 5px; align-items: center; }
  .typing-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--muted); animation: tdot 1.2s infinite;
  }
  .typing-dot:nth-child(2) { animation-delay: .2s; }
  .typing-dot:nth-child(3) { animation-delay: .4s; }
  @keyframes tdot { 0%,60%,100%{transform:translateY(0)} 30%{transform:translateY(-7px)} }
  /* 마이크 파동 */
  .mic-wave {
    display: flex; align-items: center; gap: 3px;
  }
  .mic-bar {
    width: 3px; background: var(--blue); border-radius: 2px;
    animation: mbar 0.8s infinite ease-in-out;
  }
  .mic-bar:nth-child(1){height:8px;  animation-delay:0s}
  .mic-bar:nth-child(2){height:16px; animation-delay:.15s}
  .mic-bar:nth-child(3){height:10px; animation-delay:.3s}
  .mic-bar:nth-child(4){height:20px; animation-delay:.45s}
  .mic-bar:nth-child(5){height:12px; animation-delay:.6s}
  @keyframes mbar { 0%,100%{transform:scaleY(0.4)} 50%{transform:scaleY(1)} }

  /* 메뉴 그리드 */
  #menu-section { width: 100%; }
  .menu-tabs { display: flex; gap: 8px; margin-bottom: 16px; flex-wrap: wrap; }
  .menu-tab {
    padding: 8px 18px;
    border-radius: 999px;
    border: 1px solid var(--bg3);
    background: var(--bg2);
    color: var(--muted);
    font-size: 0.85rem;
    cursor: pointer;
    transition: all .2s;
  }
  .menu-tab.active, .menu-tab:hover { background: var(--accent); color: #fff; border-color: var(--accent); }
  .menu-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
    gap: 10px;
  }
  .menu-card {
    background: var(--bg2);
    border: 1px solid var(--bg3);
    border-radius: 12px;
    padding: 14px;
    cursor: pointer;
    transition: all .15s;
  }
  .menu-card:hover { border-color: var(--accent); transform: translateY(-2px); }
  .menu-card-name  { font-weight: 600; font-size: 0.9rem; margin-bottom: 4px; }
  .menu-card-price { color: var(--accent); font-size: 0.85rem; }

  /* 결제 화면 */
  #screen-checkout { width: 100%; align-items: flex-start; justify-content: flex-start; flex-direction: row !important; gap: 24px; }
  #checkout-left {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: 0;
    background: var(--bg2);
    border: 1px solid var(--bg3);
    border-radius: var(--radius);
    overflow: hidden;
    min-width: 0;
  }
  #checkout-left-header {
    padding: 12px 16px;
    border-bottom: 1px solid var(--bg3);
    font-size: 0.78rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1px;
    background: var(--bg);
  }
  #checkout-convo {
    overflow-y: auto;
    max-height: 420px;
    padding: 14px;
    display: flex;
    flex-direction: column;
    gap: 12px;
    scroll-behavior: smooth;
  }
  #checkout-right {
    width: 300px;
    flex-shrink: 0;
    display: flex;
    flex-direction: column;
    gap: 16px;
  }
  .checkout-title { font-size: 1.4rem; font-weight: 800; }
  #checkout-summary {
    background: var(--bg2);
    border: 1px solid var(--bg3);
    border-radius: var(--radius);
    padding: 18px;
    text-align: left;
  }
  .summary-item {
    display: flex;
    justify-content: space-between;
    padding: 8px 0;
    font-size: 0.9rem;
    border-bottom: 1px solid var(--bg3);
  }
  .summary-item:last-child { border: none; }
  .summary-total {
    display: flex;
    justify-content: space-between;
    font-size: 1.1rem;
    font-weight: 700;
    padding-top: 10px;
    margin-top: 4px;
  }
  .pay-buttons { display: flex; flex-direction: column; gap: 10px; }

  /* 처리 중 화면 */
  #screen-payment_processing { text-align: center; }
  .spinner {
    width: 60px; height: 60px;
    border: 4px solid var(--bg3);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin 1s linear infinite;
    margin: 0 auto 24px;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* 목소리 저장 화면 */
  #screen-voice_save_prompt { text-align: center; max-width: 420px; margin: 0 auto; }
  .voice-icon { font-size: 4rem; margin-bottom: 20px; }

  /* 등록 화면 */
  #screen-register { max-width: 420px; margin: 0 auto; width: 100%; }
  .register-title { font-size: 1.4rem; font-weight: 700; margin-bottom: 8px; }
  .register-hint  { color: var(--muted); font-size: 0.9rem; margin-bottom: 28px; }
  .input-group { margin-bottom: 16px; }
  .input-label { font-size: 0.8rem; color: var(--muted); margin-bottom: 6px; display: block; }
  .kiosk-input {
    width: 100%;
    background: var(--bg2);
    border: 1px solid var(--bg3);
    border-radius: 10px;
    padding: 14px 16px;
    color: var(--text);
    font-size: 1rem;
    outline: none;
    transition: border-color .2s;
  }
  .kiosk-input:focus { border-color: var(--accent); }

  /* 완료 화면 */
  #screen-complete { text-align: center; }
  .complete-icon { font-size: 5rem; margin-bottom: 20px; }
  .complete-title { font-size: 2rem; font-weight: 800; margin-bottom: 12px; color: var(--green); }

  /* 잠금 화면 */
  #screen-locked { text-align: center; max-width: 400px; margin: 0 auto; }
  .lock-icon { font-size: 4rem; margin-bottom: 20px; }
  .lock-title { font-size: 1.6rem; font-weight: 700; margin-bottom: 12px; color: var(--red); }

  /* ── 공통 버튼 ── */
  .btn {
    padding: 14px 28px;
    border-radius: 12px;
    border: none;
    font-size: 1rem;
    font-weight: 600;
    cursor: pointer;
    transition: all .15s;
    width: 100%;
  }
  .btn:hover { opacity: .88; transform: translateY(-1px); }
  .btn:active { transform: translateY(0); }
  .btn-primary  { background: var(--accent);  color: #fff; }
  .btn-success  { background: var(--green);   color: #fff; }
  .btn-danger   { background: var(--red);     color: #fff; }
  .btn-outline  { background: transparent; color: var(--text); border: 1px solid var(--bg3); }
  .btn-yellow   { background: var(--yellow);  color: #000; }
  #screen-card_insert, #screen-app_payment { text-align: center; max-width: 420px; margin: 0 auto; }

  .section-title {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--muted);
    margin-bottom: 14px;
  }
  .big-title { font-size: 1.6rem; font-weight: 800; margin-bottom: 10px; }
  .sub-text  { color: var(--muted); font-size: 0.95rem; margin-bottom: 28px; line-height: 1.6; }
</style>
</head>
<body>

<!-- 상단 상태바 -->
<div id="statusbar">
  <div class="logo">AI Cashier</div>
  <div class="chip">
    <div class="dot dot-active" id="mic-dot"></div>
    <span id="mic-text">마이크</span>
  </div>
  <div class="chip">
    <div class="dot dot-idle" id="conv-dot"></div>
    <span id="conv-text">대기 중</span>
  </div>
  <div class="chip">
    <div class="dot dot-pending" id="spk-dot"></div>
    <span id="spk-text">화자인증</span>
  </div>
  <button id="tts-btn" class="chip" style="cursor:pointer;background:rgba(255,255,255,.9);border:1px solid rgba(255,255,255,.5);color:var(--accent);font-size:.78rem;font-weight:600;" onclick="toggleTTS()">🔊 음성 ON</button>
  <button class="chip" style="cursor:pointer;background:rgba(255,255,255,.15);border:1px solid rgba(255,255,255,.4);color:#fff;font-size:.78rem;" onclick="resetKiosk()">↩ 처음으로</button>
</div>

<!-- 메인 -->
<div id="main">
  <!-- 왼쪽: 화면 콘텐츠 -->
  <div id="content">

    <!-- 대기 화면 -->
    <div class="screen active" id="screen-waiting">
      <div class="waiting-icon">🎤</div>
      <div class="waiting-title">안녕하세요!</div>
      <div class="waiting-hint">말씀하시거나 아래 버튼을 눌러 시작하세요</div>
      <div style="display:flex;gap:12px;margin-top:32px;flex-wrap:wrap;justify-content:center;">
        <button class="btn btn-primary" style="width:200px;" onclick="startOrder()">시작하기</button>
        <button class="btn btn-outline" style="width:200px;" onclick="window.open('/app','_blank')">📱 앱 등록 미리보기</button>
      </div>
    </div>

    <!-- 주문 화면 -->
    <div class="screen" id="screen-ordering">
      <!-- 대화 패널 -->
      <div id="conv-panel">
        <div id="conv-header">
          <div class="conv-avatar ai">🤖</div>
          <span class="ai-name">AI 캐셔</span>
          <span style="margin-left:auto;font-size:0.75rem;" id="conv-header-status">대기 중</span>
        </div>
        <div id="chat-log"></div>
        <div id="conv-footer">
          <div id="conv-status-icon">🤖</div>
          <div id="conv-stream-text">
            <span style="color:var(--muted);font-size:.88rem;">말씀하세요...</span>
          </div>
        </div>
      </div>
      <!-- 메뉴 -->
      <div id="menu-section">
        <div class="menu-tabs">
          <div class="menu-tab active" onclick="showCategory('버거')">버거</div>
          <div class="menu-tab" onclick="showCategory('사이드')">사이드</div>
          <div class="menu-tab" onclick="showCategory('음료')">음료</div>
          <div class="menu-tab" onclick="showCategory('세트')">세트</div>
        </div>
        <div class="menu-grid" id="menu-grid"></div>
      </div>
    </div>

    <!-- 결제 화면 -->
    <div class="screen" id="screen-checkout">
      <!-- 왼쪽: 대화 내용 -->
      <div id="checkout-left">
        <div id="checkout-left-header">대화 내용</div>
        <div id="checkout-convo"></div>
      </div>
      <!-- 오른쪽: 주문 내역 + 결제 -->
      <div id="checkout-right">
        <div class="checkout-title">주문 확인</div>
        <div id="checkout-summary"></div>
        <div class="pay-buttons">
          <button class="btn btn-primary" onclick="selectPayment('app_card')">📱 앱 카드 결제</button>
          <button class="btn btn-outline" onclick="selectPayment('physical_card')">💳 현장 카드 결제</button>
        </div>
      </div>
    </div>

    <!-- 결제 처리 중 -->
    <div class="screen" id="screen-payment_processing">
      <div class="spinner"></div>
      <div class="big-title">결제 처리 중...</div>
      <div class="sub-text">잠시만 기다려주세요</div>
    </div>

    <!-- 목소리 저장 질문 -->
    <div class="screen" id="screen-voice_save_prompt">
      <div class="voice-icon">🎙️</div>
      <div class="big-title">목소리를 저장할까요?</div>
      <div class="sub-text">다음 방문 시 바로 주문할 수 있어요.<br>저장하지 않아도 오늘 주문은 완료됩니다.</div>
      <div style="display:flex;gap:12px;width:100%;max-width:320px;">
        <button class="btn btn-success" onclick="saveVoice(true)">예, 저장할게요</button>
        <button class="btn btn-outline" onclick="saveVoice(false)">아니요</button>
      </div>
    </div>

    <!-- 이름 + 전화번호 등록 -->
    <div class="screen" id="screen-register">
      <div class="register-title">간단히 등록해주세요</div>
      <div class="register-hint">이름과 전화번호만 입력하면 끝납니다.<br>다음 방문부터 목소리만으로 주문하세요!</div>
      <div class="input-group" style="width:100%">
        <label class="input-label">이름</label>
        <input class="kiosk-input" id="reg-name" type="text" placeholder="홍길동" autocomplete="off">
      </div>
      <div class="input-group" style="width:100%">
        <label class="input-label">전화번호</label>
        <input class="kiosk-input" id="reg-phone" type="tel" placeholder="01012345678" autocomplete="off">
      </div>
      <div id="reg-error" style="color:var(--red);font-size:.85rem;margin-bottom:12px;display:none;width:100%;"></div>
      <button class="btn btn-primary" onclick="submitRegister()" style="max-width:320px;">등록 완료</button>
    </div>

    <!-- 카드 삽입 대기 -->
    <div class="screen" id="screen-card_insert">
      <div style="font-size:4rem;margin-bottom:20px;animation:float 2s ease-in-out infinite">💳</div>
      <div class="big-title">카드를 꽂아주세요</div>
      <div class="sub-text">단말기에 카드를 삽입하면 자동으로 결제됩니다.<br><span style="color:var(--muted);font-size:.85rem">잠시 후 자동 결제됩니다...</span></div>
      <div class="spinner" style="margin-top:12px"></div>
    </div>

    <!-- 앱카드 결제수단 선택 -->
    <div class="screen" id="screen-app_payment">
      <div style="font-size:3rem;margin-bottom:16px">📱</div>
      <div class="big-title" style="margin-bottom:8px">결제 수단 선택</div>
      <div class="sub-text">앱에 등록된 결제 수단을 선택하세요</div>
      <div style="display:flex;flex-direction:column;gap:12px;width:100%;max-width:360px">
        <button class="btn btn-primary" onclick="confirmAppPayment('신용카드')">
          💳 신용카드
        </button>
        <button class="btn btn-outline" onclick="confirmAppPayment('체크카드')">
          🏦 체크카드
        </button>
        <button class="btn btn-outline" onclick="confirmAppPayment('계좌이체')">
          📤 계좌이체
        </button>
      </div>
      <button class="btn btn-outline" style="max-width:200px;margin-top:20px;font-size:.85rem;padding:10px" onclick="post('/action/checkout',{})">← 결제수단 변경</button>
    </div>

    <!-- 주문 완료 -->
    <div class="screen" id="screen-complete">
      <div class="complete-icon">✅</div>
      <div class="complete-title" id="complete-title">주문 완료!</div>
      <div class="sub-text" id="complete-sub">음식이 준비되면 안내드립니다.</div>
    </div>

    <!-- 화자인증 실패 잠금 -->
    <div class="screen" id="screen-locked">
      <div class="lock-icon">🔒</div>
      <div class="lock-title">본인 확인이 필요합니다</div>
      <div class="sub-text">처음 말씀하신 분이 맞으신가요?<br>다시 말씀해 주시면 확인하겠습니다.</div>
      <button class="btn btn-yellow" onclick="retryVerification()" style="max-width:280px;">다시 말하기</button>
    </div>

  </div><!-- /#content -->

  <!-- 오른쪽: 장바구니 -->
  <div id="cart-panel" class="hidden">
    <div id="cart-header">🛒 장바구니</div>
    <div id="cart-items">
      <div class="cart-empty">아직 담긴 메뉴가 없어요</div>
    </div>
    <div id="cart-footer">
      <div id="cart-total">
        <span>합계</span>
        <span id="cart-total-price">0원</span>
      </div>
      <button class="btn btn-primary" onclick="checkout()">주문하기</button>
    </div>
  </div>

</div><!-- /#main -->

<script>
// ── 메뉴 데이터 (서버에서 주입 가능, 지금은 클라이언트에 직접) ──
const MENU = {
  "버거":  [
    {name:"치즈버거",   price:6500},
    {name:"더블버거",   price:8500},
    {name:"베이컨버거", price:7500},
    {name:"새우버거",   price:7000},
    {name:"불고기버거", price:7000},
  ],
  "사이드": [
    {name:"감자튀김",   price:2500},
    {name:"양파링",     price:3000},
    {name:"치킨텐더",   price:4500},
    {name:"코울슬로",   price:2000},
  ],
  "음료": [
    {name:"콜라",       price:2000},
    {name:"사이다",     price:2000},
    {name:"아이스티",   price:2500},
    {name:"오렌지주스", price:3000},
    {name:"물",         price:1000},
  ],
  "세트": [
    {name:"치즈버거 세트",   price:9500},
    {name:"더블버거 세트",   price:12000},
    {name:"베이컨버거 세트", price:10500},
  ],
};

let currentCategory = "버거";
let currentState    = {};
let ttsEnabled      = true;
let _lastLogLen     = 0;
let _lastAiCount    = 0;

// ── TTS ──
function toggleTTS() {
  ttsEnabled = !ttsEnabled;
  const btn = document.getElementById('tts-btn');
  btn.textContent = ttsEnabled ? '🔊 음성 ON' : '🔇 음성 OFF';
  btn.style.background = ttsEnabled ? 'rgba(255,255,255,.9)' : 'rgba(255,255,255,.2)';
  btn.style.color = ttsEnabled ? 'var(--accent)' : '#fff';
  if (!ttsEnabled) window.speechSynthesis && window.speechSynthesis.cancel();
}

function speakKo(text) {
  if (!ttsEnabled || !text || !window.speechSynthesis) return;
  window.speechSynthesis.cancel();
  const utt = new SpeechSynthesisUtterance(text);
  utt.lang  = 'ko-KR';
  utt.rate  = 1.05;
  utt.pitch = 1.0;
  // 한국어 voice 우선 선택
  const voices = speechSynthesis.getVoices();
  const koVoice = voices.find(v => v.lang.startsWith('ko'));
  if (koVoice) utt.voice = koVoice;
  speechSynthesis.speak(utt);
}

// ── 화면 전환 ──
function showScreen(name) {
  document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
  const el = document.getElementById('screen-' + name);
  if (el) el.classList.add('active');
  document.getElementById('cart-panel').classList.toggle('hidden', !['ordering','checkout','card_insert','app_payment'].includes(name));
}

// ── 상태 적용 ──
function applyState(state) {
  currentState = state;
  showScreen(state.screen);
  updateStatusBar(state);

  if (state.conversation_log !== undefined) renderChatLog(state.conversation_log);
  updateConvFooter(state);

  renderCart(state.cart_items || [], state.cart_total || 0);
  if (state.screen === 'checkout') renderCheckoutSummary(state.cart_items || [], state.cart_total || 0);
  if (state.screen === 'complete') {
    const nm = state.user_name;
    document.getElementById('complete-title').textContent = nm ? `감사합니다, ${nm}님!` : '주문 완료!';
    document.getElementById('complete-sub').textContent =
      state.transaction_id ? `결제 완료 (${state.transaction_id})\n음식이 준비되면 안내드립니다.` : '음식이 준비되면 안내드립니다.';
  }
}

// ── 상태바 ──
function updateStatusBar(state) {
  const LABELS = {idle:'대기 중', listening:'듣는 중', processing:'처리 중', speaking:'말하는 중'};
  const convDot = document.getElementById('conv-dot');
  convDot.className = 'dot dot-' + state.conversation;
  document.getElementById('conv-text').textContent = LABELS[state.conversation] || state.conversation;

  const spkDot = document.getElementById('spk-dot');
  if (state.speaker_verified === true)   { spkDot.className='dot dot-ok';   document.getElementById('spk-text').textContent='인증됨'; }
  else if (state.speaker_verified===false){ spkDot.className='dot dot-fail'; document.getElementById('spk-text').textContent='불일치'; }
  else { spkDot.className='dot dot-pending'; document.getElementById('spk-text').textContent='화자인증'; }
}

// ── 대화 하단 풋터 (스트리밍 / 상태 표시) ──
function updateConvFooter(state) {
  const icon   = document.getElementById('conv-status-icon');
  const stream = document.getElementById('conv-stream-text');
  const header = document.getElementById('conv-header-status');

  if (state.ai_text) {
    // AI 텍스트 스트리밍 중
    icon.textContent = '🤖';
    stream.innerHTML = `<span style="color:var(--accent)">${escHtml(state.ai_text)}<span style="display:inline-block;width:2px;height:1em;background:var(--accent);vertical-align:text-bottom;margin-left:2px;animation:blink .7s infinite"></span></span>`;
    header && (header.textContent = 'AI 응답 중...');
  } else if (state.conversation === 'listening') {
    // 사용자 말하는 중
    icon.textContent = '🎤';
    stream.innerHTML = `<div class="mic-wave"><div class="mic-bar"></div><div class="mic-bar"></div><div class="mic-bar"></div><div class="mic-bar"></div><div class="mic-bar"></div></div><span style="margin-left:8px;color:var(--blue)">듣는 중...</span>`;
    header && (header.textContent = '손님 발화 중');
  } else if (state.conversation === 'processing') {
    icon.textContent = '🤖';
    stream.innerHTML = `<div class="typing-dots"><div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div></div><span style="margin-left:8px;color:var(--muted)">생각 중...</span>`;
    header && (header.textContent = '처리 중...');
  } else {
    icon.textContent = '🤖';
    stream.innerHTML = `<span style="color:var(--muted);font-size:.88rem">말씀하세요...</span>`;
    header && (header.textContent = '대기 중');
  }
}

// ── 대화 로그 렌더링 ──
function renderChatLog(log) {
  // TTS: 새 AI 메시지가 추가되었을 때만 읽기
  const aiMsgs = log.filter(m => m.role === 'ai');
  if (aiMsgs.length > _lastAiCount) {
    speakKo(aiMsgs[aiMsgs.length - 1].text);
    _lastAiCount = aiMsgs.length;
  }

  if (log.length === _lastLogLen) return;
  _lastLogLen = log.length;

  const html = log.map(msg => {
    const isAi = msg.role === 'ai';
    return `
      <div class="msg-row ${msg.role}">
        <div class="msg-av ${msg.role}">${isAi ? '🤖' : '👤'}</div>
        <div class="msg-body">
          <div class="msg-name">${isAi ? 'AI 캐셔' : '손님'}</div>
          <div class="msg-bubble ${msg.role}">${escHtml(msg.text)}</div>
        </div>
      </div>`;
  }).join('');

  // 주문 화면 + 결제 화면 둘 다 업데이트
  const mainLog     = document.getElementById('chat-log');
  const checkoutLog = document.getElementById('checkout-convo');
  if (mainLog)     { mainLog.innerHTML     = html; mainLog.scrollTop     = mainLog.scrollHeight; }
  if (checkoutLog) { checkoutLog.innerHTML = html; checkoutLog.scrollTop = checkoutLog.scrollHeight; }
}

function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// ── 장바구니 렌더링 ──
function renderCart(items, total) {
  const container = document.getElementById('cart-items');
  const totalEl   = document.getElementById('cart-total-price');
  if (!items || items.length === 0) {
    container.innerHTML = '<div class="cart-empty">아직 담긴 메뉴가 없어요</div>';
    totalEl.textContent = '0원';
    return;
  }
  container.innerHTML = items.map(item => `
    <div class="cart-item">
      <div>
        <div class="cart-item-name">${item.name}</div>
        ${item.includes ? `<div class="cart-item-qty" style="color:var(--accent);font-size:.75rem">${item.includes}</div>` : ''}
        <div class="cart-item-qty">× ${item.quantity}</div>
      </div>
      <div class="cart-item-price">${(item.price * item.quantity).toLocaleString()}원</div>
    </div>
  `).join('');
  totalEl.textContent = total.toLocaleString() + '원';
}

// ── 결제 요약 ──
function renderCheckoutSummary(items, total) {
  const el = document.getElementById('checkout-summary');
  if (!items || items.length === 0) { el.innerHTML = ''; return; }
  el.innerHTML = `
    ${items.map(i => `
      <div class="summary-item">
        <span>${i.name} × ${i.quantity}</span>
        <span>${(i.price * i.quantity).toLocaleString()}원</span>
      </div>
    `).join('')}
    <div class="summary-total">
      <span>총 결제금액</span>
      <span style="color:var(--accent)">${total.toLocaleString()}원</span>
    </div>
  `;
}

// ── 메뉴 탭 / 그리드 ──
function showCategory(cat) {
  currentCategory = cat;
  document.querySelectorAll('.menu-tab').forEach(t =>
    t.classList.toggle('active', t.textContent === cat)
  );
  renderMenuGrid(cat);
}

function renderMenuGrid(cat) {
  const grid = document.getElementById('menu-grid');
  const items = MENU[cat] || [];
  grid.innerHTML = items.map(item => `
    <div class="menu-card" onclick="addMenuByClick('${item.name}')">
      <div class="menu-card-name">${item.name}</div>
      <div class="menu-card-price">${item.price.toLocaleString()}원</div>
    </div>
  `).join('');
}

async function addMenuByClick(name) {
  // 화면 클릭으로도 장바구니 추가 가능 (보조 수단)
  await post('/action/add_menu', {name});
}

// ── 액션 요청 ──
async function post(url, data) {
  try {
    await fetch(url, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(data)});
  } catch(e) { console.error(e); }
}

function startOrder()            { post('/action/start', {}); }
function checkout()              { post('/action/checkout', {}); }
function selectPayment(method)   { post('/action/payment', {method}); }
function saveVoice(save)         { post('/action/save_voice', {save}); }
function retryVerification()     { post('/action/retry_verification', {}); }

function resetKiosk() {
  if (!confirm('처음으로 돌아가시겠어요?\n주문 내용이 모두 초기화됩니다.')) return;
  post('/action/reset', {});
}

function confirmAppPayment(method) {
  post('/action/app_payment_confirm', {method});
}

function submitRegister() {
  const name  = document.getElementById('reg-name').value.trim();
  const phone = document.getElementById('reg-phone').value.trim();
  const errEl = document.getElementById('reg-error');
  if (!name) { errEl.textContent='이름을 입력해주세요.'; errEl.style.display='block'; return; }
  if (!/^01[0-9]{8,9}$/.test(phone)) {
    errEl.textContent='올바른 전화번호를 입력해주세요 (예: 01012345678)';
    errEl.style.display='block'; return;
  }
  errEl.style.display='none';
  post('/action/register', {name, phone});
}

// ── SSE 연결 ──
const es = new EventSource('/events');
es.onmessage = e => applyState(JSON.parse(e.data));

// TTS 음성 목록 로드 (일부 브라우저는 비동기)
if (window.speechSynthesis) {
  speechSynthesis.onvoiceschanged = () => speechSynthesis.getVoices();
  speechSynthesis.getVoices();
}

// ── 초기 메뉴 렌더링 ──
renderMenuGrid('버거');
</script>
</body>
</html>"""


def _build_app_html() -> str:
    """앱 등록 가상 화면 — /app 에서 접근."""
    return r"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Cashier 앱 — 등록</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --bg: #f8fafc;
    --card: #ffffff;
    --accent: #6366f1;
    --accent2: #8b5cf6;
    --green: #22c55e;
    --red: #ef4444;
    --text: #0f172a;
    --muted: #64748b;
    --border: #e2e8f0;
    --radius: 20px;
  }
  body {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    font-family: 'Apple SD Gothic Neo', 'Noto Sans KR', sans-serif;
    padding: 24px;
  }

  /* 폰 프레임 */
  .phone {
    background: var(--card);
    border-radius: 40px;
    width: 390px;
    min-height: 760px;
    box-shadow: 0 40px 80px rgba(0,0,0,0.35);
    overflow: hidden;
    display: flex;
    flex-direction: column;
    position: relative;
  }
  .phone-bar {
    height: 50px;
    background: var(--card);
    display: flex;
    align-items: center;
    justify-content: center;
    border-bottom: 1px solid var(--border);
    flex-shrink: 0;
    position: relative;
  }
  .phone-notch {
    width: 120px; height: 28px;
    background: #0f172a;
    border-radius: 0 0 18px 18px;
  }
  .phone-time {
    position: absolute;
    left: 20px;
    font-size: 0.8rem;
    font-weight: 700;
    color: #0f172a;
  }

  /* 앱 헤더 */
  .app-header {
    padding: 20px 24px 12px;
    background: var(--accent);
    color: white;
    flex-shrink: 0;
  }
  .app-logo { font-size: 1.3rem; font-weight: 800; letter-spacing: -0.5px; }
  .app-tagline { font-size: 0.78rem; opacity: 0.85; margin-top: 2px; }

  /* 스텝 인디케이터 */
  .steps {
    display: flex;
    gap: 0;
    background: var(--accent);
    padding: 0 24px 16px;
    flex-shrink: 0;
  }
  .step {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    cursor: pointer;
    opacity: 0.5;
    transition: opacity .2s;
  }
  .step.active { opacity: 1; }
  .step-dot {
    width: 28px; height: 28px;
    border-radius: 50%;
    background: rgba(255,255,255,0.3);
    display: flex; align-items: center; justify-content: center;
    font-size: 0.75rem;
    font-weight: 700;
    color: white;
    margin-bottom: 4px;
  }
  .step.active .step-dot { background: white; color: var(--accent); }
  .step.done .step-dot { background: var(--green); }
  .step-label { font-size: 0.6rem; color: white; text-align: center; opacity: 0.9; }
  .step-line { flex: 1; height: 2px; background: rgba(255,255,255,0.3); margin-top: 14px; }
  .step-line.done { background: var(--green); }

  /* 스크롤 콘텐츠 */
  .app-content {
    flex: 1;
    overflow-y: auto;
    padding: 24px;
  }

  /* 각 단계 패널 */
  .panel { display: none; }
  .panel.active { display: block; }

  .panel-title { font-size: 1.3rem; font-weight: 800; color: var(--text); margin-bottom: 6px; }
  .panel-sub   { font-size: 0.85rem; color: var(--muted); margin-bottom: 24px; line-height: 1.6; }

  .input-group { margin-bottom: 16px; }
  .input-label { font-size: 0.78rem; color: var(--muted); font-weight: 600; margin-bottom: 6px; display: block; }
  .app-input {
    width: 100%;
    border: 1.5px solid var(--border);
    border-radius: 12px;
    padding: 13px 16px;
    font-size: 0.95rem;
    color: var(--text);
    outline: none;
    transition: border-color .2s;
    background: var(--bg);
  }
  .app-input:focus { border-color: var(--accent); background: white; }

  /* 목소리 등록 UI */
  .voice-recorder {
    border: 2px dashed var(--border);
    border-radius: var(--radius);
    padding: 32px 20px;
    text-align: center;
    margin-bottom: 16px;
    transition: all .2s;
  }
  .voice-recorder.recording {
    border-color: var(--red);
    background: #fff5f5;
    animation: pulse-border 1s infinite;
  }
  .voice-recorder.done {
    border-color: var(--green);
    background: #f0fdf4;
  }
  @keyframes pulse-border { 0%,100%{border-color:var(--red)} 50%{border-color:#fca5a5} }
  .record-icon { font-size: 3rem; margin-bottom: 12px; }
  .record-text { font-size: 0.9rem; color: var(--muted); margin-bottom: 16px; }
  .record-bar {
    height: 6px; border-radius: 3px;
    background: var(--border);
    overflow: hidden;
    margin: 0 auto 12px;
    width: 80%;
  }
  .record-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    width: 0%;
    transition: width .1s;
  }

  /* 카드 등록 */
  .card-preview {
    background: linear-gradient(135deg, #667eea, #764ba2);
    border-radius: 16px;
    padding: 24px;
    color: white;
    margin-bottom: 16px;
    position: relative;
    overflow: hidden;
  }
  .card-preview::before {
    content: '';
    position: absolute;
    top: -30px; right: -30px;
    width: 120px; height: 120px;
    border-radius: 50%;
    background: rgba(255,255,255,0.1);
  }
  .card-chip { font-size: 1.5rem; margin-bottom: 16px; }
  .card-number { font-size: 1.1rem; letter-spacing: 3px; font-weight: 600; margin-bottom: 8px; }
  .card-name { font-size: 0.78rem; opacity: 0.8; }

  /* 취향 태그 */
  .taste-tags { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 16px; }
  .taste-tag {
    padding: 8px 16px;
    border-radius: 999px;
    border: 1.5px solid var(--border);
    font-size: 0.85rem;
    cursor: pointer;
    transition: all .15s;
    color: var(--muted);
  }
  .taste-tag.selected { background: var(--accent); border-color: var(--accent); color: white; }

  /* 완료 화면 */
  .complete-wrap { text-align: center; padding: 20px 0; }
  .complete-check { font-size: 4rem; margin-bottom: 16px; }
  .complete-name  { font-size: 1.4rem; font-weight: 800; color: var(--text); margin-bottom: 8px; }
  .complete-desc  { font-size: 0.88rem; color: var(--muted); line-height: 1.7; margin-bottom: 24px; }
  .badge {
    display: inline-block;
    background: #f0f4ff;
    color: var(--accent);
    border-radius: 999px;
    padding: 6px 16px;
    font-size: 0.8rem;
    font-weight: 600;
    margin: 4px;
  }

  /* 버튼 */
  .btn-app {
    width: 100%;
    padding: 15px;
    border-radius: 14px;
    border: none;
    font-size: 1rem;
    font-weight: 700;
    cursor: pointer;
    transition: all .15s;
    margin-top: 8px;
  }
  .btn-app:hover { opacity: 0.88; transform: translateY(-1px); }
  .btn-primary-app { background: var(--accent); color: white; }
  .btn-outline-app { background: transparent; color: var(--accent); border: 1.5px solid var(--accent); }

  /* 하단 네비게이션 */
  .app-nav {
    display: flex;
    border-top: 1px solid var(--border);
    background: var(--card);
    flex-shrink: 0;
  }
  .nav-item {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 10px;
    font-size: 0.65rem;
    color: var(--muted);
    cursor: pointer;
  }
  .nav-item.active { color: var(--accent); }
  .nav-icon { font-size: 1.2rem; margin-bottom: 2px; }

  .divider { height: 1px; background: var(--border); margin: 16px 0; }
  .info-box {
    background: #f0f4ff;
    border-radius: 12px;
    padding: 14px;
    font-size: 0.82rem;
    color: var(--accent);
    margin-bottom: 16px;
    line-height: 1.6;
  }
</style>
</head>
<body>
<div class="phone">
  <!-- 상단 노치 -->
  <div class="phone-bar">
    <div class="phone-time">9:41</div>
    <div class="phone-notch"></div>
  </div>

  <!-- 앱 헤더 -->
  <div class="app-header">
    <div class="app-logo">AI Cashier</div>
    <div class="app-tagline">목소리로 주문하는 스마트 캐셔</div>
  </div>

  <!-- 스텝 인디케이터 -->
  <div class="steps" id="steps">
    <div class="step active" id="step-0" onclick="goStep(0)">
      <div class="step-dot">1</div>
      <div class="step-label">본인 확인</div>
    </div>
    <div class="step-line" id="line-0"></div>
    <div class="step" id="step-1" onclick="goStep(1)">
      <div class="step-dot">2</div>
      <div class="step-label">목소리 등록</div>
    </div>
    <div class="step-line" id="line-1"></div>
    <div class="step" id="step-2" onclick="goStep(2)">
      <div class="step-dot">2</div>
      <div class="step-label">카드 등록</div>
    </div>
    <div class="step-line" id="line-2"></div>
    <div class="step" id="step-3" onclick="goStep(3)">
      <div class="step-dot">3</div>
      <div class="step-label">취향 설정</div>
    </div>
    <div class="step-line" id="line-3"></div>
    <div class="step" id="step-4" onclick="goStep(4)">
      <div class="step-dot">✓</div>
      <div class="step-label">완료</div>
    </div>
  </div>

  <!-- 콘텐츠 -->
  <div class="app-content">

    <!-- Step 0: 본인 확인 -->
    <div class="panel active" id="panel-0">
      <div class="panel-title">안녕하세요!</div>
      <div class="panel-sub">이름과 전화번호로 간단히 등록하세요.<br>키오스크 방문 기록이 있으면 자동으로 연결됩니다.</div>
      <div class="input-group">
        <label class="input-label">이름</label>
        <input class="app-input" id="app-name" type="text" placeholder="홍길동" value="">
      </div>
      <div class="input-group">
        <label class="input-label">전화번호</label>
        <input class="app-input" id="app-phone" type="tel" placeholder="01012345678" value="">
      </div>
      <div class="info-box" id="kiosk-link-box" style="display:none">
        🔗 키오스크 방문 기록을 찾았어요!<br>
        <strong id="kiosk-found-text"></strong> — 연결하고 목소리를 다시 등록하면 더 정확해집니다.
      </div>
      <button class="btn-app btn-primary-app" onclick="step0Next()">다음</button>
    </div>

    <!-- Step 1: 목소리 등록 -->
    <div class="panel" id="panel-1">
      <div class="panel-title">목소리 등록</div>
      <div class="panel-sub">조용한 환경에서 아래 문장을 읽어주세요.<br>키오스크에서보다 훨씬 정확하게 등록됩니다.</div>
      <div class="voice-recorder" id="voice-recorder">
        <div class="record-icon" id="rec-icon">🎤</div>
        <div class="record-text" id="rec-text">버튼을 눌러 녹음을 시작하세요</div>
        <div style="background:#f0f4ff;border-radius:12px;padding:14px;margin-bottom:16px;font-size:0.9rem;color:#334155;line-height:1.7;font-style:italic;">
          "안녕하세요, 저는 AI 캐셔를 이용하고 싶어요.<br>치즈버거 하나랑 콜라 주세요!"
        </div>
        <div class="record-bar"><div class="record-fill" id="rec-fill"></div></div>
        <button class="btn-app btn-primary-app" id="rec-btn" onclick="toggleRecord()">녹음 시작</button>
      </div>
      <div id="rec-status" style="display:none;text-align:center;color:var(--green);font-weight:600;margin-bottom:12px;">
        ✅ 녹음 완료! (5.2초)
      </div>
      <button class="btn-app btn-primary-app" id="voice-next-btn" onclick="goStep(2)" style="display:none">다음</button>
      <button class="btn-app btn-outline-app" onclick="goStep(2)" style="margin-top:8px">건너뛰기 (나중에)</button>
    </div>

    <!-- Step 2: 카드 등록 -->
    <div class="panel" id="panel-2">
      <div class="panel-title">카드 등록</div>
      <div class="panel-sub">앱 카드를 등록하면 키오스크에서<br>목소리만으로 결제할 수 있어요.</div>
      <div class="card-preview" id="card-preview">
        <div class="card-chip">▣</div>
        <div class="card-number" id="card-num-display">•••• •••• •••• ••••</div>
        <div class="card-name" id="card-name-display">홍 길 동</div>
      </div>
      <div class="input-group">
        <label class="input-label">카드 번호</label>
        <input class="app-input" id="card-num" type="tel" placeholder="0000 0000 0000 0000"
          oninput="formatCardNum(this)" maxlength="19">
      </div>
      <div style="display:flex;gap:10px;">
        <div class="input-group" style="flex:1">
          <label class="input-label">유효기간</label>
          <input class="app-input" id="card-exp" type="tel" placeholder="MM/YY" maxlength="5"
            oninput="formatExp(this)">
        </div>
        <div class="input-group" style="flex:1">
          <label class="input-label">CVC</label>
          <input class="app-input" id="card-cvc" type="tel" placeholder="•••" maxlength="3">
        </div>
      </div>
      <button class="btn-app btn-primary-app" onclick="goStep(3)">카드 등록</button>
      <button class="btn-app btn-outline-app" onclick="goStep(3)" style="margin-top:8px">건너뛰기 (나중에)</button>
    </div>

    <!-- Step 3: 취향 설정 -->
    <div class="panel" id="panel-3">
      <div class="panel-title">취향 설정</div>
      <div class="panel-sub">AI가 맞춤 메뉴를 추천할 때 활용해요.<br>여러 개 선택 가능합니다.</div>
      <div class="input-label" style="margin-bottom:10px">선호 카테고리</div>
      <div class="taste-tags" id="taste-tags">
        <div class="taste-tag" onclick="toggleTag(this)">🍔 버거</div>
        <div class="taste-tag" onclick="toggleTag(this)">🍟 사이드</div>
        <div class="taste-tag" onclick="toggleTag(this)">🥤 음료</div>
        <div class="taste-tag" onclick="toggleTag(this)">🎁 세트</div>
      </div>
      <div class="divider"></div>
      <div class="input-label" style="margin-bottom:10px">식이 제한</div>
      <div class="taste-tags">
        <div class="taste-tag" onclick="toggleTag(this)">🥗 채식</div>
        <div class="taste-tag" onclick="toggleTag(this)">🌶 매운맛 좋아요</div>
        <div class="taste-tag" onclick="toggleTag(this)">🧀 치즈 좋아요</div>
        <div class="taste-tag" onclick="toggleTag(this)">🦐 해산물 좋아요</div>
      </div>
      <div class="divider"></div>
      <div class="input-label" style="margin-bottom:10px">알림 설정</div>
      <div class="taste-tags">
        <div class="taste-tag selected" onclick="toggleTag(this)">🔔 주문 완료 알림</div>
        <div class="taste-tag selected" onclick="toggleTag(this)">⭐ 신메뉴 알림</div>
        <div class="taste-tag" onclick="toggleTag(this)">🎫 할인 쿠폰 알림</div>
      </div>
      <button class="btn-app btn-primary-app" style="margin-top:16px" onclick="goStep(4)">완료</button>
    </div>

    <!-- Step 4: 완료 -->
    <div class="panel" id="panel-4">
      <div class="complete-wrap">
        <div class="complete-check">🎉</div>
        <div class="complete-name" id="complete-name">홍길동님, 등록 완료!</div>
        <div class="complete-desc">
          이제 키오스크에서 말씀만 하시면<br>
          바로 주문이 시작됩니다.<br><br>
          등록된 기능:
        </div>
        <div>
          <span class="badge">🎤 목소리 인식</span>
          <span class="badge" id="badge-card">💳 앱 카드</span>
          <span class="badge">⭐ 맞춤 추천</span>
        </div>
        <div class="divider"></div>
        <div style="background:#f0fdf4;border-radius:12px;padding:16px;text-align:left;font-size:0.85rem;color:#166534;line-height:1.7;">
          💡 <strong>다음 키오스크 방문 시</strong><br>
          말씀하시면 자동으로 인식되어<br>
          "어서오세요 홍길동님!"으로 맞이합니다.
        </div>
        <button class="btn-app btn-outline-app" style="margin-top:20px" onclick="window.close()">
          키오스크 화면으로 돌아가기
        </button>
      </div>
    </div>

  </div><!-- /.app-content -->

  <!-- 하단 네비 -->
  <div class="app-nav">
    <div class="nav-item active"><div class="nav-icon">🏠</div>홈</div>
    <div class="nav-item"><div class="nav-icon">📋</div>주문내역</div>
    <div class="nav-item"><div class="nav-icon">👤</div>내 정보</div>
    <div class="nav-item"><div class="nav-icon">⚙️</div>설정</div>
  </div>
</div>

<script>
let currentStep = 0;
let isRecording = false;
let recInterval = null;
let recProgress = 0;
let userName = '';

function goStep(n) {
  // 뒤로 가기 막기 (완료 후)
  if (currentStep === 4 && n < 4) return;

  document.querySelectorAll('.panel').forEach((p,i) => p.classList.toggle('active', i === n));

  // 스텝 인디케이터 업데이트
  for (let i = 0; i <= 4; i++) {
    const el = document.getElementById('step-' + i);
    if (!el) continue;
    el.classList.remove('active','done');
    if (i < n) el.classList.add('done');
    else if (i === n) el.classList.add('active');
  }
  for (let i = 0; i <= 3; i++) {
    const line = document.getElementById('line-' + i);
    if (line) line.classList.toggle('done', i < n);
  }
  currentStep = n;

  // 완료 화면 이름 업데이트
  if (n === 4) {
    const name = document.getElementById('app-name').value || '고객';
    document.getElementById('complete-name').textContent = name + '님, 등록 완료!';
    const cardNum = document.getElementById('card-num').value;
    if (!cardNum) document.getElementById('badge-card').style.display = 'none';
  }
}

function step0Next() {
  const name = document.getElementById('app-name').value.trim();
  const phone = document.getElementById('app-phone').value.trim();
  if (!name) { alert('이름을 입력해주세요.'); return; }
  if (!/^01[0-9]{8,9}$/.test(phone)) { alert('전화번호를 올바르게 입력해주세요.'); return; }
  userName = name;
  // 키오스크 연동 시뮬레이션 — 010-1234-5678이면 기존 기록 있는 척
  if (phone === '01012345678') {
    document.getElementById('kiosk-link-box').style.display = 'block';
    document.getElementById('kiosk-found-text').textContent = '키오스크 방문 1회 기록';
  }
  goStep(1);
}

// 목소리 녹음 시뮬레이션
function toggleRecord() {
  if (isRecording) {
    stopRecord();
  } else {
    startRecord();
  }
}

function startRecord() {
  isRecording = true;
  recProgress = 0;
  const recorder = document.getElementById('voice-recorder');
  const btn = document.getElementById('rec-btn');
  const fill = document.getElementById('rec-fill');
  const icon = document.getElementById('rec-icon');
  const text = document.getElementById('rec-text');
  recorder.classList.add('recording');
  btn.textContent = '⏹ 녹음 중지';
  icon.textContent = '🔴';
  text.textContent = '녹음 중... 문장을 읽어주세요';
  recInterval = setInterval(() => {
    recProgress = Math.min(recProgress + 2, 100);
    fill.style.width = recProgress + '%';
    if (recProgress >= 100) stopRecord();
  }, 100);
}

function stopRecord() {
  isRecording = false;
  clearInterval(recInterval);
  const recorder = document.getElementById('voice-recorder');
  const btn = document.getElementById('rec-btn');
  const icon = document.getElementById('rec-icon');
  const text = document.getElementById('rec-text');
  const status = document.getElementById('rec-status');
  const nextBtn = document.getElementById('voice-next-btn');
  recorder.classList.remove('recording');
  recorder.classList.add('done');
  btn.textContent = '다시 녹음';
  icon.textContent = '✅';
  text.textContent = '목소리 등록 완료!';
  status.style.display = 'block';
  nextBtn.style.display = 'block';
}

// 카드 번호 포맷
function formatCardNum(input) {
  let v = input.value.replace(/\D/g,'').slice(0,16);
  input.value = v.replace(/(.{4})/g,'$1 ').trim();
  const display = v ? v.replace(/(.{4})/g,'$1 ').trim() : '•••• •••• •••• ••••';
  document.getElementById('card-num-display').textContent = display;
}
function formatExp(input) {
  let v = input.value.replace(/\D/g,'');
  if (v.length >= 2) v = v.slice(0,2) + '/' + v.slice(2,4);
  input.value = v;
}

// 취향 태그 토글
function toggleTag(el) { el.classList.toggle('selected'); }

// 카드 이름 연동
document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('app-name').addEventListener('input', e => {
    document.getElementById('card-name-display').textContent =
      e.target.value ? e.target.value.split('').join(' ') : '홍 길 동';
  });
});
</script>
</body>
</html>"""
