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
    --bg: #0f172a;
    --bg2: #1e293b;
    --bg3: #334155;
    --accent: #6366f1;
    --accent2: #8b5cf6;
    --green: #22c55e;
    --yellow: #f59e0b;
    --red: #ef4444;
    --blue: #3b82f6;
    --text: #f1f5f9;
    --muted: #94a3b8;
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
    background: var(--bg2);
    border-bottom: 1px solid var(--bg3);
    flex-shrink: 0;
  }
  #statusbar .logo {
    font-size: 1.1rem;
    font-weight: 800;
    color: var(--accent);
    margin-right: auto;
    letter-spacing: -0.5px;
  }
  .chip {
    display: flex;
    align-items: center;
    gap: 6px;
    background: var(--bg);
    border: 1px solid var(--bg3);
    border-radius: 999px;
    padding: 5px 12px;
    font-size: 0.78rem;
    color: var(--muted);
  }
  .dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
  }
  .dot-active    { background: var(--green);  box-shadow: 0 0 6px var(--green); }
  .dot-listening { background: var(--blue);   box-shadow: 0 0 6px var(--blue);  animation: blink 1s infinite; }
  .dot-processing{ background: var(--yellow); box-shadow: 0 0 6px var(--yellow);}
  .dot-speaking  { background: var(--accent2);box-shadow: 0 0 6px var(--accent2); animation: blink .7s infinite; }
  .dot-idle      { background: var(--bg3); }
  .dot-ok        { background: var(--green);  box-shadow: 0 0 6px var(--green); }
  .dot-fail      { background: var(--red);    box-shadow: 0 0 6px var(--red); }
  .dot-pending   { background: var(--bg3); }

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
  .cart-item-price{ color: var(--accent); font-weight: 600; }
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
  #screen-ordering { width: 100%; align-items: flex-start; justify-content: flex-start; }
  #ai-speech-box {
    background: var(--bg2);
    border: 1px solid var(--bg3);
    border-radius: var(--radius);
    padding: 20px 24px;
    width: 100%;
    margin-bottom: 20px;
    min-height: 80px;
  }
  #ai-speech-label { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 1px; color: var(--muted); margin-bottom: 8px; }
  #ai-speech-text  { font-size: 1.15rem; line-height: 1.7; color: #a78bfa; min-height: 1.7em; }

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
  #screen-checkout { text-align: center; max-width: 480px; margin: 0 auto; width: 100%; }
  .checkout-title { font-size: 1.6rem; font-weight: 800; margin-bottom: 24px; }
  #checkout-summary {
    background: var(--bg2);
    border: 1px solid var(--bg3);
    border-radius: var(--radius);
    padding: 20px;
    width: 100%;
    margin-bottom: 24px;
    text-align: left;
  }
  .summary-item {
    display: flex;
    justify-content: space-between;
    padding: 8px 0;
    font-size: 0.95rem;
    border-bottom: 1px solid var(--bg3);
  }
  .summary-item:last-child { border: none; }
  .summary-total {
    display: flex;
    justify-content: space-between;
    font-size: 1.2rem;
    font-weight: 700;
    padding-top: 12px;
    margin-top: 4px;
  }
  .pay-buttons { display: flex; gap: 12px; width: 100%; }

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
</div>

<!-- 메인 -->
<div id="main">
  <!-- 왼쪽: 화면 콘텐츠 -->
  <div id="content">

    <!-- 대기 화면 -->
    <div class="screen active" id="screen-waiting">
      <div class="waiting-icon">🎤</div>
      <div class="waiting-title">안녕하세요!</div>
      <div class="waiting-hint">말씀하시면 바로 주문을 도와드립니다</div>
    </div>

    <!-- 주문 화면 -->
    <div class="screen" id="screen-ordering">
      <div id="ai-speech-box">
        <div id="ai-speech-label">AI 캐셔</div>
        <div id="ai-speech-text">말씀하세요...</div>
      </div>
      <div id="menu-section">
        <div class="section-title">메뉴</div>
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
      <div class="checkout-title">주문 확인</div>
      <div id="checkout-summary"></div>
      <div class="pay-buttons">
        <button class="btn btn-primary" onclick="selectPayment('app_card')">📱 앱 카드</button>
        <button class="btn btn-outline" onclick="selectPayment('physical_card')">💳 현장 카드</button>
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
let currentState = {};

// ── 화면 전환 ──
function showScreen(name) {
  document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
  const el = document.getElementById('screen-' + name);
  if (el) el.classList.add('active');

  // 장바구니 패널: 주문 중에만 표시
  const cartPanel = document.getElementById('cart-panel');
  cartPanel.classList.toggle('hidden', name !== 'ordering');
}

// ── 상태 적용 ──
function applyState(state) {
  currentState = state;

  // 화면 전환
  showScreen(state.screen);

  // 상태바
  updateStatusBar(state);

  // AI 텍스트
  if (state.ai_text !== undefined) {
    document.getElementById('ai-speech-text').textContent = state.ai_text || '말씀하세요...';
  }

  // 장바구니
  renderCart(state.cart_items || [], state.cart_total || 0);

  // 결제 화면 요약
  if (state.screen === 'checkout') {
    renderCheckoutSummary(state.cart_items || [], state.cart_total || 0);
  }

  // 완료 화면 메시지 커스텀
  if (state.screen === 'complete') {
    const name = state.user_name;
    document.getElementById('complete-title').textContent = name ? `감사합니다, ${name}님!` : '주문 완료!';
    document.getElementById('complete-sub').textContent =
      state.transaction_id
        ? `결제 완료 (${state.transaction_id})\n음식이 준비되면 안내드립니다.`
        : '음식이 준비되면 안내드립니다.';
  }
}

function updateStatusBar(state) {
  const CONV_LABELS = {idle:'대기 중', listening:'듣는 중', processing:'처리 중', speaking:'말하는 중'};
  const convDot  = document.getElementById('conv-dot');
  const convText = document.getElementById('conv-text');
  convDot.className  = 'dot dot-' + state.conversation;
  convText.textContent = CONV_LABELS[state.conversation] || state.conversation;

  const spkDot  = document.getElementById('spk-dot');
  const spkText = document.getElementById('spk-text');
  if (state.speaker_verified === true)  { spkDot.className='dot dot-ok';   spkText.textContent='인증됨'; }
  else if (state.speaker_verified===false){ spkDot.className='dot dot-fail'; spkText.textContent='불일치'; }
  else { spkDot.className='dot dot-pending'; spkText.textContent='화자인증'; }
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

function checkout()              { post('/action/checkout', {}); }
function selectPayment(method)   { post('/action/payment', {method}); }
function saveVoice(save)         { post('/action/save_voice', {save}); }
function retryVerification()     { post('/action/retry_verification', {}); }

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

// ── 초기 메뉴 렌더링 ──
renderMenuGrid('버거');
</script>
</body>
</html>"""
