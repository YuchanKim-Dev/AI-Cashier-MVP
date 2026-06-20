"""
키오스크 프론트엔드 — FastAPI 서버 (프리미엄 UI v3).

화면(screen) 종류:
  waiting             대기 화면
  ordering            주문 중 (장바구니 + AI 대화)
  checkout            결제 화면 (주문 요약 + 결제 방법 선택)
  payment_processing  결제 처리 중
  voice_save_prompt   목소리 저장 질문 (신규 + 3초 이상)
  register            이름 + 전화번호 입력
  complete            주문 완료
  locked              화자인증 실패 잠금
"""

import asyncio
import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, Response, StreamingResponse

app = FastAPI(title="Voice AI Cashier Kiosk")

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
_tts_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="tts")


def set_context(action_queue: asyncio.Queue, loop: asyncio.AbstractEventLoop):
    global _action_queue, _main_loop
    _action_queue = action_queue
    _main_loop = loop


def push_state(updates: dict):
    _state.update(updates)
    data = json.dumps(_state, ensure_ascii=False)
    for q in _sse_queues:
        try:
            q.put_nowait(data)
        except asyncio.QueueFull:
            pass


def _enqueue_action(action: dict):
    if _main_loop and _action_queue:
        _main_loop.call_soon_threadsafe(_action_queue.put_nowait, action)


# ─── HTTP 엔드포인트 ────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse(content=_build_html())


@app.get("/events")
async def sse():
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


@app.get("/tts")
async def tts_endpoint(text: str):
    """OpenAI TTS — 더 자연스러운 한국어 음성 (nova 모델)."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key or not text.strip():
        return Response(status_code=204)

    def _synthesize() -> bytes:
        from openai import OpenAI as OpenAISync
        client = OpenAISync(api_key=api_key)
        resp = client.audio.speech.create(
            model="tts-1-hd",
            voice="nova",
            input=text[:600],
            response_format="mp3",
        )
        return resp.content

    loop = asyncio.get_event_loop()
    try:
        audio_bytes = await loop.run_in_executor(_tts_executor, _synthesize)
        return Response(
            content=audio_bytes,
            media_type="audio/mpeg",
            headers={"Cache-Control": "no-cache"},
        )
    except Exception as e:
        print(f"[TTS] 오류: {e}")
        return Response(status_code=500)


@app.post("/action/checkout")
async def action_checkout():
    _enqueue_action({"type": "checkout"})
    return {"ok": True}


@app.post("/action/payment")
async def action_payment(request: Request):
    body = await request.json()
    _enqueue_action({"type": "payment", "method": body.get("method", "physical_card")})
    return {"ok": True}


@app.post("/action/save_voice")
async def action_save_voice(request: Request):
    body = await request.json()
    _enqueue_action({"type": "save_voice", "save": body.get("save", False)})
    return {"ok": True}


@app.post("/action/register")
async def action_register(request: Request):
    body = await request.json()
    _enqueue_action({
        "type": "register",
        "name": body.get("name", "").strip(),
        "phone": body.get("phone", "").strip(),
    })
    return {"ok": True}


@app.post("/action/retry_verification")
async def action_retry():
    _enqueue_action({"type": "retry_verification"})
    return {"ok": True}


@app.post("/action/add_menu")
async def action_add_menu(request: Request):
    body = await request.json()
    _enqueue_action({"type": "add_menu", "name": body.get("name", "")})
    return {"ok": True}


@app.post("/action/start")
async def action_start():
    _enqueue_action({"type": "start"})
    return {"ok": True}


@app.post("/action/reset")
async def action_reset():
    _enqueue_action({"type": "reset"})
    return {"ok": True}


@app.post("/action/app_payment_confirm")
async def action_app_payment_confirm(request: Request):
    body = await request.json()
    _enqueue_action({"type": "app_payment_confirm", "method": body.get("method", "신용카드")})
    return {"ok": True}


@app.get("/app", response_class=HTMLResponse)
async def app_demo():
    return HTMLResponse(content=_build_app_html())


# ─── HTML ──────────────────────────────────────────────────────────────────────

def _build_html() -> str:
    return r"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AI Cashier</title>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg:        #f9fafb;
  --bg2:       #ffffff;
  --bg3:       #f3f4f6;
  --border:    #e5e7eb;
  --border2:   #d1d5db;
  --accent:    #c2410c;
  --accent-lt: #fed7aa;
  --green:     #15803d;
  --green-lt:  #dcfce7;
  --red:       #b91c1c;
  --blue:      #1d4ed8;
  --blue-lt:   #dbeafe;
  --text:      #111827;
  --text2:     #374151;
  --muted:     #6b7280;
  --muted2:    #9ca3af;
  --header:    #0f172a;
  --header2:   #1e293b;
  --radius:    14px;
  --shadow:    0 1px 3px rgba(0,0,0,.08), 0 1px 2px rgba(0,0,0,.06);
  --shadow-md: 0 4px 6px rgba(0,0,0,.07), 0 2px 4px rgba(0,0,0,.06);
  --shadow-lg: 0 10px 15px rgba(0,0,0,.08), 0 4px 6px rgba(0,0,0,.05);
}

body {
  background: var(--bg);
  color: var(--text);
  font-family: -apple-system, 'Apple SD Gothic Neo', 'Noto Sans KR', 'Malgun Gothic', sans-serif;
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  font-size: 15px;
  line-height: 1.5;
}

/* ── 헤더 ── */
#header {
  background: var(--header);
  padding: 0 28px;
  height: 60px;
  display: flex;
  align-items: center;
  gap: 16px;
  flex-shrink: 0;
  border-bottom: 1px solid rgba(255,255,255,.06);
}
#header .brand {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-right: auto;
}
#header .brand-icon {
  width: 34px; height: 34px;
  background: var(--accent);
  border-radius: 10px;
  display: flex; align-items: center; justify-content: center;
  font-size: 1.1rem;
}
#header .brand-name {
  font-size: 1.05rem;
  font-weight: 700;
  color: #fff;
  letter-spacing: -.3px;
}
#header .brand-sub {
  font-size: .65rem;
  color: #64748b;
  font-weight: 400;
  margin-top: 1px;
}

.hchip {
  display: flex;
  align-items: center;
  gap: 7px;
  background: var(--header2);
  border: 1px solid rgba(255,255,255,.08);
  border-radius: 8px;
  padding: 6px 12px;
  font-size: .75rem;
  color: #94a3b8;
  white-space: nowrap;
}
.hchip .dot {
  width: 7px; height: 7px;
  border-radius: 50%;
  flex-shrink: 0;
}
.dot-active    { background: #4ade80; }
.dot-listening { background: #60a5fa; animation: blink 1s infinite; }
.dot-processing{ background: #fbbf24; animation: blink .6s infinite; }
.dot-speaking  { background: #c084fc; animation: blink .7s infinite; }
.dot-idle      { background: #334155; }
.dot-ok        { background: #4ade80; }
.dot-fail      { background: #f87171; }
.dot-pending   { background: #334155; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.3} }

.hbtn {
  display: flex; align-items: center; gap: 6px;
  background: rgba(255,255,255,.06);
  border: 1px solid rgba(255,255,255,.1);
  border-radius: 8px;
  padding: 6px 14px;
  font-size: .78rem;
  font-weight: 500;
  color: #cbd5e1;
  cursor: pointer;
  transition: all .15s;
}
.hbtn:hover { background: rgba(255,255,255,.12); color: #fff; }
.hbtn.tts-on { background: #c2410c; border-color: transparent; color: #fff; }

/* ── 레이아웃 ── */
#main { display: flex; flex: 1; overflow: hidden; }

#content {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow-y: auto;
  padding: 28px;
  gap: 20px;
}

/* ── 장바구니 패널 ── */
#cart-panel {
  width: 288px;
  flex-shrink: 0;
  background: var(--bg2);
  border-left: 1px solid var(--border);
  display: flex;
  flex-direction: column;
}
#cart-panel.hidden { display: none; }

#cart-header {
  padding: 18px 20px 14px;
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: .9rem;
  font-weight: 700;
  color: var(--text);
}
#cart-header .cart-count {
  background: var(--accent);
  color: #fff;
  border-radius: 99px;
  padding: 1px 8px;
  font-size: .7rem;
  font-weight: 700;
}
#cart-items { flex: 1; overflow-y: auto; }
.cart-item {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  padding: 13px 20px;
  border-bottom: 1px solid var(--border);
  gap: 10px;
}
.cart-item-info { flex: 1; min-width: 0; }
.cart-item-name {
  font-weight: 600;
  font-size: .9rem;
  color: var(--text);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.cart-item-includes {
  font-size: .73rem;
  color: var(--accent);
  margin-top: 2px;
}
.cart-item-qty {
  font-size: .75rem;
  color: var(--muted);
  margin-top: 2px;
}
.cart-item-price {
  font-weight: 700;
  font-size: .9rem;
  color: var(--text);
  white-space: nowrap;
}
.cart-empty {
  padding: 48px 20px;
  text-align: center;
  color: var(--muted2);
  font-size: .85rem;
}
.cart-empty-icon { font-size: 2.4rem; margin-bottom: 10px; opacity: .4; }

#cart-footer {
  border-top: 1px solid var(--border);
  padding: 16px 20px;
}
.cart-total-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 14px;
  font-size: .92rem;
}
.cart-total-label { color: var(--muted); }
.cart-total-price { font-weight: 800; font-size: 1.1rem; color: var(--text); }

/* ── 화면 ── */
.screen { display: none; flex-direction: column; align-items: center; justify-content: center; flex: 1; }
.screen.active { display: flex; }

/* 대기 */
#screen-waiting { text-align: center; }
.waiting-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: var(--green-lt);
  color: var(--green);
  border-radius: 99px;
  padding: 5px 14px;
  font-size: .78rem;
  font-weight: 600;
  margin-bottom: 28px;
}
.waiting-icon { font-size: 4.5rem; margin-bottom: 20px; }
.waiting-title {
  font-size: 2.2rem;
  font-weight: 800;
  letter-spacing: -.5px;
  color: var(--text);
  margin-bottom: 10px;
}
.waiting-desc { color: var(--muted); font-size: 1rem; margin-bottom: 36px; line-height: 1.7; }

/* 주문 화면 */
#screen-ordering { width: 100%; align-items: flex-start; justify-content: flex-start; }

/* 대화 패널 */
#conv-panel {
  width: 100%;
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  display: flex;
  flex-direction: column;
  overflow: hidden;
  box-shadow: var(--shadow);
  flex-shrink: 0;
}
#conv-header {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 13px 16px;
  border-bottom: 1px solid var(--border);
  background: var(--bg3);
}
.conv-ai-avatar {
  width: 32px; height: 32px;
  background: var(--accent);
  border-radius: 10px;
  display: flex; align-items: center; justify-content: center;
  font-size: 1rem;
  flex-shrink: 0;
}
.conv-ai-name { font-weight: 700; font-size: .88rem; color: var(--text); }
.conv-ai-status {
  font-size: .73rem;
  color: var(--muted);
  margin-top: 1px;
}
#conv-header-badge {
  margin-left: auto;
  font-size: .7rem;
  color: var(--muted);
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 3px 8px;
}

/* 채팅 로그 */
#chat-log {
  overflow-y: auto;
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 12px;
  max-height: 260px;
  scroll-behavior: smooth;
  background: var(--bg3);
}

.msg-row { display: flex; gap: 9px; align-items: flex-end; }
.msg-row.user { flex-direction: row-reverse; }

.msg-av {
  width: 30px; height: 30px;
  border-radius: 8px;
  display: flex; align-items: center; justify-content: center;
  font-size: .9rem;
  flex-shrink: 0;
  align-self: flex-start;
}
.msg-av.ai   { background: var(--accent); }
.msg-av.user { background: var(--header); }

.msg-body { display: flex; flex-direction: column; max-width: 78%; }
.msg-row.user .msg-body { align-items: flex-end; }

.msg-name { font-size: .68rem; color: var(--muted); margin-bottom: 4px; font-weight: 500; }

.msg-bubble {
  padding: 10px 14px;
  border-radius: 12px;
  font-size: .92rem;
  line-height: 1.55;
  word-break: break-word;
  box-shadow: var(--shadow);
}
.msg-bubble.ai {
  background: var(--bg2);
  color: var(--text);
  border-radius: 2px 12px 12px 12px;
  border-left: 3px solid var(--accent);
}
.msg-bubble.user {
  background: var(--header);
  color: #f1f5f9;
  border-radius: 12px 2px 12px 12px;
}

/* 푸터 */
#conv-footer {
  border-top: 1px solid var(--border);
  padding: 11px 16px;
  min-height: 50px;
  display: flex;
  align-items: center;
  gap: 10px;
  background: var(--bg2);
}
#conv-footer-icon {
  width: 28px; height: 28px;
  border-radius: 8px;
  background: var(--bg3);
  border: 1px solid var(--border);
  display: flex; align-items: center; justify-content: center;
  font-size: .9rem;
  flex-shrink: 0;
}
#conv-footer-text { flex: 1; font-size: .88rem; color: var(--muted); }

/* 타이핑 점 */
.typing-dots { display: flex; gap: 4px; align-items: center; }
.typing-dot {
  width: 6px; height: 6px;
  border-radius: 50%;
  background: var(--muted2);
  animation: tdot 1.2s infinite;
}
.typing-dot:nth-child(2) { animation-delay: .2s; }
.typing-dot:nth-child(3) { animation-delay: .4s; }
@keyframes tdot { 0%,60%,100%{transform:translateY(0)} 30%{transform:translateY(-5px)} }

/* 마이크 파동 */
.mic-wave { display: flex; align-items: center; gap: 3px; }
.mic-bar {
  width: 3px;
  background: var(--blue);
  border-radius: 2px;
  animation: mbar .8s infinite ease-in-out;
}
.mic-bar:nth-child(1){height:7px;  animation-delay:0s}
.mic-bar:nth-child(2){height:14px; animation-delay:.15s}
.mic-bar:nth-child(3){height:9px;  animation-delay:.3s}
.mic-bar:nth-child(4){height:18px; animation-delay:.45s}
.mic-bar:nth-child(5){height:11px; animation-delay:.6s}
@keyframes mbar { 0%,100%{transform:scaleY(.35)} 50%{transform:scaleY(1)} }

/* 메뉴 */
#menu-section { width: 100%; }
.section-label {
  font-size: .72rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: .8px;
  color: var(--muted);
  margin-bottom: 12px;
}
.menu-tabs {
  display: flex;
  gap: 6px;
  margin-bottom: 14px;
}
.menu-tab {
  padding: 7px 16px;
  border-radius: 8px;
  border: 1px solid var(--border);
  background: var(--bg2);
  color: var(--muted);
  font-size: .82rem;
  font-weight: 500;
  cursor: pointer;
  transition: all .15s;
}
.menu-tab.active {
  background: var(--header);
  color: #fff;
  border-color: var(--header);
}
.menu-tab:not(.active):hover {
  border-color: var(--border2);
  color: var(--text);
}

.menu-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(148px, 1fr));
  gap: 10px;
}
.menu-card {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 16px 14px;
  cursor: pointer;
  transition: all .15s;
  box-shadow: var(--shadow);
}
.menu-card:hover {
  border-color: var(--accent);
  box-shadow: 0 0 0 3px rgba(194,65,12,.08), var(--shadow-md);
  transform: translateY(-1px);
}
.menu-card-emoji { font-size: 1.6rem; margin-bottom: 8px; }
.menu-card-name { font-weight: 600; font-size: .88rem; color: var(--text); margin-bottom: 4px; }
.menu-card-price { font-size: .82rem; color: var(--accent); font-weight: 700; }

/* 결제 화면 */
#screen-checkout {
  width: 100%;
  align-items: flex-start;
  justify-content: flex-start;
  flex-direction: row !important;
  gap: 20px;
}
#checkout-left {
  flex: 1;
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  overflow: hidden;
  display: flex;
  flex-direction: column;
  min-width: 0;
  box-shadow: var(--shadow);
}
#checkout-left-header {
  padding: 12px 16px;
  border-bottom: 1px solid var(--border);
  font-size: .72rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: .8px;
  color: var(--muted);
  background: var(--bg3);
}
#checkout-convo {
  overflow-y: auto;
  max-height: 440px;
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 12px;
  scroll-behavior: smooth;
  background: var(--bg3);
}
#checkout-right {
  width: 296px;
  flex-shrink: 0;
  display: flex;
  flex-direction: column;
  gap: 14px;
}
.checkout-title {
  font-size: 1.3rem;
  font-weight: 800;
  letter-spacing: -.3px;
  color: var(--text);
}

#checkout-summary {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  overflow: hidden;
  box-shadow: var(--shadow);
}
.summary-header {
  padding: 12px 16px;
  background: var(--bg3);
  border-bottom: 1px solid var(--border);
  font-size: .72rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: .8px;
  color: var(--muted);
}
.summary-body { padding: 8px 0; }
.summary-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 9px 16px;
  font-size: .88rem;
  border-bottom: 1px solid var(--border);
}
.summary-item:last-child { border: none; }
.summary-item-name { color: var(--text2); }
.summary-item-price { font-weight: 600; color: var(--text); }
.summary-total {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 13px 16px;
  font-size: .95rem;
  font-weight: 800;
  border-top: 2px solid var(--border2);
}
.summary-total-price { color: var(--accent); font-size: 1.1rem; }

.pay-section-title {
  font-size: .72rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: .8px;
  color: var(--muted);
  margin-bottom: 10px;
}
.pay-buttons { display: flex; flex-direction: column; gap: 8px; }

/* 처리 중 */
#screen-payment_processing { text-align: center; }
.spinner {
  width: 52px; height: 52px;
  border: 3px solid var(--border2);
  border-top-color: var(--accent);
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin: 0 auto 24px;
}
@keyframes spin { to { transform: rotate(360deg); } }

/* 완료 */
#screen-complete { text-align: center; }
.complete-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: var(--green-lt);
  color: var(--green);
  border-radius: 99px;
  padding: 6px 16px;
  font-size: .82rem;
  font-weight: 700;
  margin-bottom: 20px;
}
.complete-icon { font-size: 4rem; margin-bottom: 16px; }
.complete-title { font-size: 2rem; font-weight: 800; color: var(--text); margin-bottom: 8px; letter-spacing: -.3px; }

/* 등록 화면 */
#screen-register { max-width: 400px; margin: 0 auto; width: 100%; text-align: left; }
#screen-voice_save_prompt { text-align: center; max-width: 420px; margin: 0 auto; }
#screen-locked { text-align: center; max-width: 400px; margin: 0 auto; }
#screen-card_insert { text-align: center; max-width: 400px; margin: 0 auto; }
#screen-app_payment { text-align: center; max-width: 400px; margin: 0 auto; }

.screen-icon { font-size: 3.5rem; margin-bottom: 18px; }
.screen-title {
  font-size: 1.5rem;
  font-weight: 800;
  color: var(--text);
  margin-bottom: 8px;
  letter-spacing: -.3px;
}
.screen-desc {
  color: var(--muted);
  font-size: .92rem;
  line-height: 1.7;
  margin-bottom: 28px;
}

.input-group { margin-bottom: 14px; }
.input-label {
  font-size: .78rem;
  font-weight: 600;
  color: var(--text2);
  margin-bottom: 6px;
  display: block;
}
.kiosk-input {
  width: 100%;
  background: var(--bg2);
  border: 1.5px solid var(--border);
  border-radius: 10px;
  padding: 12px 14px;
  color: var(--text);
  font-size: .95rem;
  outline: none;
  transition: border-color .2s, box-shadow .2s;
}
.kiosk-input:focus {
  border-color: var(--accent);
  box-shadow: 0 0 0 3px rgba(194,65,12,.1);
}

/* ── 버튼 ── */
.btn {
  padding: 12px 24px;
  border-radius: 10px;
  border: none;
  font-size: .92rem;
  font-weight: 600;
  cursor: pointer;
  transition: all .15s;
  width: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 7px;
}
.btn:hover { opacity: .88; transform: translateY(-1px); box-shadow: var(--shadow-md); }
.btn:active { transform: translateY(0); opacity: 1; }

.btn-primary  { background: var(--accent); color: #fff; }
.btn-dark     { background: var(--header); color: #fff; }
.btn-success  { background: #15803d; color: #fff; }
.btn-danger   { background: var(--red); color: #fff; }
.btn-outline  {
  background: transparent;
  color: var(--text2);
  border: 1.5px solid var(--border2);
}
.btn-outline:hover { border-color: var(--text2); }

.btn-lg { padding: 14px 28px; font-size: 1rem; border-radius: 12px; }
.btn-sm { padding: 8px 16px; font-size: .82rem; border-radius: 8px; }

/* 카드 삽입 */
.card-float { animation: float 2.5s ease-in-out infinite; }
@keyframes float { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-10px)} }

/* 잠금 */
.lock-title { color: var(--red); }

/* 결제수단 */
.pay-method-btn {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 14px 16px;
  background: var(--bg2);
  border: 1.5px solid var(--border);
  border-radius: var(--radius);
  cursor: pointer;
  transition: all .15s;
  text-align: left;
}
.pay-method-btn:hover { border-color: var(--accent); background: #fff7ed; }
.pay-method-icon {
  width: 40px; height: 40px;
  border-radius: 10px;
  background: var(--bg3);
  display: flex; align-items: center; justify-content: center;
  font-size: 1.2rem;
  flex-shrink: 0;
}
.pay-method-name { font-weight: 600; font-size: .9rem; color: var(--text); }
.pay-method-desc { font-size: .75rem; color: var(--muted); margin-top: 1px; }

.divider {
  height: 1px;
  background: var(--border);
  margin: 4px 0;
}

</style>
</head>
<body>

<!-- ── 헤더 ── -->
<div id="header">
  <div class="brand">
    <div class="brand-icon">🤖</div>
    <div>
      <div class="brand-name">AI Cashier</div>
      <div class="brand-sub">Voice Ordering System</div>
    </div>
  </div>

  <div class="hchip">
    <div class="dot dot-active" id="mic-dot"></div>
    <span id="mic-text">마이크</span>
  </div>
  <div class="hchip">
    <div class="dot dot-idle" id="conv-dot"></div>
    <span id="conv-text">대기 중</span>
  </div>
  <div class="hchip">
    <div class="dot dot-pending" id="spk-dot"></div>
    <span id="spk-text">화자인식</span>
  </div>

  <button id="tts-btn" class="hbtn tts-on" onclick="toggleTTS()">
    <span id="tts-icon">🔊</span><span id="tts-label">음성</span>
  </button>
  <button class="hbtn" onclick="resetKiosk()">
    ↩ 처음으로
  </button>
</div>

<!-- ── 메인 ── -->
<div id="main">
<div id="content">

  <!-- 대기 -->
  <div class="screen active" id="screen-waiting">
    <div class="waiting-badge">
      <span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:#15803d"></span>
      시스템 준비 완료
    </div>
    <div class="waiting-icon">🎤</div>
    <div class="waiting-title">안녕하세요!</div>
    <div class="waiting-desc">말씀하시거나 아래 버튼을 눌러 주문을 시작하세요</div>
    <div style="display:flex;gap:10px;">
      <button class="btn btn-primary btn-lg" style="width:200px;" onclick="startOrder()">주문 시작하기</button>
      <button class="btn btn-outline btn-lg" style="width:200px;" onclick="window.open('/app','_blank')">📱 앱 미리보기</button>
    </div>
  </div>

  <!-- 주문 -->
  <div class="screen" id="screen-ordering">
    <!-- 대화 패널 -->
    <div id="conv-panel">
      <div id="conv-header">
        <div class="conv-ai-avatar">🤖</div>
        <div>
          <div class="conv-ai-name">AI 캐셔</div>
          <div class="conv-ai-status" id="conv-ai-status-text">대기 중</div>
        </div>
        <div id="conv-header-badge">대기 중</div>
      </div>
      <div id="chat-log"></div>
      <div id="conv-footer">
        <div id="conv-footer-icon">🤖</div>
        <div id="conv-footer-text">
          <span style="color:var(--muted2);font-size:.85rem;">말씀해 주세요...</span>
        </div>
      </div>
    </div>

    <!-- 메뉴 -->
    <div id="menu-section">
      <div class="section-label">메뉴</div>
      <div class="menu-tabs">
        <div class="menu-tab active" onclick="showCategory('버거')">🍔 버거</div>
        <div class="menu-tab" onclick="showCategory('사이드')">🍟 사이드</div>
        <div class="menu-tab" onclick="showCategory('음료')">🥤 음료</div>
        <div class="menu-tab" onclick="showCategory('세트')">🎁 세트</div>
      </div>
      <div class="menu-grid" id="menu-grid"></div>
    </div>
  </div>

  <!-- 결제 확인 -->
  <div class="screen" id="screen-checkout">
    <div id="checkout-left">
      <div id="checkout-left-header">대화 내용</div>
      <div id="checkout-convo"></div>
    </div>
    <div id="checkout-right">
      <div class="checkout-title">주문 확인</div>
      <div id="checkout-summary"></div>
      <div>
        <div class="pay-section-title">결제 수단</div>
        <div class="pay-buttons">
          <button class="pay-method-btn" onclick="selectPayment('app_card')">
            <div class="pay-method-icon">📱</div>
            <div>
              <div class="pay-method-name">앱 카드</div>
              <div class="pay-method-desc">등록된 앱 카드로 결제</div>
            </div>
          </button>
          <button class="pay-method-btn" onclick="selectPayment('physical_card')">
            <div class="pay-method-icon">💳</div>
            <div>
              <div class="pay-method-name">현장 카드</div>
              <div class="pay-method-desc">카드 단말기에 삽입</div>
            </div>
          </button>
        </div>
      </div>
    </div>
  </div>

  <!-- 결제 처리 -->
  <div class="screen" id="screen-payment_processing">
    <div class="spinner"></div>
    <div class="screen-title">결제 처리 중...</div>
    <div class="screen-desc">잠시만 기다려주세요</div>
  </div>

  <!-- 목소리 저장 질문 -->
  <div class="screen" id="screen-voice_save_prompt">
    <div class="screen-icon">🎙️</div>
    <div class="screen-title">목소리를 저장할까요?</div>
    <div class="screen-desc">다음 방문 시 목소리만으로 바로 주문하실 수 있어요.<br>저장하지 않아도 오늘 주문은 완료됩니다.</div>
    <div style="display:flex;gap:10px;width:100%;max-width:320px;">
      <button class="btn btn-success" onclick="saveVoice(true)">저장할게요</button>
      <button class="btn btn-outline" onclick="saveVoice(false)">괜찮아요</button>
    </div>
  </div>

  <!-- 등록 -->
  <div class="screen" id="screen-register">
    <div class="screen-title" style="margin-bottom:6px;">간단히 등록해주세요</div>
    <div class="screen-desc" style="margin-bottom:24px;">이름과 전화번호만 입력하면 다음부터 목소리로 바로 주문하실 수 있어요.</div>
    <div class="input-group" style="width:100%">
      <label class="input-label">이름</label>
      <input class="kiosk-input" id="reg-name" type="text" placeholder="홍길동" autocomplete="off">
    </div>
    <div class="input-group" style="width:100%">
      <label class="input-label">전화번호</label>
      <input class="kiosk-input" id="reg-phone" type="tel" placeholder="01012345678" autocomplete="off">
    </div>
    <div id="reg-error" style="color:var(--red);font-size:.83rem;margin-bottom:10px;display:none;"></div>
    <button class="btn btn-primary" onclick="submitRegister()" style="max-width:320px;">등록 완료</button>
  </div>

  <!-- 카드 삽입 -->
  <div class="screen" id="screen-card_insert">
    <div class="screen-icon card-float">💳</div>
    <div class="screen-title">카드를 꽂아주세요</div>
    <div class="screen-desc">단말기에 카드를 삽입하면 자동으로 결제됩니다.<br><span style="font-size:.82rem;color:var(--muted2)">잠시 후 자동으로 처리됩니다...</span></div>
    <div class="spinner" style="margin-top:4px;"></div>
  </div>

  <!-- 앱카드 수단 선택 -->
  <div class="screen" id="screen-app_payment">
    <div class="screen-icon">📱</div>
    <div class="screen-title">결제 수단 선택</div>
    <div class="screen-desc">앱에 등록된 결제 수단을 선택해주세요</div>
    <div style="display:flex;flex-direction:column;gap:10px;width:100%;max-width:360px;">
      <button class="pay-method-btn" onclick="confirmAppPayment('신용카드')">
        <div class="pay-method-icon">💳</div>
        <div><div class="pay-method-name">신용카드</div></div>
      </button>
      <button class="pay-method-btn" onclick="confirmAppPayment('체크카드')">
        <div class="pay-method-icon">🏦</div>
        <div><div class="pay-method-name">체크카드</div></div>
      </button>
      <button class="pay-method-btn" onclick="confirmAppPayment('계좌이체')">
        <div class="pay-method-icon">📤</div>
        <div><div class="pay-method-name">계좌이체</div></div>
      </button>
    </div>
    <button class="btn btn-outline btn-sm" style="max-width:200px;margin-top:16px;" onclick="post('/action/checkout',{})">← 결제수단 변경</button>
  </div>

  <!-- 완료 -->
  <div class="screen" id="screen-complete">
    <div class="complete-badge">✓ 결제 완료</div>
    <div class="complete-icon">🎉</div>
    <div class="complete-title" id="complete-title">주문 완료!</div>
    <div class="screen-desc" id="complete-sub" style="margin-bottom:0;">음식이 준비되면 안내드립니다.</div>
  </div>

  <!-- 잠금 -->
  <div class="screen" id="screen-locked">
    <div class="screen-icon">🔒</div>
    <div class="screen-title lock-title">본인 확인 필요</div>
    <div class="screen-desc">처음 말씀하신 분이 맞으신가요?<br>다시 한 번 말씀해 주시면 확인하겠습니다.</div>
    <button class="btn btn-dark" style="max-width:240px;" onclick="retryVerification()">다시 말하기</button>
  </div>

</div><!-- /#content -->

<!-- 장바구니 -->
<div id="cart-panel" class="hidden">
  <div id="cart-header">
    <span>장바구니</span>
    <span class="cart-count" id="cart-count">0</span>
  </div>
  <div id="cart-items">
    <div class="cart-empty">
      <div class="cart-empty-icon">🛒</div>
      아직 담긴 메뉴가 없어요
    </div>
  </div>
  <div id="cart-footer">
    <div class="cart-total-row">
      <span class="cart-total-label">합계</span>
      <span class="cart-total-price" id="cart-total-price">0원</span>
    </div>
    <button class="btn btn-primary" onclick="checkout()">주문하기</button>
  </div>
</div>

</div><!-- /#main -->

<script>
const MENU = {
  "버거": [
    {name:"치즈버거",   price:6500, emoji:"🍔"},
    {name:"더블버거",   price:8500, emoji:"🍔"},
    {name:"베이컨버거", price:7500, emoji:"🥓"},
    {name:"새우버거",   price:7000, emoji:"🦐"},
    {name:"불고기버거", price:7000, emoji:"🥩"},
  ],
  "사이드": [
    {name:"감자튀김",   price:2500, emoji:"🍟"},
    {name:"양파링",     price:3000, emoji:"🧅"},
    {name:"치킨텐더",   price:4500, emoji:"🍗"},
    {name:"코울슬로",   price:2000, emoji:"🥗"},
  ],
  "음료": [
    {name:"콜라",       price:2000, emoji:"🥤"},
    {name:"사이다",     price:2000, emoji:"🥤"},
    {name:"아이스티",   price:2500, emoji:"🧋"},
    {name:"오렌지주스", price:3000, emoji:"🍊"},
    {name:"물",         price:1000, emoji:"💧"},
  ],
  "세트": [
    {name:"치즈버거 세트",   price:9500,  emoji:"🎁"},
    {name:"더블버거 세트",   price:12000, emoji:"🎁"},
    {name:"베이컨버거 세트", price:10500, emoji:"🎁"},
  ],
};

let currentCategory = "버거";
let currentState    = {};
let ttsEnabled      = true;
let _lastLogLen     = 0;
let _lastAiCount    = 0;
let _ttsAudio       = null;  // 현재 재생 중인 Audio 객체

// ── TTS (OpenAI API) ──
function toggleTTS() {
  ttsEnabled = !ttsEnabled;
  const btn  = document.getElementById('tts-btn');
  const icon = document.getElementById('tts-icon');
  const lbl  = document.getElementById('tts-label');
  if (ttsEnabled) {
    btn.classList.add('tts-on');
    icon.textContent = '🔊';
    lbl.textContent  = '음성';
  } else {
    btn.classList.remove('tts-on');
    icon.textContent = '🔇';
    lbl.textContent  = '음성';
    if (_ttsAudio) { _ttsAudio.pause(); _ttsAudio = null; }
  }
}

async function speakText(text) {
  if (!ttsEnabled || !text) return;
  try {
    if (_ttsAudio) { _ttsAudio.pause(); _ttsAudio = null; }
    const url = '/tts?text=' + encodeURIComponent(text);
    const audio = new Audio(url);
    _ttsAudio = audio;
    audio.play().catch(() => {});
    audio.onended = () => { if (_ttsAudio === audio) _ttsAudio = null; };
  } catch(e) { console.warn('TTS 오류:', e); }
}

// ── 화면 전환 ──
function showScreen(name) {
  document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
  const el = document.getElementById('screen-' + name);
  if (el) el.classList.add('active');
  const cartScreens = ['ordering', 'checkout', 'card_insert', 'app_payment'];
  document.getElementById('cart-panel').classList.toggle('hidden', !cartScreens.includes(name));
}

// ── 상태 적용 ──
function applyState(state) {
  currentState = state;
  showScreen(state.screen);
  updateHeader(state);
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

// ── 헤더 상태 ──
function updateHeader(state) {
  const LABELS = {idle:'대기 중', listening:'듣는 중', processing:'생각 중', speaking:'말하는 중'};
  const convDot = document.getElementById('conv-dot');
  convDot.className = 'dot dot-' + (state.conversation || 'idle');
  document.getElementById('conv-text').textContent = LABELS[state.conversation] || '대기 중';

  const spkDot = document.getElementById('spk-dot');
  const spkTxt = document.getElementById('spk-text');
  if (state.speaker_verified === true)    { spkDot.className='dot dot-ok';   spkTxt.textContent='인증됨'; }
  else if (state.speaker_verified===false){ spkDot.className='dot dot-fail'; spkTxt.textContent='불일치'; }
  else                                    { spkDot.className='dot dot-pending'; spkTxt.textContent='인식 중'; }
}

// ── 대화 풋터 ──
function updateConvFooter(state) {
  const icon   = document.getElementById('conv-footer-icon');
  const stream = document.getElementById('conv-footer-text');
  const badge  = document.getElementById('conv-header-badge');
  const status = document.getElementById('conv-ai-status-text');

  if (state.ai_text) {
    icon.textContent = '🤖';
    stream.innerHTML = `<span style="color:var(--accent);font-size:.9rem">${escHtml(state.ai_text)}<span style="display:inline-block;width:2px;height:.9em;background:var(--accent);vertical-align:middle;margin-left:2px;animation:blink .6s infinite"></span></span>`;
    if (badge) badge.textContent = 'AI 응답 중...';
    if (status) status.textContent = '응답 중';
  } else if (state.conversation === 'listening') {
    icon.textContent = '🎤';
    stream.innerHTML = `<div style="display:flex;align-items:center;gap:10px"><div class="mic-wave"><div class="mic-bar"></div><div class="mic-bar"></div><div class="mic-bar"></div><div class="mic-bar"></div><div class="mic-bar"></div></div><span style="color:var(--blue);font-size:.85rem;">듣는 중...</span></div>`;
    if (badge) badge.textContent = '손님 발화 중';
    if (status) status.textContent = '듣는 중';
  } else if (state.conversation === 'processing') {
    icon.textContent = '🤖';
    stream.innerHTML = `<div style="display:flex;align-items:center;gap:10px"><div class="typing-dots"><div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div></div><span style="color:var(--muted);font-size:.85rem;">생각 중...</span></div>`;
    if (badge) badge.textContent = '처리 중...';
    if (status) status.textContent = '처리 중';
  } else {
    icon.textContent = '🤖';
    stream.innerHTML = `<span style="color:var(--muted2);font-size:.85rem;">말씀해 주세요...</span>`;
    if (badge) badge.textContent = '대기 중';
    if (status) status.textContent = '대기 중';
  }
}

// ── 대화 로그 ──
function renderChatLog(log) {
  const aiMsgs = log.filter(m => m.role === 'ai');
  if (aiMsgs.length > _lastAiCount) {
    speakText(aiMsgs[aiMsgs.length - 1].text);
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

  const mainLog     = document.getElementById('chat-log');
  const checkoutLog = document.getElementById('checkout-convo');
  if (mainLog)     { mainLog.innerHTML = html;     mainLog.scrollTop     = mainLog.scrollHeight; }
  if (checkoutLog) { checkoutLog.innerHTML = html; checkoutLog.scrollTop = checkoutLog.scrollHeight; }
}

function escHtml(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// ── 장바구니 ──
function renderCart(items, total) {
  const container = document.getElementById('cart-items');
  const totalEl   = document.getElementById('cart-total-price');
  const countEl   = document.getElementById('cart-count');
  if (!items || items.length === 0) {
    container.innerHTML = '<div class="cart-empty"><div class="cart-empty-icon">🛒</div>아직 담긴 메뉴가 없어요</div>';
    totalEl.textContent  = '0원';
    if (countEl) countEl.textContent = '0';
    return;
  }
  const totalQty = items.reduce((s, i) => s + i.quantity, 0);
  if (countEl) countEl.textContent = String(totalQty);
  container.innerHTML = items.map(item => `
    <div class="cart-item">
      <div class="cart-item-info">
        <div class="cart-item-name">${item.name}</div>
        ${item.includes ? `<div class="cart-item-includes">${item.includes}</div>` : ''}
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
    <div class="summary-header">주문 내역</div>
    <div class="summary-body">
      ${items.map(i => `
        <div class="summary-item">
          <span class="summary-item-name">${i.name} × ${i.quantity}</span>
          <span class="summary-item-price">${(i.price * i.quantity).toLocaleString()}원</span>
        </div>
      `).join('')}
    </div>
    <div class="summary-total">
      <span>총 결제금액</span>
      <span class="summary-total-price">${total.toLocaleString()}원</span>
    </div>
  `;
}

// ── 메뉴 ──
function showCategory(cat) {
  currentCategory = cat;
  document.querySelectorAll('.menu-tab').forEach(t =>
    t.classList.toggle('active', t.textContent.includes(cat.replace('버거','버거').replace('사이드','사이드').replace('음료','음료').replace('세트','세트')))
  );
  renderMenuGrid(cat);
}

function renderMenuGrid(cat) {
  const grid  = document.getElementById('menu-grid');
  const items = MENU[cat] || [];
  grid.innerHTML = items.map(item => `
    <div class="menu-card" onclick="addMenuByClick('${item.name}')">
      <div class="menu-card-emoji">${item.emoji || '🍽'}</div>
      <div class="menu-card-name">${item.name}</div>
      <div class="menu-card-price">${item.price.toLocaleString()}원</div>
    </div>
  `).join('');
}

async function addMenuByClick(name) {
  await post('/action/add_menu', {name});
}

// ── 액션 ──
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
function confirmAppPayment(m)    { post('/action/app_payment_confirm', {method: m}); }

function resetKiosk() {
  if (!confirm('처음으로 돌아가시겠어요?\n주문 내용이 모두 초기화됩니다.')) return;
  if (_ttsAudio) { _ttsAudio.pause(); _ttsAudio = null; }
  post('/action/reset', {});
}

function submitRegister() {
  const name  = document.getElementById('reg-name').value.trim();
  const phone = document.getElementById('reg-phone').value.trim();
  const errEl = document.getElementById('reg-error');
  if (!name)  { errEl.textContent='이름을 입력해주세요.'; errEl.style.display='block'; return; }
  if (!/^01[0-9]{8,9}$/.test(phone)) {
    errEl.textContent='올바른 전화번호를 입력해주세요 (예: 01012345678)';
    errEl.style.display='block'; return;
  }
  errEl.style.display='none';
  post('/action/register', {name, phone});
}

// ── SSE ──
const es = new EventSource('/events');
es.onmessage = e => applyState(JSON.parse(e.data));
es.onerror   = () => { /* 자동 재연결 */ };

// 초기 메뉴 렌더
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
<title>AI Cashier 앱</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --bg: #f8fafc; --card: #ffffff; --accent: #c2410c; --accent2: #ea580c;
    --green: #15803d; --red: #b91c1c; --text: #111827; --muted: #6b7280;
    --border: #e5e7eb; --radius: 18px;
  }
  body {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    min-height: 100vh; display: flex; align-items: center; justify-content: center;
    font-family: -apple-system, 'Apple SD Gothic Neo', 'Noto Sans KR', sans-serif; padding: 24px;
  }
  .phone {
    background: var(--card); border-radius: 40px; width: 390px; min-height: 760px;
    box-shadow: 0 40px 80px rgba(0,0,0,.5); overflow: hidden;
    display: flex; flex-direction: column;
  }
  .phone-bar {
    height: 48px; background: var(--card); display: flex; align-items: center;
    justify-content: center; border-bottom: 1px solid var(--border); flex-shrink: 0; position: relative;
  }
  .phone-notch { width: 110px; height: 26px; background: #0f172a; border-radius: 0 0 16px 16px; }
  .phone-time { position: absolute; left: 20px; font-size: .8rem; font-weight: 700; }
  .app-header { padding: 18px 24px 12px; background: var(--text); color: white; flex-shrink: 0; }
  .app-logo { font-size: 1.2rem; font-weight: 800; letter-spacing: -.3px; }
  .app-tagline { font-size: .73rem; opacity: .6; margin-top: 2px; }
  .steps {
    display: flex; gap: 0; background: var(--text); padding: 0 24px 14px; flex-shrink: 0;
  }
  .step { flex: 1; display: flex; flex-direction: column; align-items: center; cursor: pointer; opacity: .4; transition: opacity .2s; }
  .step.active { opacity: 1; }
  .step-dot {
    width: 26px; height: 26px; border-radius: 50%;
    background: rgba(255,255,255,.15);
    display: flex; align-items: center; justify-content: center;
    font-size: .72rem; font-weight: 700; color: white; margin-bottom: 4px;
  }
  .step.active .step-dot { background: var(--accent); }
  .step.done .step-dot   { background: var(--green); }
  .step-label { font-size: .58rem; color: rgba(255,255,255,.7); text-align: center; }
  .step-line  { flex: 1; height: 1px; background: rgba(255,255,255,.15); margin-top: 13px; }
  .step-line.done { background: var(--green); }
  .app-content { flex: 1; overflow-y: auto; padding: 24px; }
  .panel { display: none; }
  .panel.active { display: block; }
  .panel-title { font-size: 1.25rem; font-weight: 800; color: var(--text); margin-bottom: 6px; }
  .panel-sub   { font-size: .83rem; color: var(--muted); margin-bottom: 22px; line-height: 1.6; }
  .input-group { margin-bottom: 14px; }
  .input-label { font-size: .75rem; color: var(--muted); font-weight: 600; margin-bottom: 5px; display: block; }
  .app-input {
    width: 100%; border: 1.5px solid var(--border); border-radius: 10px;
    padding: 12px 14px; font-size: .9rem; color: var(--text); outline: none;
    transition: border-color .2s; background: var(--bg);
  }
  .app-input:focus { border-color: var(--accent); background: white; }
  .voice-recorder {
    border: 2px dashed var(--border); border-radius: var(--radius);
    padding: 28px 20px; text-align: center; margin-bottom: 14px; transition: all .2s;
  }
  .voice-recorder.recording { border-color: var(--red); background: #fef2f2; }
  .voice-recorder.done      { border-color: var(--green); background: #f0fdf4; }
  .record-icon  { font-size: 2.6rem; margin-bottom: 10px; }
  .record-text  { font-size: .83rem; color: var(--muted); margin-bottom: 14px; }
  .record-bar   { height: 5px; border-radius: 3px; background: var(--border); overflow: hidden; margin: 0 auto 12px; width: 80%; }
  .record-fill  { height: 100%; background: var(--accent); width: 0%; transition: width .1s; }
  .card-preview {
    background: linear-gradient(135deg, #0f172a, #334155);
    border-radius: 14px; padding: 22px; color: white; margin-bottom: 14px; position: relative; overflow: hidden;
  }
  .card-preview::before {
    content: ''; position: absolute; top: -30px; right: -30px;
    width: 110px; height: 110px; border-radius: 50%; background: rgba(255,255,255,.07);
  }
  .card-chip   { font-size: 1.4rem; margin-bottom: 14px; }
  .card-number { font-size: 1rem; letter-spacing: 3px; font-weight: 600; margin-bottom: 7px; }
  .card-name   { font-size: .73rem; opacity: .7; }
  .taste-tags  { display: flex; flex-wrap: wrap; gap: 7px; margin-bottom: 14px; }
  .taste-tag {
    padding: 7px 14px; border-radius: 999px; border: 1.5px solid var(--border);
    font-size: .82rem; cursor: pointer; transition: all .15s; color: var(--muted);
  }
  .taste-tag.selected { background: var(--text); border-color: var(--text); color: white; }
  .complete-wrap { text-align: center; padding: 16px 0; }
  .complete-check { font-size: 3.6rem; margin-bottom: 14px; }
  .complete-name  { font-size: 1.3rem; font-weight: 800; color: var(--text); margin-bottom: 7px; }
  .complete-desc  { font-size: .83rem; color: var(--muted); line-height: 1.7; margin-bottom: 20px; }
  .badge {
    display: inline-block; background: #fff7ed; color: var(--accent);
    border-radius: 999px; padding: 5px 14px; font-size: .75rem; font-weight: 600; margin: 3px;
  }
  .btn-app {
    width: 100%; padding: 13px; border-radius: 12px; border: none;
    font-size: .92rem; font-weight: 700; cursor: pointer; transition: all .15s; margin-top: 7px;
  }
  .btn-app:hover { opacity: .88; transform: translateY(-1px); }
  .btn-primary-app { background: var(--text); color: white; }
  .btn-outline-app { background: transparent; color: var(--text); border: 1.5px solid var(--border); }
  .app-nav {
    display: flex; border-top: 1px solid var(--border);
    background: var(--card); flex-shrink: 0;
  }
  .nav-item {
    flex: 1; display: flex; flex-direction: column; align-items: center;
    padding: 9px; font-size: .62rem; color: var(--muted); cursor: pointer;
  }
  .nav-item.active { color: var(--accent); }
  .nav-icon { font-size: 1.15rem; margin-bottom: 2px; }
  .divider { height: 1px; background: var(--border); margin: 14px 0; }
  .info-box {
    background: #fff7ed; border-radius: 10px; padding: 12px;
    font-size: .8rem; color: var(--accent); margin-bottom: 14px; line-height: 1.6;
  }
</style>
</head>
<body>
<div class="phone">
  <div class="phone-bar">
    <div class="phone-time">9:41</div>
    <div class="phone-notch"></div>
  </div>
  <div class="app-header">
    <div class="app-logo">🤖 AI Cashier</div>
    <div class="app-tagline">목소리로 주문하는 스마트 캐셔</div>
  </div>
  <div class="steps" id="steps">
    <div class="step active" id="step-0" onclick="goStep(0)"><div class="step-dot">1</div><div class="step-label">본인확인</div></div>
    <div class="step-line" id="line-0"></div>
    <div class="step" id="step-1" onclick="goStep(1)"><div class="step-dot">2</div><div class="step-label">목소리</div></div>
    <div class="step-line" id="line-1"></div>
    <div class="step" id="step-2" onclick="goStep(2)"><div class="step-dot">3</div><div class="step-label">카드</div></div>
    <div class="step-line" id="line-2"></div>
    <div class="step" id="step-3" onclick="goStep(3)"><div class="step-dot">4</div><div class="step-label">취향</div></div>
    <div class="step-line" id="line-3"></div>
    <div class="step" id="step-4" onclick="goStep(4)"><div class="step-dot">✓</div><div class="step-label">완료</div></div>
  </div>
  <div class="app-content">
    <div class="panel active" id="panel-0">
      <div class="panel-title">안녕하세요!</div>
      <div class="panel-sub">이름과 전화번호로 간단히 등록하세요.</div>
      <div class="input-group">
        <label class="input-label">이름</label>
        <input class="app-input" id="app-name" type="text" placeholder="홍길동">
      </div>
      <div class="input-group">
        <label class="input-label">전화번호</label>
        <input class="app-input" id="app-phone" type="tel" placeholder="01012345678">
      </div>
      <div class="info-box" id="kiosk-link-box" style="display:none">
        🔗 키오스크 방문 기록을 찾았어요! 연결 후 목소리를 다시 등록하면 더 정확해집니다.
      </div>
      <button class="btn-app btn-primary-app" onclick="step0Next()">다음</button>
    </div>
    <div class="panel" id="panel-1">
      <div class="panel-title">목소리 등록</div>
      <div class="panel-sub">조용한 환경에서 아래 문장을 읽어주세요.</div>
      <div class="voice-recorder" id="voice-recorder">
        <div class="record-icon" id="rec-icon">🎤</div>
        <div class="record-text" id="rec-text">버튼을 눌러 녹음을 시작하세요</div>
        <div style="background:#f1f5f9;border-radius:10px;padding:12px;margin-bottom:14px;font-size:.85rem;color:#334155;line-height:1.7;font-style:italic;">
          "안녕하세요, 저는 AI 캐셔를 이용하고 싶어요. 치즈버거 하나랑 콜라 주세요!"
        </div>
        <div class="record-bar"><div class="record-fill" id="rec-fill"></div></div>
        <button class="btn-app btn-primary-app" id="rec-btn" onclick="toggleRecord()">녹음 시작</button>
      </div>
      <div id="rec-status" style="display:none;text-align:center;color:var(--green);font-weight:700;margin-bottom:10px;">✅ 녹음 완료!</div>
      <button class="btn-app btn-primary-app" id="voice-next-btn" onclick="goStep(2)" style="display:none">다음</button>
      <button class="btn-app btn-outline-app" onclick="goStep(2)">건너뛰기</button>
    </div>
    <div class="panel" id="panel-2">
      <div class="panel-title">카드 등록</div>
      <div class="panel-sub">앱 카드를 등록하면 목소리만으로 결제할 수 있어요.</div>
      <div class="card-preview">
        <div class="card-chip">▣</div>
        <div class="card-number" id="card-num-display">•••• •••• •••• ••••</div>
        <div class="card-name" id="card-name-display">홍 길 동</div>
      </div>
      <div class="input-group">
        <label class="input-label">카드 번호</label>
        <input class="app-input" id="card-num" type="tel" placeholder="0000 0000 0000 0000" oninput="formatCardNum(this)" maxlength="19">
      </div>
      <div style="display:flex;gap:10px;">
        <div class="input-group" style="flex:1">
          <label class="input-label">유효기간</label>
          <input class="app-input" id="card-exp" type="tel" placeholder="MM/YY" maxlength="5" oninput="formatExp(this)">
        </div>
        <div class="input-group" style="flex:1">
          <label class="input-label">CVC</label>
          <input class="app-input" id="card-cvc" type="tel" placeholder="•••" maxlength="3">
        </div>
      </div>
      <button class="btn-app btn-primary-app" onclick="goStep(3)">카드 등록</button>
      <button class="btn-app btn-outline-app" onclick="goStep(3)">건너뛰기</button>
    </div>
    <div class="panel" id="panel-3">
      <div class="panel-title">취향 설정</div>
      <div class="panel-sub">AI 맞춤 추천에 활용됩니다.</div>
      <div class="input-label" style="margin-bottom:8px">선호 카테고리</div>
      <div class="taste-tags">
        <div class="taste-tag" onclick="toggleTag(this)">🍔 버거</div>
        <div class="taste-tag" onclick="toggleTag(this)">🍟 사이드</div>
        <div class="taste-tag" onclick="toggleTag(this)">🥤 음료</div>
        <div class="taste-tag" onclick="toggleTag(this)">🎁 세트</div>
      </div>
      <div class="divider"></div>
      <div class="input-label" style="margin-bottom:8px">알림 설정</div>
      <div class="taste-tags">
        <div class="taste-tag selected" onclick="toggleTag(this)">🔔 주문 완료 알림</div>
        <div class="taste-tag selected" onclick="toggleTag(this)">⭐ 신메뉴 알림</div>
        <div class="taste-tag" onclick="toggleTag(this)">🎫 할인 쿠폰 알림</div>
      </div>
      <button class="btn-app btn-primary-app" style="margin-top:14px" onclick="goStep(4)">완료</button>
    </div>
    <div class="panel" id="panel-4">
      <div class="complete-wrap">
        <div class="complete-check">🎉</div>
        <div class="complete-name" id="complete-name">등록 완료!</div>
        <div class="complete-desc">이제 키오스크에서 말씀만 하시면<br>목소리로 바로 주문이 시작됩니다.<br><br>등록된 기능:</div>
        <div>
          <span class="badge">🎤 목소리 인식</span>
          <span class="badge" id="badge-card">💳 앱 카드</span>
          <span class="badge">⭐ 맞춤 추천</span>
        </div>
        <div class="divider"></div>
        <div style="background:#fff7ed;border-radius:10px;padding:14px;text-align:left;font-size:.82rem;color:var(--accent);line-height:1.7;">
          💡 <strong>다음 키오스크 방문 시</strong><br>말씀하시면 자동으로 인식되어 이름으로 맞이합니다.
        </div>
        <button class="btn-app btn-outline-app" style="margin-top:16px" onclick="window.close()">키오스크로 돌아가기</button>
      </div>
    </div>
  </div>
  <div class="app-nav">
    <div class="nav-item active"><div class="nav-icon">🏠</div>홈</div>
    <div class="nav-item"><div class="nav-icon">📋</div>주문내역</div>
    <div class="nav-item"><div class="nav-icon">👤</div>내 정보</div>
    <div class="nav-item"><div class="nav-icon">⚙️</div>설정</div>
  </div>
</div>
<script>
let currentStep = 0; let isRecording = false; let recInterval = null; let recProgress = 0;
function goStep(n) {
  if (currentStep === 4 && n < 4) return;
  document.querySelectorAll('.panel').forEach((p,i) => p.classList.toggle('active', i === n));
  for (let i = 0; i <= 4; i++) {
    const el = document.getElementById('step-' + i);
    if (!el) continue;
    el.classList.remove('active','done');
    if (i < n) el.classList.add('done'); else if (i === n) el.classList.add('active');
  }
  for (let i = 0; i <= 3; i++) {
    const line = document.getElementById('line-' + i);
    if (line) line.classList.toggle('done', i < n);
  }
  currentStep = n;
  if (n === 4) {
    const name = document.getElementById('app-name').value || '고객';
    document.getElementById('complete-name').textContent = name + '님, 등록 완료!';
    if (!document.getElementById('card-num').value) document.getElementById('badge-card').style.display = 'none';
  }
}
function step0Next() {
  const name = document.getElementById('app-name').value.trim();
  const phone = document.getElementById('app-phone').value.trim();
  if (!name) { alert('이름을 입력해주세요.'); return; }
  if (!/^01[0-9]{8,9}$/.test(phone)) { alert('전화번호를 올바르게 입력해주세요.'); return; }
  goStep(1);
}
function toggleRecord() { isRecording ? stopRecord() : startRecord(); }
function startRecord() {
  isRecording = true; recProgress = 0;
  const recorder = document.getElementById('voice-recorder');
  recorder.classList.add('recording');
  document.getElementById('rec-btn').textContent = '⏹ 중지';
  document.getElementById('rec-icon').textContent = '🔴';
  document.getElementById('rec-text').textContent = '녹음 중... 문장을 읽어주세요';
  recInterval = setInterval(() => {
    recProgress = Math.min(recProgress + 2, 100);
    document.getElementById('rec-fill').style.width = recProgress + '%';
    if (recProgress >= 100) stopRecord();
  }, 100);
}
function stopRecord() {
  isRecording = false; clearInterval(recInterval);
  const recorder = document.getElementById('voice-recorder');
  recorder.classList.remove('recording'); recorder.classList.add('done');
  document.getElementById('rec-btn').textContent = '다시 녹음';
  document.getElementById('rec-icon').textContent = '✅';
  document.getElementById('rec-text').textContent = '목소리 등록 완료!';
  document.getElementById('rec-status').style.display = 'block';
  document.getElementById('voice-next-btn').style.display = 'block';
}
function formatCardNum(input) {
  let v = input.value.replace(/\D/g,'').slice(0,16);
  input.value = v.replace(/(.{4})/g,'$1 ').trim();
  document.getElementById('card-num-display').textContent = v ? v.replace(/(.{4})/g,'$1 ').trim() : '•••• •••• •••• ••••';
}
function formatExp(input) {
  let v = input.value.replace(/\D/g,'');
  if (v.length >= 2) v = v.slice(0,2) + '/' + v.slice(2,4);
  input.value = v;
}
function toggleTag(el) { el.classList.toggle('selected'); }
document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('app-name').addEventListener('input', e => {
    document.getElementById('card-name-display').textContent = e.target.value ? e.target.value.split('').join(' ') : '홍 길 동';
  });
});
</script>
</body>
</html>"""
