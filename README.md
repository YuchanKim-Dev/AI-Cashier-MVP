# Voice AI Cashier / 음성 AI 캐셔 키오스크

손님 목소리로 주문·결제하는 AI 캐셔. **두 모델이 동시에** 동작한다.

## 구현 현황

### ✅ 완료된 기능 (Phase 1 + 2)

| 기능 | 설명 |
|------|------|
| 음성 주문 | 마이크 → Realtime API → 텍스트+TTS 응답 |
| 대화 로그 | 손님/AI 말풍선 실시간 표시 (순서 보장) |
| 장바구니 | 음성 function calling으로 추가/삭제/조회 |
| 세트 메뉴 | 포함 구성 항목 표시 (감자튀김+음료 등) |
| 결제 | 현장카드(카드 삽입 화면 3초 mock) / 앱카드(수단 선택) |
| 사용자 등록 | 이름+전화번호 → JSON 파일 영구 저장 |
| 화자인식 | ECAPA-TDNN 목소리 임베딩 비교 — 등록 사용자 자동 인식 + 이름 인사 |
| 재방문 인식 | 등록 사용자 JSON 로드, 다음 발화에 이름으로 인사 |
| 처음으로 | 세션 전체 초기화 버튼 |
| 주문 완료 후 | 완료·결제·등록 화면에서 마이크 전송 중단 |
| TTS | OpenAI TTS API (tts-1-hd / nova 보이스) — 자연스러운 한국어 음성 |
| 화면 | 대기→주문→결제(대화내용 포함)→처리→완료 전체 흐름 |

### 🔜 예정 (Phase 3)
- ECAPA-TDNN 화자인증 (PostgreSQL + pgvector)
- 실제 사용자 목소리 임베딩 저장·매칭

---

## 아키텍처

```
                         ┌─ [화자인증] ECAPA-TDNN → pgvector 매칭 (Phase 3)
   마이크(24kHz PCM) ─분기┤
                         └─ [Realtime API] WebSocket → function calling → 텍스트 응답
                                                   ↓
                                        키오스크 화면 (FastAPI + HTML/JS)
                                        브라우저 TTS (Web Speech API)
```

## 사용 기술

| 영역 | 기술 |
|------|------|
| 대화 엔진 | OpenAI Realtime API `gpt-realtime-2` (GA, text output) |
| 캐셔 로직 | Realtime function calling |
| 화자인증 | SpeechBrain ECAPA-TDNN (Phase 3) |
| 임베딩 저장 | PostgreSQL + pgvector (Phase 3) |
| 오디오 입력 | sounddevice (PortAudio) 24kHz mono |
| 음성 출력 | OpenAI TTS API (`tts-1-hd`, nova 보이스) |
| 화자인식 | SpeechBrain ECAPA-TDNN (목소리 임베딩 비교) |
| 프론트엔드 | FastAPI + SSE + HTML/JS |
| 사용자 DB | JSON 파일 (Phase 2 mock) → PostgreSQL (Phase 3) |

## 변경 이력

| 날짜 | 내용 |
|------|------|
| 2026-06-20 | GA Realtime API 적용 (output_modalities text, 24kHz, session.audio 구조) |
| 2026-06-20 | 전체 Phase 2 구현 — 장바구니·결제·화면 전체 |
| 2026-06-20 | 브라우저 TTS, 대화 로그 UI, 결제 화면 대화 내용 표시 |
| 2026-06-20 | 처음으로 버튼, 현장/앱카드 결제 화면 분리 |
| 2026-06-20 | 밝은 매장 테마 (크림/오렌지-레드) |
| 2026-06-21 | 다중 function call 동시 호출 버그 수정 (response.create 중복 방지) |
| 2026-06-21 | 사용자 등록 JSON 영구 저장 + 재시작 시 자동 인식 |
| 2026-06-21 | 주문 완료 후 마이크 전송 중단 (complete·register·card_insert 등) |
| 2026-06-21 | 대화 순서 보장 (손님 발화 → AI 응답) |
| 2026-06-21 | 화자인식 구현 — ECAPA-TDNN 목소리 임베딩 비교, 등록 시 임베딩 저장 |
| 2026-06-21 | TTS 교체 — Web Speech API → OpenAI TTS API (tts-1-hd/nova, 훨씬 자연스러움) |
| 2026-06-21 | 프리미엄 UI 리디자인 — 다크 헤더 + 클린 라이트 콘텐츠, 카드형 메뉴 |

## 디렉터리 구조

```
mvp-ai_Cashier/
├── .env                        # API 키 (gitignore)
├── requirements.txt
├── data/
│   └── users.json              # 등록 사용자 DB (gitignore)
├── src/
│   ├── audio/                  # 마이크 캡처(capture.py) / 재생(playback.py)
│   ├── realtime/               # Realtime API WebSocket 클라이언트
│   ├── tools/
│   │   ├── menu.py             # 메뉴 데이터
│   │   ├── cart.py             # 장바구니
│   │   ├── payment.py          # 결제 게이트웨이 (mock)
│   │   ├── handlers.py         # function call 핸들러
│   │   └── user_store.py       # 사용자 JSON 저장소
│   ├── frontend/               # 키오스크 화면 (FastAPI + HTML/JS)
│   └── orchestrator/           # 메인 파이프라인 통합
└── tests/                      # 단위 테스트 (54개)
```

## 실행

```bash
# 가상환경 활성화
source .venv/bin/activate

# .env에 OPENAI_API_KEY 설정 후
python -m src.orchestrator.main
```

브라우저에서 `http://localhost:8000` 접속

## 필요 환경변수

```
OPENAI_API_KEY=sk-...
OPENAI_REALTIME_MODEL=gpt-realtime-2   # 기본값
OPENAI_REALTIME_VOICE=alloy            # 기본값
```
