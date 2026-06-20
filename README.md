# Voice AI Cashier / 음성 AI 캐셔 키오스크

손님 목소리로 주문·결제하는 AI 캐셔. **두 모델이 동시에** 동작한다.

## 아키텍처

```
                         ┌─ [화자인증] raw PCM 누적 → ECAPA 임베딩 → pgvector 매칭
   마이크(16kHz PCM) ─분기┤   (비동기/백그라운드. 1초 응답 경로에 영향 없음. 세션 내내 지속)
                         └─ [Realtime API] WebSocket 스트리밍 → function calling → 음성 출력
                                                   ↓
                                        키오스크 화면 (FastAPI + HTML/JS)
```

- **화자인증**은 응답 지연 경로 밖에서 비동기로 돌며 세션 중 지속 검증
- **대화 엔진**은 WebSocket으로 음성을 실시간 스트리밍해 1초 이내 응답

## 사용 기술과 선택 이유

| 영역 | 기술 | 선택 이유 |
|------|------|-----------|
| 대화 엔진 | OpenAI Realtime API (`gpt-realtime-2`) | 음성-투-음성 통합. STT→LLM→TTS 직렬 조합으로는 1초 응답 불가 |
| 캐셔 로직 | Realtime function calling | 모델이 맥락에 맞게 메뉴/장바구니/결제 tool 호출 |
| 화자인증 | SpeechBrain ECAPA-TDNN | 192차원 임베딩, 8GB 맥에서 단독 구동 가능 |
| 임베딩 저장 | PostgreSQL + pgvector | 코사인 유사도 검색. 생체정보를 클라우드에 위탁하지 않음 |
| 오디오 I/O | sounddevice (PortAudio) | 마이크 raw PCM(16kHz mono) 캡처 후 두 갈래로 분기 |
| 프론트엔드 | FastAPI + HTML/JS | 키오스크 화면. Python 단일 스택으로 복잡도 최소화 |
| 언어 | Python 3.11 | 전 영역을 단일 스택으로 통일 |

## 화자인증 보안 로직

- 대화 시작 시 1회 등록/식별 → 세션 주인(session owner) 고정
- 매 N초마다 들어오는 음성이 세션 주인과 코사인 유사도 임계값 이상인지 지속 검증
- 다른 사람으로 판정되면 결제 등 민감 동작 즉시 잠금 + UI 차단
- 연속 N회 불일치 시 세션 종료/재인증 요구
- 파라미터(임계값/주기/허용 횟수)는 `.env`에서 조정

## 디렉터리 구조

```
voice-cashier/
├── .env.example              # 필요한 환경변수 목록
├── requirements.txt
├── docs/
│   └── history_for_newsession.md  # 세션 인계 파일 (gitignore)
├── src/
│   ├── audio/                # 마이크 캡처/재생/PCM 분기
│   ├── realtime/             # Realtime API WebSocket 클라이언트·이벤트
│   ├── tools/                # function calling 핸들러 (메뉴/장바구니/결제)
│   ├── speaker/              # ECAPA 임베딩, pgvector 식별/지속검증
│   ├── frontend/             # 키오스크 화면 (FastAPI + HTML/JS)
│   └── orchestrator/         # 두 파이프라인 통합
└── tests/                    # 단위 테스트
```

## 설치 및 실행 (VS Code 기준)

### 사전 준비

1. Python 3.11 설치 확인
2. PortAudio 설치: `brew install portaudio`
3. PostgreSQL 설치: `brew install postgresql` (3단계부터 필요)

### 환경 설정

```bash
# 1. Python 가상환경 생성
python3.11 -m venv .venv
source .venv/bin/activate

# 2. 의존성 설치
pip install -r requirements.txt

# 3. 환경변수 설정
cp .env.example .env
# .env 파일을 열어 OPENAI_API_KEY 등 입력
```

### 실행

```bash
# (개발 중 — 단계별로 업데이트 예정)
python -m src.orchestrator.main
```

## 필요한 API / 계정

| 서비스 | 용도 | 발급 경로 |
|--------|------|-----------|
| OpenAI API Key | Realtime API 사용 | platform.openai.com |
| HuggingFace Token | SpeechBrain 모델 다운로드 (공개 모델이면 불필요) | huggingface.co |

## 단계별 진행 상황

- [ ] **1단계**: 마이크 → Realtime → 음성 응답 + 최소 화면 (`feature/realtime-voice`)
- [ ] **2단계**: function calling — 캐셔 로직 + 화면 연동 (`feature/cart-checkout`)
- [ ] **3단계**: 화자인증 지속 검증 (`feature/speaker-verification`)
- [ ] **4단계**: 오케스트레이터 통합 (`feature/orchestrator`)

## Git 컨벤션

- 브랜치: `main` ← `dev` ← `feature/<기능>`
- 커밋: Conventional Commits — `feat(scope): 설명`
  - 타입: `feat / fix / docs / refactor / test / chore / style`
  - 본문에 무엇을 왜 했는지 기록
