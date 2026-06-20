"""
세션 상태 관리.

하나의 손님 세션 동안의 모든 상태를 보관.
orchestrator와 frontend가 이 객체를 공유한다.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SessionState:
    # 화면 전환 상태 — frontend가 이 값으로 어떤 화면을 보여줄지 결정
    # waiting|ordering|checkout|payment_processing|voice_save_prompt|register|complete|locked
    screen: str = "waiting"

    # 대화 상태
    conversation: str = "idle"       # idle|listening|processing|speaking
    mic: str = "active"
    ai_text: str = ""                # AI 현재 발화 텍스트 (실시간 스트리밍)
    user_text: str = ""              # 사용자 마지막 발화 전사
    conversation_log: list = field(default_factory=list)  # [{role, text}] 전체 대화 이력

    # 사용자 정보
    user_name: Optional[str] = None  # 인식된/입력된 이름 (None = 신규)
    is_new_user: bool = True         # False = DB에서 찾은 기존 사용자
    speaker_verified: Optional[bool] = None  # None|True|False (3단계에서 실제 동작)
    failed_verifications: int = 0    # 연속 화자 불일치 횟수

    # 장바구니 (frontend에 실시간 전달용 — 실제 데이터는 CartManager)
    cart_items: list = field(default_factory=list)
    cart_total: int = 0

    # 결제
    payment_method: Optional[str] = None  # app_card|physical_card
    transaction_id: Optional[str] = None

    # 목소리 누적 (3초 이상이면 저장 프롬프트 표시)
    voice_duration: float = 0.0      # 누적 발화 시간(초)
    _speech_start_time: Optional[float] = None  # 내부용

    def to_dict(self) -> dict:
        """SSE로 frontend에 전달할 직렬화 가능한 dict."""
        return {
            "screen": self.screen,
            "conversation": self.conversation,
            "mic": self.mic,
            "ai_text": self.ai_text,
            "user_text": self.user_text,
            "conversation_log": self.conversation_log,
            "user_name": self.user_name,
            "is_new_user": self.is_new_user,
            "speaker_verified": self.speaker_verified,
            "failed_verifications": self.failed_verifications,
            "cart_items": self.cart_items,
            "cart_total": self.cart_total,
            "payment_method": self.payment_method,
            "transaction_id": self.transaction_id,
            "voice_duration": round(self.voice_duration, 1),
        }

    def on_speech_start(self, timestamp: float):
        """speech_started 이벤트 시각 기록."""
        self._speech_start_time = timestamp

    def on_speech_end(self, timestamp: float):
        """speech_stopped 이벤트로 누적 발화 시간 업데이트."""
        if self._speech_start_time is not None:
            self.voice_duration += timestamp - self._speech_start_time
            self._speech_start_time = None

    @property
    def enough_voice(self) -> bool:
        """저장 가능한 발화 길이(3초 이상)인지 확인."""
        return self.voice_duration >= 3.0

    def reset(self):
        """새 손님 세션 시작 시 초기화."""
        self.__init__()
