"""
결제 처리 모듈.

PG사 연동을 고려한 추상화 레이어.
MVP: MockPaymentGateway (결제 된척).
실제 PG 연동 시 PaymentGateway를 상속해 구현체만 교체하면 된다.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Optional


class PaymentGateway(ABC):
    """PG사 연동 추상 인터페이스. 실제 연동 시 이 클래스를 상속."""

    @abstractmethod
    async def process(self, amount: int, method: str, user_id: Optional[str] = None) -> dict:
        """
        결제 처리.
        Returns: {"success": bool, "transaction_id": str, "amount": int, "error": str|None}
        """
        pass

    @abstractmethod
    async def refund(self, transaction_id: str) -> dict:
        pass


class MockPaymentGateway(PaymentGateway):
    """
    MVP용 결제 된척 구현체.
    실제 PG 연동 전까지 이 클래스를 사용.
    PG 연동 시 이 파일만 수정하면 orchestrator는 변경 없음.
    """

    _counter = 0  # 트랜잭션 ID 생성용

    async def process(self, amount: int, method: str, user_id: Optional[str] = None) -> dict:
        # 실제 PG라면 여기서 HTTP 요청. MVP는 1.5초 딜레이로 흉내.
        await asyncio.sleep(1.5)
        MockPaymentGateway._counter += 1
        return {
            "success": True,
            "transaction_id": f"MOCK-{MockPaymentGateway._counter:04d}",
            "amount": amount,
            "method": method,
            "error": None,
        }

    async def refund(self, transaction_id: str) -> dict:
        await asyncio.sleep(0.5)
        return {"success": True, "transaction_id": transaction_id}


# 싱글턴 — orchestrator에서 import해서 사용
payment_gateway: PaymentGateway = MockPaymentGateway()
