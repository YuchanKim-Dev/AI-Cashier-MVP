"""
Realtime API function call 디스패처.

모델이 function calling 이벤트를 보내면 이 모듈이 받아서
적절한 핸들러(menu/cart/payment)를 호출하고 결과를 반환한다.
"""

import json
from typing import Callable, Awaitable

from src.tools.menu import recommend_menu
from src.tools.cart import CartManager
from src.orchestrator.session import SessionState

# Realtime API에 등록할 tool 정의 목록
TOOLS = [
    {
        "type": "function",
        "name": "recommend_menu",
        "description": "메뉴를 추천하거나 카테고리별 메뉴를 안내합니다. 손님이 뭘 먹을지 모를 때나 메뉴를 물어볼 때 호출합니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": ["버거", "사이드", "음료", "세트"],
                    "description": "보여줄 메뉴 카테고리. 없으면 전체 메뉴 반환.",
                }
            },
        },
    },
    {
        "type": "function",
        "name": "add_to_cart",
        "description": "손님이 주문한 메뉴를 장바구니에 추가합니다. 메뉴명은 정확하게 전달하세요.",
        "parameters": {
            "type": "object",
            "properties": {
                "item_name": {"type": "string", "description": "추가할 메뉴 이름 (예: 치즈버거)"},
                "quantity": {"type": "integer", "description": "수량. 기본값 1.", "default": 1},
            },
            "required": ["item_name"],
        },
    },
    {
        "type": "function",
        "name": "remove_from_cart",
        "description": "손님이 취소 요청한 메뉴를 장바구니에서 제거합니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "item_name": {"type": "string", "description": "제거할 메뉴 이름"},
            },
            "required": ["item_name"],
        },
    },
    {
        "type": "function",
        "name": "view_cart",
        "description": "현재 장바구니 내용을 확인합니다. 손님이 주문 내역을 물어볼 때 호출합니다.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "type": "function",
        "name": "checkout",
        "description": "결제를 시작합니다. 손님이 주문을 끝내고 결제하겠다고 할 때 호출합니다. 장바구니가 비어있으면 호출하지 마세요.",
        "parameters": {"type": "object", "properties": {}},
    },
]


class FunctionCallHandler:
    """
    Realtime API function_call_arguments.done 이벤트를 받아
    실제 함수를 실행하고 결과를 반환하는 클래스.
    """

    def __init__(
        self,
        cart: CartManager,
        session: SessionState,
        on_checkout: Callable[[], Awaitable[None]],
    ):
        self.cart = cart
        self.session = session
        # checkout은 결제 화면 전환이 필요해서 orchestrator 콜백으로 처리
        self.on_checkout = on_checkout

    async def handle(self, call_id: str, name: str, arguments_json: str) -> str:
        """
        function call을 실행하고 JSON 문자열 결과를 반환.
        Realtime API는 function_call_output.output을 string으로 받는다.
        """
        try:
            args = json.loads(arguments_json) if arguments_json.strip() else {}
        except json.JSONDecodeError:
            args = {}

        if name == "recommend_menu":
            result = recommend_menu(args.get("category"))

        elif name == "add_to_cart":
            result = self.cart.add_item(
                item_name=args.get("item_name", ""),
                quantity=args.get("quantity", 1),
            )
            if result.get("success"):
                # 세션 상태의 장바구니 미러 업데이트 → SSE로 화면 반영
                self.session.cart_items = result["cart"]["items"]
                self.session.cart_total = result["cart"]["total"]
                if self.session.screen == "waiting":
                    self.session.screen = "ordering"

        elif name == "remove_from_cart":
            result = self.cart.remove_item(args.get("item_name", ""))
            if result.get("success"):
                self.session.cart_items = result["cart"]["items"]
                self.session.cart_total = result["cart"]["total"]

        elif name == "view_cart":
            result = self.cart.view()

        elif name == "checkout":
            if self.cart.is_empty:
                result = {"success": False, "error": "장바구니가 비어있습니다."}
            else:
                result = {"success": True, "message": "결제 화면으로 이동합니다.", "cart": self.cart.view()}
                # 결제 화면 전환은 orchestrator 콜백에서 처리
                await self.on_checkout()

        else:
            result = {"error": f"알 수 없는 함수: {name}"}

        return json.dumps(result, ensure_ascii=False)
