"""
장바구니 관리.

CartManager 인스턴스를 세션마다 하나씩 생성해 사용.
add/remove/view/clear + 합계 계산.
"""

from typing import Optional
from src.tools.menu import find_item


class CartManager:
    def __init__(self):
        # [{"name": str, "price": int, "quantity": int, "category": str}]
        self.items: list[dict] = []

    def add_item(self, item_name: str, quantity: int = 1) -> dict:
        """
        장바구니에 메뉴 추가.
        이미 있으면 수량만 증가. 메뉴에 없으면 error 반환.
        """
        menu_item = find_item(item_name)
        if not menu_item:
            return {"success": False, "error": f"'{item_name}'을(를) 메뉴에서 찾을 수 없습니다."}

        for cart_item in self.items:
            if cart_item["name"] == menu_item["name"]:
                cart_item["quantity"] += quantity
                return {"success": True, "action": "updated", "item": cart_item, "cart": self.to_dict()}

        new_item = {
            "name": menu_item["name"],
            "price": menu_item["price"],
            "quantity": quantity,
            "category": menu_item["category"],
        }
        self.items.append(new_item)
        return {"success": True, "action": "added", "item": new_item, "cart": self.to_dict()}

    def remove_item(self, item_name: str) -> dict:
        """장바구니에서 메뉴 제거."""
        for i, item in enumerate(self.items):
            if item_name in item["name"] or item["name"] in item_name:
                removed = self.items.pop(i)
                return {"success": True, "removed": removed["name"], "cart": self.to_dict()}
        return {"success": False, "error": f"장바구니에 '{item_name}'이(가) 없습니다."}

    def view(self) -> dict:
        """현재 장바구니 전체 조회."""
        return self.to_dict()

    def clear(self):
        self.items = []

    def to_dict(self) -> dict:
        return {
            "items": list(self.items),
            "total": self.total,
            "count": sum(i["quantity"] for i in self.items),
            "is_empty": self.is_empty,
        }

    @property
    def total(self) -> int:
        return sum(item["price"] * item["quantity"] for item in self.items)

    @property
    def is_empty(self) -> bool:
        return len(self.items) == 0
