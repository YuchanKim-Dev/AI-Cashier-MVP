"""
tools 모듈 단위 테스트 — menu, cart, payment, handlers.
"""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.tools.menu import recommend_menu, find_item, MENU_DATA
from src.tools.cart import CartManager
from src.tools.payment import MockPaymentGateway
from src.tools.handlers import FunctionCallHandler
from src.orchestrator.session import SessionState


# ── 메뉴 ──────────────────────────────────────────────────────────────────────

class TestMenu:
    def test_recommend_all(self):
        result = recommend_menu()
        assert "all" in result
        assert "버거" in result["all"]

    def test_recommend_category(self):
        result = recommend_menu("버거")
        assert result["category"] == "버거"
        assert len(result["items"]) > 0

    def test_find_exact(self):
        item = find_item("치즈버거")
        assert item is not None
        assert item["price"] == 6500

    def test_find_partial(self):
        item = find_item("치즈")
        assert item is not None

    def test_find_not_found(self):
        assert find_item("존재하지않는메뉴") is None


# ── 장바구니 ───────────────────────────────────────────────────────────────────

class TestCart:
    def setup_method(self):
        self.cart = CartManager()

    def test_add_item(self):
        result = self.cart.add_item("치즈버거")
        assert result["success"] is True
        assert self.cart.total == 6500

    def test_add_same_item_increments_quantity(self):
        self.cart.add_item("치즈버거")
        self.cart.add_item("치즈버거")
        assert self.cart.items[0]["quantity"] == 2
        assert self.cart.total == 13000

    def test_add_with_quantity(self):
        self.cart.add_item("콜라", quantity=3)
        assert self.cart.total == 6000

    def test_add_unknown_item(self):
        result = self.cart.add_item("없는메뉴")
        assert result["success"] is False

    def test_remove_item(self):
        self.cart.add_item("치즈버거")
        result = self.cart.remove_item("치즈버거")
        assert result["success"] is True
        assert self.cart.is_empty

    def test_remove_not_in_cart(self):
        result = self.cart.remove_item("없는메뉴")
        assert result["success"] is False

    def test_view(self):
        self.cart.add_item("치즈버거")
        self.cart.add_item("콜라")
        view = self.cart.view()
        assert view["count"] == 2
        assert view["total"] == 8500

    def test_clear(self):
        self.cart.add_item("치즈버거")
        self.cart.clear()
        assert self.cart.is_empty


# ── 결제 ──────────────────────────────────────────────────────────────────────

class TestPayment:
    @pytest.mark.asyncio
    async def test_mock_payment_success(self):
        gw = MockPaymentGateway()
        result = await gw.process(amount=10000, method="physical_card")
        assert result["success"] is True
        assert result["amount"] == 10000
        assert "MOCK" in result["transaction_id"]

    @pytest.mark.asyncio
    async def test_mock_refund(self):
        gw = MockPaymentGateway()
        result = await gw.refund("MOCK-0001")
        assert result["success"] is True


# ── function call 핸들러 ───────────────────────────────────────────────────────

class TestFunctionCallHandler:
    def setup_method(self):
        self.cart    = CartManager()
        self.session = SessionState()
        self.checkout_called = False

        async def on_checkout():
            self.checkout_called = True

        self.handler = FunctionCallHandler(
            cart=self.cart,
            session=self.session,
            on_checkout=on_checkout,
        )

    @pytest.mark.asyncio
    async def test_add_to_cart(self):
        result_json = await self.handler.handle("id1", "add_to_cart", '{"item_name":"치즈버거"}')
        result = json.loads(result_json)
        assert result["success"] is True
        assert self.cart.total == 6500
        # 화면이 ordering으로 전환됐는지 확인
        assert self.session.screen == "ordering"

    @pytest.mark.asyncio
    async def test_recommend_menu(self):
        result_json = await self.handler.handle("id2", "recommend_menu", '{"category":"버거"}')
        result = json.loads(result_json)
        assert result["category"] == "버거"

    @pytest.mark.asyncio
    async def test_view_cart(self):
        self.cart.add_item("콜라")
        result_json = await self.handler.handle("id3", "view_cart", '{}')
        result = json.loads(result_json)
        assert result["total"] == 2000

    @pytest.mark.asyncio
    async def test_remove_from_cart(self):
        self.cart.add_item("치즈버거")
        result_json = await self.handler.handle("id4", "remove_from_cart", '{"item_name":"치즈버거"}')
        result = json.loads(result_json)
        assert result["success"] is True
        assert self.cart.is_empty

    @pytest.mark.asyncio
    async def test_checkout_empty_cart(self):
        result_json = await self.handler.handle("id5", "checkout", '{}')
        result = json.loads(result_json)
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_checkout_with_items(self):
        self.cart.add_item("치즈버거")
        result_json = await self.handler.handle("id6", "checkout", '{}')
        result = json.loads(result_json)
        assert result["success"] is True
        assert self.checkout_called is True

    @pytest.mark.asyncio
    async def test_unknown_function(self):
        result_json = await self.handler.handle("id7", "unknown_fn", '{}')
        result = json.loads(result_json)
        assert "error" in result
