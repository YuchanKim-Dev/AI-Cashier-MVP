"""
메뉴 데이터 및 recommend_menu function call 핸들러.

MVP: 하드코딩 메뉴. 추후 DB로 교체 가능하도록 MENU_BY_NAME 딕셔너리로 단순화.
"""

from typing import Optional

MENU_DATA: dict = {
    "버거": [
        {"id": "b1", "name": "치즈버거",   "price": 6500},
        {"id": "b2", "name": "더블버거",   "price": 8500},
        {"id": "b3", "name": "베이컨버거", "price": 7500},
        {"id": "b4", "name": "새우버거",   "price": 7000},
        {"id": "b5", "name": "불고기버거", "price": 7000},
    ],
    "사이드": [
        {"id": "s1", "name": "감자튀김",   "price": 2500},
        {"id": "s2", "name": "양파링",     "price": 3000},
        {"id": "s3", "name": "치킨텐더",   "price": 4500},
        {"id": "s4", "name": "코울슬로",   "price": 2000},
    ],
    "음료": [
        {"id": "d1", "name": "콜라",       "price": 2000},
        {"id": "d2", "name": "사이다",     "price": 2000},
        {"id": "d3", "name": "아이스티",   "price": 2500},
        {"id": "d4", "name": "오렌지주스", "price": 3000},
        {"id": "d5", "name": "물",         "price": 1000},
    ],
    "세트": [
        {"id": "set1", "name": "치즈버거 세트",  "price": 9500,  "includes": "치즈버거+감자튀김+음료"},
        {"id": "set2", "name": "더블버거 세트",  "price": 12000, "includes": "더블버거+감자튀김+음료"},
        {"id": "set3", "name": "베이컨버거 세트","price": 10500, "includes": "베이컨버거+감자튀김+음료"},
    ],
}

# 이름으로 빠르게 찾기 위한 플랫 맵 (add_to_cart 에서 사용)
MENU_BY_NAME: dict = {
    item["name"]: {**item, "category": cat}
    for cat, items in MENU_DATA.items()
    for item in items
}


def recommend_menu(category: Optional[str] = None) -> dict:
    """
    메뉴 추천 function call 핸들러.
    category 없으면 전체, 있으면 해당 카테고리만 반환.
    """
    if category and category in MENU_DATA:
        return {"category": category, "items": MENU_DATA[category]}
    return {"all": MENU_DATA}


def find_item(name: str) -> Optional[dict]:
    """메뉴 이름으로 아이템 조회. 없으면 None."""
    # 정확히 일치 우선, 없으면 부분 일치
    if name in MENU_BY_NAME:
        return MENU_BY_NAME[name]
    for menu_name, item in MENU_BY_NAME.items():
        if name in menu_name or menu_name in name:
            return item
    return None
