"""
사용자 정보 파일 저장소 (Phase 2 mock).
Phase 3에서 PostgreSQL + pgvector로 교체 예정.
"""

import json
import os

_DB_PATH = os.path.join(os.path.dirname(__file__), "../../data/users.json")


def _load() -> list[dict]:
    try:
        with open(_DB_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def _save(users: list[dict]):
    os.makedirs(os.path.dirname(_DB_PATH), exist_ok=True)
    with open(_DB_PATH, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)


def save_user(name: str, phone: str, embedding: list = None):
    """사용자 등록 (이미 있으면 업데이트)."""
    users = _load()
    for u in users:
        if u["phone"] == phone:
            u["name"] = name
            if embedding is not None:
                u["embedding"] = embedding
            _save(users)
            return
    user: dict = {"name": name, "phone": phone}
    if embedding is not None:
        user["embedding"] = embedding
    users.append(user)
    _save(users)


def get_all_users() -> list[dict]:
    return _load()


def get_first_user() -> dict | None:
    """데모용: 가장 마지막으로 등록한 사용자 반환."""
    users = _load()
    return users[-1] if users else None
