from __future__ import annotations

import json
import secrets
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
import hashlib


# Simple file-backed user store for local usage
USERS_PATH = Path("data/users.json")


def _ensure_store() -> None:
    USERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not USERS_PATH.exists():
        USERS_PATH.write_text(json.dumps({}), encoding="utf-8")


def _load_users() -> Dict[str, dict]:
    _ensure_store()
    try:
        return json.loads(USERS_PATH.read_text(encoding="utf-8"))
    except Exception:
        # If file is corrupted, reset to empty for local dev usage
        return {}


def _save_users(users: Dict[str, dict]) -> None:
    _ensure_store()
    USERS_PATH.write_text(json.dumps(users, indent=2), encoding="utf-8")


def _hash_password(password: str, salt: Optional[str] = None) -> Tuple[str, str]:
    if salt is None:
        salt = secrets.token_hex(16)
    # PBKDF2 with SHA256, 100k iterations (sufficient for local dev)
    dk = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), bytes.fromhex(salt), 100_000
    )
    return dk.hex(), salt


def create_user(username: str, password: str) -> dict:
    username = username.strip().lower()
    if not username or not password:
        raise ValueError("Username and password are required")

    users = _load_users()
    if username in users:
        raise ValueError("User already exists")

    pwd_hash, salt = _hash_password(password)
    users[username] = {
        "username": username,
        "password_hash": pwd_hash,
        "salt": salt,
        "api_key": None,
        "created_at": int(time.time()),
        "last_login_at": None,
    }
    _save_users(users)
    return {k: v for k, v in users[username].items() if k != "password_hash" and k != "salt"}


def authenticate(username: str, password: str) -> Optional[dict]:
    username = username.strip().lower()
    users = _load_users()
    user = users.get(username)
    if not user:
        return None
    calc_hash, _ = _hash_password(password, user.get("salt"))
    if secrets.compare_digest(calc_hash, user.get("password_hash", "")):
        return user
    return None


def generate_api_key() -> str:
    # 32 bytes -> urlsafe ~43 chars
    return secrets.token_urlsafe(32)


def set_user_api_key(username: str, api_key: str) -> dict:
    username = username.strip().lower()
    users = _load_users()
    if username not in users:
        raise ValueError("User not found")
    users[username]["api_key"] = api_key
    users[username]["last_login_at"] = int(time.time())
    _save_users(users)
    return {k: v for k, v in users[username].items() if k != "password_hash" and k != "salt"}


def get_user_by_api_key(api_key: str) -> Optional[dict]:
    if not api_key:
        return None
    users = _load_users()
    for u in users.values():
        if u.get("api_key") == api_key:
            return u
    return None


def get_public_user(username: str) -> Optional[dict]:
    username = username.strip().lower()
    users = _load_users()
    u = users.get(username)
    if not u:
        return None
    return {k: v for k, v in u.items() if k not in {"password_hash", "salt"}}

