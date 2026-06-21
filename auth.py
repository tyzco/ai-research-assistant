"""JWT 用户认证 + 多租户隔离模块。

用法：
    from auth import create_token, get_current_user, hash_password, verify_password

    # 注册
    hashed = hash_password("mypassword")
    users_db[user_id] = {"password": hashed, ...}

    # 登录
    token = create_token(user_id)
    返回 {"access_token": token}

    # 接口保护
    @app.get("/topics")
    async def list(user_id = Depends(get_current_user)):
        ...
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext

logger = logging.getLogger(__name__)

# ---- 配置 ----
SECRET_KEY = "aia-research-secret-key-change-in-production-2024"
ALGORITHM = "HS256"
TOKEN_EXPIRE_DAYS = 7

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer(auto_error=False)  # auto_error=False 允许无鉴权访问公开端点

# ---- 简易用户数据库（JSON 文件持久化） ----
USERS_FILE = Path("data/users.json")
USERS_FILE.parent.mkdir(parents=True, exist_ok=True)


def _load_users() -> dict:
    if USERS_FILE.exists():
        try:
            return json.loads(USERS_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_users(users: dict):
    USERS_FILE.write_text(json.dumps(users, indent=2, ensure_ascii=False))


# ---- 密码处理 ----
def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


# ---- Token 处理 ----
def create_token(user_id: str, expires_delta: Optional[timedelta] = None) -> str:
    expire = datetime.utcnow() + (expires_delta or timedelta(days=TOKEN_EXPIRE_DAYS))
    return jwt.encode({"sub": user_id, "exp": expire}, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> Optional[str]:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("sub")
    except JWTError:
        return None


# ---- 用户注册/登录 ----
def register_user(username: str, password: str, email: str = "") -> tuple[bool, str]:
    """注册新用户。返回 (成功, 消息)"""
    users = _load_users()
    if username in users:
        return False, "用户名已存在"
    users[username] = {
        "password": hash_password(password),
        "email": email,
        "created_at": datetime.utcnow().isoformat(),
    }
    _save_users(users)
    return True, "注册成功"


def login_user(username: str, password: str) -> Optional[str]:
    """验证用户并返回 token。失败返回 None。"""
    users = _load_users()
    user = users.get(username)
    if not user or not verify_password(password, user["password"]):
        return None
    return create_token(username)


# ---- FastAPI 依赖注入 ----
async def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> str:
    """从 JWT Token 中提取 user_id。无有效 token 时返回 "anonymous" 允许公开访问。"""
    if credentials and credentials.credentials:
        user_id = decode_token(credentials.credentials)
        if user_id:
            return user_id
    return "anonymous"


async def require_user(
    user_id: str = Depends(get_current_user),
) -> str:
    """必须登录才能访问的端点使用此依赖。未登录返回 401。"""
    if user_id == "anonymous":
        raise HTTPException(status_code=401, detail="请先登录")
    return user_id
