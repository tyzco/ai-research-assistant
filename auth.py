"""JWT 用户认证 + SQLite 持久化。

用法：
    from auth import create_token, get_current_user, hash_password, verify_password

    # 注册
    hashed = hash_password("mypassword")
    db.register_user("alice", hashed, "a@b.com")

    # 登录
    token = login_user("alice", "mypassword")

    # 接口保护
    @app.get("/topics")
    async def list(user_id = Depends(get_current_user)):
        ...
"""

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext

logger = logging.getLogger(__name__)

# ---- 配置 ----
SECRET_KEY = os.getenv(
    "JWT_SECRET_KEY", "paperlink-development-key-change-in-production"
)
_IS_DEV_KEY = "change-in-production" in SECRET_KEY
if _IS_DEV_KEY:
    logger.warning(
        "JWT_SECRET_KEY is using the default value. "
        "Set JWT_SECRET_KEY env var for production. "
        "Server will continue with dev key (not suitable for production)."
    )

ALGORITHM = "HS256"
TOKEN_EXPIRE_DAYS = 7

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer(auto_error=False)


# ---- 密码处理 ----
def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


# ---- Token 处理 ----
def create_token(user_id: str, expires_delta: Optional[timedelta] = None) -> str:
    """签发 JWT token。payload 包含 sub(用户ID)、exp(过期时间)、iat(签发时间)。"""
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(days=TOKEN_EXPIRE_DAYS)
    )
    return jwt.encode(
        {"sub": user_id, "exp": expire, "iat": datetime.now(timezone.utc)},
        SECRET_KEY,
        algorithm=ALGORITHM,
    )


def decode_token(token: str) -> Optional[str]:
    """解码 JWT token，返回 user_id。无效/过期返回 None。"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("sub")
    except jwt.ExpiredSignatureError:
        logger.debug("Token expired")
        return None
    except JWTError:
        return None


# ---- 用户注册/登录（SQLite 持久化）----
def register_user(username: str, password: str, email: str = "") -> tuple[bool, str]:
    from database import get_db

    db = get_db()
    return db.register_user(username, hash_password(password), email)


def login_user(username: str, password: str) -> Optional[str]:
    """验证用户并返回 token。失败返回 None。
    日志中区分用户不存在 vs 密码错误 (DeepSeek 对话建议)。"""
    from database import get_db

    db = get_db()
    user = db.get_user(username)
    if not user:
        logger.info(f"Login failed: user '{username}' not found")
        return None
    if not verify_password(password, user["password"]):
        logger.info(f"Login failed: wrong password for '{username}'")
        return None
    return create_token(username)


# ---- FastAPI 依赖注入 ----
def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[str]:
    """从 JWT Token 提取 user_id。无 token 返回 None，无效 token 返回 None。
    注意：如需区分「未提供」和「已过期」，调用方可检查 token 是否存在。"""
    if credentials and credentials.credentials:
        user_id = decode_token(credentials.credentials)
        if user_id:
            return user_id
        # token 存在但无效 → 记录 debug (可能已过期)
        logger.debug("Invalid or expired token presented")
    return None


def require_user(
    user_id: Optional[str] = Depends(get_current_user),
) -> str:
    """必须登录才能访问。未登录返回 401 + WWW-Authenticate 头。
    注意：user_id 现在是 Optional[str]，未认证时为 None (DeepSeek 对话建议)。"""
    if user_id is None:
        raise HTTPException(
            status_code=401,
            detail="请先登录",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user_id
