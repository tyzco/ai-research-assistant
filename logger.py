"""结构化日志模块：全局统一日志格式 + 请求 ID 追踪。

用法：
    from logger import get_logger, set_request_id
    log = get_logger(__name__)
    log.info("message")

    # 在请求入口设置请求 ID (FastAPI 中间件)
    set_request_id(uuid4().hex[:8])
    # 该请求内所有日志自动带 [rid=xxxx]

所有日志自动带时间戳、模块名、日志级别、请求 ID。
生产环境可通过 LOG_FORMAT=json 切换 JSON 输出 (ELK/Loki 可解析)。

v5.2 修复 (DeepSeek 对话审查):
  - 请求 ID 追踪 (contextvars + logging.Filter, 异步安全)
  - JSON 格式输出 (LOG_FORMAT=json)
  - 自动初始化 (导入时配置, 防遗漏手动调用)
  - 初始化消息用本模块 logger
"""

import contextvars
import json
import logging
import os
import sys
from datetime import datetime, timezone

# 异步安全的请求 ID 上下文 (DeepSeek 建议: contextvars 传递请求 ID)
_request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default=""
)

logger = logging.getLogger(__name__)

# 噪音日志抑制列表 (DeepSeek 建议: 移到可配置)
_NOISY_LOGGERS = ("httpx", "httpcore", "urllib3", "asyncio", "sentence_transformers",
                  "datasets", "PIL", "numexpr")


def set_request_id(request_id: str):
    """在请求入口设置请求 ID。该请求内所有日志自动携带。
    应在 FastAPI 中间件中调用: 每次请求生成/传递 UUID。"""
    _request_id_var.set(request_id)


class RequestIdFilter(logging.Filter):
    """日志过滤器: 为每条日志注入 request_id 字段。"""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id_var.get() or "-"
        return True


class JsonFormatter(logging.Formatter):
    """JSON 日志格式 (DeepSeek 建议: 生产环境集中日志系统可解析)。"""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "time": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": getattr(record, "request_id", "-"),
        }
        # 附加 extra 字段
        if hasattr(record, "key") or hasattr(record, "value"):
            for k in ("key", "value"):
                if hasattr(record, k):
                    entry[k] = getattr(record, k)
        if record.exc_info:
            entry["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(entry, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    """文本日志格式 (开发环境, 带请求 ID)。直接读 record 字段, 不重复创建 Formatter。"""

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created).strftime("%m-%d %H:%M:%S")
        rid = getattr(record, "request_id", "-")
        rid_part = f" [rid={rid}]" if rid != "-" else ""
        msg = record.getMessage()
        if record.exc_info:
            msg += "\n" + self.formatException(record.exc_info)
        return f"{ts} [{record.levelname:<5}] {record.name}{rid_part}: {msg}"


_initialized = False


def setup_logging(level: str = None):
    """初始化全局日志配置。幂等, 可多次调用。"""
    global _initialized
    if _initialized:
        return
    _initialized = True

    lvl = getattr(
        logging, (level or os.getenv("LOG_LEVEL", "INFO")).upper(), logging.INFO
    )

    root = logging.getLogger()
    if not root.handlers:
        # LOG_FORMAT=json → JSON 输出, 否则文本
        if os.getenv("LOG_FORMAT", "text") == "json":
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(JsonFormatter())
        else:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(TextFormatter())
        root.addHandler(handler)

    # 注入请求 ID 过滤器到每个 handler (关键: filter 加在 handler 而非 logger,
    # 因为子 logger 的 record 传播到 root handler 时不经过 root logger 的 filter)
    for h in root.handlers:
        if not any(isinstance(f, RequestIdFilter) for f in h.filters):
            h.addFilter(RequestIdFilter())
    root.setLevel(lvl)

    # 抑制噪音日志
    for noisy in _NOISY_LOGGERS:
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logger.info("Logging initialized at level %s", logging.getLevelName(lvl))


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


# 自动初始化 (DeepSeek 建议: 防遗漏手动调用)
setup_logging()
