"""工具调用管理 — 参考 pi-agent 的参数校验 + 安全检查 + 错误恢复。

核心能力:
  1. 参数验证: JSON Schema 校验 + 类型转换, 为模型兜底修正错误指令
  2. 安全检查: 工具调用前做安全审计, 调用后处理结果
  3. 错误恢复: 工具执行出错时将报错信息返回模型, 让模型自行纠错

使用:
    from agent.tool_manager import ToolManager
    tm = ToolManager()
    result = await tm.execute(tool_name, tool_args, safe_mode=True)
"""

import json
import logging
import threading
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# 危险工具黑名单 (需额外确认; 仅审计, 不自动阻止 — 见 security_check)
DANGEROUS_TOOLS = {
    "download_and_index": "涉及网络下载和执行文件操作",
    "bash": "涉及 Shell 命令执行",
    "write_file": "涉及文件写入",
}

# 安全域名白名单 (download 工具)
SAFE_DOMAINS = [
    "arxiv.org",
    "openalex.org",
    "semanticscholar.org",
    "api.core.ac.uk",
    "crossref.org",
    "doaj.org",
    "ncbi.nlm.nih.gov",
    "europepmc.org",
    "paperswithcode.com",
]


def _is_safe_domain(url: str) -> bool:
    """严格域名匹配: 白名单域名或其子域名。
    DeepSeek 审查 #1: 原 `d in url` 子串匹配可被 fake-arxiv.org 绕过。"""
    try:
        host = (urlparse(url).hostname or "").lower()
    except Exception:
        return False
    return any(host == d or host.endswith("." + d) for d in SAFE_DOMAINS)


# 瞬时故障 (值得重试): 网络抖动/超时/限流/5xx —— 重试通常能过
_TRANSIENT_MARKERS = (
    "timeout", "timed out", "connection", "connect", "temporarily",
    "rate limit", "too many requests", "502", "503", "504", "429",
    "server error", "reset by peer", "unavailable",
)
# 永久故障 (不该重试): 参数错/找不到/权限 —— 重试只是白等
_PERMANENT_MARKERS = (
    "缺少必填", "invalid", "not found", "不存在", "permission", "forbidden",
    "unauthorized", "schema", "jsondecode", "值的类型", "400", "401", "403", "404",
)


def classify_tool_error(exc: BaseException | str) -> str:
    """把工具异常分类为 transient / permanent。分类决定是否重试。

    宁可判成 permanent 也不要滥重试: 参数错误重试 3 次只是把 3 秒的失败拖成 9 秒,
    对用户没有任何收益 (且与"保速度"原则冲突)。
    """
    name = type(exc).__name__ if isinstance(exc, BaseException) else ""
    msg = (str(exc) or "").lower()
    blob = f"{name.lower()} {msg}"

    if any(m in blob for m in _PERMANENT_MARKERS):
        return "permanent"
    if name in ("TimeoutError", "ConnectError", "ReadTimeout", "ConnectTimeout",
                "APITimeoutError", "APIConnectionError", "ReadError", "RemoteProtocolError",
                "RateLimitError", "InternalServerError", "ServiceUnavailableError"):
        return "transient"
    if any(m in blob for m in _TRANSIENT_MARKERS):
        return "transient"
    return "permanent"


class ToolManager:
    """工具调用管理器: 校验 + 安全 + 恢复。"""

    def __init__(self):
        self.call_history: list[dict] = []
        self.error_count: dict[str, int] = {}  # tool_name → error count
        self._lock = threading.Lock()  # DeepSeek 审查 #4: 并发写保护

    def validate_params(
        self, tool_name: str, params: dict, schema: dict
    ) -> tuple[bool, dict, str]:
        """JSON Schema 校验 + 自动类型转换。

        Returns: (is_valid, corrected_params, error_message)
        """
        corrected = dict(params)
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))

        # 检查必填参数
        missing = required - set(corrected.keys())
        if missing:
            return False, corrected, f"缺少必填参数: {', '.join(missing)}"

        # 类型转换 + 校验
        for key, prop in properties.items():
            if key not in corrected:
                continue
            expected_type = prop.get("type", "string")
            value = corrected[key]

            try:
                # DeepSeek 审查 #2: bool 是 int 子类, 先显式处理 bool 避免误转
                if expected_type == "integer":
                    if isinstance(value, bool):
                        corrected[key] = int(value)
                    elif not isinstance(value, int):
                        corrected[key] = int(value)
                elif expected_type == "number":
                    if isinstance(value, bool):
                        corrected[key] = float(value)
                    elif not isinstance(value, (int, float)):
                        corrected[key] = float(value)
                elif expected_type == "boolean":
                    if not isinstance(value, bool):
                        if isinstance(value, (int, float)):
                            corrected[key] = bool(value)
                        else:
                            corrected[key] = (
                                str(value).strip().lower() in ("true", "1", "yes", "是")
                            )
                elif expected_type == "array" and isinstance(value, str):
                    # 尝试解析 JSON array 字符串
                    try:
                        corrected[key] = json.loads(value)
                    except json.JSONDecodeError:
                        corrected[key] = [value]
                elif expected_type == "string" and not isinstance(value, str):
                    corrected[key] = str(value)
            except (ValueError, TypeError) as e:
                logger.debug(f"Type coercion failed for {tool_name}.{key}: {e}")
                # 不阻断，保留原值让工具自己处理

        # === 约束校验 (前置参数校验钩子) ===
        # 原实现只查"必填 + 类型", 越界值会一路传到实现里**静默出错**:
        #   - enum: citation_chain.direction 传 "forward" → 实现两个分支都不进 → 返回空结果,
        #     模型误以为"这篇没有引用关系"
        #   - maxItems: download_and_index.urls 传 20 条 → 实现 urls[:5] 静默截断,
        #     模型以为 20 篇都入库了
        # 这里在**调用前**拦下并给出可执行的报错, 让模型重新生成参数, 而不是让它猜为什么结果不对。
        for key, prop in properties.items():
            if key not in corrected:
                continue
            value = corrected[key]

            enum_vals = prop.get("enum")
            if enum_vals and value not in enum_vals:
                return (
                    False,
                    corrected,
                    f"参数 {key}={value!r} 不在允许取值内，只能取 {enum_vals} 之一",
                )

            if isinstance(value, (list, str)):
                n = len(value)
                mx, mn = prop.get("maxItems"), prop.get("minItems")
                if mx is not None and n > mx:
                    return (
                        False,
                        corrected,
                        f"参数 {key} 有 {n} 项，超过上限 {mx}（超出部分不会被处理）。"
                        f"请缩减到 {mx} 项以内，或分多次调用",
                    )
                if mn is not None and n < mn:
                    return False, corrected, f"参数 {key} 至少需要 {mn} 项，当前 {n} 项"
                # 字符串长度
                if isinstance(value, str):
                    smx, smn = prop.get("maxLength"), prop.get("minLength")
                    if smx is not None and n > smx:
                        return (
                            False,
                            corrected,
                            f"参数 {key} 长度 {n} 超过上限 {smx}，请精简后重试",
                        )
                    if smn is not None and n < smn:
                        return (
                            False,
                            corrected,
                            f"参数 {key} 长度 {n} 少于下限 {smn}，请补充内容后重试",
                        )

            if isinstance(value, (int, float)) and not isinstance(value, bool):
                lo, hi = prop.get("minimum"), prop.get("maximum")
                if lo is not None and value < lo:
                    return False, corrected, f"参数 {key}={value} 小于最小值 {lo}"
                if hi is not None and value > hi:
                    return False, corrected, f"参数 {key}={value} 大于最大值 {hi}"

        return True, corrected, ""

    def security_check(self, tool_name: str, params: dict) -> tuple[bool, str]:
        """工具执行前安全检查。

        Returns: (is_safe, reason)
        """
        try:
            from config import TOOL_SECURITY_BLOCK_NON_WHITELIST
        except Exception:
            TOOL_SECURITY_BLOCK_NON_WHITELIST = False
        # URL 参数检查
        for key in ("url", "urls", "pdf_url"):
            urls = params.get(key, [])
            if isinstance(urls, str):
                urls = [urls]
            for url in urls:
                url_str = str(url).lower()
                # 检查协议
                if not url_str.startswith(("http://", "https://")):
                    return False, f"不安全的 URL 协议: {url}"
                # DeepSeek 审查 #1: 严格域名匹配; 默认仅告警, 配置可升级为阻止
                if not _is_safe_domain(url_str):
                    logger.warning(
                        f"Non-academic domain in {tool_name}.{key}: {url_str[:80]}"
                    )
                    if TOOL_SECURITY_BLOCK_NON_WHITELIST:
                        return False, f"非白名单域名已拦截: {url_str[:80]}"

        # 危险工具标记 (仅审计, 不自动阻止)
        # download_and_index 是科研团队核心工具; 人工确认由 tool_registry 的
        # need_confirm 机制负责 (DeepSeek 审查 #3: 默认阻止会破坏检索→下载主流程)
        if tool_name in DANGEROUS_TOOLS:
            logger.info(
                f"Dangerous tool called: {tool_name} — {DANGEROUS_TOOLS[tool_name]}"
            )

        return True, ""

    def format_error_for_llm(self, tool_name: str, error: str, params: dict) -> str:
        """将工具错误格式化为 LLM 可理解的反馈。

        pi-agent 风格: 错误信息返回给模型，让模型自行纠错，而非直接中断。
        """
        with self._lock:
            self.error_count[tool_name] = self.error_count.get(tool_name, 0) + 1
            error_count = self.error_count[tool_name]

        return json.dumps(
            {
                "error": True,
                "tool": tool_name,
                "message": error,
                "attempt": error_count,
                "params_used": {k: str(v)[:100] for k, v in params.items()},
                "suggestion": (
                    f"工具 {tool_name} 执行失败 (第 {error_count} 次)。"
                    f"请检查参数是否正确，或尝试其他方法完成任务。"
                ),
            },
            ensure_ascii=False,
        )

    async def execute_with_retry(
        self, tool_name: str, tool_def, args: dict, timeout: float | None = None
    ):
        """执行工具, 仅对**瞬时故障**做指数退避重试。返回 (ok, result_or_error)。

        timeout: **每次尝试**的超时 (不是总预算), 由调用方按剩余时间预算传入。

        为什么要分级: 网络抖动重试有意义, 参数写错重试纯属浪费时间。分类见
        classify_tool_error; 重试次数/退避见 config.TOOL_RETRY_*。
        """
        import asyncio

        from config import TOOL_RETRY_BASE_DELAY, TOOL_RETRY_MAX

        max_attempts = TOOL_RETRY_MAX + 1
        if tool_def is not None and getattr(tool_def, "retryable", True) is False:
            # 非重放安全的工具: 禁止自动重试。
            # 超时不代表没生效 —— 可能已经写完只是响应没回来, 重试会重复执行。
            max_attempts = 1
            logger.info(f"tool {tool_name} is not retryable → auto-retry disabled")

        last_err: BaseException | None = None
        for attempt in range(max_attempts):
            try:
                _call = tool_def.execute(**args)
                result = (
                    await asyncio.wait_for(_call, timeout=timeout)
                    if timeout
                    else await _call
                )
                if attempt:
                    logger.info(
                        f"tool {tool_name} succeeded on retry #{attempt} (前 {attempt} 次瞬时失败)"
                    )
                return True, result
            except Exception as e:
                last_err = e
                kind = classify_tool_error(e)
                if kind == "permanent" or attempt >= max_attempts - 1:
                    logger.warning(
                        f"tool {tool_name} failed ({kind}, attempt {attempt + 1}/"
                        f"{max_attempts}): {e}"
                    )
                    break
                delay = TOOL_RETRY_BASE_DELAY * (2**attempt)
                logger.warning(
                    f"tool {tool_name} transient failure (attempt {attempt + 1}/"
                    f"{max_attempts}), retry in {delay:.1f}s: {e}"
                )
                await asyncio.sleep(delay)
        return False, str(last_err)

    def record_call(self, tool_name: str, params: dict, result: str, success: bool):
        with self._lock:
            self.call_history.append(
                {
                    "tool": tool_name,
                    "params": {k: str(v)[:200] for k, v in params.items()},
                    "success": success,
                    "result_len": len(result),
                    "result_preview": result[:100],
                }
            )
            # 只保留最近 50 条
            if len(self.call_history) > 50:
                self.call_history = self.call_history[-50:]

    def get_stats(self) -> dict:
        return {
            "total_calls": len(self.call_history),
            "error_counts": dict(self.error_count),
            "recent_calls": self.call_history[-5:],
        }


# 全局单例
_tool_manager: Optional[ToolManager] = None


def get_tool_manager() -> ToolManager:
    global _tool_manager
    if _tool_manager is None:
        _tool_manager = ToolManager()
    return _tool_manager
