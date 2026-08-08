"""上下文工程模块 — 参考 pi-agent 的 transformContext 设计。

核心能力:
  1. 工具输出截断: 过长结果截断 + 完整内容写本地文件, 模型可选择读取
  2. Token 阈值压缩: 剩余 token 低于阈值时, 按结构化模板压缩历史上下文
  3. 上下文窗口管理: token 计数 + 预留空间 + 主动/被动压缩触发

使用:
    from agent.context_engine import ContextEngine
    ctx = ContextEngine(max_tokens=64000, reserve_tokens=8192)
    result = ctx.maybe_compact(messages, tool_results)
"""

import logging
import re
import threading
from dataclasses import dataclass
from pathlib import Path

from config import AGENT_CONTEXT_MAX_TOKENS, AGENT_CONTEXT_RESERVE_TOKENS

logger = logging.getLogger(__name__)

# tiktoken cl100k_base 编码缓存 (DeepSeek 审查 #1: 英文/代码 token 估算更准)
_tok_cache = None

# 压缩阈值: 剩余 token 低于此值时触发压缩
DEFAULT_RESERVE_TOKENS = 8192
# 单条工具输出最大 token (估算: 1 token ≈ 3 chars for Chinese, 4 chars for English)
MAX_TOOL_OUTPUT_TOKENS = 2000
# 压缩模板: 类似于 Anthropic 的 summary 格式
COMPACT_TEMPLATE = """以下为对话历史摘要（保留关键信息，省略详细过程）：

【任务目标】
{goal}

【已使用的工具】
{tools_used}

【关键发现】
{key_findings}

【当前状态】
{status}

---
以上为摘要。请基于以上信息和后续对话继续。"""


@dataclass
class ContextEngine:
    """上下文管理引擎。"""

    max_tokens: int = AGENT_CONTEXT_MAX_TOKENS  # 模型上下文窗口 (默认 1M, 用满真实窗口)
    reserve_tokens: int = AGENT_CONTEXT_RESERVE_TOKENS  # 预留 token 数 (给系统 prompt + 新输出)
    fulltext_dir: str = "data/tool_outputs"

    # 内部状态
    _compact_count: int = 0

    def __post_init__(self):
        Path(self.fulltext_dir).mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()  # DeepSeek 审查 #9: 并发计数保护

    def estimate_tokens(self, text: str) -> int:
        """token 估算。

        DeepSeek 审查 #1: 优先 tiktoken cl100k_base (英文/代码/JSON 更准),
        任一边界情况失败时回退字符启发式 (中文 ~1.5 char/token, 英文 ~3.5)。
        """
        if not text:
            return 0
        try:
            global _tok_cache
            if _tok_cache is None:
                import tiktoken

                _tok_cache = tiktoken.get_encoding("cl100k_base")
            return len(_tok_cache.encode(text))
        except Exception:
            cn_chars = len(re.findall(r"[一-鿿]", text))
            other_chars = len(text) - cn_chars
            return int(cn_chars / 1.5 + other_chars / 3.5)

    def estimate_messages_tokens(self, messages: list[dict]) -> int:
        total = 0
        for m in messages:
            content = m.get("content", "")
            if isinstance(content, str):
                total += self.estimate_tokens(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        total += self.estimate_tokens(str(part.get("text", "")))
        return total

    def remaining_tokens(self, messages: list[dict]) -> int:
        used = self.estimate_messages_tokens(messages)
        return max(0, self.max_tokens - used - self.reserve_tokens)

    def should_compact(self, messages: list[dict]) -> bool:
        """判断是否需要压缩。"""
        return self.remaining_tokens(messages) < self.reserve_tokens // 2

    def truncate_tool_output(
        self, tool_name: str, output: str
    ) -> tuple[str, str | None]:
        """截断过长工具输出，完整内容写入文件。

        Returns:
            (truncated_text, fulltext_path_or_None)
        """
        token_est = self.estimate_tokens(output)
        if token_est <= MAX_TOOL_OUTPUT_TOKENS:
            return output, None

        # DeepSeek 审查 #3: 清洗孤立代理项, 防 write_text/JSON 序列化崩溃
        import uuid

        clean = output.encode("utf-8", errors="ignore").decode("utf-8")
        # 截断 + 写文件 (uuid4 文件名, 防 hash 碰撞覆盖)
        safe_name = tool_name.replace("/", "_").replace(" ", "_")[:40]
        file_path = Path(self.fulltext_dir) / f"{safe_name}_{uuid.uuid4().hex[:12]}.txt"
        file_path.write_text(clean, encoding="utf-8")

        truncated = output[: MAX_TOOL_OUTPUT_TOKENS * 4]  # ~8000 chars
        truncated = truncated.encode("utf-8", errors="ignore").decode("utf-8")
        hint = (
            f"\n\n[输出已截断。完整结果 ({len(output)} 字符) 保存在: {file_path}。"
            f"如需查看完整内容，请使用 read_file 工具读取此路径。]"
        )
        return truncated + hint, str(file_path)

    def compact_history(
        self,
        messages: list[dict],
        goal: str = "",
        tools_used: list[str] = None,
        key_findings: list[str] = None,
    ) -> list[dict]:
        """按结构化模板压缩历史上下文。

        保留最后 2 轮完整对话，其余用摘要替换。
        """
        if len(messages) <= 4:
            return messages

        with self._lock:
            self._compact_count += 1
        # 分离: 最后 2 轮保留，其余压缩为摘要
        keep = messages[-4:]  # 最后 2 轮 = 4 条消息 (user+assistant × 2)
        old = messages[:-4]

        summary = COMPACT_TEMPLATE.format(
            goal=goal or "进行学术文献调研",
            tools_used=", ".join(tools_used or ["无"]),
            key_findings="; ".join(key_findings or ["无"]),
            status=f"已压缩 {len(old)} 条历史消息，当前第 {self._compact_count} 次压缩",
        )

        # DeepSeek 审查 #2: 摘要用 system 角色注入, 避免破坏 user/assistant 交替规则
        return [{"role": "system", "content": summary}] + keep

    def get_stats(self) -> dict:
        return {
            "compact_count": self._compact_count,
            "max_tokens": self.max_tokens,
            "reserve_tokens": self.reserve_tokens,
        }


# ===== 全局单例 =====
_ctx_engine: ContextEngine | None = None


def get_context_engine() -> ContextEngine:
    global _ctx_engine
    if _ctx_engine is None:
        _ctx_engine = ContextEngine()
    return _ctx_engine
