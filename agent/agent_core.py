"""Agent 核心：ReAct 执行循环 + 任务规划。

Agent 接收高层次目标 → 思考规划 → 调用工具 → 观察结果 → 循环直至完成。
"""

import json
import logging
import traceback

from openai import AsyncOpenAI

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL

from .tool_registry import TOOLS, get_tool, get_tools_schema

logger = logging.getLogger(__name__)

AGENT_SYSTEM_PROMPT = """你是一个学术研究 Agent。你可以使用以下工具来完成用户的调研任务。

## 可用工具
{tool_descriptions}

## 工作流程
1. generate_search_strategy 生成检索策略（必须第一步）
2. search_papers 搜索论文（最多 2 次，使用不同关键词）
3. 如果任务要求深入分析，用 download_and_index 下载论文构建知识库
4. ask_knowledge_base 基于知识库深度问答
5. 所有步骤完成后，给出结构化的调研报告

## 输出格式
每次回复使用以下格式：

Thought: 分析当前状态，思考下一步应该做什么
Action: 工具名称
Action Input: 工具参数的 JSON 对象

或者任务完成时：
Thought: 任务已完成
Final Answer: 最终调研报告（Markdown，含论文标题、年份、方法）

## 重要规则
- 每次只调用一个工具
- 搜索最多 2 轮，之后必须基于已有结果给出回答
- 最终报告必须引用具体的论文标题和年份
- 如果没有找到完全匹配的论文，说明搜索局限性"""


class ResearchAgent:
    """学术调研 Agent。"""

    def __init__(self, topic_id: str | None = None, max_steps: int = 10):
        self.topic_id = topic_id
        self.max_steps = max_steps
        self.client = AsyncOpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
        self.history: list[dict] = []
        self.steps: list[dict] = []  # 执行日志
        self.result: str = ""

    def _build_tool_descriptions(self) -> str:
        lines = []
        for t in TOOLS:
            lines.append(f"- **{t.name}**: {t.description}")
            params = t.parameters.get("properties", {})
            if params:
                for pn, pi in params.items():
                    lines.append(f"  - {pn}: {pi.get('description', '')}")
        return "\n".join(lines)

    async def run(self, task: str) -> dict:
        """运行 Agent 完成调研任务。返回 {result, steps, success}。"""
        self.history = [
            {
                "role": "system",
                "content": AGENT_SYSTEM_PROMPT.format(
                    tool_descriptions=self._build_tool_descriptions()
                ),
            },
            {"role": "user", "content": f"调研任务：{task}"},
        ]
        self.steps = []

        for step_num in range(self.max_steps):
            self.steps.append({"step": step_num + 1, "type": "thinking", "content": ""})
            logger.info(f"Agent step {step_num + 1}/{self.max_steps}")

            try:
                # 调用 LLM 推理
                resp = await self.client.chat.completions.create(
                    model=DEEPSEEK_MODEL,
                    messages=self.history,
                    tools=get_tools_schema(),
                    tool_choice="auto",
                    temperature=0.3,
                    max_tokens=1000,
                )
                msg = resp.choices[0].message

                # 如果 LLM 返回了工具调用
                if msg.tool_calls:
                    tc = msg.tool_calls[0]
                    tool_name = tc.function.name
                    tool_args = json.loads(tc.function.arguments)

                    self.steps[-1][
                        "thought"
                    ] = f"调用 {tool_name} → {str(tool_args)[:100]}"
                    logger.info(f"  Tool: {tool_name}({tool_args})")

                    tool = get_tool(tool_name)
                    if not tool:
                        tool_result = f"工具 {tool_name} 不存在"
                    else:
                        # 需要人工确认
                        if tool.need_confirm:
                            self.steps[-1]["need_confirm"] = True
                            self.steps[-1]["tool_name"] = tool_name
                            self.steps[-1]["tool_args"] = tool_args
                            return {
                                "success": True,
                                "paused": True,
                                "steps": self.steps,
                                "message": f"需要确认执行 {tool_name}",
                            }

                        tool_result = await tool.execute(**tool_args)

                    # 将工具调用和结果加入对话
                    self.history.append(
                        {"role": "assistant", "content": None, "tool_calls": [tc]}
                    )
                    self.history.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": str(tool_result)[:3000],
                        }
                    )
                    self.steps[-1]["tool_result"] = str(tool_result)[:200]

                # LLM 认为任务完成 → 反思检查
                else:
                    answer = msg.content or ""
                    # 自反思：检查答案是否完整，有无遗漏
                    self.steps[-1]["thought"] = "生成答案，进行自反思检查..."
                    self.steps[-1]["type"] = "reflecting"
                    self.history.append({"role": "assistant", "content": answer})
                    try:
                        reflection_prompt = "请反思你上面的回答：是否遗漏了重要方面？是否引用了足够的论文依据？如果有遗漏请补充，如果已完整请以'✅'开头复述最终答案。"
                        if step_num + 1 < self.max_steps:
                            self.history.append(
                                {"role": "user", "content": reflection_prompt}
                            )
                            ref_resp = await self.client.chat.completions.create(
                                model=DEEPSEEK_MODEL,
                                messages=self.history,
                                temperature=0.2,
                                max_tokens=600,
                            )
                            refined = ref_resp.choices[0].message.content or answer
                            answer = refined if len(refined) > 20 else answer
                            self.steps[-1]["reflection"] = refined[:200]
                    except Exception:
                        pass  # 反思失败就用原答案
                    self.result = answer
                    self.steps[-1]["result"] = answer[:500]
                    return {"success": True, "result": answer, "steps": self.steps}

            except Exception as e:
                self.steps[-1]["error"] = str(e)
                logger.error(f"Agent error: {e}\n{traceback.format_exc()}")
                # 继续尝试下一步
                self.history.append(
                    {"role": "user", "content": f"上一步出错了：{e}。请尝试其他方法继续。"}
                )

        # 达到最大步数
        return {
            "success": False,
            "error": f"超过最大步数 {self.max_steps}",
            "steps": self.steps,
        }


# 快捷入口
async def run_research_agent(task: str, topic_id: str | None = None) -> dict:
    agent = ResearchAgent(topic_id=topic_id)
    return await agent.run(task)
