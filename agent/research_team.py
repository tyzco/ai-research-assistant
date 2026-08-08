"""虚拟科研团队 — 13 个细分 Agent + 双轨审批机制。

三层架构:
  Upstream (调研层):   选题评估 → 多元检索 → 质量筛选    [3 agents]
  Midstream (产出层):  综述合成 → 分章写作 → 图表生成    [6 agents]
  Downstream (智控层): 同行审稿 → 格式校验 → 伦理审查 → 终稿定稿 [4 agents]

双轨机制:
  - Auto track:  Agent 自动执行各环节，遇到不确定时请求人工确认
  - Human track: 三个审批闸门 (选题方向 / 核心文献 / 最终结论) 处暂停，
                 人工确认后任务进入下一环节，全程可追溯可干预

使用:
    from agent.research_team import ResearchTeam
    team = ResearchTeam()
    result = await team.execute("研究 Transformer 注意力机制的最新进展")
"""

import asyncio
import json
import logging
import threading
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from openai import AsyncOpenAI

from config import (
    AGENT_MAX_TOKENS,
    AGENT_TEMPERATURE,
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
)

logger = logging.getLogger(__name__)


# ============================================================
# 审批闸门定义
# ============================================================


class ApprovalGate(str, Enum):
    """三个人工审批闸门——必须人工确认后才能继续。"""

    TOPIC_DIRECTION = "topic_direction"  # 选题方向确认
    CORE_LITERATURE = "core_literature"  # 核心文献入选
    FINAL_CONCLUSION = "final_conclusion"  # 最终结论定稿


@dataclass
class ApprovalRequest:
    """审批请求——发送给前端展示，等待用户确认。"""

    gate: ApprovalGate
    title: str
    summary: str
    options: list[str] = field(default_factory=list)
    agent_output: str = ""
    created_at: str = ""


# ============================================================
# 13 个 Agent 定义
# ============================================================

AGENT_DEFINITIONS = {
    # ========== UPSTREAM: 调研层 ==========
    "topic_evaluator": {
        "name": "选题评估 Agent",
        "layer": "upstream",
        "role": "评估研究选题的可行性、创新性和学术价值。识别研究空白，判断选题是否值得深入。",
        "prompt": """你是选题评估专家。收到研究方向后，完成以下评估：

1. **可行性分析**: 该方向是否有足够的文献支撑？数据/方法是否可获取？
2. **创新性判断**: 与现有研究相比，该方向的差异化价值在哪？
3. **研究空白**: 识别该领域尚未被充分探索的子方向
4. **推荐聚焦**: 建议 2-3 个更具体的切入角度

输出格式:
## 选题评估报告
### 可行性: [高/中/低] — 说明
### 创新性: [高/中/低] — 说明
### 研究空白: [列出 2-3 个]
### 推荐聚焦方向:
1. [具体方向1]
2. [具体方向2]""",
        "tools": ["search_knowledge_bases", "agent_ask", "analyze_research_gap"],
        "needs_approval": ApprovalGate.TOPIC_DIRECTION,
        "approval_summary": "选题方向确认：请确认以下研究方向是否继续，或提供修正意见。",
    },
    "multi_source_searcher": {
        "name": "多元文献检索 Agent",
        "layer": "upstream",
        "role": "12 源并行搜索学术论文，覆盖中英文数据库，确保文献覆盖的广度和深度。",
        "prompt": """你是文献检索专家。根据已确认的选题方向，执行多源检索：

1. 调用 generate_search_strategy 生成中英文检索策略
2. 调用 search_papers 在 12 个学术数据库中并行搜索
3. 调用 search_knowledge_bases 检查本地是否已有相关知识库
4. 汇总结果，标注每个来源的检索命中数

输出格式:
## 文献检索报告
### 检索策略: [中英文关键词]
### 检索结果:
| 数据库 | 命中数 | OA可下载 |
|--------|--------|----------|
| arXiv  | XX     | XX       |
...""",
        "tools": [
            "generate_search_strategy",
            "search_papers",
            "search_knowledge_bases",
        ],
    },
    "quality_screener": {
        "name": "文献质量筛选 Agent",
        "layer": "upstream",
        "role": "按引用数/来源可信度/年份/相关性筛选高价值论文，过滤低质量文献。",
        "prompt": """你是文献质量评估专家。对检索到的论文进行质量筛选：

1. **引用加权**: 高引论文优先（citation_count > 50 标记为"必读"）
2. **来源可信度**: Semantic Scholar > OpenAlex > arXiv > 其他 (按 SOURCE_WEIGHTS)
3. **时效性**: 优先近 3 年论文，经典论文（高引旧文）单独标注
4. **相关性**: 标题+摘要与选题的语义匹配度
5. **去重去噪**: 剔除会议通知/征稿/撤稿等垃圾条目

输出格式:
## 文献质量筛选报告
### 必读论文 (Top 5-10):
1. [标题] ([年份], 引用: XX) — 入选理由
### 推荐阅读 (10-20):
...
### 已过滤: XX 篇 (低质量/不相关/重复)""",
        "tools": ["search_papers", "agent_ask"],
        "needs_approval": ApprovalGate.CORE_LITERATURE,
        "approval_summary": "核心文献入选确认：请勾选确认最终纳入分析的论文列表。",
    },
    # ========== MIDSTREAM: 产出层 ==========
    "review_synthesizer": {
        "name": "文献综述合成 Agent",
        "layer": "midstream",
        "role": "基于已筛选的核心文献，生成结构化文献综述（摘要/引言/方法论/关键发现/未来方向）。",
        "prompt": """你是文献综述撰写专家。基于已入选的核心论文，撰写结构化综述：

## 综述结构
### 摘要 (200-300字): 概述该领域的研究现状、主要方法和未解决问题
### 1. 引言: 研究背景与意义
### 2. 相关工作:
   - 2.1 方法A类 (引用论文[1],[3],[5])
   - 2.2 方法B类 (引用论文[2],[4],[6])
   - 2.3 其他方法
### 3. 关键方法比较: 用表格对比各方法的优缺点
### 4. 开放问题与未来方向
### 参考文献: [1] 标题, 作者, 年份, DOI

每个论点至少引用 1 篇论文。引用格式: [N]""",
        "tools": ["agent_ask", "ask_knowledge_base"],
    },
    "chapter_writer_intro": {
        "name": "引言章节写作 Agent",
        "layer": "midstream",
        "role": "撰写论文引言章节：研究背景→问题陈述→研究目标→论文贡献→组织结构。",
        "prompt": """你是论文章节写作专家（引言部分）。基于文献综述内容，撰写引言：

## 引言结构 (Introduction)
1. **研究背景** (2-3段): 该领域的宏观背景和重要性
2. **问题陈述** (1段): 明确指出现有方法的不足或研究空白
3. **研究目标** (1段): 本文要解决的核心问题
4. **主要贡献** (3-4点): 本文的创新点
5. **论文组织结构** (1段): 后续章节安排

每段用引用支撑。语言: 学术中文或英文。""",
        "tools": ["agent_ask"],
    },
    "chapter_writer_method": {
        "name": "方法章节写作 Agent",
        "layer": "midstream",
        "role": "撰写方法章节：模型架构→算法流程→实验设置→评估指标。",
        "prompt": """你是论文章节写作专家（方法部分）。撰写方法章节：

## 方法结构 (Method)
1. **总体架构**: 描述模型的整体设计，用文字说明（禁用图）
2. **核心模块**: 逐个子模块的详细设计（算法流程用编号列表描述）
3. **训练策略**: 损失函数、优化器、超参数配置
4. **实验设置**: 数据集、对比基线、评估指标

参考已有论文的描述方式。语言: 学术中文或英文。""",
        "tools": ["agent_ask"],
    },
    "chapter_writer_experiment": {
        "name": "实验章节写作 Agent",
        "layer": "midstream",
        "role": "撰写实验章节：实验设计→结果分析→消融实验→可视化描述。",
        "prompt": """你是论文章节写作专家（实验部分）。撰写实验章节：

## 实验结构 (Experiments)
1. **实验设计**: 数据集规模、评估协议、对比基线
2. **主实验结果**: 用表格形式总结各方法在核心指标上的对比
3. **消融实验**: 每个设计模块的贡献分析
4. **案例分析**: 选取 2-3 个典型案例做深入分析
5. **讨论**: 结果的意义、局限性和未来工作

数据用表格呈现，每个发现要有论文引用支撑。""",
        "tools": ["agent_ask"],
    },
    "figure_generator": {
        "name": "图表说明生成 Agent",
        "layer": "midstream",
        "role": "基于实验结果生成图表描述和说明文字（不生成实际图片）。",
        "prompt": """你是学术图表说明撰写专家。基于实验数据和论文内容，生成图表说明：

## 输出格式

### Figure 1: [图标题]
**Caption**: [一段 2-3 句的图注，描述图中展示的内容]
**Description**: [在正文中引用此图的段落，描述图中展示的关键模式和结论]

### Table 1: [表标题]
**设计**: [表格的行列结构说明]
**关键发现**: [表格揭示的核心结论]

注意: 你生成的是图表描述文字，不是实际图片。""",
        "tools": ["agent_ask"],
    },
    "bib_manager": {
        "name": "参考文献管理 Agent",
        "layer": "midstream",
        "role": "整理所有引用的参考文献，输出标准格式（BibTeX / GB/T 7714 / APA）。",
        "prompt": """你是参考文献管理专家。整理论文中所有引用的文献，输出多种格式：

1. **按引用顺序**: 编号 [1] [2] [3] ...
2. **输出格式**: BibTeX (可导入 Zotero/Mendeley)
3. **中文格式**: GB/T 7714-2015
4. **英文格式**: APA 7th
5. **DOI 校验**: 检查每个条目的 DOI 是否有效

输出: 完整的参考文献列表 + BibTeX 代码块。""",
        "tools": ["agent_ask"],
    },
    # ========== DOWNSTREAM: 智控层 ==========
    "peer_reviewer": {
        "name": "模拟同行审稿 Agent",
        "layer": "downstream",
        "role": "以审稿人视角审查论文：方法有效性→实验说服力→贡献清晰度→写作质量。",
        "prompt": """你是模拟同行审稿人。以顶会审稿标准审查论文：

## 审稿维度 (1-5 分)
1. **技术正确性**: 方法是否有理论支撑？推导是否严密？
2. **实验说服力**: 实验设计是否合理？对比基线是否充分？
3. **贡献清晰度**: 核心贡献是否明确表述？与已有工作的区别是否清楚？
4. **写作质量**: 结构是否清晰？语言是否准确？
5. **可复现性**: 是否提供足够的细节供他人复现？

## 审稿意见格式
### Overall: [Strong Accept / Accept / Borderline / Reject]
### Strengths: (列出 2-3 个优点)
### Weaknesses: (列出 2-3 个需要改进的地方)
### Detailed Comments: (逐节的详细修改建议)
### Questions to Authors: (2-3 个需要作者澄清的问题)""",
        "tools": ["agent_ask", "check_draft_quality", "review_draft"],
    },
    "citation_verifier": {
        "name": "引用精度验证 Agent",
        "layer": "downstream",
        "role": "逐条验证论文中每个引用是否真实存在于知识库中，消除幻觉引用。（借鉴 PaperQA2 三轮验证设计）",
        "prompt": """你是引用精度验证专家。逐条检查以下文本中的引用，验证其真实性。

## 验证规则 (借鉴 PaperQA2 三轮验证 + Research Radar 矛盾检测)
1. **存在性检查**: 每条 [N] 引用的论文是否真实存在于知识库中？标题/作者/年份是否匹配？
2. **语义一致性**: 引用的论文内容是否真的支撑正文中的论点？（不为引用而引用）
3. **幻觉检测**: 标识任何找不到匹配的引用（标记为 [幻觉]）
4. **缺失补充**: 如果有更好的论文可以支撑某论点但未被引用，建议补充

## 输出格式
### 验证通过的引用: [列出] (共 N 条)
### 疑似幻觉引用: [列出] (共 N 条) — 每条约需标注原引用编号和判断原因
### 需要补充的引用: [列出] (共 N 条) — 每条约需标注对应正文位置和推荐论文

请对整个文档的引用进行全面审查。""",
        "tools": ["agent_ask", "search_knowledge_bases"],
    },
    "format_checker": {
        "name": "格式规范校验 Agent",
        "layer": "downstream",
        "role": "检查论文格式：引用一致性→图表编号→章节标题→页边距/字体要求。",
        "prompt": """你是学术格式校验专家。检查论文格式是否符合期刊/会议要求：

## 校验清单
1. **引用一致性**: 正文引用的每一篇是否都出现在参考文献列表中？反过来呢？
2. **图表编号**: Figure/Table 编号是否连续？正文是否引用了每一张图表？
3. **章节标题**: 层级是否正确？（1 → 1.1 → 1.1.1）
4. **术语一致性**: 同一概念在全文中是否使用了统一的术语？
5. **数学符号**: 公式中的符号是否在正文中都有定义？

输出: 逐条列出发现的问题，标注严重程度 [严重/一般/建议]。""",
        "tools": ["agent_ask"],
    },
    "ethics_reviewer": {
        "name": "伦理合规审查 Agent",
        "layer": "downstream",
        "role": "审查研究的伦理合规性：数据隐私→公平性→潜在滥用风险→IRB 要求。",
        "prompt": """你是学术伦理审查专家。审查论文是否满足学术伦理要求：

## 审查维度
1. **数据隐私**: 使用的数据集是否合规？是否涉及个人敏感信息？
2. **公平性与偏见**: 方法是否存在对特定群体的系统性偏见？
3. **潜在滥用**: 技术是否可能被用于 harmful 目的？是否讨论了 mitigation？
4. **IRB 合规**: 涉及人类受试者的研究是否有 IRB 审批？
5. **利益冲突**: 作者是否声明了潜在的利益冲突？

输出: 逐项评估，标注风险等级 [高/中/低/无]。""",
        "tools": ["agent_ask"],
    },
    "final_drafter": {
        "name": "终稿定稿 Agent",
        "layer": "downstream",
        "role": "整合所有章节和修改意见，生成最终完整论文稿件。",
        "prompt": """你是终稿整合专家。将前面所有 agent 的产出整合为完整论文：

1. **章节拼接**: 按学术论文标准顺序组装（Title→Abstract→Introduction→...→Conclusion→References）
2. **风格统一**: 确保全文语言风格一致、术语一致、引用格式一致
3. **交叉引用修正**: 确保所有内部引用（"see Section 3.2"）正确
4. **最终检查**: 运行格式校验和伦理审查中标记的所有问题是否已修复

输出: 完整论文 Markdown + BibTeX 参考文献。""",
        "tools": ["agent_ask", "export_report"],
        "needs_approval": ApprovalGate.FINAL_CONCLUSION,
        "approval_summary": "最终结论定稿确认：请审核完整论文稿件，确认无误后输出。",
    },
}


# ============================================================
# 虚拟科研团队 — 主循环
# ============================================================


@dataclass
class ResearchTeam:
    """13 agent 虚拟科研团队 + 双轨审批机制。

    工作流:
      User 输入研究方向
        → [Upstream] 选题评估 → 🔴 审批闸门1: 选题方向确认
        → [Upstream] 多元检索 → 质量筛选 → 🔴 审批闸门2: 核心文献入选
        → [Midstream] 综述合成 → 分章写作(×3) → 图表生成 → 参考文献
        → [Downstream] 同行审稿 → 格式校验 → 伦理审查 → 终稿定稿
        → 🔴 审批闸门3: 最终结论定稿
    """

    # 工作流定义: (agent_key, description, required)
    PIPELINE = [
        # === UPSTREAM ===
        ("topic_evaluator", "选题评估", True),
        ("multi_source_searcher", "多元文献检索", False),  # 可跳过(已有KB)
        ("quality_screener", "文献质量筛选", True),
        # === MIDSTREAM ===
        ("review_synthesizer", "文献综述合成", True),
        ("chapter_writer_intro", "引言章节写作", False),
        ("chapter_writer_method", "方法章节写作", False),
        ("chapter_writer_experiment", "实验章节写作", False),
        ("figure_generator", "图表说明生成", False),
        ("bib_manager", "参考文献管理", False),
        # === DOWNSTREAM ===
        ("peer_reviewer", "模拟同行审稿", False),
        ("citation_verifier", "引用精度验证", False),
        ("format_checker", "格式规范校验", False),
        ("ethics_reviewer", "伦理合规审查", False),
        ("final_drafter", "终稿定稿", True),
    ]

    def __init__(self, run_id: str = ""):
        """创建团队实例。run_id 为空 → 自动生成 (每个 /team/start 独立 run)。
        DeepSeek 审查 #7: 弃用全局单例, 按 run_id 隔离多用户并发。"""
        self.run_id: str = run_id or f"run_{uuid.uuid4().hex[:12]}"
        self._client: Optional[AsyncOpenAI] = None
        self.state: dict[str, Any] = {}  # agent_key → output
        self.pending_approval: Optional[ApprovalRequest] = None
        self.current_stage: str = "idle"
        self.history: list[dict] = []  # 执行日志
        self._lock = asyncio.Lock()  # 并发防护: 同一 run 禁止并发执行/审批

    @property
    def client(self):
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL
            )
        return self._client

    def get_agents_by_layer(self, layer: str) -> list[str]:
        return [k for k, v in AGENT_DEFINITIONS.items() if v["layer"] == layer]

    def get_layer_name(self, layer: str) -> str:
        return {"upstream": "上游调研层", "midstream": "中游产出层", "downstream": "下游智控层"}.get(
            layer, layer
        )

    # ---- 快照持久化 (DeepSeek 审查 #2) ----

    def _pending_dict(self) -> dict | None:
        """pending_approval → 可 JSON 序列化 dict。"""
        if not self.pending_approval:
            return None
        p = self.pending_approval
        return {
            "gate": p.gate.value,
            "title": p.title,
            "summary": p.summary,
            "options": p.options,
            "agent_output": p.agent_output,
            "created_at": p.created_at,
        }

    def _save_snapshot(self, status: str = "active") -> None:
        """持久化当前团队状态到 SQLite, 崩溃/重启后可恢复。"""
        try:
            from database import get_db

            get_db().save_team_run(
                run_id=self.run_id,
                task=self.state.get("task", ""),
                # 只存字符串值 (agent 产出/任务/反馈), 避免 JSON 序列化失败
                state={k: v for k, v in self.state.items() if isinstance(v, str)},
                history=self.history,
                current_stage=self.current_stage,
                pending_approval=self._pending_dict(),
                status=status,
            )
        except Exception as e:
            logger.warning(f"Failed to save team snapshot {self.run_id}: {e}")

    @classmethod
    def restore(cls, run_id: str) -> Optional["ResearchTeam"]:
        """从 SQLite 快照恢复团队 (崩溃后 /team/approve 可继续)。"""
        from database import get_db

        d = get_db().get_team_run(run_id)
        if not d:
            return None
        team = cls(run_id=run_id)
        team.state = d.get("state", {}) or {}
        team.history = d.get("history", []) or []
        team.current_stage = d.get("current_stage", "idle")
        p = d.get("pending_approval")
        if p:
            team.pending_approval = ApprovalRequest(
                gate=ApprovalGate(p["gate"]),
                title=p.get("title", ""),
                summary=p.get("summary", ""),
                options=p.get("options", []) or [],
                agent_output=p.get("agent_output", ""),
                created_at=p.get("created_at", ""),
            )
        logger.info(f"Team {run_id} restored from snapshot (stage={team.current_stage})")
        return team

    async def _run_pipeline(self, start_idx: int) -> dict:
        """从 start_idx 开始执行流水线 (DeepSeek 审查 #6: 去重 execute/approve_and_continue)。
        遇审批闸门 → 返回 awaiting_approval; 全部完成 → 返回 completed。"""
        for i in range(start_idx, len(self.PIPELINE)):
            agent_key, description, required = self.PIPELINE[i]
            if not required and not self._should_run(agent_key):
                continue

            self.current_stage = agent_key
            agent_def = AGENT_DEFINITIONS[agent_key]
            logger.info(f"Running: {agent_def['name']} ({description})")

            result = await self._run_agent(agent_key, agent_def)

            self.state[agent_key] = result
            self.history.append(
                {
                    "agent": agent_key,
                    "name": agent_def["name"],
                    "description": description,
                    "layer": agent_def["layer"],
                    "output_preview": result[:200],
                }
            )
            self._save_snapshot()  # 每步落盘, 崩溃可恢复

            # 检查是否需要人工审批
            gate = agent_def.get("needs_approval")
            if gate:
                self.pending_approval = ApprovalRequest(
                    gate=gate,
                    title=agent_def.get("approval_summary", f"{agent_def['name']} 审批"),
                    summary=result[:500],
                    agent_output=result,
                )
                self._save_snapshot()  # 待审批状态落盘
                return {
                    "status": "awaiting_approval",
                    "gate": gate.value,
                    "agent": agent_key,
                    "name": agent_def["name"],
                    "request": {
                        "title": self.pending_approval.title,
                        "summary": self.pending_approval.summary,
                    },
                    "completed_stages": [h["agent"] for h in self.history],
                }

        # 全部完成
        final = self.state.get("final_drafter", "")
        self._save_snapshot(status="completed")
        return {
            "status": "completed",
            "result": final or self._build_summary(),
            "history": self.history,
            "completed_stages": [h["agent"] for h in self.history],
        }

    async def execute(self, task: str) -> dict:
        """执行完整研究流程。返回 {status, result|request, run_id}"""
        async with self._lock:
            self.state = {"task": task}
            self.history = []
            return await self._run_pipeline(0)

    async def approve_and_continue(self, feedback: str = "") -> dict:
        """人工确认后继续执行。feedback 可选，用于修正方向。"""
        async with self._lock:
            if not self.pending_approval:
                return {"status": "error", "message": "没有待处理的审批"}

            gate = self.pending_approval.gate
            self.pending_approval = None

            if feedback:
                self.state[f"feedback_{gate.value}"] = feedback

            # 从当前 agent 的下一个开始继续
            current_idx = next(
                (i for i, (k, _, _) in enumerate(self.PIPELINE) if k == self.current_stage),
                -1,
            )
            return await self._run_pipeline(current_idx + 1)

    async def _run_agent(self, agent_key: str, agent_def: dict) -> str:
        """运行单个 agent——构建 system prompt + 上下文 → LLM 调用 + 工具调用循环。

        DeepSeek 审查 #1: 原实现传入 tools/tool_choice 但只读 message.content,
        tool_calls 被静默丢弃, 依赖工具的 agent (检索/问答) 全部失效。
        现实现标准 Function Calling 循环: 检测 tool_calls → 执行 → 结果回填 → 再调用。
        #8: schema 只暴露 agent 声明过的工具, 防止误调无 topic_id 的工具。
        #5: LLM 调用带 max_retries 自动重试。
        """
        tools_desc = ", ".join(agent_def.get("tools", []))
        context = self._build_context(agent_def["layer"])

        # 人工审批反馈注入 (修复: 原 feedback 只存 state 无 agent 读取 → "修改后通过"是摆设)
        feedbacks = [
            self.state[k]
            for k in sorted(self.state)
            if k.startswith("feedback_") and self.state[k]
        ]
        feedback_txt = "\n".join(f"【人工修正意见（必须遵守）】{f}" for f in feedbacks)

        messages = [
            {
                "role": "system",
                "content": (
                    f"你是 PaperLink 虚拟科研团队的 {agent_def['name']}。"
                    f"所属层级: {self.get_layer_name(agent_def['layer'])}。"
                    f"你的职责: {agent_def['role']}\n"
                    f"可用工具: {tools_desc}\n\n"
                    f"{agent_def['prompt']}"
                ),
            },
            {
                "role": "user",
                "content": f"研究任务: {self.state.get('task', '')}\n\n"
                + (feedback_txt + "\n\n" if feedback_txt else "")
                + f"已有成果 (上游 agent 产出):\n{context}\n\n"
                f"请基于以上信息，完成你的专业环节。",
            },
        ]

        try:
            from agent.tool_registry import get_tool, get_tools_schema

            declared = set(agent_def.get("tools", []))
            tools_schema = [
                s
                for s in get_tools_schema()
                if s["function"]["name"] in declared
            ] or None

            # 工具调用循环: 上限 8 轮, 防止模型死循环
            for _ in range(8):
                resp = await self.client.chat.completions.create(
                    model=DEEPSEEK_MODEL,
                    messages=messages,
                    tools=tools_schema,
                    tool_choice="auto" if tools_schema else None,
                    temperature=AGENT_TEMPERATURE,
                    max_tokens=AGENT_MAX_TOKENS,
                    extra_body={"thinking": {"type": "disabled"}},
                    max_retries=2,
                )
                msg = resp.choices[0].message
                tool_calls = getattr(msg, "tool_calls", None)

                # 无工具调用 → 最终回答 (若 finish_reason==length 被截断则续写)
                if not tool_calls:
                    content = msg.content or ""
                    if resp.choices[0].finish_reason == "length" and content:
                        _cont = [{"role": "assistant", "content": content}]
                        for _ in range(4):
                            _cont.append(
                                {"role": "user", "content": "（回答被截断，请从断点继续，不要重复已写内容）"}
                            )
                            _r2 = await self.client.chat.completions.create(
                                model=DEEPSEEK_MODEL,
                                messages=messages + _cont,
                                temperature=AGENT_TEMPERATURE,
                                max_tokens=AGENT_MAX_TOKENS,
                                extra_body={"thinking": {"type": "disabled"}},
                                max_retries=2,
                            )
                            _chunk = _r2.choices[0].message.content or ""
                            content += _chunk
                            if _r2.choices[0].finish_reason != "length" or not _chunk:
                                break
                            _cont.append({"role": "assistant", "content": _chunk})
                    return content.strip()

                # 回填 assistant 消息(含 tool_calls)保持对话连续性
                messages.append(msg.model_dump(exclude_none=True))
                for tc in tool_calls:
                    try:
                        args = json.loads(tc.function.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    tool = get_tool(tc.function.name)
                    if tool is None:
                        result = f"❌ 未知工具: {tc.function.name}"
                    else:
                        result = await tool.execute(**args)
                    messages.append(
                        {"role": "tool", "tool_call_id": tc.id, "content": result}
                    )

            return "（工具调用轮次超限，已停止）"
        except Exception as e:
            logger.error(f"Agent {agent_key} failed: {e}")
            return f"[{agent_def['name']} 执行失败: {e}]"

    def _build_context(self, current_layer: str) -> str:
        """为下游 agent 构建上游产出摘要。
        DeepSeek 审查 #3: 截断上限 500 → 1500, 减少下游上下文丢失。"""
        layer_order = ["upstream", "midstream", "downstream"]
        context_parts = []
        for layer in layer_order:
            if layer == current_layer:
                break
            agents_in_layer = self.get_agents_by_layer(layer)
            for agent_key in agents_in_layer:
                output = self.state.get(agent_key)
                if output:
                    context_parts.append(
                        f"### {AGENT_DEFINITIONS[agent_key]['name']}\n"
                        f"{output[:1500]}..."
                    )
        return "\n\n".join(context_parts) if context_parts else "（无上游产出）"

    def _should_run(self, agent_key: str) -> bool:
        """判断是否跳过可选 agent。
        DeepSeek 审查 #4: 原 has_kb magic flag 全项目无处设置, 属死代码。
        当前无可自动跳过的依据 → 一律执行 (required=False 仅表达"可选")。
        后续接入会话持久化后可在此按资源/偏好动态跳过。"""
        return True

    def _build_summary(self) -> str:
        parts = ["# 研究流程总结\n"]
        for h in self.history:
            layer_cn = self.get_layer_name(h["layer"])
            parts.append(f"## [{layer_cn}] {h['name']}\n{h['output_preview']}\n")
        return "\n".join(parts)


# ============================================================
# 团队实例注册表 — 按 run_id 隔离 (DeepSeek 审查 #7: 弃用全局单例)
# ============================================================
_teams: dict[str, ResearchTeam] = {}
_teams_lock = threading.Lock()


def get_team(run_id: str = "") -> ResearchTeam:
    """按 run_id 获取团队实例。

    - run_id 为空: 创建全新实例 (每个 /team/start 独立 run, 多用户隔离)。
    - run_id 在内存: 直接返回。
    - run_id 仅存于 DB: 从快照恢复 (崩溃恢复)。
    - run_id 未知: 抛 KeyError (由 api 层转 404)。
    """
    if run_id:
        with _teams_lock:
            team = _teams.get(run_id)
            if team is not None:
                return team
        team = ResearchTeam.restore(run_id)
        if team is None:
            raise KeyError(f"run_id 不存在: {run_id}")
        with _teams_lock:
            _teams[run_id] = team
        return team

    team = ResearchTeam()
    with _teams_lock:
        _teams[team.run_id] = team
        _prune_teams()
    return team


def _prune_teams() -> None:
    """注册表上限 200。超限先释放无待审批的 run (可从 DB 恢复), 再丢最旧。"""
    if len(_teams) < 200:
        return
    for rid in [r for r, t in _teams.items() if t.pending_approval is None]:
        _teams.pop(rid, None)
    while len(_teams) > 200:
        _teams.pop(next(iter(_teams)), None)
