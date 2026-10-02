"""Edward Cheng | I-Wu 技能系统 — 可插拔的 Agent 能力扩展。

参考 OpenAI Assistants API 的 tools-as-skills 设计 + pi-agent 的扩展机制。
每个 Skill 是独立的能力单元，Agent 可以根据任务动态启用/禁用。

七大核心能力映射:
  1. 文献调研 → LiteratureReviewSkill
  2. 论文写作 → PaperWritingSkill
  3. 深度问答 → DeepQASkill
  4. 科研决策 → ResearchDecisionSkill
  5. 引用链追踪 → CitationExplorationSkill
  6. 论文批判 → PaperCritiqueSkill
  7. 实验设计 → ExperimentDesignSkill
"""

import logging
import threading
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Skill:
    """可插拔技能单元。"""

    name: str
    description: str
    category: str  # literature / writing / qa / decision
    tools: list[str] = field(default_factory=list)
    system_prompt_appendix: str = ""
    enabled: bool = True

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "tools": self.tools,
            "enabled": self.enabled,
            # DeepSeek 审查 #6: 暴露提示预览, 便于前端/监控查看完整定义
            "prompt_preview": self.system_prompt_appendix[:200],
        }


# ===== 七大核心技能定义 =====

LITERATURE_REVIEW = Skill(
    name="literature_review",
    description="全链路文献调研与综述合成: 12源搜索 → 知识库构建 → 综述生成",
    category="literature",
    tools=[
        "generate_search_strategy",
        "search_papers",
        "download_and_index",
        "search_knowledge_bases",
        "agent_ask",
        "export_report",
        "doi_lookup",
        "arxiv_lookup",
    ],
    system_prompt_appendix="""
【文献调研模式】
1. 收到调研主题后，先调用 search_knowledge_bases 检查是否已有相关 KB
2. 如已有相关 KB（或系统已指定当前课题库）→ 直接 agent_ask 基于该库问答，不要再 generate_search_strategy / search_papers
3. 如确无可 KB → 才调用 generate_search_strategy → search_papers → download_and_index → agent_ask
4. 最后综合所有结果，以结构化 Markdown 格式输出综述
   - 包含: 摘要/引言/方法论/关键发现/未来方向
   - 每段引用 [论文: 标题, 年份]
""",
)

PAPER_WRITING = Skill(
    name="paper_writing",
    description="论文全流程写作辅助: 大纲生成 → 段落扩展 → 引用管理 → 格式排版",
    category="writing",
    tools=[
        "agent_ask",
        "search_papers",
        "export_report",
        "check_draft_quality",
        "review_draft",
        "check_factual_support",
    ],
    system_prompt_appendix="""
【论文写作模式】
1. 理解用户写作需求 (论文类型/目标期刊/语言/章节要求)
2. 基于知识库提供文献支撑
3. 生成结构化大纲 (Abstract/Introduction/Related Work/Method/Experiments/Conclusion)
4. 逐节扩展内容，每个论点引用 1-2 篇相关论文
5. 输出格式: 学术 Markdown + BibTeX 参考文献
6. 初稿完成前必须调用 check_draft_quality 自检 (研究问题/空白定位/结构/引用/诚实性/可续写);
   有未达标项则修订后再次自检, 直至全部通过再交付
7. **引用真实性必须核验** (学术场景不容编造): 交付前用 check_factual_support 抽查关键论断
   是否被库内证据支撑; 判为 contradicted/unsupported 的论断必须改写或明确标注为「通用学术共识」。
   这是硬性步骤, 不得跳过 —— 幻觉引文属学术不端, 是高风险不可错项。
""",
)

DEEP_QA = Skill(
    name="deep_qa",
    description="学术深度问答与知识管理: 多轮指代消解 → 自适应检索 → 级联重排 → 精确回答",
    category="qa",
    tools=["agent_ask", "ask_knowledge_base", "get_topic_state"],
    system_prompt_appendix="""
【深度问答模式】
1. 理解问题的研究背景和上下文
2. 基于知识库检索最相关论文段落
3. 优先使用全文资料，次选摘要补充
4. 回答格式: 结论先行 → 论据支撑 → 引用标注 → 补充说明
5. 如信息不足，明确告知并提供替代搜索建议
""",
)

RESEARCH_DECISION = Skill(
    name="research_decision",
    description="数据驱动的科研决策支持: 研究空白分析 → 引用分析 → 趋势追踪 → 影响力评估",
    category="decision",
    tools=["analyze_research_gap", "search_papers", "get_topic_state"],
    system_prompt_appendix="""
【科研决策模式】
1. 分析特定领域的论文发表趋势 (按年份/引用数)
2. 识别高影响力论文和关键学者
3. 对比不同方法/模型的效果 (基于论文报告的指标)
4. 输出: 趋势图表描述 + 关键论文列表 + 研究建议
""",
)

CITATION_EXPLORATION = Skill(
    name="citation_exploration",
    description="引用链追踪: 追溯论文的引用/被引关系, 理清研究脉络与关键节点",
    category="literature",
    tools=["citation_chain", "search_papers"],
    system_prompt_appendix="""
【引用链追踪模式】
1. 用 citation_chain 查论文的引用与被引关系
2. 沿引用链梳理: 奠基工作 → 关键改进 → 后续发展
3. 识别高影响节点 (被引多/开创性)
4. 输出: 研究脉络图 + 关键节点论文列表
""",
)

PAPER_CRITIQUE = Skill(
    name="paper_critique",
    description="论文批判性分析: 审阅论文的核心主张/证据/方法论弱点/贡献",
    category="qa",
    tools=["agent_ask", "search_knowledge_bases", "doi_lookup", "arxiv_lookup"],
    system_prompt_appendix="""
【论文批判模式】
1. 定位论文 (通过知识库检索或 agent_ask)
2. 分析: 核心主张 → 证据是否支撑 → 方法论弱点 → 与相关工作对比
3. 判断贡献真实性: 结果是否被证据支持, 实验是否公平对比
4. 输出: 批判性评估 (贡献 + 局限 + 可改进点 + 复现难点)
""",
)

EXPERIMENT_DESIGN = Skill(
    name="experiment_design",
    description="实验设计: 为研究方向设计实验/消融/评测方案",
    category="writing",
    tools=["search_papers", "agent_ask"],
    system_prompt_appendix="""
【实验设计模式】
1. 明确研究假设与要验证的问题
2. 检索参考相关工作的评测方式 (数据集/基线/指标)
3. 设计: 主实验 + 消融实验 + 边界情况
4. 输出: 实验方案 (每个实验验证什么假设、对比哪些基线、用什么指标)
""",
)


# ===== 技能注册表 =====

# 基础能力工具地板: 任何技能模式都保留, 防止技能选择造成能力缺口
# (DeepSeek 审查 #2: 动态激活时不能丢掉检索/KB 发现/问答核心能力)
CORE_TOOLS = {"search_papers", "search_knowledge_bases", "agent_ask"}

# 任务 → 技能 意图关键词 (auto 模式按任务动态激活)
SKILL_INTENT_KEYWORDS = {
    "literature_review": [
        "综述", "调研", "文献", "搜索", "最新进展", "前沿", "研究现状",
        "survey", "literature", "review",
    ],
    "paper_writing": [
        "写", "论文", "大纲", "引言", "方法", "实验", "章节", "段落", "润色",
        "writing", "outline", "draft",
    ],
    "deep_qa": [
        "什么是", "为什么", "如何", "解释", "对比", "区别", "机制", "问答",
        "有哪些", "包含哪些", "列出", "有什么", "介绍一下这个库",
        "explain", "what is", "how",
    ],
    "research_decision": [
        "趋势", "引用", "影响力", "h-index", "高引", "分析", "决策",
        "trend", "impact",
    ],
    "citation_exploration": [
        "引用链", "引文", "脉络", "被引", "被引用", "引用关系", "相关文献链",
        "引用了", "谁引用", "引用网络", "citation", "cited",
    ],
    "paper_critique": [
        "批判", "审阅这篇", "分析这篇论文", "评估这篇", "这篇论文怎么样", "弱点", "不足",
        "critique", "weakness", "review this paper",
    ],
    "experiment_design": [
        "实验设计", "消融", "评测方案", "怎么验证", "实验方案", "设计实验", "基线对比",
        "experiment", "ablation",
    ],
}


class SkillRegistry:
    """管理所有技能，支持动态启用/禁用。"""

    def __init__(self):
        self._skills: dict[str, Skill] = {
            "literature_review": LITERATURE_REVIEW,
            "paper_writing": PAPER_WRITING,
            "deep_qa": DEEP_QA,
            "research_decision": RESEARCH_DECISION,
            "citation_exploration": CITATION_EXPLORATION,
            "paper_critique": PAPER_CRITIQUE,
            "experiment_design": EXPERIMENT_DESIGN,
        }
        # 获取当前会话的活跃技能 (默认全部启用)
        self.active: set[str] = set(self._skills.keys())
        self._validate_tools()

    def _validate_tools(self) -> None:
        """DeepSeek 审查 #1: 校验技能声明的工具在注册表中存在, 缺失时警告。
        防止技能引用了未实现/改名后的工具导致运行时错误。"""
        try:
            from agent.tool_registry import get_tool
        except Exception:
            return
        for name, skill in self._skills.items():
            missing = [t for t in skill.tools if get_tool(t) is None]
            if missing:
                logger.warning(
                    f"Skill {name} references unregistered tools: {missing}"
                )

    def enable(self, name: str):
        if name in self._skills:
            self.active.add(name)
            self._skills[name].enabled = True

    def disable(self, name: str):
        if name in self._skills:
            self.active.discard(name)
            self._skills[name].enabled = False

    def list_all(self) -> list[dict]:
        return [s.to_dict() for s in self._skills.values()]

    def get(self, name: str) -> Optional[Skill]:
        return self._skills.get(name)

    def select_for_task(self, task: str, max_skills: int = 2) -> list[str]:
        """DeepSeek 审查 #2: 按意图关键词动态激活最相关技能 (仅限激活技能)。

        无匹配 → 默认 literature_review; 未激活 → 取第一个激活技能。
        返回技能名列表, 供 tool 裁剪与 system prompt 拼接使用。
        """
        task_lower = (task or "").lower()
        scored: list[tuple[int, str]] = []
        for name, kws in SKILL_INTENT_KEYWORDS.items():
            if name not in self.active:
                continue
            s = sum(1 for kw in kws if kw.lower() in task_lower)
            if s:
                scored.append((s, name))
        scored.sort(key=lambda x: -x[0])
        if scored:
            return [n for _, n in scored[:max_skills]]
        # 第 2 层: 规则未命中 → 嵌入语义技能路由 (三层漏斗中间层, 仅限已激活技能)
        try:
            from intent_classifier import classify_skill

            sem = classify_skill(task, active=tuple(self.active))
            if sem:
                return [sem[0]]
        except Exception:
            pass  # 语义层不可用 → 静默降级为默认技能
        if "literature_review" in self.active:
            return ["literature_review"]
        return sorted(self.active)[:1]

    def tools_for_skills(self, skill_names: list[str]) -> list[str]:
        """激活技能工具并集 + 核心地板; 并集为空 → 全量注册工具兜底。"""
        tools = set(CORE_TOOLS)
        for n in skill_names:
            skill = self._skills.get(n)
            if skill:
                tools.update(skill.tools)
        return sorted(tools)

    def prompt_for_skills(self, skill_names: list[str]) -> str:
        """拼接所选技能的 system prompt 附录 (供 pi_agent 注入)。"""
        parts = []
        for n in skill_names:
            skill = self._skills.get(n)
            if skill and skill.system_prompt_appendix:
                parts.append(f"## {skill.description}\n{skill.system_prompt_appendix}")
        return "\n\n".join(parts)

    def register(self, skill: Skill) -> bool:
        """注册/覆盖一个技能 (DeepSeek 审查 #8: 支持第三方动态扩展)。"""
        self._skills[skill.name] = skill
        if skill.enabled:
            self.active.add(skill.name)
        logger.info(f"Skill registered: {skill.name} (category={skill.category})")
        return True


# 全局单例 (DeepSeek 审查 #10: 加锁防多线程重复初始化)
_skill_registry: Optional[SkillRegistry] = None
_skill_registry_lock = threading.Lock()


def get_skills() -> SkillRegistry:
    global _skill_registry
    if _skill_registry is None:
        with _skill_registry_lock:
            if _skill_registry is None:
                _skill_registry = SkillRegistry()
    return _skill_registry


def schemas_for_tools(tool_names: list[str]) -> list[dict]:
    """按工具名取 OpenAI schema; 未注册名自动跳过。供 pi_agent 裁剪工具集用。"""
    try:
        from agent.tool_registry import get_tools_schema

        all_schemas = get_tools_schema()
        by_name = {s["function"]["name"]: s for s in all_schemas}
        return [by_name[n] for n in tool_names if n in by_name]
    except Exception:
        return []
