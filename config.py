"""配置管理：所有可调参数集中于此。其他模块通过 from config import XXX 引用。

约定: 可调旋钮 (数值/字符串/列表) 一律 env 驱动 + 默认值; 内容型数据
(意图关键词/垃圾词/注入模式/翻译字典) 作为代码常驻 — 那是内容, 不是配置旋钮。
"""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

# HF/Transformers 强制离线: 所有模型本地缓存, 禁止联网检查。
# 修复: 以前只在 embedder.py 设 HF_HUB_OFFLINE, 任何不先 import embedder 的模型加载
# (如 reranker) 都会尝试连 huggingface.co → 本机网络抖, 挂起 30-60s。config 被所有模块先导入,
# 在此全局设默认值 (尊重用户显式 env 覆盖)。
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def _env(name: str, default, cast=str):
    """读环境变量; 缺失/非法时回退默认。统一 cast, 消除 os.getenv 样板。"""
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return cast(raw)
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    return _env(name, default, lambda v: str(v).lower() in ("1", "true", "yes", "on"))


def _env_list(name: str, default: list):
    """读逗号分隔环境变量为列表。"""
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    return [x.strip() for x in raw.split(",") if x.strip()]


# ===================================================================
# LLM (DeepSeek)
# ===================================================================
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = _env("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = _env("DEEPSEEK_MODEL", "deepseek-v4-pro")

# LLM 容错 (企业级第一条: 单点故障容错)。单 DeepSeek 端点抖动即失败 →
# 超时 + 指数退避重试 + 可选降级模型。fallback 三项留空 = 不降级 (仅重试)。
LLM_TIMEOUT = _env("LLM_TIMEOUT", 60, int)
LLM_MAX_RETRIES = _env("LLM_MAX_RETRIES", 2, int)
DEEPSEEK_FALLBACK_MODEL = _env("DEEPSEEK_FALLBACK_MODEL", "", str)
DEEPSEEK_FALLBACK_BASE_URL = _env("DEEPSEEK_FALLBACK_BASE_URL", "", str)
DEEPSEEK_FALLBACK_API_KEY = os.getenv("DEEPSEEK_FALLBACK_API_KEY", "")

# ===================================================================
# Embedding (本地模型)
# ===================================================================
LOCAL_EMBEDDING_MODEL = _env("LOCAL_EMBEDDING_MODEL", "BAAI/bge-small-zh")
LOCAL_EMBEDDING_DIM = _env("LOCAL_EMBEDDING_DIM", 512, int)
# 嵌入后端: "sentence_transformers" | "flag_embedding" | "" (留空按模型名自动检测)
EMBEDDING_BACKEND = _env("EMBEDDING_BACKEND", "")
# 嵌入缓存条数 (LRU, 0 = 关闭)。同一文本+同一模型 → 向量恒定, 缓存绝对安全;
# 而嵌入是最重的本地算力: 重复上传同一篇论文/重建库会整篇重算, 缓存后直接复用。
EMBEDDING_CACHE_SIZE = _env("EMBEDDING_CACHE_SIZE", 20000, int)

# ===================================================================
# Rerank 模型 (延迟加载)
# ===================================================================
RERANK_MODEL = _env("RERANK_MODEL", "BAAI/bge-reranker-base")
# 重排精度阈值: bge-reranker 得分低于此值视为低置信度, 过滤掉。0 = 不过滤 (默认)。
# **实测标定** (43 篇真实论文 KB, 8 题 eval): bge-reranker-base 原始分数落在
# 0.0001–0.24, 中位 0.0003 —— 尺度被压缩 3 个数量级且随查询漂移, **不存在可用的
# 绝对阈值**。设 0.3 会滤掉 100% 的块, 设 0.05 会滤掉 90%, 而 R@5/MRR 几乎不变
# (说明分数与相关性没有干净的分界)。因此默认关闭; 若要开启, 必须针对**当前
# reranker 模型**重新标定, 不能照搬别处的经验值。
RERANK_MIN_SCORE = _env("RERANK_MIN_SCORE", 0.0, float)
# 全部候选都低于阈值时: True=保留最高分那条(保证有上下文) / False=返回空(如实说没找到)
RERANK_KEEP_BEST_IF_NONE = _env_bool("RERANK_KEEP_BEST_IF_NONE", True)

# HyDE (Hypothetical Document Embeddings) 查询改写 — 默认关 (每次检索+1 LLM调用, 与速度优先原则冲突)
# 开启后: 为什么是/如何/为什么类解释性问题生成假设答案段落作检索变体, 提升弱召回场景命中
# 质量水位管控 (load-aware degradation): 并发水位阈值 — 按负载动态升降检索质量档位
# 成本闸: 高成本操作前先回报估算并等人工确认 (防误操作烧钱/耗时)
# DeepSeek 真实单价 (从官方定价页填入; 默认 0 = 未配置 → 成本闸只显示次数/时间, 绝不显示假¥)
DEEPSEEK_PRICE_INPUT_PER_MTOK = _env("DEEPSEEK_PRICE_INPUT_PER_MTOK", 0.0, float)
DEEPSEEK_PRICE_OUTPUT_PER_MTOK = _env("DEEPSEEK_PRICE_OUTPUT_PER_MTOK", 0.0, float)
COST_CURRENCY = _env("COST_CURRENCY", "¥", str)

COST_GATE_ENABLED = _env_bool("COST_GATE_ENABLED", True)
COST_GATE_MIN_PAPERS = _env("COST_GATE_MIN_PAPERS", 20, int)   # 下载>=此篇数需确认
COST_GATE_SEC_PER_PAPER = _env("COST_GATE_SEC_PER_PAPER", 12, float)  # 每篇下载+解析+嵌入约12s
COST_GATE_TEAM_ESTIMATE = _env("COST_GATE_TEAM_ESTIMATE", 13, int)    # 科研团队约13次LLM调用

LOAD_HIGH_THRESHOLD = _env("LOAD_HIGH_THRESHOLD", 8, int)      # 并发>=此值 → 跳 stage2精排+查询变体
LOAD_OVERLOAD_THRESHOLD = _env("LOAD_OVERLOAD_THRESHOLD", 16, int)  # 并发>=此值 → 极致降级(单路+单级)

HYDE_ENABLED = _env_bool("HYDE_ENABLED", False)
HYDE_TIMEOUT = _env("HYDE_TIMEOUT", 6, int)
HYDE_MAX_TOKENS = _env("HYDE_MAX_TOKENS", 120, int)
RERANK_STAGE1_MODEL = _env("RERANK_STAGE1_MODEL", "all-MiniLM-L6-v2")
# 重排精度阈值: bge-reranker 得分低于此值的 chunk 过滤掉 (0=不过滤, 提升精度; 太高伤召回)

# 学术范围白名单闸门 (锁定用户只能问论文/学术相关问题, 默认开启)
SCOPE_GATE_ENABLED = _env_bool("SCOPE_GATE_ENABLED", True)

# ===================================================================
# Academic Search
# ===================================================================
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
APIFY_API_KEY = os.getenv("APIFY_API_KEY", "")
MAX_SEARCH_RESULTS = _env("MAX_SEARCH_RESULTS", 100, int)

# ===================================================================
# Storage Paths
# ===================================================================
PAPER_CACHE_DIR = PROJECT_ROOT / os.getenv("PAPER_CACHE_DIR", "paper_cache")
LANCEDB_DIR = PROJECT_ROOT / os.getenv("LANCEDB_DIR", "data/lancedb_data")
IMAGE_DIR = PROJECT_ROOT / os.getenv("IMAGE_DIR", "data/images")

# ===================================================================
# 文档解析质量闸门 (拦截"解析出来是垃圾"的 PDF, 不污染知识库)
# ===================================================================
# 实测 43 篇 arXiv 论文: 每页中位 3351 字 (最小 668), 乱码率中位 0.01% (最大 0.83%)。
# 阈值取得远离正常分布 → 只拦真坏的 (扫描件/字体未嵌入), 正常论文零误杀。
DOC_QUALITY_GATE_ENABLED = _env_bool("DOC_QUALITY_GATE_ENABLED", True)
DOC_QUALITY_MIN_CHARS_PER_PAGE = _env("DOC_QUALITY_MIN_CHARS_PER_PAGE", 200, int)
DOC_QUALITY_MAX_GARBAGE_RATIO = _env("DOC_QUALITY_MAX_GARBAGE_RATIO", 0.15, float)
# 字符单一度: 最高频字符占比。真实论文最大 12.4%, CID 乱码可达 90%+ → 50% 留 4 倍余量
DOC_QUALITY_MAX_DOMINATION_RATIO = _env("DOC_QUALITY_MAX_DOMINATION_RATIO", 0.50, float)

# 双栏重排: 页内正文块左右栏切换率 >= 此值 → 判定串栏, 按栏重排 (只换顺序, 不增删内容)
# 实测 29 篇有正文可判定的论文: 25 篇按栏正常 (切换率低), 4 篇中招 (切换率 0.32-0.43)
TWO_COLUMN_SWITCH_RATIO = _env("TWO_COLUMN_SWITCH_RATIO", 0.30, float)
TWO_COLUMN_REORDER_ENABLED = _env_bool("TWO_COLUMN_REORDER_ENABLED", True)
# PDF 表格结构化: get_text 把表格读成"竖着一列一列的数字流" (列名与数值糊在一起),
# 用 find_tables 检出后把该区域替换成 Markdown 管道表。只重新组织同一批单元格文本, 不增删。
TABLE_EXTRACT_ENABLED = _env_bool("TABLE_EXTRACT_ENABLED", True)
# PDF 章节识别: 按字号启发式(>正文字号 1.12 倍)检出标题并绑定到 chunk。
# PDF 没有 DOCX 那样的显式层级, 这是排版线索推断 —— 命中才用, 不命中留空, 不影响切分。
PDF_SECTION_ENABLED = _env_bool("PDF_SECTION_ENABLED", True)

# ===================================================================
# 工具调用重试 (仅对瞬时故障: 超时/连接/限流/5xx)
# ===================================================================
# 参数类错误 (缺少必填/invalid/404) 判为永久故障, 不重试 —— 重试只是把 3s 失败拖成 9s
TOOL_RETRY_MAX = _env("TOOL_RETRY_MAX", 2, int)
TOOL_RETRY_BASE_DELAY = _env("TOOL_RETRY_BASE_DELAY", 0.8, float)

# 死循环第二道防线 (第一道=轮次限制 max_steps, 第三道=超时 max_time):
# 同一工具用**相同参数**连续调用达到此次数 → 判定原地打转, 把观察结果换成警示让模型换路。
# 阈值 3 = 允许"重试一次仍失败"的正常情形, 第 3 次才判定为转圈。
AGENT_REPEAT_CALL_LIMIT = _env("AGENT_REPEAT_CALL_LIMIT", 3, int)

# ReAct 第三类故障: **错误累积**。单次失败让模型自行纠错是对的; 连续失败说明路线错了,
# 继续烧步数只会把错误累积进上下文, 最终交出建立在失败之上的答案。达到此次数即止损。
AGENT_MAX_CONSECUTIVE_FAILURES = _env("AGENT_MAX_CONSECUTIVE_FAILURES", 4, int)

# 三级兜底的分级参数:
# 一级 (参数错): 给模型几次**修正参数**的机会, 用完就禁止再调这个工具
#   —— 参数错是确定性的, 改不对说明模型没理解工具契约, 再试只是烧步数
# 三级 (多次失败): AGENT_MAX_CONSECUTIVE_FAILURES 触发后终止整个任务
AGENT_PARAM_CORRECTION_BUDGET = _env("AGENT_PARAM_CORRECTION_BUDGET", 1, int)

# ── 工具治理 ──
# 静态黑名单 (逗号分隔): 运维可在不发版的情况下停用某个工具 (如上游 API 挂了、
# 或某工具被判定有风险)。此前只有"危险工具"审计日志, **没有任何东西会真的拦**。
TOOL_DISABLED = _env_list("TOOL_DISABLED", [])
# 单次任务内, 同一工具最多被调用几次 (防"换着参数反复调"造成的成本失控 ——
# 重复动作检测只抓**同参数**的原地打转, 换参数连调 N 次它看不到)。
TOOL_CALL_LIMIT_PER_RUN = _env("TOOL_CALL_LIMIT_PER_RUN", 8, int)

# 工具结果缓存 (只对 Tool.cacheable=True 的只读工具生效)
TOOL_CACHE_ENABLED = _env_bool("TOOL_CACHE_ENABLED", True)
# TTL 必须存在: 缓存的语义是"短时间内答案不变", 不是"永远不变" ——
# 知识库更新后不能被旧结果挡住。
TOOL_CACHE_TTL = _env("TOOL_CACHE_TTL", 180, int)
TOOL_CACHE_MAX = _env("TOOL_CACHE_MAX", 512, int)

# 单个 agent 的 wall-clock 上限 (科研团队每个 agent 独立计时)。
# 此前只有轮次上限: 单次 LLM 调用各自有 timeout, 但 agent 可以一直"下一轮再说"。
AGENT_AGENT_TIMEOUT = _env("AGENT_AGENT_TIMEOUT", 180, int)

# 科研团队单个 agent 的**循环轮次**上限。
# 刻意与 pi_agent 的 AGENT_MAX_STEPS 分开: 团队一次跑 14 个 agent, 轮次预算翻倍
# 就是成本翻倍; 原来是硬编码的 8, 提为配置但**保持原值**, 不做无依据的放宽。
TEAM_AGENT_MAX_ROUNDS = _env("TEAM_AGENT_MAX_ROUNDS", 8, int)

# 科研团队规划器 (规划与执行分离): 按任务裁剪本次要跑的 agent。
# 兜底: 规划失败/空结果/关闭 → 回退完整管线; required=True 的 agent (含全部 3 个审批闸门)
# 永远必跑, 规划器碰不到。
TEAM_PLANNER_ENABLED = _env_bool("TEAM_PLANNER_ENABLED", True)

# ===================================================================
# 知识库版本/失效管理
# ===================================================================
# 只检索 active 的条目。老库没有 status 列(或补列后为空) → 视为 active, 否则检索会全空。
# 注: 条件里显式包含 status = '' 是为了兼容迁移前写入的行。
KB_ACTIVE_FILTER = "is_fulltext = true AND is_image = false AND (status = 'active' OR status = '')"

# 中文关键词召回 (BM25) 开关。
# 背景: LanceDB 默认 FTS 分词器是 simple/English, 无空格中文被当一个巨大 token ——
# "曲轴箱温度报警阈值是九十五摄氏度" 搜 "曲轴箱温度" 命中 0, 即 BM25 支路对中文
# **一直是死的**, "混合检索"实际只有向量一路 (实测 8/8 题 BM25 零召回)。
# 修复: 入库时用 jieba 分词另存 text_fts 列并索引该列, 查询侧同样分词。
#
# **实测权衡** (8 题 eval, 修复前后同一题库):
#   R@5  0.7694 → 0.7733 (微升)
#   MRR  0.6667 → 0.6042 (下降)
# 下降来自两条分支的候选池不对称: 有 BM25 时融合后只给重排 top_k*3, 无 BM25 时
# 直接给 top_k*6 —— 多一路召回反而让重排候选池减半。已在 hybrid_retriever 标注。
# n=8 且度量是近似语义相似度, 不足以据此调参; 保留修复 (精确标识符检索对工业手册
# 场景是刚需: 固件版本号/故障码/条款编号), 并留此开关以便按场景关闭。
KB_BM25_ENABLED = _env_bool("KB_BM25_ENABLED", True)

# ===================================================================
# Retrieval Limits
# ===================================================================
# 分块大小 (字符): 论文长, 2048 太小 → 检索到的段落覆盖不全 (答案跨块会丢)。
# 4096 让每个检索单元覆盖更全; 配合 1M 上下文窗口, prompt 可容纳整块。
CHUNK_SIZE = _env("CHUNK_SIZE", 4096, int)
# 分块重叠比例。docstring 一直写着 15%, 但实现是 min(200, ...) —— 在 4096 的块大小下
# 上限 200 先触发, **真实重叠只有 4.88%**, 与声称值和通行建议(15%-20%)都不符。
# 现改为比例驱动, 上限只做安全兜底 (不绑定)。
CHUNK_OVERLAP_RATIO = _env("CHUNK_OVERLAP_RATIO", 0.15, float)
CHUNK_OVERLAP_MAX = _env("CHUNK_OVERLAP_MAX", 1024, int)
RETRIEVAL_TOP_K = _env("RETRIEVAL_TOP_K", 5, int)

# ===================================================================
# LLM Call Parameters (all LLM calls use these defaults)
# ===================================================================
# Generation (answers, reports)
LLM_GEN_TEMPERATURE = _env("LLM_GEN_TEMPERATURE", 0.3, float)
# deepseek-v4-pro 是推理模型, max_tokens 同时计入推理+输出; 1024 在复杂 RAG prompt 下
# 推理可能耗尽预算 → 输出空串。提高兜底 (实际输出已在推理禁用后显著小于此值)。
LLM_GEN_MAX_TOKENS = _env("LLM_GEN_MAX_TOKENS", 2048, int)
LLM_GEN_TIMEOUT = _env("LLM_GEN_TIMEOUT", 30, int)

# Quick responses (agent/quick, translations)
LLM_QUICK_TEMPERATURE = _env("LLM_QUICK_TEMPERATURE", 0.1, float)
LLM_QUICK_MAX_TOKENS = _env("LLM_QUICK_MAX_TOKENS", 512, int)
LLM_QUICK_TIMEOUT = _env("LLM_QUICK_TIMEOUT", 15, int)

# Strategy generation (search strategy, entity extraction)
LLM_STRATEGY_TEMPERATURE = _env("LLM_STRATEGY_TEMPERATURE", 0.3, float)
LLM_STRATEGY_MAX_TOKENS = _env("LLM_STRATEGY_MAX_TOKENS", 2000, int)
LLM_STRATEGY_TIMEOUT = _env("LLM_STRATEGY_TIMEOUT", 30, int)

# Translation (CN→EN keyword translation)
LLM_TRANSLATE_TEMPERATURE = _env("LLM_TRANSLATE_TEMPERATURE", 0.0, float)
LLM_TRANSLATE_MAX_TOKENS = _env("LLM_TRANSLATE_MAX_TOKENS", 50, int)
LLM_TRANSLATE_TIMEOUT = _env("LLM_TRANSLATE_TIMEOUT", 6, int)

# Coreference resolution (multi-turn)
LLM_COREF_TEMPERATURE = _env("LLM_COREF_TEMPERATURE", 0.1, float)
LLM_COREF_MAX_TOKENS = _env("LLM_COREF_MAX_TOKENS", 100, int)
LLM_COREF_TIMEOUT = _env("LLM_COREF_TIMEOUT", 8, int)

# Query expansion
LLM_EXPAND_TEMPERATURE = _env("LLM_EXPAND_TEMPERATURE", 0.3, float)
LLM_EXPAND_MAX_TOKENS = _env("LLM_EXPAND_MAX_TOKENS", 200, int)
LLM_EXPAND_TIMEOUT = _env("LLM_EXPAND_TIMEOUT", 10, int)

# ===================================================================
# Search Timeouts (per source, seconds)
# ===================================================================
SEARCH_TIMEOUT_ARXIV = _env("SEARCH_TIMEOUT_ARXIV", 6, int)
SEARCH_TIMEOUT_OPENALEX = _env("SEARCH_TIMEOUT_OPENALEX", 6, int)
SEARCH_TIMEOUT_OPENALEX_CN = _env("SEARCH_TIMEOUT_OPENALEX_CN", 6, int)
SEARCH_TIMEOUT_S2 = _env("SEARCH_TIMEOUT_S2", 4, int)
SEARCH_TIMEOUT_CORE = _env("SEARCH_TIMEOUT_CORE", 5, int)
SEARCH_TIMEOUT_CROSSREF = _env("SEARCH_TIMEOUT_CROSSREF", 5, int)
SEARCH_TIMEOUT_DOAJ = _env("SEARCH_TIMEOUT_DOAJ", 5, int)
SEARCH_TIMEOUT_PUBMED = _env("SEARCH_TIMEOUT_PUBMED", 5, int)
SEARCH_TIMEOUT_PWC = _env("SEARCH_TIMEOUT_PWC", 5, int)
SEARCH_TIMEOUT_EPMC = _env("SEARCH_TIMEOUT_EPMC", 5, int)
SEARCH_TIMEOUT_GS_APIFY = _env("SEARCH_TIMEOUT_GS_APIFY", 3, int)
SEARCH_TIMEOUT_CNKI = _env("SEARCH_TIMEOUT_CNKI", 2, int)
SEARCH_TIMEOUT_UNPAYWALL = _env("SEARCH_TIMEOUT_UNPAYWALL", 4, int)

# ===================================================================
# Search Limits (per API call)
# ===================================================================
SEARCH_LIMIT_ARXIV = _env("SEARCH_LIMIT_ARXIV", 30, int)
SEARCH_LIMIT_OPENALEX = _env("SEARCH_LIMIT_OPENALEX", 60, int)
SEARCH_LIMIT_OPENALEX_CN = _env("SEARCH_LIMIT_OPENALEX_CN", 60, int)
SEARCH_LIMIT_S2 = _env("SEARCH_LIMIT_S2", 20, int)
SEARCH_LIMIT_CORE = _env("SEARCH_LIMIT_CORE", 15, int)
SEARCH_LIMIT_CROSSREF = _env("SEARCH_LIMIT_CROSSREF", 15, int)
SEARCH_LIMIT_DOAJ = _env("SEARCH_LIMIT_DOAJ", 15, int)
SEARCH_LIMIT_PUBMED = _env("SEARCH_LIMIT_PUBMED", 15, int)
SEARCH_LIMIT_PWC = _env("SEARCH_LIMIT_PWC", 15, int)
SEARCH_LIMIT_EPMC = _env("SEARCH_LIMIT_EPMC", 20, int)
SEARCH_LIMIT_UNPAYWALL = _env("SEARCH_LIMIT_UNPAYWALL", 15, int)

# ===================================================================
# Agent
# ===================================================================
# 默认 15 = 前端实际发送值 (此前 config=10 / api 默认=20 / 前端=15 三方不一致, 配置从未生效)。
# 写初稿路径前端会显式传更大的值 (30)。
AGENT_MAX_STEPS = _env("AGENT_MAX_STEPS", 15, int)
# Agent 单轮输出上限: 1000 → 4096。Agent 产出是"交付物"(综述/章节/审稿),
# 1000 token 会截断长回答 → 体验差。设宽以容纳完整交付物; 代价是单轮延迟/成本上升。
# 更健壮的方案是检测 finish_reason=="length" 后续写 (未来可做)。
AGENT_MAX_TOKENS = _env("AGENT_MAX_TOKENS", 4096, int)
AGENT_TEMPERATURE = _env("AGENT_TEMPERATURE", 0.3, float)
# Agent 上下文窗口 (context_engine): DeepSeek v4-pro 实际 1M 上下文。
# 论文问答高精度场景: 窗口越大模型看到的证据越多, 幻觉越少 → 用满模型真实窗口。
# reserve 是给系统 prompt + 新输出预留的空间。
AGENT_CONTEXT_MAX_TOKENS = _env("AGENT_CONTEXT_MAX_TOKENS", 1_000_000, int)
AGENT_CONTEXT_RESERVE_TOKENS = _env("AGENT_CONTEXT_RESERVE_TOKENS", 8192, int)
# Agent 反思重试闭环 (loop-engineering 第一环): 直答后轻量自批判,
# 答案不完整 → 带批判意见重试一轮 (有界, 最多 1 次)。false=关闭 (省一次 LLM 调用)。
AGENT_REFLEXION_ENABLED = _env_bool("AGENT_REFLEXION_ENABLED", True)
# Agent 可选 plan 阶段 (loop-engineering 规划层): 长任务先让 LLM 拆 3-6 步计划
# 再执行, 防纯 ReAct 长任务 (整篇论文) 跑偏。仅对长度 ≥ 阈值的任务触发 (短问答不规划省延迟)。
AGENT_PLAN_ENABLED = _env_bool("AGENT_PLAN_ENABLED", True)

# Plan 模式: 生成计划后暂停等人工确认 (计划是长任务的"方向闸", 错方向先拦下)
AGENT_PLAN_REQUIRE_APPROVAL = _env_bool("AGENT_PLAN_REQUIRE_APPROVAL", True)
AGENT_PLAN_MIN_CHARS = _env("AGENT_PLAN_MIN_CHARS", 60, int)

# Hook BEFORE_EMIT 敏感词表 (空列表 = 不拦截)。默认仅保留高危命令类,
# 不含 "execute arbitrary code" 等学术安全论文常见词组 (避免误杀)。
HOOK_SENSITIVE_WORDS = _env_list("HOOK_SENSITIVE_WORDS", ["rm -rf /", "DROP TABLE"])
# 工具安全: 非白名单域名是否直接阻止 (True = 严格模式; 默认 False 仅告警,
# 避免阻断出版商等非白名单合法下载源)
TOOL_SECURITY_BLOCK_NON_WHITELIST = _env_bool(
    "TOOL_SECURITY_BLOCK_NON_WHITELIST", False
)

# ===================================================================
# Upload / Download 资源限制 (DeepSeek 审查 #14/#15)
# ===================================================================
# 单 PDF 上传大小上限 (字节)
MAX_UPLOAD_BYTES = _env("MAX_UPLOAD_BYTES", 50 * 1024 * 1024, int)
# /download_bulk 并发下载数 (信号量)
DOWNLOAD_CONCURRENCY = _env("DOWNLOAD_CONCURRENCY", 6, int)
DOWNLOAD_TIMEOUT = _env("DOWNLOAD_TIMEOUT", 20, int)

# ===================================================================
# Conversation Memory
# ===================================================================
# 记忆上限: key_findings 最多保留条数 (超限丢最旧) — 防无限增长
MEMORY_MAX_FINDINGS = _env("MEMORY_MAX_FINDINGS", 100, int)
# 记忆数据层过期: 已结题/归档课题的记忆保留天数, 超期清理 (0=不清理)
MEMORY_ARCHIVE_TTL_DAYS = _env("MEMORY_ARCHIVE_TTL_DAYS", 90, int)
MEMORY_MAX_RECENT = _env(
    "MEMORY_MAX_RECENT", 5, int
)  # max recent turns kept in full text
MEMORY_HISTORY_TURNS = _env(
    "MEMORY_HISTORY_TURNS", 8, int
)  # max history turns sent to LLM

# ===================================================================
# Rate Limiting
# ===================================================================
RATE_LIMIT_REQUESTS_PER_SEC = _env("RATE_LIMIT_REQUESTS_PER_SEC", 30, int)

# ===================================================================
# 按量付费配额 (付费闭环: 免费额度体验 → 用量追踪 → 超额充值)
# ===================================================================
QUOTA_ENABLED = _env_bool("QUOTA_ENABLED", True)
QUOTA_FREE_TOKENS = _env("QUOTA_FREE_TOKENS", 100_000, int)  # 免费额度 (LLM token 数)

# Stripe 支付 (真实收款: 充值配额)
STRIPE_SECRET_KEY = os.getenv("STRIPE_SECRET_KEY", "")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET", "")
STRIPE_PRICE_PER_1000_TOKENS = _env(
    "STRIPE_PRICE_PER_1000_TOKENS", 0.05, float
)  # $ / 千 token

# ===================================================================
# API Server
# ===================================================================
API_HOST = _env("API_HOST", "0.0.0.0", str)
API_PORT = _env("API_PORT", 8001, int)

# ===================================================================
# CORS
# ===================================================================
CORS_ORIGINS = _env_list(
    "CORS_ORIGINS", ["http://localhost:8001", "http://127.0.0.1:8001"]
)

# ===================================================================
# Frontend Model Options
# ===================================================================

# ===================================================================
# RAG Pipeline Parameters
# ===================================================================
# Query rewrite: skip expansion for short queries
REWRITE_MIN_LENGTH = _env("REWRITE_MIN_LENGTH", 15, int)
# 注: 改写语义校验阈值见下方 EXPANSION_SIMILARITY_THRESHOLD_BASE / _ADAPTIVE
# Coreference resolution: only try if question < this length
COREF_MAX_LENGTH = _env("COREF_MAX_LENGTH", 20, int)
# Complex query: trigger rerank if intent matches these types
COMPLEX_INTENT_TYPES = _env_list("COMPLEX_INTENT_TYPES", ["对比", "原理", "最新"])

# ===================================================================
# Intent Classification Keywords (5 classes)
# ===================================================================
# 内容型数据: 意图分类关键词表, 代码常驻 (非配置旋钮)
INTENT_KEYWORDS = {
    "定义": ["什么是", "是什么", "定义", "概念", "define", "what is"],
    "对比": ["对比", "区别", "优缺点", "比较", "vs", "compare", "哪个更好", "差异"],
    "实操": ["如何", "怎么", "怎样", "方法", "步骤", "how to", "流程", "实现"],
    "最新": ["最新", "前沿", "趋势", "2024", "2025", "2026", "recent", "latest", "突破", "进展"],
    "原理": ["为什么", "原理", "机制", "原因", "why", "mechanism", "证明"],
}

# Intent complexity scores — 替代二元匹配，为意图路由提供连续权重
# 高值 → 更可能触发重排序、更保守的改写阈值
INTENT_COMPLEXITY_SCORES = {
    "定义": 0.3,  # 简单事实检索，不触发重排
    "实操": 0.5,  # 中等复杂度
    "最新": 0.7,  # 较复杂，触发 light 重排
    "原理": 0.8,  # 复杂，需要深度推理
    "对比": 0.9,  # 最复杂，触发 full 重排
    "事实": 0.2,  # default fallback
}

# ===================================================================
# Intent Semantic Layer (三层漏斗第 2 层: 嵌入语义分类)
# ===================================================================
# 规则层未命中时, 用嵌入语义相似度兜底分类意图/技能 (见 intent_classifier.py)。
# 判定标准 (bge 系绝对相似度挤在 ~0.7-0.9 窄带, 单阈值不可靠):
#   INTENT_SEMANTIC_MARGIN — 最优 vs 次优 余弦间距 ≥ 此值才采信
#   INTENT_SEMANTIC_FLOOR  — 且最优绝对分数 ≥ 此值
# 不满足 → 返回 None, 交给大模型+工具层 (pi_agent Function Calling) 兜底。
# MARGIN 越大越保守 (更多请求交给 LLM), 越小越激进 (语义层接管更多常规意图)。
INTENT_SEMANTIC_ENABLED = _env_bool("INTENT_SEMANTIC_ENABLED", True)
INTENT_SEMANTIC_MARGIN = _env("INTENT_SEMANTIC_MARGIN", 0.06, float)
INTENT_SEMANTIC_FLOOR = _env("INTENT_SEMANTIC_FLOOR", 0.60, float)

# ===================================================================
# Intelligent Rerank Trigger (替代 hardcoded COMPLEX_INTENT_TYPES)
# ===================================================================
# 重排序触发条件: 意图复杂度 >= 此阈值 → 触发
RERANK_INTENT_THRESHOLD = _env("RERANK_INTENT_THRESHOLD", 0.6, float)
# 长问句兜底: 超过此长度即使意图分低也重排 (长问句常含多约束/多跳)
RERANK_LONG_QUERY_MIN_LEN = _env("RERANK_LONG_QUERY_MIN_LEN", 15, int)

# ===================================================================
# Adaptive Retrieval Parameters
# ===================================================================
# _compute_adaptive_weight coefficients
ADAPTIVE_SHORT_QUERY_BOOST = _env(
    "ADAPTIVE_SHORT_QUERY_BOOST", 0.2, float
)  # <15 chars → BM25 boost
ADAPTIVE_ACRONYM_BOOST = _env(
    "ADAPTIVE_ACRONYM_BOOST", 0.15, float
)  # acronyms → BM25 boost
ADAPTIVE_QUESTION_PENALTY = _env(
    "ADAPTIVE_QUESTION_PENALTY", -0.15, float
)  # question-form → BM25 penalty
ADAPTIVE_LONG_QUERY_PENALTY = _env(
    "ADAPTIVE_LONG_QUERY_PENALTY", -0.1, float
)  # >40 chars → BM25 penalty
ADAPTIVE_SHORT_THRESHOLD = _env("ADAPTIVE_SHORT_THRESHOLD", 15, int)
ADAPTIVE_LONG_THRESHOLD = _env("ADAPTIVE_LONG_THRESHOLD", 40, int)
ADAPTIVE_DEFAULT_ALPHA = _env("ADAPTIVE_DEFAULT_ALPHA", 0.5, float)  # balanced default
ADAPTIVE_ALPHA_MIN = _env("ADAPTIVE_ALPHA_MIN", 0.05, float)
ADAPTIVE_ALPHA_MAX = _env("ADAPTIVE_ALPHA_MAX", 0.95, float)

# ===================================================================
# Image Filter — 多阶段规则预筛 (在调用多模态模型前过滤无用图)
# ===================================================================
IMAGE_MIN_PIXELS = _env("IMAGE_MIN_PIXELS", 100, int)
IMAGE_MAX_ASPECT_RATIO = _env("IMAGE_MAX_ASPECT_RATIO", 6.0, float)
IMAGE_MIN_COLORS = _env("IMAGE_MIN_COLORS", 5, int)

# Stage 1: 尺寸+比例 (快速淘汰, ~90%噪声)
IMAGE_FILTER_STAGE1_ENABLED = _env_bool("IMAGE_FILTER_STAGE1_ENABLED", True)
# Stage 2: 位置过滤 — 首页/末页跳过
IMAGE_FILTER_SKIP_FIRST_PAGE = _env_bool("IMAGE_FILTER_SKIP_FIRST_PAGE", True)
IMAGE_FILTER_SKIP_LAST_PAGE = _env_bool("IMAGE_FILTER_SKIP_LAST_PAGE", True)

# ===================================================================
# Spam/Quality Filter Keywords (内容型数据, 代码常驻)
# ===================================================================
SPAM_KEYWORDS = [
    "征稿",
    "会议通知",
    "征文",
    "订阅",
    "广告",
    "约稿",
    "稿约",
    "投稿须知",
    "撤稿",
    "抄袭",
    "学术不端",
    "书评",
    "读者来信",
    "简讯",
    "新闻",
]

# Source credibility weights (higher = more trustworthy)
SOURCE_WEIGHTS = {
    "Semantic Scholar": 10,
    "OpenAlex": 8,
    "arXiv": 7,
    "baidu_xueshu": 3,
    "nssd": 5,
}

# ===================================================================
# Prompt Injection Detection Patterns (内容型数据, 代码常驻)
# ===================================================================
INJECTION_PATTERNS = [
    r"忽略.*指令",
    r"ignore.*instruction",
    r"system\s*:",
    r"你是一个",
    r"you are a",
    r"忘记.*规则",
    r"forget.*rule",
    r"切换角色",
    r"switch.*role",
    r"<<SYS>>",
    r"\[INST\]",
    r"\[SYSTEM\]",
    r"DAN\s*:",
    r"jailbreak",
]

# ===================================================================
# CN→EN Translation — 智能化 (内容型数据, 代码常驻)
# ===================================================================
# LLM 翻译缓存: 已翻译过的查询存本地 JSON, 避免重复调用 LLM
CN_EN_TERMS = {
    "计算机视觉": "computer vision",
    "机器学习": "machine learning",
    "深度学习": "deep learning",
    "自然语言处理": "natural language processing",
    "强化学习": "reinforcement learning",
    "图像识别": "image recognition",
    "目标检测": "object detection",
    "语义分割": "semantic segmentation",
    "人脸识别": "face recognition",
    "知识图谱": "knowledge graph",
    "迁移学习": "transfer learning",
    "联邦学习": "federated learning",
    "图神经网络": "graph neural network",
    "自动驾驶": "autonomous driving",
    "人工智能": "artificial intelligence",
    "大语言模型": "large language model",
    "扩散模型": "diffusion model",
    "多模态": "multimodal",
    "神经网络": "neural network",
    "数据挖掘": "data mining",
    "机器人": "robot",
    "视觉": "vision",
    "图像": "image",
    "视频": "video",
    "检测": "detection",
    "分割": "segmentation",
    "识别": "recognition",
    "分类": "classification",
}

# 翻译缓存: 本地 JSON 文件, 命中即用, 无需 LLM 调用
TRANSLATION_CACHE_ENABLED = _env_bool("TRANSLATION_CACHE_ENABLED", True)
TRANSLATION_CACHE_FILE = _env("TRANSLATION_CACHE_FILE", "data/translation_cache.json")
TRANSLATION_CACHE_MAX_ENTRIES = _env("TRANSLATION_CACHE_MAX_ENTRIES", 500, int)

# jieba 分词: 中文输入自动分词, 组合为多词英文查询提升覆盖率
USE_JIEBA_TOKENIZATION = _env_bool("USE_JIEBA_TOKENIZATION", True)
# 分词后取前 N 个关键词 (按 TF-IDF 权重排序, 若 jieba 不可用则取前 N 个按长度)
CN_TOKENIZATION_TOP_K = _env("CN_TOKENIZATION_TOP_K", 5, int)

# ===================================================================
# Query Expansion — 自适应语义校验阈值
# ===================================================================
# 基础阈值 (原 EXPANSION_SIMILARITY_THRESHOLD)
EXPANSION_SIMILARITY_THRESHOLD_BASE = _env(
    "EXPANSION_SIMILARITY_THRESHOLD_BASE", 0.75, float
)
# 自适应调整: 短查询 (<10字) → 更严格 (0.85), 长查询 (>50字) → 更宽松 (0.60)
EXPANSION_SIMILARITY_ADAPTIVE = {
    "short": {"threshold": 0.85, "max_len": 10},
    "long": {"threshold": 0.60, "min_len": 50},
}
