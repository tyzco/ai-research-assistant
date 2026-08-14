# AI Agent 工程原理 × PaperLink 落地对照手册

> 从「AI Agent 工程知识要点」出发整理。每个主题分三层：
> **① 问题/概念** —— 这到底是什么、难在哪
> **② 通用解决方案** —— 业界/大厂怎么做
> **③ 本项目落地** —— PaperLink 怎么做的（可指着代码讲）
>
> PaperLink = 面向 C 端研究者的「选题 → 初稿」Agent 即服务：检索 12 源论文 → 建私有知识库（LanceDB）→ 证据问答 → 辅助写作 → 质量验收 → 审稿批注。按量付费。

---

## 0. 全景图：Agent 工程六大模块

概念层把工业级 AI Agent 拆成 **上下文工程、工具编排、验证机制、状态管理、可观测性、人类接管** 六大模块；执行层是 **感知用户任务 → 规划拆分步骤 → 调用工具执行 → 反思结果并循环**。

```
用户任务
  │
  ├─ 上下文工程  角色隔离 / 信息裁剪 / 格式控制 / 来源分层   (context_engine.py)
  ├─ 工具编排    何时调用 / 参数校验 / 结果规范化              (tool_registry + tool_manager)
  ├─ 验证机制    参数 schema 校验 / 工具超时重试 / 质量门       (Tool.execute + draft_quality)
  ├─ 状态管理    多轮记忆 / 上下文压缩 / 证据保留               (conversation_memory + context_engine)
  ├─ 可观测性    trace / metrics / 日志 / 逐步推送              (monitor + SSE)
  └─ 人类接管    危险工具审批门 / 主动路由建议                  (need_confirm + /suggest)
```

**PaperLink 模块地图**（可指着 `CLAUDE.md` 讲）：

| 概念层模块 | PaperLink 实现文件 |
|---|---|
| 上下文工程 | `agent/context_engine.py`（1M 窗口 + key_findings 证据保留 + 压缩）|
| 工具编排 | `agent/tool_registry.py`（17 工具）+ `tools_extra/` 插件 + `agent/tool_manager.py` |
| 验证机制 | `Tool.execute` schema 校验 + 3 次退避重试；`draft_quality.py` 6 项质量门 |
| 状态管理 | `conversation_memory.py` + `agent/skills.py`(P3 技能记忆) + `feedback_lessons.py`(P2) |
| 可观测性 | `monitor.py` + `/metrics` + 全局异常处理 + agent 执行 trace（SSE）|
| 人类接管 | `need_confirm` 审批门 + `research_team` 3 道审批闸 + `/suggest` 主动路由 |

---

## 一、Agent 核心机制

### 1.1 Agent 定义与执行循环

**① 概念/问题**：Agent 的早期定义是"大模型智能对话"，现在工业级标准是**执行循环**：感知任务 → 规划拆分 → 调工具 → 反思结果 → 再循环。难点在：规划要能拆到可执行步骤、循环要能收敛不死循环、反思要能识别失败并修正。

**② 通用方案**：主流架构四类 —— **ReAct**（推理+行动交错，最通用）、**Plan-and-Execute**（先规划再执行，适合长任务）、**路由模式**（意图→不同链路）、**Workflow 嵌入**（固定流程里嵌 Agent，适合确定性高的业务）。实现上分 **Framework 路线**（ReActAgent，Hooks 增强/终止，可维护性好）和 **Graph 路线**（节点/状态/边，与流程图一一映射，复杂场景适配强）。

**③ 本项目落地**：`agent/pi_agent.py` 走 **Framework + Hook 路线**——HookRegistry 生命周期钩子（trace_start / tool_call / trace_end 等），内置 `_max_steps_guard`（步数上限）、`_tool_allowlist`（工具白名单）等守卫；`_reflexion_critique` 实现**反思重试**：直接回答后自我批判一轮，不合格再修订。`research_team.py` 的 13-agent 团队则像 Graph 路线：上游（选题评估→检索→质筛）→中游（综述/章节写作）→下游（审稿/格式/伦理/终稿），每段之间有明确节点。`langgraph_workflow.py` 早期原型已被 pi_agent 取代（框架路线更贴合本项目的审批流）。

### 1.2 Function Calling 与 Skill 机制

**① 概念/问题**：LLM 本身不能执行动作，需要把能力暴露为**函数签名**（名称/描述/JSON Schema），模型选函数、填参数、返回 tool_call。**Skill** 本质基于 Function Call：用 Markdown 文件（元数据 + 执行指令）实现**能力按需加载**——不把所有工具塞进每轮，而是按任务动态挂载对应技能集，省 token、减干扰。

**② 通用方案**：OpenAI Function Calling 是事实标准；Spring AI Alibaba 内置 Skill 实现（Skill 文件 + 脚本即可对接任意大模型）；工具侧做 schema 校验、参数补全、自动重试。

**③ 本项目落地**：`agent/tool_registry.py` 用统一 `Tool` 类声明 17 个工具（name/description/JSON Schema/execute/need_confirm），转成 OpenAI schema 发给 DeepSeek；`Tool.execute` 做必填校验 + 3 次退避重试。`agent/skills.py` 定义 7 个技能（文献综述/论文写作/深度问答/科研决策/引用链/批判/实验设计），`SkillRegistry.select_for_task` 按意图关键词 + 语义路由**动态裁剪工具集**（保留 CORE_TOOLS 地板，防技能选择造成能力缺口）。`tools_extra/` 插件目录丢 .py 即注册，`/agent/tools/reload` 热加载——这就是"能力按需加载"的轻量实现。

### 1.3 Agent 编排四层架构

**① 概念/问题**：Agent 编排让 AI 从"单轮问答工具"变成"业务流程全环节执行者"。四层：**工具编排**（自主判断调哪个工具）、**流程编排**（任务顺序/校验/返工）、**多 Agent 编排**（拆任务分工协作）、**运行时编排**（状态/权限/人工确认/失败重试）。

**② 通用方案**：工具编排靠 ReAct 循环；流程编排用 Workflow/DAG；多 Agent 用角色拆分 + 消息传递；运行时编排是工程底线（状态机 + 审批门 + 重试）。

**③ 本项目落地**：四层都有——
- 工具编排：pi_agent ReAct 循环 + 17 工具；
- 流程编排：`research_team` 13-agent 流水线（上游→中游→下游）；
- 多 Agent：选题评估/检索/质筛/综述/写作/审稿各司其职，`topic_manager` 共享状态；
- 运行时：**3 道人工审批门**（🔴 Gate1 选题方向 / Gate2 核心文献 / Gate3 终稿结论）——双轨机制：Agent 自动跑，到闸门前端暂停、用户确认或给反馈才继续。这就是"人类接管"的落地。

### 1.4 Agent 死循环 / 长任务跑偏

**① 概念/问题**：Agent 可能反复调同一工具不收敛（死循环），或长任务中途偏离原目标（跑偏）。面试高频题。

**② 通用方案**：步数/时间/成本上限；循环去重（相同 tool_call 幂等）；反思阶段校验是否偏离目标；Plan-and-Execute 把目标写进上下文持续对齐。

**③ 本项目落地**：`hooks._max_steps_guard`（步数上限，超限终止）+ `max_time` 上限；`Tool.execute` 对相同参数自动重试但设上限；`_reflexion_critique` 每轮对照 `goal` 自批判防跑偏；`context_engine` 把 `key_findings`（已确立的关键结论）持久注入，避免长任务丢目标。

### 1.5 多 Agent 协作

**① 概念/问题**：单一 Agent 上下文有限，复杂任务拆给多个角色 Agent 并行/流水线协作，但引入通信成本、职责重叠、状态一致性问题。

**② 通用方案**：角色拆分（每个 Agent 专一职责）+ 编排器汇总；消息传递 vs 共享状态；审批门做安全绳。

**③ 本项目落地**：`research_team.py` 13 个 Agent 分三段（上游 3 / 中游 6 / 下游 4），职责单一是**为审计方便**（每步可查哪个 Agent 产出了什么）；`run_id` 多用户隔离；3 道审批门保证协作不失控。

---

## 二、上下文与记忆

### 2.1 上下文工程

**① 概念/问题**：Agent 的质量上限 = 上下文质量上限。核心四件事：**角色隔离**（系统/工具/用户分层）、**信息裁剪**（只给相关的）、**格式控制**（工具结果结构化）、**来源分层**（区分用户输入/检索证据/模型生成）。做不到就**目标漂移与决策污染**——模型被无关信息带偏。

**② 通用方案**：系统提示词 + 检索证据 + 对话历史分层拼装；证据带来源标注；对长上下文做压缩/截断。

**③ 本项目落地**：`agent/context_engine.py` 直接把上下文窗口设成 `AGENT_CONTEXT_MAX_TOKENS`（默认 1M，用满真实窗口），`key_findings` 进压缩模板保留"已确立结论"；工具输出截断（大结果只留关键字段）；`qa_engine.py` 三层 Prompt（指令/证据/问题），回答带引用标注，**来源可追溯**。

### 2.2 上下文压缩与 token 成本（kv 缓存 / 前缀缓存）

**① 概念/问题**：长对话/长任务 token 爆炸 → 成本高、超窗口。优化手段：上下文压缩（历史摘要）、**前缀缓存 / kv 缓存**（相同 system prompt + 历史前缀复用计算）、工具输出瘦身。

**② 通用方案**：滑动窗口 + 摘要替换老历史；system prompt 固定放最前以命中前缀缓存；结构化输出省 token。

**③ 本项目落地**：`ContextEngine.maybe_compact()` 超限自动压缩（`estimate_tokens` 记账 + `reserve_tokens` 预留）；`finish_reason=="length"` **断点续写**（不污染 messages，从断点继续）；工具结果 `str()[:3000]` 截断；skill 裁剪工具集 = 每轮省 token 的前置手段。

### 2.3 记忆机制（短时 / 长时 / 永久 + 向量库 + 知识图谱）

**① 概念/问题**：工业级 Agent 需要**短时记忆**（会话内）、**长时记忆**（缓存）、**永久记忆**（跨会话），配合向量数据库检索，并用**知识图谱为精准数据兜底**。多数中小公司因成本放弃知识图谱——不是不需要，是做不起。

**② 通用方案**：会话历史存内存/DB；关键结论存向量库按需召回；用户画像/偏好存用户表；图谱做实体关系精准检索。

**③ 本项目落地**：三层记忆全有——
- 短时：`conversation_memory.py`（SQLite 多轮，按 topic 键）；
- 永久：`feedback_lessons.py`（P2：低分反馈 → LLM 提炼教训 → 注入下次 system prompt）与 `agent_skills.py`（P3：成功任务模式 → 跨会话复用）；
- 图谱：`knowledge_graph.py` 实体抽取 + D3 可视化（`/kg_data`）——本项目做的是**轻量图谱**（实体关系图），不冒充完整 GraphRAG，诚实标注边界。

### 2.4 多轮上下文保持

**① 问题**：追问"那它呢"这种指代，脱离上文无法回答；刷新页面/换会话就丢记忆。

**② 通用方案**：指代消解（coref）+ 历史回填；会话持久化；关键词记忆。

**③ 本项目落地**：`qa_engine` 第 1 步**指代消解**（先改写问题再检索）；`/chat` 带 history 数组；`topic_manager` + `conversation_memory` 支持**刷新恢复会话**（前端 localStorage + 后端 SQLite）。

---

## 三、RAG 检索增强（垂直落地）

### 3.1 垂直 RAG 落地原则

**① 概念/问题**：垂直领域 RAG 的核心不是"能用"，是**数据规划围绕召回率、精准排序率**。领域数据噪声大、术语私、分布长尾，直接上通用 pipeline 效果差。

**② 通用方案**：数据清洗与标注（去噪、标注业务实体）→ 分层分块 → 多路召回 → 重排 → 反馈闭环。

**③ 本项目落地**：私有 KB 是护城河。`downloader.parse_pdf`（PyMuPDF 提全文）+ `knowledge_base` 双层 LanceDB（向量 + FTS）；检索策略针对**不常见选题**专门调过：双语重排 + 引用对数提权 + 领域联想词 + CN→EN 兜底 + 4096 分块（`docs/BENCHMARK.md` 记录 A/B：4096 > 2048 +1.4%）。

### 3.2 数据清洗与分块

**① 概念/问题**：**切分力度**要兼顾细粒度检索与上下文连贯性——块太小丢上下文，块太大召回噪声；表格/扫描件是难点。

**② 通用方案**：语义分块（按段落/标题）；父子文档（父块供上下文、子块供检索）；表格转 Markdown；扫描件 OCR。

**③ 本项目落地**：`CHUNK_SIZE=4096` 大块 + **邻块 ±2 扩展**（检索命中后拉上下文窗口，`context_engine`/`qa_engine` 拼接）；`downloader` 支持图像提取（`parse_pdf` 返回全文 + 图片）；处理的是 PDF 文本流（扫描件 OCR 为预留扩展点）。

### 3.3 Query 改写

**① 概念/问题**：用户问题口语化/含指代/中英混，直接检索匹配差。

**② 通用方案**：同义词替换、逻辑扩展、中英互译兜底。

**③ 本项目落地**：`qa_engine` 第 1-2 步 coref → 查询改写；`search.py._translate_cn_query` **CN→EN 翻译兜底**（中文 query 检索不到时翻英文再查）；`_quick_keywords` 本地生成关键词。

### 3.4 层级召回与混合检索（BM25 vs Embedding）

**① 概念/问题**：BM25 精确词匹配（术语/编号强）vs Embedding 语义匹配（同义/语义强），单用任一都偏。**层级召回**= 先粗召回再精排，控制候选集。

**② 通用方案**：BM25 + 向量双路召回 → RRF/加权融合；两阶段重排（轻量粗排 → 精排模型）。

**③ 本项目落地**：`hybrid_retriever.py` **自适应 RRF**：按问题特征（长度/是否英文术语/是否短查询）动态调 BM25 与向量权重（`_compute_adaptive_weight`）；召回后**两级级联重排**：Stage1 MiniLM 快速过滤 → Stage2 bge-reranker 精排；`search.py` 12 源并行检索（arXiv/OpenAlex×2/CNKI/Semantic Scholar…）先做论文级召回。

### 3.5 重排序与融合（Rerank / RRF / 引用提权）

**① 概念/问题**：向量相似度≠学术相关度。领域信号（引用量、领域术语）能显著提权，但要让长尾论文也能浮出。

**② 通用方案**：BERT 类 rerank 模型重排；RF 倒序融合多路；业务规则加权。

**③ 本项目落地**：`search.py._rerank_by_relevance` **双语 query 嵌入 + 引用对数提权**（`log(1+citations)` 压缩，防止高引论文霸榜）；`hybrid_retriever` 两阶段重排；`qa_engine` 父子分块扩展增强上下文关联。

### 3.6 GraphRAG vs 普通向量 RAG

**① 概念/问题**：向量 RAG 擅长"语义相似"，弱于"关系/多跳/聚合"问题（谁引用了谁、跨实体总结）。GraphRAG 建实体-关系图做精准检索，成本高。

**② 通用方案**：小团队一般向量 RAG + 轻量图谱；真图谱 RAG 需要抽取 + 图存储 + 图查询。

**③ 本项目落地**：`knowledge_graph.py` 做**论文实体抽取 + D3 可视化**（供展示/辅助）；核心链路仍走向量 RAG（成本可控）。在文档里诚实声明：**本项目图谱是可视化辅助，不是完整 GraphRAG**——这正好回答"为什么小公司不做知识图谱"。

### 3.7 RAG 评测（MRR / R@5，表格/扫描件）

**① 概念/问题**：RAG 评测关键指标 **MRR**（平均倒数排名：答案排第 1=1 分，排第 2=0.5…）、召回率（R@k）。要防止"评测集太简单 → 分数虚高"的自欺。

**② 通用方案**：多主题、真实标注问答对；划分训练/测试；指标拆解验证。

**③ 本项目落地**：`run_benchmark.py` 多主题基准（43 篇 / 30 题 / 5 主题）→ **MRR 0.72 / R@top5 0.78**，逐主题拆解 + 环境/局限自动写入报告（`data/benchmark_report.json`）。**诚实声明**：这是小样本 sanity 分数，不做大规模性能承诺（`docs/BENCHMARK.md`）。

---

## 四、工程与可靠性

### 4.1 输出稳定性（稳定输出 JSON / 结构化）

**① 问题**：LLM 偶发输出非 JSON / 字段缺失 / 工具参数类型错，直接崩下游。

**② 通用方案**：schema 约束 + 校验 + 重试；结果规范化。

**③ 本项目落地**：工具参数 JSON Schema 校验（`Tool.execute` 查 required）；所有工具返回**统一 dict 结构**（`json.dumps` 下发）；解析失败重试；`monitor.safe_call` 兜底。

### 4.2 系统可靠性 / 失败重试 / 降级

**① 问题**：LLM 端点超时、embedding 模型加载失败、第三方检索挂掉 → 整体不可用。

**② 通用方案**：超时 + 退避重试 + 优雅降级；全局异常兜底。

**③ 本项目落地**：`Tool.execute` 3 次退避重试；`search.py` 12 源**逐源 `_safe` 包裹**（单源挂不影响其他）；`scope_gate` 嵌入不可用降级为规则闸；**全局异常处理器**（`api.py`）未捕获异常返回统一 JSON 不暴露内部；`logger.py` 统一结构化日志。

### 4.3 性能（QPS / 延迟 / TTF，压力测试）

**① 问题**：首字响应（TTF）影响交互体感；检索慢拖累整体；需要压测找推理/DB 瓶颈。

**② 通用方案**：流式输出降首字延迟；缓存命中；异步并发；压测定位瓶颈。

**③ 本项目落地**：SSE 流式（`/ask/stream`、`/agent/pi/stream`）；异步 httpx 并发 12 源；检索策略/翻译/搜索三层缓存；`benchmark.py` 性能基准 + `/metrics` 吞吐统计。这是本项目的"加速度"迭代结果（几十秒→秒级体感）。

### 4.4 基础设施（FastAPI / ORM / 向量库 / 缓存）

**① 问题**：选型影响扩展性与运维成本。向量数据库、缓存、ORM 怎么配。

**② 通用方案**：API 框架 + 关系库 + 向量库 + Redis 缓存；微服务拆分按需。

**③ 本项目落地**：**FastAPI**（单进程即可，`fastapi跨语言调用`/`分布式部署`是**预留边界**，当前单容器零运维足够）；**SQLite WAL**（thread-local 连接、FK 级联，够个人工具用）；**LanceDB** 向量 + FTS 双层；`config.py` 全量环境变量配置（`用环境变量配置替代硬编码`——专项整改过）；无 Redis（缓存用进程内 dict + 过期）。

### 4.5 模型降级策略

**① 问题**：主力模型限流/超时/涨价时，服务不能断。

**② 通用方案**：多模型 + 失败切换；按任务分级用模型。

**③ 本项目落地**：当前单 DeepSeek（OpenAI 兼容）；`embedder.py` 双后端（ST + FlagEmbedding）自动切换；降级策略为**预留扩展点**（`config.py` 换模型即切）。

---

## 五、安全与风控

### 5.1 线上安全风控 / 敏感信息过滤

**① 问题**：用户输入可能含隐私/敏感/恶意内容；免费通道会被当通用 LLM 滥用。

**② 通用方案**：输入清洗 + 关键词过滤 + 内容安全审核 + 配额。

**③ 本项目落地**：`monitor.sanitize_input`（输入清洗/截断）+ `hooks._sensitive_word_filter`（敏感词拦截）；**`scope_gate.py` 学术范围白名单闸门**（刚建的专项）：三层漏斗（续接短句→规则词→嵌入语义 margin）锁定用户只能问论文/学术问题，拒绝后返回 422 `SCOPE_REJECTED`，嵌入降级不挂服务；`quota.py` 按量付费 + 免费额度（成本风控 + 防滥用）。

### 5.2 防 Agent 注入 / 工具白名单 / 权限体系

**① 问题**：检索到的网页/论文文本可能含恶意指令注入 Agent；工具可被诱导执行危险操作；多用户无隔离。

**② 通用方案**：工具白名单 + 参数校验 + 敏感工具人工确认；来源分层（工具输出不当指令执行）；统一权限（JWT）。

**③ 本项目落地**：`hooks._tool_allowlist` 工具白名单守卫；`tool_manager` 参数校验 + 安全检查；**危险工具 `need_confirm=True`**（如 `download_and_index`）执行前记审计日志、审批门暂停；`auth.py` JWT + bcrypt，`get_current_user` 依赖注入；`/upload/data` 仅白名单扩展名、`/file/download` 限制在 `data/exports/`（**路径穿越防护**，刚修的）。

### 5.3 幻觉率监控

**① 问题**：生成内容与事实不一致（幻觉）是学术工具的大忌——引用编造、结论无据。

**② 通用方案**：证据锚定 + 人工标注一致性校验 + 质量门。

**③ 本项目落地**：**防幻觉三件套**——`key_findings` 证据保留 + 1M 上下文（事实在上下文里，减少编造）；答案**带引用标注**（`[论文: 标题, 年份]`，来源可查）；`draft_quality.py` 初稿 6 项自检（研究问题/空白定位/结构/引用/诚实性/可续写），**诚实性单列一项**，不合格必须修订。`research_gap.py` 明确**不承诺"没人做过"**，改为"在现有文献中找到位置 + 诚实标注不确定"。

### 5.4 风控证据聚合与处置边界（合法红线）

**① 问题**：学术工具天然踩版权红线；被用于非法用途要能挡。

**② 通用方案**：来源白名单 + 合规边界声明 + 处置闭环（证据→决策→记录）。

**③ 本项目落地**：**法律红线写死**——只处理 OA 内容 + 用户上传论文，绝不碰付费/盗版（Sci-Hub 类）；`fetch_url` 只读用户提供的链接 + OA 公开页面；`scope_gate` 拦截违法犯罪类请求；合规声明在 `docs/PITCH_FACTS.md`。

---

## 六、可观测与优化闭环

### 6.1 链路可观测体系

**① 问题**：Agent 多步执行，哪一步慢/错/耗 token，黑盒查不了。

**② 通用方案**：trace（每步工具/耗时/token）、metrics、日志、SSE 逐步推送。

**③ 本项目落地**：`monitor.py` 指标 + `/metrics`；请求日志；`/ask/trace` 流水线逐阶段耗时可视化（`pipeline.html`）；`/agent/pi/stream` SSE 逐步推送工具调用事件；`dev.html` 开发者中心聚合（工具/配额/评测/建议学习）。

### 6.2 闭环优化流程

**① 问题**：检索质量差却说不清差在哪、怎么改。

**② 通用方案**：数据处理 → query 预处理 → 混合检索 → 指标拆解验证 → 回注。

**③ 本项目落地**：`/feedback` 数据飞轮（用户 👍👎 进 `learning_feedback` 表）；`run_eval.py`（R@5 + MRR + 回归门）+ `run_benchmark.py`（多主题）；**自我进化三环**（`docs/SELF_EVOLUTION.md`）：P1 反思重试 / P2 反馈教训（低分→LLM 提炼注入）/ P3 技能记忆（成功模式跨会话）；主动路由 `/suggest` 按接受度学习。安全约束：**Agent 从不改自己代码，学习结果人类审批**。

---

## 附 A：面试速查表（概念 → 本项目一句话）

| 概念/问题 | 本项目一句话 |
|---|---|
| Agent 定义 | 感知→规划→工具→反思循环（pi_agent）|
| ReAct vs Plan-and-Execute | pi_agent = ReAct + 反思；research_team = 流水线式 |
| Framework vs Graph | 走 Framework(Hook) 路线，Graph 路线被取代 |
| Skill 机制 | 7 技能按意图动态裁剪工具集，保留核心地板 |
| 编排四层 | 工具/流程/多Agent/运行时（3 审批门）全有 |
| 死循环 | max_steps_guard + 幂等重试 + 反思防跑偏 |
| 上下文工程 | 1M 窗口 + key_findings 证据保留 + 工具输出截断 |
| 前缀缓存/token 成本 | 压缩 + 断点续写 + skill 裁剪 |
| 记忆三层次 | SQLite 短时 / P2 教训 / P3 技能记忆 + 轻量图谱 |
| 垂直 RAG | 双层 LanceDB + 双语重排 + 引用提权 + 4096 分块 |
| 混合检索 | 自适应 RRF（BM25+向量）+ 两级级联重排 |
| Rerank/融合 | MiniLM 粗排 → bge-reranker 精排 + 引用对数提权 |
| GraphRAG vs 向量 RAG | 向量 RAG 核心 + 轻量图谱可视化，诚实不冒充 |
| RAG 评测 | MRR 0.72 / R@top5 0.78（43篇/30题/5主题，诚实声明）|
| 输出稳定 | JSON Schema 校验 + 统一 dict + 重试 |
| 可靠性/降级 | 3 次退避重试 + 12 源单源隔离 + 全局异常兜底 |
| 性能 | SSE 流式 + 三层缓存 + 异步并发 |
| 安全风控 | sanitize + 敏感词 hook + **scope_gate 学术白名单** + quota |
| 防注入/权限 | 工具白名单 + need_confirm 审批 + JWT |
| 幻觉 | key_findings 证据 + 引用锚定 + 初稿 6 项自检 |
| 合法边界 | 仅 OA + 用户上传，绝不碰盗版 |
| 可观测 | trace + metrics + SSE 步骤推送 + dev 中心 |
| 优化闭环 | feedback 飞轮 + benchmark + 自我进化三环 |

## 附 B：本项目关键文件索引

| 文件 | 对应主题 |
|---|---|
| `agent/pi_agent.py` | ReAct 循环 / 反思 / 断点续写 / Hook |
| `agent/hooks.py` | 死循环守卫 / 敏感词 / 工具白名单 |
| `agent/context_engine.py` | 上下文压缩 / 证据保留 / token 记账 |
| `agent/tool_registry.py` + `tools_extra/` | 工具编排 / 插件化（fetch_url / dataio）|
| `agent/skills.py` | Skill 机制 / 动态裁剪 |
| `agent/research_team.py` | 多 Agent 编排 / 3 审批门 |
| `search.py` | 12 源检索 / 双语重排 / 引用提权 |
| `hybrid_retriever.py` | 自适应 RRF / 两级重排 |
| `qa_engine.py` | RAG 8 步链路 / 指代消解 / 引用 |
| `knowledge_base.py` + `downloader.py` | LanceDB 双层 / PDF 解析分块 |
| `scope_gate.py` | 学术白名单闸门（三层漏斗）|
| `quota.py` + `api.py` Stripe | 配额 / 按量付费 |
| `feedback_lessons.py` / `agent_skills.py` / `suggestions.py` | P2 教训 / P3 技能记忆 / 主动路由学习 |
| `run_benchmark.py` / `run_eval.py` | RAG 评测 |
| `monitor.py` / `static/dev.html` | 可观测 / 开发者中心 |

> 本篇与 `docs/ARCHITECTURE.md`（系统架构）、`docs/BENCHMARK.md`（评测方法）、`docs/SELF_EVOLUTION.md`（自我进化）互为补充；总纲见根目录 `PROJECT_SUMMARY.md`。
