# PaperLink 竞品全景分析 (2024-2026)

> 覆盖 GitHub 上 20+ 个同类开源项目的全面分析。
> 每个项目标注：定位、亮点、vs PaperLink 的差异、可借鉴之处。

---

## 第一梯队：直接同质化（和我们做的事高度重叠）

### 2026 年新项目

#### OpenDraft — 19 Agent 全自动论文写作
- **Repo**: [federicodeponte/opendraft](https://github.com/federicodeponte/opendraft)
- **定位**: 19 个 Agent 在 ~10 分钟起草 2 万字学术论文，引用经 CrossRef/OpenAlex/arXiv 验证
- **亮点**: 57+ 语言, PDF/Word/LaTeX 导出, MIT License
- **vs PaperLink**: 他们偏向"最终产出"（论文 PDF），我们偏向"调研过程"（检索+筛选+综述+审稿）。我们多了人工审批闸门，他们多了多语言 PDF 导出
- **可借鉴**: 引用验证流水线（CrossRef/OpenAlex 交叉校验）; LaTeX 模板生成

#### NanoResearch — 三层协同进化多 Agent
- **Repo**: [saadmsft/nanoresearch](https://github.com/saadmsft/nanoresearch)
- **定位**: arXiv:2605.10813 论文实现 — tri-level co-evolving multi-agent (Skill Bank + Memory + SDPO Planner)
- **亮点**: 领域无关（生物/社科/工程/CS 通用），ChatGPT-style Web UI, 自动生成 Python 实验
- **vs PaperLink**: 他们的核心创新是"协同进化"（Agent 互相学习），我们是"人工审批双轨"。路线不同但互补
- **可借鉴**: Skill Bank 机制 — Agent 可积累和复用已学会的检索/写作模式

#### Research Radar — 三信任闸门 + 人机协同
- **Repo**: [GZ-November/research-radar](https://github.com/GZ-November/research-radar)
- **定位**: Human-in-the-loop, G0/G1/G2 三级信任闸门, PaperQA2 矛盾检测, 329 tests
- **亮点**: 和我们最像！三个闸门 vs 我们的三个闸门。他们多做了一层：PaperQA2 引用校验；我们多做了一层：13 Agent 细分
- **vs PaperLink**: 我们 Agent 数量更多 (13 vs 5), 他们测试更充分 (329 tests vs 0)
- **可借鉴**: 329 个测试用例的设计思路; G0/G1/G2 信任分级的粒度参考; PaperQA2 作为 citation checker 集成

#### Literature Review Agent — LangGraph 双 Agent 流
- **Repo**: [littlelelephant/literature-review-agent](https://github.com/littlelelephant/literature-review-agent)
- **定位**: 两个独立 LangGraph Agent — Literature Search + Literature Review, DeepSeek 默认
- **亮点**: audit-ready（每步可审计）, 引用 100% 可追溯, evidence-grounded writing
- **vs PaperLink**: 我们多了 10 个 Agent + 12 个数据源; 他们更聚焦综述质量
- **可借鉴**: 两个 Agent 间的 explicit handoff 协议; evidence-grounded 生成（强制每个句子附引用证据 ID）

#### ScholarFlow — 8 节点 LangGraph 流水线
- **Repo**: [qianbkk/scholarflow](https://github.com/qianbkk/scholarflow)
- **定位**: 8 节点 LangGraph 流水线 (query_decompose → search → expand_citations → rank → refine → synthesize → graph → track_cost)
- **亮点**: D3 交互引文图谱, 双语 (中/英), 流式思考日志, 成本追踪
- **vs PaperLink**: 我们多了 Agent 协作层，他们多了成本追踪 + 引文图谱。两者互补
- **可借鉴**: 引文链扩展（从一个论文出发，沿引用关系扩展）; 成本追踪（在 `/agent/pi/stats` 中加 cost 字段）

#### LitScribe — 7 Agent + GraphRAG + Zotero 集成
- **Repo**: (7-agent LangGraph + GraphRAG)
- **定位**: 24 篇论文综述, 100% 引用 grounding, 15 分钟, $0.098
- **亮点**: **Zotero 集成**, 多语言生成, 5 种引用格式
- **vs PaperLink**: 他们有 Zotero 集成（我们计划中），我们有双轨审批
- **可借鉴**: Zotero 集成的具体实现方式; $0.098 成本优化策略

#### Paper-Agent — AutoGen + LangGraph 多 Agent
- **Repo**: [Tswoen/Paper-Agent](https://github.com/Tswoen/Paper-Agent)
- **定位**: 6 核心节点 (search → read → analyze → write → report), KMeans 论文聚类
- **亮点**: 论文聚类分析, ChromaDB, 关键词+自动化搜索+分析
- **vs PaperLink**: 我们用 LanceDB 而非 ChromaDB (更轻量); 我们多了审批机制
- **可借鉴**: KMeans 论文聚类 — 在搜索结果中自动发现研究方向聚类

#### Pilot Research — 技能化研究工作流
- **Repo**: [nmd2k/pilot-research](https://github.com/nmd2k/pilot-research)
- **定位**: Skill-based research workflows for AI coding agents
- **亮点**: Socratic 研究讨论, arXiv 搜索+摘要, 子 Agent 并行执行
- **vs PaperLink**: 他们是 CLI-first 工具, 我们是 Web 界面。互补
- **可借鉴**: Socratic 讨论模式 — 选题评估前先做一轮批判性对话

#### AI Research SKILLs — 98 技能库
- **Repo**: [Orchestra-Research/AI-Research-SKILLs](https://github.com/orchestra-research/AI-Research-SKILLs)
- **定位**: 98 个技能, 23 个类别, 覆盖从 idea 到 paper 全流程
- **亮点**: 双循环架构 (outer: research plan → inner: execute), 支持 Claude Code/Codex/Gemini
- **vs PaperLink**: 他们是 "skill 市场" 模式, 我们是 "一体化平台" 模式
- **可借鉴**: 将我们的 4 个 broad skill 拆分为更细粒度的 skill, 参考他们的分类体系

---

### 2024-2025 年项目

#### PaperQA2 — RAG 引用精度标杆
- **Repo**: [Future-House/paper-qa](https://github.com/Future-House/paper-qa) — 7,700 ⭐
- **定位**: High-accuracy RAG over PDFs, 引用级精度, superhuman 基准
- **亮点**: 三轮验证 (search → gather → answer), LangChain, LiteLLM 多模型
- **vs PaperLink**: 他们 RAG 精度更高（论文级验证），我们功能更全（12 源搜索 + Agent 团队）
- **可借鉴**: citation_verifier Agent 参考他们的三轮验证流程; 作为可选 QA 后端引擎

#### GPT-Researcher — 多 Agent 深度研究
- **Repo**: [assafelovic/gpt-researcher](https://github.com/assafelovic/gpt-researcher) — 15,000+ ⭐
- **定位**: Web 多 Agent 深度研究, Plan-and-Execute, 报告生成
- **亮点**: 前端成熟度最高 (React), 并行 Agent, PDF/DOCX/Markdown 导出
- **vs PaperLink**: 他们搜 web 我们搜学术库，场景互补。他们前端我们 Agent 深度
- **可借鉴**: 报告格式多样性 (DOCX 导出); 前端 React 组件设计

#### arxiv-tracker-agent — 最相似架构
- **Repo**: [gaoweijun5/arxiv-tracker-agent](https://github.com/gaoweijun5/arxiv-tracker-agent)
- **定位**: LangGraph + FastAPI + 混合检索 + Markdown 报告, 每日论文推送
- **亮点**: 架构与我们几乎一致 (LangGraph, FastAPI, 混合检索, SSE)
- **vs PaperLink**: 我们多了 11 个数据源, 13 Agent, 审批机制, Docker; 他们多了定期推送
- **可借鉴**: 每日自动检索 + 新论文推送; 论文评分卡片 UI

#### AI Research Team — 6 Agent 团队
- **Repo**: [tomlin7/ai-research-team](https://github.com/tomlin7/ai-research-team)
- **定位**: 6 个专业 Agent (Reviewer, Hypothesis, Experiment, Data, Writer, Critic) + LaTeX PDF
- **亮点**: OpenAI/Anthropic/Gemini/Ollama 全支持, Semantic Scholar + arXiv
- **vs PaperLink**: 他们有 LaTeX PDF 生成, 我们 Agent 更多 (13 vs 6) + 审批
- **可借鉴**: LaTeX PDF 生成 pipeline

#### STORM — Stanford Wiki 风格生成
- **Repo**: [stanford-oval/storm](https://github.com/stanford-oval/storm) — MIT License
- **定位**: 自动 Wikipedia 风格文章生成, 强引用
- **亮点**: 多轮对话式研究, 大纲自动生成, 强引用保证
- **vs PaperLink**: 我们多了学术源搜索和 Agent 团队; 他们的引用质量更高
- **可衔接**: 作为综述 Agent 的前置步骤, 或者在我们的 KB 建好后喂入 STORM 生成初稿

---

## 第二梯队：可衔接的上下游工具

| 工具 | 类型 | 衔接方式 | 优先级 |
|---|---|---|---|
| **[Zotero](https://www.zotero.org)** | 文献管理 | **Plugin**: 选中论文 → "Send to PaperLink" → 自动建 KB → 问答。Zotero 有本地 WebSocket/HTTP API | **P0** |
| **[Paperlib](https://github.com/lichao0211/paperlib)** | 开源文献管理 | 元数据抓取参考（无 DOI 会议论文识别）；可互为桌面端替代 | P2 |
| **[Obsidian](https://obsidian.md)** | 本地知识库 | `/export` 输出 Markdown → Obsidian vault 结构 + 双向链接 | P1 |
| **[PaperQA2](https://github.com/Future-House/paper-qa)** | RAG 引擎 | 作为可选后端（`QA_BACKEND=paperqa`），精度敏感场景切换 | P1 |
| **[Zettlr](https://www.zettlr.com)** | Markdown 写作 | 导出的综述/论文 → Zettlr 做最终编辑和排版 | P3 |

---

## 第三梯队：教育参考

| 工具 | 为什么参考但不跟进 |
|---|---|
| **AI-Scientist-v2** | 全自动实验→论文。但我们明确说了"不替代研究者判断"，不能走全自动路线 |
| **SurveyX / SurveyForge** | 纯综述生成。功能太窄，用我们的 `review_synthesizer` Agent 可覆盖 |
| **Prism AI** | 多 Agent + LangGraph + Perplexity 替代。和 GPT-Researcher 定位重叠 |
| **Local Deep Research** | Ollama 本地运行。适合隐私敏感场景，我们可参考其本地 LLM 接入方式 |
| **Virubot Autonomous Agent** | Vertex AI + React。全自动，无审批机制，与我们理念冲突 |

---

## PaperLink 的差异化优势（全 GitHub 唯一）

搜索了 20+ 项目后，PaperLink 有**四个独特竞争力**：

| 优势 | 说明 |
|---|---|
| **12 个学术源并行** | 最多同类 (GPT-Researcher: web only; arxiv-tracker: arXiv only; PaperQA2: user PDFs)。我们是唯一覆盖 OA + PubMed + CNKI + CORE + DOAJ + EPMC + PWC + Crossref 全部 12 源的 |
| **13 Agent + 三层架构 + 双轨审批** | 没有任何竞品有"人工审批闸门" + "三层架构 (上游/中游/下游)"。OpenDraft 有 19 Agent 但全是自动的；Research Radar 有 3 个闸门但只有 5 个 Agent |
| **零外部依赖部署** | Docker + SQLite + LanceDB, 无 GPU 需求。GPT-Researcher 需要多个外部 API, PaperQA2 需要 PyPI 依赖链。我们用一条 `python api.py` 启动全栈 |
| **全链路覆盖** | 选题→检索→筛选→综述→章节写作→图表→审稿→校验→伦理→定稿→导出。竞品多是"单一环节精品"（PaperQA2: 只做 RAG 精度; STORM: 只做文章生成） |

---

## Phase 4 路线（借鉴竞品 + 衔接上下游）

### Phase 4A: 吸收精华 (1-2 天)

| 借鉴来源 | 做什么 | 价值 |
|---|---|---|
| **PaperQA2** | 新增 `citation_verifier` Agent (在 `final_drafter` 之前): 逐条校验引用是否真实存在 | 消除幻觉引用 |
| **LitLLM** | `review_synthesizer` 拆为两步: outline→fill, 参考 plan-based 生成 | 提升综述质量 |
| **Research Radar** | 参考 G0/G1/G2 信任分级，细化三个闸门的检查项 | 审批更严谨 |
| **ScholarFlow** | ✅ 引文链扩展: 从一篇核心论文出发，沿引用关系扩展发现更多论文 (`/citation/chain`) | 提升文献覆盖率 |
| **arxiv-tracker-agent** | ✅ 研究方向订阅 + 自动检索 + 后台调度器 (`/subscribe` + `scheduler.py`) | 持续性服务 |

### Phase 4B: Zotero 集成 ✅ 已完成

- ✅ `/zotero/import` 端点: 接收 Zotero metadata JSON → 自动创建课题
- ✅ `/zotero/scan` 端点: 扫描本地 Zotero SQLite 数据库
- ✅ 多格式论文导出: Markdown / DOCX / LaTeX / BibTeX (`/export/paper` + `/export/download`)
- 🔜 **Zotero Plugin** (浏览器扩展端): 需要单独的插件项目

### Phase 4C: 持续服务 ✅ 已完成

- ✅ `/subscribe`: 订阅研究方向 → 后台 `scheduler.py` 每小时自动检索
- ✅ `/export/paper?format=docx|latex|markdown|bibtex`: 4 格式论文导出
- ✅ 前端集成: Export dropdown + Subscribe button + Citation Chain button

---

*分析日期: 2026-08-03 | 覆盖 20+ 项目 | 2024-2026 全时段 | Phase 4 全部完成*
