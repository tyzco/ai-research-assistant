# PaperLink 项目总总结（与需求对齐版）

> 本文件是项目的唯一总纲：把「需求 → 设计 → 实现 → 验证 → 交付」一次对齐。
> 读这一份，加上 docs/ 索引，即可完整接手。

---

## 一、项目是什么（一句话）

**面向 C 端研究者的「选题 → 初稿」Agent 即服务工具**：输入研究方向，系统回答「有没有人做过」→ 检索建私有知识库 → 基于证据问答 → 辅助写作 → 质量验收 → 审稿批注。免费额度体验，按量付费，完成一篇退场。

## 二、需求与实现对齐表

| 原始需求 | 实现 | 验证 |
|---|---|---|
| C 端个人学术工具 | 用户侧仅：聊天框 + 瀑布流 + 3 模式按钮（写论文/找论文/科研团队）| ✅ |
| 选题→初稿 一条龙 | 9 步 pitch + 12 演示模块 | ✅ |
| Agent 即服务 + 按量付费 | 14 工具 / 7 技能 + 免费额度→Stripe 收款 | ✅（HKD 下限已修）|
| 用户自供论文建私有库 | 上传→双层 LanceDB KB→RAG | ✅ |
| 合法边界 | 仅 OA 下载 + 闭源「学校账号」提示 | ✅ |
| 检索质量（不常见选题出关键论文）| 双语重排 + 引用提权 + 领域联想词 + CN→EN 兜底 + 4096 分块 | ✅ A/B +1.4% |
| 自我进化 + 主动路由 | P1 反思/P2 反馈教训/P3 技能记忆 + 预判/执行/学习 | ✅ |
| 用户侧简洁/开发侧聚合 | 3 按钮 header + `/static/dev.html` 开发者中心 | ✅ |
| 诚实可宣传 | 检索基准 MRR 0.72（方法可复核）+ 局限文档 + 宣传话术 | ✅ |

## 三、产品故事（对人讲）

```
输入选题
 → 研究空白分析（有没有人做过，锚定文献）
 → 12 源检索 + 关键论文浮出
 → 用户建私有知识库（OA + 自供论文，合法）
 → 基于证据问答 / Agent 调研写作
 → 初稿质量清单 6 项自检（货真价实）
 → 同行评审批注
 → 自我进化 + 主动路由（越用越懂你）
```

## 四、系统架构

```
前端: index.html (用户) / dev.html, pipeline.html, test.html, kg.html (开发)
API:  api.py (FastAPI, 44+ 端点)
检索: search.py (12 源) → hybrid_retriever (BM25+向量 RRF) → 级联重排
RAG:  qa_engine.py (8 步链路) + knowledge_base (LanceDB 双层)
Agent: pi_agent (Hook 驱动) / research_team (13 Agent)
工具:  tool_registry (14 工具) + tools_extra 插件 + tool_manager 安全
技能:  skills.py (7 技能) + intent_classifier (三层漏斗)
进化:  feedback_lessons (P2) + agent_skills (P3) + draft_quality (验收线)
商业化: quota (配额) + Stripe (收款)
```

## 五、交付物清单

**12 演示模块**（`demo/`，独立可跑、真实调用、可串联）：
检索/研究空白/建库/问答/Agent/团队/写作/质量/审稿/自我进化/主动路由/引用链

**8 份文档**（`docs/`）：
| 文档 | 作用 |
|---|---|
| `PITCH_FACTS.md` | 宣传怎么说、什么不能说（面试/融资）|
| `BENCHMARK.md` | 评估方法 + 声明边界 |
| `DELIVERY.md` | 交付状态 + 启动 |
| `SELF_EVOLUTION.md` | 自我进化三环 |
| `REVIEW_DEEPSEEK.md` | 链接驱动修复清单 |
| `DEMO_SCRIPT.md` / `ARCHITECTURE.md` / `COMPETITIVE_ANALYSIS.md` | 演示/架构/竞品 |

**诚实数据**：检索基准 MRR 0.72 / R@top5 0.78（43 篇 / 30 题 / 5 主题，`data/benchmark_report.json`）

## 六、技术亮点（可讲深）

1. **检索质量**：相关度重排（双语 query + 引用对数提权）让不常见选题（如「可微神经架构搜索」）浮出 DARTS 关键论文；4096 分块 + 邻块±2 覆盖更全（A/B +1.4%）。
2. **自我进化**：反思重试（直答自批判）+ 反馈教训（低分→LLM 提炼注入）+ 技能记忆（成功做法跨会话复用）。
3. **主动路由**：阶段预判（选题→初稿旅程）→ 一键执行 → 接受度学习（越点越优先）。
4. **防幻觉**：1M 上下文窗口 + key_findings 证据保留 + 初稿质量清单。
5. **插件化**：`tools_extra/` 丢文件即注册，`/agent/tools/reload` 不重启生效。

## 七、诚实声明（交工必读）

1. **本机网络抖**：LLM 端点间歇超时（环境），稳定网络正常。
2. **真实数据未积累**：自我进化/主动路由的学习要真实用户使用才开始。
3. **生产部署**：Stripe 换 live 密钥 + webhook secret + 真实域名。
4. **检索数字**：0.72 是 43 篇/30 题小样本，可作 sanity 不作大规模性能声明。
5. **测试集**：8 题 run_eval 是内部冒烟，非对外基准（对外用 run_benchmark）。

## 八、启动与使用

```bash
cp .env.example .env   # 填 DEEPSEEK_API_KEY (+ 可选 Stripe)
python api.py          # http://localhost:8001
python demo/pitch.py "研究方向"      # 9 步产品演示
python demo/run_all.py "方向"        # 12 模块
python run_benchmark.py --kb papers_benchmark_mt_kb2  # 检索基准
```

## 九、与「我」的对齐（本项目按我的视角组织）

- **定位**：帮用户完成「选题→初稿」，不贪多；交付货真价实的初稿（质量清单兜底）。
- **变现**：按量付费（写论文是阶段性高强度，完成一篇退场），非订阅。
- **合法**：仅 OA + 用户上传，绝不碰盗版。
- **原则**：Agent 自主决策 × 固定流程规范（流程是骨架，Agent 是血肉，审批门是安全绳）。
- **风格**：用户侧极简（聊天+瀑布流），开发侧聚合；诚实数据，不虚报。
