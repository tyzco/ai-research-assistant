# PaperLink 代码审查修复清单（DeepSeek 链接驱动）

> **范围**：你提供的 DeepSeek 审查分享链接驱动的全部修复。
> **依据**：代码内 `DeepSeek 审查 #N` / `DeepSeek 建议` / `DeepSeek 对话建议` 注释**逐条复制原文**核对。
> **近期的非链接驱动修复**（自主排查：/metrics RLock 死锁、api.py 模块级 logger 未定义、HF 离线挂起、多用户 ownership、搜索相关度重排）不在本清单，见各文件内注释可追溯。

---

## 一、全局未采纳 / 部分采纳

| # | 建议 | 处置 | 原因 |
|---|---|---|---|
| G1 | ragas 全套 LLM 判定（faithfulness/answer_relevancy/context_recall 走 ragas LLM judge） | **未采纳** → faithfulness/answer_relevancy 改 DeepSeek 直判，context_recall 改嵌入语义召回，仅 context_precision 走 ragas | ragas 0.4.3 + langchain_community 0.4.x 与 DeepSeek 输出格式不兼容：faithfulness 恒 0、answer_relevancy nan、context_recall 卡死（IncompleteOutputException）。代码原注：`faithfulness/answer_relevancy/context_recall 的 LLM 判定与 DeepSeek 输出格式不兼容 (faithfulness 恒 0 / answer_relevancy nan / context_recall 卡死)。故: faithfulness+answer_relevancy 用 DeepSeek 直判, context_recall 用嵌入语义召回` |
| G2 | ragas `InstructorLLM + InstructorModelArgs(max_tokens=2048)` | **未采纳** | 全指标 nan，instructor 结构化输出与 DeepSeek 不兼容。原注：`LangchainLLMWrapper (非 instructor): 纯 chat 生成, 对 DeepSeek 等 OpenAI 兼容模型更稳定` |
| G3 | ragas `llm_factory(max_tokens=...)` 传播 | **未采纳** | 参数未生效，DeepSeek 仍被默认 max_tokens=1024 截断 |
| G4 | LanceDB `db.rename_table` 原子换名 | **未采纳** → `_copy_to` 数据复制 | LanceDB OSS 不支持 rename。原注：`rename 不支持(...), 改用数据复制` |
| G5 | 表扫描用 `tbl.search()`（无查询向量） | **未采纳** → `tbl.to_arrow()` | 原注：`原 search() 无查询向量会崩溃, 全表加载易 OOM` |
| G6 | 工具非白名单域名**默认阻止** | **部分采纳** → 默认仅告警 + `TOOL_SECURITY_BLOCK_NON_WHITELIST` 可升级 | 原注：`默认阻止会破坏检索→下载主流程` |
| G7 | DeepSeek 推理链耗尽 max_tokens → 空输出 | **采纳** → 生成调用 `extra_body={"thinking":{"type":"disabled"}}` + `LLM_GEN_MAX_TOKENS` 1024→2048 | deepseek-v4-pro 是推理模型，max_tokens 同时计推理+输出 |

---

## 二、核心检索与问答

### qa_engine.py
```python
_reranker_lock = threading.Lock()              # DeepSeek 审查 #3: 防并发重复加载
"""异步重排 (DeepSeek 审查 2: CrossEncoder.predict 同步阻塞事件循环)。"""
DeepSeek 审查 #2: pid 白名单校验, 防止拼接进 where 子句引发语法错误/注入
DeepSeek 审查 #3: 去重键 DOI/paper_id 优先, 避免同名论文误去重
DeepSeek 审查 #4   # (指代消解/查询改写失败时 warning 而非中断)
DeepSeek 审查 #5: _is_complex_query 内部已含 len>15, 外层冗余判断删除
DeepSeek 审查 #6)  # COMPLEX_EXPAND_KEYWORDS 由 config.INTENT_KEYWORDS+COMPLEX_INTENT_TYPES 派生
DeepSeek 审查 #7: 预初始化, 避免降级分支抛异常时 UnboundLocalError
DeepSeek 审查 #8: 历史放开头, 避免污染末端的回答指令
extra_body={"thinking": {"type": "disabled"}}  # (G7 推理模型适配)
```

### hybrid_retriever.py
```python
双检锁: 线程安全加载 (DeepSeek 审查 #1)                  # 重排器并发加载
v5.3 (DeepSeek 审查 2): stage1_k 可配置, 默认 max(top_k*3, 15)
logger.error(f"Stage1 rerank failed: {e}")  DeepSeek 审查 #5
logger.error(f"Stage2 rerank failed: {e}")  DeepSeek 审查 #5
归一化后点积 = 余弦相似度 (DeepSeek 审查 #8)
```

### search.py
```python
v5.3 修复 (DeepSeek 审查 1): 原实现 POST 后立即解析结果——但 Apify /runs 端点 (改轮询)
重试一次 (DeepSeek 审查 #2: 降低温度提高结构化输出稳定性)
jieba 分词: 中文输入自动分词，组合为多词英文查询 (DeepSeek 对话建议)
DeepSeek 审查 #6)  # 检索结果排序来源权重接入
```

### knowledge_base.py
```python
LanceDB 表不存在抛 ValueError (DeepSeek 审查 #1: 精确捕获, 其他异常向上抛避免数据丢失)
表不存在 → 自动创建 (DeepSeek 审查 #4: 原静默 return 丢数据无感知)
"""稳定 RRF 键 (DeepSeek 审查 6): chunk_id 优先, 否则 paper_id+chunk_index 组合。"""
```

---

## 三、Agent 系统

### agent/pi_agent.py
```python
DeepSeek 审查: 显式技能名覆盖 auto 意图选择 (None = 按任务自动激活)
=== 技能解析: 显式指定 > 意图自动选择 (DeepSeek 审查 #2/#3/#4) ===
DeepSeek 审查 #1: 任何路径都推进 steps, 防死循环
DeepSeek 审查 #2/#9: compact_history 保留 messages[-4:],
DeepSeek 审查 #3: 全量处理 tool_calls, 不再丢弃多余调用
DeepSeek 审查 #5: assistant 消息只追加一次, 每条工具结果都回填
DeepSeek 审查 #6: 用剩余时间做超时保护, 工具卡住可中止
DeepSeek 审查 #9: 直答轮也记录 turn, 使 trace.steps 完整
DeepSeek 审查 #4: 移除硬编码四大能力块 (由技能系统按任务动态提供)
"""触发 hook，异常隔离 (DeepSeek 审查 4): 单个 hook 崩溃不中断 agent。"""
DeepSeek 审查 #5: 校验/拦截/失败都回填结果, 避免对话停滞
```

### agent/research_team.py
```python
团队实例注册表 — 按 run_id 隔离 (DeepSeek 审查 #7: 弃用全局单例)
"""从 start_idx 开始执行流水线 (DeepSeek 审查 6: 去重 execute/approve_and_continue)。
"""持久化科研团队快照 (DeepSeek 审查 2: 崩溃可恢复, 多 run 并存)。"""
DeepSeek 审查 1: 原实现传入 tools/tool_choice 但只读 message.content,  # → 解析工具输出
DeepSeek 审查 3: 截断上限 500 → 1500, 减少下游上下文丢失。"""
DeepSeek 审查 4: 原 has_kb magic flag 全项目无处设置, 属死代码。   # → 删除
```

### agent/skills.py
```python
"""DeepSeek 审查 1: 校验技能声明的工具在注册表中存在, 缺失时警告。"""
(DeepSeek 审查 #2: 动态激活时不能丢掉检索/KB 发现/问答核心能力)    # CORE_TOOLS 地板
"""DeepSeek 审查 2: 按意图关键词动态激活最相关技能 (仅限激活技能)。"""
DeepSeek 审查 #6: 暴露提示预览, 便于前端/监控查看完整定义
"""注册/覆盖一个技能 (DeepSeek 审查 8: 支持第三方动态扩展)。"""
全局单例 (DeepSeek 审查 #10: 加锁防多线程重复初始化)
```

### agent/hooks.py
```python
DeepSeek 审查 #1: 空列表视为"不修改"(防止误清空工具集);
tool_calls: 交集式过滤, 同样防空集 (DeepSeek 审查 #3)
messages: first-writer-wins (DeepSeek 审查 #5: 冲突时告警)
DeepSeek 审查 #4: 加锁防多线程/多事件循环重复初始化
"""BEFORE_EMIT: 敏感词检测 (DeepSeek 审查 6: 可配置 + 记录命中词)。
DeepSeek 审查 7: 记录每次工具尝试 (审计), 未来可加黑名单。"""
DeepSeek 审查 #9: 带堆栈, 便于排查 hook 内部错误
```

### agent/tool_registry.py
```python
DeepSeek 审查 1: 退避改用 await asyncio.sleep, 不阻塞事件循环。
DeepSeek 审查 2: parse_pdf/chunk_text 是 CPU 密集同步操作,  # → 线程池
DeepSeek 审查 #4: 幂等注册, 防止模块重复导入/热重载时工具累积重复项
query 为占位参数 (DeepSeek 审查 5/#10: 当前不参与过滤, 保持可选)。"""
"""导出课题报告。统一返回 dict, 与其他工具格式一致 (DeepSeek 审查 8)。"""
```

### agent/tool_manager.py
```python
DeepSeek 审查 1: 原 `d in url` 子串匹配可被 fake-arxiv.org 绕过。"""  # → 严格域名匹配
DeepSeek 审查 #2: bool 是 int 子类, 先显式处理 bool 避免误转
need_confirm 机制负责 (DeepSeek 审查 #3: 默认阻止会破坏检索→下载主流程)  # G6
self._lock = threading.Lock()  DeepSeek 审查 #4: 并发写保护
```

### agent/context_engine.py
```python
DeepSeek 审查 1: 优先 tiktoken cl100k_base (英文/代码/JSON 更准),
tiktoken cl100k_base 编码缓存 (DeepSeek 审查 #1: 英文/代码 token 估算更准)
DeepSeek 审查 #2: 摘要用 system 角色注入, 避免破坏 user/assistant 交替规则
DeepSeek 审查 #3: 清洗孤立代理项, 防 write_text/JSON 序列化崩溃
self._lock = threading.Lock()  DeepSeek 审查 #9: 并发计数保护
```

---

## 四、基础设施

### api.py
```python
DeepSeek 审查 #1: 保存消息移到 return 之前 (原代码在 return 后, 不可达),
DeepSeek 审查 #2: 反向代理下取 X-Forwarded-For 首个 IP, 否则用直连 IP
DeepSeek 审查 #3: 保存完整答案, 原代码 add_turn(question, "") 污染记忆
"""统一 LanceDB 表列举 (DeepSeek 审查 4: 兼容不同版本 API, 去重各处脆弱代码)。"""
DeepSeek 审查 #5: getattr 安全访问, 防 progress/stats 缺失时 500
翻译缓存检查 (DeepSeek 对话建议: 避免重复 LLM 调用)
DeepSeek 审查 #7: 加锁保护并发读写, 防止 JSON 文件损坏。"""
DeepSeek 审查 #9: 记录失败源 (原静默返回空列表, 无法排查限流/超时)
DeepSeek 审查 #13: CPU 密集解析放入线程池, 不阻塞事件循环
sem = asyncio.Semaphore(DOWNLOAD_CONCURRENCY)  DeepSeek 审查 #14: 配置化
DeepSeek 审查 #15: 大小上限 + PDF 魔数校验 (真 PDF 以 %PDF 开头)
```

### config.py
```python
DeepSeek 审查 #14/#15   # Upload / Download 资源限制 (MAX_UPLOAD_BYTES / DOWNLOAD_CONCURRENCY)
# (另有 INTENT_KEYWORDS / INTENT_COMPLEXITY_SCORES / COMPLEX_INTENT_TYPES 集中配置)
```

### logger.py
```python
异步安全的请求 ID 上下文 (DeepSeek 建议: contextvars 传递请求 ID)
请求 ID 追踪: 每次请求生成唯一 ID, 该请求内所有日志携带 (DeepSeek 建议)
噪音日志抑制列表 (DeepSeek 建议: 移到可配置)
"""JSON 日志格式 (DeepSeek 建议: 生产环境集中日志系统可解析)。"""
自动初始化 (DeepSeek 建议: 防遗漏手动调用)
```

### embedder.py
```python
后端判断: 显式配置优先, 否则按模型名启发 (DeepSeek 建议: 避免仅靠命名)
维度校验 (DeepSeek 建议: 配置与实际不符提前报错, 而非静默写错 LanceDB)
"""统一 L2 归一化 (DeepSeek 建议: 所有后端输出单位向量, 保证余弦相似度可比)。"""
双检锁: 线程安全加载 (DeepSeek 审查 #1)
"""异步批量编码 (DeepSeek 建议: 避免同步 CPU 密集阻塞事件循环)。"""
```

### database.py
```python
只初始化一次 (DeepSeek 建议: 避免高频操作反复执行 DDL)
v5.2: 支持 user_id 隔离 (DeepSeek 建议: 启动加载不泄漏其他用户课题)。"""
"""持久化科研团队快照 (DeepSeek 审查 2: 崩溃可恢复, 多 run 并存)。"""
"""事务上下文管理器 (DeepSeek 建议: 多操作原子性)。
```

### topic_manager.py
```python
状态反向映射 (DeepSeek 对话建议: 避免 BUILDING 吞掉其他状态)
完整状态映射 (DeepSeek 对话建议: 不丢失 ERROR 等状态)
JSON 反序列化 (DeepSeek 对话建议: 写入/读取对称)
"""增量保存单个课题 (DeepSeek 对话建议: 原实现名不副实——叫 save_topic_state 却做全量保存)。"""
"""增量保存单个课题 (DeepSeek 对话建议: 替代全量保存的写放大问题)。"""
```

### models.py
```python
===== TopicProgress (高频更新字段 — DeepSeek 对话建议拆分) =====
===== Search Strategy (具体化 — DeepSeek 对话建议) =====
```

### downloader.py
```python
首页/尾页丢弃 (DeepSeek 审查 #1: 原 `>= page_count-1` 误丢倒数第二页, 尾页泄漏)
v5.3: 支持显式 chunk_overlap (DeepSeek 审查 3)。"""
v5.3: 修正注释——函数每次创建新实例, 非单例 (DeepSeek 审查 2)。"""
图片上限 (DeepSeek 审查 #4: 防止大 PDF 内存/磁盘激增)
```

### knowledge_graph.py / conversation_memory.py / auth.py / monitor.py / image_filter.py
```python
# knowledge_graph.py: 轻量 JSON 图 + 前端 D3, 不引入 Neo4j (DeepSeek 审查)
# conversation_memory.py: v5.2 修复 get_context 时序 bug; 结构化 key_findings (DeepSeek 建议); TTL 惰性清理
# auth.py: 注意：user_id 现在是 Optional[str]，未认证时为 None (DeepSeek 对话建议)。"""
#         日志中区分用户不存在 vs 密码错误 (DeepSeek 对话建议)。"""
# monitor.py: 细粒度错误分类 (DeepSeek 对话建议)
# image_filter.py: 规则 + ML 两级过滤 (DeepSeek 对话参考: 多模态调用前筛掉无用图)
```

---

## 五、评估与工具脚本

### evaluate_rag_standards.py（采纳 + 关键未采纳 G1-G3）
```python
# ragas 0.4.3 与 langchain_community 0.4.x 兼容 stub (vertexai 已拆分为独立包, 本评估不使用)
# 注: faithfulness/answer_relevancy/context_recall 的 LLM 判定与 DeepSeek 输出格式不兼容
#     (faithfulness 恒 0 / answer_relevancy nan / context_recall 卡死)。
#     故: faithfulness+answer_relevancy 用 DeepSeek 直判, context_recall 用嵌入语义召回。
# LangchainLLMWrapper (非 instructor): 纯 chat 生成, 对 DeepSeek 等 OpenAI 兼容模型更稳定
# 同步客户端: ragas 的 nest_asyncio 会破坏后续 async 调用, fallback 用同步避免 (实测可靠)
```

### run_eval.py / benchmark.py / eval_common.py（脚本整合）
- 采纳：3 处重复逻辑（load_eval_qa / pick_kb / semantic_recall / mrr）抽取 `eval_common.py` 共享；每题独立 try/except 失败不中断；hybrid 失败降级 vector；回归门禁（hybrid recall 下降 > 0.02 阻断）；benchmark 空表防护/CLI 参数(#3)/数据校验(#4)/命中率修正(#1) + `--limit`。

### test_e2e.py
```python
DeepSeek 审查改造: 从"人工验收检查清单"升级为自动化测试。
# 逐步骤断言 / 独特 topic_id / 用例间隔离 / 3 个不同问题 / JSON 报告 / 退出码 / 自动清理
```

### reindex_all.py（采纳 + 未采纳 G4/G5）
```python
DeepSeek 审查修复:
  #1/#10 原子双表替换: 先写临时表并验证, 成功后才删旧表+rename, 失败保留原表
  #2/#3  用 to_arrow() 表扫描 + 分批编码写入 (原 search() 无查询向量会崩溃, 全表加载易 OOM)
  #4     保留原表 schema, 仅替换 vector 字段维度 (原硬编码 14 字段会丢额外列)
  #6     FTS 失败告警 (原静默 pass)
  #7     写入后校验行数一致
  #8     argparse + --dry-run
  #9     用 schema 读维度 (原靠首行样本)
# 未采纳: db.rename_table (OSS 不支持) → _copy_to 数据复制; tbl.search() 无查询向量 (崩溃) → to_arrow()
```

---

## 六、前端

| 文件 | 采纳 |
|---|---|
| static/index.html | Agent 默认模式；模式下拉（Auto/团队/文献调研/写作/深度问答）；停止按钮；trace 折叠时间线；专业头部；focus-visible；JS 安全引号（`"`/`&quot;`，禁 `\x27`） |
| static/kg.html | Obsidian 风格力导向图谱（glow 滤镜、透镜悬停、隐藏标签、图例） |
| static/pipeline.html | teal 主题、错误处理、停止/复制按钮、`find()` 渲染 stats |
| static/test.html | API 测试面板、参数输入、curl 转义、AbortController、健康自刷新 |

---

## 七、附注

- 本清单为**链接驱动的采纳记录**，逐条对应代码内原注释，可复核。
- 未采纳项（G1-G6）均经**实测验证**后放弃（nan / 卡死 / 崩溃 / 破坏主流程），不是主观跳过，代码内保留注释。
- 非链接驱动的自主修复（RLock / logger 未定义 / HF 离线 / ownership / 搜索相关度）建议另开评审文档。
