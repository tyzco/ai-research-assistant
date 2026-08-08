# PaperLink 自我进化系统（loop-engineering）

> 从「harness 工程」（固定循环 + 固定工具）进化到「loop-engineering」（反馈→行为闭环）。
> 安全原则：**数据驱动，人类把关，Agent 只改进当次输出，不改写自己。**

## 三环闭环

```
P1 反思重试   — 当次答案: 生成 → 自批判 → 不足则修订
P2 反馈教训   — 跨会话: 低分反馈 → LLM 提炼教训 → 注入提示 (避免失败)
P3 技能记忆   — 跨会话: 成功任务 → 记录做法 → 相似任务复用 (复用成功)
```

| 环 | 文件 | 触发 | 数据源 | 效果 |
|---|---|---|---|---|
| **P1 反思重试** | `agent/pi_agent.py` | 每次直答后 | 当次答案 | 答案不完整 → 带批判意见重试一轮（有界 1 次） |
| **P2 反馈教训** | `feedback_lessons.py` + `database.learning_feedback` | 系统提示构建时 | 用户低分反馈 (rating≤2) | 「易失败教训」注入，规避同类错误 |
| **P3 技能记忆** | `agent_skills.py` + `database.agent_skill_memory` | 成功任务后记录 / 任务开始时匹配 | 成功任务的做法 | 「相似任务成功经验」注入，复用已验证方法 |

## 各环实现要点

### P1 反思重试
- 直答分支：答案生成后，独立 LLM 调用自批判（`_reflexion_critique`）
- 返回 PASS → 交付；非 PASS → `【反思】批判 + 请修订` 注入 → 续跑一轮
- 有界：`_reflexed` 标志 + `turn_idx < max_steps-1` 双保险，防死循环
- 配置：`AGENT_REFLEXION_ENABLED`（默认 true，false 省一次 LLM 调用）

### P2 反馈教训
- 低分反馈（rating≤2，≥3 条才蒸馏，防过拟合噪声）→ LLM 提炼 2-3 条可执行教训
- TTL 缓存 5 分钟（反馈变化慢）
- 注入 `_build_system_prompt` 的【历史反馈教训】段
- 安全：只提炼抽象教训，不扩散用户原始问答明文

### P3 技能记忆
- 成功任务（success=True）→ `record_success` 写入 `agent_skill_memory`（保留最新 100 条）
- 新任务 → `get_similar_strategies` 二元组重叠匹配（≥0.5）相似成功任务
- 注入【过往相似任务的成功经验】段：`相似任务「X」成功做法: 调用工具 [...]`
- 轻量：不做 LLM 蒸馏，直接用工具序列 + 结果预览作为"策略"

## 配套能力（对抗幻觉）

| 能力 | 文件 | 作用 |
|---|---|---|
| 1M 上下文窗口 | `config.AGENT_CONTEXT_MAX_TOKENS` | 论文高精度场景窗口最大化，证据全保留 |
| key_findings 填充 | `agent/pi_agent.py` | 压缩摘要带真实证据（原恒为"无"） |
| 初稿质量清单 | `draft_quality.py` | 6 项自检 → pass/revise，交付前把关 |
| 工具安全层 | `agent/tool_manager.py` | 参数校验 + 域名白名单 + 审计 |

## 演进路径

- [x] P1 反思重试
- [x] P2 反馈教训
- [x] P3 技能记忆
- [ ] 细粒度：P3 按任务类型精确匹配（当前二元组保守匹配）
- [ ] 规划级反思：工具任务也可反思（当前只覆盖直答路径）
- [ ] 评估闭环：自动评测分数 → 驱动提示/工具改进（人类批准）

## 关键文件

- `agent/pi_agent.py` — 反思环 + 反馈/技能注入点
- `feedback_lessons.py` — P2 教训蒸馏
- `agent_skills.py` — P3 技能记忆
- `database.py` — `learning_feedback` + `agent_skill_memory` 表
- `config.py` — `AGENT_REFLEXION_ENABLED` / `AGENT_CONTEXT_*`
