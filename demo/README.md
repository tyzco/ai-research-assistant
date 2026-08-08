# PaperLink 分段产品演示

面向投资 / 项目展示 / 吸引协作的分段演示。每个产品模块独立可跑、真实调用后端、输出即演示稿，模块间经 `demo/output/state.json` 状态衔接。

## 快速开始

```bash
# 产品演示总纲: 选题→初稿 8 步故事线, 带旁白+商业定调, 输出完整演示稿
python demo/pitch.py "你的研究方向"

# 一键串联全部模块（真实 LLM + 12 学术源 + 私有知识库）
python demo/run_all.py "你的研究方向"

# 或逐模块演示
python demo/demo_01_search.py "大语言模型代码漏洞自动修复"   # 检索
python demo/demo_09_gap_analysis.py "选题"                  # 研究空白分析（有没有人做过）
python demo/demo_02_kb.py --pdf 你的论文.pdf                 # 私有知识库（可追加全文）
python demo/demo_03_ask.py "这些论文主要解决了什么问题？"      # RAG 问答
python demo/demo_04_agent.py "调研RAG研究现状"               # Agent 即服务
python demo/demo_05_team.py [--full]                        # 13 Agent 科研团队
python demo/demo_06_writing.py                              # 论文写作与多格式导出
python demo/demo_08_draft_quality.py                        # 初稿质量清单 (选题→初稿 验收线)
python demo/demo_10_draft_review.py                         # 初稿审稿与批注 (模拟同行评审)
python demo/demo_11_self_evolution.py                       # 自我进化 (反思+反馈+技能记忆)
python demo/demo_12_proactive.py                            # 主动路由 (预判+执行+学习)
python demo/demo_07_review.py [paper_id]                    # 审稿 / 引用链验证
```

演示产物（Markdown 报告）保存到 `demo/output/`，可直接出示。

## 模块 → 产品定位

| 模块 | 脚本 | 产品能力 | 变现指向 |
|---|---|---|---|
| 01 检索 | demo_01 | 研究方向 → 检索策略 → 12 源并行搜索 | 引流 / 付费起点 |
| 09 研究空白 | demo_09 | 有没有人做过 → 真实检索 → 空白/可行性报告 | **选题差异化** |
| 02 私有知识库 | demo_02 | 用户自供论文 → 双层索引 + FTS | **核心护城河**（用户数据） |
| 03 知识库问答 | demo_03 | RAG 8 步链路 + 可溯源引用 | 免费层（DeepSeek 低成本） |
| 04 Agent 即服务 | demo_04 | 技能路由 + 工具链 + 人工确认门 | **付费层**（按结果卖） |
| 05 科研团队 | demo_05 | 13 Agent 三层流水线 + 3 闸门 | 高阶付费 |
| 06 写作辅助 | demo_06 | 章节合成 → MD/DOCX/LaTeX/BibTeX | 付费 / 增值 |
| 08 初稿质量清单 | demo_08 | 选题→初稿验收线：6 项自检 → pass/revise（研究问题/空白定位/结构/引用/诚实性/可续写） | **交付验收标准** |
| 10 审稿批注 | demo_10 | 模拟同行评审：5 维评分 + 逐条批注 + 待澄清问题 | 付费 / 增值 |
| 11 自我进化 | demo_11 | 反思重试 + 反馈教训 + 技能记忆 三环闭环 | **产品差异化** |
| 12 主动路由 | demo_12 | 预判下一步 → 一键执行 → 接受度学习 | **知用户·越用越懂** |
| 07 审稿验证 | demo_07 | 引用链追踪 / 同行评审 / 伦理审查 | 付费 / 增值 |

## 说明

- **合法边界**：仅处理 OA 内容 + 用户自己上传的论文，不含任何盗版获取能力。
- **可衔接**：模块间状态衔接（`state.json`），既可整体交付，也可拆分独立售卖/展示。
- **演示模式**：`demo_05` 默认展示架构，`--full` 跑真实全流程（多轮 LLM，自动通过人工闸门）。
- **真实调用**：所有演示走真实代码路径（LLM / 检索 / 向量库），非伪造数据。
