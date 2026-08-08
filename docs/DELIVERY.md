# PaperLink 交付总结

> 一句话: 面向 C 端研究者的「选题 → 初稿」Agent 即服务工具，能演示、能收钱、会自我进化。

## 验收状态（2026-08 全系统回归）

- ✅ 全部 Python 编译通过
- ✅ 全部前端 HTML JS 语法通过
- ✅ 全量模块导入通过
- ✅ 9 个关键读端点 200 + create_topic 正常
- ⚠️ LLM 类端点受本机网络抖动影响（间歇超时），稳定网络下正常（demo/单元测试已验证）

## 快速启动

```bash
cp .env.example .env   # 填 DEEPSEEK_API_KEY (+ 可选 Stripe 测试密钥)
python api.py          # 打开 http://localhost:8001
python demo/pitch.py "研究方向"   # 9 步产品演示
python demo/run_all.py "方向"     # 12 模块完整演示
python run_benchmark.py --kb papers_benchmark_mt_kb2   # 检索基准
```

## 产品能力（12 演示模块）

| # | 模块 | 能力 |
|---|---|---|
| 1 | 检索 | 12 源并行 + 相关度重排 + 研究空白分析 |
| 2-3 | 建库 | 私有双层知识库（4096 块 + 邻块±2） |
| 4-5 | 问答/Agent | RAG 证据回答 + 工具循环 + 人工确认门 |
| 6-8 | 写作/质量/审稿 | 质量清单 6 项自检 + 同行评审 |
| 9-10 | 自我进化/主动路由 | 反思+反馈+技能记忆；预判+执行+学习 |
| 11-12 | 引用链/全景 | 引用脉络 + 完整故事线 |

## 商业化

- **免费额度 → 用量追踪 → 超额 402 → Stripe 充值**（测试密钥已配）
- 定位: 选题→初稿, 完成一篇退场

## 可宣传弹药

- 产品结果: 选题→初稿 一条龙 + 差异化（研究空白/质量清单/自我进化/主动路由）
- 检索基准: MRR 0.72 / R@top5 0.78（43篇/30题/5主题，方法可复核）
- 文档: `PITCH_FACTS.md`（怎么说）/ `BENCHMARK.md`（怎么测）/ `SELF_EVOLUTION.md`（怎么进化）

## 关键文档索引

- `docs/PITCH_FACTS.md` — 宣传话术（面试/融资应对）
- `docs/BENCHMARK.md` — 评估方法与声明边界
- `docs/SELF_EVOLUTION.md` — 自我进化三环
- `docs/REVIEW_DEEPSEEK.md` — 链接驱动修复清单
- `docs/DEMO_SCRIPT.md` / `docs/ARCHITECTURE.md`

## 遗留事项（下一步）

1. **真实用户验证** — 建议接受数据/反馈积累, 让自我进化与主动路由真正学习
2. **生产部署** — Stripe live 密钥 + 真实域名 + 鉴权中间件
3. **多主题 KB 扩到 100 题** — 检索基准更可信
