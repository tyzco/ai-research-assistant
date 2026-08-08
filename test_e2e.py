"""端到端自动化测试：检索策略 → 嵌入模型 → 知识库构建 → 问答。

用法: python test_e2e.py
前提: 已配置 .env (DEEPSEEK_API_KEY 必填)

DeepSeek 审查改造: 从"人工验收检查清单"升级为自动化测试。
- #1 每步骤加断言 (关键词数量/嵌入维度/库行数/答案长度与引用)
- #3 唯一 topic_id + 结束后清理测试表 (可重复运行)
- #4 逐步骤独立记录, 失败不中断后续, 汇总报告
- #5 问答覆盖 事实/英文/无关问题 三类
- #7 环境预检 (KEY + 嵌入维度)
- #6 输出结构化 JSON 报告
"""
import asyncio
import json
import logging
import os
import time
import traceback
import uuid

# 离线/本地缓存加载模型, 避免 HF 联网检查卡顿
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("test_e2e")


async def main() -> int:
    from config import DEEPSEEK_API_KEY, LANCEDB_DIR, LOCAL_EMBEDDING_DIM

    # ---- 环境预检 (#7) ----
    if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY == "sk-your-key-here":
        print("❌ 缺少 DEEPSEEK_API_KEY，请先配置 .env")
        return 1

    query = "大模型代码漏洞自动修复"
    topic_id = f"test_e2e_{uuid.uuid4().hex[:8]}"  # #3 唯一 ID, 可重复运行
    t0 = time.time()
    table_name = ""
    report = {"steps": {}, "passed": 0, "failed": 0}

    async def step(name: str, fn):
        """执行一步并断言; 失败记录但不中断后续步骤 (#4)。"""
        try:
            await fn()
            report["steps"][name] = {"status": "PASS"}
            report["passed"] += 1
            print(f"  ✅ {name}")
        except AssertionError as e:
            report["steps"][name] = {"status": "FAIL", "error": str(e)}
            report["failed"] += 1
            print(f"  ❌ {name}: {e}")
        except Exception as e:
            report["steps"][name] = {
                "status": "ERROR",
                "error": f"{type(e).__name__}: {str(e)[:200]}",
            }
            report["failed"] += 1
            print(f"  ⚠️  {name}: {type(e).__name__}: {e}")
            traceback.print_exc()

    # ---- 1. 检索策略生成 (#1 断言) ----
    async def s1():
        from search import generate_search_strategy

        s = await generate_search_strategy(query)
        cn, en = s.get("keywords_cn", []), s.get("keywords_en", [])
        assert len(cn) >= 3, f"中文关键词不足 ({len(cn)})"
        assert len(en) >= 3, f"英文关键词不足 ({len(en)})"
        assert s.get("recommended_databases"), "推荐数据库为空"
        assert len(s.get("boolean_queries", [])) >= 1, "检索式为空"
        print(
            f"  关键词 {cn[:4]}/{en[:4]} | 库 {len(s.get('recommended_databases', []))} "
            f"| 式 {len(s.get('boolean_queries', []))}"
        )

    # ---- 2. 嵌入模型 (#1/#7 断言维度) ----
    async def s2():
        from knowledge_base import _embed_texts_sync

        texts = [
            "大模型代码漏洞自动修复研究",
            "基于深度学习的程序修复方法",
            "代码漏洞检测与安全分析",
        ]
        vecs = _embed_texts_sync(texts)
        assert len(vecs) == len(texts), f"向量数 {len(vecs)} != {len(texts)}"
        assert len(vecs[0]) == LOCAL_EMBEDDING_DIM, (
            f"嵌入维度 {len(vecs[0])} != 期望 {LOCAL_EMBEDDING_DIM}"
        )
        print(f"  维度 {len(vecs[0])} ✓")

    # ---- 3. 知识库构建 (#1/#3) ----
    async def s3():
        nonlocal table_name
        from knowledge_base import build_knowledge_base
        from models import PaperMeta

        papers = [
            PaperMeta(
                paper_id="p1",
                title="基于LLM的代码漏洞修复",
                authors="Zhang et al.",
                year=2024,
                abstract="本研究提出了一个基于大语言模型的代码漏洞自动修复框架...",
            ),
            PaperMeta(
                paper_id="p2",
                title="Deep Learning for Program Repair",
                authors="Chen et al.",
                year=2023,
                abstract="This paper surveys deep learning approaches for automated program repair...",
            ),
            PaperMeta(
                paper_id="p3",
                title="代码安全分析综述",
                authors="Wang et al.",
                year=2024,
                abstract="本文综述了近年来代码安全分析的主要方法...",
            ),
        ]
        fulltext_chunks = {
            "p1": [
                "我们提出 LLM-Repair 框架，结合代码表示学习和提示工程实现漏洞自动修复。首先使用 CodeBERT 生成代码嵌入...",
                "实验在 Defects4J 数据集上进行，修复成功率达到 67.3%，比基线方法提高 12 个百分点...",
            ],
            "p2": [
                "Deep learning-based program repair has emerged as a promising approach. Key techniques include sequence-to-sequence models, graph neural networks...",
                "The main challenges are: limited training data, difficulty in generating correct patches...",
            ],
            "p3": [
                "代码安全分析技术分为三大类：静态分析（如抽象解释、符号执行）、动态分析（如模糊测试）和基于AI的方法...",
            ],
        }
        table_name = await build_knowledge_base(papers, fulltext_chunks, topic_id)
        assert table_name and table_name.startswith("papers_"), f"表名异常: {table_name}"
        import lancedb

        db = lancedb.connect(str(LANCEDB_DIR))
        rows = db.open_table(table_name).count_rows()
        assert rows > 0, "知识库行数为 0"
        print(f"  表 {table_name} | {rows} 行")

    # ---- 4. 问答 (#1/#5: 事实/英文/无关三类 + 断言) ----
    async def s4():
        from qa_engine import ask_question

        cases = [
            ("这些论文提出了什么方法来进行代码漏洞修复？", ["修复", "LLM", "框架"]),
            ("Deep Learning for Program Repair 这篇论文的主要方法是什么？", ["Deep", "repair"]),
            ("与代码漏洞完全无关的宇宙起源问题，你能回答吗？", []),  # 边界: 无关问题也应有合理回复
        ]
        for q, expect in cases:
            result = await ask_question(table_name, q)
            assert len(result.answer) > 50, f"答案过短 ({len(result.answer)} 字)"
            assert len(result.references) >= 1, "无引用"
            if expect:
                hit = [k for k in expect if k.lower() in result.answer.lower()]
                assert hit, f"答案未包含预期词 {expect}"
            print(f"  ✓ {q[:26]}... ({len(result.answer)} 字 / {len(result.references)} 引用)")

    await step("检索策略生成", s1)
    await step("嵌入模型", s2)
    await step("知识库构建", s3)
    await step("问答测试", s4)

    # ---- 清理测试表 (#3) ----
    if table_name:
        try:
            import lancedb

            db = lancedb.connect(str(LANCEDB_DIR))
            raw = db.list_tables()
            tables = list(raw.tables) if hasattr(raw, "tables") else [str(t) for t in raw]
            if table_name in tables:
                db.drop_table(table_name)
                print(f"  已清理测试表 {table_name}")
        except Exception as e:
            print(f"  [警告] 清理失败: {e}")

    elapsed = time.time() - t0
    ok = report["failed"] == 0
    print("\n" + "=" * 60)
    print(f"{'✅ 端到端测试通过' if ok else '❌ 端到端测试失败'} "
          f"| 通过 {report['passed']} / 失败 {report['failed']} | 耗时 {elapsed:.1f}s")
    for name, r in report["steps"].items():
        print(f"  [{r['status']}] {name}" + (f" — {r['error']}" if r.get("error") else ""))

    try:
        with open("data/e2e_report.json", "w", encoding="utf-8") as f:
            json.dump(
                {**report, "query": query, "topic_id": topic_id, "elapsed_s": round(elapsed, 1)},
                f,
                ensure_ascii=False,
                indent=2,
            )
        print("报告已保存: data/e2e_report.json")
    except Exception:
        pass

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
