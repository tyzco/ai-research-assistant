"""端到端测试：新流程 → 检索策略生成 → 嵌入测试 → 知识库构建 → 问答。

用法：python test_e2e.py
前提：已配置 .env（DEEPSEEK_API_KEY 必填，VISION_API_KEY 可选）
"""

import asyncio
import logging
import time

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("test_e2e")


async def main():
    from config import DEEPSEEK_API_KEY

    if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY == "sk-your-key-here":
        logger.error("缺少 DEEPSEEK_API_KEY，请先配置 .env")
        return

    query = "大模型代码漏洞自动修复"
    topic_id = "test_e2e_v2"
    t0 = time.time()

    # ---- 1. 检索策略生成 ----
    print("\n" + "=" * 60)
    print("第 1 步：生成检索策略")
    print("=" * 60)
    try:
        from search import generate_search_strategy

        strategy = await generate_search_strategy(query)
        print(f"中文关键词: {strategy.get('keywords_cn', [])}")
        print(f"英文关键词: {strategy.get('keywords_en', [])}")
        print(f"推荐数据库: {strategy.get('recommended_databases', [])}")
        print(f"检索式数量: {len(strategy.get('boolean_queries', []))}")
        print(f"关键学者: {len(strategy.get('top_authors', []))} 位")
    except Exception as e:
        logger.error(f"检索策略生成失败: {e}")
        return

    # ---- 2. 嵌入模型测试 ----
    print("\n" + "=" * 60)
    print("第 2 步：本地嵌入模型测试")
    print("=" * 60)
    try:
        from knowledge_base import _embed_texts_sync

        t_emb = time.time()
        test_texts = [
            "大模型代码漏洞自动修复研究",
            "基于深度学习的程序修复方法",
            "代码漏洞检测与安全分析",
        ]
        vecs = _embed_texts_sync(test_texts)
        print(
            f"编码 {len(test_texts)} 条文本，维度 {len(vecs[0])}，耗时 {time.time() - t_emb:.1f}s"
        )
    except Exception as e:
        logger.error(f"嵌入测试失败: {e}")
        return

    # ---- 3. 知识库构建（用模拟数据） ----
    print("\n" + "=" * 60)
    print("第 3 步：构建知识库（模拟数据）")
    print("=" * 60)
    try:
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
                abstract="本文综述了近年来代码安全分析的主要方法，包括静态分析、动态分析和基于AI的方法...",
            ),
        ]

        fulltext_chunks = {
            "p1": [
                "我们提出 LLM-Repair 框架，结合代码表示学习和提示工程实现漏洞自动修复。首先使用 CodeBERT 生成代码嵌入...",
                "实验在 Defects4J 数据集上进行，修复成功率达到 67.3%，比基线方法提高 12 个百分点...",
            ],
            "p2": [
                "Deep learning-based program repair has emerged as a promising approach. Key techniques include sequence-to-sequence models, graph neural networks...",
                "The main challenges are: limited training data, difficulty in generating correct patches, and lack of explainability...",
            ],
            "p3": [
                "代码安全分析技术分为三大类：静态分析（如抽象解释、符号执行）、动态分析（如模糊测试）和基于AI的方法...",
            ],
        }

        t_kb = time.time()
        table_name = await build_knowledge_base(papers, fulltext_chunks, topic_id)
        print(f"表名: {table_name}")
        print(f"知识库构建耗时: {time.time() - t_kb:.1f}s")
    except Exception as e:
        logger.error(f"知识库构建失败: {e}")
        import traceback

        traceback.print_exc()
        return

    # ---- 4. 问答测试 ----
    print("\n" + "=" * 60)
    print("第 4 步：问答测试")
    print("=" * 60)
    try:
        from qa_engine import ask_question

        questions = [
            "这些论文提出了什么方法来进行代码漏洞修复？",
            "各方法的主要优缺点是什么？",
        ]

        for q in questions:
            print(f"\n--- 提问: {q} ---")
            result = await ask_question(table_name, q)
            print(f"\n回答:\n{result.answer[:600]}...")
            print(f"\n引用文献 ({len(result.references)} 篇):")
            for r in result.references:
                print(f"  [{r['source']}] {r['title']} ({r.get('year', 'N/A')})")
            if result.supplement:
                print(f"\n建议补充 ({len(result.supplement)} 篇):")
                for s in result.supplement[:3]:
                    print(f"  - {s['title']} ({s.get('year', 'N/A')})")
    except Exception as e:
        logger.error(f"问答失败: {e}")
        import traceback

        traceback.print_exc()
        return

    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print(f"✅ 端到端测试通过，总耗时 {elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
