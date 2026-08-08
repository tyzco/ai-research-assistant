"""产品模块 01 · 检索与资料获取
输入研究方向 → LLM 检索策略 → 12 学术源并行搜索 → 质量筛选 → 论文列表。

演示:
    python demo/demo_01_search.py "大语言模型代码漏洞自动修复"
"""
import sys

import common

QUERY = sys.argv[1] if len(sys.argv) > 1 else "大语言模型代码漏洞自动修复"


async def main():
    common.banner(
        "产品模块 01 · 检索与资料获取",
        "输入研究方向 → LLM 检索策略 → 12 学术源并行搜索 → 质量筛选",
    )
    from topic_manager import create_topic
    from search import generate_search_strategy, search_papers_for_topic

    common.section("1/3 创建课题")
    state = create_topic(QUERY)
    common.show("研究方向", QUERY)
    common.show("topic_id", state.topic_id)

    common.section("2/3 LLM 生成检索策略")
    common.step("生成中英关键词 / 布尔查询 / 推荐数据库")
    strat = {}
    try:
        strat = await generate_search_strategy(QUERY)
    except Exception as e:
        common.ok("LLM 策略生成失败，降级为默认关键词（" + str(e)[:60] + "）")
        strat = {"keywords_cn": [QUERY], "keywords_en": [QUERY], "boolean_queries": []}
    common.show("keywords_cn", strat.get("keywords_cn", []))
    common.show("keywords_en", strat.get("keywords_en", []))
    for bq in (strat.get("boolean_queries") or [])[:2]:
        common.show(
            "boolean_query",
            str(bq.get("database", "")) + ": " + str(bq.get("query", "")),
        )

    common.section("3/3 12 源并行搜索（真实调用）")
    common.step("arXiv / OpenAlex / Semantic Scholar / Crossref / CORE / PubMed ...")
    papers = await search_papers_for_topic(
        QUERY,
        strat.get("keywords_en"),
        strat.get("keywords_cn"),
        related_en=strat.get("related_terms_en"),
        related_cn=strat.get("related_terms_cn"),
    )
    common.ok("检索到 " + str(len(papers)) + " 篇（质量筛选后）")
    for p in papers[:8]:
        src = getattr(p, "source", "") or ""
        cite = " · 引用 " + str(p.citation_count) if p.citation_count else ""
        print(
            "    - " + (p.title or "")[:66] + " [" + str(p.year or "") + "] ("
            + src + ")" + cite
        )
        print("      OA=" + ("是" if p.is_oa else "否") + " · DOI=" + (p.doi or "-"))

    # 状态衔接 → demo_02(建库) / demo_03(问答) / demo_07(审稿)
    common.save_state(
        topic_id=state.topic_id,
        query=QUERY,
        n_papers=len(papers),
        papers=[p.model_dump() for p in papers],
    )
    md = (
        "# 模块 01 · 检索与资料获取\n\n"
        + "- 研究方向: " + QUERY + "\n"
        + "- topic_id: " + state.topic_id + "\n"
        + "- 检索策略: " + json_dumps(strat) + "\n"
        + "- 论文 " + str(len(papers)) + " 篇\n\n"
        + "\n".join(
            "- " + (p.title or "") + " [" + str(p.year or "") + "] OA=" + str(p.is_oa)
            for p in papers[:20]
        )
    )
    common.save_artifact("01_search.md", md)
    common.next_hint("python demo/demo_02_kb.py", "把检索结果构建成私有知识库")


def json_dumps(obj):
    import json

    return json.dumps(obj, ensure_ascii=False, default=str)[:800]


if __name__ == "__main__":
    common.run(main)
