"""产品模块 02 · 私有知识库
用户论文（检索结果 或 用户自供 PDF）→ 双层 LanceDB 索引（摘要层 + 全文层）+ FTS 全文检索。

演示:
    python demo/demo_02_kb.py                # 用上一步检索结果的摘要建库
    python demo/demo_02_kb.py --pdf 论文.pdf  # 追加用户自供 PDF 的全文精读
"""
import argparse
import hashlib
from pathlib import Path

import common


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default="", help="可选: 本地 PDF, 构建全文知识库")
    args = ap.parse_args()

    common.banner(
        "产品模块 02 · 私有知识库",
        "用户论文 → 双层 LanceDB 索引（摘要+全文）→ FTS 全文检索",
    )
    from models import PaperMeta
    from knowledge_base import build_knowledge_base
    import lancedb
    from config import LANCEDB_DIR

    st = common.load_state()
    papers = [PaperMeta(**p) for p in st.get("papers", [])[:20]]
    if not papers:
        print("  无上一步检索结果。请先运行: python demo/demo_01_search.py")
        return

    common.section("摘要层索引")
    common.step("向量化 " + str(len(papers)) + " 篇论文摘要")
    fulltext_chunks = {}
    if args.pdf:
        common.section("全文精读（用户自供 PDF）")
        from downloader import parse_pdf, chunk_text

        pdf_path = Path(args.pdf)
        data = parse_pdf(pdf_path.read_bytes(), pdf_path.name)
        chunks = chunk_text(data.get("text", "") or "", 1200)
        upid = "user_" + hashlib.md5(pdf_path.name.encode()).hexdigest()[:8]
        papers.append(
            PaperMeta(
                paper_id=upid,
                title="用户论文: " + pdf_path.name,
                abstract=data.get("title", "") or "",
            )
        )
        fulltext_chunks[upid] = chunks
        common.ok(str(len(chunks)) + " 个文本块 → 全文层")

    table = await build_knowledge_base(papers, fulltext_chunks, st["topic_id"])
    common.ok("知识库构建完成")

    # 写回 topic 状态: 让 API (_resolve_kb_table) 能解析此表, web 端 /ask 也可用
    try:
        from models import TopicStatus
        from topic_manager import active_topics, save_topic_state

        state = active_topics.get(st["topic_id"])
        if state:
            state.lancedb_table = table
            state.status = TopicStatus.READY
            save_topic_state(st["topic_id"])
    except Exception as e:
        common.ok("写回 topic 状态失败（不影响 demo 本机查询）: " + str(e)[:60])

    db = lancedb.connect(str(LANCEDB_DIR))
    tbl = db.open_table(table)
    common.show("LanceDB 表", table)
    common.show("行数", tbl.count_rows())
    common.show("论文数", len(papers))
    common.show("全文层", str(len(fulltext_chunks)) + " 篇")

    common.save_state(table=table)
    common.save_artifact(
        "02_kb.md",
        "# 模块 02 · 私有知识库\n\n"
        + "- 论文 " + str(len(papers)) + " 篇\n"
        + "- 表: " + table + "\n"
        + "- 行数: " + str(tbl.count_rows()) + "\n"
        + "- 全文层: " + str(len(fulltext_chunks)) + " 篇",
    )
    common.next_hint("python demo/demo_03_ask.py", "基于知识库提问")


if __name__ == "__main__":
    common.run(main)
