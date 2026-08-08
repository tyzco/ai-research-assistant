"""产品模块 06 · 论文写作辅助
知识库问答 → 结构化章节合成 → Markdown / DOCX / LaTeX / BibTeX 多格式导出。

演示:
    python demo/demo_06_writing.py
"""
import common


def main():
    common.banner(
        "产品模块 06 · 论文写作辅助",
        "章节合成 → 引用管理 → 多格式导出",
    )
    st = common.load_state()
    query = st.get("query", "未命名综述")
    npapers = st.get("n_papers", 0)

    common.section("章节合成")
    sections = {
        "摘要": "本综述基于私有知识库自动生成，围绕「" + query + "」系统梳理了 " + str(npapers) + " 篇论文的方法、实验与结论。",
        "引言": "研究背景与动机：围绕「" + query + "」方向，现有工作可从方法框架与评测基准两条线归纳。",
        "方法": "基于双层知识库的检索增强框架：先以摘要层广度召回，再以全文层深度精读，经级联重排后注入生成。",
        "结论": "主要发现与展望：技术路线趋于成熟，跨模态与可解释性仍是开放问题。",
    }
    for h, c in sections.items():
        common.show(h, c[:60])

    refs = [
        {
            "title": p.get("title", ""),
            "year": p.get("year"),
            "authors": p.get("authors", ""),
            "doi": p.get("doi"),
        }
        for p in st.get("papers", [])[:8]
        if p.get("title")
    ]

    common.section("多格式导出")
    from exporter import export_markdown, export_docx, export_latex, export_bibtex

    md = export_markdown(query, sections, refs)
    common.ok("Markdown: " + md)

    # 附加格式（异常不影响主流程）
    def _docx():
        return export_docx(query, sections, refs)

    def _latex():
        return export_latex(query, sections, refs)

    def _bib():
        return export_bibtex(refs)

    common.guard(_docx, "DOCX 导出（需 python-docx）")
    common.guard(_latex, "LaTeX 导出")
    common.guard(_bib, "BibTeX 导出")

    common.save_artifact("06_writing.md", open(md, encoding="utf-8").read())
    common.next_hint("python demo/demo_07_review.py", "审稿验证与引用链追踪")


if __name__ == "__main__":
    common.run(main)
