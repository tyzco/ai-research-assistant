"""学术论文导出模块 — DOCX / LaTeX / BibTeX / Markdown。

使用:
    from exporter import export_docx, export_latex, export_bibtex
    filepath = export_docx("论文标题", sections, references, output_dir="exports/")
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

EXPORT_DIR = Path("data/exports")
EXPORT_DIR.mkdir(parents=True, exist_ok=True)


def export_markdown(
    title: str, sections: dict[str, str], references: list[dict], output_path: str = ""
) -> str:
    """导出 Markdown 格式论文。"""
    path = Path(output_path) if output_path else EXPORT_DIR / f"{_safe_name(title)}.md"

    lines = [f"# {title}", "", "*由 PaperLink AI 生成*", ""]
    for heading, content in sections.items():
        lines.append(f"## {heading}")
        lines.append("")
        lines.append(content)
        lines.append("")
    lines.append("## 参考文献")
    lines.append("")
    for i, ref in enumerate(references, 1):
        authors = ref.get("authors", "Unknown")
        year = ref.get("year", "n.d.")
        title_r = ref.get("title", "Untitled")
        doi = ref.get("doi", "")
        url = f"https://doi.org/{doi}" if doi else ref.get("url", "")
        lines.append(f"[{i}] {authors}. ({year}). *{title_r}*. {url}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def export_docx(
    title: str, sections: dict[str, str], references: list[dict], output_path: str = ""
) -> str:
    """导出 Word (.docx) 论文。"""
    from docx import Document
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    path = (
        Path(output_path) if output_path else EXPORT_DIR / f"{_safe_name(title)}.docx"
    )
    doc = Document()

    # Title
    h = doc.add_heading(title, level=0)
    h.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph("由 PaperLink AI 生成").alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_paragraph()

    # Sections
    for heading, content in sections.items():
        doc.add_heading(heading, level=1)
        for para_text in content.split("\n\n"):
            para_text = para_text.strip()
            if para_text:
                doc.add_paragraph(para_text)

    # References
    doc.add_heading("参考文献", level=1)
    for i, ref in enumerate(references, 1):
        authors = ref.get("authors", "Unknown")
        year = ref.get("year", "n.d.")
        title_r = ref.get("title", "Untitled")
        doi = ref.get("doi", "")
        url = f"https://doi.org/{doi}" if doi else ref.get("url", "")
        p = doc.add_paragraph()
        p.add_run(f"[{i}] ").bold = False
        p.add_run(f"{authors}. ({year}). {title_r}. {url}")

    doc.save(str(path))
    return str(path)


def export_latex(
    title: str, sections: dict[str, str], references: list[dict], output_path: str = ""
) -> str:
    """导出 LaTeX 论文模板。"""
    path = Path(output_path) if output_path else EXPORT_DIR / f"{_safe_name(title)}.tex"

    lines = [
        r"\documentclass[11pt,a4paper]{article}",
        r"\usepackage[UTF8]{ctex}",
        r"\usepackage{hyperref}",
        r"\usepackage{geometry}",
        r"\geometry{margin=2.5cm}",
        r"\usepackage{natbib}",
        r"\bibliographystyle{unsrt}",
        r"",
        r"\title{%s}" % _tex_escape(title),
        r"\author{Powered by PaperLink AI}",
        r"\date{\today}",
        r"",
        r"\begin{document}",
        r"\maketitle",
        r"",
    ]

    for heading, content in sections.items():
        lines.append(r"\section{%s}" % _tex_escape(heading))
        lines.append("")
        for para_text in content.split("\n\n"):
            para_text = para_text.strip()
            if para_text:
                lines.append(_tex_escape(para_text))
                lines.append("")

    lines.append(r"\section*{参考文献}")
    if references:
        lines.append(r"\begin{thebibliography}{99}")
        for i, ref in enumerate(references, 1):
            authors = ref.get("authors", "Unknown")
            year = ref.get("year", "n.d.")
            title_r = ref.get("title", "Untitled")
            doi = ref.get("doi", "")
            url = f"https://doi.org/{doi}" if doi else ref.get("url", "")
            cite_key = ref.get("cite_key", f"ref{i}")
            lines.append(
                r"\bibitem{%s} %s. (%s). \\textit{%s}. %s"
                % (
                    cite_key,
                    _tex_escape(authors),
                    year,
                    _tex_escape(title_r),
                    _tex_escape(url) if url else "",
                )
            )
        lines.append(r"\end{thebibliography}")

    lines.append(r"\end{document}")
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def export_bibtex(references: list[dict], output_path: str = "") -> str:
    """导出 BibTeX 参考文献。"""
    path = Path(output_path) if output_path else EXPORT_DIR / "references.bib"

    entries = []
    for i, ref in enumerate(references, 1):
        authors = (
            ref.get("authors", "Unknown").split(",")[0].strip().split()[-1]
            if ref.get("authors")
            else "Unknown"
        )
        year = ref.get("year", "2024")
        title_r = ref.get("title", "Untitled")
        doi = ref.get("doi", "")
        cite_key = ref.get("cite_key", f"ref{i}")

        entries.append(
            "@article{%s,\n"
            "  author = {%s},\n"
            "  title = {%s},\n"
            "  year = {%s},\n"
            "  doi = {%s}\n"
            "}" % (cite_key, _tex_escape(authors), _tex_escape(title_r), year, doi)
        )

    path.write_text("\n\n".join(entries), encoding="utf-8")
    return str(path)


def _safe_name(title: str) -> str:
    """清理文件名中的非法字符。"""
    import re

    name = re.sub(r'[<>:"/\\|?*]', "_", title)[:80]
    return name.strip().replace(" ", "_")


def _tex_escape(text: str) -> str:
    """转义 LaTeX 特殊字符。

    `\\` 必须排第一: 其余每条规则都会**插入反斜杠**(`&`→`\\&`)。若 `\\` 最后执行,
    它会把这些刚插入的反斜杠再转义成 `\\textbackslash{}`, 结果是 `a_b` → `a\\textbackslash{}_b`
    —— 下划线没转义、还多出垃圾, 导出 .tex 无法编译。
    """
    replacements = {
        "\\": r"\textbackslash{}",  # 先跑, 只转义源文本里原有的反斜杠
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\^{}",
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    return text
