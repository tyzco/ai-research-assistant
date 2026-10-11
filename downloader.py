"""论文处理模块：PDF 文本提取 + 图片提取 + 分块。"""

import logging
import re
from pathlib import Path

import fitz  # PyMuPDF

import config
from config import CHUNK_SIZE, IMAGE_DIR, PAPER_CACHE_DIR

logger = logging.getLogger(__name__)

PAPER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)


def _column_order_switches(blocks: list, page_width: float) -> tuple[int, int]:
    """统计"正文块"在给定顺序下的左右栏切换次数。

    只看窄块 (<=70% 页宽) 且文本量足够的块 —— 跨栏标题/宽图/宽表会污染判断。
    返回 (切换次数, 参与统计的块数); 块太少返回 (0, 0) 表示无法判定。
    """
    narrow = [
        b for b in blocks
        if (b[2] - b[0]) <= page_width * 0.7 and len((b[4] or "").strip()) >= 60
    ]
    if len(narrow) < 6:
        return 0, 0
    mid = page_width / 2
    seq = [0 if (b[0] + b[2]) / 2 < mid else 1 for b in narrow]
    return sum(1 for a, b in zip(seq, seq[1:]) if a != b), len(seq)


def two_column_blocks(page) -> list | None:
    """双栏页面按栏重排后的文本块 (只换顺序, 不增删)。未检测到串栏 → None。

    背景: PyMuPDF 默认按内容流顺序输出, 部分双栏论文会串栏 —— 实测某论文第 4 页
    顺序为 左→右→右→右→左→左 (左栏的 Table II/III 排到了右栏正文之后), 一个 chunk
    里左栏半句接右栏半句再跳回左栏。抽样 29 篇有正文可判定的论文, 4 篇中招。

    安全性: 只**重排同一批文本块**, 不增删内容; 未命中检测的页面完全走原路径,
    因此对正常的 86% 零影响。返回 None 时调用方应回退 page.get_text()。
    """
    try:
        blocks = [b for b in page.get_text("blocks") if (b[4] or "").strip()]
    except Exception:
        return None
    if len(blocks) < 8:
        return None

    width = page.rect.width
    switches, n = _column_order_switches(blocks, width)
    if n == 0:
        return None
    # 切换率低 → 本来就是按栏顺序, 不动
    if switches / n < config.TWO_COLUMN_SWITCH_RATIO:
        return None
    # 两侧都要有足够的正文块, 否则不是双栏页
    mid = width / 2
    narrow = [
        b for b in blocks
        if (b[2] - b[0]) <= width * 0.7 and len((b[4] or "").strip()) >= 60
    ]
    left = [b for b in narrow if (b[0] + b[2]) / 2 < mid]
    right = [b for b in narrow if (b[0] + b[2]) / 2 >= mid]
    if len(left) < 3 or len(right) < 3:
        return None

    # 跨栏块 (页眉/宽图/宽表) 归到其 y 所在的位置, 与左栏同组 → 保持原有纵向顺序
    def key(b):
        is_narrow = (b[2] - b[0]) <= width * 0.7
        col = 0 if (not is_narrow or (b[0] + b[2]) / 2 < mid) else 1
        return (col, b[1])

    ordered = sorted(blocks, key=key)
    # 内容守恒校验: 重排前后必须是同一批文本, 否则宁可放弃 (不冒丢字风险)
    if sorted(b[4] for b in ordered) != sorted(b[4] for b in blocks):
        return None
    return ordered


def _rows_to_markdown(rows: list[list[str]]) -> str:
    """二维单元格 → Markdown 管道表。单元格内换行用空格连接 (保持 token 边界)。

    用空格而非 `<br>`: `<br>` 会把一个 token 拆成两个, 使上下游比对与后续检索都失真。
    """
    def _flat(c: str) -> str:
        return " ".join((c or "").split())

    # 展开"合并行": find_tables 在无横线的分组表里会把多行并成一个逻辑行
    # (单元格内用 \n 分隔), 例如 4 个模型 4 个指标被并成 1 行 × 4 格。
    # 当同一行的各单元格行数一致时还原成多行 —— 否则模型看到一个格子里挤着 4 个数值,
    # 无法判断哪个数属于哪个模型。
    expanded: list[list[str]] = []
    for r in rows:
        sub = [[c for c in (cell or "").split("\n")] for cell in r]
        if len(r) > 1 and all(len(x) == len(sub[0]) and len(x) > 1 for x in sub):
            for i in range(len(sub[0])):
                expanded.append([x[i] for x in sub])
        else:
            expanded.append(list(r))
    rows = expanded

    cols = max(len(r) for r in rows)
    norm = [list(r) + [""] * (cols - len(r)) for r in rows]
    head = "| " + " | ".join(_flat(c) for c in norm[0]) + " |"
    sep = "| " + " | ".join("---" for _ in range(cols)) + " |"
    body = [
        "| " + " | ".join(_flat(c) for c in r) + " |" for r in norm[1:] if any(r)
    ]
    return "\n".join([head, sep] + body)


def _page_headings(page) -> list[str]:
    """按字号启发式检出本页标题 (**只读, 不改文本**)。

    PDF 没有 DOCX 那样的显式标题层级, 只能靠排版线索。实测同一篇 arXiv 论文里
    正文 9.96pt、标题 11.96/17.2/20pt, 分离得很干净; 检出结果正是
    CONVNET CONFIGURATIONS / CLASSIFICATION FRAMEWORK / CONCLUSION / REFERENCES。

    判据: 字号 > 正文最高频字号 × 1.12 且 行文本 2-80 字且 不是纯数字/页码。
    返回按阅读顺序的标题文本 (供按偏移绑定到 chunk)。
    """
    try:
        d = page.get_text("dict")
    except Exception:
        return []
    lines: list[tuple[float, str]] = []
    for b in d.get("blocks", []):
        for ln in b.get("lines", []):
            txt = "".join(sp.get("text", "") for sp in ln.get("spans", [])).strip()
            if not txt:
                continue
            sz = max((sp.get("size", 0) for sp in ln.get("spans", [])), default=0)
            lines.append((round(sz, 1), txt))
    if len(lines) < 6:
        return []
    from collections import Counter as _C

    body = _C(sz for sz, _ in lines).most_common(1)[0][0]
    out = []
    for sz, t in lines:
        if sz <= body * 1.12 or not (2 <= len(t) <= 80):
            continue
        if re.fullmatch(r"[\d\s.\-]+", t):
            continue
        # 页眉/页脚误判: arXiv 预印本戳、纯日期、"Published as a conference paper" 之类
        # 也是大字号, 但它们不是章节 —— 实测第一条章节曾被判成
        # "arXiv:1409.1556v6 [cs.CV] 10 Apr 2015"。
        low = t.lower()
        if low.startswith("arxiv:") or "conference paper at" in low:
            continue
        if re.search(r"\b\d{1,2}\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b", low):
            continue
        out.append(t)
    return out


def _table_regions(page) -> list[dict]:
    """检出本页表格, 返回 [{bbox, markdown}] (失败返回空 list, 不抛)。"""
    out: list[dict] = []
    try:
        tabs = page.find_tables()
    except Exception:
        return out
    for t in getattr(tabs, "tables", []) or []:
        try:
            rows = getattr(t, "row_count", 0) or 0
            cols = getattr(t, "col_count", 0) or 0
            # 太小的"表格"多是版式误检 (两格边框), 换掉反而更糟
            if rows < 2 or cols < 2:
                continue
            # 用 extract() 的**原始单元格文本**自己拼 markdown, 不用 to_markdown():
            # 实测 to_markdown() 会改坏数值 —— 单元格里的 `76.1` 被输出成 `76_._1`
            # (find_tables 的 markdown 转义), 对指标表这是硬伤, 还会让下游的覆盖校验
            # 永远匹配不上。extract() 拿到的是干净文本。
            rows = t.extract()
            rows = [[(c or "").strip() for c in row] for row in rows if row]
            rows = [r for r in rows if any(r)]
            if len(rows) < 2 or len(rows[0]) < 2:
                continue
            md = _rows_to_markdown(rows)
            if not md:
                continue
            out.append({"bbox": tuple(t.bbox), "markdown": md})
        except Exception:
            continue
    return out


def _render_blocks_with_tables(page, blocks: list) -> str | None:
    """把落在表格区域内的文本块**替换**为该表格的 Markdown, 其余块原样保留。

    为什么要替换而不是追加: PDF 表格用 get_text 读出来是**竖着一列一列的数字流**
    (实测某页输出 "Data c l N |V | |Vpre| Test MR 2 20 10662 18765 16448 CV SST-1"),
    列名与数值糊在一起, 模型无法判断哪个数属于哪一行。追加会造成同一批数字出现两遍;
    替换后表格既可读又不重复。

    内容守恒: Markdown 用的就是同一批单元格文本 —— 只是重新组织, 不增删。
    返回 None 表示本页没有表格 (调用方走原路径, 行为不变)。
    """
    regions = _table_regions(page)
    if not regions:
        return None, 0

    def _in_table(b) -> dict | None:
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        for r in regions:
            x0, y0, x1, y1 = r["bbox"]
            if x0 - 2 <= cx <= x1 + 2 and y0 - 2 <= cy <= y1 + 2:
                return r
        return None

    import re as _re

    from collections import Counter as _Ctr

    def _chars(s: str) -> "_Ctr":
        """**token 多重集** —— 判定"表格 markdown 是否完整覆盖了扁平文本"。

        试过三种判据, 前两种都被实测否掉:
          1. 子串: markdown 是**列优先**(一格塞进整列值, 用 <br> 分隔), 扁平块是**行优先**,
             顺序不同 → 必然判不覆盖 → 原块保留 → 同一批数字出现两遍
          2. 字符多重集: 与顺序无关了, 但**丢 token 边界** —— 扁平块里的 `1024` 在 markdown
             里被拆进两格时, 字符还是那些字符, 判定通过 → token 静默消失
             (实测 5 篇论文丢了 c/a/and/mIoU/FPS/52.5)
          3. token 多重集(本实现): 与顺序无关 **且** 保留 token 身份。
             代价是 <br>/下划线造成的 token 形态差异会让部分表格判不覆盖 → 退回扁平文本
             (可读性差但不丢字)。宁可漏修, 不冒丢字。
        """
        return _Ctr(t.lower() for t in _re.findall(r"[A-Za-z0-9_]{1,}", s or ""))


    # === 逐表判定 (不是逐块, 也不是整页) ===
    # 为什么按表整体判: 逐块比对时, 一个块里多出的 'a' 可以被表格里**其它单元格**的
    # 'a' 抵消掉 —— 实测这样会静默丢掉 c/a/and 这类高频字符 (5 篇论文中招)。
    # 表格是原子单元: 要么该表整体被 markdown 完整覆盖(丢弃扁平文本), 要么整表退回原文。
    # 判定用字符多重集而非子串: markdown 是列优先、扁平块是行优先, 顺序不同但字符相同。
    drop_ids: set[int] = set()
    for r in regions:
        mc = _chars(r["markdown"])
        inside = [b for b in blocks if _in_table(b) is r]
        if not inside:
            continue
        union = _Ctr()
        for b in inside:
            union += _chars((b[4] or "").strip())
        if union and not (union - mc):
            drop_ids |= {id(b) for b in inside}

    out: list[str] = []
    emitted: set[int] = set()
    for b in blocks:
        txt = (b[4] or "").strip()
        r = _in_table(b)
        if r is not None:
            key = id(r)
            if key not in emitted:
                emitted.add(key)
                out.append(r["markdown"])
            if id(b) not in drop_ids and txt:
                out.append(txt)  # 该表未被完整覆盖 → 保留扁平文本(接受重复, 不冒丢字)
            continue
        if txt:
            out.append(txt)
    for r in regions:
        if id(r) not in emitted:
            out.append(r["markdown"])
    # 结构化成功 = 该表的扁平文本确实被替换掉了
    _structured = sum(
        1 for r in regions if any(id(b) in drop_ids for b in blocks if _in_table(b) is r)
    )
    return ("\n".join(out) if out else None), _structured


def page_text_two_column(page) -> str | None:
    """双栏页面按栏重排文本。**仅在检测到串栏时**返回重排结果, 否则 None。
    调用方 (parse_pdf) 在 None 时回退到 page.get_text(), 保证未命中页面行为不变。
    """
    blocks = two_column_blocks(page)
    if blocks is None:
        return None
    return "\n".join((b[4] or "").strip() for b in blocks)


def _docx_table_to_text(table) -> str:
    """把 Word 表格转成 Markdown 管道表 —— 保留行列结构, 不被切碎成散句。"""
    rows = []
    for row in table.rows:
        cells = [" ".join(c.text.split()) for c in row.cells]
        if any(cells):
            rows.append("| " + " | ".join(cells) + " |")
    if not rows:
        return ""
    header_sep = "| " + " | ".join("---" for _ in table.rows[0].cells) + " |"
    return "\n".join([rows[0], header_sep] + rows[1:])


_MATH_NS = "http://schemas.openxmlformats.org/officeDocument/2006/math"


def _docx_extra_text(p_element) -> str:
    """提取 python-docx 段落 API **读不到**的 Word 内容: OMML 公式 + 文本框。

    背景 (实测): 富文本排版的东西用普通 API 会**静默丢失** ——
      - 内嵌公式存为 OMML (`m:oMath`), 不在 paragraph.text 里
      - 悬浮/嵌入文本框存在 run 内的 `w:txbxContent`, 整段不在 doc.paragraphs 里
    对一个读论文的系统, 丢公式和丢"图注/补充说明"都是硬伤: 读者看到的正文会
    缺一截, 而且**没有任何报错**。这里按 XML 把两类内容捞回来。

    公式线性化成 $...$ (下游检索/展示都能看出这是公式, 而不是混进正文的乱码);
    文本框加 [文本框] 前缀, 让模型知道这段不在正文流里。
    """
    try:
        from docx.oxml.ns import qn

        parts: list[str] = []
        for omath in p_element.iter(f"{{{_MATH_NS}}}oMath"):
            txt = "".join(
                (t.text or "") for t in omath.iter(f"{{{_MATH_NS}}}t")
            ).strip()
            if txt:
                parts.append(f"${txt}$")
        for tb in p_element.iter(qn("w:txbxContent")):
            txt = "".join((t.text or "") for t in tb.iter(qn("w:t"))).strip()
            if txt:
                parts.append(f"[文本框] {txt}")
        return " ".join(parts)
    except Exception:
        return ""


def parse_docx(docx_bytes: bytes, name: str) -> dict:
    """解析 .docx: 按标题层级切分, 每个段落绑定其所属标题路径。

    与 PDF 的根本差异: Word 的标题层级 (Heading 1/2/3) 是**显式结构**, 不需要靠
    字号/加粗去猜。因此这里能做真正的结构感知切分 —— 每个 chunk 都知道自己属于
    「2.1 实验设置」, 检索命中后可以给出可核验的章节定位 (PDF 那条路做不到)。

    返回 {
        "full_text": str,
        "sections": [{"heading": str, "level": int, "start": int, "end": int}],
        "tables": int,
        "quality": ...  (由 assess_text_quality 复用 pages 字段)
    }
    """
    import io

    import docx  # python-docx

    out = {"full_text": "", "sections": [], "tables": 0, "pages": []}
    try:
        d = docx.Document(io.BytesIO(docx_bytes))
    except Exception as e:
        logger.warning(f"Failed to open DOCX '{name}': {e}")
        return out

    parts: list[str] = []
    stack: list[tuple[int, str]] = []  # (level, title) 当前标题路径
    cur_title, cur_start, cur_level = "（正文）", 0, 1
    table_count = 0

    def flush(end_pos: int, next_title: str, next_level: int = 1):
        """收尾当前小节 (标题路径变化时调用)。"""
        nonlocal cur_title, cur_start, cur_level
        if end_pos > cur_start:
            out["sections"].append(
                {"heading": cur_title, "level": cur_level,
                 "start": cur_start, "end": end_pos}
            )
        cur_title, cur_start, cur_level = next_title, end_pos, next_level

    def heading_text(p):
        """取段落文本; 标题段落返回 (层级, 文本), 否则 None。"""
        style = (p.style.name or "") if p.style is not None else ""
        if style.startswith("Heading"):
            try:
                lvl = int(style.split()[-1])
            except (ValueError, IndexError):
                lvl = 1
            t = p.text.strip()
            return (lvl, t) if t else None
        if style in ("Title", "Subtitle") and p.text.strip():
            return (0, p.text.strip())
        return None

    # 按**文档顺序**遍历 body: d.paragraphs / d.tables 是两个独立集合,
    # 分开遍历会把所有表格挪到文末 (实测结果表被排到"结论"之后, 章节归属全错)。
    from docx.table import Table as _DocxTable
    from docx.text.paragraph import Paragraph as _DocxParagraph

    for child in d.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            p = _DocxParagraph(child, d)
            h = heading_text(p)
            if h is not None:
                lvl, title = h
                # 维护标题路径: 同级或更高级 → 弹栈
                while stack and stack[-1][0] >= lvl:
                    stack.pop()
                stack.append((lvl, title))
                path = " > ".join(t for _, t in stack)
                pos = len("".join(parts))
                flush(pos, path, lvl or 1)
                parts.append(f"{'#' * max(1, lvl)} {title}\n")
                continue
            txt = p.text.strip()
            # 段落 API 之外的内容 (公式/文本框) 单独捞, 不与正文混在一起
            extra = _docx_extra_text(p._p)
            if txt or extra:
                parts.append((txt + ("\n" if txt else "")) + (extra + "\n" if extra else ""))
        elif tag == "tbl":
            md = _docx_table_to_text(_DocxTable(child, d))
            if md:
                table_count += 1
                # 表格归属当前章节 (不另起节), 这样引用它的章节定位才是对的;
                # 表格自身仍在文档原位, 不被挪到文末
                parts.append(md + "\n")

    flush(len("".join(parts)), "（结尾）", 1)

    out["full_text"] = "".join(parts)
    out["tables"] = table_count
    # 复用 assess_text_quality: 它读 pages 算每页字数, 这里以"节"计数不适用,
    # 故提供单页语义的 pages 占位 (整篇算 1 页), 阈值判定交给调用方按需关闭
    out["pages"] = [{"page_num": 1, "text": out["full_text"]}]
    return out


def chunk_text_sectioned(
    text: str,
    sections: list[dict],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int | None = None,
) -> list[dict]:
    """**按章节**切分, 每块严格属于一个章节: [{"text", "page_num": 0, "section": str}]。

    为什么按节切而不是整篇按字数切: 整篇切会让一个块横跨多个章节, 而块只能带一个
    section 标签 —— 标签就变成了谎话 (实测一块从"3.2 主实验结果"跨到"4 结论", 却标成 3.2)。
    既然要拿这个标签做可核验的溯源, 就必须保证块不跨节。

    节内仍用同一个 splitter (递归语义切分 + 15% 重叠), 单节超长时正常分成多块。
    """
    if not sections:
        return [{"text": c, "page_num": 0, "section": ""}
                for c in chunk_text(text, chunk_size, chunk_overlap)]

    out: list[dict] = []
    for s in sections:
        seg = text[s["start"]:s["end"]]
        if not seg.strip():
            continue
        for c in chunk_text(seg, chunk_size, chunk_overlap):
            out.append({"text": c, "page_num": 0, "section": s["heading"]})
    return out


def assess_text_quality(parsed: dict) -> dict:
    """解析质量评估闸门: 拦截"解析出来是垃圾"的文档, 不要污染知识库。

    三个信号 (阈值见 config, 默认远离正常分布 → 只拦真坏的, 不误杀):
      - 每页有效字符数过低 → 扫描版/纯图 PDF, 没有文本层 (需 OCR, 当前不具备)
      - 乱码率过高 → 替换符/私用区/控制字符占比高
      - 字符单一度过高 → 字体未嵌入时 PyMuPDF 会把 CID 替换成**同一个合法字符**
        (实测: 私用区乱码被替换成 '·', 占 97.5%), 乱码率检查抓不到, 但字符单一度能抓

    参照 43 篇真实 arXiv 论文实测:
      每页中位 3351 字 (最小 668) | 乱码率中位 0.01% (最大 0.83%)
      最常见字符占比中位 10.3% (最大 12.4%) → 阈值 50% 留 4 倍余量

    返回 {"ok", "reason", "chars_per_page", "garbage_ratio", "domination_ratio"}
    """
    import unicodedata
    from collections import Counter

    pages = parsed.get("pages") or []
    text = (parsed.get("full_text") or "").strip()
    n_pages = max(1, len(pages))
    n = len(text)
    chars_per_page = n / n_pages

    if n == 0:
        return {
            "ok": False,
            "reason": "PDF 无可提取文本 (可能是扫描件/图片版, 需 OCR)",
            "chars_per_page": 0,
            "garbage_ratio": 1.0,
            "domination_ratio": 1.0,
        }

    # 乱码字符: 替换符 + 私用区 (未嵌入字体的 CID) + 控制/未分配字符
    bad = sum(
        1
        for c in text
        if c == "�"
        or 0xE000 <= ord(c) <= 0xF8FF
        or (unicodedata.category(c) in ("Cc", "Co", "Cn") and c not in "\n\r\t")
    )
    garbage_ratio = bad / n

    # 字符单一度: 最高频非空白字符的占比 (真实论文 <=12%, CID 乱码可达 90%+)
    counts = Counter(ch for ch in text if not ch.isspace())
    total_nonspace = sum(counts.values())
    domination = (counts.most_common(1)[0][1] / total_nonspace) if total_nonspace else 1.0

    def _verdict(ok: bool, reason: str) -> dict:
        return {
            "ok": ok,
            "reason": reason,
            "chars_per_page": int(chars_per_page),
            "garbage_ratio": garbage_ratio,
            "domination_ratio": domination,
        }

    if not config.DOC_QUALITY_GATE_ENABLED:
        return _verdict(True, "")
    if chars_per_page < config.DOC_QUALITY_MIN_CHARS_PER_PAGE:
        return _verdict(
            False,
            f"每页仅 {chars_per_page:.0f} 字符 (阈值 {config.DOC_QUALITY_MIN_CHARS_PER_PAGE}), 疑扫描件",
        )
    if garbage_ratio > config.DOC_QUALITY_MAX_GARBAGE_RATIO:
        return _verdict(
            False,
            f"乱码率 {garbage_ratio:.1%} (阈值 {config.DOC_QUALITY_MAX_GARBAGE_RATIO:.0%}), 疑字体未嵌入",
        )
    if domination > config.DOC_QUALITY_MAX_DOMINATION_RATIO:
        return _verdict(
            False,
            f"字符单一度 {domination:.0%} (阈值 {config.DOC_QUALITY_MAX_DOMINATION_RATIO:.0%}), 疑字体未嵌入导致乱码",
        )
    return _verdict(True, "")


def parse_pdf(pdf_bytes: bytes, pdf_name: str) -> dict:
    """
    解析单个 PDF：提取全文 + 图片。

    返回 {
        "full_text": str,
        "pages": [{"page_num": int, "text": str}],
        "page_offsets": [{"start": int, "end": int, "page_num": int}],  # 供分块反查真实页码
        "images": [
            {"image_path": str, "page_num": int, "bbox": tuple}
        ]
    }
    """
    result = {"full_text": "", "pages": [], "page_offsets": [], "sections": [], "images": []}
    # (绝对偏移, 标题文本) —— 逐页记录, 最后拼成 sections
    _heading_marks: list[tuple[int, str]] = []

    safe_name = Path(pdf_name).stem.replace(" ", "_")[:40]

    # 图片上限 (DeepSeek 审查 #4: 防止大 PDF 内存/磁盘激增)
    max_images = getattr(config, "MAX_IMAGES_PER_PDF", 200)

    reordered_pages = 0  # 触发双栏重排的页数 (观测用)
    tables_found = 0  # 结构化出来的表格数 (观测用)

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        logger.warning(f"Failed to open PDF '{pdf_name}': {e}")
        return result

    try:
        for page_idx, page in enumerate(doc):
            page_num = page_idx + 1
            try:
                text = page.get_text().strip()
                # 双栏串栏修正: 命中才重排 (只换顺序, 已做内容守恒校验), 未命中完全不改
                _blocks = None
                if config.TWO_COLUMN_REORDER_ENABLED:
                    _blocks = two_column_blocks(page)
                    if _blocks:
                        text = "\n".join((b[4] or "").strip() for b in _blocks)
                        reordered_pages += 1
                # 表格结构化: 命中才替换表格区域 (内容守恒, 只重新组织), 未命中不改
                if config.TABLE_EXTRACT_ENABLED:
                    _src = _blocks or [
                        b for b in page.get_text("blocks") if (b[4] or "").strip()
                    ]
                    _tbl, _nstruct = _render_blocks_with_tables(page, _src)
                    if _tbl:
                        text = _tbl.strip()
                        tables_found += _nstruct  # 只统计真正被结构化的
            except Exception as e:
                logger.warning(f"Page {page_num} text extraction failed: {e}")
                text = ""
            result["pages"].append({"page_num": page_num, "text": text})
            _start = len(result["full_text"])
            result["full_text"] += text + "\n\n"
            result["page_offsets"].append(
                {"start": _start, "end": _start + len(text), "page_num": page_num}
            )
            # 标题偏移: 按阅读顺序在该页文本里逐个定位 (用运行指针, 不重定位到同名前文)
            if config.PDF_SECTION_ENABLED and text:
                _hp = 0
                for _h in _page_headings(page):
                    _idx = text.find(_h, _hp)
                    if _idx < 0:
                        continue
                    _hp = _idx + 1
                    _heading_marks.append((_start + _idx, _h))

            # 提取图片（快速规则预筛）
            for img_idx, img in enumerate(page.get_images(full=True)):
                if len(result["images"]) >= max_images:
                    logger.warning(f"Image cap {max_images} reached, skipping rest of {pdf_name}")
                    return result
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    w, h = base_image.get("width", 0), base_image.get("height", 0)
                    ext = base_image["ext"]

                    if not _fast_image_filter(w, h, image_bytes, page_num, len(doc)):
                        continue

                    # 方案4: ML 轻量级分类器二次过滤 (14维特征, 无LLM)
                    if config.IMAGE_FILTER_STAGE1_ENABLED:
                        try:
                            from image_filter import get_image_filter

                            keep, score = get_image_filter().predict(image_bytes)
                            if not keep:
                                continue  # ML 判定为噪音, 跳过
                        except Exception:
                            pass  # ML 过滤失败 → 保守保留

                    image_filename = f"{safe_name}_p{page_num}_img{img_idx}.{ext}"
                    image_path = IMAGE_DIR / image_filename
                    image_path.write_bytes(image_bytes)

                    result["images"].append(
                        {
                            "image_path": str(image_path),
                            "page_num": page_num,
                            "ext": ext,
                            "width": w,
                            "height": h,
                        }
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to extract image {img_idx} on page {page_num}: {e}"
                    )
    finally:
        doc.close()

    # === 标题 → 章节区间 (PDF 没有显式层级, 只能靠字号线索; 命中才给, 不命中为空) ===
    if _heading_marks:
        _full = result["full_text"]
        for i, (pos, h) in enumerate(_heading_marks):
            end = _heading_marks[i + 1][0] if i + 1 < len(_heading_marks) else len(_full)
            result["sections"].append(
                {"heading": h, "level": 2, "start": pos, "end": end}
            )

    result["reordered_pages"] = reordered_pages
    result["tables_found"] = tables_found
    if reordered_pages:
        logger.debug(f"{pdf_name}: 双栏重排 {reordered_pages} 页")
    return result


def _fast_image_filter(
    width: int, height: int, image_bytes: bytes, page_num: int, page_count: int
) -> bool:
    """快速规则预筛：过滤明显噪声图片（Logo/图标/纯色图）。返回 True=可能有用。"""
    # 尺寸过滤：太小的图通常是 Logo/图标
    if width < config.IMAGE_MIN_PIXELS or height < config.IMAGE_MIN_PIXELS:
        return False
    if max(width, height) / max(1, min(width, height)) > config.IMAGE_MAX_ASPECT_RATIO:
        return False
    # 首页/尾页丢弃 (DeepSeek 审查 #1: 原 `>= page_count-1` 误丢倒数第二页, 尾页泄漏)
    # 开关见 IMAGE_FILTER_SKIP_FIRST_PAGE / IMAGE_FILTER_SKIP_LAST_PAGE
    if config.IMAGE_FILTER_SKIP_FIRST_PAGE and page_num <= 1:
        return False
    if config.IMAGE_FILTER_SKIP_LAST_PAGE and page_num >= page_count:
        return False
    # 颜色过滤：色彩太少的可能是单色图标
    try:
        import io

        from PIL import Image

        img = Image.open(io.BytesIO(image_bytes))
        colors = img.convert("RGB").getcolors(maxcolors=10)
        if colors and len(colors) < config.IMAGE_MIN_COLORS:
            return False
    except Exception:
        pass  # 解码失败就保留（保守策略）
    return True


def _get_splitter(chunk_size: int | None = None, chunk_overlap: int | None = None):
    """创建 LangChain RecursiveCharacterTextSplitter 分割器。
    分隔符优先级：中英文句号 → 换行 → 空格 → 字符级切分。
    准备2(技术) §1545 建议递归切分是多数项目的默认选择。
    v5.3: 修正注释——函数每次创建新实例, 非单例 (DeepSeek 审查 #2)。"""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    cs = chunk_size or CHUNK_SIZE
    if chunk_overlap is not None:
        co = chunk_overlap
    else:
        # 比例驱动 (config.CHUNK_OVERLAP_RATIO, 默认 15%)。
        # 修复: 原实现 min(200, max(50, cs*0.15)) 的上限 200 在 cs=4096 时先触发,
        # 真实重叠被压到 4.88% —— 与 docstring 声称的 15% 和通行建议(15-20%)都不符。
        # 上限保留但抬高到 CHUNK_OVERLAP_MAX, 只做安全兜底, 不绑定正常取值。
        from config import CHUNK_OVERLAP_MAX, CHUNK_OVERLAP_RATIO

        co = max(50, min(int(cs * CHUNK_OVERLAP_RATIO), CHUNK_OVERLAP_MAX, cs // 2))
    return RecursiveCharacterTextSplitter(
        separators=[
            "\n\n",
            "\n",
            "。",
            "！",
            "？",  # 段落 + 中文标点
            ".",
            "!",
            "?",  # 英文标点
            " ",
            "",  # 空格 → 字符级兜底
        ],
        chunk_size=cs,
        chunk_overlap=co,
        length_function=len,
        is_separator_regex=False,
    )


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
               chunk_overlap: int | None = None) -> list[str]:
    """使用 LangChain RecursiveCharacterTextSplitter 进行递归语义切分。
    按 准备2(技术) §364 建议：NLP 语义感知动态切分 + 15% token 重叠窗口。
    v5.3: 支持显式 chunk_overlap (DeepSeek 审查 #3)。"""
    if not text or not text.strip():
        return []
    splitter = _get_splitter(chunk_size, chunk_overlap)
    return splitter.split_text(text)


def chunk_text_paged(
    text: str,
    page_offsets: list[dict],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int | None = None,
    sections: list[dict] | None = None,
) -> list[dict]:
    """同 chunk_text 的切分结果, 但每块附带**真实页码**: [{"text": str, "page_num": int}]。

    背景: 此前 knowledge_base 写入的 page_number 恒为 0, 而 QA prompt 却要求模型
    标注 `[论文: 标题, 年份, pN]` —— 模型拿到无意义页码却必须给页码 → 引用页码全属编造。
    此处按块在 full_text 中的字符偏移反查所属页, 使溯源可核验。

    切分逻辑与 chunk_text 完全一致 (同一 splitter), 只是补元数据 → 检索行为不变。
    """
    chunks = chunk_text(text, chunk_size, chunk_overlap)
    if not chunks or not page_offsets:
        return [
            {"text": c, "page_num": 0, "section": _section_at(text, c, sections)}
            for c in chunks
        ]

    def _page_at(pos: int) -> int:
        for o in page_offsets:
            if o["start"] <= pos <= o["end"]:
                return o["page_num"]
        # 落在页间分隔符 (\n\n) 上 → 归到上一页
        prev = 0
        for o in page_offsets:
            if o["start"] <= pos:
                prev = o["page_num"]
            else:
                break
        return prev

    out: list[dict] = []
    pos = 0  # 运行搜索位置: 块间有重叠, 用上一块起点 +1 保证顺序定位不倒退
    for c in chunks:
        idx = text.find(c, pos)
        if idx == -1:  # 兜底: 从全文重新定位 (splitter 极少改字, 但避免丢页码)
            idx = text.find(c)
        if idx == -1:
            out.append({"text": c, "page_num": 0, "section": ""})
            continue
        pos = idx + 1
        out.append(
            {
                "text": c,
                "page_num": _page_at(idx),
                "section": _heading_at(sections, idx),
            }
        )
    return out


def _heading_at(sections: list[dict] | None, pos: int) -> str:
    """按字符偏移取所属章节标题 (无 sections 或未命中 → 空)。

    边界: 位置**早于第一个标题**时, 此前返回空 —— 结果论文第一块 (标题/摘要区)
    没有章节标签。现归到第一个标题下 (那正是它所属的开篇部分)。
    """
    if not sections:
        return ""
    if pos < sections[0]["start"]:
        return sections[0]["heading"]
    for s in sections:
        if s["start"] <= pos <= s["end"]:
            return s["heading"]
    prev = ""
    for s in sections:
        if s["start"] <= pos:
            prev = s["heading"]
        else:
            break
    return prev


def _section_at(text: str, chunk: str, sections: list[dict] | None) -> str:
    """无 page_offsets 时的兜底: 现算块位置再取章节。"""
    if not sections:
        return ""
    idx = text.find(chunk)
    return _heading_at(sections, idx) if idx >= 0 else ""


