"""论文处理模块：PDF 文本提取 + 图片提取 + 分块。"""

import logging
from pathlib import Path

import fitz  # PyMuPDF

import config
from config import CHUNK_SIZE, IMAGE_DIR, PAPER_CACHE_DIR

logger = logging.getLogger(__name__)

PAPER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)


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
    result = {"full_text": "", "pages": [], "page_offsets": [], "images": []}

    safe_name = Path(pdf_name).stem.replace(" ", "_")[:40]

    # 图片上限 (DeepSeek 审查 #4: 防止大 PDF 内存/磁盘激增)
    max_images = getattr(config, "MAX_IMAGES_PER_PDF", 200)

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
            except Exception as e:
                logger.warning(f"Page {page_num} text extraction failed: {e}")
                text = ""
            result["pages"].append({"page_num": page_num, "text": text})
            _start = len(result["full_text"])
            result["full_text"] += text + "\n\n"
            result["page_offsets"].append(
                {"start": _start, "end": _start + len(text), "page_num": page_num}
            )

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
    co = (
        chunk_overlap
        if chunk_overlap is not None
        else min(200, max(50, int(cs * 0.15)))
    )  # 15% overlap, capped
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
) -> list[dict]:
    """同 chunk_text 的切分结果, 但每块附带**真实页码**: [{"text": str, "page_num": int}]。

    背景: 此前 knowledge_base 写入的 page_number 恒为 0, 而 QA prompt 却要求模型
    标注 `[论文: 标题, 年份, pN]` —— 模型拿到无意义页码却必须给页码 → 引用页码全属编造。
    此处按块在 full_text 中的字符偏移反查所属页, 使溯源可核验。

    切分逻辑与 chunk_text 完全一致 (同一 splitter), 只是补元数据 → 检索行为不变。
    """
    chunks = chunk_text(text, chunk_size, chunk_overlap)
    if not chunks or not page_offsets:
        return [{"text": c, "page_num": 0} for c in chunks]

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
            out.append({"text": c, "page_num": 0})
            continue
        pos = idx + 1
        out.append({"text": c, "page_num": _page_at(idx)})
    return out


