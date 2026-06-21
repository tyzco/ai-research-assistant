"""论文处理模块：PDF 文本提取 + 图片提取 + 分块。"""

import logging
from pathlib import Path

import fitz  # PyMuPDF

from config import CHUNK_SIZE, IMAGE_DIR, PAPER_CACHE_DIR

logger = logging.getLogger(__name__)

PAPER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)


def parse_pdf(pdf_bytes: bytes, pdf_name: str) -> dict:
    """
    解析单个 PDF：提取全文 + 图片。

    返回 {
        "full_text": str,
        "pages": [{"page_num": int, "text": str}],
        "images": [
            {"image_path": str, "page_num": int, "bbox": tuple}
        ]
    }
    """
    result = {"full_text": "", "pages": [], "images": []}

    safe_name = Path(pdf_name).stem.replace(" ", "_")[:40]

    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page_idx, page in enumerate(doc):
            page_num = page_idx + 1
            text = page.get_text().strip()
            result["pages"].append({"page_num": page_num, "text": text})
            result["full_text"] += text + "\n\n"

            # 提取图片（快速规则预筛）
            for img_idx, img in enumerate(page.get_images(full=True)):
                try:
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    w, h = base_image.get("width", 0), base_image.get("height", 0)
                    ext = base_image["ext"]

                    if not _fast_image_filter(w, h, image_bytes, page_num, len(doc)):
                        continue

                    image_filename = f"{safe_name}_p{page_num}_img{img_idx}.{ext}"
                    image_path = IMAGE_DIR / image_filename
                    image_path.write_bytes(image_bytes)

                    result["images"].append(
                        {
                            "image_path": str(image_path),
                            "page_num": page_num,
                            "ext": ext,
                            "width": w, "height": h,
                        }
                    )
                except Exception as e:
                    logger.warning(
                        f"Failed to extract image {img_idx} on page {page_num}: {e}"
                    )

    return result



def _fast_image_filter(width: int, height: int, image_bytes: bytes, page_num: int, page_count: int) -> bool:
    """快速规则预筛：过滤明显噪声图片（Logo/图标/纯色图）。返回 True=可能有用。"""
    # 尺寸过滤：太小的图通常是 Logo/图标
    if width < 100 or height < 100:
        return False
    if max(width, height) / max(1, min(width, height)) > 6.0:
        return False
    if page_num <= 1 or page_num >= page_count - 1:
        return False
    # 颜色过滤：少于 3 种颜色的可能是单色图标
    try:
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(image_bytes))
        colors = img.convert("RGB").getcolors(maxcolors=10)
        if colors and len(colors) < 5:
            return False
    except Exception:
        pass  # 解码失败就保留（保守策略）
    return True


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    """简单滑窗分块，每块不超过 chunk_size 字符，100 字符重叠。"""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - 100
    return chunks


def semantic_chunk_text(
    text: str, max_chars: int = 1024, overlap: int = 2
) -> list[str]:
    """按句子边界切分，合并到接近 max_chars，带 overlap 句重叠。
    更适合学术论文——不会把一句话切成两半。"""
    import re

    sentences = re.split(r"(?<=[。！？.!?])\s*", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return []

    chunks = []
    i = 0
    while i < len(sentences):
        chunk = sentences[i]
        j = i + 1
        while j < len(sentences) and len(chunk) + len(sentences[j]) < max_chars:
            chunk += " " + sentences[j]
            j += 1
        if len(chunk) > 50:  # 太短的不存
            chunks.append(chunk)
        i = max(i + 1, j - overlap)
    return chunks if chunks else [text[:max_chars]]
