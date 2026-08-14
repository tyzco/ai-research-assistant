"""数据文件工具插件 — Excel 导出 + CSV 读取 (dataio)。

放在 agent/tools_extra/ 下, 注册表启动时自动发现注册。
能力:
  1. export_references_xlsx: 参考文献列表 → Excel (写入 data/exports/, 经现有 /export/download 下载)
  2. read_csv: 读取 CSV (服务端路径或内联内容) → 列名/行数/预览 + 论文元数据列自动识别
依赖: openpyxl (已装) + csv (stdlib)。

边界: 仅处理用户上传/生成的数据文件, 不触碰付费/盗版内容。
"""
import asyncio
import csv
import io
import logging
import re
import urllib.parse

logger = logging.getLogger(__name__)

_CSV_ENCODINGS = ("utf-8-sig", "utf-8", "gb18030", "gbk")

# 论文元数据列别名 → 标准列名 (read_csv 自动识别用)
_PAPER_COLUMNS = {
    "title": ("标题", "论文标题", "题目", "title", "name"),
    "authors": ("作者", "authors", "author"),
    "year": ("年份", "year", "时间"),
    "venue": ("期刊", "会议", "venue", "journal", "conference"),
    "doi": ("doi", "url"),
}


def _decode(raw: bytes) -> str:
    for enc in _CSV_ENCODINGS:
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def _detect_columns(header: list[str]) -> dict:
    """按别名识别论文元数据列, 返回 {标准列名: 表头原文}。"""
    mapped: dict[str, str] = {}
    for col in header:
        c = col.strip().lower()
        for key, aliases in _PAPER_COLUMNS.items():
            if key not in mapped and c in aliases:
                mapped[key] = col
                break
    return mapped


async def _read_csv(filepath: str = "", content: str = "") -> dict:
    """读取 CSV 数据: 返回列名/行数/预览, 并自动识别论文元数据列。"""
    try:
        if content:
            text = content
        elif filepath:
            import os

            if not os.path.exists(filepath):
                return {"error": f"文件不存在: {filepath}"}
            with open(filepath, "rb") as f:
                text = _decode(f.read())
        else:
            return {"error": "需要 filepath (上传的数据文件) 或 content (内联 CSV 文本) 参数"}

        rows = [r for r in csv.reader(io.StringIO(text)) if any(c.strip() for c in r)]
        if not rows:
            return {"error": "CSV 为空"}
        header = [c.strip() for c in rows[0]]
        data = rows[1:]

        mapped = _detect_columns(header)
        preview = [dict(zip(header, r)) for r in data[:8]]
        tip = (
            "已识别论文元数据列: " + ", ".join(f"{k}={v}" for k, v in mapped.items())
            if mapped
            else "未识别到论文元数据列 (title/doi/year/author 等)"
        )
        return {
            "columns": header,
            "rows": len(data),
            "detected_paper_columns": mapped,
            "preview": preview,
            "tip": tip,
        }
    except Exception as e:
        return {"error": f"CSV 解析失败: {e}"}


async def _export_references_xlsx(references: list, filename: str = "") -> dict:
    """参考文献列表 → Excel 文件 (data/exports/, 经 /export/download 下载)。"""
    try:
        import openpyxl
        from openpyxl.utils import get_column_letter

        from exporter import EXPORT_DIR

        refs = [r for r in (references or []) if isinstance(r, dict)]
        cols = ["title", "authors", "year", "venue", "doi"]

        def _cell(v):
            if isinstance(v, list):
                return "; ".join(str(x) for x in v)
            return v or ""

        wb = openpyxl.Workbook()
        ws = wb.active
        ws.title = "references"
        ws.append([c.capitalize() for c in cols])
        for r in refs:
            ws.append([_cell(r.get(c)) for c in cols])
        for i in range(1, len(cols) + 1):
            ws.column_dimensions[get_column_letter(i)].width = 42

        safe = re.sub(r"[^\w\-]+", "_", (filename or "references").strip()) or "references"
        path = EXPORT_DIR / f"{safe}.xlsx"
        # 防覆盖: 已存在 → 追加时间戳 (Date 不可用, 用递增后缀)
        n = 2
        while path.exists():
            path = EXPORT_DIR / f"{safe}_{n}.xlsx"
            n += 1
        wb.save(path)

        url = "/file/download?filepath=" + urllib.parse.quote(str(path))
        return {
            "ok": True,
            "rows": len(refs),
            "filepath": str(path),
            "download_url": url,
            "markdown_link": f"[下载 Excel]({url})",
        }
    except Exception as e:
        return {"error": f"Excel 导出失败: {e}"}


from agent.tool_registry import Tool  # noqa: E402  (必须放文件尾, 避免循环导入)

TOOLS = [
    Tool(
        "export_references_xlsx",
        "把参考文献列表导出为 Excel 表格 (data/exports/)。"
        "输入 references 数组 (每项含 title/authors/year/venue/doi), 返回下载链接。",
        {
            "type": "object",
            "properties": {
                "references": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "参考文献列表, 每项含 title/authors/year/venue/doi",
                },
                "filename": {"type": "string", "description": "文件名 (可选, 默认 references)"},
            },
            "required": ["references"],
        },
        _export_references_xlsx,
    ),
    Tool(
        "read_csv",
        "读取 CSV 数据文件或内联 CSV 文本: 返回列名/行数/预览, 并自动识别论文元数据列"
        "(title/doi/year/author 等)。配合 /upload/data 上传的数据文件使用。",
        {
            "type": "object",
            "properties": {
                "filepath": {
                    "type": "string",
                    "description": "服务端数据文件路径 (经 /upload/data 上传后返回的 filepath)",
                },
                "content": {"type": "string", "description": "内联 CSV 文本 (二选一, 无需上传)"},
            },
        },
        _read_csv,
    ),
]
