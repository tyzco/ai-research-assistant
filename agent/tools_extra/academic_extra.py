"""学术核心工具插件 — 查重 + 期刊/会议等级 (academic_extra)。

放在 agent/tools_extra/ 下, 注册表启动时自动发现注册。
能力:
  1. check_plagiarism: 文本与私有知识库做字符三元组包含度查重, 返回相似度 + 来源
  2. lookup_venue_rank: 查期刊影响因子(近似) / 会议 CCF 等级
依赖: hybrid_retriever + lancedb (均已在)。

边界: 查重仅对比私有 KB (用户自供论文), 不调用外部盗版库; 影响因子为近似值, 以官方为准。
"""
import asyncio
import logging
import re

logger = logging.getLogger(__name__)


# ============ 工具 1: 查重 ============

async def _best_kb() -> str:
    """选内容最丰富的用户知识库 (papers_* 且非 eval)。"""
    import lancedb

    from config import LANCEDB_DIR

    db = lancedb.connect(str(LANCEDB_DIR))
    kbs = [
        t for t in db.table_names()
        if t.startswith("papers_") and "eval" not in t.lower()
    ]
    if not kbs:
        return ""
    scored = []
    for t in kbs:
        try:
            scored.append((db.open_table(t).count_rows(), t))
        except Exception:
            pass
    scored.sort(key=lambda x: -x[0])
    return scored[0][1] if scored else ""


def _char_ngrams(s: str, n: int = 3) -> set[str]:
    """字符 n-gram (中文无需分词, 英文混排也可用)。"""
    s = re.sub(r"\s+", "", s.lower())
    return {s[i:i + n] for i in range(len(s) - n + 1)}


async def _check_plagiarism(text: str, table_name: str = "") -> dict:
    """查重: 文本 vs 私有知识库, 字符三元组包含度 (0-1)。"""
    text = (text or "").strip()
    if len(text) < 20:
        return {"error": "文本太短，无法查重（至少 20 字符）"}
    if not table_name:
        table_name = await _best_kb()
    if not table_name:
        return {"error": "没有可对比的知识库。请先构建知识库（上传/下载 OA 论文）再查重，或传入 table_name"}

    from hybrid_retriever import AdaptiveHybridRetriever

    try:
        hr = AdaptiveHybridRetriever(table_name)
        results, alpha = hr.search(text, top_k=8)
    except Exception as e:
        return {"error": f"知识库检索失败: {e}"}

    # 按段落拆分, 逐段算包含度 (更精准定位重合片段)
    paras = [
        p.strip() for p in re.split(r"\n+|(?<=[。！？.!?])\s*", text)
        if len(p.strip()) >= 10
    ] or [text]
    text_ng = _char_ngrams(text)

    matches = []
    for r in results:
        chunk_ng = _char_ngrams(r.get("text") or "")
        if not chunk_ng:
            continue
        # 整体包含度 (文本 n-gram 有多少出现在该 chunk)
        whole = len(text_ng & chunk_ng) / max(len(text_ng), 1)
        # 最重合的单段
        best_sim, best_para = 0.0, ""
        for p in paras:
            pn = _char_ngrams(p)
            if not pn:
                continue
            sim = len(pn & chunk_ng) / max(len(pn), 1)
            if sim > best_sim:
                best_sim, best_para = sim, p
        sim = max(whole, best_sim)
        if sim >= 0.30:  # 只报有明显重合的
            matches.append({
                "title": r.get("title", "N/A"),
                "year": r.get("year"),
                "similarity": round(sim, 3),
                "source_excerpt": (r.get("text") or "")[:200],
                "matched_passage": best_para[:200],
            })

    matches.sort(key=lambda m: -m["similarity"])
    matches = matches[:5]
    overall = matches[0]["similarity"] if matches else 0.0
    if overall >= 0.85:
        verdict, msg = "high", "高度相似：大段文本与已有文献重合，建议改写并用正确引用标注"
    elif overall >= 0.60:
        verdict, msg = "medium", "中度相似：存在明显重合段落，需改写降重后使用"
    else:
        verdict, msg = "low", "低相似：未发现明显重合，原创性较高"

    return {
        "verdict": verdict,
        "overall_similarity": round(overall, 3),
        "message": msg,
        "matches": matches,
        "compared_kb": table_name,
        "tip": "相似度为字符三元组包含度（0-1），>0.85 高度重合、0.60-0.85 需改写。只对比私有知识库，不查外部库。",
    }


# ============ 工具 2: 期刊/会议等级 ============

# 常见 CS/AI 期刊会议 (CCF 等级 + 期刊近似 IF)。IF 为近年近似值, 以出版社官方为准。
VENUE_TABLE: list[dict] = [
    # ---- CCF A 会议 ----
    {"venue": "NeurIPS", "type": "conference", "rank": "CCF-A"},
    {"venue": "ICML", "type": "conference", "rank": "CCF-A"},
    {"venue": "ICLR", "type": "conference", "rank": "CCF-A"},
    {"venue": "CVPR", "type": "conference", "rank": "CCF-A"},
    {"venue": "ICCV", "type": "conference", "rank": "CCF-A"},
    {"venue": "ACL", "type": "conference", "rank": "CCF-A"},
    {"venue": "AAAI", "type": "conference", "rank": "CCF-A"},
    {"venue": "IJCAI", "type": "conference", "rank": "CCF-A"},
    {"venue": "KDD", "type": "conference", "rank": "CCF-A"},
    {"venue": "SIGIR", "type": "conference", "rank": "CCF-A"},
    {"venue": "WWW", "type": "conference", "rank": "CCF-A"},
    {"venue": "ICDE", "type": "conference", "rank": "CCF-A"},
    {"venue": "INFOCOM", "type": "conference", "rank": "CCF-A"},
    # ---- CCF B 会议 ----
    {"venue": "ECCV", "type": "conference", "rank": "CCF-B"},
    {"venue": "EMNLP", "type": "conference", "rank": "CCF-B"},
    {"venue": "NAACL", "type": "conference", "rank": "CCF-B"},
    {"venue": "COLING", "type": "conference", "rank": "CCF-B"},
    {"venue": "ICDM", "type": "conference", "rank": "CCF-B"},
    {"venue": "COLT", "type": "conference", "rank": "CCF-B"},
    {"venue": "WSDM", "type": "conference", "rank": "CCF-B"},
    {"venue": "RecSys", "type": "conference", "rank": "CCF-B"},
    {"venue": "CIKM", "type": "conference", "rank": "CCF-B"},
    {"venue": "ESEC/FSE", "type": "conference", "rank": "CCF-B"},
    {"venue": "USENIX ATC", "type": "conference", "rank": "CCF-B"},
    # ---- CCF C 会议 ----
    {"venue": "IJCNN", "type": "conference", "rank": "CCF-C"},
    {"venue": "ICANN", "type": "conference", "rank": "CCF-C"},
    {"venue": "ACCV", "type": "conference", "rank": "CCF-C"},
    {"venue": "PRICAI", "type": "conference", "rank": "CCF-C"},
    # ---- 综合/期刊 ----
    {"venue": "Nature", "type": "journal", "rank": "综合顶刊", "if": "~50"},
    {"venue": "Science", "type": "journal", "rank": "综合顶刊", "if": "~45"},
    {"venue": "IEEE TPAMI", "type": "journal", "rank": "CCF-A", "if": "~24"},
    {"venue": "JMLR", "type": "journal", "rank": "CCF-A", "if": "~5"},
    {"venue": "Artificial Intelligence", "type": "journal", "rank": "CCF-A", "if": "~10"},
    {"venue": "IJCV", "type": "journal", "rank": "CCF-A", "if": "~12"},
    {"venue": "IEEE TIP", "type": "journal", "rank": "CCF-A", "if": "~10"},
    {"venue": "IEEE TKDE", "type": "journal", "rank": "CCF-A", "if": "~9"},
    {"venue": "IEEE TNNLS", "type": "journal", "rank": "CCF-B", "if": "~10"},
    {"venue": "Neural Networks", "type": "journal", "rank": "CCF-B", "if": "~7"},
    {"venue": "Pattern Recognition", "type": "journal", "rank": "CCF-B", "if": "~7"},
    {"venue": "Information Sciences", "type": "journal", "rank": "CCF-B", "if": "~8"},
    {"venue": "Neurocomputing", "type": "journal", "rank": "CCF-C", "if": "~5"},
    {"venue": "Applied Intelligence", "type": "journal", "rank": "CCF-C", "if": "~5"},
    {"venue": "Expert Systems with Applications", "type": "journal", "rank": "CCF-C", "if": "~8"},
    {"venue": "Knowledge-Based Systems", "type": "journal", "rank": "CCF-C", "if": "~8"},
]


def _fuzzy_venue_candidates(low: str) -> list[dict]:
    """保守模糊匹配, 防短名误配 (如 "science" 匹配 "Journal of Applied Polymer Science")。
    只接受: ① query = venue + 少量后缀 (cvpr 2025); ② venue 以 query 开头且 query ≥4 字符。"""
    cands = []
    for e in VENUE_TABLE:
        ev = e["venue"].lower()
        if ev == low:
            return [e]
        if low.startswith(ev) and len(ev) >= 4:  # query = venue + 后缀
            extra = low[len(ev):].strip(" 0-9()/·,.")
            if len(extra) <= 12:
                cands.append((len(ev), e))
        elif ev.startswith(low) and len(low) >= 4:  # venue 以 query 开头
            cands.append((len(ev), e))
    cands.sort(key=lambda x: -x[0])
    return [e for _, e in cands]


async def _lookup_venue_rank(venue: str) -> dict:
    """查期刊/会议等级: 期刊返回近似影响因子 + CCF, 会议返回 CCF 等级。"""
    name = (venue or "").strip()
    if not name:
        return {"error": "需要 venue 参数（期刊或会议名）"}
    low = name.lower()
    exact = [e for e in VENUE_TABLE if e["venue"].lower() == low]
    if exact:
        return _venue_result(exact[0])
    fuzzy = _fuzzy_venue_candidates(low)
    if fuzzy:
        return _venue_result(fuzzy[0], fuzzy=True)
    return {
        "venue": name,
        "found": False,
        "message": "未收录该期刊/会议。当前表仅覆盖常见 CS/AI 期刊会议；可提供官方信息后补充。",
    }


def _venue_result(e: dict, fuzzy: bool = False) -> dict:
    out = {
        "venue": e["venue"],
        "type": e["type"],
        "rank": e["rank"],
        "found": True,
    }
    if e.get("if"):
        out["impact_factor_approx"] = e["if"]
    if fuzzy:
        out["message"] = f"模糊匹配到「{e['venue']}」，如需更精确请用全称"
    return out


from agent.tool_registry import Tool  # noqa: E402  (必须放文件尾, 避免循环导入)

TOOLS = [
    Tool(
        "check_plagiarism",
        "查重: 输入文本, 与私有知识库对比, 返回相似度(0-1) + 最重合片段 + 来源论文。"
        "写论文初稿前调用, 高相似(>0.85)或中相似(0.60-0.85)需改写降重。",
        {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "要查重的文本/段落/初稿"},
                "table_name": {"type": "string", "description": "知识库表名（可选，默认自动选内容最丰富的）"},
            },
            "required": ["text"],
        },
        _check_plagiarism,
    ),
    Tool(
        "lookup_venue_rank",
        "查期刊/会议等级: 期刊返回近似影响因子+CCF等级, 会议返回CCF等级。投稿/选题决策用。",
        {
            "type": "object",
            "properties": {
                "venue": {"type": "string", "description": "期刊或会议名，如 IEEE TPAMI / NeurIPS"}
            },
            "required": ["venue"],
        },
        _lookup_venue_rank,
    ),
]
