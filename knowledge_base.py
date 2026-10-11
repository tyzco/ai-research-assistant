"""LanceDB 知识库模块：双层向量索引（全文深度层 + 摘要广度层）。
嵌入：本地 BAAI/bge-small-zh，CPU 推理，零成本。
"""

import logging
import re

import lancedb
import pyarrow as pa

from config import IMAGE_DIR, LANCEDB_DIR, LOCAL_EMBEDDING_DIM, RETRIEVAL_TOP_K
from models import PaperMeta

logger = logging.getLogger(__name__)

LANCEDB_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# ---- 嵌入模型 ----
# 嵌入模型已迁移到 embedder.py，此处保留向后兼容别名
from embedder import _embed_texts_sync

# ---- LanceDB 表管理 ----


def _get_table(db, table_name: str):
    try:
        t = db.open_table(table_name)
        # 存量库迁移 (幂等): 老表无 section 列 → 补上, 否则带 section 的行写不进去
        if "section" not in {f.name for f in t.schema}:
            try:
                t.add_columns({"section": "''"})
                logger.info(f"Migrated {table_name}: added column 'section'")
            except Exception as e:
                logger.warning(f"Add column 'section' on {table_name} failed: {e}")
        # 版本管理列 (存量库补列; 老行 status 留空 → 读取时按 active 处理, 见 ACTIVE_FILTER)
        _names = {f.name for f in t.schema}
        _add = {}
        if "doc_key" not in _names:
            _add["doc_key"] = "''"
        if "status" not in _names:
            _add["status"] = "'active'"
        if "version" not in _names:
            _add["version"] = "1"
        if "text_fts" not in _names:
            _add["text_fts"] = "''"
        if _add:
            try:
                t.add_columns(_add)
                logger.info(f"Migrated {table_name}: added {sorted(_add)}")
            except Exception as e:
                logger.warning(f"Migrate version columns on {table_name} failed: {e}")
        return t
    except ValueError as e:
        # LanceDB 表不存在抛 ValueError (DeepSeek 审查 #1: 精确捕获, 其他异常向上抛避免数据丢失)
        if "not found" not in str(e).lower():
            raise
        dim = LOCAL_EMBEDDING_DIM
        return db.create_table(
            table_name,
            schema=pa.schema(
                [
                    pa.field("chunk_id", pa.string()),
                    pa.field("paper_id", pa.string()),
                    pa.field("title", pa.string()),
                    pa.field("authors", pa.string()),
                    pa.field("year", pa.int32()),
                    pa.field("abstract", pa.string()),
                    pa.field("doi", pa.string()),
                    pa.field("is_fulltext", pa.bool_()),
                    pa.field("is_image", pa.bool_()),
                    pa.field("image_path", pa.string()),
                    pa.field("page_number", pa.int32()),
                    pa.field("chunk_index", pa.int32()),
                    # 章节定位: DOCX 有显式标题层级可绑定; PDF 无 (留空)
                    pa.field("section", pa.string()),
                    # === 版本与失效管理 ===
                    # doc_key: 同一份文档跨版本的稳定标识 (规范手册改版时 title 可能不变)
                    # status : active(可检索) | superseded(已被新版取代) | archived(人工废弃)
                    #          —— 检索阶段过滤掉非 active, 避免模型引用作废条文/旧固件参数
                    # version: 同一 doc_key 的版本号, 每次改版 +1
                    pa.field("doc_key", pa.string()),
                    pa.field("status", pa.string()),
                    pa.field("version", pa.int32()),
                    pa.field("text", pa.string()),
                    # jieba 分词后的副本, **只用于 FTS 索引**。
                    # 为什么不直接索引 text: LanceDB 默认分词器是 simple/English,
                    # 无空格中文被当成**一个巨大 token** —— 实测 "曲轴箱温度报警阈值是九十五
                    # 摄氏度" 搜 "曲轴箱温度" 命中 0。即 BM25 支路对中文一直是死的,
                    # "混合检索"实际退化成纯向量。分词后索引可恢复 (实测 4/4 命中)。
                    pa.field("text_fts", pa.string()),
                    pa.field("vector", pa.list_(pa.float32(), dim)),
                ]
            ),
        )


# FTS 停用词: 中文虚词/疑问词。LanceDB 的 remove_stop_words 用的是 English 词表,
# 对中文无效 —— 不过滤的话 "什么是ArcFace算法" 会被切成 什么/是/算法, 其中
# 什么/是 把所有含这些字的文档都捞上来 (实测 BM25 每题都返回同一批无关期刊页,
# 修复后 MRR 反而从 0.667 掉到 0.625)。这里在分词阶段就滤掉。
_FTS_STOPWORDS = {
    "的", "了", "是", "在", "和", "与", "及", "或", "有", "为", "对", "从", "到",
    "这", "那", "其", "被", "把", "给", "让", "使", "等", "所", "以", "而", "则",
    "什么", "哪些", "哪个", "怎么", "怎样", "如何", "为什么", "是否", "可以", "一个",
    "我们", "他们", "它们", "这个", "那个", "这些", "那些", "以及", "并且", "但是",
    "因为", "所以", "如果", "就是", "还是", "或者", "然后", "通过", "进行", "具有",
    "the", "a", "an", "of", "is", "are", "and", "or", "to", "in", "for", "on",
    "what", "which", "how", "why", "does", "do", "with",
}


def _fts_tokens(text: str) -> list[str]:
    """分词并滤掉停用词/单字 (留 ASCII 词和长度>=2 的中文词)。"""
    try:
        import jieba

        toks = list(jieba.cut(text or ""))
    except Exception:
        return [t for t in (text or "").split() if t]
    out = []
    for t in toks:
        t = t.strip()
        if not t or t in _FTS_STOPWORDS:
            continue
        # 单字中文噪声大 (的/是 已滤, 但"有""对"之类仍会误配); ASCII 单字符保留 (如 "C")
        if len(t) == 1 and not t.isascii():
            continue
        if not any(ch.isalnum() for ch in t):
            continue
        out.append(t)
    return out


def _segment_for_fts(text: str) -> str:
    """把文本切成空格分隔的词, 供 FTS 索引。

    LanceDB 这个版本的可用分词器只有 simple/raw/whitespace (无 jieba/ngram),
    默认 simple 对无空格中文等于不切 —— 所以**我们自己切好再入库**。
    jieba 不可用时原样返回 (退化为原行为, 不会更差)。
    """
    t = text or ""
    if not t:
        return ""
    return " ".join(_fts_tokens(t))[:20000]


def _doc_key(p) -> str:
    """同一份文档跨版本的稳定标识。

    为什么不用 paper_id: paper_id 是**内容哈希** —— 文档改版后哈希就变了, 会变成
    一份"全新文档", 旧版本仍留在库里且仍是 active, 检索时会同时命中新旧两版。
    doc_key 取标题 (去空白/标点), 让"同一份手册的不同版本"能被识别为同一份文档。
    """
    import re as _re

    raw = (p.title or "").strip() or (p.paper_id or "")
    return _re.sub(r"[\s\-_/\\]+", "", raw)[:120]


# ---- 知识库构建 ----


async def build_knowledge_base(
    papers: list[PaperMeta],
    fulltext_chunks: dict[str, list[str]],
    topic_id: str,
    progress_callback=None,
) -> str:
    """
    构建双层知识库：
    a. 为所有论文各插入一条摘要记录（is_fulltext=False）
    b. 为每篇上传全文的论文插入多个文本块（is_fulltext=True）

    返回 LanceDB 表名。
    """
    table_name = f"papers_{topic_id}"
    db = lancedb.connect(str(LANCEDB_DIR))
    table = _get_table(db, table_name)
    n = len(papers)
    _versions: dict[str, int] = {}   # paper_id → version
    _doc_keys: dict[str, str] = {}   # paper_id → doc_key

    # --- 版本管理: 同一 doc_key 的旧版本标记为 superseded (而非删除) ---
    # 保留旧版是为了可追溯 (工业场景常需要"上一版规范怎么说"), 但检索阶段会过滤掉,
    # 所以不会拿作废条文/旧固件参数去回答。
    for p in papers:
        _key = _doc_key(p)
        if not re.fullmatch(r"[A-Za-z0-9_\-\u4e00-\u9fff]{1,128}", _key):
            continue
        _prev = 0
        try:
            old_rows = (
                table.search()
                .where(f"doc_key = '{_key}'")
                .limit(1000)
                .to_list()
            )
            for r in old_rows:
                _prev = max(_prev, int(r.get("version") or 0))
            if old_rows:
                # **标记**为失效而非删除 (与"废弃旧版规范打上失效标签"一致):
                # 保留旧版可追溯 ("上一版规范怎么说"), 检索阶段靠 status 过滤掉,
                # 所以不会拿作废条文/旧固件参数去回答。真正清理交给 prune_superseded()。
                table.update(
                    where=f"doc_key = '{_key}' AND paper_id != '{p.paper_id}'",
                    values={"status": "superseded"},
                )
                logger.info(
                    f"Marked {len(old_rows)} row(s) of doc_key={_key} as superseded "
                    f"(prev version {_prev})"
                )
        except Exception as e:
            logger.warning(f"supersede old version failed for {_key}: {e}")
        _versions[p.paper_id] = _prev + 1
        _doc_keys[p.paper_id] = _key
    # 同一 paper_id 的旧行 (重复上传同一份文件) 直接替换
    # --- 幂等: 先清掉这批论文的旧行 ---
    # 重复上传同一份文件时, 若只做 add 会得到**重复块** (实测 21 → 42),
    # 检索时 top-k 被重复内容占满, 实际召回被稀释。先删后插 → 重复上传变成替换。
    _cleared = 0
    for p in papers:
        pid = p.paper_id or ""
        # paper_id 由我们生成 (内容哈希/uuid hex), 但仍做白名单校验防止拼进 where 注入
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", pid):
            continue
        try:
            table.delete(f"paper_id = '{pid}'")
            _cleared += 1
        except Exception as e:
            logger.warning(f"clear old rows for {pid} failed: {e}")
    if _cleared:
        logger.info(f"Cleared existing rows for {_cleared} paper(s) before re-index")

    # --- 摘要广度层 ---
    if progress_callback:
        await progress_callback(0, n, "indexing_abstracts")
    abstract_texts = [p.abstract for p in papers]
    if abstract_texts:
        vectors = _embed_texts_sync(abstract_texts)
        rows = []
        for p, vec in zip(papers, vectors):
            rows.append(
                {
                    "chunk_id": f"{p.paper_id}_abstract",
                    "paper_id": p.paper_id,
                    "title": p.title,
                    "authors": p.authors,
                    "year": p.year or 0,
                    "abstract": p.abstract,
                    "doi": p.doi or "",
                    "is_fulltext": False,
                    "is_image": False,
                    "image_path": "",
                    "page_number": 0,
                    "chunk_index": 0,
                    "section": "",
                    "doc_key": _doc_keys.get(p.paper_id, ""),
                    "status": "active",
                    "version": _versions.get(p.paper_id, 1),
                    "text": p.abstract,
                    "text_fts": _segment_for_fts(p.abstract),
                    "vector": [float(v) for v in vec],
                }
            )
        table.add(rows)
        if progress_callback:
            await progress_callback(n, n, "indexing_abstracts")

    # --- 全文深度层 ---
    fulltext_papers = [p for p in papers if p.paper_id in fulltext_chunks]
    total_chunks = sum(len(chunks) for chunks in fulltext_chunks.values())
    if progress_callback:
        await progress_callback(0, total_chunks, "indexing_fulltext")

    indexed = 0
    batch_texts: list[str] = []
    batch_rows: list[dict] = []

    for p in fulltext_papers:
        for chunk_idx, _entry in enumerate(fulltext_chunks[p.paper_id]):
            # 兼容两种形态: str (旧调用方, 无页码) / {"text","page_num"[,"section"]}
            if isinstance(_entry, dict):
                chunk_text = _entry.get("text", "")
                page_num = int(_entry.get("page_num") or 0)
                section = _entry.get("section") or ""
            else:
                chunk_text, page_num, section = _entry, 0, ""
            batch_texts.append(chunk_text)
            batch_rows.append(
                {
                    "chunk_id": f"{p.paper_id}_chunk_{chunk_idx}",
                    "paper_id": p.paper_id,
                    "title": p.title,
                    "authors": p.authors,
                    "year": p.year or 0,
                    "abstract": p.abstract,
                    "doi": p.doi or "",
                    "is_fulltext": True,
                    "is_image": False,
                    "image_path": "",
                    "page_number": page_num,  # 真实页码 (此前恒 0 → 引用页码全是编的)
                    "chunk_index": chunk_idx + 1,  # 0=abstract, 1+=fulltext
                    "section": section,  # 所属章节 (DOCX 可绑定; PDF 为空)
                    "doc_key": _doc_keys.get(p.paper_id, ""),
                    "status": "active",
                    "version": _versions.get(p.paper_id, 1),
                    "text": chunk_text,
                    "text_fts": _segment_for_fts(chunk_text),
                    "vector": [],
                }
            )
            indexed += 1

            if len(batch_texts) >= 32:
                vecs = _embed_texts_sync(batch_texts)
                for r, v in zip(batch_rows, vecs):
                    r["vector"] = [float(x) for x in v]
                table.add(batch_rows)
                if progress_callback:
                    await progress_callback(indexed, total_chunks, "indexing_fulltext")
                batch_texts, batch_rows = [], []

    if batch_texts:
        vecs = _embed_texts_sync(batch_texts)
        for r, v in zip(batch_rows, vecs):
            r["vector"] = [float(x) for x in v]
        table.add(batch_rows)
        if progress_callback:
            await progress_callback(indexed, total_chunks, "indexing_fulltext")

    logger.info(
        f"Knowledge base built: {n} abstracts + {total_chunks} fulltext chunks in {table_name}"
    )
    # 创建全文索引——LanceDB 的 BM25/FTS 依赖此索引，缺失则全部 BM25 查询静默失败
    try:
        # 必须**重新打开**表再建索引: 本函数前面做过 delete + add, 而最初打开的
        # `table` 句柄仍指向删除前的快照 —— 在它上面建 FTS 会漏掉刚写入的新行。
        # 实测: 版本替换后 BM25 搜不到新内容 (查"螺旋桨桨距校准"命中 0 条),
        # 混合检索静默退化成纯向量。
        fresh = _get_table(db, table_name)
        # 优先索引 jieba 分词列 (中文可检索); 老表没有该列时回退到 text
        _cols = {f.name for f in fresh.schema}
        _fts_col = "text_fts" if "text_fts" in _cols else "text"
        fresh.create_fts_index(_fts_col, replace=True)
        logger.info(f"FTS index created on {table_name} (column={_fts_col})")
    except Exception as e:
        logger.warning(f"FTS index creation failed on {table_name}: {e}")
    return table_name


def prune_superseded(table_name: str, keep_versions: int = 1) -> int:
    """清理被取代的旧版本切片 (定时巡检用)。

    保留最近 keep_versions 个已失效版本以便追溯, 更旧的真正删除 ——
    对应"定时巡检知识库, 删除旧切片"。返回删除的行数。
    """
    db = lancedb.connect(str(LANCEDB_DIR))
    try:
        table = db.open_table(table_name)
    except Exception:
        return 0
    try:
        rows = table.search().where("status = 'superseded'").limit(10000).to_list()
    except Exception:
        return 0
    if not rows:
        return 0
    # 按 doc_key 分组, 保留 version 最大的 keep_versions 个
    by_key: dict[str, list[dict]] = {}
    for r in rows:
        by_key.setdefault(r.get("doc_key") or "", []).append(r)
    drop = 0
    for _k, items in by_key.items():
        # 按**版本**分组, 而不是按行: 一个版本有摘要行+多个全文行,
        # 按行迭代会对同一 paper_id 反复 delete (计数虚高, 实测把 1 版报成 3 组)。
        by_ver: dict[int, set[str]] = {}
        for r in items:
            v = int(r.get("version") or 0)
            pid = r.get("paper_id") or ""
            if pid:
                by_ver.setdefault(v, set()).add(pid)
        for v in sorted(by_ver, reverse=True)[keep_versions:]:
            for pid in by_ver[v]:
                if re.fullmatch(r"[A-Za-z0-9_-]{1,64}", pid):
                    try:
                        table.delete(f"paper_id = '{pid}'")
                        drop += 1
                    except Exception:
                        pass
    if drop:
        logger.info(f"Pruned {drop} superseded document version(s) from {table_name}")
    return drop


def migrate_fts_column(table_name: str, batch: int = 500) -> dict:
    """给存量表补 `text_fts` 并重建 FTS 索引 (**复用已有向量, 不重新嵌入**)。

    为什么需要: 修复中文 BM25 之前建的库都缺 `text_fts` 列, 它们的 FTS 索引建在
    `text` 上 —— 而 LanceDB 默认分词器对无空格中文等于不切, 关键词检索形同虚设
    (实测中文查询命中率 0)。只加列 + 重分词 + 重建索引即可, 无需重算嵌入。
    """
    import lancedb

    from config import LANCEDB_DIR

    db = lancedb.connect(str(LANCEDB_DIR))
    try:
        t = db.open_table(table_name)
    except Exception as e:
        return {"ok": False, "error": f"表不存在: {e}"}

    cols = {f.name for f in t.schema}
    if "text_fts" not in cols:
        try:
            t.add_columns({"text_fts": "''"})
        except Exception as e:
            return {"ok": False, "error": f"加列失败: {e}"}
        t = db.open_table(table_name)  # add_columns 后重开, 避免陈旧句柄

    rows = t.search().limit(1_000_000).to_list()
    done = 0
    for i in range(0, len(rows), batch):
        chunk = rows[i : i + batch]
        # 逐行 update (LanceDB 无"按行批量赋值"): 用 chunk_id 作 where
        for r in chunk:
            cid = (r.get("chunk_id") or "").replace("'", "")
            if not cid:
                continue
            try:
                t.update(
                    where=f"chunk_id = '{cid}'",
                    values={"text_fts": _segment_for_fts(r.get("text") or "")},
                )
                done += 1
            except Exception:
                pass
    fresh = db.open_table(table_name)
    fresh.create_fts_index("text_fts", replace=True)
    logger.info(f"migrate_fts_column: {table_name} 更新 {done}/{len(rows)} 行, 索引已重建")
    return {"ok": True, "table": table_name, "rows": len(rows), "updated": done}


# ---- 图片描述存储 ----


def store_image_descriptions(
    table_name: str,
    image_records: list[dict],
) -> None:
    """将图片描述存入 LanceDB（is_image=True）。"""
    if not image_records:
        return
    db = lancedb.connect(str(LANCEDB_DIR))
    try:
        table = db.open_table(table_name)
    except ValueError as e:
        if "not found" not in str(e).lower():
            raise
        # 表不存在 → 自动创建 (DeepSeek 审查 #4: 原静默 return 丢数据无感知)
        logger.warning(f"Table {table_name} not found, creating for image descriptions")
        table = _get_table(db, table_name)

    texts = [r["text"] for r in image_records]
    vecs = _embed_texts_sync(texts)
    rows = []
    for r, v in zip(image_records, vecs):
        r["vector"] = [float(x) for x in v]
        r.setdefault("chunk_id", r.get("image_path", ""))
        r.setdefault("is_fulltext", True)
        r.setdefault("is_image", True)
        r.setdefault("paper_id", "")
        r.setdefault("title", "")
        r.setdefault("authors", "")
        r.setdefault("year", 0)
        r.setdefault("abstract", "")
        r.setdefault("doi", "")
        r.setdefault("page_number", 0)
        r.setdefault("section", "")
        rows.append(r)
    table.add(rows)

    logger.info(f"Stored {len(rows)} image descriptions in {table_name}")


# ---- 检索 ----


def _cosine_search(
    table, query_vec: list[float], top_k: int, filter_expr: str | None = None
):
    """LanceDB 向量相似度搜索。"""
    q = table.search(query_vec, vector_column_name="vector").limit(top_k)
    if filter_expr:
        q = q.where(filter_expr, prefilter=True)
    try:
        return q.to_list()
    except Exception:
        return []


def _stable_key(r: dict, rank: int) -> str:
    """稳定 RRF 键 (DeepSeek 审查 #6): chunk_id 优先, 否则 paper_id+chunk_index 组合。"""
    cid = r.get("chunk_id", "")
    if cid:
        return cid
    pid = r.get("paper_id", "")
    ci = r.get("chunk_index", "")
    if pid:
        return f"{pid}#{ci}"
    return f"rank#{rank}"


def _hybrid_search(table, q_str, q_vec, top_k, flt=None):
    """向量 + BM25 混合检索，RRF 融合"""
    vec = _cosine_search(table, q_vec, top_k * 2, flt)
    bm25 = []
    try:
        bm25 = table.search(q_str, query_type="fts").limit(top_k * 2).to_list()
    except Exception:
        pass
    if not bm25:
        return vec[:top_k]

    def rrf(rank, k=60):
        return 1.0 / (k + rank)

    fused = {}
    for rank, r in enumerate(vec):
        k = _stable_key(r, rank)
        fused[k] = fused.get(k, 0) + rrf(rank)
    for rank, r in enumerate(bm25):
        k = _stable_key(r, rank)
        fused[k] = fused.get(k, 0) + rrf(rank)

    sorted_ids = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]
    vm = {_stable_key(r, i): r for i, r in enumerate(vec)}
    bm = {_stable_key(r, i): r for i, r in enumerate(bm25)}
    result = []
    for cid, _ in sorted_ids:
        result.append(vm.get(cid) or bm.get(cid) or {"chunk_id": cid})
    return result


def retrieve_with_images(
    table_name: str,
    query: str,
    top_k: int = RETRIEVAL_TOP_K,
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    三层混合检索（BM25+向量）：全文文本 + 图片描述 + 摘要。
    """
    db = lancedb.connect(str(LANCEDB_DIR))
    try:
        table = db.open_table(table_name)
    except Exception:
        return [], [], []

    query_vec = _embed_texts_sync([query])[0]
    text_res = _hybrid_search(
        table, query, query_vec, top_k, "is_fulltext = true AND is_image = false"
    )
    img_res = _hybrid_search(table, query, query_vec, top_k, "is_image = true")
    abs_res = _hybrid_search(table, query, query_vec, top_k, "is_fulltext = false")
    return text_res, img_res, abs_res
