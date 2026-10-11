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
                    pa.field("text", pa.string()),
                    pa.field("vector", pa.list_(pa.float32(), dim)),
                ]
            ),
        )


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
                    "text": p.abstract,
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
                    "text": chunk_text,
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
        table.create_fts_index("text", replace=True)
        logger.info(f"FTS index created on {table_name}")
    except Exception as e:
        logger.warning(f"FTS index creation failed on {table_name}: {e}")
    return table_name


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
