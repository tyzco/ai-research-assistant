"""LanceDB 知识库模块：双层向量索引（全文深度层 + 摘要广度层）。
嵌入：本地 BAAI/bge-small-zh，CPU 推理，零成本。
"""

import logging

import lancedb
import pyarrow as pa
from sentence_transformers import SentenceTransformer

from config import (
    IMAGE_DIR,
    LANCEDB_DIR,
    LOCAL_EMBEDDING_DIM,
    LOCAL_EMBEDDING_MODEL,
    RETRIEVAL_TOP_K,
)
from models import PaperMeta

logger = logging.getLogger(__name__)

LANCEDB_DIR.mkdir(parents=True, exist_ok=True)
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# ---- 嵌入模型 ----

_embed_model = None


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        logger.info(f"Loading local embedding model: {LOCAL_EMBEDDING_MODEL}")
        _embed_model = SentenceTransformer(LOCAL_EMBEDDING_MODEL)
    return _embed_model


def _embed_texts_sync(texts: list[str]) -> list[list[float]]:
    """本地批量编码（sentence-transformers, CPU, 免费）。"""
    if not texts:
        return []
    model = _get_embed_model()
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return embeddings.tolist()


# ---- LanceDB 表管理 ----


def _get_table(db, table_name: str):
    try:
        return db.open_table(table_name)
    except Exception:
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
        for chunk_idx, chunk_text in enumerate(fulltext_chunks[p.paper_id]):
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
                    "page_number": 0,
                    "chunk_index": chunk_idx + 1,  # 0=abstract, 1+=fulltext
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
    except Exception:
        return

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


async def dual_retrieve(
    table_name: str,
    query: str,
    top_k: int = RETRIEVAL_TOP_K,
) -> tuple[list[dict], list[dict]]:
    """
    双层检索：全文层 top-k + 摘要层 top-k。
    返回 (fulltext_results, abstract_results)。
    """
    db = lancedb.connect(str(LANCEDB_DIR))
    try:
        table = db.open_table(table_name)
    except Exception:
        return [], []

    query_vec = _embed_texts_sync([query])[0]
    ft_res = _cosine_search(table, query_vec, top_k, "is_fulltext = true")
    abs_res = _cosine_search(table, query_vec, top_k, "is_fulltext = false")
    return ft_res, abs_res


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
        fused[r.get("chunk_id", str(rank))] = fused.get(
            r.get("chunk_id", str(rank)), 0
        ) + rrf(rank)
    for rank, r in enumerate(bm25):
        fused[r.get("chunk_id", str(rank))] = fused.get(
            r.get("chunk_id", str(rank)), 0
        ) + rrf(rank)

    sorted_ids = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:top_k]
    result, vm, bm = (
        [],
        {r.get("chunk_id", ""): r for r in vec},
        {r.get("chunk_id", ""): r for r in bm25},
    )
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
