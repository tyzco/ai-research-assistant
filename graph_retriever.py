"""轻量图引导检索: chunk-关键术语二部图 → 检索后按术语重叠扩展跨论文图邻居上下文。

定位: 提高 faithfulness — 检索 top-5 常接不住答案的具体论据, 通过"共享关键术语的图邻居 chunk"
把支撑段落补进上下文 (例如检索到 Double DQN 摘要, 顺图拉到含"target network"机制的具体段落)。

不引入 LLM 实体抽取 (贵), 用正则提取关键术语 (英文驼峰词/大写缩写 + 中文连续名词) —
与项目"自研轻量"一致。索引惰性构建 + 进程级缓存, 线程安全。

用法:
    gr = TermGraphRetriever("papers_xxx")
    neighbors = gr.expand(seed_chunks, k=3)   # [(chunk_id, shared_terms_count)]
    row = gr.get_row(chunk_id)                 # 邻居 chunk 完整行 (含 text/title/year)
"""

import logging
import re
import threading

logger = logging.getLogger(__name__)

# 关键术语正则: 首字母大写驼峰词 / 全大写缩写 / 中文连续名词
_CAMEL_RE = re.compile(r"[A-Z][a-zA-Z0-9]{1,30}")
_ACRONYM_RE = re.compile(r"[A-Z]{2,}[0-9]*")
_CN_RE = re.compile(r"[一-鿿]{2,6}")

# 高频英文停用词 (避免 "The/Figure/Table/This" 等泛词连成假图边)
_EN_STOP = {
    "The", "This", "These", "That", "Figure", "Table", "Section", "Fig", "Our",
    "We", "For", "With", "From", "Using", "Based", "In", "On", "And", "An",
    "A", "To", "Is", "Are", "At", "By", "of", "et", "al", "pro", "St",
    "Data", "Model", "Models", "Results", "Result", "Method", "Methods",
    "However", "Moreover", "Therefore", "Can", "Will", "Not", "have", "has",
}


def extract_terms(text: str) -> set[str]:
    """提取 chunk/查询的关键术语。英文驼峰词 + 全大写缩写 + 中文 2-6 字名词。"""
    terms: set[str] = set()
    if not text:
        return terms
    head = text[:2000]
    terms.update(_CAMEL_RE.findall(head))
    terms.update(_ACRONYM_RE.findall(head))
    terms.update(_CN_RE.findall(head))
    return {t for t in terms if len(t) >= 2 and t not in _EN_STOP}


class TermGraphRetriever:
    """chunk-术语二部图: term → {chunk_id}, 另存 chunk_id → 行。惰性构建 + 进程级缓存。"""

    _term_index: dict[str, dict] = {}
    _rows: dict[str, dict] = {}
    _lock = threading.Lock()

    def __init__(self, table_name: str, scan_limit: int = 5000):
        self.table_name = table_name
        self.scan_limit = scan_limit

    def _load(self):
        if self.table_name in TermGraphRetriever._term_index:
            return TermGraphRetriever._term_index[self.table_name]
        with TermGraphRetriever._lock:
            if self.table_name in TermGraphRetriever._term_index:
                return TermGraphRetriever._term_index[self.table_name]
            import lancedb

            from config import LANCEDB_DIR
            from knowledge_base import _get_table

            db = lancedb.connect(str(LANCEDB_DIR))
            tbl = _get_table(db, self.table_name)
            rows = (
                tbl.search()
                .where("is_fulltext = true AND is_image = false")
                .limit(self.scan_limit)
                .to_list()
            )
            index: dict[str, set[str]] = {}
            row_map: dict[str, dict] = {}
            for r in rows:
                cid = r.get("chunk_id", "")
                if not cid:
                    continue
                row_map[cid] = r
                for t in extract_terms(r.get("text", "")):
                    index.setdefault(t, set()).add(cid)
            TermGraphRetriever._term_index[self.table_name] = index
            TermGraphRetriever._rows[self.table_name] = row_map
            logger.info(
                f"TermGraph index built: {self.table_name} "
                f"({len(row_map)} chunks, {len(index)} terms)"
            )
            return index

    def get_row(self, chunk_id: str) -> dict | None:
        """取某个 chunk 的完整行 (含 text/title/year/page_number)。"""
        self._load()
        return (TermGraphRetriever._rows.get(self.table_name) or {}).get(chunk_id)

    def expand(
        self, seed_chunks: list[dict], k: int = 3, min_shared: int = 2
    ) -> list[tuple[str, int]]:
        """找与 seed_chunks 共享 ≥min_shared 个术语的图邻居 chunk。

        返回 [(chunk_id, 共享术语数)] 按共享数降序取 top-k, 已排除 seed 自身。
        """
        if not seed_chunks:
            return []
        index = self._load()
        seen = {c.get("chunk_id", "") for c in seed_chunks}
        seed_terms: set[str] = set()
        for c in seed_chunks:
            seed_terms |= extract_terms(c.get("text", ""))
        if not seed_terms:
            return []
        # 邻居打分: 共享术语越多越可能是答案支撑段落
        score: dict[str, set[str]] = {}
        for t in seed_terms:
            for cid in index.get(t, set()):
                if cid in seen:
                    continue
                score.setdefault(cid, set()).add(t)
        ranked = sorted(score.items(), key=lambda x: -len(x[1]))
        out = []
        for cid, terms in ranked:
            if len(terms) >= min_shared:
                out.append((cid, len(terms)))
            if len(out) >= k:
                break
        return out
