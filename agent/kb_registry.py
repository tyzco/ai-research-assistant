"""知识库表注册表 (进程内缓存)。

Agent 需要知道"有哪些库可选" —— 否则它只能猜表名。实测出现过模型编造
`papers_rrf` / `rrf` 的情况, 而工具返回"暂无知识库内容", 模型误以为库为空,
继续硬写。这里统一提供:
  - list_kbs(): 真实表名 + 行数 (缓存 30s, 避免每次调用扫 LanceDB)
  - resolve_kb(name): topic_id / papers_<id> / 表名 → 真实表名, 不存在返回 None
  - available_hint(): 给模型/错误信息用的可用列表 (排除评测表)
"""

import threading
import time

from config import LANCEDB_DIR

_TTL = 30.0
_lock = threading.Lock()
_state: dict = {"at": 0.0, "rows": []}

# 评测/临时库不作为 Agent 的默认可用库, 避免把 benchmark 当课题资料
_EXCLUDE = ("eval", "benchmark", "test", "tmp", "probe")


def list_kbs(force: bool = False) -> list[dict]:
    now = time.time()
    with _lock:
        if not force and _state["rows"] and now - _state["at"] < _TTL:
            return list(_state["rows"])
    import lancedb

    db = lancedb.connect(str(LANCEDB_DIR))
    try:
        names = list(db.list_tables().tables)
    except Exception:
        names = list(db.table_names())
    rows = []
    for t in names:
        if not str(t).startswith("papers_"):
            continue
        try:
            n = db.open_table(t).count_rows()
        except Exception:
            n = 0
        rows.append({"table": str(t), "rows": n})
    rows.sort(key=lambda x: -x["rows"])
    with _lock:
        _state.update(at=time.time(), rows=rows)
    return rows


def resolve_kb(name: str) -> str | None:
    """把用户/模型给的名称解析成真实表名; 不存在返回 None。"""
    if not name:
        return None
    raw = str(name).strip()
    names = {r["table"] for r in list_kbs()}
    if raw in names:
        return raw
    cand = f"papers_{raw}"
    if cand in names:
        return cand
    low = {n.lower(): n for n in names}
    return low.get(raw.lower()) or low.get(cand.lower())


def available_hint(limit: int = 8, include_eval: bool = False) -> list[str]:
    rows = list_kbs()
    if not include_eval:
        filtered = [
            r
            for r in rows
            if not any(x in r["table"].lower() for x in _EXCLUDE)
        ]
        rows = filtered or rows
    return [f"{r['table']} ({r['rows']} 行)" for r in rows[:limit]]
