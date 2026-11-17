"""LanceDB 表列举 (兼容不同版本 API)。"""


def list_lance_tables(db) -> list[str]:
    raw = db.list_tables() if hasattr(db, "list_tables") else db.table_names()
    return list(raw.tables) if hasattr(raw, "tables") else [str(t) for t in raw]
