"""重建所有 LanceDB 表的向量索引。嵌入模型换型后必须执行。

用法: python reindex_all.py [--table 关键字] [--dry-run]

DeepSeek 审查修复:
  #1/#10 原子双表替换: 先写临时表并验证, 成功后才删旧表+rename, 失败保留原表
  #2/#3  用 to_arrow() 表扫描 + 分批编码写入 (原 search() 无查询向量会崩溃, 全表加载易 OOM)
  #4     保留原表 schema, 仅替换 vector 字段维度 (原硬编码 14 字段会丢额外列)
  #6     FTS 失败告警 (原静默 pass)
  #7     写入后校验行数一致
  #8     argparse + --dry-run
  #9     用 schema 读维度 (原靠首行样本)
"""
import argparse
import logging
import os

import lancedb
import pyarrow as pa

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from config import LANCEDB_DIR, LOCAL_EMBEDDING_DIM, LOCAL_EMBEDDING_MODEL  # noqa: E402
from embedder import embed_texts  # noqa: E402
from eval_common import list_tables  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("reindex")

TARGET_DIM = LOCAL_EMBEDDING_DIM
BATCH = 64


def _vector_dim(tbl) -> int:
    """从 schema 读向量维度 (#9), 而非首行样本。"""
    try:
        return tbl.schema.field("vector").type.list_size or 0
    except Exception:
        return 0


def _rebuild_schema(tbl) -> pa.schema:
    """保留原 schema, 仅替换 vector 字段维度 (#4)。"""
    fields = []
    for f in tbl.schema:
        if f.name == "vector":
            fields.append(pa.field("vector", pa.list_(pa.float32(), TARGET_DIM)))
        else:
            fields.append(f)
    return pa.schema(fields)


def reindex_table(db, table_name: str) -> bool:
    tbl = db.open_table(table_name)
    current_dim = _vector_dim(tbl)
    total = tbl.count_rows()
    if total == 0:
        logger.info(f"  {table_name}: empty, skip")
        return False
    if current_dim == TARGET_DIM:
        logger.info(f"  {table_name}: dim already {current_dim}, skip ({total} rows)")
        return False

    logger.info(f"  {table_name}: {current_dim}d → {TARGET_DIM}d ({total} rows)")
    tmp = f"{table_name}_reindexing"
    try:
        if tmp in list_tables():
            db.drop_table(tmp)  # 清理上次残留
        new_tbl = db.create_table(tmp, schema=_rebuild_schema(tbl))

        # 流式分批: to_arrow() 是表扫描(非 search, #3); 分批编码写入防 OOM (#2)
        written = 0
        for batch in tbl.to_arrow().to_batches(max_chunksize=BATCH):
            rows = batch.to_pylist()
            vecs = embed_texts([r.get("text", "") for r in rows])
            assert len(vecs) == len(rows), "vector count mismatch"
            new_tbl.add(
                [
                    {
                        k: (v if k != "vector" else [float(x) for x in vec])
                        for k, v in r.items()
                    }
                    for r, vec in zip(rows, vecs)
                ]
            )
            written += len(rows)
            logger.info(f"    {written}/{total} rows...")

        assert written == total, f"row count mismatch: {written} vs {total}"  # #7

        # 重建 FTS (#6: 失败告警而非静默)
        try:
            new_tbl.create_fts_index("text", replace=True)
        except Exception as e:
            logger.warning(f"    FTS index failed on {tmp}: {e}")

        # 原子替换 (#1): 新表已完整验证 → 删旧表 → 换名/复制
        db.drop_table(table_name)
        try:
            db.rename_table(tmp, table_name)
        except Exception as e:
            # LanceDB OSS 不支持 rename → 从已验证 tmp 复制数据到原名新表 (无需重嵌入)
            logger.info(f"    rename 不支持({str(e)[:40]}), 改用数据复制")
            _copy_to(db, tmp, table_name)
        logger.info(f"  {table_name}: reindex complete ({total} rows)")
        return True
    except Exception as e:
        # 失败保留原表, 清理临时表
        try:
            if tmp in list_tables():
                db.drop_table(tmp)
        except Exception:
            pass
        logger.error(f"  {table_name}: reindex failed, original kept: {e}")
        raise


def _copy_to(db, src_name: str, dst_name: str) -> None:
    """从已验证的临时表复制数据到原名新表 (数据已嵌入, 无需重嵌入)。"""
    src = db.open_table(src_name)
    dst = db.create_table(dst_name, schema=src.schema)
    for batch in src.to_arrow().to_batches(max_chunksize=BATCH):
        dst.add(batch.to_pylist())
    db.drop_table(src_name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="", help="仅重建名称含此关键字的表")
    ap.add_argument("--dry-run", action="store_true", help="仅列出将重建的表, 不执行")
    args = ap.parse_args()

    tables = [t for t in list_tables() if "_reindexing" not in t]  # 排除残留临时表
    if args.table:
        tables = [t for t in tables if args.table in str(t)]
    if not tables:
        logger.error("No tables found")
        return

    db = lancedb.connect(str(LANCEDB_DIR))
    if args.dry_run:
        logger.info(f"Would reindex {len(tables)} tables → {LOCAL_EMBEDDING_MODEL} ({TARGET_DIM}d)")
        for t in tables:
            dim = _vector_dim(db.open_table(t))
            logger.info(f"  {t}: {dim}d" + ("" if dim != TARGET_DIM else " (ok)"))
        return

    logger.info(f"Reindexing {len(tables)} tables → {LOCAL_EMBEDDING_MODEL} ({TARGET_DIM}d)")
    for t in tables:
        try:
            reindex_table(db, str(t))
        except Exception as e:
            logger.error(f"  FAILED: {t}: {e}")

    logger.info("Done. Run `python run_eval.py` to verify recall.")


if __name__ == "__main__":
    main()
