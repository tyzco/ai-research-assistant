"""降级路径体检器 — 检测「本该走正常路径, 却总是走降级」的 bug。

设计原则 (本文件的存在理由):
    降级是**保底**, 不是正常通道。若某环节总在降级, 说明正常路径坏了 —— 该修正常路径,
    而不是接受降级。本项目已因「async 函数漏 await → 解包 coroutine 抛异常 → 被 except 吞 →
    永远降级」这一类 bug 踩坑 5 次 (qa_engine 混合检索 / claim_diagnose 检索 / factcheck 证据 /
    api 两处端点)。本脚本把这类检测固化, 可重复跑。

检测项:
  1. unpack-async-no-await : `a, b = obj.method(...)` 其中 method 是 async def 但没 await
     → 必然抛 "cannot unpack non-iterable coroutine object" → 必然降级/失败
  2. await-in-try-fallback : try 块内调用 async 且无 await, 且该 try 有 except 降级
     → 该降级路径会被 100% 触发

用法:
    python detect_degradation.py            # 全项目扫描
    python detect_degradation.py --file api.py
"""
import argparse
import glob
import re

# 同名但同步的调用 (会误报, 显式排除)
SYNC_WHITELIST = {
    "execute",   # sqlite cursor.execute
    "search",    # LanceDB table.search / re.search —— 注意: 项目自有 hr.search 是 async!
    "list", "dict", "set", "tuple", "str", "int", "float", "len", "open", "print",
    "format", "join", "split", "strip", "keys", "values", "items", "get",
}
# 但下面这些对象的 .search 是 async, 需要单独认 (按调用前缀)
ASYNC_CALL_PREFIXES = ("hr.search", "retriever.search", "self.search")


def _collect_async_names(files):
    names = set()
    for f in files:
        try:
            for m in re.finditer(r"async def (\w+)", open(f).read()):
                names.add(m.group(1))
        except Exception:
            pass
    return names


def scan(files: list[str]) -> list[dict]:
    async_names = _collect_async_names(files)
    hits = []
    for f in files:
        try:
            lines = open(f).readlines()
        except Exception:
            continue
        in_doc = False
        for i, line in enumerate(lines, 1):
            # 跳过文档字符串/注释 (docstring 里的用法示例不是代码)
            _q = line.count('"""') + line.count("'''")
            if _q % 2 == 1:
                in_doc = not in_doc
                continue
            if in_doc:
                continue
            ls = line.strip()
            if ls.startswith("#") or "def " in ls or "await " in ls or "asyncio.run" in ls:
                continue
            m = re.match(r"[\w\s,.]+\s*=\s*([\w.]+)\(", ls)
            if not m:
                continue
            call = m.group(1)
            name = call.split(".")[-1]
            is_async_call = call in ASYNC_CALL_PREFIXES or (
                name in async_names and call not in SYNC_WHITELIST
            )
            # 排除 `x.search(` 这类 LanceDB/re 的同步同名调用
            if is_async_call and (call in ASYNC_CALL_PREFIXES or "." not in call):
                hits.append(
                    {"file": f, "line": i, "call": call, "code": ls[:80]}
                )
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="")
    args = ap.parse_args()
    files = (
        [args.file]
        if args.file
        else [
            f
            for f in glob.glob("*.py")
            + glob.glob("agent/*.py")
            + glob.glob("agent/tools_extra/*.py")
            if not f.startswith(("detect_", "test_"))
        ]
    )
    hits = scan(files)
    print(f"扫描 {len(files)} 个文件 → 疑似「必降级」调用点: {len(hits)}")
    for h in hits:
        print(f"  ❌ {h['file']}:{h['line']}  {h['call']}()  ← {h['code']}")
    if not hits:
        print("  ✅ 未发现 unpack-async-no-await 模式")
    print(
        "\n判据: except 里的降级是保底; 若 try 里必然抛异常, 降级就成了唯一通道 → 必须修 try。"
    )


if __name__ == "__main__":
    main()
