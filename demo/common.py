"""Demo 共享工具：统一环境 / 分节打印 / 状态衔接 / 产物存档 / 鲁棒降级。

每个 demo 独立可跑，输出即演示稿；状态经 demo/output/state.json 在模块间衔接。
"""
import json
import os
import sys
import time
from pathlib import Path

# 统一环境：离线 HF（模型本地缓存）+ 项目根进 sys.path
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEMO_DIR = Path(__file__).resolve().parent
OUT = DEMO_DIR / "output"
OUT.mkdir(parents=True, exist_ok=True)
STATE = OUT / "state.json"


def banner(title: str, sub: str = ""):
    """模块大标题。"""
    w = 72
    print("\n" + "=" * w)
    print("  " + title)
    if sub:
        print("  " + sub)
    print("=" * w)


def section(title: str):
    print("\n── " + title + " " + "─" * max(2, 56 - len(title)))


def step(label: str):
    print("  · " + label + " ...")


def show(k, v):
    s = str(v)
    if len(s) > 130:
        s = s[:127] + "..."
    print("  " + k + ": " + s)


def ok(msg: str):
    print("  [OK] " + msg)


def save_artifact(name: str, md: str):
    """保存 Markdown 产物到 demo/output（演示可出示）。"""
    p = OUT / name
    p.write_text(md, encoding="utf-8")
    print("  [保存] " + p.relative_to(ROOT).as_posix())


def load_state() -> dict:
    if STATE.exists():
        try:
            return json.loads(STATE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_state(**kw):
    """模块间状态衔接：topic_id / papers / table 逐段向后传。"""
    s = load_state()
    s.update(kw)
    STATE.write_text(
        json.dumps(s, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def next_hint(cmd: str, desc: str):
    print("\n  → 下一步: " + cmd + "   (" + desc + ")")


def guard(fn, msg: str):
    """鲁棒降级：模块内单步失败不中断演示。"""
    try:
        return fn()
    except Exception as e:
        print("  [跳过] " + msg + " —— " + str(e)[:100])
        return None


def run(fn):
    """统一入口：捕获异常，演示时输出干净报错而非 traceback（适合面向观众）。"""
    import asyncio

    try:
        if asyncio.iscoroutinefunction(fn):
            asyncio.run(fn())
        else:
            fn()
    except KeyboardInterrupt:
        print("\n  [演示中断] 用户终止。")
    except Exception as e:
        print("\n  [演示中断] " + type(e).__name__ + ": " + str(e)[:150])
        print("  请检查网络 / API Key 后重试。")
