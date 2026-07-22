"""数据飞轮：机器人模拟用户使用 → 收集 BadCase → 分析 → 驱动优化。

准备2(技术) §51: 数据飞轮 = 上线获取真实反馈 → 反哺系统优化。
准备2(技术) §298: 定期分析 BadCase，反哺数据切分、模型微调或规则优化。
准备2(技术) §300: 根据线上数据调整底层模型，形成"数据→模型→效果"正循环。

本模块用脚本模拟用户行为（学术问答场景），自动收集检索失败/答案不满意案例，
生成 BadCase 报告，供人工 review 后反哺 Prompt/检索策略/分块参数。"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

OUTPUT_DIR = Path("data/flywheel")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ===== 模拟用户模板 =====
SIMULATED_QUERIES = [
    # (问题类型, 示例问题)
    ("定义查询", "什么是{term}？"),
    ("方法对比", "{method_a}和{method_b}有什么区别？"),
    ("应用场景", "{term}在{domain}领域有什么应用？"),
    ("优缺点", "{term}的主要优缺点是什么？"),
    ("数据集", "{dataset}数据集有什么特点？"),
    ("最新进展", "{term}领域最近有哪些突破？"),
    ("方法细节", "{method}的核心创新点是什么？"),
    ("评估指标", "如何评估{task}的性能？"),
]

TERMS = [
    "深度学习",
    "迁移学习",
    "强化学习",
    "图神经网络",
    "注意力机制",
    "BERT",
    "GPT",
    "Transformer",
    "ResNet",
    "GAN",
    "自监督学习",
    "联邦学习",
    "元学习",
    "对比学习",
    "知识蒸馏",
]
DOMAINS = ["医疗", "金融", "教育", "自动驾驶", "自然语言处理"]
METHODS = ["ArcFace", "FaceNet", "DeepFace", "CosFace", "SphereFace"]
DATASETS = ["ImageNet", "LFW", "CIFAR-10", "MS-Celeb-1M", "MegaFace"]


class FlywheelCollector:
    """数据飞轮收集器：运行模拟查询 → 记录指标 → 生成 BadCase 报告。"""

    def __init__(self):
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results: list[dict[str, Any]] = []
        self.start_time = time.time()

    def record(self, query: str, question_type: str, metrics: dict, bad: bool = False):
        self.results.append(
            {
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "type": question_type,
                "metrics": metrics,
                "is_bad_case": bad,
                "notes": metrics.get("notes", ""),
            }
        )

    def analyze(self) -> dict:
        """分析收集的数据，生成优化建议。"""
        total = len(self.results)
        if not total:
            return {"error": "无数据"}

        bad_cases = [r for r in self.results if r["is_bad_case"]]
        bad_rate = len(bad_cases) / total

        # 按问题类型统计
        by_type = {}
        for r in self.results:
            t = r["type"]
            by_type.setdefault(t, {"total": 0, "bad": 0, "avg_recall": 0})
            by_type[t]["total"] += 1
            if r["is_bad_case"]:
                by_type[t]["bad"] += 1
            by_type[t]["avg_recall"] += r["metrics"].get("recall", 0)
        for t in by_type:
            by_type[t]["avg_recall"] /= max(1, by_type[t]["total"])

        # 生成建议
        suggestions = []
        if bad_rate > 0.3:
            suggestions.append("BadCase 率过高 (>30%)：检查检索策略、Chunk 大小或 Embedding 模型是否需要调整")
        worst_type = max(
            by_type, key=lambda t: by_type[t]["bad"] / max(1, by_type[t]["total"])
        )
        if by_type[worst_type]["bad"] > 0:
            suggestions.append(f"问题类型「{worst_type}」BadCase 率高：考虑针对性优化 Prompt 或增加该领域论文")

        return {
            "session": self.session_id,
            "total_queries": total,
            "bad_cases": len(bad_cases),
            "bad_rate": round(bad_rate, 4),
            "by_type": by_type,
            "suggestions": suggestions,
            "duration_seconds": round(time.time() - self.start_time, 1),
        }

    def save(self):
        """持久化到 data/flywheel/。"""
        report = self.analyze()
        ts = self.session_id
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / f"report_{ts}.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False)
        )
        (OUTPUT_DIR / f"raw_{ts}.json").write_text(
            json.dumps(self.results, indent=2, ensure_ascii=False)
        )
        return report


async def simulate_user_session(num_queries: int = 20) -> dict:
    """运行一次模拟用户会话。"""
    import random

    import lancedb

    from config import LANCEDB_DIR
    from qa_engine import ask_question

    # 找可用的 KB
    db = lancedb.connect(str(LANCEDB_DIR))
    raw = db.list_tables()
    tables = raw.tables if hasattr(raw, "tables") else []
    kb = tables[0] if tables else None
    if not kb:
        return {"error": "无可用知识库"}

    collector = FlywheelCollector()
    collector.session_id = f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    for _ in range(num_queries):
        tmpl_type, tmpl = random.choice(SIMULATED_QUERIES)
        query = tmpl.format(
            term=random.choice(TERMS),
            method_a=random.choice(METHODS),
            method_b=random.choice(METHODS),
            domain=random.choice(DOMAINS),
            method=random.choice(METHODS),
            task=random.choice(["人脸识别", "图像分类", "目标检测", "语义分割"]),
            dataset=random.choice(DATASETS),
        )

        try:
            result = await ask_question(str(kb), query, use_rerank=True)
            answer = result.answer or ""
            # Heuristic: answer < 50 chars = bad (probably couldn't retrieve)
            is_bad = len(answer) < 50 or "信息不足" in answer or "无法回答" in answer
            collector.record(
                query,
                tmpl_type,
                {
                    "answer_length": len(answer),
                    "references": len(result.references),
                    "supplements": len(result.supplement),
                    "recall": min(1.0, len(result.references) / 3),
                    "notes": "retrieval_failure"
                    if is_bad and len(result.references) < 2
                    else "",
                },
                bad=is_bad,
            )
        except Exception:
            collector.record(
                query, tmpl_type, {"recall": 0, "notes": "api_error"}, bad=True
            )

    return collector.save()
