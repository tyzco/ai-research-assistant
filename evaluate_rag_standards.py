"""RAG 标准评估 (ragas): faithfulness + context_precision + answer_relevancy + context_recall。

用法:
    python evaluate_rag_standards.py                    # 全部题
    python evaluate_rag_standards.py --limit 5          # 只评前 5 题
    python evaluate_rag_standards.py --kb papers_xxx    # 指定 KB

评估器: DeepSeek (llm_factory) + 本地 bge 嵌入 (LocalBgeEmbeddings)。
注: 每个样本会生成真实答案 (qa_engine.ask_with_context) 再交由 ragas 打分, 耗时较长。
"""
import argparse
import asyncio
import json
import logging
import os
import sys
import types


# 离线/本地缓存加载模型, 避免 sentence-transformers 每次启动联网检查 HuggingFace 卡顿
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

logging.basicConfig(level=logging.WARNING)

# ragas 0.4.3 与 langchain_community 0.4.x 兼容 stub (vertexai 已拆分为独立包, 本评估不使用)
_mod = types.ModuleType("langchain_community.chat_models.vertexai")


class _ChatVertexAI:
    def __init__(self, *a, **k):
        raise NotImplementedError("VertexAI 未配置, 本评估使用 DeepSeek")


_mod.ChatVertexAI = _ChatVertexAI
sys.modules.setdefault("langchain_community.chat_models.vertexai", _mod)

from langchain_openai import ChatOpenAI  # noqa: E402
from openai import OpenAI  # noqa: E402
from ragas import EvaluationDataset, SingleTurnSample, evaluate  # noqa: E402
from ragas.embeddings.base import BaseRagasEmbeddings  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.metrics import context_precision  # noqa: E402
# 注: faithfulness/answer_relevancy/context_recall 的 LLM 判定与 DeepSeek 输出格式不兼容
# (faithfulness 恒 0 / answer_relevancy nan / context_recall 卡死)。
# 故: faithfulness+answer_relevancy 用 DeepSeek 直判, context_recall 用嵌入语义召回。

from config import DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL  # noqa: E402
from eval_common import load_eval_qa, pick_kb, semantic_recall  # noqa: E402
from knowledge_base import LANCEDB_DIR, _embed_texts_sync  # noqa: E402


class LocalBgeEmbeddings(BaseRagasEmbeddings):
    """用项目本地 bge 模型提供 ragas 评估嵌入。"""

    def embed_documents(self, texts):
        return _embed_texts_sync(texts)

    def embed_query(self, text):
        return _embed_texts_sync([text])[0]

