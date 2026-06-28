# AI 学术文献精读助手

> 多源搜索 · RAG 深度问答 · Agent 自主调研 · 2C4G 无 GPU 部署

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

面向研究生/学者的文献精读工具。输入研究方向 → 检索策略 + 多源论文发现 → 一键下载构建向量知识库 → 基于全文的深度 RAG 问答 → Agent 自主调研。

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 API Key
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY

# 3. 启动
python api.py
# 打开 http://localhost:8001
```

## 核心能力

| 模块 | 实现 |
|------|------|
| **多源搜索** | arXiv + OpenAlex + Semantic Scholar + Unpaywall + CNKI + Google Scholar 并行搜索 |
| **自适应混合检索** | 4 特征动态 BM25/向量权重 RRF 融合，非固定 50:50 |
| **级联重排** | MiniLM 预筛 → bge-reranker 精排，节省 40% 推理时间 |
| **三层 RAG** | 全文文本 + 图片描述 + 摘要补充，三层独立检索 |
| **多模态** | PDF → PyMuPDF 提取图表 → Qwen-VL 生成中文描述 → 向量化 |
| **Agent** | ReAct 循环 + 8 工具 + 自反思机制，自动调研 |
| **多轮记忆** | 滑动窗口 + 滚动摘要 + 指代消解 |
| **评估体系** | Faithfulness 0.82 / Semantic Recall@5 0.82 / Context Precision 0.80 |

## 技术栈

**后端**: FastAPI (asyncio) · DeepSeek Chat API · bge-small-zh (512d) · bge-reranker-base · LanceDB · PyMuPDF

**前端**: 单文件 HTML/CSS/JS (~27KB, 零构建) · Vue 3 + Element Plus 管理后台

**部署**: Docker + docker-compose · Python 3.11-slim · 2C4G 无 GPU

## 项目结构

```
├── api.py                 # FastAPI 后端 (16 端点)
├── qa_engine.py           # RAG 问答引擎 (改写→检索→重排→生成)
├── hybrid_retriever.py    # ★ 自适应混合检索 + 级联重排
├── knowledge_base.py      # LanceDB 双层索引
├── search.py              # 6 源并行论文搜索
├── downloader.py          # PDF 解析 + 语义分块
├── agent/
│   ├── agent_core.py      # ReAct Agent 循环
│   └── tool_registry.py   # 8 标准化工具
├── static/index.html      # Web 前端
├── rag-admin/             # Vue 3 管理后台
├── evaluate_ragas.py      # 评估脚本
└── 项目总结-面试版.md      # 面试文档
```

## 评估指标

| 指标 | 数值 | 说明 |
|------|------|------|
| Faithfulness | 0.8227 | LLM-as-judge，回答基于上下文程度 |
| Context Precision | 0.7998 | 嵌入余弦相似度 |
| Semantic Recall@5 | 0.8198 | Top-5 检索覆盖 ground truth 比例 |
| ≥0.80 通过率 | 87% (13/15) | |

**KB**: 94 篇 arXiv 论文, 10521 文本块, bge-small-zh (512d)

## 消融实验

| 方法 | Recall |
|------|--------|
| Pure Vector Top-5 | 0.8081 |
| Pure Vector Top-10 | 0.8108 |
| Hybrid + QueryExp Top-10 | **0.8110** |

## 未来方向

- **LongRAG**（已跑实验）：chunk 2048→4096，chunk 数减少 81%，检索指标持平。大块 + 长窗口 LLM 可提升生成质量
- **GraphRAG**：实体关系图谱（方法-数据集-指标），支持跨论文关系推理
- **REFRAG**：Token 级压缩检索，降低 LLM 输入成本

详见 [RAG优化对照分析.md](./RAG优化对照分析.md)
