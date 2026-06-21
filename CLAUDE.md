# AI Research Assistant — CLAUDE.md

## Commands

```bash
# API server (port 8001, serves frontend at /)
python api.py

# Vue admin dashboard (dev, port 5173)
cd rag-admin && npm run dev

# Evaluation
python evaluate_ragas.py          # Lightweight 3-metric eval
python evaluate_rag_standards.py   # Embedding-based eval (50 papers, 5622 chunks)
python test_e2e.py                 # End-to-end smoke test

# Run in background
nohup python api.py > /tmp/api_server.log 2>&1 &
```

## Architecture

```
User input (index.html) → FastAPI (api.py, port 8001)
  ├── /create_topic    → search.py → DeepSeek generates search strategy
  ├── /search_papers   → search.py → 6-source parallel search (arXiv, OpenAlex, S2, Unpaywall, CNKI, Apify)
  ├── /download_bulk   → downloader.py → PyMuPDF → semantic chunking → LanceDB
  ├── /ask             → qa_engine.py → rewrite → hybrid retrieve → cascade rerank → generate
  ├── /ask/stream      → SSE streaming version
  ├── /agent/quick     → Direct KB retrieval + single DeepSeek call (<10s)
  └── /agent/run       → agent_core.py → ReAct loop with 8 tools
```

**RAG Pipeline**: Query rewrite (long queries only, cos≥0.75 validation) → AdaptiveHybridRetriever (4-feature dynamic BM25/vector weight) → CascadeRerank (MiniLM pre-filter → bge-reranker refine) → DeepSeek generation with citation format

**Knowledge Base**: LanceDB (local embedded) — dual-layer: abstract (is_fulltext=false) + fulltext chunks (is_fulltext=true) + images (is_image=true). Embedding: bge-small-zh (512d), CPU, ~0.1s per text.

**Agent**: 8 standardized tools (OpenAI Function Calling format), ReAct loop, max_steps=10. `/agent/quick` bypasses ReAct for simple questions (3-5s vs 25-30s).

## Key Files

| File | Purpose |
|------|---------|
| `api.py` | 16 endpoints, CORS, static file serving |
| `search.py` | Strategy generation + multi-source parallel search |
| `qa_engine.py` | Core RAG pipeline (rewrite → retrieve → rerank → generate) |
| `hybrid_retriever.py` | Adaptive RRF fusion + cascade rerank |
| `knowledge_base.py` | LanceDB CRUD, hybrid search, dual-layer indexing |
| `downloader.py` | PDF parsing (PyMuPDF), semantic chunking, image filter |
| `config.py` | All config via env vars |
| `models.py` | Pydantic models (AskRequest, TopicState, PaperMeta) |
| `agent/agent_core.py` | ResearchAgent class, ReAct loop |
| `agent/tool_registry.py` | 8 standardized tools |
| `monitor.py` | Input sanitization, metrics, safe_call wrapper |
| `auth.py` | JWT + bcrypt, user DB in data/users.json |
| `static/index.html` | ~27KB single-file frontend, no build step |

## Gotchas

- **LanceDB API**: `db.list_tables()` returns `ListTablesResponse` object — use `.tables` attribute to get the list. NOT a tuple or plain list.
- **API field names**: `/search_papers` returns `{papers: [...]}` — frontend reads `pd.papers` not `pd.pps`.
- **Frontend JS is minified**: `C()` and `W()` functions are single-line (~500 chars). Use exact string matching in edits. Beware line-number prefixes leaking into content.
- **Model selector**: `AskRequest` must include `model` and `vision_model` fields, or frontend model selector is silently ignored. Same for `CreateTopicRequest`.
- **active_topics is in-memory**: Lost on restart. LanceDB data persists on disk.
- **bge-small-zh > bge-large-zh for sentence retrieval**: Large model scored 73.6% vs small 81.6% on semantic recall. MTEB confirms small models better at sentence-level tasks.
- **DEEPSEEK_MODEL**: Some endpoints import from config locally (`from config import DEEPSEEK_MODEL`). Don't add it as function param without checking local imports.
- **Embedding model loads at startup**: First API start takes ~10s for model download/load. Subsequent restarts are instant (cached).

## Environment

Required `.env`:
```
DEEPSEEK_API_KEY=sk-xxx
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
```

Optional:
```
VISION_API_KEY=xxx       # Qwen-VL (ENABLE_VISION=false by default)
APIFY_API_KEY=xxx        # Google Scholar via Apify
S2_API_KEY=xxx           # Semantic Scholar (skipped without key)
```

## Testing/Evaluation

- `test_e2e.py`: 4-step full pipeline test (strategy, embedding, KB build, QA)
- `evaluate_ragas.py`: Faithfulness + Context Precision + Answer Relevance on 8 QA pairs
- `evaluate_rag_standards.py`: Semantic Recall@5 on 50 papers, 15 QA pairs
- `data/eval_qa.json`: 8 annotated QA pairs (face recognition domain)
- `data/eval_papers_big/`: 94 arXiv PDFs for evaluation KB

Current metrics (94 papers, 10521 chunks, bge-small-zh):
- Faithfulness: 0.8227
- Context Precision: 0.7998
- Semantic Recall@5: 0.8198
