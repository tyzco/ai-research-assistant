# PaperLink — CLAUDE.md (v4.1)

## Quick Commands

```bash
python api.py                                    # Dev server (port 8001)
docker-compose up -d                             # Production deploy
python run_eval.py                               # Retrieval eval (R@5+MRR)
LOCAL_EMBEDDING_MODEL=BAAI/bge-m3 LOCAL_EMBEDDING_DIM=1024 python reindex_all.py  # Rebuild vectors
```

## Architecture Overview

```
user input → static/index.html (SPA, ~45KB, IIFE, zero-build)
  │
  ├─ /create_topic → topic_manager → SQLite
  ├─ /search_papers/stream → search.py (12-parallel sources)
  ├─ /ask → qa_engine.ask_question() → hybrid_retriever → DeepSeek
  ├─ /chat → DeepSeek (no RAG, with history)
  ├─ /agent/pi → pi_agent.PiAgent (hook-driven, 10 lifecycle hooks)
  ├─ /team/start → research_team.ResearchTeam (15 agents, 3 approval gates)
  ├─ /citation/chain → integrations.expand_citation_chain (Semantic Scholar)
  ├─ /subscribe → integrations.subscribe_topic (auto-scheduled via scheduler.py)
  ├─ /zotero/import + /zotero/scan → Zotero bridge
  ├─ /export/paper + /export/download → Markdown/DOCX/LaTeX/BibTeX
  ├─ /kg_data → knowledge_graph → D3.js force-graph
  └─ /feedback → learning_feedback table (data flywheel)
```

## Module Map (32 Python files)

| Layer | File | Responsibility |
|-------|------|----------------|
| **Infra** | `config.py` | All config constants + env vars (NO hardcoded values) |
| | `database.py` | SQLite 5 tables, WAL, thread-safe |
| | `logger.py` | Unified structured logging |
| | `embedder.py` | Embedding model abstraction (ST + FlagEmbedding dual backends) |
| | `auth.py` | JWT + bcrypt, SQLite-backed user DB |
| | `monitor.py` | Input sanitization, metrics, safe_call wrapper |
| | `models.py` | Pydantic models (TopicState, PaperMeta, AskRequest) |
| | `topic_manager.py` | Topic lifecycle, SQLite persistence |
| | `conversation_memory.py` | Multi-turn memory, SQLite persistence |
| **Data** | `search.py` | Strategy generation + 12-source parallel search + quality filter |
| | `downloader.py` | PDF parsing (PyMuPDF), image extraction, semantic chunking |
| **Knowledge** | `knowledge_base.py` | LanceDB dual-layer index + FTS auto-creation |
| | `hybrid_retriever.py` | Adaptive RRF fusion + 2-stage cascade rerank |
| | `knowledge_graph.py` | Entity extraction + D3.js graph data |
| | `qa_engine.py` | Full RAG pipeline (8 steps: coref→rewrite→retrieve→rerank→expand→prompt→generate) |
| | `intent_classifier.py` | 三层漏斗第 2 层: 嵌入语义意图分类 (规则→语义→LLM 工具, margin 判定) |
| **Agent** | `agent/research_team.py` | **13-agent virtual team + dual-track approval (3 gates)** |
| | `agent/pi_agent.py` | Hook-driven agent loop (trace/turn lifecycle) |
| | `agent/hooks.py` | 10 lifecycle intervention hooks |
| | `agent/context_engine.py` | Token compaction + tool output truncation |
| | `agent/tool_manager.py` | Param validation + security check + error recovery |
| | `agent/skills.py` | 4 core capability skills |
| | `agent/tool_registry.py` | 8+2 tool registry (Function Calling) |
| **Eval** | `run_eval.py` | Automated retrieval eval (R@5+MRR+regression gate) |
| | `reindex_all.py` | Vector rebuild tool for model migration |

## 13-Agent Research Team

```
UPSTREAM (3):   topic_evaluator → multi_source_searcher → quality_screener
                    [🔴 Gate 1: topic_direction]        [🔴 Gate 2: core_literature]

MIDSTREAM (6):  review_synthesizer → chapter_writer_intro/method/experiment
                    → figure_generator → bib_manager

DOWNSTREAM (4): peer_reviewer → format_checker → ethics_reviewer → final_drafter
                                                          [🔴 Gate 3: final_conclusion]
```

**Dual-track mechanism**: Auto execution with 3 human approval gates. Agents run automatically through pipeline. At each gate, frontend pauses with approval bar — user must confirm or provide feedback before continuing.

## Key Gotchas

- **LanceDB**: `db.list_tables()` returns object with `.tables` attribute. Use `raw.tables` not raw iteration.
- **FTS index**: MUST call `table.create_fts_index("text")` after `build_knowledge_base`. Without it, BM25 silently fails.
- **Config imports**: Use `from config import XXX` at top of file. Some endpoints do local `from config import ...` inside functions for hot-reload safety.
- **Frontend JS**: All inside IIFE `(function(){ ... })()`. Only window-exposed functions are accessible from HTML onclick handlers.
- **SQLite**: Thread-local connections with `threading.local()`. WAL mode enabled. Foreign keys cascade (delete topic → auto-delete messages/memories).
- **ConversationMemory**: Key is `topic_id` now (was `table_name` before fix). `/ask` endpoint passes `memory_key=req.topic_id`.
- **search.py requires `SPAM_KEYWORDS`**: The formatter/linter tends to strip `SPAM_KEYWORDS` from the import block. If search returns 0 results, check `from config import ... SPAM_KEYWORDS`.
- **LLM_GEN_MAX_TOKENS=1024**: Changed from 500 to prevent answer truncation. Agent agents use `AGENT_MAX_TOKENS=1000`.
- **Frontend JS**: IIFE `(function(){ ... })()`, brackets must be exactly 232/232. If buttons stop working, check brace balance first. Common breakage cause: Python-style `\UXXXXXXXX` escapes in inline JS strings.
- **Frontend escape safety**: NEVER use `\\x27` or raw `\\'` inside single-quoted JS strings in generated HTML. Use `"` quoting for attribute values, `&quot;` for quotes in data.
- **monitor MetricsTracker 用 `threading.RLock`**: to_dict() 持锁时调 estimate_cost() 也加锁, 普通 Lock 自死锁 → /metrics 永久挂起 (已修复, 勿改回 Lock)。

## Environment

Required `.env`:
```
DEEPSEEK_API_KEY=sk-xxx
```

Optional:
```
LOCAL_EMBEDDING_MODEL=BAAI/bge-m3   # upgrade embedding
LOCAL_EMBEDDING_DIM=1024            # bge-m3 dimension
JWT_SECRET_KEY=random-string        # production must change
CORS_ORIGINS=http://your-domain     # deployment config
```

## Testing / Eval

- `run_eval.py` — Full retrieval eval (Recall@5, MRR, Vector vs Hybrid, regression gate)
- `test_e2e.py` — 4-step end-to-end (strategy → embed → kb → qa)
- `benchmark.py` — Performance benchmark (latency + throughput)
- Baseline: bge-small-zh 512d, R@5=0.7703, MRR=0.75
