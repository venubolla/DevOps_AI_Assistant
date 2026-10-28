# DevOps AI Assistant

[![CI](https://github.com/venubolla/DevOps_AI_Assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/venubolla/DevOps_AI_Assistant/actions/workflows/ci.yml)

A fully local, citation-backed RAG assistant for DevOps knowledge (runbooks, incidents, Kubernetes and deployment docs). It combines hybrid retrieval (vector + keyword) with reranking, a LangGraph agent with query rewriting and retry, grounded answer generation, and an evaluation harness built on custom retrieval metrics and [RAGAS](https://github.com/explodinggradients/ragas). Everything runs locally with Ollama and PostgreSQL/pgvector, so there is no API cost.

## Features

- **Hybrid retrieval**: pgvector semantic search plus keyword search, fused and reranked.
- **LangGraph agent**: retrieval tool, relevance gate, query-rewrite retry, grounded answer.
- **Conversation memory**: follow-up questions keep the context of earlier turns.
- **Citations**: every answer returns the sources it used, with scores.
- **Metadata filters**: filter by `service`, `environment` and `document_type`.
- **FastAPI backend** and a **Streamlit chat UI** that streams the agent's progress.
- **Evaluation**: Hit@K, Recall@K, MRR, an LLM answer judge, and RAGAS (Faithfulness, Context Recall, Factual Correctness).

## Architecture

### Offline: ingestion

```
 data/knowledge_base/          ┌───────────────────────────────────────────────┐
  .md / .txt / .pdf  ────────► │ Ingestion  (python -m app.rag.ingestion)      │
  + YAML front matter          │  1. load file, read metadata (service,        │
                               │     environment, document_type, version)      │
                               │  2. hash content + metadata + path            │
                               │     (re-ingest replaces changed documents)    │
                               │  3. chunk: 800 chars, 120 overlap             │
                               │  4. embed: MiniLM-L6-v2, 384-dim, normalized  │
                               └───────────────────────┬───────────────────────┘
                                                       ▼
                               ┌───────────────────────────────────────────────┐
                               │ PostgreSQL 16 + pgvector  (Docker, port 5433) │
                               │  documents · document_chunks                  │
                               │  HNSW index (cosine) · GIN full-text index    │
                               └───────────────────────────────────────────────┘
```

### Online: answering a question

```
 Streamlit chat UI ──► FastAPI ──────────► LangGraph agent
  history + filters     /api/v1/agent/ask     (Ollama qwen3:8b with a search tool)
  streamed progress     /api/v1/agent/ask/stream
                                                    │
                          ┌─────────────────────────┴─────────────────────────┐
                          │ model requests the search tool                    │ model replies without it
                          ▼                                                   ▼
              ┌────────────────────────────────────────────┐     ┌───────────────────────────┐
              │ Hybrid reranked search                     │     │ Scope check               │
              │                                            │     │  search anyway, then      │
              │  vector search      keyword search         │     │  classify the message     │
              │  pgvector cosine    Postgres full-text     │     │  ├ KB is relevant         │
              │  top 20             top 20                 │     │  │   → force retrieval    │
              │        └──────────┬──────────┘             │     │  ├ small talk → reply     │
              │                   ▼                        │     │  ├ DevOps, no KB match    │
              │   RRF fusion: rank merge, k = 60           │     │  │   → not-found message  │
              │                   ▼                        │     │  └ anything else          │
              │   Cross-encoder rerank (ms-marco MiniLM)   │     │      → refuse             │
              │                   ▼                        │     └───────────────────────────┘
              │   top 5 chunks with reranker scores        │
              └─────────────────────┬──────────────────────┘
                                    ▼
                   Relevance gate: best reranker score >= RELEVANCE_THRESHOLD (-5.0)
                      ├─ pass ─────────────────────────────────────────────────┐
                      ├─ fail, first time ─► rewrite query (fast LLM) ─► search │
                      │                      again ─► back to the gate          │
                      └─ fail, after the retry ─► "not enough information"      │
                                                                                │
                                    ┌───────────────────────────────────────────┘
                                    ▼
                   Answer generation: context-only prompt, citations [1] [2] [3]
                                    ▼
                   Grounding check: independent judge model (qwen2.5:7b-instruct)
                      ├─ supported ─► answer + sources (title, path, score)
                      └─ unsupported or judge error ─► safe fallback (fails closed)
```

`/api/v1/rag/ask` runs the same hybrid reranked search followed by answer generation, without the agent loop. The evaluation scripts use it as the non-agentic baseline.

The "top 20" candidates and "top 5" results are the defaults: the API requests `top_k` results (5) from `max(top_k * 4, 10)` candidates (20).

### Offline: evaluation

```
 data/evaluation/rag_eval.json ──► scripts/evaluate_retrieval.py ──► Hit@K · Recall@K · MRR
                               └─► scripts/evaluate_answers.py   ──► keyword coverage · citation validity
                                                                       · semantic similarity · LLM judge
 data/ragas_dataset.json ────────► scripts/evaluate_ragas.py     ──► RAGAS (judge: qwen2.5:7b-instruct)
                                                                       Faithfulness · Context Recall
                                                                       · Factual Correctness
                                                                       → data/ragas_results/
 scripts/evaluate_langgraph_comparison.py ──► direct RAG vs agent on the same questions
 scripts/calibrate_relevance_threshold.py ──► reranker score vs. relevance label table
```

## Tech stack

FastAPI · LangGraph · LangChain · PostgreSQL + pgvector · SQLAlchemy · sentence-transformers · Ollama · RAGAS · Streamlit

## Project layout

```
app/
  api/          FastAPI routes and schemas
  graph/        LangGraph agent (state, nodes, tools, graph)
  rag/          ingestion, chunking, embeddings, search, reranker, generation
  database/     SQLAlchemy models and connection
  evaluation/   datasets, metrics and RAGAS configuration
ui/             Streamlit chat app
tests/          pytest suite
scripts/        evaluation, calibration and manual smoke scripts
data/           knowledge base and evaluation datasets
```

## Getting started

### Prerequisites

- Python 3.14
- Docker (for PostgreSQL with pgvector)
- [Ollama](https://ollama.com) with the models below

```bash
ollama pull qwen3:8b
ollama pull qwen2.5:7b-instruct
```

### Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env     # then edit the values, especially DATABASE_URL and the password
docker compose up -d     # starts PostgreSQL + pgvector on port 5433
```

Make sure the password in `.env` matches the one in `docker-compose.yml`.

Initialise the schema and ingest the knowledge base:

```bash
python -c "import app.database.models; from app.database.init_db import init_db; init_db()"
python -m app.rag.ingestion
```

### Run

```bash
# Terminal 1: API
uvicorn app.main:app --reload

# Terminal 2: UI
streamlit run ui/streamlit_app.py
```

- API docs: http://localhost:8000/docs
- UI: http://localhost:8501 (set `API_URL` if the API runs elsewhere)

The UI is a single chat backed by the agent. `/api/v1/rag/ask` remains available
as the non-agentic baseline used by the evaluation scripts.

### Example request

```bash
curl -X POST http://localhost:8000/api/v1/agent/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I troubleshoot a CrashLoopBackOff pod?", "top_k": 5}'
```

```json
{
  "question": "...",
  "answer": "... [1]",
  "sources": [{"citation": "[1]", "title": "...", "source": "...", "score": 0.0}]
}
```

## API endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Service health |
| GET | `/health/database` | Database connectivity |
| POST | `/api/v1/rag/search` | Semantic search, returns chunks |
| POST | `/api/v1/rag/ask` | Hybrid search + reranker + grounded answer |
| POST | `/api/v1/agent/ask` | LangGraph agent with query rewrite and retry |
| POST | `/api/v1/agent/ask/stream` | Same agent, streaming NDJSON progress then the answer |

`/api/v1/agent/ask` and its streaming variant accept an optional `history` list
of `{"role": "user" \| "assistant", "content": "..."}` turns.

## Evaluation

```bash
python scripts/evaluate_retrieval.py       # Hit@K, Recall@K, MRR
python scripts/evaluate_answers.py         # custom LLM judge
python scripts/build_ragas_dataset.py      # build the saved RAGAS dataset
python scripts/evaluate_ragas.py           # RAGAS metrics
```

Latest RAGAS run (20 questions, local `qwen2.5:7b-instruct` judge):

| Metric | Score |
|---|---|
| Faithfulness |0.8669|
| Context Recall | 1.000 |
| Factual Correctness (precision) | 0.4710 |

### Limitations

- The knowledge base is small (3 documents), so these numbers are not a claim of production quality.
- The judge is a local 7B model, which can be noisy and biased. Treat the scores as relative signals.
- A Context Recall of 1.0 on every sample should be read with caution given the small corpus.
- RAGAS runs on a laptop-class model are slow. `evaluate_ragas.py` uses a single worker and a long timeout to avoid request timeouts.

## Configuration

Settings are read from `.env` (see [.env.example](.env.example)): `DATABASE_URL`, `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `RAGAS_MODEL`, `JUDGE_MODEL`, `RELEVANCE_THRESHOLD`.

`RELEVANCE_THRESHOLD` defaults to `-5.0`. On the labeled set the 20 in-scope
questions score at or above `-0.44` while out-of-scope questions score around
`-11.0`, so the threshold sits inside that gap. Re-run
[scripts/calibrate_relevance_threshold.py](scripts/calibrate_relevance_threshold.py)
after the knowledge base grows.

## Tests

```bash
pytest                    # unit tests (no Ollama or database needed)
pytest -m integration     # also needs the PostgreSQL container running
```

The remaining `scripts/test_*.py` files are manual smoke checks that need a live Ollama and/or database.

## Linting and formatting

```bash
ruff check .
ruff format .
```

## License

MIT. See [LICENSE](LICENSE).
