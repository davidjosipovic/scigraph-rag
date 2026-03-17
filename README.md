# SCIGRAPH-RAG: Knowledge Graph RAG for Scientific Papers

A **GraphRAG** pipeline that uses the [Open Research Knowledge Graph (ORKG)](https://orkg.org/) to answer questions about scientific papers. Retrieval is deterministic — SPARQL queries against a structured KG, not vector similarity — so every answer cites exact paper URIs and DOIs from the graph.

## Key Features

- **10-step pipeline**: classify → extract → normalize → retrieve (parallel SPARQL) → rank → hard/soft filter → truncate → context → generate → sources
- **Query classification** (6 types): `topic_search`, `method_comparison`, `dataset_search`, `claim_verification`, `method_usage`, `paper_lookup`
- **Scientific NER**: Extracts methods, datasets, tasks, research fields, and metrics using Llama 3 with few-shot prompting; keyword fallback when NER returns nothing
- **Entity normalization**: ~90 method synonyms, ~55 dataset synonyms, 28 task synonyms, 11 field synonyms — expands abbreviations like CNN → convolutional neural network, NER → named entity recognition
- **Parallel SPARQL retrieval**: `asyncio.gather()` fires all strategy queries concurrently; 10 s timeout with title-keyword fallback
- **SPARQL injection protection**: `_sanitize()` strips dangerous characters; word-boundary regex (`\bTERM\b`) for short terms (≤ 4 chars) to prevent substring false-positives (e.g. "NER" matching "mineral")
- **Thread-safe query cache**: module-level 256-entry FIFO cache with `threading.Lock()`
- **Heuristic ranking**: +2 method match, +2 dataset match, +1 title keyword; `hard_filter` requires both method and dataset when both are present; `soft_filter` drops score-0 noise
- **Year validation**: rejects ORKG values like "9" or "12" (months stored in year predicates) — only 4-digit years 1900–2099 accepted
- **Fully local LLM**: Ollama (Llama 3) via persistent `httpx.Client` connection pool — zero API costs
- **138 tests** across unit, integration, and end-to-end pipeline tests with `pytest-asyncio`

## Architecture

```
User Question
      │
      ▼
┌─────────────┐
│  Classify   │  → query_type (6 types)          ┐
└──────┬──────┘                                   │ parallel
┌──────▼──────┐                                   │
│   Extract   │  → methods, datasets, tasks, ...  ┘
└──────┬──────┘
┌──────▼──────┐
│  Normalize  │  → synonym/abbreviation expansion
└──────┬──────┘
┌──────▼──────┐
│  Retrieve   │  → parallel SPARQL (asyncio.gather) + timeout fallback
└──────┬──────┘
┌──────▼──────┐
│    Rank     │  → score rows by entity overlap
└──────┬──────┘
┌──────▼──────┐
│   Filter    │  → hard_filter (method+dataset), soft_filter (score ≥ 1)
└──────┬──────┘
┌──────▼──────┐
│  Truncate   │  → top 8 papers
└──────┬──────┘
┌──────▼──────┐
│   Context   │  → structured per-paper blocks with scores
└──────┬──────┘
┌──────▼──────┐
│  Generate   │  → Llama 3 via Ollama (run_in_executor, non-blocking)
└──────┬──────┘
┌──────▼──────┐
│   Sources   │  → deduplicated papers with URI, DOI, year, methods, datasets
└─────────────┘
```

## Project Structure

```
scigraph-rag/
├── backend/
│   ├── api/
│   │   ├── routes.py          # FastAPI endpoints: POST /ask, GET /health
│   │   └── schemas.py         # Pydantic v2 request/response models
│   ├── kg/
│   │   ├── queries.py         # SPARQL query builders (method, dataset, title, field)
│   │   └── sparql_client.py   # Thread-safe SPARQL client with 256-entry cache
│   ├── rag/
│   │   ├── pipeline.py        # Main orchestrator (10-step pipeline)
│   │   ├── query_classifier.py
│   │   ├── entity_extractor.py
│   │   ├── entity_normalization.py  # METHOD/DATASET/TASK/FIELD synonyms
│   │   ├── query_builder.py   # Multi-strategy retrieval planner
│   │   ├── ranking.py         # Heuristic scoring + hard/soft filter
│   │   └── context_builder.py # Context + sources in one pass
│   ├── llm/
│   │   └── ollama_client.py   # Persistent httpx.Client, prompt templates
│   ├── config.py              # Pydantic-settings config (env-driven)
│   └── main.py                # FastAPI app + CORS
├── tests/
│   ├── test_pipeline.py       # Pipeline unit + end-to-end integration tests
│   └── test_kg.py             # SPARQL query builder tests
├── pytest.ini                 # asyncio_mode = auto
├── requirements.txt
└── Makefile                   # make run, make test
```

## Quick Start

### 1. Environment Setup

```bash
# Install pyenv and pyenv-virtualenv if not already installed:
# https://github.com/pyenv/pyenv#installation
# https://github.com/pyenv/pyenv-virtualenv#installation

pyenv install 3.13.6
pyenv virtualenv 3.13.6 scigraph-rag-3.13
pyenv local scigraph-rag-3.13

pip install -r requirements.txt
cp .env.example .env
```

### 2. Start Ollama

```bash
ollama serve
ollama pull llama3
```

### 3. Run the API

```bash
make run   # uvicorn on http://localhost:8000
```

### 4. Ask a question

```bash
curl -s -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Which papers compare CNN and SVM on image classification?"}' \
  | python -m json.tool
```

## API

### `POST /ask`

```json
{
  "question": "Which papers use BERT for NER?"
}
```

Response includes:
- `query_type` — detected query type
- `entities` — extracted methods, datasets, tasks, fields, metrics
- `sparql_queries` — all SPARQL queries executed
- `strategies_used` — retrieval strategies that ran
- `answer` — LLM-generated answer grounded in KG context
- `sources` — cited papers (title, URI, DOI, year, methods, datasets)
- `kg_results_count` — total KG rows retrieved

### `GET /health`

```json
{
  "llm":      {"status": "ok", "model": "llama3"},
  "sparql":   {"status": "ok"},
  "pipeline": "ready"
}
```

## Configuration

All settings are in `.env` (loaded by `pydantic-settings`):

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API URL |
| `OLLAMA_MODEL` | `llama3` | Model to use |
| `OLLAMA_TIMEOUT` | `120` | LLM request timeout (seconds) |
| `SPARQL_ENDPOINT` | `https://orkg.org/triplestore` | ORKG SPARQL endpoint |
| `SPARQL_TIMEOUT` | `10` | SPARQL query timeout (seconds) |
| `MAX_CONTEXT_PAPERS` | `8` | Max papers fed to LLM |
| `CORS_ORIGINS` | `http://localhost:3000,...` | Allowed CORS origins |

## Why GraphRAG vs Vector RAG?

| | Vector RAG | This project (KG-RAG) |
|---|---|---|
| Retrieval | Approximate (cosine similarity) | Exact (SPARQL triples) |
| Hallucination risk | Medium–high | Low (cites exact KG nodes) |
| Reasoning | Text context only | Graph traversal (multi-hop) |
| Traceability | Text chunk | Paper URI + DOI |
| Index maintenance | Re-embed on update | ORKG is live |

## Running Tests

```bash
make test
# or
pytest tests/ -v
```

---

**Author**: David | Built on Arch Linux | **License**: MIT
