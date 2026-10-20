# SCIGRAPH-RAG: Knowledge Graph RAG for Scientific Papers

A **GraphRAG** pipeline that uses the [Open Research Knowledge Graph (ORKG)](https://orkg.org/) to answer questions about scientific papers. Retrieval is deterministic — SPARQL queries against a structured KG, not vector similarity — so every answer cites exact paper URIs and DOIs from the graph.

## Key Features

- **9-stage pipeline**: classify → extract → normalize → retrieve (parallel SPARQL) → rank → hard/soft filter → truncate → context+sources → generate
- **Query classification** (6 types): `topic_search`, `method_comparison`, `dataset_search`, `claim_verification`, `method_usage`, `paper_lookup`
- **Scientific NER**: Extracts methods, datasets, tasks, research fields, and metrics using Llama 3 with few-shot prompting; keyword fallback when NER returns nothing
- **Entity normalization**: 117 method synonyms, 62 dataset synonyms, 28 task synonyms, 11 field synonyms — expands abbreviations like CNN → convolutional neural network, NER → named entity recognition
- **Parallel SPARQL retrieval**: `asyncio.gather()` fires all strategy queries concurrently; 10 s timeout with title-keyword fallback
- **SPARQL injection protection**: `_sanitize()` strips dangerous characters; word-boundary regex (`\bTERM\b`) for short terms (≤ 4 chars) to prevent substring false-positives (e.g. "NER" matching "mineral")
- **Thread-safe query cache**: module-level 256-entry FIFO cache with `threading.Lock()`
- **Heuristic ranking**: +2 method match, +2 dataset match, +1 title keyword; `hard_filter` requires both method and dataset when both are present; `soft_filter` drops score-0 noise
- **Year validation**: rejects ORKG values like "9" or "12" (months stored in year predicates) — only 4-digit years 1900–2099 accepted
- **Fully local LLM**: Ollama (Llama 3) via persistent `httpx.Client` connection pool — zero API costs
- **142 tests** across unit, integration, and end-to-end pipeline tests with `pytest-asyncio`

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
│   │   ├── pipeline.py        # Main orchestrator (9-stage pipeline)
│   │   ├── query_classifier.py
│   │   ├── entity_extractor.py
│   │   ├── entity_normalization.py  # METHOD/DATASET/TASK/FIELD synonyms
│   │   ├── query_builder.py   # Multi-strategy retrieval planner
│   │   ├── ranking.py         # Heuristic scoring + hard/soft filter
│   │   └── context_builder.py # Context + sources in one pass
│   ├── llm/
│   │   ├── base.py            # BaseLLMClient ABC + prompt templates
│   │   ├── factory.py         # Provider factory (resolves LLM_PROVIDER)
│   │   ├── ollama_client.py   # Ollama (local) — persistent httpx.Client
│   │   ├── openai_client.py   # OpenAI + Groq (OpenAI-compatible API)
│   │   ├── anthropic_client.py
│   │   └── gemini_client.py
│   ├── config.py              # Pydantic-settings config (env-driven)
│   └── main.py                # FastAPI app + CORS
├── eval/
│   ├── benchmark.py           # Multi-model benchmark runner (LLM-as-judge)
│   ├── ablation.py            # Ablation study (normalization × hard filter)
│   ├── judge.py               # LLM-as-judge scorer (groundedness/relevance/completeness)
│   ├── rejudge.py             # Retroactively re-score existing results with new judge
│   ├── retrieval_eval.py      # nDCG / recall of retrieval against the graded gold set
│   ├── answer_eval.py         # Gold-based answer metrics (citations, abstention, verdicts)
│   ├── faithfulness.py        # Claim-level faithfulness judge (local qwen3:8b) + kappa check
│   ├── vector_baseline.py     # Vector-RAG and hybrid (RRF) baselines, evaluation only
│   ├── ANSWER_EVAL_REPORT.md  # Current answer-quality results (9 models)
│   ├── RETRIEVAL_REPORT.md    # Retrieval results (dev/test)
│   └── results/               # JSON + CSV output from benchmark and ablation runs
├── data/
│   ├── eval_questions.json    # 50 evaluation questions (6 types, 3 difficulty levels)
│   └── gold/                  # Graded relevance gold set + claim verdicts
├── tests/
│   ├── test_pipeline.py       # Pipeline unit + end-to-end integration tests
│   ├── test_kg.py             # SPARQL query builder + sanitization tests
│   └── test_api.py            # FastAPI endpoint tests
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
| `ORKG_SPARQL_ENDPOINT` | `https://orkg.org/triplestore` | ORKG SPARQL endpoint |
| `SPARQL_TIMEOUT` | `10` | SPARQL query timeout (seconds) |
| `SPARQL_FULLTEXT` | `true` | Use Virtuoso full-text index (`bif:contains`) for label matching |
| `MAX_CONTEXT_PAPERS` | `8` | Max papers fed to LLM |
| `CORS_ORIGINS` | `http://localhost:3000,...` | Allowed CORS origins |

## Evaluation

All evaluation runs locally (Ollama), with optional cloud models via Groq. 50 questions in 6 types, split into dev (odd IDs) and test (even IDs); a graded relevance gold set (0/1/2, 2,633 judgments) and gold verdicts for claim questions are in `data/gold/`.

```bash
# retrieval: nDCG@8 etc. against the gold set (oracle or end-to-end)
python -m eval.retrieval_eval [--split test] [--model ollama:qwen2.5:7b]

# answers: run models, then score without an LLM judge
python -m eval.benchmark --models ollama:qwen2.5:7b --skip-judge
python -m eval.answer_eval --results eval/results/benchmark_<ts>.json [--split test] [--by-type]
python -m eval.faithfulness --results eval/results/benchmark_<ts>.json   # resumes where it stopped

# baselines (evaluation only; the system itself stays pure GraphRAG)
python -m eval.vector_baseline build --dump data/orkg_dump_<date>.nt   # ORKG RDF dump
python -m eval.vector_baseline embed                                   # mxbai-embed-large via Ollama
python -m eval.vector_baseline retrieve [--split test]
python -m eval.vector_baseline answer [--hybrid] --models ollama:qwen2.5:7b
```

Results and methodology: [`eval/RETRIEVAL_REPORT.md`](eval/RETRIEVAL_REPORT.md) and [`eval/ANSWER_EVAL_REPORT.md`](eval/ANSWER_EVAL_REPORT.md).

## GraphRAG vs Vector RAG: measured

Same pipeline, only the choice of the 8 papers differs (`eval/vector_baseline.py`):

| nDCG@8, all 50 questions | Graph (oracle) | Vector (mxbai) |
|---|---|---|
| Dataset search / method comparison / claim verification | **0.82 / 0.66 / 0.66** | 0.69 / 0.54 / 0.56 |
| Method usage / paper lookup / topic search | 0.71 / 0.58 / 0.85 | **0.82 / 0.72 / 0.98** |
| All | 0.72 | 0.73 |

- Overall the two are on par, but they retrieve largely different papers (0.42 of 8 shared). An RRF hybrid run in the pipeline retrieves significantly better than graph retrieval alone (0.746 vs 0.682, end-to-end).
- Answer quality (strict end-to-end score) is equal within the variance between repeated runs; graph retrieval leads on method comparisons and its answers stay slightly closer to the context.
- Graph retrieval's real advantage is traceability: every paper comes from an explicit SPARQL path, and the context shows each value's role (method, baseline, dataset, result).

## Running Tests

```bash
make test
# or
pytest tests/ -v
```

---

**Author**: David | Built on Arch Linux | **License**: MIT
