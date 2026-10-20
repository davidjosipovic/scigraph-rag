"""
Vector-RAG baseline: the same pipeline, with graph retrieval swapped for
dense similarity search. Used only for comparison in the evaluation — the
system itself stays pure GraphRAG.

What changes and what stays the same:

  GraphRAG (backend/rag)          VectorPipeline (here)
  ─────────────────────────       ─────────────────────────
  classify + extract entities     same (same LLM)
  SPARQL strategies, ranking,     cosine similarity between the question
  filtering, enrichment           and one embedded text card per paper
  → top 8 papers                  → top 8 papers
  context from the papers'        same: full records fetched live
  ORKG records (detail rows)      (papers_details), same row selection
  LLM answer                      same prompt and model

So only the choice of the 8 papers differs; the context format, the
records shown and the answering prompt are identical.

Index: every ORKG paper in the public RDF dump (one card per paper: title,
research field, and the labels of its contributions' values, research
problems first), embedded with mxbai-embed-large via Ollama.

Usage:
    python -m eval.vector_baseline build  --dump data/orkg_dump_20261018.nt
    python -m eval.vector_baseline embed
    python -m eval.vector_baseline retrieve [--split test]     # nDCG vs gold
    python -m eval.vector_baseline answer --models ollama:qwen2.5:7b groq:openai/gpt-oss-20b
    python -m eval.vector_baseline answer --hybrid --models ollama:qwen2.5:7b   # graph + vector, RRF

Another embedding model: VECTOR_EMBED_MODEL=nomic-embed-text (see EMBEDDERS).
"""

import argparse
import asyncio
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import httpx
import numpy as np
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import settings  # noqa: E402
from backend.rag.pipeline import RAGPipeline  # noqa: E402

INDEX_DIR = PROJECT_ROOT / "data" / "vector"
CARDS_FILE = INDEX_DIR / "cards.jsonl"
# Embedding models: (query prefix, document prefix, dimension, file). Each
# model is used with the instruction prefixes it was trained with.
EMBEDDERS = {
    "mxbai-embed-large": ("Represent this sentence for searching relevant passages: ", "", 1024, "embeddings.npy"),
    "nomic-embed-text": ("search_query: ", "search_document: ", 768, "embeddings_nomic.npy"),
}
EMBED_MODEL = os.environ.get("VECTOR_EMBED_MODEL", "mxbai-embed-large")
QUERY_PREFIX, DOC_PREFIX, EMB_DIM, _emb_name = EMBEDDERS[EMBED_MODEL]
EMB_FILE = INDEX_DIR / _emb_name
CARD_CHARS = 1500  # ~400 tokens, inside the model's 512-token window

ORKG = "http://orkg.org/orkg/"
LABEL = "<http://www.w3.org/2000/01/rdf-schema#label>"
TYPE = "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>"
PAPER = f"<{ORKG}class/Paper>"
P_CONTRIB, P_FIELD, P_PROBLEM = (f"<{ORKG}predicate/{p}>" for p in ("P31", "P30", "P32"))
P_BENCH = f"<{ORKG}predicate/PWC_HAS_BENCHMARK>"
SKIP_PREDS = {TYPE, LABEL}

_NT = re.compile(r'^(<[^>]+>|_:\S+) (<[^>]+>) (.+) \.\s*$')
_LIT = re.compile(r'^"(.*)"(?:\^\^<[^>]+>|@[\w-]+)?$', re.DOTALL)


def _literal(obj: str) -> str | None:
    m = _LIT.match(obj)
    if not m:
        return None
    return m.group(1).encode().decode("unicode_escape", "ignore").encode("latin-1", "ignore").decode("utf-8", "ignore")


def _triples(path: Path):
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = _NT.match(line)
            if m:
                yield m.groups()


# ── build: one text card per paper ───────────────────────────────

def build(dump: Path) -> None:
    labels: dict[str, str] = {}
    papers: set[str] = set()
    contribs: dict[str, list[str]] = defaultdict(list)   # paper → contributions
    field: dict[str, str] = {}
    logger.info("pass 1: labels, papers, contributions")
    for s, p, o in _triples(dump):
        if p == LABEL:
            lit = _literal(o)
            if lit:
                labels[s] = lit
        elif p == TYPE and o == PAPER:
            papers.add(s)
        elif p == P_CONTRIB:
            contribs[s].append(o)
        elif p == P_FIELD:
            field[s] = o
    contrib_set = {c for p in papers for c in contribs.get(p, [])}
    logger.info(f"{len(papers)} papers, {len(contrib_set)} contributions, {len(labels)} labels")

    logger.info("pass 2: contribution values")
    values: dict[str, list[tuple[str, str]]] = defaultdict(list)  # contribution → (pred, value)
    for s, p, o in _triples(dump):
        if s not in contrib_set or p in SKIP_PREDS:
            continue
        val = _literal(o) if o.startswith('"') else labels.get(o)
        if val and not val.startswith("http"):
            values[s].append((p, val))

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(CARDS_FILE, "w") as out:
        for paper in sorted(papers):
            title = labels.get(paper)
            if not title:
                continue
            problems, other, seen = [], [], set()
            for c in contribs.get(paper, []):
                for p, v in values.get(c, []):
                    key = v.lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    if p == P_PROBLEM:
                        problems.append(v)
                    else:
                        pred = "benchmark" if p == P_BENCH else labels.get(p, p.rsplit("/", 1)[-1].strip(">"))
                        other.append(f"{pred}: {v}")
            parts = [title]
            if paper in field and field[paper] in labels:
                parts.append(f"Research field: {labels[field[paper]]}")
            if problems:
                parts.append("Research problem: " + "; ".join(problems))
            parts += other
            text = "\n".join(parts)[:CARD_CHARS]
            out.write(json.dumps({"uri": paper.strip("<>"), "title": title, "text": text}, ensure_ascii=False) + "\n")
            n += 1
    logger.info(f"{n} cards → {CARDS_FILE}")


# ── embed ────────────────────────────────────────────────────────

def _embed(texts: list[str], client: httpx.Client) -> np.ndarray:
    r = client.post(f"{settings.ollama_base_url}/api/embed",
                    json={"model": EMBED_MODEL, "input": texts, "truncate": True})
    if r.status_code == 400 and len(texts) > 1:
        # One over-long card can fail the whole batch: retry one by one.
        return np.vstack([_embed([t], client) for t in texts])
    if r.status_code == 400:
        # Token-dense text (formulas, IDs) can exceed the window despite the
        # character cap; halve it until it fits.
        logger.warning(f"embed 400 ({r.text[:80]}), shortening card of {len(texts[0])} chars")
        return _embed([texts[0][: len(texts[0]) // 2]], client)
    r.raise_for_status()
    v = np.asarray(r.json()["embeddings"], dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def embed(batch: int = 64) -> None:
    cards = [json.loads(line) for line in open(CARDS_FILE)]
    out = np.zeros((len(cards), EMB_DIM), dtype=np.float16)
    with httpx.Client(timeout=300) as client:
        for i in range(0, len(cards), batch):
            out[i:i + batch] = _embed([DOC_PREFIX + c["text"] for c in cards[i:i + batch]], client)
            if i // batch % 50 == 0:
                logger.info(f"{i + batch}/{len(cards)}")
    np.save(EMB_FILE, out)
    logger.info(f"embeddings {out.shape} → {EMB_FILE}")


class VectorIndex:
    def __init__(self) -> None:
        self.cards = [json.loads(line) for line in open(CARDS_FILE)]
        self.emb = np.load(EMB_FILE).astype(np.float32)
        self.client = httpx.Client(timeout=120)

    def search(self, question: str, k: int) -> list[str]:
        q = _embed([QUERY_PREFIX + question], self.client)[0]
        top = np.argsort(-(self.emb @ q))[:k]
        return [self.cards[i]["uri"] for i in top]


# ── the baseline pipeline ────────────────────────────────────────

class VectorPipeline(RAGPipeline):
    """RAGPipeline with steps 4–7 replaced by vector search (see module doc)."""

    index: VectorIndex | None = None

    async def retrieve(self, question, query_type=None, entities=None):
        from functools import partial

        from backend.rag.entity_extractor import ExtractedEntities, extract_entities, extract_keywords
        from backend.rag.entity_normalization import expand_entities
        from backend.rag.query_builder import RetrievalResult
        from backend.rag.query_classifier import classify_query

        if VectorPipeline.index is None:
            VectorPipeline.index = VectorIndex()
        # Steps 1–3 as in GraphRAG: the query type picks the answer prompt and
        # the entities steer which record rows are shown, exactly as there.
        loop = asyncio.get_running_loop()
        if query_type is None:
            query_type = await loop.run_in_executor(None, partial(classify_query, question, self.llm))
        if entities is None:
            entities = await loop.run_in_executor(None, partial(extract_entities, question, self.llm))
        if entities.is_empty() and (keywords := extract_keywords(question)):
            entities = ExtractedEntities(methods=keywords)
        base = {"query_type": query_type, "entities": entities}
        expanded = expand_entities(entities)

        uris = self.index.search(question, settings.max_context_papers)
        rows = await self._record_rows(uris, expanded)
        rank = {u: i for i, u in enumerate(uris)}
        rows.sort(key=lambda r: rank.get(r.get("paper"), len(uris)))

        retrieval = RetrievalResult()
        retrieval.add(rows, f"vector top-{len(uris)}", "vector")
        return {"query_type": base["query_type"], "entities": base["entities"],
                "retrieval": retrieval, "rows": rows}

    async def _record_rows(self, uris, expanded):
        """Context rows for papers found without graph queries: the key rows of their full record."""
        from backend.rag.ranking import select_detail_rows

        details = await self._fetch_details(uris)
        rows = select_detail_rows(details, expanded, settings.max_detail_pairs)
        # A paper whose record has no key rows still reaches the context by title.
        present = {r["paper"] for r in rows}
        rows += [next(r for r in details if r["paper"] == u) | {"predLabel": "", "valueLabel": ""}
                 for u in uris if u not in present and any(r["paper"] == u for r in details)]
        return rows


RRF_K = 60


def rrf(rankings: list[list[str]], k: int) -> list[str]:
    """Reciprocal rank fusion (Cormack et al., 2009) of ranked URI lists, top k."""
    score: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for r, uri in enumerate(ranking, start=1):
            score[uri] += 1 / (RRF_K + r)
    first: dict[str, int] = {}
    for u in (u for ranking in rankings for u in ranking):
        first.setdefault(u, len(first))  # ties go to the earlier appearance
    return sorted(score, key=lambda u: (-score[u], first[u]))[:k]


class HybridPipeline(VectorPipeline):
    """
    Graph and vector retrieval fused by RRF; evaluation only.

    Both retrievers return their top 8 papers, RRF picks 8 of the union.
    Papers from the graph keep exactly the rows graph retrieval gave them;
    papers found only by the vector index get their record rows as in
    VectorPipeline. Classification and entities are those of the graph run.
    """

    async def retrieve(self, question, query_type=None, entities=None):
        from backend.rag.entity_normalization import expand_entities
        from backend.rag.query_builder import RetrievalResult

        if VectorPipeline.index is None:
            VectorPipeline.index = VectorIndex()
        k = settings.max_context_papers
        graph = await RAGPipeline.retrieve(self, question, query_type, entities)
        graph_uris = list(dict.fromkeys(r["paper"] for r in graph["rows"] if r.get("paper")))
        vector_uris = self.index.search(question, k)
        fused = rrf([graph_uris, vector_uris], k)

        rows = [r for r in graph["rows"] if r.get("paper") in fused]
        vector_only = [u for u in fused if u not in graph_uris]
        if vector_only:
            rows += await self._record_rows(vector_only, expand_entities(graph["entities"]))
        rank = {u: i for i, u in enumerate(fused)}
        rows.sort(key=lambda r: rank.get(r.get("paper"), k))

        retrieval = graph["retrieval"]
        retrieval.add([], f"rrf(graph top-{len(graph_uris)}, vector top-{k})", "hybrid_rrf")
        return {"query_type": graph["query_type"], "entities": graph["entities"],
                "retrieval": retrieval, "rows": rows}


# ── retrieve: nDCG against the gold set ──────────────────────────

def retrieve_eval(split: str | None, k: int) -> None:
    from eval.retrieval_eval import GOLD_FILE, QUESTIONS_FILE, RESULTS_DIR, in_split, score, summarize, METRICS

    index = VectorIndex()
    questions = {q["id"]: q for q in json.load(open(QUESTIONS_FILE))["questions"]}
    gold = {q: v for q, v in json.load(open(GOLD_FILE))["questions"].items() if in_split(q, split)}
    titles = {c["uri"]: c["title"] for c in index.cards}
    results = []
    for qid, g in gold.items():
        grades = {j["uri"]: j["grade"] for j in g["judgments"]}
        uris = index.search(questions[qid]["question"], k)
        results.append({"id": qid, "query_type": questions[qid]["query_type"], **score(uris, grades, k),
                        "retrieved": [{"uri": u, "title": titles.get(u), "grade": grades.get(u)} for u in uris]})
    summary = summarize(results)
    print(f"\n{'query_type':<20} {'n':>3} " + " ".join(f"{m:>8}" for m in METRICS))
    for s in summary:
        print(f"{s['query_type']:<20} {s['n']:>3} " + " ".join(f"{s[m]:>8.3f}" for m in METRICS))
    mode = "vector" + ("" if EMBED_MODEL == "mxbai-embed-large" else "_" + EMBED_MODEL.split("-")[0]) + (f"_{split}" if split else "")
    path = RESULTS_DIR / f"retrieval_{mode}_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps({"mode": mode, "k": k, "split": split, "summary": summary, "questions": results},
                               indent=1, ensure_ascii=False))
    print(f"\nSaved → {path}")


# ── answer: full runs in the benchmark format ────────────────────

async def answer(models: list[str], delay: float, only: list[str] | None = None,
                 hybrid: bool = False) -> None:
    from eval.benchmark import QUESTIONS_FILE, run_model, load_questions

    questions = load_questions(QUESTIONS_FILE)
    if only:
        questions = [q for q in questions if q["id"] in only]
    results = []
    for spec in models:
        provider, model = spec.split(":", 1)
        results += await run_model(provider, model, questions,
                                   delay=delay if provider != "ollama" else 0.0,
                                   pipeline_cls=HybridPipeline if hybrid else VectorPipeline)
    name = "hybrid" if hybrid else "vector"
    path = PROJECT_ROOT / "eval" / "results" / f"benchmark_{name}_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps([asdict(r) for r in results], indent=2, ensure_ascii=False))
    print(f"Saved → {path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--dump", type=Path, required=True)
    sub.add_parser("embed")
    r = sub.add_parser("retrieve")
    r.add_argument("--split", choices=("dev", "test"))
    r.add_argument("--k", type=int, default=settings.max_context_papers)
    a = sub.add_parser("answer")
    a.add_argument("--models", nargs="+", required=True, metavar="PROVIDER:MODEL")
    a.add_argument("--delay", type=float, default=30.0, help="pause between questions for cloud models")
    a.add_argument("--hybrid", action="store_true", help="RRF of graph and vector retrieval instead of vector only")
    a.add_argument("--questions", nargs="+", metavar="QID", help="only these question ids (e.g. to re-run failures)")
    args = p.parse_args()

    if args.cmd == "build":
        build(args.dump)
    elif args.cmd == "embed":
        embed()
    elif args.cmd == "retrieve":
        retrieve_eval(args.split, args.k)
    else:
        asyncio.run(answer(args.models, args.delay, args.questions, args.hybrid))


if __name__ == "__main__":
    main()
