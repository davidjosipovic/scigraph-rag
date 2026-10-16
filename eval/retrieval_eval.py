"""
Retrieval evaluation against the graded gold set (data/gold/retrieval_gold.json).

Measures what the LLM actually gets to see: the top-K papers left after
ranking, filtering and truncation (RAGPipeline.retrieve, steps 1–7).
No LLM-as-judge is involved, so these numbers do not depend on a judge's
leniency.

Two modes:
  oracle (default)   gold query type + hand-written expected_entities from
                     data/eval_questions.json — isolates the retrieval stage.
  --model P:M        end-to-end: the LLM classifies the question and extracts
                     entities, as in a real run.

Metrics per question, averaged per query type and overall (K = max_context_papers):
  P@K        share of retrieved papers that are relevant (grade >= 1)
  P2@K       share of retrieved papers that are directly relevant (grade 2)
  R@K        share of the grade-2 gold papers that were retrieved
  nDCG@K     graded ranking quality (gains 2 / 1 / 0)
  Hit@K      1 if at least one grade-2 paper was retrieved
  empty      1 if nothing reached the LLM context
Retrieved papers that are not in the gold pool count as non-relevant; their
share is reported as `unjudged` so pool coverage stays visible.

Usage:
    python -m eval.retrieval_eval                      # oracle mode
    python -m eval.retrieval_eval --model groq:openai/gpt-oss-20b
    python -m eval.retrieval_eval --ids Q16 Q18 --show # per-paper detail

Output:
    eval/results/retrieval_<mode>_YYYYMMDD_HHMMSS.json
"""

import argparse
import asyncio
import json
import math
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.config import settings  # noqa: E402
from backend.rag.entity_extractor import ExtractedEntities  # noqa: E402
from backend.rag.query_classifier import QueryType  # noqa: E402

QUESTIONS_FILE = PROJECT_ROOT / "data" / "eval_questions.json"
GOLD_FILE = PROJECT_ROOT / "data" / "gold" / "retrieval_gold.json"
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
METRICS = ("P@K", "P2@K", "R@K", "nDCG@K", "Hit@K", "empty", "unjudged")


def _ranked_papers(rows: list[dict]) -> list[str]:
    """Distinct paper URIs in the order the context builder sees them."""
    seen: list[str] = []
    for r in rows:
        uri = r.get("paper")
        if uri and uri not in seen:
            seen.append(uri)
    return seen


def score(retrieved: list[str], grades: dict[str, int], k: int) -> dict[str, float]:
    top = retrieved[:k]
    g = [grades.get(u, 0) for u in top]
    n_rel2 = sum(1 for v in grades.values() if v == 2)
    ideal = sorted(grades.values(), reverse=True)[:k]
    dcg = sum(x / math.log2(i + 2) for i, x in enumerate(g))
    idcg = sum(x / math.log2(i + 2) for i, x in enumerate(ideal))
    return {
        "P@K": sum(x >= 1 for x in g) / len(top) if top else 0.0,
        "P2@K": sum(x == 2 for x in g) / len(top) if top else 0.0,
        "R@K": sum(x == 2 for x in g) / n_rel2 if n_rel2 else float("nan"),
        "nDCG@K": dcg / idcg if idcg else float("nan"),
        "Hit@K": float(any(x == 2 for x in g)) if n_rel2 else float("nan"),
        "empty": float(not top),
        "unjudged": sum(u not in grades for u in top) / len(top) if top else 0.0,
    }


def _mean(rows: list[dict], key: str) -> float:
    vals = [r[key] for r in rows if not math.isnan(r[key])]
    return sum(vals) / len(vals) if vals else float("nan")


async def evaluate(model: str | None, ids: list[str] | None, k: int) -> list[dict]:
    from backend.llm.factory import create_llm_client
    from backend.rag.pipeline import RAGPipeline

    questions = {q["id"]: q for q in json.load(open(QUESTIONS_FILE))["questions"]}
    gold = json.load(open(GOLD_FILE))["questions"]
    if ids:
        gold = {q: v for q, v in gold.items() if q in ids}

    llm = create_llm_client(*model.split(":", 1)) if model else None
    # The LLM client is only used for classification/extraction in end-to-end
    # mode; oracle mode never calls it, so any configured client will do.
    pipeline = RAGPipeline(llm_client=llm or create_llm_client())

    out = []
    for qid, g in gold.items():
        q = questions[qid]
        if model:
            r = await pipeline.retrieve(q["question"])
        else:
            e = q["expected_entities"]
            r = await pipeline.retrieve(
                q["question"],
                query_type=QueryType(q["query_type"]),
                entities=ExtractedEntities(
                    methods=e.get("methods", []), datasets=e.get("datasets", []),
                    tasks=e.get("tasks", []), fields=e.get("fields", []),
                ),
            )
        grades = {j["uri"]: j["grade"] for j in g["judgments"]}
        titles = {j["uri"]: j["title"] for j in g["judgments"]}
        retrieved = _ranked_papers(r["rows"])
        m = score(retrieved, grades, k)
        out.append({
            "id": qid, "query_type": q["query_type"], "detected_type": r["query_type"].value,
            **m, "retrieved": [{"uri": u, "title": titles.get(u), "grade": grades.get(u)} for u in retrieved[:k]],
        })
        logger.info(f"{qid}: P@K={m['P@K']:.2f} R@K={m['R@K']:.2f} nDCG={m['nDCG@K']:.2f}")
    return out


def summarize(results: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        groups[r["query_type"]].append(r)
    rows = [{"query_type": t, "n": len(rs), **{m: _mean(rs, m) for m in METRICS}} for t, rs in sorted(groups.items())]
    rows.append({"query_type": "ALL", "n": len(results), **{m: _mean(results, m) for m in METRICS}})
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", metavar="PROVIDER:MODEL", help="End-to-end mode with this LLM (default: oracle)")
    p.add_argument("--ids", nargs="+", metavar="QID", help="Only these questions")
    p.add_argument("--k", type=int, default=settings.max_context_papers, help="Cut-off K (default: max_context_papers)")
    p.add_argument("--show", action="store_true", help="Print retrieved papers with their grades")
    args = p.parse_args()

    results = asyncio.run(evaluate(args.model, args.ids, args.k))
    summary = summarize(results)

    if args.show:
        for r in results:
            print(f"\n{r['id']} [{r['query_type']} → {r['detected_type']}]")
            for x in r["retrieved"]:
                print(f"   {x['grade'] if x['grade'] is not None else '?'}  {(x['title'] or x['uri'])[:90]}")
    print(f"\n{'query_type':<20} {'n':>3} " + " ".join(f"{m:>8}" for m in METRICS))
    for s in summary:
        print(f"{s['query_type']:<20} {s['n']:>3} " + " ".join(f"{s[m]:>8.3f}" for m in METRICS))

    mode = args.model.replace(":", "_").replace("/", "_") if args.model else "oracle"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"retrieval_{mode}_{datetime.now():%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps({"mode": mode, "k": args.k, "summary": summary, "questions": results}, indent=1, ensure_ascii=False))
    print(f"\nSaved → {path}")


if __name__ == "__main__":
    main()
