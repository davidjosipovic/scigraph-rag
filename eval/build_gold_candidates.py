"""
Pool candidate ORKG papers for the retrieval gold set.

For every evaluation question, candidates are collected from several
independent sources so that the gold set is not biased towards the
pipeline under test (TREC-style pooling):

  pipeline   current RAG retrieval + ranking, run with the hand-written
             expected_entities (top papers, in the order the LLM would see them)
  claim      claim_evidence over the hand-written search terms
             (contributions ranked by how many terms they cover)
  title      title_keyword_search over the search terms
  entity     broad_entity_search per search term (any contribution value)
  lookup     paper_lookup_by_title for the exact titles of lookup questions

Every candidate is then enriched with its full ORKG contribution record so it
can be judged without leaving the file.

Usage:
    python -m eval.build_gold_candidates                # all questions
    python -m eval.build_gold_candidates --ids Q18 Q43  # subset

Output:
    data/gold/candidates.json
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.kg import queries as Q  # noqa: E402
from backend.kg.sparql_client import SPARQLClient  # noqa: E402
from backend.rag.entity_extractor import ExtractedEntities  # noqa: E402
from backend.rag.entity_normalization import expand_entities  # noqa: E402
from backend.rag.query_builder import retrieve_async  # noqa: E402
from backend.rag.query_classifier import QueryType  # noqa: E402
from backend.rag.ranking import hard_filter, rank_results, soft_filter  # noqa: E402

QUESTIONS_FILE = PROJECT_ROOT / "data" / "eval_questions.json"
SPEC_FILE = PROJECT_ROOT / "data" / "gold" / "search_spec.json"
OUTPUT_FILE = PROJECT_ROOT / "data" / "gold" / "candidates.json"

PIPELINE_TOP = 10      # pipeline papers kept (the LLM sees the top 8)
SOURCE_LIMIT = 10      # papers per non-pipeline source
MAX_CANDIDATES = 30    # pool cap per question
MAX_VALUE_LEN = 160    # long literals (abstracts) are truncated for reading
MAX_PAIRS = 60         # contribution (predicate, value) pairs kept per paper


def _papers(rows: list[dict]) -> list[dict]:
    """Distinct papers from result rows, in first-seen order."""
    seen: dict[str, dict] = {}
    for r in rows:
        uri = r.get("paper")
        if uri and uri not in seen:
            seen[uri] = {"uri": uri, "title": r.get("title", ""), "year": r.get("year")}
    return list(seen.values())


async def _run(client: SPARQLClient, sem: asyncio.Semaphore, query: str) -> list[dict]:
    if not query:
        return []
    async with sem:
        try:
            return await client.execute_async(query)
        except Exception as e:  # one failing source must not sink the question
            logger.warning(f"query failed: {e}")
            return []


async def pool_question(q: dict, spec: dict, client: SPARQLClient, sem: asyncio.Semaphore) -> dict:
    terms = spec["terms"]
    e = q["expected_entities"]
    expanded = expand_entities(ExtractedEntities(
        methods=e.get("methods", []), datasets=e.get("datasets", []),
        tasks=e.get("tasks", []), fields=e.get("fields", []),
    ))

    # Current pipeline retrieval, ranked and filtered exactly as in RAGPipeline
    async with sem:
        retrieval = await retrieve_async(q["question"], QueryType(q["query_type"]), expanded, client)
    ranked = soft_filter(hard_filter(rank_results(retrieval.results, expanded), expanded), min_score=1)
    sources: dict[str, list[dict]] = {"pipeline": _papers(ranked)[:PIPELINE_TOP]}

    jobs = {"claim": Q.claim_evidence(terms, SOURCE_LIMIT), "title": Q.title_keyword_search(terms, SOURCE_LIMIT)}
    for t in terms[:3]:
        jobs[f"entity:{t}"] = Q.broad_entity_search(t, 5)
    for t in spec.get("titles", []):
        jobs[f"lookup:{t}"] = Q.paper_lookup_by_title(t, 3)
    results = await asyncio.gather(*(_run(client, sem, qq) for qq in jobs.values()))
    for name, rows in zip(jobs, results):
        sources[name] = _papers(rows)[:SOURCE_LIMIT]

    # Merge: lookups first (the exact paper must be judged), then pipeline order
    order = [k for k in sources if k.startswith("lookup")] + ["pipeline"] + \
            [k for k in sources if not k.startswith("lookup") and k != "pipeline"]
    pool: dict[str, dict] = {}
    for name in order:
        for rank, p in enumerate(sources[name], 1):
            c = pool.setdefault(p["uri"], {**p, "found_by": [], "pipeline_rank": None})
            c["found_by"].append(name.split(":")[0])
            if name == "pipeline":
                c["pipeline_rank"] = rank
    candidates = list(pool.values())[:MAX_CANDIDATES]
    for c in candidates:
        c["found_by"] = sorted(set(c["found_by"]))

    # Enrich with the full contribution record
    details = await asyncio.gather(*(_run(client, sem, Q.paper_full_contributions(c["uri"], 150)) for c in candidates))
    for c, rows in zip(candidates, details):
        contribs: dict[str, list[list[str]]] = {}
        n = 0
        for r in rows:
            pred, val = r.get("predLabel", ""), r.get("valueLabel", "")
            if not pred or not val or val.startswith("http") or n >= MAX_PAIRS:
                continue
            pair = [pred, val if len(val) <= MAX_VALUE_LEN else val[:MAX_VALUE_LEN] + "…"]
            lst = contribs.setdefault(r.get("contribLabel", "Contribution"), [])
            if pair not in lst:
                lst.append(pair)
                n += 1
        c["contributions"] = contribs

    logger.info(f"{q['id']}: {len(candidates)} candidates "
                f"({len(sources['pipeline'])} from pipeline, {retrieval.total_results} raw rows)")
    return {
        "question": q["question"], "query_type": q["query_type"], "difficulty": q["difficulty"],
        "criterion": spec["criterion"], "candidates": candidates,
    }


async def main_async(ids: list[str] | None) -> None:
    questions = json.load(open(QUESTIONS_FILE))["questions"]
    spec = json.load(open(SPEC_FILE))["questions"]
    if ids:
        questions = [q for q in questions if q["id"] in ids]
    client = SPARQLClient()
    sem = asyncio.Semaphore(4)  # stay polite to the public ORKG endpoint

    out = json.load(open(OUTPUT_FILE)) if (ids and OUTPUT_FILE.exists()) else {}
    for q in questions:  # sequential per question; queries inside run in parallel
        out[q["id"]] = await pool_question(q, spec[q["id"]], client, sem)
    out = dict(sorted(out.items()))
    OUTPUT_FILE.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    logger.info(f"Wrote {sum(len(v['candidates']) for v in out.values())} candidates → {OUTPUT_FILE}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ids", nargs="+", metavar="QID", help="Only (re)build these questions")
    asyncio.run(main_async(p.parse_args().ids))


if __name__ == "__main__":
    main()
