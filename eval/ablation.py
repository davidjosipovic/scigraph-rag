"""
Ablation study: impact of entity normalization and the hard filter on retrieval.

Runs the full eval question set through the pipeline under four configurations
(normalization on/off × hard filter on/off) using a single fixed LLM (Ollama,
local and free — no judge scoring, no API cost) and compares retrieval-level
metrics: KG results returned, sources kept after filtering, and latency.

This isolates the effect of two pipeline stages (Step 3 normalization, Step 6
hard filter) on retrieval recall/precision, independent of LLM answer quality.

Usage:
    python -m eval.ablation
    python -m eval.ablation --limit 10 --model llama3
"""

import argparse
import asyncio
import csv
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

QUESTIONS_FILE = PROJECT_ROOT / "data" / "eval_questions.json"
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"

CONFIGS = [
    ("norm_on_filter_on", True, True),
    ("norm_off_filter_on", False, True),
    ("norm_on_filter_off", True, False),
    ("norm_off_filter_off", False, False),
]


@dataclass
class AblationResult:
    config: str
    question_id: str
    query_type: str
    latency_seconds: float
    kg_results_count: int
    sources_count: int
    error: str | None


async def run_config(
    config_name: str,
    enable_normalization: bool,
    enable_hard_filter: bool,
    questions: list[dict],
    model: str,
) -> list[AblationResult]:
    from backend.llm.factory import create_llm_client
    from backend.rag.pipeline import RAGPipeline

    logger.info(f"=== Config: {config_name} (norm={enable_normalization}, hard_filter={enable_hard_filter}) ===")

    llm_client = create_llm_client(provider="ollama", model=model)
    pipeline = RAGPipeline(
        llm_client=llm_client,
        enable_normalization=enable_normalization,
        enable_hard_filter=enable_hard_filter,
    )

    results: list[AblationResult] = []
    for i, q in enumerate(questions, 1):
        logger.info(f"[{config_name}] ({i}/{len(questions)}) {q['id']}: {q['question'][:70]}...")
        t0 = time.perf_counter()
        error: str | None = None
        result: dict = {}
        try:
            result = await pipeline.ask(q["question"])
        except Exception as e:
            error = str(e)
            logger.error(f"[{config_name}] {q['id']} FAILED: {e}")
        latency = round(time.perf_counter() - t0, 3)

        results.append(AblationResult(
            config=config_name,
            question_id=q["id"],
            query_type=q["query_type"],
            latency_seconds=latency,
            kg_results_count=result.get("kg_results_count", 0),
            sources_count=len(result.get("sources", [])),
            error=error,
        ))
    return results


def compute_summary(results: list[AblationResult]) -> list[dict]:
    from collections import defaultdict

    groups: dict[str, list[AblationResult]] = defaultdict(list)
    for r in results:
        groups[r.config].append(r)

    rows = []
    for config_name, _, _ in CONFIGS:
        group = groups.get(config_name, [])
        ok = [r for r in group if not r.error]

        def avg(vals: list) -> str:
            return f"{sum(vals) / len(vals):.3f}" if vals else "N/A"

        rows.append({
            "config": config_name,
            "questions_total": len(group),
            "questions_failed": len(group) - len(ok),
            "avg_latency_s": avg([r.latency_seconds for r in ok]),
            "avg_kg_results": avg([r.kg_results_count for r in ok]),
            "avg_sources": avg([r.sources_count for r in ok]),
        })
    return rows


def print_summary(rows: list[dict]) -> None:
    print("\n" + "=" * 80)
    print("ABLATION SUMMARY (retrieval-level — no judge, no LLM-quality scoring)")
    print("=" * 80)
    header = f"{'Config':<22} {'Latency':>9} {'KG Results':>11} {'Sources':>9} {'Failed':>7}"
    print(header)
    print("-" * 80)
    for row in rows:
        print(
            f"{row['config']:<22} "
            f"{row['avg_latency_s']:>9} "
            f"{row['avg_kg_results']:>11} "
            f"{row['avg_sources']:>9} "
            f"{row['questions_failed']:>7}"
        )
    print("=" * 80 + "\n")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="llama3", help="Ollama model to use (default: llama3)")
    p.add_argument("--limit", type=int, default=None, metavar="N", help="Run only the first N questions")
    p.add_argument("--query-types", nargs="+", metavar="TYPE", help="Filter questions by query type")
    return p


async def main_async(args: argparse.Namespace) -> None:
    with open(QUESTIONS_FILE) as f:
        data = json.load(f)
    questions = data["questions"]
    if args.query_types:
        questions = [q for q in questions if q["query_type"] in args.query_types]
    if args.limit:
        questions = questions[: args.limit]
    logger.info(f"Loaded {len(questions)} questions for ablation")

    all_results: list[AblationResult] = []
    for config_name, norm, hard_filter_on in CONFIGS:
        all_results.extend(
            await run_config(config_name, norm, hard_filter_on, questions, args.model)
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    json_path = RESULTS_DIR / f"ablation_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)

    summary = compute_summary(all_results)
    csv_path = RESULTS_DIR / f"ablation_{timestamp}_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    print_summary(summary)
    print(f"Results saved:\n  {json_path}\n  {csv_path}")


def main() -> None:
    args = build_parser().parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
