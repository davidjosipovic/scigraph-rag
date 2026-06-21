"""
GraphRAG evaluation benchmark.

Runs the full pipeline (classify → extract → retrieve → generate) for each
configured model over the evaluation question set, measures latency, and scores
answer quality with an LLM-as-judge.

Usage examples:
    # Ollama only (no API keys needed):
    python -m eval.benchmark --models ollama:llama3

    # Compare three models:
    python -m eval.benchmark \\
        --models ollama:llama3 openai:gpt-4o-mini anthropic:claude-haiku-4-5-20251001 \\
        --judge openai:gpt-4o

    # Quick smoke test (5 questions, skip judge):
    python -m eval.benchmark --models ollama:llama3 --limit 5 --skip-judge

    # Only method_comparison questions:
    python -m eval.benchmark --models ollama:llama3 --query-types method_comparison dataset_search

Output:
    eval/results/benchmark_YYYYMMDD_HHMMSS.json        full per-question results
    eval/results/benchmark_YYYYMMDD_HHMMSS_summary.csv per-model summary table
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
from typing import Any

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

QUESTIONS_FILE = PROJECT_ROOT / "data" / "eval_questions.json"
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"


# ── Data model ────────────────────────────────────────────────────

@dataclass
class QuestionResult:
    """All data collected for one question × one model run."""

    question_id: str
    question: str
    query_type: str
    difficulty: str
    provider: str
    model: str
    latency_seconds: float
    answer: str
    kg_results_count: int
    sources_count: int
    sources: list[dict]
    query_type_detected: str
    entities_methods: list[str]
    entities_datasets: list[str]
    error: str | None
    # Filled by judge (None when --skip-judge or judge fails)
    groundedness: float | None
    relevance: float | None
    completeness: float | None
    overall_score: float | None
    judge_reasoning: str | None


# ── Question loading ──────────────────────────────────────────────

def load_questions(
    path: Path,
    limit: int | None = None,
    query_types: list[str] | None = None,
) -> list[dict]:
    with open(path) as f:
        data = json.load(f)
    questions: list[dict] = data["questions"]
    if query_types:
        questions = [q for q in questions if q["query_type"] in query_types]
    if limit:
        questions = questions[:limit]
    logger.info(f"Loaded {len(questions)} questions from {path.name}")
    return questions


# ── Pipeline runner ───────────────────────────────────────────────

async def run_model(
    provider: str,
    model: str,
    questions: list[dict],
    delay: float = 0.0,
) -> list[QuestionResult]:
    """Run all questions through the pipeline for one provider:model pair."""
    from backend.llm.factory import create_llm_client
    from backend.rag.pipeline import RAGPipeline

    label = f"{provider}:{model}"
    logger.info(f"=== Starting model: {label} ({len(questions)} questions) ===")

    llm_client = create_llm_client(provider=provider, model=model)
    pipeline = RAGPipeline(llm_client=llm_client)

    results: list[QuestionResult] = []

    for i, q in enumerate(questions, 1):
        if delay > 0 and i > 1:
            logger.info(f"[{label}] Waiting {delay}s before next question (rate limit delay)...")
            await asyncio.sleep(delay)

        logger.info(f"[{label}] ({i}/{len(questions)}) {q['id']}: {q['question'][:70]}...")

        t0 = time.perf_counter()
        error: str | None = None
        pipeline_result: dict[str, Any] = {}

        try:
            pipeline_result = await pipeline.ask(q["question"])
        except Exception as e:
            error = str(e)
            logger.error(f"[{label}] {q['id']} FAILED: {e}")

        latency = round(time.perf_counter() - t0, 3)
        entities = pipeline_result.get("entities", {})

        raw_sources = pipeline_result.get("sources", [])
        results.append(QuestionResult(
            question_id=q["id"],
            question=q["question"],
            query_type=q["query_type"],
            difficulty=q["difficulty"],
            provider=provider,
            model=model,
            latency_seconds=latency,
            answer=pipeline_result.get("answer", ""),
            kg_results_count=pipeline_result.get("kg_results_count", 0),
            sources_count=len(raw_sources),
            sources=raw_sources,
            query_type_detected=pipeline_result.get("query_type", ""),
            entities_methods=entities.get("methods", []),
            entities_datasets=entities.get("datasets", []),
            error=error,
            groundedness=None,
            relevance=None,
            completeness=None,
            overall_score=None,
            judge_reasoning=None,
        ))

        logger.info(
            f"[{label}] {q['id']} done — "
            f"latency={latency}s, kg_results={pipeline_result.get('kg_results_count', 0)}, "
            f"sources={len(pipeline_result.get('sources', []))}"
        )

    return results


# ── Judge scoring ─────────────────────────────────────────────────

def apply_judge(
    results: list[QuestionResult],
    questions_by_id: dict[str, dict],
    judge_provider: str,
    judge_model: str,
    delay: float = 0.0,
) -> None:
    """Score all results in-place using the LLM judge."""
    import time
    from backend.llm.factory import create_llm_client
    from eval.judge import score_answer

    judge_label = f"{judge_provider}:{judge_model}"
    logger.info(f"=== Scoring {len(results)} answers with judge: {judge_label} (delay={delay}s) ===")

    judge_client = create_llm_client(provider=judge_provider, model=judge_model)
    total = len(results)

    for i, r in enumerate(results, 1):
        if delay > 0 and i > 1:
            time.sleep(delay)
        if r.error:
            logger.warning(f"Skipping judge for {r.question_id} ({r.provider}:{r.model}) — pipeline error")
            r.judge_reasoning = "skipped: pipeline error"
            continue

        q = questions_by_id[r.question_id]
        logger.info(f"[judge] ({i}/{total}) {r.question_id} × {r.provider}:{r.model}")

        scores = score_answer(
            question=r.question,
            answer=r.answer,
            sources=r.sources,
            answer_hints=q.get("answer_hints", []),
            judge_client=judge_client,
        )
        r.groundedness = scores["groundedness"]
        r.relevance = scores["relevance"]
        r.completeness = scores["completeness"]
        r.overall_score = scores["overall"]
        r.judge_reasoning = scores["reasoning"]


# ── Output ────────────────────────────────────────────────────────

def save_results(results: list[QuestionResult], timestamp: str) -> tuple[Path, Path]:
    """Save full JSON results and CSV summary."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    json_path = RESULTS_DIR / f"benchmark_{timestamp}.json"
    csv_path = RESULTS_DIR / f"benchmark_{timestamp}_summary.csv"

    # Full results JSON
    with open(json_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    logger.info(f"Full results → {json_path}")

    # Per-model summary CSV
    summary = _compute_summary(results)
    with open(csv_path, "w", newline="") as f:
        if summary:
            writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            writer.writeheader()
            writer.writerows(summary)
    logger.info(f"Summary CSV → {csv_path}")

    return json_path, csv_path


def _compute_summary(results: list[QuestionResult]) -> list[dict]:
    """Aggregate per-model statistics."""
    from collections import defaultdict

    groups: dict[str, list[QuestionResult]] = defaultdict(list)
    for r in results:
        groups[f"{r.provider}:{r.model}"].append(r)

    rows = []
    for model_label, group in sorted(groups.items()):
        answered = [r for r in group if not r.error]
        scored = [r for r in answered if r.overall_score is not None]

        def avg(vals: list) -> str:
            v = [x for x in vals if x is not None]
            return f"{sum(v) / len(v):.3f}" if v else "N/A"

        rows.append({
            "model": model_label,
            "questions_total": len(group),
            "questions_answered": len(answered),
            "questions_failed": len(group) - len(answered),
            "avg_latency_s": avg([r.latency_seconds for r in answered]),
            "avg_kg_results": avg([r.kg_results_count for r in answered]),
            "avg_sources": avg([r.sources_count for r in answered]),
            "avg_groundedness": avg([r.groundedness for r in scored]),
            "avg_relevance": avg([r.relevance for r in scored]),
            "avg_completeness": avg([r.completeness for r in scored]),
            "avg_overall_score": avg([r.overall_score for r in scored]),
            "questions_scored": len(scored),
        })
    return rows


def print_summary(results: list[QuestionResult]) -> None:
    """Print a readable summary table to stdout."""
    summary = _compute_summary(results)
    if not summary:
        return

    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    header = f"{'Model':<40} {'Latency':>9} {'Overall':>9} {'Ground.':>9} {'Relev.':>9} {'Compl.':>9}"
    print(header)
    print("-" * 80)
    for row in summary:
        print(
            f"{row['model']:<40} "
            f"{row['avg_latency_s']:>9} "
            f"{row['avg_overall_score']:>9} "
            f"{row['avg_groundedness']:>9} "
            f"{row['avg_relevance']:>9} "
            f"{row['avg_completeness']:>9}"
        )
    print("=" * 80 + "\n")


# ── CLI ───────────────────────────────────────────────────────────

def parse_model(spec: str) -> tuple[str, str]:
    """Parse 'provider:model' spec into (provider, model)."""
    if ":" not in spec:
        raise argparse.ArgumentTypeError(
            f"Model spec must be 'provider:model', got: {spec!r}"
        )
    provider, model = spec.split(":", 1)
    return provider.strip(), model.strip()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="GraphRAG benchmark — compare LLMs on ORKG question answering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--models",
        nargs="+",
        required=True,
        metavar="PROVIDER:MODEL",
        help="Models to benchmark, e.g. ollama:llama3 openai:gpt-4o-mini",
    )
    p.add_argument(
        "--judge",
        default="ollama:qwen3:8b",
        metavar="PROVIDER:MODEL",
        help="Judge model for answer scoring (default: ollama:qwen3:8b — "
        "must not also appear in --models, see self-judging bias check)",
    )
    p.add_argument(
        "--questions",
        type=Path,
        default=QUESTIONS_FILE,
        help=f"Path to eval questions JSON (default: {QUESTIONS_FILE})",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR,
        help=f"Directory for result files (default: {RESULTS_DIR})",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Run only the first N questions (useful for quick tests)",
    )
    p.add_argument(
        "--query-types",
        nargs="+",
        metavar="TYPE",
        help="Filter questions by query type (e.g. topic_search method_comparison)",
    )
    p.add_argument(
        "--skip-judge",
        action="store_true",
        help="Skip LLM-as-judge scoring — only measure latency and retrieval stats",
    )
    p.add_argument(
        "--delay",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="Seconds to wait between questions (use 15-20 for Gemini free tier rate limits)",
    )
    p.add_argument(
        "--judge-delay",
        type=float,
        default=0.0,
        metavar="SECONDS",
        help="Seconds to wait between judge calls (use 4-5 for Gemini free tier, 15 RPM limit)",
    )
    return p


async def main_async(args: argparse.Namespace) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    global RESULTS_DIR
    RESULTS_DIR = args.output_dir

    questions = load_questions(args.questions, limit=args.limit, query_types=args.query_types)
    questions_by_id = {q["id"]: q for q in questions}

    if not questions:
        logger.error("No questions loaded — check --questions path and --query-types filter")
        return

    all_results: list[QuestionResult] = []

    for spec in args.models:
        provider, model = parse_model(spec)
        model_results = await run_model(provider, model, questions, delay=args.delay)
        all_results.extend(model_results)

    if not args.skip_judge:
        judge_provider, judge_model = parse_model(args.judge)
        apply_judge(all_results, questions_by_id, judge_provider, judge_model, delay=args.judge_delay)
    else:
        logger.info("Skipping judge scoring (--skip-judge)")

    json_path, csv_path = save_results(all_results, timestamp)
    print_summary(all_results)
    print(f"Results saved:\n  {json_path}\n  {csv_path}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    # Validate model specs early
    for spec in args.models:
        parse_model(spec)
    parse_model(args.judge)

    # The judge must not also be a candidate model — otherwise it would be
    # scoring its own answers (self-judging bias), silently inflating its score.
    if not args.skip_judge and args.judge in args.models:
        parser.error(
            f"--judge {args.judge!r} is also listed in --models. "
            "The judge cannot score its own answers (self-judging bias) — "
            "pick a judge model that is not one of the candidates."
        )

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
