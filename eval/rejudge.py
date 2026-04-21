"""
Re-judge an existing benchmark results JSON.

Loads a results file produced by benchmark.py, scores all unscored answers
using the specified judge model (with rate-limit delay), and saves updated files.

Usage:
    python -m eval.rejudge \
        --results eval/results/benchmark_20260413_143541.json \
        --judge gemini:gemma-3-4b-it \
        --delay 7
"""

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

QUESTIONS_FILE = PROJECT_ROOT / "data" / "eval_questions.json"


def load_questions_by_id(path: Path) -> dict[str, dict]:
    with open(path) as f:
        data = json.load(f)
    return {q["id"]: q for q in data["questions"]}


def compute_summary(results: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        groups[f"{r['provider']}:{r['model']}"].append(r)

    rows = []
    for model_label, group in sorted(groups.items()):
        answered = [r for r in group if not r.get("error")]
        scored = [r for r in answered if r.get("overall_score") is not None]

        def avg(vals):
            v = [x for x in vals if x is not None]
            return f"{sum(v) / len(v):.3f}" if v else "N/A"

        rows.append({
            "model": model_label,
            "questions_total": len(group),
            "questions_answered": len(answered),
            "questions_failed": len(group) - len(answered),
            "avg_latency_s": avg([r["latency_seconds"] for r in answered]),
            "avg_kg_results": avg([r["kg_results_count"] for r in answered]),
            "avg_sources": avg([r["sources_count"] for r in answered]),
            "avg_groundedness": avg([r.get("groundedness") for r in scored]),
            "avg_relevance": avg([r.get("relevance") for r in scored]),
            "avg_completeness": avg([r.get("completeness") for r in scored]),
            "avg_overall_score": avg([r.get("overall_score") for r in scored]),
            "questions_scored": len(scored),
        })
    return rows


def print_summary(summary: list[dict]) -> None:
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)
    print(f"{'Model':<40} {'Latency':>9} {'Overall':>9} {'Ground.':>9} {'Relev.':>9} {'Compl.':>9} {'Scored':>7}")
    print("-" * 80)
    for row in summary:
        print(
            f"{row['model']:<40} "
            f"{row['avg_latency_s']:>9} "
            f"{row['avg_overall_score']:>9} "
            f"{row['avg_groundedness']:>9} "
            f"{row['avg_relevance']:>9} "
            f"{row['avg_completeness']:>9} "
            f"{row['questions_scored']:>7}"
        )
    print("=" * 80 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-run judge scoring on existing benchmark results")
    parser.add_argument("--results", required=True, type=Path, help="Path to benchmark JSON results file")
    parser.add_argument("--judge", required=True, metavar="PROVIDER:MODEL", help="Judge model, e.g. gemini:gemma-3-4b-it")
    parser.add_argument("--delay", type=float, default=7.0, metavar="SECONDS",
                        help="Seconds between judge calls (default: 7 — safe for 15K TPM free tier)")
    parser.add_argument("--questions", type=Path, default=QUESTIONS_FILE)
    parser.add_argument("--force", action="store_true", help="Re-score already-scored answers too")
    args = parser.parse_args()

    if ":" not in args.judge:
        parser.error("--judge must be PROVIDER:MODEL, e.g. gemini:gemma-3-4b-it")
    judge_provider, judge_model = args.judge.split(":", 1)

    from backend.llm.factory import create_llm_client
    from eval.judge import score_answer

    with open(args.results) as f:
        results = json.load(f)

    questions_by_id = load_questions_by_id(args.questions)
    judge_client = create_llm_client(provider=judge_provider, model=judge_model)

    to_score = [r for r in results if args.force or r.get("overall_score") is None]
    already_scored = len(results) - len(to_score)
    logger.info(f"Results: {len(results)} total, {already_scored} already scored, {len(to_score)} to score")
    logger.info(f"Judge: {args.judge}, delay: {args.delay}s between calls")

    for i, r in enumerate(to_score, 1):
        if i > 1:
            time.sleep(args.delay)

        qid = r["question_id"]
        model_label = f"{r['provider']}:{r['model']}"
        logger.info(f"[judge] ({i}/{len(to_score)}) {qid} × {model_label}")

        if r.get("error"):
            r["judge_reasoning"] = "skipped: pipeline error"
            continue

        q = questions_by_id.get(qid, {})
        scores = score_answer(
            question=r["question"],
            answer=r["answer"],
            sources=r.get("sources", []),
            answer_hints=q.get("answer_hints", []),
            judge_client=judge_client,
        )
        r["groundedness"] = scores["groundedness"]
        r["relevance"] = scores["relevance"]
        r["completeness"] = scores["completeness"]
        r["overall_score"] = scores["overall"]
        r["judge_reasoning"] = scores["reasoning"]

        if scores["overall"] is not None:
            logger.info(f"  → overall={scores['overall']:.2f} (G={scores['groundedness']} R={scores['relevance']} C={scores['completeness']})")
        else:
            logger.warning(f"  → scoring failed: {scores['reasoning'][:80]}")

    # Save updated results back to same file
    with open(args.results, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Updated results → {args.results}")

    # Update summary CSV
    csv_path = args.results.with_name(args.results.stem + "_summary.csv")
    summary = compute_summary(results)
    with open(csv_path, "w", newline="") as f:
        if summary:
            writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            writer.writeheader()
            writer.writerows(summary)
    logger.info(f"Updated summary → {csv_path}")

    print_summary(summary)


if __name__ == "__main__":
    main()
