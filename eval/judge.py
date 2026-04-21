"""
LLM-as-judge scorer for the GraphRAG evaluation.

Scores each model answer on three criteria (0-5 each):
  - Groundedness:  Is the answer strictly based on retrieved sources?
  - Relevance:     Does the answer address the question?
  - Completeness:  Does the answer cover the expected key aspects?

Overall score = mean(groundedness, relevance, completeness).

The judge is a separate LLM call — ideally a capable model (GPT-4o, Claude Sonnet)
so the scoring is reliable. Configure with --judge provider:model in benchmark.py.

Usage:
    from eval.judge import score_answer
    scores = score_answer(question, answer, sources, answer_hints, judge_client)
"""

import json
import re

from loguru import logger

from backend.llm.base import BaseLLMClient


_JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for a scientific question-answering system built on the Open Research Knowledge Graph (ORKG).

Your task is to evaluate the quality of an AI-generated answer. The system retrieves papers from ORKG and generates answers grounded in those papers.

Score the answer on THREE criteria (integer 0–5 each):

**Groundedness** — Is the answer strictly based on the retrieved sources?
  5 = Every claim is directly supported by the listed sources
  4 = Almost all claims supported, trivial inferences only
  3 = Most claims supported, some unsupported statements
  2 = About half the claims are unsupported or uncertain
  1 = Most claims not in sources, or answer ignores sources
  0 = Answer fabricates papers, results, or authors not in sources

**Relevance** — Does the answer address what was asked?
  5 = Fully and directly answers the question
  4 = Answers the question with minor gaps
  3 = Partially answers the question
  2 = Addresses the topic but not the specific question
  1 = Barely related to the question
  0 = Does not address the question at all

**Completeness** — Does the answer cover the key aspects expected for this type of question?
  5 = Covers all key aspects listed in expected hints
  4 = Covers most key aspects
  3 = Covers some key aspects
  2 = Covers only one key aspect or gives a very thin answer
  1 = Misses nearly all expected aspects
  0 = No relevant content

Respond ONLY with a valid JSON object — no explanation outside it:
{"groundedness": <int>, "relevance": <int>, "completeness": <int>, "reasoning": "<one concise sentence explaining the scores>"}"""


def score_answer(
    question: str,
    answer: str,
    sources: list[dict],
    answer_hints: list[str],
    judge_client: BaseLLMClient,
) -> dict[str, int | float | str | None]:
    """
    Score one answer using an LLM judge.

    Args:
        question:      The original user question.
        answer:        The model's generated answer.
        sources:       List of source dicts returned by the pipeline.
        answer_hints:  Expected key aspects from eval_questions.json.
        judge_client:  LLM client to use as judge (should be a capable model).

    Returns:
        Dict with keys: groundedness, relevance, completeness, overall, reasoning.
        On judge failure all score fields are None.
    """
    sources_text = "\n".join(
        f"  - {s.get('title', 'Unknown')} ({s.get('year') or 'N/A'})"
        for s in sources
    ) or "  (no sources retrieved)"

    hints_text = ", ".join(answer_hints) if answer_hints else "N/A"

    prompt = f"""Question: {question}

Retrieved sources (papers found in ORKG):
{sources_text}

Expected key aspects a good answer should mention: {hints_text}

Answer to evaluate:
{answer.strip() if answer else "(empty — model returned nothing)"}

Score the answer using the criteria in your instructions."""

    try:
        raw = judge_client.generate(
            prompt=prompt,
            system=_JUDGE_SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=300,
        )
        match = re.search(r"\{.*?\}", raw, re.DOTALL)
        if not match:
            logger.warning(f"Judge returned no JSON: {raw[:100]!r}")
            return _failed_scores("no JSON in judge response")

        data = json.loads(match.group())
        g = _clamp(data.get("groundedness"))
        r = _clamp(data.get("relevance"))
        c = _clamp(data.get("completeness"))
        overall = round((g + r + c) / 3, 2) if all(x is not None for x in (g, r, c)) else None

        return {
            "groundedness": g,
            "relevance": r,
            "completeness": c,
            "overall": overall,
            "reasoning": str(data.get("reasoning", ""))[:300],
        }

    except json.JSONDecodeError as e:
        logger.warning(f"Judge JSON parse error: {e}")
        return _failed_scores(f"JSON parse error: {e}")
    except Exception as e:
        logger.error(f"Judge scoring failed: {e}")
        return _failed_scores(str(e))


def _clamp(value: object) -> int | None:
    """Clamp a score value to 0–5 integer, or None if invalid."""
    try:
        v = int(value)  # type: ignore[arg-type]
        return max(0, min(5, v))
    except (TypeError, ValueError):
        return None


def _failed_scores(reason: str) -> dict:
    return {
        "groundedness": None,
        "relevance": None,
        "completeness": None,
        "overall": None,
        "reasoning": f"Judge failed: {reason}",
    }
