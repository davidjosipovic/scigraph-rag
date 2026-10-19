"""
Claim-level faithfulness of benchmark answers to the knowledge graph context.

The old judge (eval/judge.py) only saw source titles, so it could not tell
whether "the Oromo paper trains a CNN-RNN baseline" is in the context or
invented. This judge sees the full context the answering model saw and works
in two steps:

  1. extract   split the answer body (without the closing "Sources" list) into
               short, self-contained factual claims
  2. verify    label every claim against the context, one call per claim:
                 S  supported     stated in, or directly follows from, the context
                 U  unsupported   not in the context (invented detail, outside knowledge)
                 C  contradicted  the context says otherwise

For S and C the judge must copy the deciding evidence line; a quote that is
not in the context turns the label into U (an absence claim judged S needs no
quote). This keeps a small judge from waving claims through.

Faithfulness = S / (S + U + C) per answer; answers without claims are left out
of the mean. The judge runs locally (Ollama, default qwen3:8b) and must not be
one of the benchmarked models. Its labels are checked against a human-labelled
sample (--export-sample / --agreement) before the numbers are reported.

Usage:
    python -m eval.faithfulness --results eval/results/benchmark_X.json
    python -m eval.faithfulness --results ... --export-sample 60
    python -m eval.faithfulness --agreement eval/results/faithfulness_sample_labelled.json

Output (resumable — finished answers are skipped on re-run):
    eval/results/faithfulness_<benchmark name>.json
"""

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from eval.answer_eval import split_sections  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
LABELS = {"S": "supported", "U": "unsupported", "C": "contradicted"}

_EXTRACT_SYSTEM = """You split an answer into atomic factual claims for fact-checking.

Rules:
- One claim per item: a single fact about a paper, method, dataset, result or the research area.
- Make each claim self-contained: replace "it", "the first paper", "this study" with the paper title or method name.
- Mark a claim "absence" when it says some information is missing or that no paper does something
  (e.g. "No datasets are specified for paper X"); every other claim is "fact".
- Skip claims that only say a listed paper is about the question's topic, section headings,
  restatements of the question and generic filler ("Further research is needed").
- Keep descriptions of what a method is or does, numbers, datasets, comparisons and verdicts.
- At most 25 claims.

Respond ONLY with a JSON array: [{"claim": "...", "kind": "fact"}, ...]"""

_VERIFY_SYSTEM = """You are a strict fact-checker. You receive the ONLY evidence available (records from the Open Research Knowledge Graph) and ONE claim made by a question-answering system.

Label the claim:
  S = supported: stated in the evidence, or a direct, obvious consequence of it
  U = unsupported: not in the evidence — added detail, background knowledge, a guess, or a paper/result that does not appear
  C = contradicted: the evidence says something different (wrong number, wrong paper, wrong method/dataset)

Be strict: background knowledge that is true in general but not in the evidence is U.
For S or C, copy the evidence line that decides it, exactly as written. For U write "NONE".
The quoted line must itself contain the claim's specific content (the number, dataset, property or result).
A line that merely mentions the same paper or method name does NOT support a claim about what it is, does or achieves — that is U.
If the claim says something is MISSING from the evidence: S if it really is missing (quote "NONE"), C if the evidence contains it (quote that line).

Respond ONLY with JSON: {"quote": "<exact evidence line or NONE>", "label": "S|U|C"}"""


def _json_block(raw: str, opener: str, closer: str):
    match = re.search(re.escape(opener) + r".*" + re.escape(closer), raw, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON in judge response: {raw[:120]!r}")
    return json.loads(match.group())


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def quote_in_context(quote: str, context: str) -> bool:
    """Every part of the judge's quote must really be in the context.

    Small judges sometimes join several evidence lines with ";" or newlines.
    """
    ctx = _squash(context)
    parts = [_squash(p.strip(" \"'`.")).removeprefix("- ") for p in re.split(r"[;\n]", quote)]
    parts = [p for p in parts if p]
    return bool(parts) and any(len(p) >= 8 for p in parts) and all(p in ctx for p in parts)


def extract_claims(answer: str, question: str, judge) -> list[dict]:
    sections = split_sections(answer or "")
    body = "\n".join(v for k, v in sections.items() if k != "sources").strip()
    if not body:
        return []
    raw = judge.generate(
        prompt=f"Question: {question}\n\nAnswer:\n{body}\n\nList the atomic factual claims.",
        system=_EXTRACT_SYSTEM, temperature=0.0, max_tokens=2500,
    )
    claims = []
    for c in _json_block(raw, "[", "]"):
        text = str(c.get("claim", "") if isinstance(c, dict) else c).strip()
        kind = c.get("kind", "fact") if isinstance(c, dict) else "fact"
        if text:
            claims.append({"claim": text, "kind": "absence" if kind == "absence" else "fact"})
    return claims[:25]


def verify_claim(claim: dict, context: str, judge) -> dict:
    """Label one claim. S/C need a quote that is really in the context, else U.

    The context comes first in the prompt, so Ollama reuses the cached prefix
    across the claims of one answer.
    """
    raw = judge.generate(
        prompt=f"EVIDENCE:\n{context}\n\nCLAIM: {claim['claim']}\n\nLabel the claim.",
        system=_VERIFY_SYSTEM, temperature=0.0, max_tokens=400,
    )
    try:
        data = _json_block(raw, "{", "}")
    except ValueError:
        # Long multi-line quotes sometimes break the JSON: read the fields directly.
        label = re.search(r'"label"\s*:\s*"([SUC])', raw)
        quote = re.search(r'"quote"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
        if not label:
            raise
        data = {"label": label.group(1),
                "quote": json.loads(f'"{quote.group(1)}"') if quote else ""}
    label = str(data.get("label", "")).strip().upper()[:1]
    quote = str(data.get("quote", "")).strip()
    if label not in LABELS:
        raise ValueError(f"invalid label: {data}")
    quote_ok = quote_in_context(quote, context)
    final = label
    # An absence claim judged S needs no quote; everything else judged S or C does.
    if label in "SC" and not (label == "S" and claim["kind"] == "absence") and not quote_ok:
        final = "U"
    return {**claim, "judge_label": label, "quote": quote, "quote_ok": quote_ok, "label": final}


def faithfulness(labels: list[str]) -> float | None:
    return round(labels.count("S") / len(labels), 4) if labels else None


def run(results_path: Path, judge_spec: str, out_dir: Path = RESULTS_DIR) -> Path:
    from backend.llm.factory import create_llm_client

    results = json.loads(results_path.read_text())
    if not all("context" in r for r in results):
        sys.exit(f"{results_path.name} has no saved context — re-run eval/benchmark.py")

    provider, model = judge_spec.split(":", 1)
    if model in {r["model"] for r in results}:
        sys.exit(f"judge {judge_spec} is also a benchmarked model — pick another judge")
    judge = create_llm_client(provider=provider, model=model)

    out_path = out_dir / f"faithfulness_{results_path.stem}.json"
    done = json.loads(out_path.read_text()) if out_path.exists() else {"judge": judge_spec, "answers": {}}
    answers = done["answers"]

    todo = [r for r in results if not r.get("error") and f"{r['model']}|{r['question_id']}" not in answers]
    logger.info(f"{len(todo)} answers to judge with {judge_spec} ({len(answers)} already done)")
    for i, r in enumerate(todo, 1):
        key = f"{r['model']}|{r['question_id']}"
        entry: dict = {"model": r["model"], "qid": r["question_id"], "query_type": r["query_type"]}
        try:
            claims = [verify_claim(c, r["context"], judge)
                      for c in extract_claims(r["answer"], r["question"], judge)]
            labels = [c["label"] for c in claims]
            entry.update(claims=claims, labels=labels, faithfulness=faithfulness(labels))
        except Exception as e:  # keep going; failed answers are retried on the next run
            logger.warning(f"{key}: {e}")
            continue
        answers[key] = entry
        logger.info(f"({i}/{len(todo)}) {key}: {len(claims)} claims, faithfulness={entry['faithfulness']}")
        out_path.write_text(json.dumps(done, indent=1, ensure_ascii=False))
    return out_path


def summarize(out_path: Path) -> None:
    answers = json.loads(out_path.read_text())["answers"].values()
    by_model: dict[str, list[dict]] = defaultdict(list)
    for a in answers:
        by_model[a["model"]].append(a)

    print(f"\n{'model':<36}{'n':>5}{'claims/ans':>12}{'Faithful':>10}{'Unsupp':>9}{'Contra':>9}")
    for model, rows in sorted(by_model.items()):
        labels = [lab for a in rows for lab in a["labels"]]
        scored = [a["faithfulness"] for a in rows if a["faithfulness"] is not None]
        n = len(labels) or 1
        print(f"{model:<36}{len(rows):>5}{len(labels) / len(rows):>12.1f}"
              f"{sum(scored) / len(scored) if scored else 0:>10.3f}"
              f"{labels.count('U') / n:>9.3f}{labels.count('C') / n:>9.3f}")


def export_sample(out_path: Path, results_path: Path, n: int, seed: int = 13) -> Path:
    """Random claims (with their context) for a human to label blind."""
    results = {f"{r['model']}|{r['question_id']}": r for r in json.loads(results_path.read_text())}
    pool = [(key, i, c["claim"], c["label"])
            for key, a in json.loads(out_path.read_text())["answers"].items()
            for i, c in enumerate(a["claims"])]
    random.Random(seed).shuffle(pool)
    sample = [{"key": key, "claim_index": i, "claim": c, "context": results[key]["context"],
               "human_label": "", "judge_label": lab}
              for key, i, c, lab in pool[:n]]
    path = out_path.with_name(out_path.stem + "_sample.json")
    path.write_text(json.dumps(sample, indent=1, ensure_ascii=False))
    print(f"{len(sample)} claims → {path}\nFill in human_label (S/U/C) without looking at judge_label.")
    return path


def agreement(path: Path) -> None:
    rows = [r for r in json.loads(path.read_text()) if r["human_label"].strip().upper()[:1] in LABELS]
    h = [r["human_label"].strip().upper()[:1] for r in rows]
    j = [r["judge_label"] for r in rows]
    n = len(rows)
    po = sum(a == b for a, b in zip(h, j)) / n
    pe = sum((h.count(k) / n) * (j.count(k) / n) for k in LABELS)
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
    print(f"n={n}  accuracy={po:.3f}  Cohen's kappa={kappa:.3f}")
    print("confusion (rows = human, cols = judge):")
    print("     " + "".join(f"{k:>5}" for k in LABELS))
    for a in LABELS:
        print(f"{a:>5}" + "".join(f"{sum(x == a and y == b for x, y in zip(h, j)):>5}" for b in LABELS))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", type=Path, help="benchmark JSON (saved with context)")
    p.add_argument("--judge", default="ollama:qwen3:8b", metavar="PROVIDER:MODEL")
    p.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    p.add_argument("--export-sample", type=int, metavar="N", help="write N random claims for human labelling")
    p.add_argument("--agreement", type=Path, metavar="FILE", help="judge vs human agreement on a labelled sample")
    args = p.parse_args()

    if args.agreement:
        agreement(args.agreement)
        return
    if not args.results:
        p.error("--results is required")
    out_path = run(args.results, args.judge, args.output_dir)
    summarize(out_path)
    if args.export_sample:
        export_sample(out_path, args.results, args.export_sample)


if __name__ == "__main__":
    main()
