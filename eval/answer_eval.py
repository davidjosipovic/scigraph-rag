"""
Gold-based answer evaluation — no LLM judge.

The LLM judge (eval/judge.py) sees only source titles and rewards answers that
say "insufficient evidence" (groundedness 5/5 even with zero sources). This
script scores the answers of a benchmark run against the graded retrieval gold
set instead: every answer references papers by exact title, so the papers an
answer relies on can be read off deterministically and looked up in
data/gold/retrieval_gold.json.

Per answer:
  cited        source papers whose title appears in the answer body
               (the closing "Sources" list is ignored — models list every
               source there, even ones they just called irrelevant)
  abstain      the answer relies on no paper: nothing cited in the body, or a
               claim verdict of INSUFFICIENT EVIDENCE
  fabricated   titles in the closing "Sources" list that match none of the
               retrieved sources (heuristic — audit with --show before quoting)

Metrics (K = papers in the LLM context):
  answerable   the context holds at least one grade-2 paper
  CitePrec     mean gold grade / 2 of the cited papers (graded precision)
  CiteUtil     share of the grade-2 papers in the context that are cited
  FalseAbst    answerable, but the model abstained
  CorrAbst     not answerable, and the model abstained
  Fabr         answers with at least one fabricated title
  Score        per question: answerable → F1(CitePrec, CiteUtil), 0 on
               abstention; not answerable → 1 on abstention, else CitePrec
               (at most 0.5, since nothing cited can be grade 2); claim
               questions with a gold verdict (data/gold/claim_verdicts.json)
               score 1 if the verdict is accepted, else 0
  Verdict      verdict accuracy on those claim questions; INSUFFICIENT is
               also accepted when none of the deciding papers was retrieved

Usage:
    python -m eval.answer_eval --results eval/results/benchmark_X.json
    python -m eval.answer_eval --results ... --split test --show
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

GOLD_FILE = PROJECT_ROOT / "data" / "gold" / "retrieval_gold.json"
VERDICT_FILE = PROJECT_ROOT / "data" / "gold" / "claim_verdicts.json"
_VERDICTS = ("SUPPORTED", "REFUTED", "INSUFFICIENT")

_SECTION_RE = re.compile(
    r"^\s*(?:#+\s*)?(?:\d+\.\s*)?\**\s*(overview|relevant papers|methods(?: used)?|"
    r"datasets(?: used)?|sources|verdict|evidence)\s*\**\s*:?\s*\**",
    re.IGNORECASE | re.MULTILINE,
)
_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+\.)\s+(.+?)\s*$", re.MULTILINE)


def norm(text: str) -> str:
    text = text.lower().replace("‑", "-").replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text)).strip()


def split_sections(answer: str) -> dict[str, str]:
    """Answer text per template section; text before any header is 'overview'."""
    marks = [(m.start(), m.end(), m.group(1).lower().split()[0]) for m in _SECTION_RE.finditer(answer)]
    if not marks:
        return {"body": answer}
    sections: dict[str, str] = {"overview": answer[: marks[0][0]]}
    for i, (_, end, name) in enumerate(marks):
        stop = marks[i + 1][0] if i + 1 < len(marks) else len(answer)
        sections[name] = sections.get(name, "") + "\n" + answer[end:stop]
    return sections


def mentions(text: str, title: str) -> bool:
    t, n = norm(title), norm(text)
    if not t:
        return False
    # Long titles are sometimes shortened by the model — accept a long prefix.
    return t in n or (len(t) > 60 and t[:60] in n)


def is_fabricated(item: str, titles: list[str]) -> bool:
    """A listed title that matches none of the retrieved sources."""
    if re.match(r"^\W*[\w ]{1,20}:\s", item) or re.search(r"\bR\d{3,}\b", item):
        return False  # "DOI: …", "Field: …" or a resource-id label, not a title
    item = re.sub(r"\(\d{4}\)|[*_\"“”]", "", item).strip(" .:-")
    n = norm(item)
    if len(n.split()) < 4 or n in {"none", "none found", "n a"}:
        return False  # too short to be a paper title (or an explicit "None")
    for t in titles:
        nt = norm(t)
        if nt in n or n in nt or SequenceMatcher(None, n, nt).ratio() > 0.8:
            return False
    return True


def parse_verdict(verdict_section: str) -> str | None:
    """First verdict word in the Verdict section ("PARTIALLY SUPPORTED" → None)."""
    m = re.search(r"\b(partially\s+)?(supported|refuted|insufficient)\b", verdict_section, re.IGNORECASE)
    return None if not m or m.group(1) else m.group(2).upper()


def evaluate_answer(answer: str, sources: list[dict], grades: dict[str, int],
                    gold_verdict: dict | None = None) -> dict:
    sections = split_sections(answer or "")
    body = "\n".join(v for k, v in sections.items() if k != "sources")
    titles = [s.get("title", "") for s in sources]

    cited = [s["uri"] for s in sources if mentions(body, s.get("title", ""))]
    verdict = sections.get("verdict", "")
    insufficient = bool(re.search(r"insufficient", verdict, re.IGNORECASE))
    abstain = not cited or insufficient

    listed = _BULLET_RE.findall(sections.get("sources", ""))
    fabricated = sorted({i.strip() for i in listed if is_fabricated(i, titles)})

    ctx2 = [s["uri"] for s in sources if grades.get(s["uri"]) == 2]
    answerable = bool(ctx2)
    prec = sum(grades.get(u, 0) for u in cited) / (2 * len(cited)) if cited else None
    util = len([u for u in cited if grades.get(u) == 2]) / len(ctx2) if ctx2 else None

    if answerable:
        score = 0.0 if abstain or not prec or not util else 2 * prec * util / (prec + util)
    else:
        score = 1.0 if abstain else (prec or 0.0)

    # Claim questions with a gold verdict are scored on the verdict instead:
    # a relevant paper in the context does not mean the claim can be decided.
    verdict, verdict_ok = parse_verdict(verdict), None
    if gold_verdict and gold_verdict["verdict"] != "MIXED":
        accept = set(gold_verdict["accept"])
        uris = {s["uri"] for s in sources}
        if not uris & set(gold_verdict["key_papers"]):
            accept.add("INSUFFICIENT")
        verdict_ok = verdict in accept
        score = float(verdict_ok)

    return {
        "cited": cited,
        "abstain": abstain,
        "answerable": answerable,
        "cite_prec": prec,
        "cite_util": util,
        "fabricated": fabricated,
        "verdict": verdict,
        "verdict_ok": verdict_ok,
        "unjudged_cited": [u for u in cited if u not in grades],
        "score": round(score, 4),
    }


def _mean(vals: list) -> float | None:
    v = [x for x in vals if x is not None]
    return round(sum(v) / len(v), 3) if v else None


def summarize(rows: list[dict]) -> dict:
    # Abstention rates exclude verdict-scored claim questions (see evaluate_answer).
    ans = [r for r in rows if r["answerable"] and r["verdict_ok"] is None]
    una = [r for r in rows if not r["answerable"] and r["verdict_ok"] is None]
    return {
        "n": len(rows),
        "answerable": len(ans),
        "Score": _mean([r["score"] for r in rows]),
        "CitePrec": _mean([r["cite_prec"] for r in rows]),
        "CiteUtil": _mean([r["cite_util"] for r in ans]),
        "FalseAbst": _mean([float(r["abstain"]) for r in ans]),
        "CorrAbst": _mean([float(r["abstain"]) for r in una]),
        "Fabr": _mean([float(bool(r["fabricated"])) for r in rows]),
        "Verdict": _mean([float(r["verdict_ok"]) for r in rows if r["verdict_ok"] is not None]),
        "Judge": _mean([r.get("judge_overall") for r in rows]),
    }


def in_split(qid: str, split: str) -> bool:
    if split == "all":
        return True
    odd = int(re.sub(r"\D", "", qid)) % 2 == 1
    return odd if split == "dev" else not odd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", type=Path, nargs="+", required=True)
    p.add_argument("--split", choices=["all", "dev", "test"], default="all")
    p.add_argument("--show", action="store_true", help="per-answer detail")
    p.add_argument("--by-type", action="store_true", help="also break down by query type")
    args = p.parse_args()

    gold = json.loads(GOLD_FILE.read_text())["questions"]
    grades = {qid: {j["uri"]: j["grade"] for j in q["judgments"]} for qid, q in gold.items()}
    verdicts = json.loads(VERDICT_FILE.read_text())["questions"]

    by_model: dict[str, list[dict]] = defaultdict(list)
    for path in args.results:
        for r in json.loads(path.read_text()):
            if r.get("error") or not in_split(r["question_id"], args.split):
                continue
            row = evaluate_answer(r["answer"], r["sources"], grades[r["question_id"]],
                                  verdicts.get(r["question_id"]))
            row.update(qid=r["question_id"], qtype=r["query_type"], judge_overall=r.get("overall_score"))
            by_model[f"{r['provider']}:{r['model']}"].append(row)
            if args.show:
                print(f"{r['model']:<28} {r['question_id']} score={row['score']:.2f} "
                      f"abst={row['abstain']!s:<5} ans={row['answerable']!s:<5} "
                      f"prec={row['cite_prec']} util={row['cite_util']} fabr={row['fabricated']} "
                      f"judge={row['judge_overall']}")

    cols = ["n", "answerable", "Score", "CitePrec", "CiteUtil", "FalseAbst", "CorrAbst", "Fabr", "Verdict", "Judge"]
    print(f"\n{'model':<40}" + "".join(f"{c:>11}" for c in cols))
    for model, rows in sorted(by_model.items(), key=lambda kv: -(summarize(kv[1])["Score"] or 0)):
        s = summarize(rows)
        print(f"{model:<40}" + "".join(f"{'-' if s[c] is None else s[c]!s:>11}" for c in cols))
        if args.by_type:
            types = defaultdict(list)
            for r in rows:
                types[r["qtype"]].append(r)
            for t, tr in sorted(types.items()):
                s = summarize(tr)
                print(f"  {t:<38}" + "".join(f"{'-' if s[c] is None else s[c]!s:>11}" for c in cols))
    unj = sum(len(r["unjudged_cited"]) for rows in by_model.values() for r in rows)
    print(f"\nunjudged cited papers (counted as grade 0): {unj}")


if __name__ == "__main__":
    main()
