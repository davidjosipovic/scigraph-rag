"""
Retrieval ranking, hard filtering, and truncation.

Every retrieval strategy returns rows with different fields — method/dataset
labels, research problems, claim-evidence (predicate, value) pairs, title
lookups — so ranking first reduces each paper to the same profile:

    method labels   values in a method-like role (method, model, baseline, ...)
    dataset labels  values in a dataset-like role, incl. benchmark results
    any labels      every matched value regardless of role
    title           plus whether a title lookup matched it (exactly)

Scoring system (per paper):
    +2  per queried method found in a method role   (+1 if only in another role)
    +2  dataset found in a dataset role              (+1 if only in another role)
    +1  task / research problem found
    +1  keyword match in the paper title
    +2  found by a title lookup, +10 more for an exact title match
    +1  per question keyword in the title, for papers found by a title
        keyword search (paper lookups)
    +1  the paper carries results (benchmark scores) and matches an entity

Terms of up to 4 characters must match a whole word (an optional plural
"s" allowed), so "GAN" does not match "organ" and "NAS" does not match
"nasal"; longer terms match as substrings. This mirrors _label_match in
backend/kg/queries.py, which decides what SPARQL returns in the first place.

Hard filtering:
    When the query contains **both** method(s) AND dataset(s), only papers
    matching at least one method AND at least one dataset are kept. Papers
    found by a title lookup (or matching 2+ title keywords) are exempt —
    the user named the paper itself.

Truncation:
    Only the top N papers (default 8) are forwarded to the LLM context
    window to stay within token limits.

Usage:
    from backend.rag.ranking import rank_results, hard_filter, truncate_to_top_papers

    ranked  = rank_results(results, entities)
    filtered = hard_filter(ranked, entities)
    top     = truncate_to_top_papers(filtered, max_papers=8)
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

from backend.rag.entity_normalization import ExpandedEntities


# ── Score weights ────────────────────────────────────────────────

SCORE_METHOD_MATCH = 2
SCORE_METHOD_OTHER_ROLE = 1
SCORE_DATASET_MATCH = 2
SCORE_DATASET_OTHER_ROLE = 1
SCORE_TASK_MATCH = 1
SCORE_TITLE_KEYWORD = 1
SCORE_TITLE_LOOKUP = 2
SCORE_EXACT_TITLE = 10  # the user named this paper; nothing else should outrank it
SCORE_HAS_RESULTS = 1

# Predicate labels that put a claim-evidence value into a role
# (kept in sync with _METHOD_PRED_RE / _DATASET_PRED_RE in backend/kg/queries.py).
_METHOD_ROLE = re.compile(r"method|model|approach|algorithm|architecture|technique|classifier|baseline", re.I)
_DATASET_ROLE = re.compile(r"dataset|data set|benchmark|corpus|corpora|evaluat", re.I)
_RESULT_ROLE = re.compile(r"result|score|accuracy|performance", re.I)

_METHOD_KEYS = ("methodLabel", "methodALabel", "methodBLabel", "entityLabel")
_DATASET_KEYS = ("datasetLabel",)
_OTHER_KEYS = ("problemLabel", "fieldLabel")


@lru_cache(maxsize=4096)
def _term_re(term: str) -> re.Pattern:
    if len(term) <= 4:
        return re.compile(rf"(?<![a-z0-9]){re.escape(term)}s?(?![a-z0-9])")
    return re.compile(re.escape(term))


def _matches(term: str, labels: set[str]) -> bool:
    """Match a lowercased term in any of the labels (see module docstring)."""
    rx = _term_re(term)
    return any(rx.search(label) for label in labels)


def _split_values(value: str) -> list[str]:
    """claim_evidence concatenates the values of one predicate with ' | '."""
    return [v.strip().lower() for v in value.split(" | ") if v.strip()]


class _PaperProfile:
    """Everything the retrieval strategies said about one paper."""

    __slots__ = ("title", "method_labels", "dataset_labels", "any_labels",
                 "title_lookup", "exact_title", "title_hits", "has_results")

    def __init__(self, title: str) -> None:
        self.title = title
        self.method_labels: set[str] = set()
        self.dataset_labels: set[str] = set()
        self.any_labels: set[str] = set()
        self.title_lookup = False
        self.exact_title = False
        self.title_hits = 0
        self.has_results = False

    def add_row(self, row: dict[str, Any]) -> None:
        for key in _METHOD_KEYS:
            if row.get(key):
                self.method_labels.add(row[key].lower())
        for key in _DATASET_KEYS:
            if row.get(key):
                self.dataset_labels.add(row[key].lower())
        for key in _OTHER_KEYS:
            if row.get(key):
                self.any_labels.add(row[key].lower())

        # claim_evidence rows: (predicate, value) pairs
        pred, value = row.get("predLabel") or "", row.get("valueLabel") or ""
        if value:
            values = _split_values(value)
            self.any_labels.update(values)
            if _METHOD_ROLE.search(pred):
                self.method_labels.update(values)
            if _DATASET_ROLE.search(pred) or pred.endswith(" result"):
                self.dataset_labels.update(values)
            if _RESULT_ROLE.search(pred):
                self.has_results = True

        # paper_lookup_by_title rows carry the title length they ranked by
        if "titleLength" in row:
            self.title_lookup = True
            if str(row.get("exact")) == "1":
                self.exact_title = True

        # title_keyword_search rows count title keyword hits; claim_evidence
        # rows also carry ?hits, but always with a predicate
        if "hits" in row and not pred:
            self.title_hits = max(self.title_hits, int(row["hits"]))

        self.any_labels |= self.method_labels | self.dataset_labels


def _match_forms(names: list[str], forms_of) -> list[set[str]]:
    """
    One set of lowercased match forms per queried entity: the canonical
    form plus its variants. Variants shorter than 3 characters (e.g. "dl",
    "ml", "rl") are skipped — too ambiguous even as whole words.
    """
    out = []
    for name in names:
        canonical, *variants = forms_of(name)
        out.append({canonical.lower()} | {v.lower() for v in variants if len(v) >= 3})
    return out


def _profiles(results: list[dict[str, Any]]) -> dict[str, _PaperProfile]:
    profiles: dict[str, _PaperProfile] = {}
    for row in results:
        uri = row.get("paper", "")
        if not uri:
            continue
        if uri not in profiles:
            profiles[uri] = _PaperProfile((row.get("title") or "").lower())
        profiles[uri].add_row(row)
    return profiles


def rank_results(
    results: list[dict[str, Any]],
    entities: ExpandedEntities,
) -> list[dict[str, Any]]:
    """
    Score and rank result rows by their paper's relevance.

    Rows from all retrieval strategies are grouped per paper URI (a paper
    may appear in many rows) and the paper is scored against the queried
    entities — see the module docstring for the scoring system.

    Papers with score 0 are kept but appear last.

    Args:
        results:  Flat list of SPARQL result rows (already deduplicated
                  or not — the function handles both).
        entities: Expanded entities from the normalization step.

    Returns:
        Same list of dicts, re-ordered by relevance score.
        Each dict gets an added ``_score`` key for transparency.
    """
    if not results:
        return []

    profiles = _profiles(results)
    methods = _match_forms(entities.methods, entities.all_method_forms)
    datasets = _match_forms(entities.datasets, entities.all_dataset_forms)
    tasks = _match_forms(entities.tasks, entities.all_task_forms)
    keywords = {e.lower() for e in entities.all_entities()}

    uri_scores: dict[str, int] = {}
    for uri, p in profiles.items():
        score = 0
        matched_entity = False

        # Each queried method counts, so a paper with both compared
        # methods outranks one with just one of them.
        for forms in methods:
            if any(_matches(f, p.method_labels) for f in forms):
                score += SCORE_METHOD_MATCH
                matched_entity = True
            elif any(_matches(f, p.any_labels) for f in forms):
                score += SCORE_METHOD_OTHER_ROLE
                matched_entity = True

        if any(_matches(f, p.dataset_labels) for forms in datasets for f in forms):
            score += SCORE_DATASET_MATCH
            matched_entity = True
        elif any(_matches(f, p.any_labels) for forms in datasets for f in forms):
            score += SCORE_DATASET_OTHER_ROLE
            matched_entity = True

        if any(_matches(f, p.any_labels | {p.title}) for forms in tasks for f in forms):
            score += SCORE_TASK_MATCH

        if any(_matches(kw, {p.title}) for kw in keywords):
            score += SCORE_TITLE_KEYWORD

        if p.title_lookup:
            score += SCORE_TITLE_LOOKUP + (SCORE_EXACT_TITLE if p.exact_title else 0)

        score += p.title_hits

        if p.has_results and matched_entity:
            score += SCORE_HAS_RESULTS

        uri_scores[uri] = score

    # Annotate each result row with its paper's score
    for row in results:
        row["_score"] = uri_scores.get(row.get("paper", ""), 0)

    # Sort by score descending, then by title alphabetically for stability
    return sorted(
        results,
        key=lambda r: (-r.get("_score", 0), (r.get("title") or "").lower()),
    )


# ── Hard filter ──────────────────────────────────────────────────


def hard_filter(
    results: list[dict[str, Any]],
    entities: ExpandedEntities,
    min_keep: int = 3,
) -> list[dict[str, Any]]:
    """
    Eliminate papers that do NOT match **both** a queried method AND a
    queried dataset when the user's question mentions both.

    If the query contains only methods (no datasets) or only datasets
    (no methods), no filtering is applied — all papers are kept. Papers
    found by a title lookup always pass: the user asked for that paper.

    Matching uses the same whole-word logic and variant expansion as
    ``rank_results``, over every label a paper was matched with
    (any role), so claim-evidence rows count too.

    When fewer than ``min_keep`` papers pass, nothing is removed: ORKG
    often records the compared methods without the benchmark the question
    names, and an almost-empty context is worse than one ranked by score
    (dataset matches already rank first).

    Args:
        results:  Ranked (and possibly scored) list of SPARQL result rows.
        entities: Expanded entities from the normalization step.
        min_keep: Minimum number of passing papers for the filter to apply.

    Returns:
        Filtered list — only papers satisfying both conditions.
    """
    if not entities.methods or not entities.datasets:
        return results  # nothing to hard-filter

    method_forms = set().union(*_match_forms(entities.methods, entities.all_method_forms))
    dataset_forms = set().union(*_match_forms(entities.datasets, entities.all_dataset_forms))

    passing = {
        uri for uri, p in _profiles(results).items()
        if p.title_lookup or p.title_hits >= 2 or (
            any(_matches(m, p.any_labels) for m in method_forms)
            and any(_matches(d, p.any_labels) for d in dataset_forms)
        )
    }
    if len(passing) < min_keep:
        return results
    return [r for r in results if r.get("paper", "") in passing]


# ── Context selection ────────────────────────────────────────────

_KEY_PREDICATE = re.compile(
    r"method|model|baseline|approach|dataset|benchmark|result|metric|task|"
    r"problem|superior|compar|finding|conclusion|accuracy|score|performance",
    re.I,
)


def top_paper_uris(results: list[dict[str, Any]], n: int) -> list[str]:
    """The first ``n`` distinct paper URIs of a ranked result list."""
    seen: list[str] = []
    for row in results:
        uri = row.get("paper", "")
        if uri and uri not in seen:
            seen.append(uri)
            if len(seen) >= n:
                break
    return seen


def select_detail_rows(
    results: list[dict[str, Any]],
    entities: ExpandedEntities,
    max_pairs: int = 15,
) -> list[dict[str, Any]]:
    """
    Keep the most informative enrichment rows (``_detail``) per paper.

    A full ORKG record can hold hundreds of values (abstract fragments,
    hardware, licences...). Per paper, at most ``max_pairs`` detail rows
    are kept, in this order: values mentioning a queried entity, benchmark
    results, values under a key predicate (method, baseline, dataset,
    metric, research problem, ...). Everything else is dropped. Rows from
    the retrieval queries themselves are always kept.
    """
    forms = set().union(
        *_match_forms(entities.methods, entities.all_method_forms),
        *_match_forms(entities.datasets, entities.all_dataset_forms),
        *_match_forms(entities.tasks, entities.all_task_forms),
    )

    def priority(row: dict[str, Any]) -> int | None:
        pred = row.get("predLabel") or ""
        value = (row.get("valueLabel") or "").lower()
        if any(_matches(f, {value}) for f in forms):
            return 0
        if pred.endswith(" result"):
            return 1
        if _KEY_PREDICATE.search(pred):
            return 2
        return None

    ranked: dict[str, list[tuple[int, int]]] = {}
    for i, row in enumerate(results):
        if row.get("_detail"):
            p = priority(row)
            if p is not None:
                ranked.setdefault(row.get("paper", ""), []).append((p, i))
    keep = {i for picks in ranked.values() for _, i in sorted(picks)[:max_pairs]}
    return [r for i, r in enumerate(results) if not r.get("_detail") or i in keep]


# ── Truncation ───────────────────────────────────────────────────


def soft_filter(
    results: list[dict[str, Any]],
    min_score: int = 1,
) -> list[dict[str, Any]]:
    """
    Drop results whose ``_score`` is below *min_score*.

    Unlike ``hard_filter`` — which requires **both** a method AND a dataset
    match (all-or-nothing) — this filter uses a **soft** threshold:
    any paper that earned at least *min_score* points is kept, even if
    it only matches a method or only a dataset.

    This prevents pure noise (score 0 = no entity match at all) from
    reaching the LLM context, while preserving partial matches that
    are still useful to the user.

    Args:
        results:   Ranked result rows (must already have ``_score``).
        min_score: Minimum score a paper must have to survive.

    Returns:
        Filtered list — rows with ``_score >= min_score``.
    """
    if not results:
        return []
    return [r for r in results if r.get("_score", 0) >= min_score]


def truncate_to_top_papers(
    results: list[dict[str, Any]],
    max_papers: int = 8,
) -> list[dict[str, Any]]:
    """
    Keep only rows belonging to the top *max_papers* highest-scoring papers.

    Assumes *results* are already sorted by descending ``_score``.
    The first *max_papers* unique paper URIs are retained; all rows
    associated with those papers are kept so that the context builder
    can group them properly.

    Args:
        results:    Score-sorted result rows.
        max_papers: Maximum number of distinct papers to keep.

    Returns:
        Truncated list of result rows.
    """
    seen: list[str] = []
    paper_set: set[str] = set()

    for row in results:
        uri = row.get("paper", "")
        if uri and uri not in paper_set:
            seen.append(uri)
            paper_set.add(uri)
        if len(seen) >= max_papers:
            break

    return [r for r in results if r.get("paper", "") in paper_set]
