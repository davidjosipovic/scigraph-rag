"""
Retrieval ranking, hard filtering, and truncation.

Scoring system:
    +2  method match   (paper's contribution entities contain the queried method)
    +2  dataset match  (paper's contribution entities contain the queried dataset)
    +1  keyword match  (paper title contains any query keyword)

Hard filtering:
    When the user query contains **both** method(s) AND dataset(s), only
    papers matching at least one method AND at least one dataset are kept.
    This eliminates noise like "fake news detection" when the user asked
    about "CNN on MNIST".

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

from typing import Any

from backend.rag.entity_normalization import ExpandedEntities


# ── Score weights ────────────────────────────────────────────────

SCORE_METHOD_MATCH = 2
SCORE_DATASET_MATCH = 2
SCORE_TITLE_KEYWORD = 1


def rank_results(
    results: list[dict[str, Any]],
    entities: ExpandedEntities,
) -> list[dict[str, Any]]:
    """
    Score and rank deduplicated paper results.

    Each paper dict may contain: paper, title, doi, methodLabel,
    methodALabel, methodBLabel, datasetLabel, entityLabel, etc.

    The function:
      1. Groups results by paper URI (collecting all entity labels)
      2. Scores each paper against the queried entities
      3. Returns results sorted by descending score

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

    # Collect per-paper information for scoring
    paper_info: dict[str, _PaperScore] = {}

    for row in results:
        uri = row.get("paper", "")
        if not uri:
            continue

        if uri not in paper_info:
            paper_info[uri] = _PaperScore(
                title=(row.get("title") or "").lower(),
            )

        info = paper_info[uri]

        # Collect method labels from this row
        for key in ("methodLabel", "methodALabel", "methodBLabel"):
            val = row.get(key)
            if val:
                info.method_labels.add(val.lower())

        # Collect dataset labels
        val = row.get("datasetLabel")
        if val:
            info.dataset_labels.add(val.lower())

        # Broad entity label (from fallback queries)
        val = row.get("entityLabel")
        if val:
            info.method_labels.add(val.lower())

    # Score each paper
    query_methods = {m.lower() for m in entities.methods}
    query_datasets = {d.lower() for d in entities.datasets}
    query_keywords = {e.lower() for e in entities.all_entities()}

    # Include variants in matching.
    # Skip 1–2 char variants (e.g. "dl", "ml", "rl") — too short for safe
    # substring matching and would produce false positives on unrelated labels.
    # The canonical form (index 0 from all_method_forms) is always kept.
    variant_methods: set[str] = set()
    for m in entities.methods:
        canonical, *variants = entities.all_method_forms(m)
        variant_methods.add(canonical.lower())
        variant_methods.update(v.lower() for v in variants if len(v) >= 3)
    variant_datasets: set[str] = set()
    for d in entities.datasets:
        canonical, *variants = entities.all_dataset_forms(d)
        variant_datasets.add(canonical.lower())
        variant_datasets.update(v.lower() for v in variants if len(v) >= 3)

    all_match_methods = query_methods | variant_methods
    all_match_datasets = query_datasets | variant_datasets

    uri_scores: dict[str, int] = {}
    for uri, info in paper_info.items():
        score = 0

        # Method match: +2 per matching method
        for label in info.method_labels:
            if any(m in label for m in all_match_methods):
                score += SCORE_METHOD_MATCH
                break  # one match is enough per paper

        # Dataset match: +2 per matching dataset
        for label in info.dataset_labels:
            if any(d in label for d in all_match_datasets):
                score += SCORE_DATASET_MATCH
                break

        # Title keyword match: +1 if any entity keyword appears in title
        for kw in query_keywords:
            if kw in info.title:
                score += SCORE_TITLE_KEYWORD
                break

        uri_scores[uri] = score

    # Annotate each result row with its paper's score
    for row in results:
        uri = row.get("paper", "")
        row["_score"] = uri_scores.get(uri, 0)

    # Sort by score descending, then by title alphabetically for stability
    ranked = sorted(
        results,
        key=lambda r: (-r.get("_score", 0), (r.get("title") or "").lower()),
    )

    return ranked


# ── Internal helper ──────────────────────────────────────────────


class _PaperScore:
    """Temporary accumulator for per-paper scoring data."""

    __slots__ = ("title", "method_labels", "dataset_labels")

    def __init__(self, title: str) -> None:
        self.title = title
        self.method_labels: set[str] = set()
        self.dataset_labels: set[str] = set()


# ── Hard filter ──────────────────────────────────────────────────


def hard_filter(
    results: list[dict[str, Any]],
    entities: ExpandedEntities,
) -> list[dict[str, Any]]:
    """
    Eliminate papers that do NOT match **both** a queried method AND a
    queried dataset when the user's question mentions both.

    If the query contains only methods (no datasets) or only datasets
    (no methods), no filtering is applied — all papers are kept.

    Matching uses the same substring logic and variant expansion as
    ``rank_results`` so that synonym forms count.

    Args:
        results:  Ranked (and possibly scored) list of SPARQL result rows.
        entities: Expanded entities from the normalization step.

    Returns:
        Filtered list — only papers satisfying both conditions.
    """
    if not entities.methods or not entities.datasets:
        return results  # nothing to hard-filter

    # Build match sets (canonical + variants, lowercased).
    # Short variants (< 3 chars) are excluded to prevent false-positive
    # substring matches (e.g. "ml" in "html").
    all_methods: set[str] = set()
    for m in entities.methods:
        canonical, *variants = entities.all_method_forms(m)
        all_methods.add(canonical.lower())
        all_methods.update(v.lower() for v in variants if len(v) >= 3)
    all_datasets: set[str] = set()
    for d in entities.datasets:
        canonical, *variants = entities.all_dataset_forms(d)
        all_datasets.add(canonical.lower())
        all_datasets.update(v.lower() for v in variants if len(v) >= 3)

    # Collect labels per paper URI from ALL rows
    paper_method_labels: dict[str, set[str]] = {}
    paper_dataset_labels: dict[str, set[str]] = {}

    for row in results:
        uri = row.get("paper", "")
        if not uri:
            continue
        paper_method_labels.setdefault(uri, set())
        paper_dataset_labels.setdefault(uri, set())

        for key in ("methodLabel", "methodALabel", "methodBLabel", "entityLabel"):
            val = row.get(key)
            if val:
                paper_method_labels[uri].add(val.lower())

        for key in ("datasetLabel", "entityLabel"):
            val = row.get(key)
            if val:
                paper_dataset_labels[uri].add(val.lower())

    # Determine which papers pass both conditions
    passing: set[str] = set()
    for uri in paper_method_labels:
        has_method = any(
            m in label
            for label in paper_method_labels[uri]
            for m in all_methods
        )
        has_dataset = any(
            d in label
            for label in paper_dataset_labels.get(uri, set())
            for d in all_datasets
        )
        if has_method and has_dataset:
            passing.add(uri)

    return [r for r in results if r.get("paper", "") in passing]


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
