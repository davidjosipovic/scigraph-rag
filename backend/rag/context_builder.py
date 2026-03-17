"""
Context builder: converts raw KG retrieval results into structured,
LLM-ready context strings with rich per-paper information.

The builder groups results by paper and produces a clean structured
block per paper so the LLM can reason about each paper individually.

Output format per paper:

    Paper:
    Title: Effective Handwritten Digit Recognition using DCNN
    Method: CNN
    Dataset: MNIST
    DOI: 10.48550/arXiv.2004.00331

Papers are referenced by **title** (not by number) to prevent the LLM
from hallucinating citation numbers that don't correspond to real sources.
"""

import re
from typing import Any
from collections import defaultdict

_YEAR_RE = re.compile(r"^(19|20)\d{2}$")


def build_context(results: list[dict[str, Any]], query_type: str) -> str:
    """
    Build structured LLM context from merged retrieval results.

    Dispatches to the appropriate builder based on query type.
    """
    papers = _group_by_paper(results)
    if query_type == "claim_verification":
        return _build_evidence_context_from_papers(papers)
    return _build_paper_context_from_papers(papers)


def build_context_and_sources(
    results: list[dict[str, Any]], query_type: str
) -> tuple[str, list[dict[str, Any]]]:
    """
    Build LLM context and format sources in a single pass.

    Calls ``_group_by_paper`` once instead of twice, halving the
    grouping work done by ``build_context`` + ``format_sources``.

    Returns:
        (context_string, sources_list)
    """
    papers = _group_by_paper(results)
    if query_type == "claim_verification":
        context = _build_evidence_context_from_papers(papers)
    else:
        context = _build_paper_context_from_papers(papers)
    sources = _format_sources_from_papers(papers)
    return context, sources


def build_paper_context(results: list[dict[str, Any]]) -> str:
    """Build structured context grouped by paper (public API)."""
    return _build_paper_context_from_papers(_group_by_paper(results))


def build_evidence_context(results: list[dict[str, Any]]) -> str:
    """Build structured evidence context for claim verification (public API)."""
    return _build_evidence_context_from_papers(_group_by_paper(results))


def _build_paper_context_from_papers(papers: dict[str, dict]) -> str:
    if not papers:
        return "No relevant papers were found in the knowledge graph."

    blocks: list[str] = []
    for uri, info in papers.items():
        lines: list[str] = ["Paper:"]
        lines.append(f"Title: {info['title']}")
        lines.append(f"URI: {uri}")
        if info["year"]:
            lines.append(f"Year: {info['year']}")
        if info["doi"] != "N/A":
            lines.append(f"DOI: {info['doi']}")
        if info["fields"]:
            lines.append(f"Field: {', '.join(sorted(info['fields']))}")
        if info["tasks"]:
            lines.append(f"Task: {', '.join(sorted(info['tasks']))}")

        triples: list[str] = []
        if info["methods"]:
            for m in sorted(info["methods"]):
                triples.append(f"  Method: {m}")
        if info["datasets"]:
            for d in sorted(info["datasets"]):
                triples.append(f"  Dataset: {d}")
        if info["contributions"]:
            for contrib_name, props in info["contributions"].items():
                for pred, value in props:
                    triple = f"  {pred}: {value}"
                    if triple not in triples:
                        triples.append(triple)
        if triples:
            lines.append("Triples:")
            lines.extend(triples)

        blocks.append("\n".join(lines))

    header = f"=== Retrieved Papers from ORKG ({len(papers)} papers) ===\n"
    return header + "\n\n".join(blocks)


def _build_evidence_context_from_papers(papers: dict[str, dict]) -> str:
    if not papers:
        return "No evidence was found in the knowledge graph for this claim."

    blocks: list[str] = []
    for uri, info in papers.items():
        lines: list[str] = ["Paper:"]
        lines.append(f"Title: {info['title']}")
        if info["doi"] != "N/A":
            lines.append(f"DOI: {info['doi']}")

        if info["contributions"]:
            lines.append("Evidence:")
            for contrib_name, props in info["contributions"].items():
                for pred, value in props:
                    lines.append(f"  {pred}: {value}")
        else:
            if info["methods"]:
                lines.append(f"Method: {', '.join(sorted(info['methods']))}")
            if info["datasets"]:
                lines.append(f"Dataset: {', '.join(sorted(info['datasets']))}")

        blocks.append("\n".join(lines))

    header = f"=== Evidence from ORKG ({len(papers)} sources) ===\n"
    return header + "\n\n".join(blocks)


def format_sources(results: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Extract unique source references from retrieval results (public API)."""
    return _format_sources_from_papers(_group_by_paper(results))


def _format_sources_from_papers(papers: dict[str, dict]) -> list[dict[str, Any]]:
    sources = []
    for uri, info in papers.items():
        source: dict[str, Any] = {
            "title": info["title"],
            "uri": uri,
            "doi": info["doi"],
            "year": info.get("year"),
        }
        if info["methods"]:
            source["methods"] = list(info["methods"])
        if info["datasets"]:
            source["datasets"] = list(info["datasets"])
        sources.append(source)
    return sources


# ─── INTERNAL HELPERS ────────────────────────────────────────────


def _group_by_paper(results: list[dict[str, Any]]) -> dict[str, dict]:
    """
    Group flat SPARQL result rows into per-paper structured dicts.

    Collects and deduplicates all entity information associated with
    each unique paper URI.

    Returns:
        Ordered dict: paper_uri → {title, doi, methods, datasets, fields,
                                    tasks, contributions}
    """
    papers: dict[str, dict] = {}

    for row in results:
        uri = row.get("paper", "")
        if not uri:
            continue

        if uri not in papers:
            raw_year = row.get("year")
            year = raw_year if (raw_year and _YEAR_RE.match(str(raw_year))) else None
            papers[uri] = {
                "title": row.get("title", "Unknown"),
                "doi": row.get("doi", "N/A"),
                "year": year,
                "score": row.get("_score"),
                "methods": set(),
                "datasets": set(),
                "fields": set(),
                "tasks": set(),
                "contributions": defaultdict(list),
            }

        info = papers[uri]

        # Keep the highest score seen for this paper
        row_score = row.get("_score")
        if row_score is not None:
            if info["score"] is None or row_score > info["score"]:
                info["score"] = row_score

        # Collect methods
        for key in ("methodLabel", "methodALabel", "methodBLabel"):
            val = row.get(key)
            if val:
                info["methods"].add(val)

        # Collect datasets
        val = row.get("datasetLabel")
        if val:
            info["datasets"].add(val)

        # Collect fields
        val = row.get("fieldLabel")
        if val:
            info["fields"].add(val)

        # Collect tasks / research problems
        val = row.get("problemLabel")
        if val:
            info["tasks"].add(val)

        # Collect contributions
        contrib = row.get("contribLabel")
        pred = row.get("predLabel")
        value = row.get("valueLabel")
        if contrib and pred and value:
            pair = (pred, value)
            if pair not in info["contributions"][contrib]:
                info["contributions"][contrib].append(pair)

        # Broad entity label (from fallback queries) — add only to methods
        # as a general catch-all; adding to both caused duplicate noise
        val = row.get("entityLabel")
        if val:
            info["methods"].add(val)

    return papers
