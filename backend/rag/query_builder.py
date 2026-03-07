"""
Multi-strategy query builder: generates multiple SPARQL queries per question
and merges results from graph-based retrieval strategies.

DESIGN PRINCIPLE: Instead of generating a single SPARQL query that searches
paper titles, this module generates multiple targeted queries that traverse
the knowledge graph through different relationship paths:

  - combined:      Paper → Contribution → Method + Dataset (both required)
  - method-based:  Paper → Contribution → Method entity
  - dataset-based: Paper → Contribution → Dataset entity
  - task-based:    Paper → Contribution → Research problem
  - field-based:   Paper → Research field
  - comparison:    Paper → Contribution → MethodA + MethodB
  - claim evidence: Paper → Contribution → entity matching claim terms
  - title fallback: Paper → title (only as last resort)

The retrieval strategy is selected based on query type and extracted entities.
Entity normalization is applied to search using synonym variants.

Async support:
    ``retrieve_async()`` executes all planned SPARQL queries in parallel
    using ``asyncio.gather()`` for significantly lower latency.
"""

from __future__ import annotations

import asyncio
from typing import Any
from loguru import logger

from backend.config import settings
from backend.rag.query_classifier import QueryType
from backend.rag.entity_normalization import ExpandedEntities
from backend.kg import queries as Q
from backend.kg.sparql_client import SPARQLClient, SPARQLTimeoutError


class RetrievalResult:
    """Container for multi-query retrieval results."""

    def __init__(self) -> None:
        self.results: list[dict[str, Any]] = []
        self.sparql_queries: list[str] = []
        self.strategies_used: list[str] = []

    def add(
        self,
        rows: list[dict[str, Any]],
        query: str,
        strategy: str,
    ) -> None:
        """Add results from one retrieval query."""
        self.results.extend(rows)
        self.sparql_queries.append(query.strip())
        self.strategies_used.append(strategy)

    @property
    def total_results(self) -> int:
        return len(self.results)

    def deduplicated_papers(self) -> list[dict[str, Any]]:
        """Return results deduplicated by paper URI."""
        seen: set[str] = set()
        unique: list[dict[str, Any]] = []
        for row in self.results:
            uri = row.get("paper", "")
            if uri and uri not in seen:
                seen.add(uri)
                unique.append(row)
        return unique


# ─── PUBLIC API ──────────────────────────────────────────────────


def retrieve(
    question: str,
    query_type: QueryType,
    entities: ExpandedEntities,
    sparql_client: SPARQLClient,
) -> RetrievalResult:
    """
    Execute a multi-strategy retrieval **synchronously**.

    This is the main entry point for synchronous callers.  It selects
    retrieval strategies, generates SPARQL queries, and runs them one
    by one against the SPARQL endpoint.

    Args:
        question:      Original user question.
        query_type:    Classified query type.
        entities:      Expanded entities with synonym variants.
        sparql_client: SPARQL client for executing queries.

    Returns:
        RetrievalResult with merged results from all strategies.
    """
    limit = settings.max_results
    result = RetrievalResult()

    # Plan queries
    planned = _plan_queries(query_type, entities, limit)

    # Execute sequentially
    for query, strategy in planned:
        try:
            rows = sparql_client.execute(query)
        except SPARQLTimeoutError:
            logger.warning(
                f"Timeout on {strategy} — falling back to title search"
            )
            fallback_q = _title_fallback_for(entities, limit)
            if fallback_q:
                rows = sparql_client.execute(fallback_q)
                result.add(rows, fallback_q, f"timeout_fallback({strategy})")
            continue
        result.add(rows, query, strategy)

    # Fallback if empty
    if result.total_results == 0:
        logger.info("Primary strategies returned 0 results, trying fallbacks...")
        fallback = _plan_fallback(entities, limit)
        for query, strategy in fallback:
            rows = sparql_client.execute(query)
            result.add(rows, query, strategy)

    logger.info(
        f"Retrieval complete: {result.total_results} total results from "
        f"{len(result.sparql_queries)} queries "
        f"(strategies: {result.strategies_used})"
    )

    return result


async def retrieve_async(
    question: str,
    query_type: QueryType,
    entities: ExpandedEntities,
    sparql_client: SPARQLClient,
) -> RetrievalResult:
    """
    Execute a multi-strategy retrieval **asynchronously**.

    All planned SPARQL queries are fired in parallel using
    ``asyncio.gather()``, which significantly reduces total latency
    when the ORKG endpoint is the bottleneck.

    Individual query failures (timeouts, network errors) are caught
    and logged — they do not abort the entire retrieval batch.

    Args:
        question:      Original user question.
        query_type:    Classified query type.
        entities:      Expanded entities with synonym variants.
        sparql_client: SPARQL client (must provide ``execute_async``).

    Returns:
        RetrievalResult with merged results from all strategies.
    """
    limit = settings.max_results
    result = RetrievalResult()

    # Plan queries
    planned = _plan_queries(query_type, entities, limit)

    # Execute ALL primary queries in parallel
    tasks = [sparql_client.execute_async(q) for q, _ in planned]
    all_rows = await asyncio.gather(*tasks, return_exceptions=True)

    for (query, strategy), rows in zip(planned, all_rows):
        if isinstance(rows, SPARQLTimeoutError):
            logger.warning(
                f"Timeout on {strategy} — falling back to title search"
            )
            fallback_q = _title_fallback_for(entities, limit)
            if fallback_q:
                try:
                    fb_rows = await sparql_client.execute_async(fallback_q)
                except Exception as fb_err:
                    logger.warning(f"Title fallback also failed: {fb_err}")
                    fb_rows = []
                result.add(fb_rows, fallback_q, f"timeout_fallback({strategy})")
            else:
                result.add([], query, strategy)
            continue
        if isinstance(rows, BaseException):
            logger.warning(f"Query failed ({strategy}): {rows}")
            result.add([], query, strategy)
            continue
        result.add(rows, query, strategy)

    # Fallback: if nothing found, try broader queries in parallel
    if result.total_results == 0:
        logger.info("Primary strategies returned 0 results, trying fallbacks...")
        fallback = _plan_fallback(entities, limit)
        tasks = [sparql_client.execute_async(q) for q, _ in fallback]
        all_rows = await asyncio.gather(*tasks, return_exceptions=True)

        for (query, strategy), rows in zip(fallback, all_rows):
            if isinstance(rows, BaseException):
                logger.warning(f"Fallback query failed ({strategy}): {rows}")
                result.add([], query, strategy)
                continue
            result.add(rows, query, strategy)

    logger.info(
        f"Async retrieval complete: {result.total_results} total results from "
        f"{len(result.sparql_queries)} queries "
        f"(strategies: {result.strategies_used})"
    )

    return result


# ─── QUERY PLANNER ───────────────────────────────────────────────
# Strategy functions return a list of (sparql_query, strategy_name)
# tuples without executing anything — execution is handled by
# retrieve() / retrieve_async().

PlannedQueries = list[tuple[str, str]]

# Minimum character length for variant forms used in SPARQL CONTAINS.
# Short forms like "dl", "ml", "rl" produce excessive false positives
# because CONTAINS matches substrings (e.g. "dl" matches "WSDL").
_MIN_VARIANT_LEN = 3


def _usable_forms(forms: list[str]) -> list[str]:
    """Filter out variant forms too short for safe CONTAINS matching."""
    return [f for f in forms if len(f) >= _MIN_VARIANT_LEN]


def _plan_queries(
    query_type: QueryType,
    entities: ExpandedEntities,
    limit: int,
) -> PlannedQueries:
    """Select and plan SPARQL queries based on query type."""
    strategy_map = {
        QueryType.METHOD_COMPARISON: _plan_method_comparison,
        QueryType.METHOD_USAGE: _plan_method_usage,
        QueryType.DATASET_SEARCH: _plan_dataset_search,
        QueryType.CLAIM_VERIFICATION: _plan_claim_evidence,
        QueryType.PAPER_LOOKUP: _plan_paper_lookup,
        QueryType.TOPIC_SEARCH: _plan_topic_search,
    }

    plan_fn = strategy_map.get(query_type, _plan_topic_search)
    return plan_fn(entities, limit)


# ─── STRATEGY IMPLEMENTATIONS ───────────────────────────────────


def _plan_method_comparison(
    entities: ExpandedEntities,
    limit: int,
) -> PlannedQueries:
    """
    Plan queries for method_comparison questions.

    1. If 2+ methods: papers mentioning BOTH methods
    2. If methods + dataset: combined query for each method+dataset pair
    3. Each method individually (with variants)
    """
    planned: PlannedQueries = []
    methods = entities.methods

    if len(methods) >= 2:
        query = Q.papers_comparing_methods(methods[0], methods[1], limit)
        planned.append((query, f"comparison({methods[0]},{methods[1]})"))

        for variant in _usable_forms(entities.all_method_forms(methods[0])[1:2]):
            query = Q.papers_comparing_methods(variant, methods[1], limit)
            planned.append((query, f"comparison_variant({variant},{methods[1]})"))

    for method in methods[:2]:
        for dataset in entities.datasets[:2]:
            query = Q.papers_by_method_and_dataset(method, dataset, limit)
            planned.append((query, f"combined({method},{dataset})"))

    for method in methods[:3]:
        for form in _usable_forms(entities.all_method_forms(method)[:2]):
            query = Q.papers_by_method(form, limit)
            planned.append((query, f"method({form})"))

    return planned


def _plan_method_usage(
    entities: ExpandedEntities,
    limit: int,
) -> PlannedQueries:
    """
    Plan queries for method_usage questions.

    1. Combined method+dataset if both present
    2. Method entity labels (with variants)
    3. Research problem / task queries
    """
    planned: PlannedQueries = []

    for method in entities.methods[:2]:
        for dataset in entities.datasets[:2]:
            query = Q.papers_by_method_and_dataset(method, dataset, limit)
            planned.append((query, f"combined({method},{dataset})"))

    for method in entities.methods[:3]:
        for form in _usable_forms(entities.all_method_forms(method)[:2]):
            query = Q.papers_by_method(form, limit)
            planned.append((query, f"method({form})"))

    for task in entities.tasks[:2]:
        query = Q.papers_by_research_problem(task, limit)
        planned.append((query, f"task({task})"))

    for field in entities.fields[:2]:
        query = Q.papers_by_research_field(field, limit)
        planned.append((query, f"field({field})"))

    return planned


def _plan_dataset_search(
    entities: ExpandedEntities,
    limit: int,
) -> PlannedQueries:
    """
    Plan queries for dataset_search questions.

    1. Combined method+dataset if both present
    2. Dataset entity labels (with variants)
    3. Methods if present
    """
    planned: PlannedQueries = []

    for method in entities.methods[:2]:
        for dataset in entities.datasets[:2]:
            query = Q.papers_by_method_and_dataset(method, dataset, limit)
            planned.append((query, f"combined({method},{dataset})"))

    for dataset in entities.datasets[:3]:
        for form in _usable_forms(entities.all_dataset_forms(dataset)[:2]):
            query = Q.papers_by_dataset(form, limit)
            planned.append((query, f"dataset({form})"))

    for method in entities.methods[:2]:
        for form in _usable_forms(entities.all_method_forms(method)[:2]):
            query = Q.papers_by_method(form, limit)
            planned.append((query, f"method({form})"))

    return planned


def _plan_claim_evidence(
    entities: ExpandedEntities,
    limit: int,
) -> PlannedQueries:
    """
    Plan queries for claim_verification questions.

    1. Combined if method+dataset present
    2. Contribution entities matching claim keywords
    3. Individual methods (with variants)
    """
    planned: PlannedQueries = []

    for method in entities.methods[:2]:
        for dataset in entities.datasets[:2]:
            query = Q.papers_by_method_and_dataset(method, dataset, limit)
            planned.append((query, f"combined({method},{dataset})"))

    all_terms = entities.all_entities()
    if all_terms:
        query = Q.claim_evidence(all_terms, limit + 5)
        planned.append((query, f"claim_evidence({all_terms})"))

    for method in entities.methods[:2]:
        for form in _usable_forms(entities.all_method_forms(method)[:2]):
            query = Q.papers_by_method(form, limit)
            planned.append((query, f"method({form})"))

    return planned


def _plan_paper_lookup(
    entities: ExpandedEntities,
    limit: int,
) -> PlannedQueries:
    """
    Plan queries for paper_lookup questions.

    Title search is the primary strategy here.
    """
    planned: PlannedQueries = []
    all_terms = entities.all_entities()

    if all_terms:
        query = Q.paper_lookup_by_title(" ".join(all_terms), limit)
        planned.append((query, f"title_lookup({all_terms})"))

        for term in all_terms[:3]:
            query = Q.paper_lookup_by_title(term, limit)
            planned.append((query, f"title_lookup({term})"))

    return planned


def _plan_topic_search(
    entities: ExpandedEntities,
    limit: int,
) -> PlannedQueries:
    """
    Plan queries for topic_search questions.

    Multi-path retrieval with variant expansion:
    1. Combined method+dataset
    2. Research field
    3. Research problem/task
    4. Method (with variants)
    5. Dataset (with variants)
    """
    planned: PlannedQueries = []

    for method in entities.methods[:2]:
        for dataset in entities.datasets[:2]:
            query = Q.papers_by_method_and_dataset(method, dataset, limit)
            planned.append((query, f"combined({method},{dataset})"))

    for field in entities.fields[:2]:
        query = Q.papers_by_research_field(field, limit)
        planned.append((query, f"field({field})"))

    for task in entities.tasks[:2]:
        query = Q.papers_by_research_problem(task, limit)
        planned.append((query, f"task({task})"))

    for method in entities.methods[:2]:
        for form in _usable_forms(entities.all_method_forms(method)[:2]):
            query = Q.papers_by_method(form, limit)
            planned.append((query, f"method({form})"))

    for dataset in entities.datasets[:2]:
        for form in _usable_forms(entities.all_dataset_forms(dataset)[:2]):
            query = Q.papers_by_dataset(form, limit)
            planned.append((query, f"dataset({form})"))

    return planned


def _plan_fallback(
    entities: ExpandedEntities,
    limit: int,
) -> PlannedQueries:
    """
    Plan fallback queries when primary strategies return nothing.

    1. Broad entity search (any contribution entity matching keywords)
    2. Title keyword search (last resort)
    """
    planned: PlannedQueries = []
    all_terms = entities.all_entities()

    for term in all_terms[:3]:
        query = Q.broad_entity_search(term, limit)
        planned.append((query, f"broad_entity({term})"))

    if all_terms:
        query = Q.title_keyword_search(all_terms, limit)
        planned.append((query, f"title_fallback({all_terms})"))

    return planned


def _title_fallback_for(
    entities: ExpandedEntities,
    limit: int,
) -> str | None:
    """
    Generate a simple ``rdfs:label`` title search to use when a complex
    SPARQL query times out.

    Returns the SPARQL query string, or ``None`` if no keywords are
    available to search for.
    """
    all_terms = entities.all_entities()
    if not all_terms:
        return None
    return Q.title_keyword_search(all_terms[:5], limit)
