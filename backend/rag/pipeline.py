"""
Main RAG pipeline orchestrator.

Implements the 9-stage GraphRAG pipeline:

  1.  Query Classification       → query_type
  2.  Entity Extraction          → typed entities (methods, datasets, tasks, ...)
  3.  Entity Normalization       → expanded variants (CNN → convolutional neural network, ...)
  4.  Multi-Strategy Retrieval   → combined + per-entity SPARQL queries (async parallel)
  5.  Retrieval Ranking          → score & sort papers by relevance
  6.  Filtering                  → hard filter (method+dataset) + soft filter (score-0 noise)
  7.  Truncation                 → top N papers for context window
  8.  Context + Source Building  → structured per-paper blocks + deduplicated citations
  9.  LLM Generation             → grounded answer (cites papers by title, not number)

Returns a structured JSON response with all pipeline artifacts
(entities, SPARQL queries used, answer, sources).
"""

import asyncio
from functools import partial
from typing import Any
from loguru import logger

from backend.config import settings
from backend.rag.query_classifier import classify_query, QueryType
from backend.rag.entity_extractor import extract_entities, ExtractedEntities
from backend.rag.entity_extractor import extract_keywords
from backend.rag.entity_normalization import expand_entities, ExpandedEntities
from backend.rag.query_builder import retrieve_async, RetrievalResult
from backend.rag.ranking import rank_results, hard_filter, soft_filter, truncate_to_top_papers
from backend.rag.context_builder import build_context_and_sources
from backend.kg.sparql_client import SPARQLClient
from backend.llm.base import BaseLLMClient, get_prompt_template
from backend.llm.factory import create_llm_client


class RAGPipeline:
    """
    GraphRAG pipeline for answering questions over scientific papers.

    Pipeline steps:
        User Question
              │
        ┌─────▼──────┐
        │  Classify   │ → query_type (6 types)
        └─────┬──────┘
        ┌─────▼──────┐
        │  Extract    │ → entities {methods, datasets, tasks, fields, metrics}
        └─────┬──────┘
        ┌─────▼──────┐
        │  Normalize  │ → expand entities with synonym variants
        └─────┬──────┘
        ┌─────▼──────┐
        │  Retrieve   │ → parallel SPARQL via asyncio.gather()
        └─────┬──────┘
        ┌─────▼──────┐
        │  Rank       │ → score papers (+2 method, +2 dataset, +1 title kw)
        └─────┬──────┘
        ┌─────▼──────┐
        │  Filter     │ → hard filter (method+dataset) + soft filter (score-0 noise)
        └─────┬──────┘
        ┌─────▼──────┐
        │  Truncate   │ → top 8 papers for context window
        └─────┬──────┘
        ┌─────▼──────┐
        │  Context    │ → structured Paper: blocks + deduplicated sources
        └─────┬──────┘
        ┌─────▼──────┐
        │  Generate   │ → LLM answer (cites by title, not [1] [2])
        └─────┴──────┘

    Usage:
        pipeline = RAGPipeline()
        result = await pipeline.ask("Which papers use CNN on MNIST?")

    Ablation flags (for evaluation experiments — see eval/ablation.py):
        enable_normalization=False  skips synonym expansion; only the
                                     canonical extracted term is matched.
        enable_hard_filter=False    skips the method+dataset hard filter;
                                     only the score-0 soft filter still runs.
    """

    def __init__(
        self,
        sparql_client: SPARQLClient | None = None,
        llm_client: BaseLLMClient | None = None,
        enable_normalization: bool = True,
        enable_hard_filter: bool = True,
    ) -> None:
        self.sparql = sparql_client or SPARQLClient()
        self.llm = llm_client or create_llm_client()
        self.enable_normalization = enable_normalization
        self.enable_hard_filter = enable_hard_filter

    async def ask(self, question: str) -> dict[str, Any]:
        """
        Process a user question through the full GraphRAG pipeline.

        This method is **async** — SPARQL queries are executed in
        parallel via ``asyncio.gather()`` for reduced latency.

        Args:
            question: Natural language question about scientific papers.

        Returns:
            Structured dict with:
                - question:          Original question
                - query_type:        Detected query type
                - entities:          Extracted entities {methods, datasets, ...}
                - sparql_queries:    List of SPARQL queries executed
                - answer:            LLM-generated answer
                - sources:           Deduplicated cited papers
                - kg_results_count:  Total KG results retrieved
                - strategies_used:   Retrieval strategies that were used
        """
        logger.info(f"Processing question: {question}")

        # ── Steps 1 + 2: Classify and extract entities in parallel ──
        # Both calls take the same input and are independent — run concurrently
        # to halve the LLM latency for this stage.
        # partial() is used because run_in_executor does not support kwargs,
        # and both functions require the llm client as a second argument.
        loop = asyncio.get_running_loop()
        query_type, entities = await asyncio.gather(
            loop.run_in_executor(None, partial(classify_query, question, self.llm)),
            loop.run_in_executor(None, partial(extract_entities, question, self.llm)),
        )
        logger.info(f"Step 1 — Query type: {query_type.value}")
        logger.info(
            f"Step 2 — Entities: methods={entities.methods}, "
            f"datasets={entities.datasets}, tasks={entities.tasks}, "
            f"fields={entities.fields}, metrics={entities.metrics}"
        )

        # ── Step 2b: Keyword fallback when NER is empty ──
        if entities.is_empty():
            keywords = extract_keywords(question)
            logger.info(
                f"Step 2b — NER empty, keyword fallback: {keywords}"
            )
            if keywords:
                # Treat extracted keyword phrases as candidate methods
                # so they flow through normalization → SPARQL queries
                entities = ExtractedEntities(methods=keywords)

        # ── Step 3: Normalize / expand entities ──
        # (ablation: enable_normalization=False keeps only the canonical
        # extracted term, with no synonym variants, for comparison runs)
        if self.enable_normalization:
            expanded = expand_entities(entities)
        else:
            expanded = ExpandedEntities(
                methods=list(entities.methods),
                datasets=list(entities.datasets),
                tasks=list(entities.tasks),
                fields=list(entities.fields),
                metrics=list(entities.metrics),
            )
        logger.info(
            f"Step 3 — Normalization: "
            f"method_variants={expanded.method_variants}, "
            f"dataset_variants={expanded.dataset_variants}"
        )

        # ── Step 4: Multi-strategy retrieval (parallel via asyncio.gather) ──
        retrieval = await retrieve_async(question, query_type, expanded, self.sparql)
        logger.info(
            f"Step 4 — Retrieval: {retrieval.total_results} results from "
            f"{len(retrieval.sparql_queries)} queries"
        )

        # ── Step 5: Rank results (uses ALL rows, not deduped) ──
        ranked = rank_results(retrieval.results, expanded)
        logger.info(
            f"Step 5 — Ranking: {len(ranked)} rows, "
            f"top score={ranked[0].get('_score', 0) if ranked else 0}"
        )

        # ── Step 6: Filter results ──
        # Hard filter first: when both methods and datasets are present,
        # drop papers that don't match at least one of each.
        # Soft filter after: drop any remaining score-0 noise.
        # (ablation: enable_hard_filter=False skips the hard filter step)
        filtered = hard_filter(ranked, expanded) if self.enable_hard_filter else ranked
        filtered = soft_filter(filtered, min_score=1)
        logger.info(
            f"Step 6 — Filter: {len(ranked)} → {len(filtered)} rows "
            f"(dropped {len(ranked) - len(filtered)} noise rows)"
        )

        # ── Step 7: Truncate to top N papers ──
        truncated = truncate_to_top_papers(filtered, settings.max_context_papers)
        logger.info(
            f"Step 7 — Truncation: top {settings.max_context_papers} papers"
        )

        # ── Steps 8 + 10: Build context and format sources in one pass ──
        context, sources = build_context_and_sources(truncated, query_type.value)
        logger.debug(f"Step 8 — Context: {len(context)} chars")

        # ── Step 9: LLM generation (run in executor to avoid blocking event loop) ──
        prompt_template = get_prompt_template(query_type.value)
        prompt = prompt_template.format(context=context, question=question)
        answer = await loop.run_in_executor(None, self.llm.generate, prompt)
        logger.info(f"Step 9 — Answer: {len(answer)} chars")

        return {
            "question": question,
            "query_type": query_type.value,
            "entities": entities.to_dict(),
            "sparql_queries": retrieval.sparql_queries,
            "strategies_used": retrieval.strategies_used,
            "answer": answer,
            "sources": sources,
            "kg_results_count": retrieval.total_results,
        }

    def health_check(self) -> dict[str, Any]:
        """
        Check the health of all pipeline components.

        Returns:
            Dict with component health status.
        """
        llm_ok = self.llm.is_available()

        sparql_ok = False
        try:
            test_results = self.sparql.execute(
                "SELECT ?s WHERE { ?s ?p ?o } LIMIT 1"
            )
            sparql_ok = len(test_results) > 0
        except Exception as e:
            logger.error(f"SPARQL health check failed: {e}")

        return {
            "llm": {"status": "ok" if llm_ok else "error", "model": self.llm.model},
            "sparql": {"status": "ok" if sparql_ok else "error"},
            "pipeline": "ready" if (llm_ok and sparql_ok) else "degraded",
        }
