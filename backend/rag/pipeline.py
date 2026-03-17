"""
Main RAG pipeline orchestrator.

Implements the full 10-step GraphRAG pipeline:

  1.  Query Classification       → query_type
  2.  Entity Extraction          → typed entities (methods, datasets, tasks, ...)
  3.  Entity Normalization       → expanded variants (CNN → convolutional neural network, ...)
  4.  Multi-Strategy Retrieval   → combined + per-entity SPARQL queries (async parallel)
  5.  Retrieval Ranking          → score & sort papers by relevance
  6.  Soft Score Filtering       → drop papers with score == 0 (keep partial matches)
  7.  Truncation                 → top N papers for context window
  8.  Context Construction       → structured per-paper blocks with scores
  9.  LLM Generation            → grounded answer (cites papers by title, not number)
  10. Source Formatting          → citations + metadata

Returns a structured JSON response with all pipeline artifacts
(entities, SPARQL queries used, answer, sources).
"""

import asyncio
from typing import Any
from loguru import logger

from backend.config import settings
from backend.rag.query_classifier import classify_query, QueryType
from backend.rag.entity_extractor import extract_entities, ExtractedEntities
from backend.rag.entity_extractor import extract_keywords
from backend.rag.entity_normalization import expand_entities, ExpandedEntities
from backend.rag.query_builder import retrieve_async, RetrievalResult
from backend.rag.ranking import rank_results, hard_filter, soft_filter, truncate_to_top_papers
from backend.rag.context_builder import build_context, format_sources
from backend.kg.sparql_client import SPARQLClient
from backend.llm.ollama_client import OllamaClient, ollama_client, get_prompt_template


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
        │  Filter     │ → drop score-0 noise, keep partial matches
        └─────┬──────┘
        ┌─────▼──────┐
        │  Truncate   │ → top 8 papers for context window
        └─────┬──────┘
        ┌─────▼──────┐
        │  Context    │ → structured Paper: blocks with scores
        └─────┬──────┘
        ┌─────▼──────┐
        │  Generate   │ → LLM answer (cites by title, not [1] [2])
        └─────┬──────┘
        ┌─────▼──────┐
        │  Sources    │ → deduplicated paper references
        └─────┴──────┘

    Usage:
        pipeline = RAGPipeline()
        result = await pipeline.ask("Which papers use CNN on MNIST?")
    """

    def __init__(
        self,
        sparql_client: SPARQLClient | None = None,
        llm_client: OllamaClient | None = None,
    ) -> None:
        self.sparql = sparql_client or SPARQLClient()
        self.llm = llm_client or ollama_client

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
        loop = asyncio.get_running_loop()
        query_type, entities = await asyncio.gather(
            loop.run_in_executor(None, classify_query, question),
            loop.run_in_executor(None, extract_entities, question),
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
        expanded = expand_entities(entities)
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
        filtered = hard_filter(ranked, expanded)
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

        # ── Step 8: Build context (includes scores) ──
        context = build_context(truncated, query_type.value)
        logger.debug(f"Step 8 — Context: {len(context)} chars")

        # ── Step 9: LLM generation ──
        prompt_template = get_prompt_template(query_type.value)
        prompt = prompt_template.format(context=context, question=question)
        answer = self.llm.generate(prompt)
        logger.info(f"Step 9 — Answer: {len(answer)} chars")

        # ── Step 10: Format sources ──
        sources = format_sources(truncated)

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
