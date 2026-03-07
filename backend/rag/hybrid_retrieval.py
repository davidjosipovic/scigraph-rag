"""
Hybrid retrieval: combines Knowledge Graph (SPARQL) results with
vector-based semantic search over paper titles and abstracts.

This module shows how to integrate a vector store (ChromaDB) alongside
the existing KG retrieval pipeline.  The KG provides structured facts
(methods, datasets, contributions) while the vector store captures
semantic similarity that SPARQL string-matching can miss.

────────────────────────────────────────────────────────────────
ARCHITECTURE:

    User Question
         │
    ┌────▼────┐        ┌────────────┐
    │  KG     │        │  Vector    │
    │  SPARQL │        │  ChromaDB  │
    └────┬────┘        └─────┬──────┘
         │                   │
    ┌────▼───────────────────▼────┐
    │      Reciprocal Rank        │
    │      Fusion  (RRF)          │
    └─────────────┬───────────────┘
                  │
             Merged Results
────────────────────────────────────────────────────────────────

SETUP (one-time):
    pip install chromadb sentence-transformers

    from backend.rag.hybrid_retrieval import VectorStore
    vs = VectorStore()
    vs.index_papers(papers)        # list of {title, abstract, uri}

QUERY:
    from backend.rag.hybrid_retrieval import hybrid_retrieve

    results = hybrid_retrieve(
        question    = "Which papers use CNN on MNIST?",
        kg_results  = kg_rows,        # from SPARQL pipeline
        vector_store = vs,
        top_k       = 8,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from loguru import logger


# ── Constants ────────────────────────────────────────────────────

RRF_K = 60         # Reciprocal Rank Fusion constant (standard value)
VECTOR_TOP_K = 20   # how many candidates to pull from vector search


# ── Vector Store wrapper ─────────────────────────────────────────


@dataclass
class VectorStore:
    """
    Thin wrapper around ChromaDB for paper title / abstract embeddings.

    Usage::

        vs = VectorStore(collection_name="orkg_papers")
        vs.index_papers([
            {"uri": "http://orkg.org/R1", "title": "...", "abstract": "..."},
            ...
        ])
        hits = vs.search("CNN on MNIST", top_k=10)
    """

    collection_name: str = "orkg_papers"
    _client: Any = field(default=None, repr=False)
    _collection: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        try:
            import chromadb  # type: ignore[import-untyped]

            self._client = chromadb.Client()
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                f"ChromaDB collection '{self.collection_name}' ready "
                f"({self._collection.count()} documents)"
            )
        except ImportError:
            logger.warning(
                "chromadb is not installed — vector search disabled.  "
                "Install with: pip install chromadb sentence-transformers"
            )

    @property
    def is_available(self) -> bool:
        return self._collection is not None

    def index_papers(self, papers: list[dict[str, str]]) -> int:
        """
        Add papers to the vector collection.

        Each paper dict should have at least ``uri`` and ``title``.
        An optional ``abstract`` field improves retrieval quality.

        Returns:
            Number of papers indexed.
        """
        if not self.is_available:
            return 0

        ids = [p["uri"] for p in papers]
        documents = [
            f"{p.get('title', '')} {p.get('abstract', '')}".strip()
            for p in papers
        ]
        metadatas = [
            {"title": p.get("title", ""), "uri": p["uri"]}
            for p in papers
        ]

        self._collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )
        logger.info(f"Indexed {len(papers)} papers into ChromaDB")
        return len(papers)

    def search(self, query: str, top_k: int = VECTOR_TOP_K) -> list[dict[str, Any]]:
        """
        Semantic search over indexed papers.

        Returns:
            List of dicts with ``uri``, ``title``, ``distance`` (0 = identical).
        """
        if not self.is_available:
            return []

        results = self._collection.query(
            query_texts=[query],
            n_results=min(top_k, self._collection.count() or 1),
        )

        hits: list[dict[str, Any]] = []
        for uri, meta, dist in zip(
            results["ids"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            hits.append({
                "uri": uri,
                "title": meta.get("title", ""),
                "distance": dist,
            })
        return hits


# ── Reciprocal Rank Fusion ───────────────────────────────────────


def reciprocal_rank_fusion(
    *ranked_lists: list[dict[str, Any]],
    uri_key: str = "paper",
    k: int = RRF_K,
) -> list[dict[str, Any]]:
    """
    Merge multiple ranked result lists using Reciprocal Rank Fusion.

    RRF assigns each document a score of ``1 / (k + rank)`` across all
    lists and sums them.  This is robust to score-scale differences
    between KG retrieval and vector search.

    Args:
        ranked_lists: Variable number of ranked result lists.
        uri_key:      Key used for document identity (default ``"paper"``).
        k:            RRF constant (default 60).

    Returns:
        Merged list sorted by descending RRF score, each annotated
        with ``_rrf_score``.
    """
    scores: dict[str, float] = {}
    best_row: dict[str, dict[str, Any]] = {}

    for ranked in ranked_lists:
        for rank, row in enumerate(ranked, start=1):
            uri = row.get(uri_key, row.get("uri", ""))
            if not uri:
                continue
            scores[uri] = scores.get(uri, 0.0) + 1.0 / (k + rank)
            # Keep the row with the most metadata
            if uri not in best_row or len(row) > len(best_row[uri]):
                best_row[uri] = row

    merged = []
    for uri in sorted(scores, key=scores.get, reverse=True):
        entry = dict(best_row[uri])
        entry["_rrf_score"] = round(scores[uri], 6)
        merged.append(entry)

    return merged


# ── Hybrid retrieve ──────────────────────────────────────────────


def hybrid_retrieve(
    question: str,
    kg_results: list[dict[str, Any]],
    vector_store: VectorStore | None = None,
    top_k: int = 8,
) -> list[dict[str, Any]]:
    """
    Combine KG (SPARQL) results with vector search results.

    If ChromaDB is not available, falls back to pure KG results.

    The merging uses Reciprocal Rank Fusion so that a paper
    appearing high in both KG and vector rankings gets the best
    combined score.

    Args:
        question:      Original user question.
        kg_results:    Ranked results from the SPARQL pipeline.
        vector_store:  Optional VectorStore instance.
        top_k:         Maximum number of papers to return.

    Returns:
        Merged and re-ranked list of paper dicts.
    """
    if not vector_store or not vector_store.is_available:
        logger.debug("Vector store not available — using KG results only")
        return kg_results[:top_k]

    # Get vector hits
    vector_hits = vector_store.search(question, top_k=VECTOR_TOP_K)

    # Normalize vector hits to look like KG rows (paper key = URI)
    vector_rows = [
        {"paper": h["uri"], "title": h["title"], "_vector_distance": h["distance"]}
        for h in vector_hits
    ]

    # Fuse the two ranked lists
    merged = reciprocal_rank_fusion(kg_results, vector_rows, uri_key="paper")

    logger.info(
        f"Hybrid retrieval: {len(kg_results)} KG + {len(vector_hits)} vector "
        f"→ {len(merged)} merged (returning top {top_k})"
    )

    return merged[:top_k]
