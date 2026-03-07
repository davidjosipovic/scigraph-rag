#!/usr/bin/env python3
"""
Interactive demo of the improved KG-RAG pipeline.

Shows: entity extraction → normalization → query classification
→ combined SPARQL + variant queries → ranking → hard filtering
→ truncation → structured context with scores → hybrid retrieval (RRF).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.rag.query_classifier import classify_query
from backend.rag.entity_extractor import extract_entities
from backend.rag.entity_normalization import expand_entities
from backend.rag.ranking import rank_results, hard_filter, truncate_to_top_papers
from backend.rag.context_builder import build_paper_context
from backend.rag.hybrid_retrieval import reciprocal_rank_fusion
from backend.kg.queries import (
    papers_by_method,
    papers_by_method_and_dataset,
    papers_comparing_methods,
    papers_by_dataset,
    claim_evidence,
    broad_entity_search,
)

DEMO_QUESTIONS = [
    "Which papers use CNN on MNIST?",
    "Compare CNN and LSTM for image classification",
    "Which papers use the MNIST dataset?",
    "Is transfer learning effective for medical imaging?",
    "How is BERT used in named entity recognition?",
    "Find papers by Yoshua Bengio on deep learning",
]


def separator(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def demo_classification() -> None:
    separator("Step 1 — Query Classification (6 types)")
    for q in DEMO_QUESTIONS:
        qt = classify_query(q)
        print(f"  [{qt.upper():^24s}]  {q}")


def demo_entity_extraction_and_normalization() -> None:
    separator("Step 2 — Entity Extraction + Normalization")
    for q in DEMO_QUESTIONS:
        entities = extract_entities(q)
        expanded = expand_entities(entities)
        parts = []
        if expanded.methods:
            parts.append(f"methods={expanded.methods}")
        if expanded.datasets:
            parts.append(f"datasets={expanded.datasets}")
        if expanded.tasks:
            parts.append(f"tasks={expanded.tasks}")
        # Show variants
        variant_parts = []
        for m, variants in expanded.method_variants.items():
            variant_parts.append(f"{m} → {variants[:2]}")
        for d, variants in expanded.dataset_variants.items():
            variant_parts.append(f"{d} → {variants[:2]}")
        summary = ", ".join(parts) if parts else "(no entities)"
        print(f"  Q: {q}")
        print(f"     Entities: {summary}")
        if variant_parts:
            print(f"     Variants: {'; '.join(variant_parts)}")


def demo_combined_sparql() -> None:
    separator("Step 3 — Combined SPARQL Generation")

    # Combined method + dataset query
    print("\n  papers_by_method_and_dataset('CNN', 'MNIST'):")
    sparql = papers_by_method_and_dataset("CNN", "MNIST")
    print(f"    {sparql[:250]}...")

    # Method comparison
    print("\n  papers_comparing_methods('CNN', 'LSTM'):")
    sparql2 = papers_comparing_methods("CNN", "LSTM")
    print(f"    {sparql2[:250]}...")

    # Individual method (fallback)
    print("\n  papers_by_method('CNN'):")
    sparql3 = papers_by_method("CNN")
    print(f"    {sparql3[:200]}...")


def demo_ranking() -> None:
    separator("Step 4 — Ranking + Hard Filtering + Truncation")

    fake_results = [
        {"paper": "http://example.org/R1", "title": "Random Paper", "doi": "N/A"},
        {"paper": "http://example.org/R2", "title": "CNN for image tasks", "doi": "N/A", "methodLabel": "CNN"},
        {"paper": "http://example.org/R3", "title": "CNN on MNIST benchmark", "doi": "10.1/test", "methodLabel": "CNN", "datasetLabel": "MNIST"},
        {"paper": "http://example.org/R4", "title": "SVM experiments", "doi": "N/A", "methodLabel": "SVM"},
        {"paper": "http://example.org/R5", "title": "Fake news detection via CNN", "doi": "N/A", "methodLabel": "CNN"},
    ]

    entities = extract_entities("Which papers use CNN on MNIST?")
    expanded = expand_entities(entities)

    # Rank
    ranked = rank_results(fake_results, expanded)
    print("  After ranking:")
    for r in ranked:
        print(f"    score={r['_score']}  {r['title']}")

    # Hard filter — only papers with BOTH method AND dataset
    filtered = hard_filter(ranked, expanded)
    print(f"\n  After hard filter (require CNN + MNIST): {len(ranked)} → {len(filtered)}")
    for r in filtered:
        print(f"    score={r['_score']}  {r['title']}")

    # Truncation
    truncated = truncate_to_top_papers(filtered, max_papers=3)
    print(f"\n  After truncation (top 3): {len(truncated)} rows")


def demo_structured_context() -> None:
    separator("Step 5 — Structured Context with Scores")

    results = [
        {
            "paper": "http://example.org/R1",
            "title": "Effective Handwritten Digit Recognition using DCNN",
            "doi": "10.48550/arXiv.2004.00331",
            "methodLabel": "CNN",
            "datasetLabel": "MNIST",
            "_score": 5,
        },
        {
            "paper": "http://example.org/R2",
            "title": "Multi-column Deep Neural Networks for Image Classification",
            "doi": "N/A",
            "methodLabel": "CNN",
            "datasetLabel": "MNIST",
            "_score": 4,
        },
    ]
    context = build_paper_context(results)
    print(context)


def demo_hybrid_retrieval() -> None:
    separator("Step 6 — Hybrid Retrieval: Reciprocal Rank Fusion")

    kg_results = [
        {"paper": "http://example.org/R1", "title": "Paper A (KG rank 1)"},
        {"paper": "http://example.org/R2", "title": "Paper B (KG rank 2)"},
        {"paper": "http://example.org/R3", "title": "Paper C (KG rank 3)"},
    ]
    vector_results = [
        {"paper": "http://example.org/R2", "title": "Paper B (Vector rank 1)"},
        {"paper": "http://example.org/R4", "title": "Paper D (Vector rank 2)"},
        {"paper": "http://example.org/R1", "title": "Paper A (Vector rank 3)"},
    ]

    merged = reciprocal_rank_fusion(kg_results, vector_results)
    print("  KG results:     R1, R2, R3")
    print("  Vector results:  R2, R4, R1")
    print("  RRF merged (by fused score):")
    for r in merged:
        print(f"    {r['paper'].split('/')[-1]}  rrf={r['_rrf_score']:.4f}  {r['title']}")


def demo_end_to_end() -> None:
    separator("Step 7 — Full 10-Step Pipeline: 'Which papers use CNN on MNIST?'")
    question = "Which papers use CNN on MNIST?"
    print(f"\n  Question: {question}")

    query_type = classify_query(question)
    print(f"  Query Type: {query_type}")

    entities = extract_entities(question)
    expanded = expand_entities(entities)
    print(f"  Methods: {expanded.methods}")
    print(f"  Datasets: {expanded.datasets}")
    print(f"  Method Variants: {expanded.method_variants}")
    print(f"  Dataset Variants: {expanded.dataset_variants}")

    print(f"\n  Pipeline steps:")
    print(f"    1. Classify      → {query_type.value}")
    print(f"    2. Extract       → methods={expanded.methods}, datasets={expanded.datasets}")
    print(f"    3. Normalize     → {len(expanded.method_variants)} method variants, {len(expanded.dataset_variants)} dataset variants")
    print(f"    4. Retrieve      → asyncio.gather() parallel SPARQL (LIMIT 5 each, timeout 10s)")
    print(f"    5. Rank          → +2 method, +2 dataset, +1 title keyword")
    print(f"    6. Hard filter   → require BOTH CNN + MNIST (eliminates noise)")
    print(f"    7. Truncate      → top 8 papers for context window")
    print(f"    8. Context       → structured Paper: blocks with Relevance Score")
    print(f"    9. Generate      → LLM answer (cites by title, not [1])")
    print(f"   10. Sources       → deduplicated paper references")


if __name__ == "__main__":
    print("=" * 70)
    print("  KG-RAG — Improved Pipeline Demo")
    print("=" * 70)

    demo_classification()
    demo_entity_extraction_and_normalization()
    demo_combined_sparql()
    demo_ranking()
    demo_structured_context()
    demo_hybrid_retrieval()
    demo_end_to_end()

    print(f"\n{'=' * 70}")
    print("  Demo complete. Run 'make run' to start the API server.")
    print(f"{'=' * 70}\n")
