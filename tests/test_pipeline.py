"""
Tests for the RAG pipeline components:
  - Query classifier (6 types)
  - Entity extractor
  - Entity normalization
  - Ranking, hard filtering, truncation
  - Context builder (with scores)
  - Prompt templates
  - Async query builder
  - Hybrid retrieval (RRF)
"""

import asyncio

from backend.rag.query_classifier import classify_query, QueryType
from backend.rag.entity_extractor import extract_entities, extract_keywords
from backend.rag.entity_normalization import (
    expand_entities,
    get_method_variants,
    get_dataset_variants,
)
from backend.rag.ranking import rank_results, hard_filter, soft_filter, truncate_to_top_papers
from backend.rag.context_builder import (
    build_paper_context,
    build_evidence_context,
    build_contribution_context,
    format_sources,
)
from backend.rag.hybrid_retrieval import reciprocal_rank_fusion


class TestQueryClassifier:
    """Test 6-type question classification."""

    def test_topic_search(self):
        assert classify_query("Which papers discuss machine learning?") == QueryType.TOPIC_SEARCH
        assert classify_query("Find papers about transformers") == QueryType.TOPIC_SEARCH
        assert classify_query("Research on knowledge graphs") == QueryType.TOPIC_SEARCH

    def test_method_comparison(self):
        assert classify_query("Compare CNN and SVM") == QueryType.METHOD_COMPARISON
        assert classify_query("What is the difference between BERT and GPT?") == QueryType.METHOD_COMPARISON
        assert classify_query("CNN vs RNN for text classification") == QueryType.METHOD_COMPARISON

    def test_dataset_search(self):
        assert classify_query("Papers using the MNIST dataset") == QueryType.DATASET_SEARCH
        assert classify_query("Which benchmark is used for NER?") == QueryType.DATASET_SEARCH
        assert classify_query("Models evaluated on ImageNet") == QueryType.DATASET_SEARCH

    def test_claim_verification(self):
        assert classify_query("Does paper X claim that CNN outperforms SVM?") == QueryType.CLAIM_VERIFICATION
        assert classify_query("Is it true that BERT outperforms GPT on NER?") == QueryType.CLAIM_VERIFICATION
        assert classify_query("Verify that random forests outperform decision trees") == QueryType.CLAIM_VERIFICATION

    def test_method_usage(self):
        assert classify_query("Where is reinforcement learning used in medical diagnosis?") == QueryType.METHOD_USAGE
        assert classify_query("Papers using BERT for sentiment analysis") == QueryType.METHOD_USAGE

    def test_paper_lookup(self):
        assert classify_query("Look up the paper on attention mechanisms") == QueryType.PAPER_LOOKUP
        assert classify_query("Tell me about paper titled attention is all you need") == QueryType.PAPER_LOOKUP


class TestEntityExtractor:
    """Test typed entity extraction."""

    def test_extracts_methods(self):
        entities = extract_entities("Which papers compare CNN and SVM?")
        methods_lower = {m.lower() for m in entities.methods}
        assert "cnn" in methods_lower
        assert "svm" in methods_lower

    def test_extracts_datasets(self):
        entities = extract_entities("Models evaluated on MNIST and CIFAR-10")
        datasets_lower = {d.lower() for d in entities.datasets}
        assert "mnist" in datasets_lower
        assert "cifar-10" in datasets_lower

    def test_extracts_tasks(self):
        entities = extract_entities("Papers on sentiment analysis using BERT")
        tasks_lower = {t.lower() for t in entities.tasks}
        assert "sentiment analysis" in tasks_lower

    def test_extracts_fields(self):
        entities = extract_entities("Research in natural language processing")
        fields_lower = {f.lower() for f in entities.fields}
        assert "natural language processing" in fields_lower

    def test_extracts_metrics(self):
        entities = extract_entities("Models with highest accuracy on MNIST")
        metrics_lower = {m.lower() for m in entities.metrics}
        assert "accuracy" in metrics_lower

    def test_extracts_multiple_types(self):
        entities = extract_entities(
            "Does CNN outperform SVM on MNIST for image classification?"
        )
        assert len(entities.methods) >= 2
        assert len(entities.datasets) >= 1

    def test_empty_question(self):
        entities = extract_entities("")
        assert entities.is_empty()

    def test_to_dict(self):
        entities = extract_entities("CNN on MNIST")
        d = entities.to_dict()
        assert "methods" in d
        assert "datasets" in d
        assert "tasks" in d
        assert "fields" in d
        assert "metrics" in d

    def test_all_entities_flat_list(self):
        entities = extract_entities("CNN and SVM on MNIST")
        flat = entities.all_entities()
        assert len(flat) >= 3  # CNN, SVM, MNIST


class TestEntityNormalization:
    """Test dictionary-based entity synonym expansion."""

    def test_cnn_variants(self):
        variants = get_method_variants("cnn")
        assert "convolutional neural network" in variants
        assert "dcnn" in variants

    def test_lstm_variants(self):
        variants = get_method_variants("lstm")
        assert "long short-term memory" in variants

    def test_mnist_variants(self):
        variants = get_dataset_variants("mnist")
        assert "benchmark mnist" in variants

    def test_cifar10_variants(self):
        variants = get_dataset_variants("cifar-10")
        assert "cifar10" in variants

    def test_unknown_entity_returns_empty(self):
        assert get_method_variants("unknown_method_xyz") == []
        assert get_dataset_variants("unknown_dataset_xyz") == []

    def test_expand_entities_populates_variants(self):
        entities = extract_entities("CNN on MNIST")
        expanded = expand_entities(entities)
        assert len(expanded.method_variants) >= 1
        assert "cnn" in expanded.method_variants
        assert "convolutional neural network" in expanded.method_variants["cnn"]
        assert "mnist" in expanded.dataset_variants

    def test_all_method_forms(self):
        entities = extract_entities("CNN on MNIST")
        expanded = expand_entities(entities)
        forms = expanded.all_method_forms("cnn")
        assert "cnn" in forms
        assert "convolutional neural network" in forms

    def test_all_dataset_forms(self):
        entities = extract_entities("CNN on MNIST")
        expanded = expand_entities(entities)
        forms = expanded.all_dataset_forms("mnist")
        assert "mnist" in forms
        assert "benchmark mnist" in forms


class TestRanking:
    """Test retrieval result ranking."""

    def test_method_and_dataset_match_scores_highest(self):
        from backend.rag.entity_normalization import ExpandedEntities

        results = [
            {"paper": "http://example.org/R1", "title": "Paper A", "methodLabel": "CNN"},
            {"paper": "http://example.org/R2", "title": "CNN on MNIST", "methodLabel": "CNN", "datasetLabel": "MNIST"},
            {"paper": "http://example.org/R3", "title": "Paper C"},
        ]
        entities = ExpandedEntities(methods=["CNN"], datasets=["MNIST"])
        ranked = rank_results(results, entities)
        # R2 matches method+dataset+title, should be first
        assert ranked[0]["paper"] == "http://example.org/R2"

    def test_method_match_beats_no_match(self):
        from backend.rag.entity_normalization import ExpandedEntities

        results = [
            {"paper": "http://example.org/R1", "title": "Unrelated Paper"},
            {"paper": "http://example.org/R2", "title": "Paper B", "methodLabel": "SVM"},
        ]
        entities = ExpandedEntities(methods=["SVM"], datasets=[])
        ranked = rank_results(results, entities)
        assert ranked[0]["paper"] == "http://example.org/R2"

    def test_empty_results(self):
        from backend.rag.entity_normalization import ExpandedEntities

        entities = ExpandedEntities(methods=["CNN"])
        ranked = rank_results([], entities)
        assert ranked == []

    def test_score_annotation(self):
        from backend.rag.entity_normalization import ExpandedEntities

        results = [
            {"paper": "http://example.org/R1", "title": "CNN Paper", "methodLabel": "CNN"},
        ]
        entities = ExpandedEntities(methods=["CNN"])
        ranked = rank_results(results, entities)
        assert "_score" in ranked[0]
        assert ranked[0]["_score"] >= 2  # at least method match


class TestContextBuilder:
    """Test structured context building."""

    def test_build_paper_context_groups_by_paper(self):
        results = [
            {
                "paper": "http://orkg.org/orkg/resource/R1",
                "title": "Deep Learning for NLP",
                "doi": "10.1234/test",
                "methodLabel": "BERT",
                "datasetLabel": "SQuAD",
            },
            {
                "paper": "http://orkg.org/orkg/resource/R1",
                "title": "Deep Learning for NLP",
                "doi": "10.1234/test",
                "methodLabel": "GPT",
            },
            {
                "paper": "http://orkg.org/orkg/resource/R2",
                "title": "CNN vs SVM",
                "doi": "N/A",
                "methodALabel": "CNN",
                "methodBLabel": "SVM",
            },
        ]
        context = build_paper_context(results)
        assert "Deep Learning for NLP" in context
        assert "CNN vs SVM" in context
        assert "BERT" in context
        assert "GPT" in context
        assert "SQuAD" in context
        # Uses structured format with "Paper:" blocks and "Triples:" sections
        assert "Paper:" in context
        assert "Title:" in context
        assert "Triples:" in context
        assert "  Method:" in context
        # R1 should appear only once (grouped)
        assert context.count("Deep Learning for NLP") == 1

    def test_build_paper_context_empty(self):
        context = build_paper_context([])
        assert "No relevant papers" in context

    def test_structured_paper_format(self):
        """Verify the clean structured output format with Triples section."""
        results = [
            {
                "paper": "http://orkg.org/orkg/resource/R1",
                "title": "Effective Handwritten Digit Recognition",
                "doi": "10.48550/arXiv.2004.00331",
                "methodLabel": "CNN",
                "datasetLabel": "MNIST",
            },
        ]
        context = build_paper_context(results)
        assert "Paper:" in context
        assert "Title: Effective Handwritten Digit Recognition" in context
        assert "URI: http://orkg.org/orkg/resource/R1" in context
        assert "Triples:" in context
        assert "  Method: CNN" in context
        assert "  Dataset: MNIST" in context
        assert "DOI: 10.48550/arXiv.2004.00331" in context

    def test_build_evidence_context(self):
        results = [
            {
                "paper": "http://orkg.org/orkg/resource/R1",
                "title": "Test Paper",
                "doi": "10.1/test",
                "contribLabel": "Contribution 1",
                "predLabel": "Method",
                "valueLabel": "CNN",
            }
        ]
        context = build_evidence_context(results)
        assert "Test Paper" in context
        assert "CNN" in context
        assert "Evidence" in context

    def test_build_evidence_context_empty(self):
        context = build_evidence_context([])
        assert "No evidence" in context

    def test_format_sources_deduplicates(self):
        results = [
            {"paper": "http://example.org/R1", "title": "Paper A", "doi": "10.1/a", "methodLabel": "CNN"},
            {"paper": "http://example.org/R1", "title": "Paper A", "doi": "10.1/a", "methodLabel": "SVM"},
            {"paper": "http://example.org/R2", "title": "Paper B", "doi": "N/A"},
        ]
        sources = format_sources(results)
        assert len(sources) == 2
        assert sources[0]["title"] == "Paper A"
        assert sources[1]["title"] == "Paper B"

    def test_format_sources_includes_methods_and_datasets(self):
        results = [
            {
                "paper": "http://example.org/R1",
                "title": "Paper A",
                "doi": "10.1/a",
                "methodLabel": "CNN",
                "datasetLabel": "MNIST",
            },
        ]
        sources = format_sources(results)
        assert len(sources) == 1
        assert "methods" in sources[0]
        assert "CNN" in sources[0]["methods"]
        assert "datasets" in sources[0]
        assert "MNIST" in sources[0]["datasets"]


class TestPromptTemplates:
    """Test that prompt templates exist for all 6 query types and cite by title."""

    def test_all_query_types_have_templates(self):
        from backend.llm.ollama_client import get_prompt_template

        for qt in [
            "topic_search",
            "method_comparison",
            "dataset_search",
            "claim_verification",
            "method_usage",
            "paper_lookup",
        ]:
            template = get_prompt_template(qt)
            assert "{context}" in template
            assert "{question}" in template

    def test_templates_require_title_citations(self):
        """Verify prompts tell LLM to cite by title, not by number."""
        from backend.llm.ollama_client import get_prompt_template

        for qt in [
            "topic_search",
            "method_comparison",
            "dataset_search",
            "claim_verification",
            "method_usage",
            "paper_lookup",
        ]:
            template = get_prompt_template(qt)
            assert "exact title" in template.lower() or "paper title" in template.lower()
            assert "[1]" in template  # mentioned as what NOT to do


class TestHardFilter:
    """Test that hard filtering removes papers missing required entities."""

    def test_keeps_papers_with_both_method_and_dataset(self):
        from backend.rag.entity_normalization import ExpandedEntities

        results = [
            {"paper": "http://example.org/R1", "title": "A", "methodLabel": "CNN", "datasetLabel": "MNIST"},
            {"paper": "http://example.org/R2", "title": "B", "methodLabel": "CNN"},
            {"paper": "http://example.org/R3", "title": "C", "datasetLabel": "MNIST"},
        ]
        entities = ExpandedEntities(methods=["CNN"], datasets=["MNIST"])
        filtered = hard_filter(results, entities)
        # Only R1 has both CNN and MNIST
        assert len(filtered) == 1
        assert filtered[0]["paper"] == "http://example.org/R1"

    def test_no_filter_when_only_methods(self):
        from backend.rag.entity_normalization import ExpandedEntities

        results = [
            {"paper": "http://example.org/R1", "title": "A", "methodLabel": "CNN"},
            {"paper": "http://example.org/R2", "title": "B"},
        ]
        entities = ExpandedEntities(methods=["CNN"], datasets=[])
        filtered = hard_filter(results, entities)
        assert len(filtered) == 2  # no filtering

    def test_no_filter_when_only_datasets(self):
        from backend.rag.entity_normalization import ExpandedEntities

        results = [
            {"paper": "http://example.org/R1", "title": "A", "datasetLabel": "MNIST"},
            {"paper": "http://example.org/R2", "title": "B"},
        ]
        entities = ExpandedEntities(methods=[], datasets=["MNIST"])
        filtered = hard_filter(results, entities)
        assert len(filtered) == 2

    def test_filter_uses_variants(self):
        from backend.rag.entity_normalization import ExpandedEntities

        results = [
            {"paper": "http://example.org/R1", "title": "A", "methodLabel": "convolutional neural network", "datasetLabel": "MNIST"},
        ]
        entities = ExpandedEntities(
            methods=["CNN"], datasets=["MNIST"],
            method_variants={"cnn": ["convolutional neural network"]},
        )
        filtered = hard_filter(results, entities)
        assert len(filtered) == 1

    def test_filter_empty_results(self):
        from backend.rag.entity_normalization import ExpandedEntities

        entities = ExpandedEntities(methods=["CNN"], datasets=["MNIST"])
        assert hard_filter([], entities) == []

    def test_multi_row_paper_passes_if_labels_across_rows(self):
        """A paper can have method in one row and dataset in another."""
        from backend.rag.entity_normalization import ExpandedEntities

        results = [
            {"paper": "http://example.org/R1", "title": "A", "methodLabel": "CNN"},
            {"paper": "http://example.org/R1", "title": "A", "datasetLabel": "MNIST"},
        ]
        entities = ExpandedEntities(methods=["CNN"], datasets=["MNIST"])
        filtered = hard_filter(results, entities)
        assert len(filtered) == 2  # both rows of R1 pass


class TestTruncation:
    """Test top-N paper truncation."""

    def test_truncates_to_max_papers(self):
        results = [
            {"paper": f"http://example.org/R{i}", "title": f"Paper {i}", "_score": 10 - i}
            for i in range(15)
        ]
        truncated = truncate_to_top_papers(results, max_papers=5)
        uris = {r["paper"] for r in truncated}
        assert len(uris) == 5
        # Should be the first 5 (highest score since they are pre-sorted)
        for i in range(5):
            assert f"http://example.org/R{i}" in uris

    def test_truncation_keeps_all_rows_for_kept_papers(self):
        results = [
            {"paper": "http://example.org/R1", "title": "A", "_score": 5, "methodLabel": "CNN"},
            {"paper": "http://example.org/R1", "title": "A", "_score": 5, "datasetLabel": "MNIST"},
            {"paper": "http://example.org/R2", "title": "B", "_score": 3},
            {"paper": "http://example.org/R3", "title": "C", "_score": 1},
        ]
        truncated = truncate_to_top_papers(results, max_papers=2)
        assert len(truncated) == 3  # 2 rows for R1 + 1 for R2

    def test_truncation_empty(self):
        assert truncate_to_top_papers([], max_papers=8) == []

    def test_fewer_papers_than_limit(self):
        results = [
            {"paper": "http://example.org/R1", "title": "A"},
            {"paper": "http://example.org/R2", "title": "B"},
        ]
        truncated = truncate_to_top_papers(results, max_papers=8)
        assert len(truncated) == 2


class TestContextScores:
    """Test that relevance scores appear in context output."""

    def test_score_in_paper_block(self):
        results = [
            {
                "paper": "http://example.org/R1",
                "title": "Test Paper",
                "doi": "N/A",
                "methodLabel": "CNN",
                "_score": 5,
            },
        ]
        context = build_paper_context(results)
        assert "Relevance Score: 5" in context

    def test_no_score_if_not_present(self):
        results = [
            {
                "paper": "http://example.org/R1",
                "title": "Test Paper",
                "doi": "N/A",
            },
        ]
        context = build_paper_context(results)
        assert "Relevance Score" not in context


class TestAsyncQueryBuilder:
    """Test the plan-then-execute query builder refactor."""

    def test_plan_queries_returns_tuples(self):
        from backend.rag.query_builder import _plan_queries
        from backend.rag.entity_normalization import ExpandedEntities

        entities = ExpandedEntities(methods=["CNN"], datasets=["MNIST"])
        planned = _plan_queries(QueryType.TOPIC_SEARCH, entities, limit=5)
        assert len(planned) > 0
        for query, strategy in planned:
            assert isinstance(query, str)
            assert isinstance(strategy, str)
            assert "SELECT" in query

    def test_plan_fallback_returns_tuples(self):
        from backend.rag.query_builder import _plan_fallback
        from backend.rag.entity_normalization import ExpandedEntities

        entities = ExpandedEntities(methods=["CNN"])
        planned = _plan_fallback(entities, limit=5)
        assert len(planned) > 0
        for query, strategy in planned:
            assert "SELECT" in query

    def test_sync_retrieve_still_works(self):
        """Ensure the synchronous retrieve() still exists and has the right signature."""
        from backend.rag.query_builder import retrieve
        import inspect

        sig = inspect.signature(retrieve)
        assert "question" in sig.parameters
        assert "query_type" in sig.parameters
        assert "entities" in sig.parameters
        assert "sparql_client" in sig.parameters

    def test_async_retrieve_is_coroutine(self):
        from backend.rag.query_builder import retrieve_async
        import inspect

        assert inspect.iscoroutinefunction(retrieve_async)


class TestHybridRetrieval:
    """Test Reciprocal Rank Fusion and hybrid retrieval concepts."""

    def test_rrf_merges_two_lists(self):
        list_a = [
            {"paper": "http://example.org/R1", "title": "Paper 1"},
            {"paper": "http://example.org/R2", "title": "Paper 2"},
        ]
        list_b = [
            {"paper": "http://example.org/R2", "title": "Paper 2"},
            {"paper": "http://example.org/R3", "title": "Paper 3"},
        ]
        merged = reciprocal_rank_fusion(list_a, list_b)
        uris = [r["paper"] for r in merged]
        # R2 appears in both lists → highest RRF score
        assert uris[0] == "http://example.org/R2"
        assert len(merged) == 3

    def test_rrf_score_annotation(self):
        results = [{"paper": "http://example.org/R1", "title": "A"}]
        merged = reciprocal_rank_fusion(results)
        assert "_rrf_score" in merged[0]
        assert merged[0]["_rrf_score"] > 0

    def test_rrf_empty_lists(self):
        merged = reciprocal_rank_fusion([], [])
        assert merged == []

    def test_hybrid_retrieve_without_vector_store(self):
        from backend.rag.hybrid_retrieval import hybrid_retrieve

        kg_results = [
            {"paper": "http://example.org/R1", "title": "A"},
            {"paper": "http://example.org/R2", "title": "B"},
        ]
        results = hybrid_retrieve("test", kg_results, vector_store=None, top_k=5)
        assert len(results) == 2  # falls back to KG only


class TestTimeoutFallback:
    """Test SPARQL timeout → title-search fallback mechanism."""

    def test_title_fallback_for_returns_query(self):
        from backend.rag.query_builder import _title_fallback_for
        from backend.rag.entity_normalization import ExpandedEntities

        entities = ExpandedEntities(methods=["CNN"], datasets=["MNIST"])
        query = _title_fallback_for(entities, limit=5)
        assert query is not None
        assert "SELECT" in query
        assert "?title" in query
        # Title search should check for our keywords
        assert "CNN" in query or "MNIST" in query

    def test_title_fallback_for_empty_entities(self):
        from backend.rag.query_builder import _title_fallback_for
        from backend.rag.entity_normalization import ExpandedEntities

        entities = ExpandedEntities(methods=[], datasets=[])
        query = _title_fallback_for(entities, limit=5)
        assert query is None

    def test_sparql_timeout_error_is_importable(self):
        from backend.kg.sparql_client import SPARQLTimeoutError

        err = SPARQLTimeoutError("test timeout")
        assert isinstance(err, Exception)
        assert "test timeout" in str(err)

    def test_context_includes_uri(self):
        """Paper blocks should include the URI for traceability."""
        results = [
            {
                "paper": "http://orkg.org/orkg/resource/R42",
                "title": "Test Paper",
                "doi": "N/A",
                "methodLabel": "SVM",
            },
        ]
        context = build_paper_context(results)
        assert "URI: http://orkg.org/orkg/resource/R42" in context

    def test_context_triples_section(self):
        """Paper blocks should have a Triples section with method/dataset."""
        results = [
            {
                "paper": "http://orkg.org/orkg/resource/R1",
                "title": "Test Paper",
                "doi": "N/A",
                "methodLabel": "CNN",
                "datasetLabel": "MNIST",
            },
        ]
        context = build_paper_context(results)
        assert "Triples:" in context
        assert "  Method: CNN" in context
        assert "  Dataset: MNIST" in context

    def test_entity_label_goes_to_both_methods_and_datasets(self):
        """entityLabel from broad search should appear in both method and dataset sets."""
        results = [
            {
                "paper": "http://example.org/R1",
                "title": "Paper A",
                "doi": "N/A",
                "entityLabel": "BERT",
            },
        ]
        context = build_paper_context(results)
        # entityLabel should show up in Triples as both Method and Dataset
        assert "  Method: BERT" in context
        assert "  Dataset: BERT" in context


class TestPluralExtraction:
    """Test that dictionary matching handles plural forms."""

    def test_convolutional_neural_networks_plural(self):
        entities = extract_entities(
            "What are some papers that use Convolutional Neural Networks?"
        )
        methods_lower = {m.lower() for m in entities.methods}
        assert "convolutional neural network" in methods_lower

    def test_support_vector_machines_plural(self):
        entities = extract_entities("Papers using Support Vector Machines")
        methods_lower = {m.lower() for m in entities.methods}
        assert "support vector machine" in methods_lower

    def test_decision_trees_plural(self):
        entities = extract_entities("How are Decision Trees used in XGBoost?")
        methods_lower = {m.lower() for m in entities.methods}
        assert "decision tree" in methods_lower

    def test_neural_networks_plural(self):
        entities = extract_entities("Compare neural networks and random forests")
        methods_lower = {m.lower() for m in entities.methods}
        assert "neural network" in methods_lower
        assert "random forest" in methods_lower

    def test_singular_still_works(self):
        entities = extract_entities("Papers about convolutional neural network")
        methods_lower = {m.lower() for m in entities.methods}
        assert "convolutional neural network" in methods_lower


class TestKeywordExtraction:
    """Test keyword extraction fallback for when NER returns empty."""

    def test_extracts_phrase(self):
        keywords = extract_keywords(
            "What are some papers that use Convolutional Neural Networks?"
        )
        assert len(keywords) >= 1
        # Should group consecutive significant words into a phrase
        assert any("convolutional" in kw for kw in keywords)
        assert any("neural" in kw for kw in keywords)

    def test_strips_stop_words(self):
        keywords = extract_keywords("What are the best papers about this?")
        assert "what" not in keywords
        assert "the" not in keywords
        assert "papers" not in keywords

    def test_empty_question(self):
        keywords = extract_keywords("")
        assert keywords == []

    def test_groups_consecutive_words(self):
        keywords = extract_keywords("deep reinforcement learning for robot navigation")
        # "deep reinforcement learning" and "robot navigation" should be separate phrases
        assert len(keywords) >= 1
        assert any("deep" in kw and "reinforcement" in kw for kw in keywords)


class TestSoftFilter:
    """Test soft score-based filtering."""

    def test_drops_zero_score_rows(self):
        results = [
            {"paper": "http://example.org/R1", "title": "A", "_score": 4},
            {"paper": "http://example.org/R2", "title": "B", "_score": 2},
            {"paper": "http://example.org/R3", "title": "C", "_score": 0},
        ]
        filtered = soft_filter(results, min_score=1)
        assert len(filtered) == 2
        assert all(r["_score"] >= 1 for r in filtered)

    def test_keeps_partial_matches(self):
        """Papers matching only method (+2) or only dataset (+2) should survive."""
        results = [
            {"paper": "http://example.org/R1", "title": "A", "_score": 2},  # method only
            {"paper": "http://example.org/R2", "title": "B", "_score": 2},  # dataset only
            {"paper": "http://example.org/R3", "title": "C", "_score": 0},  # no match
        ]
        filtered = soft_filter(results, min_score=1)
        assert len(filtered) == 2  # both partial matches kept

    def test_empty_results(self):
        assert soft_filter([], min_score=1) == []

    def test_custom_min_score(self):
        results = [
            {"paper": "http://example.org/R1", "_score": 4},
            {"paper": "http://example.org/R2", "_score": 2},
            {"paper": "http://example.org/R3", "_score": 1},
        ]
        filtered = soft_filter(results, min_score=3)
        assert len(filtered) == 1
        assert filtered[0]["paper"] == "http://example.org/R1"

    def test_no_score_key_treated_as_zero(self):
        results = [{"paper": "http://example.org/R1", "title": "A"}]
        filtered = soft_filter(results, min_score=1)
        assert len(filtered) == 0


class TestShortVariantFilter:
    """Test that short variant forms are excluded from SPARQL query plans."""

    def test_short_variants_excluded(self):
        from backend.rag.query_builder import _plan_queries, _usable_forms
        from backend.rag.entity_normalization import ExpandedEntities

        # "dl" is a known variant of "deep learning" — too short for CONTAINS
        assert _usable_forms(["dl", "deep learning"]) == ["deep learning"]
        assert _usable_forms(["ml", "machine learning"]) == ["machine learning"]
        assert _usable_forms(["CNN", "convolutional neural network"]) == [
            "CNN", "convolutional neural network"
        ]

    def test_plan_queries_skips_short_forms(self):
        from backend.rag.query_builder import _plan_queries
        from backend.rag.entity_normalization import ExpandedEntities

        entities = ExpandedEntities(
            methods=["deep learning"],
            method_variants={"deep learning": ["dl"]},
        )
        planned = _plan_queries(QueryType.TOPIC_SEARCH, entities, limit=5)
        # "dl" should not appear in any planned query strategy name
        strategies = [s for _, s in planned]
        assert not any("method(dl)" in s for s in strategies)
