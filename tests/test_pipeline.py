"""
Tests for the RAG pipeline components:
  - Query classifier (6 types, LLM-based with mocks)
  - Entity extractor (LLM-based with mocks)
  - Entity normalization
  - Ranking, hard filtering, soft filtering, truncation
  - Context builder
  - Prompt templates
  - Async query builder
  - Hybrid retrieval (RRF)
"""

import asyncio
import pytest
from unittest.mock import patch

from backend.rag.query_classifier import classify_query, QueryType
from backend.rag.entity_extractor import extract_entities, extract_keywords, ExtractedEntities
from backend.rag.entity_normalization import (
    expand_entities,
    get_method_variants,
    get_dataset_variants,
    TASK_SYNONYMS,
    FIELD_SYNONYMS,
)
from backend.rag.ranking import rank_results, hard_filter, soft_filter, truncate_to_top_papers
from backend.rag.context_builder import (
    build_paper_context,
    build_evidence_context,
    build_context_and_sources,
    format_sources,
)

# ── Fixtures ──────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def clear_lru_caches():
    """Clear LRU caches before each test to avoid cross-test interference."""
    classify_query.cache_clear()
    extract_entities.cache_clear()
    yield
    classify_query.cache_clear()
    extract_entities.cache_clear()


# ── Query Classifier ──────────────────────────────────────────────

class TestQueryClassifier:
    """Test 6-type question classification (LLM mocked)."""

    def _classify_with_mock(self, question: str, llm_response: str) -> QueryType:
        with patch("backend.rag.query_classifier.ollama_client.generate", return_value=llm_response):
            return classify_query(question)

    def test_topic_search(self):
        assert self._classify_with_mock("Find papers about transformers", "topic_search") == QueryType.TOPIC_SEARCH

    def test_method_comparison(self):
        assert self._classify_with_mock("Compare CNN and SVM", "method_comparison") == QueryType.METHOD_COMPARISON

    def test_dataset_search(self):
        assert self._classify_with_mock("Papers using MNIST dataset", "dataset_search") == QueryType.DATASET_SEARCH

    def test_claim_verification(self):
        assert self._classify_with_mock("Does CNN outperform SVM?", "claim_verification") == QueryType.CLAIM_VERIFICATION

    def test_method_usage(self):
        assert self._classify_with_mock("Where is BERT used?", "method_usage") == QueryType.METHOD_USAGE

    def test_paper_lookup(self):
        assert self._classify_with_mock("Look up attention is all you need", "paper_lookup") == QueryType.PAPER_LOOKUP

    def test_invalid_llm_response_falls_back_to_topic_search(self):
        assert self._classify_with_mock("Some question", "I don't know") == QueryType.TOPIC_SEARCH

    def test_llm_response_with_whitespace(self):
        assert self._classify_with_mock("Find papers", "  topic_search  ") == QueryType.TOPIC_SEARCH

    def test_ollama_unavailable_falls_back(self):
        with patch("backend.rag.query_classifier.ollama_client.generate", side_effect=Exception("Ollama down")):
            result = classify_query("Find papers about CNN")
        assert result == QueryType.TOPIC_SEARCH

    def test_result_is_cached(self):
        call_count = 0

        def fake_generate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return "topic_search"

        with patch("backend.rag.query_classifier.ollama_client.generate", side_effect=fake_generate):
            classify_query("cached question abc")
            classify_query("cached question abc")

        assert call_count == 1  # second call served from cache


# ── Entity Extractor ──────────────────────────────────────────────

class TestEntityExtractor:
    """Test typed entity extraction (LLM mocked)."""

    def _extract_with_mock(self, question: str, llm_json: str) -> ExtractedEntities:
        with patch("backend.rag.entity_extractor.ollama_client.generate", return_value=llm_json):
            return extract_entities(question)

    def test_extracts_methods(self):
        json_resp = '{"methods": ["CNN", "SVM"], "datasets": [], "tasks": [], "fields": [], "metrics": []}'
        entities = self._extract_with_mock("Compare CNN and SVM", json_resp)
        assert "CNN" in entities.methods
        assert "SVM" in entities.methods

    def test_extracts_datasets(self):
        json_resp = '{"methods": [], "datasets": ["MNIST", "CIFAR-10"], "tasks": [], "fields": [], "metrics": []}'
        entities = self._extract_with_mock("Models on MNIST and CIFAR-10", json_resp)
        assert "MNIST" in entities.datasets
        assert "CIFAR-10" in entities.datasets

    def test_extracts_tasks(self):
        json_resp = '{"methods": ["BERT"], "datasets": [], "tasks": ["sentiment analysis"], "fields": [], "metrics": []}'
        entities = self._extract_with_mock("BERT for sentiment analysis", json_resp)
        assert "sentiment analysis" in entities.tasks

    def test_extracts_fields(self):
        json_resp = '{"methods": [], "datasets": [], "tasks": [], "fields": ["natural language processing"], "metrics": []}'
        entities = self._extract_with_mock("Research in NLP", json_resp)
        assert "natural language processing" in entities.fields

    def test_extracts_metrics(self):
        json_resp = '{"methods": [], "datasets": ["MNIST"], "tasks": [], "fields": [], "metrics": ["accuracy"]}'
        entities = self._extract_with_mock("Models with highest accuracy on MNIST", json_resp)
        assert "accuracy" in entities.metrics

    def test_unparseable_json_returns_empty(self):
        entities = self._extract_with_mock("Some question", "I cannot parse this")
        assert entities.is_empty()

    def test_ollama_unavailable_returns_empty(self):
        with patch("backend.rag.entity_extractor.ollama_client.generate", side_effect=Exception("Ollama down")):
            entities = extract_entities("Find CNN papers")
        assert entities.is_empty()

    def test_json_embedded_in_prose(self):
        """LLM sometimes wraps JSON in prose — regex should extract it."""
        llm_resp = 'Sure! Here is the result: {"methods": ["BERT"], "datasets": [], "tasks": [], "fields": [], "metrics": []} Hope that helps!'
        entities = self._extract_with_mock("BERT question", llm_resp)
        assert "BERT" in entities.methods

    def test_to_dict_has_all_keys(self):
        json_resp = '{"methods": ["CNN"], "datasets": ["MNIST"], "tasks": [], "fields": [], "metrics": []}'
        entities = self._extract_with_mock("CNN on MNIST", json_resp)
        d = entities.to_dict()
        assert set(d.keys()) == {"methods", "datasets", "tasks", "fields", "metrics"}

    def test_all_entities_flat_list(self):
        json_resp = '{"methods": ["CNN", "SVM"], "datasets": ["MNIST"], "tasks": [], "fields": [], "metrics": []}'
        entities = self._extract_with_mock("CNN and SVM on MNIST", json_resp)
        flat = entities.all_entities()
        assert len(flat) == 3

    def test_result_is_cached(self):
        call_count = 0
        json_resp = '{"methods": ["CNN"], "datasets": [], "tasks": [], "fields": [], "metrics": []}'

        def fake_generate(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return json_resp

        with patch("backend.rag.entity_extractor.ollama_client.generate", side_effect=fake_generate):
            extract_entities("unique cache test question xyz")
            extract_entities("unique cache test question xyz")

        assert call_count == 1


# ── Entity Normalization ──────────────────────────────────────────

class TestEntityNormalization:
    """Test dictionary-based entity synonym expansion."""

    def test_cnn_variants(self):
        variants = get_method_variants("cnn")
        assert "convolutional neural network" in variants
        assert "dcnn" in variants

    def test_lstm_variants(self):
        assert "long short-term memory" in get_method_variants("lstm")

    def test_bert_variants(self):
        variants = get_method_variants("bert")
        assert "bidirectional encoder representations from transformers" in variants

    def test_vit_variants(self):
        assert "vision transformer" in get_method_variants("vit")

    def test_llama_variants(self):
        assert "llama2" in get_method_variants("llama 2")

    def test_mnist_variants(self):
        assert "benchmark mnist" in get_dataset_variants("mnist")

    def test_cifar10_variants(self):
        assert "cifar10" in get_dataset_variants("cifar-10")

    def test_imagenet_variants(self):
        assert "ilsvrc" in get_dataset_variants("imagenet")

    def test_unknown_entity_returns_empty(self):
        assert get_method_variants("unknown_method_xyz") == []
        assert get_dataset_variants("unknown_dataset_xyz") == []

    def test_expand_entities_populates_method_variants(self):
        entities = ExtractedEntities(methods=["cnn"], datasets=["mnist"])
        expanded = expand_entities(entities)
        assert "cnn" in expanded.method_variants
        assert "convolutional neural network" in expanded.method_variants["cnn"]

    def test_expand_entities_populates_dataset_variants(self):
        entities = ExtractedEntities(methods=[], datasets=["mnist"])
        expanded = expand_entities(entities)
        assert "mnist" in expanded.dataset_variants

    def test_all_method_forms_includes_canonical(self):
        entities = ExtractedEntities(methods=["cnn"])
        expanded = expand_entities(entities)
        forms = expanded.all_method_forms("cnn")
        assert "cnn" in forms
        assert "convolutional neural network" in forms

    def test_all_dataset_forms_includes_canonical(self):
        entities = ExtractedEntities(datasets=["mnist"])
        expanded = expand_entities(entities)
        forms = expanded.all_dataset_forms("mnist")
        assert "mnist" in forms
        assert "benchmark mnist" in forms

    # ── Task synonyms ────────────────────────────────────────────
    def test_ner_expands_to_named_entity_recognition(self):
        entities = ExtractedEntities(tasks=["NER"])
        expanded = expand_entities(entities)
        forms = expanded.all_task_forms("NER")
        assert "NER" in forms
        assert "named entity recognition" in forms
        assert "entity recognition" in forms

    def test_mt_expands_to_machine_translation(self):
        entities = ExtractedEntities(tasks=["MT"])
        expanded = expand_entities(entities)
        forms = expanded.all_task_forms("MT")
        assert "machine translation" in forms

    def test_qa_expands_to_question_answering(self):
        entities = ExtractedEntities(tasks=["QA"])
        expanded = expand_entities(entities)
        forms = expanded.all_task_forms("QA")
        assert "question answering" in forms

    def test_unknown_task_returns_only_canonical(self):
        entities = ExtractedEntities(tasks=["some unknown task xyz"])
        expanded = expand_entities(entities)
        assert expanded.all_task_forms("some unknown task xyz") == ["some unknown task xyz"]

    # ── Field synonyms ───────────────────────────────────────────
    def test_nlp_expands_to_natural_language_processing(self):
        entities = ExtractedEntities(fields=["NLP"])
        expanded = expand_entities(entities)
        forms = expanded.all_field_forms("NLP")
        assert "NLP" in forms
        assert "natural language processing" in forms

    def test_robotics_expands_to_variants(self):
        entities = ExtractedEntities(fields=["robotics"])
        expanded = expand_entities(entities)
        forms = expanded.all_field_forms("robotics")
        assert "robot learning" in forms
        assert "autonomous systems" in forms

    def test_task_synonyms_dict_not_empty(self):
        assert len(TASK_SYNONYMS) >= 10

    def test_field_synonyms_dict_not_empty(self):
        assert len(FIELD_SYNONYMS) >= 5


# ── Ranking ───────────────────────────────────────────────────────

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
        assert rank_results([], ExpandedEntities(methods=["CNN"])) == []

    def test_score_annotation(self):
        from backend.rag.entity_normalization import ExpandedEntities

        results = [{"paper": "http://example.org/R1", "title": "CNN Paper", "methodLabel": "CNN"}]
        ranked = rank_results(results, ExpandedEntities(methods=["CNN"]))
        assert "_score" in ranked[0]
        assert ranked[0]["_score"] >= 2


# ── Hard Filter ───────────────────────────────────────────────────

class TestHardFilter:

    def test_keeps_papers_with_both_method_and_dataset(self):
        from backend.rag.entity_normalization import ExpandedEntities

        results = [
            {"paper": "http://example.org/R1", "title": "A", "methodLabel": "CNN", "datasetLabel": "MNIST"},
            {"paper": "http://example.org/R2", "title": "B", "methodLabel": "CNN"},
            {"paper": "http://example.org/R3", "title": "C", "datasetLabel": "MNIST"},
        ]
        entities = ExpandedEntities(methods=["CNN"], datasets=["MNIST"])
        filtered = hard_filter(results, entities)
        assert len(filtered) == 1
        assert filtered[0]["paper"] == "http://example.org/R1"

    def test_no_filter_when_only_methods(self):
        from backend.rag.entity_normalization import ExpandedEntities

        results = [
            {"paper": "http://example.org/R1", "title": "A", "methodLabel": "CNN"},
            {"paper": "http://example.org/R2", "title": "B"},
        ]
        assert len(hard_filter(results, ExpandedEntities(methods=["CNN"], datasets=[]))) == 2

    def test_no_filter_when_only_datasets(self):
        from backend.rag.entity_normalization import ExpandedEntities

        results = [
            {"paper": "http://example.org/R1", "title": "A", "datasetLabel": "MNIST"},
            {"paper": "http://example.org/R2", "title": "B"},
        ]
        assert len(hard_filter(results, ExpandedEntities(methods=[], datasets=["MNIST"]))) == 2

    def test_filter_uses_variants(self):
        from backend.rag.entity_normalization import ExpandedEntities

        results = [
            {"paper": "http://example.org/R1", "title": "A",
             "methodLabel": "convolutional neural network", "datasetLabel": "MNIST"},
        ]
        entities = ExpandedEntities(
            methods=["CNN"], datasets=["MNIST"],
            method_variants={"cnn": ["convolutional neural network"]},
        )
        assert len(hard_filter(results, entities)) == 1

    def test_multi_row_paper_passes_if_labels_across_rows(self):
        from backend.rag.entity_normalization import ExpandedEntities

        results = [
            {"paper": "http://example.org/R1", "title": "A", "methodLabel": "CNN"},
            {"paper": "http://example.org/R1", "title": "A", "datasetLabel": "MNIST"},
        ]
        entities = ExpandedEntities(methods=["CNN"], datasets=["MNIST"])
        assert len(hard_filter(results, entities)) == 2

    def test_empty_results(self):
        from backend.rag.entity_normalization import ExpandedEntities
        assert hard_filter([], ExpandedEntities(methods=["CNN"], datasets=["MNIST"])) == []


# ── Soft Filter ───────────────────────────────────────────────────

class TestSoftFilter:

    def test_drops_zero_score_rows(self):
        results = [
            {"paper": "http://example.org/R1", "_score": 4},
            {"paper": "http://example.org/R2", "_score": 2},
            {"paper": "http://example.org/R3", "_score": 0},
        ]
        filtered = soft_filter(results, min_score=1)
        assert len(filtered) == 2

    def test_keeps_partial_matches(self):
        results = [
            {"paper": "http://example.org/R1", "_score": 2},
            {"paper": "http://example.org/R2", "_score": 0},
        ]
        assert len(soft_filter(results, min_score=1)) == 1

    def test_custom_min_score(self):
        results = [
            {"paper": "http://example.org/R1", "_score": 4},
            {"paper": "http://example.org/R2", "_score": 2},
        ]
        assert len(soft_filter(results, min_score=3)) == 1

    def test_missing_score_key_treated_as_zero(self):
        results = [{"paper": "http://example.org/R1", "title": "A"}]
        assert soft_filter(results, min_score=1) == []

    def test_empty_results(self):
        assert soft_filter([], min_score=1) == []


# ── Truncation ────────────────────────────────────────────────────

class TestTruncation:

    def test_truncates_to_max_papers(self):
        results = [
            {"paper": f"http://example.org/R{i}", "title": f"Paper {i}", "_score": 10 - i}
            for i in range(15)
        ]
        truncated = truncate_to_top_papers(results, max_papers=5)
        assert len({r["paper"] for r in truncated}) == 5

    def test_keeps_all_rows_for_kept_papers(self):
        results = [
            {"paper": "http://example.org/R1", "title": "A", "_score": 5, "methodLabel": "CNN"},
            {"paper": "http://example.org/R1", "title": "A", "_score": 5, "datasetLabel": "MNIST"},
            {"paper": "http://example.org/R2", "title": "B", "_score": 3},
            {"paper": "http://example.org/R3", "title": "C", "_score": 1},
        ]
        truncated = truncate_to_top_papers(results, max_papers=2)
        assert len(truncated) == 3  # 2 rows for R1 + 1 for R2

    def test_empty(self):
        assert truncate_to_top_papers([], max_papers=8) == []

    def test_fewer_papers_than_limit(self):
        results = [
            {"paper": "http://example.org/R1", "title": "A"},
            {"paper": "http://example.org/R2", "title": "B"},
        ]
        assert len(truncate_to_top_papers(results, max_papers=8)) == 2


# ── Context Builder ───────────────────────────────────────────────

class TestContextBuilder:

    def test_groups_by_paper(self):
        results = [
            {"paper": "http://orkg.org/orkg/resource/R1", "title": "Deep Learning for NLP",
             "doi": "10.1234/test", "methodLabel": "BERT", "datasetLabel": "SQuAD"},
            {"paper": "http://orkg.org/orkg/resource/R1", "title": "Deep Learning for NLP",
             "doi": "10.1234/test", "methodLabel": "GPT"},
            {"paper": "http://orkg.org/orkg/resource/R2", "title": "CNN vs SVM",
             "doi": "N/A", "methodALabel": "CNN", "methodBLabel": "SVM"},
        ]
        context = build_paper_context(results)
        assert "Deep Learning for NLP" in context
        assert "CNN vs SVM" in context
        assert "BERT" in context
        assert "GPT" in context
        assert "SQuAD" in context
        assert context.count("Deep Learning for NLP") == 1  # grouped, not duplicated

    def test_structured_format(self):
        results = [
            {"paper": "http://orkg.org/orkg/resource/R1",
             "title": "Effective Handwritten Digit Recognition",
             "doi": "10.48550/arXiv.2004.00331",
             "methodLabel": "CNN", "datasetLabel": "MNIST"},
        ]
        context = build_paper_context(results)
        assert "Paper:" in context
        assert "Title: Effective Handwritten Digit Recognition" in context
        assert "URI: http://orkg.org/orkg/resource/R1" in context
        assert "DOI: 10.48550/arXiv.2004.00331" in context
        assert "Triples:" in context
        assert "  Method: CNN" in context
        assert "  Dataset: MNIST" in context

    def test_relevance_score_not_in_context(self):
        """Internal scores must not leak into LLM context."""
        results = [
            {"paper": "http://example.org/R1", "title": "Test Paper",
             "doi": "N/A", "methodLabel": "CNN", "_score": 5},
        ]
        context = build_paper_context(results)
        assert "Relevance Score" not in context

    def test_empty_returns_no_papers_message(self):
        assert "No relevant papers" in build_paper_context([])

    def test_evidence_context(self):
        results = [
            {"paper": "http://orkg.org/orkg/resource/R1", "title": "Test Paper",
             "doi": "10.1/test", "contribLabel": "Contribution 1",
             "predLabel": "Method", "valueLabel": "CNN"},
        ]
        context = build_evidence_context(results)
        assert "Test Paper" in context
        assert "CNN" in context
        assert "Evidence" in context

    def test_evidence_context_empty(self):
        assert "No evidence" in build_evidence_context([])

    def test_entity_label_goes_only_to_methods(self):
        """entityLabel from broad fallback must NOT be double-added to datasets."""
        results = [
            {"paper": "http://example.org/R1", "title": "Paper A",
             "doi": "N/A", "entityLabel": "BERT"},
        ]
        context = build_paper_context(results)
        assert "  Method: BERT" in context
        assert "  Dataset: BERT" not in context

    def test_format_sources_deduplicates(self):
        results = [
            {"paper": "http://example.org/R1", "title": "Paper A", "doi": "10.1/a", "methodLabel": "CNN"},
            {"paper": "http://example.org/R1", "title": "Paper A", "doi": "10.1/a", "methodLabel": "SVM"},
            {"paper": "http://example.org/R2", "title": "Paper B", "doi": "N/A"},
        ]
        sources = format_sources(results)
        assert len(sources) == 2

    def test_format_sources_includes_methods_and_datasets(self):
        results = [
            {"paper": "http://example.org/R1", "title": "Paper A", "doi": "10.1/a",
             "methodLabel": "CNN", "datasetLabel": "MNIST"},
        ]
        sources = format_sources(results)
        assert "CNN" in sources[0]["methods"]
        assert "MNIST" in sources[0]["datasets"]

    # ── Year validation ──────────────────────────────────────────
    def test_valid_year_preserved(self):
        results = [{"paper": "http://example.org/R1", "title": "A",
                    "doi": "N/A", "year": "2021"}]
        sources = format_sources(results)
        assert sources[0]["year"] == "2021"

    def test_invalid_year_month_value_becomes_none(self):
        """Single/double digit values like '9' or '12' are months, not years."""
        for bad in ["9", "12", "7", "0", "99"]:
            results = [{"paper": "http://example.org/R1", "title": "A",
                        "doi": "N/A", "year": bad}]
            sources = format_sources(results)
            assert sources[0]["year"] is None, f"Expected None for year={bad!r}"

    def test_year_missing_becomes_none(self):
        results = [{"paper": "http://example.org/R1", "title": "A", "doi": "N/A"}]
        sources = format_sources(results)
        assert sources[0]["year"] is None

    def test_year_shown_in_paper_context(self):
        results = [{"paper": "http://example.org/R1", "title": "A",
                    "doi": "N/A", "year": "2019"}]
        context = build_paper_context(results)
        assert "Year: 2019" in context

    # ── build_context_and_sources ────────────────────────────────
    def test_build_context_and_sources_returns_both(self):
        results = [{"paper": "http://example.org/R1", "title": "Paper A",
                    "doi": "10.1/a", "methodLabel": "CNN", "year": "2020"}]
        context, sources = build_context_and_sources(results, "method_usage")
        assert "Paper A" in context
        assert len(sources) == 1
        assert sources[0]["title"] == "Paper A"
        assert sources[0]["year"] == "2020"

    def test_build_context_and_sources_claim_type(self):
        results = [{"paper": "http://example.org/R1", "title": "Claim Paper",
                    "doi": "N/A", "contribLabel": "C1",
                    "predLabel": "Has method", "valueLabel": "BERT"}]
        context, sources = build_context_and_sources(results, "claim_verification")
        assert "Evidence" in context
        assert len(sources) == 1


# ── Prompt Templates ──────────────────────────────────────────────

class TestPromptTemplates:

    def test_all_query_types_have_templates(self):
        from backend.llm.ollama_client import get_prompt_template

        for qt in ["topic_search", "method_comparison", "dataset_search",
                   "claim_verification", "method_usage", "paper_lookup"]:
            t = get_prompt_template(qt)
            assert "{context}" in t
            assert "{question}" in t

    def test_templates_cite_by_title_not_number(self):
        from backend.llm.ollama_client import get_prompt_template

        for qt in ["topic_search", "method_comparison", "dataset_search",
                   "claim_verification", "method_usage", "paper_lookup"]:
            t = get_prompt_template(qt)
            assert "exact title" in t.lower() or "paper title" in t.lower()
            assert "[1]" in t  # mentioned as what NOT to do


# ── Async Query Builder ───────────────────────────────────────────

class TestAsyncQueryBuilder:

    def test_plan_queries_returns_valid_tuples(self):
        from backend.rag.query_builder import _plan_queries
        from backend.rag.entity_normalization import ExpandedEntities

        entities = ExpandedEntities(methods=["CNN"], datasets=["MNIST"])
        planned = _plan_queries(QueryType.TOPIC_SEARCH, entities, limit=5)
        assert len(planned) > 0
        for query, strategy in planned:
            assert isinstance(query, str)
            assert isinstance(strategy, str)
            assert "SELECT" in query

    def test_plan_fallback_returns_valid_tuples(self):
        from backend.rag.query_builder import _plan_fallback
        from backend.rag.entity_normalization import ExpandedEntities

        entities = ExpandedEntities(methods=["CNN"])
        for query, _ in _plan_fallback(entities, limit=5):
            assert "SELECT" in query

    def test_retrieve_async_is_coroutine(self):
        import inspect
        from backend.rag.query_builder import retrieve_async
        assert inspect.iscoroutinefunction(retrieve_async)

    def test_retrieve_sync_signature(self):
        import inspect
        from backend.rag.query_builder import retrieve
        sig = inspect.signature(retrieve)
        assert all(p in sig.parameters for p in ["question", "query_type", "entities", "sparql_client"])

    def test_short_variants_excluded_from_contains(self):
        from backend.rag.query_builder import _usable_forms
        assert _usable_forms(["dl", "deep learning"]) == ["deep learning"]
        assert _usable_forms(["ml", "machine learning"]) == ["machine learning"]
        assert _usable_forms(["CNN", "convolutional neural network"]) == ["CNN", "convolutional neural network"]

    def test_title_fallback_returns_query_when_entities_present(self):
        from backend.rag.query_builder import _title_fallback_for
        from backend.rag.entity_normalization import ExpandedEntities

        entities = ExpandedEntities(methods=["CNN"], datasets=["MNIST"])
        query = _title_fallback_for(entities, limit=5)
        assert query is not None
        assert "SELECT" in query
        assert "?title" in query

    def test_title_fallback_returns_none_when_entities_empty(self):
        from backend.rag.query_builder import _title_fallback_for
        from backend.rag.entity_normalization import ExpandedEntities

        assert _title_fallback_for(ExpandedEntities(), limit=5) is None



# ── Keyword Extraction Fallback ───────────────────────────────────

class TestKeywordExtraction:

    def test_extracts_significant_words(self):
        keywords = extract_keywords("deep reinforcement learning for robot navigation")
        assert any("deep" in kw for kw in keywords)

    def test_strips_stop_words(self):
        keywords = extract_keywords("What are the best papers about this?")
        assert "what" not in keywords
        assert "the" not in keywords

    def test_empty_question(self):
        assert extract_keywords("") == []

    def test_groups_consecutive_words(self):
        keywords = extract_keywords("deep reinforcement learning for robot navigation")
        assert any("deep" in kw and "reinforcement" in kw for kw in keywords)


# ── Timeout / Error Handling ──────────────────────────────────────

class TestErrorHandling:

    def test_sparql_timeout_error_importable(self):
        from backend.kg.sparql_client import SPARQLTimeoutError
        err = SPARQLTimeoutError("test timeout")
        assert isinstance(err, Exception)
        assert "test timeout" in str(err)


# ── Integration: full pipeline.ask() with mocks ───────────────────

_FAKE_ROWS = [
    {
        "paper": "http://orkg.org/orkg/resource/R1",
        "title": "BERT on SQuAD",
        "doi": "10.1/bert",
        "year": "2019",
        "methodLabel": "BERT",
        "datasetLabel": "SQuAD",
    },
    {
        "paper": "http://orkg.org/orkg/resource/R2",
        "title": "ALBERT for Question Answering",
        "doi": "10.1/albert",
        "year": "2020",
        "methodLabel": "ALBERT",
        "datasetLabel": "SQuAD",
    },
]


class TestPipelineIntegration:
    """End-to-end tests for RAGPipeline.ask() with all I/O mocked."""

    def _make_pipeline(self, rows=None, answer="Test answer."):
        """Create a RAGPipeline with mocked SPARQL and LLM."""
        from unittest.mock import AsyncMock, MagicMock
        from backend.rag.pipeline import RAGPipeline

        mock_sparql = MagicMock()
        mock_sparql.execute.return_value = rows if rows is not None else _FAKE_ROWS
        mock_sparql.execute_async = AsyncMock(
            return_value=rows if rows is not None else _FAKE_ROWS
        )

        mock_llm = MagicMock()
        mock_llm.generate.return_value = answer

        return RAGPipeline(sparql_client=mock_sparql, llm_client=mock_llm)

    async def test_ask_returns_required_keys(self):
        pipeline = self._make_pipeline()
        with patch("backend.rag.pipeline.classify_query",
                   return_value=QueryType.METHOD_USAGE), \
             patch("backend.rag.pipeline.extract_entities",
                   return_value=ExtractedEntities(methods=["BERT"], datasets=["SQuAD"])):
            result = await pipeline.ask("What papers apply BERT to SQuAD?")

        assert result["question"] == "What papers apply BERT to SQuAD?"
        assert result["query_type"] == "method_usage"
        assert "entities" in result
        assert "sparql_queries" in result
        assert "strategies_used" in result
        assert "answer" in result
        assert "sources" in result
        assert "kg_results_count" in result

    async def test_ask_sources_contain_expected_papers(self):
        pipeline = self._make_pipeline()
        with patch("backend.rag.pipeline.classify_query",
                   return_value=QueryType.METHOD_USAGE), \
             patch("backend.rag.pipeline.extract_entities",
                   return_value=ExtractedEntities(methods=["BERT"], datasets=["SQuAD"])):
            result = await pipeline.ask("What papers apply BERT to SQuAD?")

        titles = [s["title"] for s in result["sources"]]
        assert "BERT on SQuAD" in titles
        assert "ALBERT for Question Answering" in titles

    async def test_ask_answer_comes_from_llm(self):
        pipeline = self._make_pipeline(answer="Mocked LLM answer here.")
        with patch("backend.rag.pipeline.classify_query",
                   return_value=QueryType.TOPIC_SEARCH), \
             patch("backend.rag.pipeline.extract_entities",
                   return_value=ExtractedEntities(fields=["NLP"])):
            result = await pipeline.ask("Papers about NLP")

        assert result["answer"] == "Mocked LLM answer here."

    async def test_ask_year_validated_in_sources(self):
        rows_with_bad_year = [
            {**_FAKE_ROWS[0], "year": "2019"},   # valid
            {**_FAKE_ROWS[1], "year": "9"},       # invalid — month
        ]
        pipeline = self._make_pipeline(rows=rows_with_bad_year)
        with patch("backend.rag.pipeline.classify_query",
                   return_value=QueryType.METHOD_USAGE), \
             patch("backend.rag.pipeline.extract_entities",
                   return_value=ExtractedEntities(methods=["BERT"], datasets=["SQuAD"])):
            result = await pipeline.ask("BERT on SQuAD")

        by_title = {s["title"]: s for s in result["sources"]}
        assert by_title["BERT on SQuAD"]["year"] == "2019"
        assert by_title["ALBERT for Question Answering"]["year"] is None

    async def test_ask_empty_kg_returns_answer_with_no_sources(self):
        pipeline = self._make_pipeline(rows=[], answer="Nothing found.")
        with patch("backend.rag.pipeline.classify_query",
                   return_value=QueryType.TOPIC_SEARCH), \
             patch("backend.rag.pipeline.extract_entities",
                   return_value=ExtractedEntities(fields=["robotics"])):
            result = await pipeline.ask("Papers on robotics")

        assert result["sources"] == []
        assert result["answer"] == "Nothing found."
        assert result["kg_results_count"] == 0

    async def test_ask_claim_verification_type(self):
        pipeline = self._make_pipeline()
        with patch("backend.rag.pipeline.classify_query",
                   return_value=QueryType.CLAIM_VERIFICATION), \
             patch("backend.rag.pipeline.extract_entities",
                   return_value=ExtractedEntities(methods=["BERT"])):
            result = await pipeline.ask("Is it true that BERT outperforms LSTM?")

        assert result["query_type"] == "claim_verification"
        assert len(result["sources"]) >= 1

    async def test_ask_entities_reflected_in_response(self):
        pipeline = self._make_pipeline()
        extracted = ExtractedEntities(
            methods=["CNN"], datasets=["MNIST"], tasks=["image classification"]
        )
        with patch("backend.rag.pipeline.classify_query",
                   return_value=QueryType.METHOD_USAGE), \
             patch("backend.rag.pipeline.extract_entities", return_value=extracted):
            result = await pipeline.ask("CNN on MNIST for image classification")

        assert "CNN" in result["entities"]["methods"]
        assert "MNIST" in result["entities"]["datasets"]
        assert "image classification" in result["entities"]["tasks"]
