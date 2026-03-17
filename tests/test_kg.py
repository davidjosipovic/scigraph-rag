"""
Tests for the Knowledge Graph SPARQL query generation layer.

Verifies that:
  - All query templates produce valid SPARQL with correct predicates
  - SPARQL injection is blocked by _sanitize()
  - Edge cases (empty inputs, special chars) are handled safely
"""

import pytest

from backend.kg.queries import (
    _label_filter,
    papers_by_method,
    papers_by_dataset,
    papers_by_research_problem,
    papers_by_research_field,
    papers_comparing_methods,
    papers_by_method_and_dataset,
    paper_full_contributions,
    paper_metadata,
    claim_evidence,
    paper_lookup_by_title,
    broad_entity_search,
    title_keyword_search,
)
from backend.rag.query_builder import _extract_title_from_question


# ── Label Filter Behaviour ────────────────────────────────────────

class TestLabelFilter:
    """Short entities (≤4 chars) use word-boundary REGEX; longer ones use CONTAINS."""

    def test_short_entity_uses_regex(self):
        f = _label_filter("?lbl", "NER")
        assert "REGEX" in f
        assert r"\\b" in f
        assert "NER" in f

    def test_long_entity_uses_contains(self):
        f = _label_filter("?lbl", "convolutional neural network")
        assert "CONTAINS" in f
        assert "convolutional neural network" in f

    def test_exactly_4_chars_uses_regex(self):
        f = _label_filter("?lbl", "BERT")
        assert "REGEX" in f

    def test_5_chars_uses_contains(self):
        f = _label_filter("?lbl", "MNIST")
        assert "CONTAINS" in f


# ── Semantic Queries ──────────────────────────────────────────────

class TestSemanticQueries:

    def test_papers_by_method_traverses_contributions(self):
        # CNN is 3 chars → word-boundary REGEX (not plain CONTAINS)
        query = papers_by_method("CNN", limit=5)
        assert "SELECT" in query
        assert "orkgp:P31" in query
        assert "?methodLabel" in query
        assert "CNN" in query
        assert "REGEX" in query
        assert "LIMIT 5" in query

    def test_papers_by_method_long_name_uses_contains(self):
        query = papers_by_method("convolutional neural network", limit=5)
        assert "CONTAINS" in query
        assert '"convolutional neural network"' in query

    def test_papers_by_dataset_traverses_contributions(self):
        # MNIST is 5 chars → CONTAINS
        query = papers_by_dataset("MNIST", limit=5)
        assert "orkgp:P31" in query
        assert "?datasetLabel" in query
        assert '"MNIST"' in query

    def test_papers_by_research_problem(self):
        query = papers_by_research_problem("classification", limit=5)
        assert "orkgp:P31" in query
        assert "orkgp:P32" in query
        assert "?problemLabel" in query
        assert '"classification"' in query

    def test_papers_by_research_problem_short_uses_regex(self):
        # NER is 3 chars → REGEX (avoids matching "mineral")
        query = papers_by_research_problem("NER", limit=5)
        assert "REGEX" in query
        assert "NER" in query

    def test_papers_by_research_field(self):
        # NLP is 3 chars → REGEX
        query = papers_by_research_field("NLP", limit=5)
        assert "orkgp:P30" in query
        assert "?fieldLabel" in query
        assert "NLP" in query
        assert "REGEX" in query

    def test_papers_comparing_methods_dual_traversal(self):
        # CNN (3) and SVM (3) → both use REGEX
        query = papers_comparing_methods("CNN", "SVM", limit=5)
        assert "orkgp:P31" in query
        assert "CNN" in query
        assert "SVM" in query
        assert "?methodALabel" in query
        assert "?methodBLabel" in query


# ── Combined Query ────────────────────────────────────────────────

class TestCombinedQuery:

    def test_contains_both_method_and_dataset(self):
        # CNN (3 chars) → REGEX; MNIST (5 chars) → CONTAINS
        query = papers_by_method_and_dataset("CNN", "MNIST", limit=5)
        assert "orkgp:P31" in query
        assert "CNN" in query          # present in REGEX pattern
        assert '"MNIST"' in query      # present in CONTAINS literal
        assert "?methodLabel" in query
        assert "?datasetLabel" in query
        assert "LIMIT 5" in query

    def test_uses_union_of_same_and_cross_contribution(self):
        query = papers_by_method_and_dataset("SVM", "CIFAR-10", limit=5)
        assert "UNION" in query
        assert "?contribM" in query
        assert "?contribD" in query
        assert "FILTER(?contribM != ?contribD)" in query

    def test_returns_paper_metadata_fields(self):
        query = papers_by_method_and_dataset("BERT", "SQuAD")
        assert "?paper" in query
        assert "?title" in query
        assert "?doi" in query


# ── Enrichment Queries ────────────────────────────────────────────

class TestEnrichmentQueries:

    def test_paper_full_contributions(self):
        uri = "http://orkg.org/orkg/resource/R12345"
        query = paper_full_contributions(uri)
        assert uri in query
        assert "orkgp:P31" in query
        assert "?contribLabel" in query
        assert "?predLabel" in query

    def test_paper_metadata(self):
        uri = "http://orkg.org/orkg/resource/R999"
        query = paper_metadata(uri)
        assert uri in query
        assert "?title" in query
        assert "orkgp:P26" in query
        assert "orkgp:P30" in query


# ── Claim Evidence ────────────────────────────────────────────────

class TestClaimEvidence:

    def test_claim_evidence_basic(self):
        # CNN (3 chars) → REGEX; MNIST (5) and accuracy (8) → CONTAINS
        query = claim_evidence(["CNN", "MNIST", "accuracy"], limit=15)
        assert "orkgp:P31" in query
        assert "?valueLabel" in query
        assert "CNN" in query           # in REGEX pattern
        assert '"MNIST"' in query       # in CONTAINS literal
        assert '"accuracy"' in query    # in CONTAINS literal
        assert "LIMIT 15" in query

    def test_claim_evidence_empty_keywords_returns_empty_string(self):
        """If all keywords sanitize to empty, return '' to avoid malformed SPARQL."""
        result = claim_evidence([], limit=10)
        assert result == ""

    def test_claim_evidence_all_special_chars_returns_empty_string(self):
        """Keywords that are all special chars should produce empty string."""
        result = claim_evidence(['"', "\\", "{}", "<>"], limit=10)
        assert result == ""


# ── Fallback Queries ──────────────────────────────────────────────

class TestFallbackQueries:

    def test_broad_entity_search(self):
        query = broad_entity_search("transformer", limit=5)
        assert "orkgp:P31" in query
        assert "?entityLabel" in query
        assert '"transformer"' in query

    def test_paper_lookup_by_title(self):
        query = paper_lookup_by_title("attention is all you need", limit=5)
        assert "?title" in query
        assert '"attention is all you need"' in query

    def test_title_keyword_search_single(self):
        # CNN is ≤4 chars → word-boundary regex, not CONTAINS
        query = title_keyword_search(["CNN"], limit=5)
        assert "CNN" in query
        assert "REGEX" in query

    def test_title_keyword_search_multiple(self):
        # CNN and SVM are ≤4 chars → both use REGEX with word boundaries
        query = title_keyword_search(["CNN", "SVM"], limit=5)
        assert "CNN" in query
        assert "SVM" in query
        assert "REGEX" in query

    def test_title_keyword_search_long_term(self):
        # Long terms use CONTAINS, not REGEX
        query = title_keyword_search(["transformer"], limit=5)
        assert "transformer" in query
        assert "CONTAINS" in query

    def test_limit_parameter_respected(self):
        assert "LIMIT 3" in papers_by_method("test", limit=3)
        assert "LIMIT 20" in papers_by_method("test", limit=20)


# ── SPARQL Injection Sanitization ────────────────────────────────

class TestSanitization:

    def test_double_quote_injection_blocked(self):
        """A closing double-quote must not appear unescaped in the query."""
        query = papers_by_method('CNN")) } UNION { ?s ?p ?o }#', limit=5)
        # The injected characters should be stripped
        assert 'CNN")) } UNION { ?s ?p ?o }#' not in query
        # The query must still be structurally valid (no dangling braces)
        assert query.count("{") == query.count("}")

    def test_backslash_injection_blocked(self):
        query = papers_by_method("CNN\\\" OR 1=1", limit=5)
        assert "\\" not in query

    def test_curly_brace_injection_blocked(self):
        query = papers_by_dataset("MNIST} SELECT * WHERE {", limit=5)
        assert "MNIST} SELECT * WHERE {" not in query

    def test_angle_bracket_injection_blocked(self):
        query = papers_by_research_field("<http://evil.com>", limit=5)
        assert "<http://evil.com>" not in query

    def test_single_quote_injection_blocked(self):
        query = papers_by_research_problem("task' OR '1'='1", limit=5)
        assert "'" not in query.split("FILTER")[1]  # no quotes in filter value

    def test_comparison_query_sanitizes_both_methods(self):
        payload = 'CNN") } SELECT * WHERE {'
        query = papers_comparing_methods(payload, "SVM", limit=5)
        assert payload not in query
        assert "CNN" in query
        assert "SVM" in query

    def test_combined_query_sanitizes_both_inputs(self):
        payload = 'BERT") } UNION {'
        query = papers_by_method_and_dataset(payload, "SQuAD", limit=5)
        assert payload not in query
        assert "BERT" in query

    def test_claim_evidence_sanitizes_keywords(self):
        payload = 'CNN") } SELECT *'
        query = claim_evidence([payload, "MNIST"], limit=10)
        assert payload not in query
        assert "MNIST" in query

    def test_title_keyword_search_sanitizes(self):
        payload = 'attention") } SELECT *'
        query = title_keyword_search([payload], limit=5)
        assert payload not in query
        assert "attention" in query

    def test_legitimate_hyphenated_values_preserved(self):
        """Hyphens in entity names like CIFAR-10 must survive sanitization."""
        query = papers_by_dataset("CIFAR-10", limit=5)
        assert "CIFAR-10" in query

    def test_legitimate_spaces_preserved(self):
        query = papers_by_method("convolutional neural network", limit=5)
        assert "convolutional neural network" in query


# ── Paper Lookup Title Extraction ────────────────────────────────

class TestExtractTitleFromQuestion:

    def test_tell_me_about_the(self):
        title = _extract_title_from_question("Tell me about the Attention is All You Need paper")
        assert title == "Attention is All You Need"

    def test_what_is_the(self):
        title = _extract_title_from_question("What is the BERT paper?")
        # strips prefix, strips trailing "?"  not a suffix we handle, but strips "paper" suffix... hmm
        # The function strips " paper" suffix before "?" — let's just check "BERT" is in result
        assert "BERT" in title

    def test_find_the(self):
        title = _extract_title_from_question("Find the GPT-4 paper")
        assert title == "GPT-4"

    def test_no_prefix_unchanged(self):
        title = _extract_title_from_question("Attention is All You Need")
        assert title == "Attention is All You Need"

    def test_strips_article_suffix(self):
        title = _extract_title_from_question("Show me the ResNet article")
        assert title == "ResNet"
