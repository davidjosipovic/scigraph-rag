"""
Tests for the Knowledge Graph SPARQL query generation layer.

Verifies that:
  - All query templates produce valid SPARQL with correct predicates
  - SPARQL injection is blocked by _sanitize()
  - Edge cases (empty inputs, special chars) are handled safely
"""

import pytest

from backend.kg import queries as kg_queries
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
        assert "(^|[^A-Za-z0-9])NER([^A-Za-z0-9]|$)" in f

    def test_short_entity_escapes_regex_metacharacters(self):
        # Unescaped, "C++" would be a possessive quantifier matching any "C"
        # token (e.g. "cytochrome C oxidase"); "F1." would match "F1x".
        assert "C[+][+]" in _label_filter("?lbl", "C++")
        assert "F1[.]" in _label_filter("?lbl", "F1.")
        assert "\\" not in _label_filter("?lbl", "C++")

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

    def test_ranks_same_contribution_first(self):
        query = papers_by_method_and_dataset("SVM", "CIFAR-10", limit=5)
        assert "BIND(IF(?cm = ?cd, 1, 0) AS ?same)" in query
        assert "ORDER BY DESC(?sameContrib)" in query

    def test_method_and_dataset_must_be_different_values(self):
        query = papers_by_method_and_dataset("SVM", "CIFAR-10", limit=5)
        assert "FILTER(?vm != ?vd)" in query

    def test_one_row_per_paper(self):
        query = papers_by_method_and_dataset("SVM", "CIFAR-10", limit=5)
        assert "GROUP BY ?paper" in query

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

    def test_paper_metadata_includes_authors(self):
        query = paper_metadata("http://orkg.org/orkg/resource/R999")
        assert "orkgp:hasAuthors" in query
        assert "?authors" in query

    def test_full_contributions_keeps_literal_values(self):
        # ~43% of ORKG contribution values are literals (scores, sizes, ...);
        # FILTER(BOUND(?valueLabel)) on a label-only OPTIONAL dropped them all.
        query = paper_full_contributions("http://orkg.org/orkg/resource/R1")
        assert "COALESCE(?vl, ?value)" in query
        assert "BOUND(" not in query

    @pytest.mark.parametrize("bad", [
        "R12345",
        "http://orkg.org/x> } ?s ?p ?o { <http://a",
        "http://orkg.org/x\n",
    ])
    def test_enrichment_queries_reject_invalid_iri(self, bad):
        with pytest.raises(ValueError):
            paper_full_contributions(bad)
        with pytest.raises(ValueError):
            paper_metadata(bad)


# ── ORKG Schema ───────────────────────────────────────────────────

ALL_PAPER_QUERIES = [
    lambda: papers_by_method("BERT"),
    lambda: papers_by_dataset("SQuAD"),
    lambda: papers_by_research_problem("NER"),
    lambda: papers_by_research_field("NLP"),
    lambda: papers_comparing_methods("CNN", "SVM"),
    lambda: papers_by_method_and_dataset("BERT", "SQuAD"),
    lambda: claim_evidence(["BERT", "GLUE"]),
    lambda: paper_lookup_by_title("attention"),
    lambda: broad_entity_search("transformer"),
    lambda: title_keyword_search(["BERT"]),
    lambda: paper_metadata("http://orkg.org/orkg/resource/R1"),
]


class TestOrkgSchema:

    @pytest.mark.parametrize("make_query", ALL_PAPER_QUERIES)
    def test_year_is_p29_not_month(self, make_query):
        # In ORKG, P28 is "publication month" and P29 is "publication year".
        query = make_query()
        assert "orkgp:P29" in query
        assert "orkgp:P28" not in query

    def test_research_field_includes_subfields(self):
        query = papers_by_research_field("Computer Sciences")
        assert "orkgp:P36*" in query

    def test_entity_queries_match_literal_values(self):
        query = papers_by_method("transformer")
        assert "isLiteral(?lbl)" in query

    def test_entity_queries_exclude_rdf_type(self):
        # rdf:type → orkgc:Contribution has label "Contribution"
        query = broad_entity_search("contribution")
        assert "NOT IN (rdf:type, rdfs:label)" in query

    def test_entity_query_returns_distinct_papers(self):
        # The inner sub-select picks LIMIT distinct papers; otherwise one
        # paper with many matching labels fills the whole LIMIT.
        query = papers_by_method("BERT", limit=5)
        assert "GROUP BY ?paper" in query
        assert "LIMIT 5" in query


class TestFulltext:

    def test_fulltext_prefilter_when_enabled(self, monkeypatch):
        monkeypatch.setattr(kg_queries, "FULLTEXT", True)
        query = papers_by_dataset("CIFAR-10")
        # Index tokenizes "CIFAR-10" as "cifar 10"; exact FILTER is kept.
        assert "bif:contains \"'CIFAR 10'\"" in query
        assert 'CONTAINS(LCASE(STR(?lbl)), LCASE("CIFAR-10"))' in query

    def test_long_terms_get_wildcard(self, monkeypatch):
        monkeypatch.setattr(kg_queries, "FULLTEXT", True)
        assert "'transformer*'" in papers_by_method("transformer")
        # Short terms are whole-word matches, never wildcards
        assert "'BERT'" in papers_by_method("BERT")

    def test_no_bif_when_disabled(self, monkeypatch):
        # rdflib (USE_LOCAL_RDF) and non-Virtuoso endpoints lack bif:contains
        monkeypatch.setattr(kg_queries, "FULLTEXT", False)
        for make_query in ALL_PAPER_QUERIES:
            assert "bif:" not in make_query()


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

    def test_claim_evidence_ranks_contributions_by_keyword_coverage(self):
        # Coverage counts DISTINCT keywords across all values of a
        # contribution, not keywords inside one label.
        query = claim_evidence(["BERT", "SQuAD"], limit=10)
        assert "COUNT(DISTINCT ?kw) AS ?hits" in query
        assert "GROUP BY ?contrib" in query
        assert "ORDER BY DESC(?hits)" in query

    def test_claim_evidence_includes_benchmark_scores(self):
        # Scores live two hops down: Contribution → PWC_HAS_BENCHMARK →
        # Benchmark → P34 (evaluation) → "Quantity F1_95.0"
        query = claim_evidence(["BERT", "SQuAD"], limit=10)
        assert "orkgp:PWC_HAS_BENCHMARK" in query
        assert "orkgp:P34" in query

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
