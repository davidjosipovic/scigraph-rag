"""
Tests for the Knowledge Graph SPARQL query generation layer.

Verifies that all query templates produce valid SPARQL with correct
predicates and graph traversal patterns (NOT title-matching).
"""

from backend.kg.queries import (
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


class TestSemanticQueries:
    """Test that semantic queries traverse graph relationships, not titles."""

    def test_papers_by_method_traverses_contributions(self):
        query = papers_by_method("CNN", limit=5)
        assert "SELECT" in query
        assert "orkgp:P31" in query  # has contribution
        assert "?methodLabel" in query
        assert '"CNN"' in query
        assert "LIMIT 5" in query

    def test_papers_by_dataset_traverses_contributions(self):
        query = papers_by_dataset("MNIST", limit=5)
        assert "orkgp:P31" in query
        assert "?datasetLabel" in query
        assert '"MNIST"' in query

    def test_papers_by_research_problem(self):
        query = papers_by_research_problem("classification", limit=5)
        assert "orkgp:P31" in query
        assert "orkgp:P32" in query  # research problem
        assert "?problemLabel" in query
        assert '"classification"' in query

    def test_papers_by_research_field(self):
        query = papers_by_research_field("NLP", limit=5)
        assert "orkgp:P30" in query  # research field
        assert "?fieldLabel" in query
        assert '"NLP"' in query

    def test_papers_comparing_methods_uses_dual_traversal(self):
        query = papers_comparing_methods("CNN", "SVM", limit=5)
        assert "orkgp:P31" in query
        assert '"CNN"' in query
        assert '"SVM"' in query
        assert "?methodALabel" in query
        assert "?methodBLabel" in query


class TestCombinedQuery:
    """Test the combined method+dataset query."""

    def test_combined_requires_both_method_and_dataset(self):
        query = papers_by_method_and_dataset("CNN", "MNIST", limit=5)
        assert "orkgp:P31" in query
        assert '"CNN"' in query
        assert '"MNIST"' in query
        assert "?methodLabel" in query
        assert "?datasetLabel" in query
        assert "LIMIT 5" in query

    def test_combined_uses_same_contribution_pattern(self):
        query = papers_by_method_and_dataset("SVM", "CIFAR-10", limit=5)
        # Uses UNION of same-contribution + cross-contribution patterns
        assert "UNION" in query
        # Same-contribution pattern uses shared ?contrib
        assert "?contrib ?predM" in query
        assert "?contrib ?predD" in query
        # Cross-contribution pattern uses separate ?contribM / ?contribD
        assert "?contribM" in query
        assert "?contribD" in query
        assert "FILTER(?contribM != ?contribD)" in query

    def test_combined_returns_paper_metadata(self):
        query = papers_by_method_and_dataset("BERT", "SQuAD")
        assert "?paper" in query
        assert "?title" in query
        assert "?doi" in query


class TestEnrichmentQueries:
    """Test paper enrichment queries."""

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
        assert "orkgp:P26" in query  # DOI
        assert "orkgp:P30" in query  # field


class TestClaimVerification:
    """Test claim evidence retrieval."""

    def test_claim_evidence_uses_contribution_entities(self):
        query = claim_evidence(["CNN", "MNIST", "accuracy"], limit=15)
        assert "orkgp:P31" in query
        assert "?valueLabel" in query
        assert '"CNN"' in query
        assert '"MNIST"' in query
        assert '"accuracy"' in query
        assert "LIMIT 15" in query


class TestFallbackQueries:
    """Test fallback queries (used only when semantic queries fail)."""

    def test_broad_entity_search(self):
        query = broad_entity_search("transformer", limit=5)
        assert "orkgp:P31" in query  # still via contributions
        assert "?entityLabel" in query
        assert '"transformer"' in query

    def test_paper_lookup_by_title(self):
        """paper_lookup_by_title is the ONLY title-matching query."""
        query = paper_lookup_by_title("attention is all you need", limit=5)
        assert "?title" in query
        assert '"attention is all you need"' in query

    def test_title_keyword_search_last_resort(self):
        query = title_keyword_search(["CNN", "SVM"], limit=5)
        assert "?title" in query
        assert '"CNN"' in query
        assert '"SVM"' in query

    def test_limit_parameter(self):
        assert "LIMIT 3" in papers_by_method("test", limit=3)
        assert "LIMIT 20" in papers_by_method("test", limit=20)
