"""
Query classifier: determines the type of user question using an LLM.

Supported query types (6):
  - topic_search:       General topic/field-based paper search
  - method_comparison:  Compare two or more methods
  - dataset_search:     Find papers using a specific dataset
  - claim_verification: Verify a scientific claim
  - method_usage:       Find papers that use a specific method
  - paper_lookup:       Look up a specific paper by name/identifier
"""

import re
from enum import Enum

from loguru import logger

from backend.llm.base import BaseLLMClient


class QueryType(str, Enum):
    """Supported query types in the GraphRAG pipeline."""

    TOPIC_SEARCH = "topic_search"
    METHOD_COMPARISON = "method_comparison"
    DATASET_SEARCH = "dataset_search"
    CLAIM_VERIFICATION = "claim_verification"
    METHOD_USAGE = "method_usage"
    PAPER_LOOKUP = "paper_lookup"


_CLASSIFIER_SYSTEM_PROMPT = """You are a query classifier for a scientific paper search system.
Your only job is to classify the user's question into exactly one of these categories:

- topic_search: General topic or field-based search for papers (e.g. "papers about transformers")
- method_comparison: Comparing two or more methods or models (e.g. "compare BERT vs GPT")
- dataset_search: Finding papers that use a specific dataset or benchmark (e.g. "papers trained on ImageNet")
- claim_verification: Verifying or checking a scientific claim or result (e.g. "does model X outperform Y?")
- method_usage: Finding papers that use a specific method or technique (e.g. "papers using attention")
- paper_lookup: Looking up a specific paper by title or identifier (e.g. "find paper titled Attention is All You Need")

Respond with ONLY the category name, nothing else. No explanation, no punctuation."""


_VALID_TYPES = {qt.value for qt in QueryType}

# Question shapes that determine the type unambiguously. Small LLMs confuse
# exactly these — "Is X better than Y?" (a claim) as a comparison, "Look up
# papers about X" as a topic search — so they are decided before the LLM.
_LOOKUP_RE = re.compile(
    r"^(find|look up|show me|get)\s+the\s+paper\b|^look up\b|\bthe paper (that|which) introduced\b"
    r"|\bpaper (titled|entitled|called)\b",
    re.I,
)
_YES_NO_RE = re.compile(r"^(does|do|is|are|can|will|has|have|did)\b", re.I)
_CLAIM_CUE_RE = re.compile(
    r"\b(better|worse|outperforms?|outperformed|more|less|faster|slower|improves?|improved"
    r"|superior|inferior|beats?|than)\b",
    re.I,
)
_PAPERS_RE = re.compile(r"\bpapers?\b", re.I)
_COMPARE_RE = re.compile(r"\b(compare[sd]?|comparing|comparison|versus|vs\.?)\b", re.I)


def rule_based_type(question: str) -> QueryType | None:
    """
    Query type from the question's shape, or None when it is not clear-cut.

    - paper_lookup:       "Find the paper titled ...", "Look up ...",
                          "... the paper that introduced X"
    - claim_verification: a yes/no question with a comparative or effect
                          cue ("Does X outperform Y?", "Is X faster than Y?")
                          that does not ask for papers
    - method_comparison:  "compare", "comparing", "versus"
    """
    q = question.strip()
    if _LOOKUP_RE.search(q):
        return QueryType.PAPER_LOOKUP
    if _YES_NO_RE.search(q) and _CLAIM_CUE_RE.search(q) and not _PAPERS_RE.search(q):
        return QueryType.CLAIM_VERIFICATION
    if _COMPARE_RE.search(q):
        return QueryType.METHOD_COMPARISON
    return None


def classify_query(question: str, client: BaseLLMClient) -> QueryType:
    """
    Classify a user question into one of six supported query types.

    Clear-cut question shapes are decided by rule_based_type(); the LLM
    classifies the rest. Falls back to topic_search if the model returns an
    unexpected value or if the client is unavailable.

    Args:
        question: The user's natural language question.
        client:   LLM client to use for classification.

    Returns:
        The detected QueryType.
    """
    ruled = rule_based_type(question)
    if ruled is not None:
        logger.info(f"Query classified as '{ruled.value}' (rule)")
        return ruled

    try:
        raw = client.generate(
            prompt=question,
            system=_CLASSIFIER_SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=16,
        )
        label = raw.strip().lower()

        if label in _VALID_TYPES:
            logger.info(f"Query classified as '{label}'")
            return QueryType(label)

        logger.warning(
            f"Unexpected classifier output: '{raw}' — falling back to topic_search"
        )
        return QueryType.TOPIC_SEARCH

    except Exception as e:
        logger.error(f"Query classification failed: {e} — falling back to topic_search")
        return QueryType.TOPIC_SEARCH
