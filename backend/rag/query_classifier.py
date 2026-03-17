"""
Query classifier: determines the type of user question using Llama 3.

Supported query types (6):
  - topic_search:       General topic/field-based paper search
  - method_comparison:  Compare two or more methods
  - dataset_search:     Find papers using a specific dataset
  - claim_verification: Verify a scientific claim
  - method_usage:       Find papers that use a specific method
  - paper_lookup:       Look up a specific paper by name/identifier
"""

from enum import Enum

from loguru import logger

from backend.llm.ollama_client import ollama_client


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


def classify_query(question: str) -> QueryType:
    """
    Classify a user question into one of six supported query types using Llama 3.

    Falls back to topic_search if the model returns an unexpected value
    or if Ollama is unavailable.

    Args:
        question: The user's natural language question.

    Returns:
        The detected QueryType.
    """
    try:
        raw = ollama_client.generate(
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