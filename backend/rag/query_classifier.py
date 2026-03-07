"""
Query classifier: determines the type of user question.

Supported query types (6):
  - topic_search:       General topic/field-based paper search
  - method_comparison:  Compare two or more methods
  - dataset_search:     Find papers using a specific dataset
  - claim_verification: Verify a scientific claim
  - method_usage:       Find papers that use a specific method
  - paper_lookup:       Look up a specific paper by name/identifier

The classifier uses keyword heuristics for simplicity and speed.
In a production system, this could be replaced with an LLM-based classifier.
"""

import re
from enum import Enum


class QueryType(str, Enum):
    """Supported query types in the GraphRAG pipeline."""

    TOPIC_SEARCH = "topic_search"
    METHOD_COMPARISON = "method_comparison"
    DATASET_SEARCH = "dataset_search"
    CLAIM_VERIFICATION = "claim_verification"
    METHOD_USAGE = "method_usage"
    PAPER_LOOKUP = "paper_lookup"


# --- Classification patterns (checked in priority order) ---

_CLAIM_PATTERNS = [
    r"\bdoes\b.*\bclaim\b",
    r"\bis it true\b",
    r"\bverify\b",
    r"\bconfirm\b",
    r"\boutperform",
    r"\bbetter than\b",
    r"\bcompared to\b.*\bresult",
    r"\baccuracy\b.*\bhigher\b",
    r"\bdoes paper\b",
    r"\bsupported\b.*\bevidence\b",
]

_METHOD_COMPARISON_PATTERNS = [
    r"\bcompare\b.*\band\b",
    r"\bcomparison\b.*\bbetween\b",
    r"\bvs\.?\b",
    r"\bversus\b",
    r"\bcompared\b.*\bto\b",
    r"\bdifference\b.*\bbetween\b",
    r"\bwhich\b.*\bbetter\b",
    r"\bcompare\b",
]

_DATASET_SEARCH_PATTERNS = [
    r"\bdataset\b",
    r"\bbenchmark\b",
    r"\btrained on\b",
    r"\bevaluated on\b",
    r"\btested on\b",
    r"\busing\b.*\bdata\b",
    r"\bpapers?\b.*\busing\b.*(?:MNIST|CIFAR|ImageNet|SQuAD|GLUE|CoNLL)",
]

_PAPER_LOOKUP_PATTERNS = [
    r"\bpaper\b.*\btitled\b",
    r"\bfind\b.*\bpaper\b.*\bcalled\b",
    r"\blook\s?up\b",
    r"\bdetails\b.*\bpaper\b",
    r"\bwhat\b.*\bpaper\b.*\babout\b",
    r"\btell me about\b.*\bpaper\b",
]

_METHOD_USAGE_PATTERNS = [
    r"\bpapers?\b.*\busing\b",
    r"\bpapers?\b.*\bthat\s+use\b",
    r"\bapplied\b.*\bmethod\b",
    r"\bapplications?\s+of\b",
    r"\bwho\s+uses?\b",
    r"\bused\b.*\bfor\b",
    r"\bwhere\s+is\b.*\bused\b",
    r"\bhow\s+is\b.*\bused\b",
]

_TOPIC_SEARCH_PATTERNS = [
    r"\bwhat are\b.*\bapproaches\b",
    r"\bsurvey\b",
    r"\boverview\b",
    r"\bstate of the art\b",
    r"\brecent\b.*\bwork\b",
    r"\bwhich papers\b",
    r"\bfind papers\b",
    r"\bsearch for\b",
    r"\blist\b.*\bpapers\b",
    r"\bresearch\b.*\bon\b",
    r"\bpapers\b.*\babout\b",
    r"\bpapers\b.*\bin\b.*\bfield\b",
]


def classify_query(question: str) -> QueryType:
    """
    Classify a user question into one of six supported query types.

    Checks patterns in priority order:
      1. claim_verification  (highest specificity)
      2. method_comparison
      3. dataset_search
      4. paper_lookup
      5. method_usage
      6. topic_search        (broadest, fallback)

    Args:
        question: The user's natural language question.

    Returns:
        The detected QueryType.
    """
    q = question.lower().strip()

    checks: list[tuple[list[str], QueryType]] = [
        (_CLAIM_PATTERNS, QueryType.CLAIM_VERIFICATION),
        (_METHOD_COMPARISON_PATTERNS, QueryType.METHOD_COMPARISON),
        (_DATASET_SEARCH_PATTERNS, QueryType.DATASET_SEARCH),
        (_PAPER_LOOKUP_PATTERNS, QueryType.PAPER_LOOKUP),
        (_METHOD_USAGE_PATTERNS, QueryType.METHOD_USAGE),
        (_TOPIC_SEARCH_PATTERNS, QueryType.TOPIC_SEARCH),
    ]

    for patterns, query_type in checks:
        for pattern in patterns:
            if re.search(pattern, q):
                return query_type

    # Default fallback
    return QueryType.TOPIC_SEARCH
