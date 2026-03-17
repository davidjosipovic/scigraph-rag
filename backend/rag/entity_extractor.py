"""
Entity extractor: identifies typed scientific entities from user questions.

Extracts entities of these types:
  - method:   ML/AI methods, algorithms, models (CNN, SVM, BERT, etc.)
  - dataset:  Benchmark datasets (MNIST, CIFAR, ImageNet, etc.)
  - task:     Research tasks/problems (classification, translation, etc.)
  - field:    Research fields/domains (NLP, computer vision, etc.)
  - metric:   Evaluation metrics (accuracy, F1, BLEU, etc.)

Uses Llama 3 via Ollama for extraction. Falls back to keyword extraction
if the model is unavailable or returns unparseable output.
"""

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache

from loguru import logger

from backend.llm.ollama_client import ollama_client


@dataclass
class ExtractedEntities:
    """Container for entities extracted from a user question."""

    methods: list[str] = field(default_factory=list)
    datasets: list[str] = field(default_factory=list)
    tasks: list[str] = field(default_factory=list)
    fields: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "methods": self.methods,
            "datasets": self.datasets,
            "tasks": self.tasks,
            "fields": self.fields,
            "metrics": self.metrics,
        }

    def all_entities(self) -> list[str]:
        """Return all extracted entities as a flat list."""
        return self.methods + self.datasets + self.tasks + self.fields + self.metrics

    def is_empty(self) -> bool:
        return not any([self.methods, self.datasets, self.tasks, self.fields, self.metrics])


_EXTRACTOR_SYSTEM_PROMPT = """You are a scientific entity extractor for a research paper search system.

Given a user question, extract ONLY the named entities that are EXPLICITLY and LITERALLY present in the question text. Do NOT infer, predict, or add entities that are not directly written in the question.

Classify the entities you find into these categories:
- methods: ML/AI methods, algorithms, architectures, or models explicitly named (e.g. BERT, CNN, SVM)
- datasets: benchmark datasets explicitly named (e.g. MNIST, ImageNet, SQuAD)
- tasks: research tasks explicitly stated (e.g. "image classification", "sentiment analysis")
- fields: research fields explicitly stated (e.g. "natural language processing", "medical imaging")
- metrics: evaluation metrics explicitly named (e.g. accuracy, F1, BLEU)

IMPORTANT RULES:
- If a method is not named in the question, leave methods as []
- If a dataset is not named in the question, leave datasets as []
- Do NOT add methods or datasets you think might be relevant — only what is written
- "deep learning" or "machine learning" alone is a field, not a method

Respond ONLY with a valid JSON object with these exact keys. Use empty lists if nothing is found.
Example: question "Which papers use CNN on MNIST?" → {"methods": ["CNN"], "datasets": ["MNIST"], "tasks": [], "fields": [], "metrics": []}
Example: question "Papers about deep learning for medical imaging" → {"methods": [], "datasets": [], "tasks": [], "fields": ["deep learning", "medical imaging"], "metrics": []}
Example: question "Which papers report F1 score on NER tasks?" → {"methods": [], "datasets": [], "tasks": ["NER", "named entity recognition"], "fields": [], "metrics": ["F1"]}
Note: NER, POS tagging, NLI, STS, QA, MT are tasks — they are NOT datasets."""


@lru_cache(maxsize=256)
def extract_entities(question: str) -> ExtractedEntities:
    """
    Extract typed scientific entities from a natural language question using Llama 3.

    Falls back to empty ExtractedEntities if the model is unavailable
    or returns unparseable output (pipeline.py handles the empty case
    via extract_keywords fallback).

    Args:
        question: The user's natural language question.

    Returns:
        ExtractedEntities with populated typed entity lists.
    """
    try:
        raw = ollama_client.generate(
            prompt=question,
            system=_EXTRACTOR_SYSTEM_PROMPT,
            temperature=0.0,
            max_tokens=256,
        )

        # Extract JSON from response (model may wrap it in prose)
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not json_match:
            logger.warning(f"Entity extractor: no JSON found in response: '{raw[:100]}'")
            return ExtractedEntities()

        data = json.loads(json_match.group())

        entities = ExtractedEntities(
            methods=_as_str_list(data.get("methods")),
            datasets=_as_str_list(data.get("datasets")),
            tasks=_as_str_list(data.get("tasks")),
            fields=_as_str_list(data.get("fields")),
            metrics=_as_str_list(data.get("metrics")),
        )
        logger.info(f"Entity extraction complete: {entities.to_dict()}")
        return entities

    except json.JSONDecodeError as e:
        logger.warning(f"Entity extractor: JSON parse error: {e} — response: '{raw[:100]}'")
        return ExtractedEntities()
    except Exception as e:
        logger.error(f"Entity extraction failed: {e}")
        return ExtractedEntities()


def _as_str_list(value: object) -> list[str]:
    """Ensure a JSON value is a list of non-empty strings."""
    if not isinstance(value, list):
        return []
    return [str(v).strip() for v in value if str(v).strip()]


# ── Keyword extraction fallback ──────────────────────────────────────────────
# Used by pipeline.py (Step 2b) when extract_entities returns empty results.

_STOP_WORDS = {
    "what", "which", "where", "when", "how", "who", "whom", "whose",
    "does", "did", "that", "this", "have", "has", "had", "been",
    "with", "from", "some", "about", "more", "most", "than",
    "them", "they", "their", "there", "these", "those",
    "paper", "papers", "article", "articles", "study", "studies",
    "research", "work", "works", "result", "results",
    "find", "list", "show", "tell", "give", "describe",
    "explain", "discuss", "use", "used", "using", "uses",
    "are", "is", "was", "were", "will", "would", "could", "should",
    "can", "may", "might", "shall", "must",
    "the", "and", "for", "but", "not", "all", "any", "each", "every",
    "also", "very", "just", "only", "other", "into",
    "compare", "compared", "comparison", "between",
    "method", "methods", "model", "models", "approach", "approaches",
}


def extract_keywords(question: str) -> list[str]:
    """
    Extract significant keywords / phrases from a question.

    Fallback for when LLM-based entity extraction returns empty results.
    Groups consecutive non-stop-words into phrases.

    Args:
        question: The original user question.

    Returns:
        List of keyword phrases, may be empty.
    """
    cleaned = re.sub(r"[^\w\s-]", "", question.lower())
    words = cleaned.split()

    keywords: list[str] = []
    current_phrase: list[str] = []

    for word in words:
        if word not in _STOP_WORDS and len(word) > 2:
            current_phrase.append(word)
        else:
            if current_phrase:
                keywords.append(" ".join(current_phrase))
                current_phrase = []
    if current_phrase:
        keywords.append(" ".join(current_phrase))

    return keywords