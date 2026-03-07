"""
Entity extractor: identifies typed scientific entities from user questions.

Extracts entities of these types:
  - method:   ML/AI methods, algorithms, models (CNN, SVM, BERT, etc.)
  - dataset:  Benchmark datasets (MNIST, CIFAR, ImageNet, etc.)
  - task:     Research tasks/problems (classification, translation, etc.)
  - field:    Research fields/domains (NLP, computer vision, etc.)
  - metric:   Evaluation metrics (accuracy, F1, BLEU, etc.)

Uses a combination of:
  1. Dictionary-based matching (known entities)
  2. Pattern-based extraction (capitalised terms, acronyms)
  3. Contextual heuristics (words near "using", "on", etc.)

This avoids the overhead of an NER model while being extensible.
"""

import re
from dataclasses import dataclass, field


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


# ── Known entity dictionaries ──────────────────────────────────────
# These are expanded as the system encounters more queries.
# Case-insensitive matching is used.

KNOWN_METHODS = {
    # Neural networks
    "cnn", "convolutional neural network", "rnn", "recurrent neural network",
    "lstm", "gru", "transformer", "bert", "gpt", "gpt-2", "gpt-3", "gpt-4",
    "roberta", "xlnet", "t5", "bart", "electra", "albert", "distilbert",
    "resnet", "vgg", "alexnet", "inception", "mobilenet", "efficientnet",
    "unet", "u-net", "yolo", "faster rcnn", "mask rcnn", "ssd",
    "gan", "generative adversarial network", "vae", "variational autoencoder",
    "autoencoder", "diffusion model", "stable diffusion",
    # Classical ML
    "svm", "support vector machine", "random forest", "decision tree",
    "logistic regression", "linear regression", "naive bayes", "knn",
    "k-nearest neighbor", "gradient boosting", "xgboost", "lightgbm",
    "catboost", "adaboost", "bagging", "ensemble",
    # Reinforcement learning
    "reinforcement learning", "q-learning", "deep q-network", "dqn",
    "ppo", "a3c", "ddpg", "sac", "td3", "policy gradient",
    # Other
    "attention mechanism", "self-attention", "multi-head attention",
    "word2vec", "glove", "fasttext", "elmo",
    "pca", "kmeans", "k-means", "dbscan", "spectral clustering",
    "neural network", "deep learning", "machine learning",
    "transfer learning", "fine-tuning", "few-shot learning",
    "zero-shot learning", "meta-learning", "contrastive learning",
    "federated learning", "graph neural network", "gnn",
}

KNOWN_DATASETS = {
    # Vision
    "mnist", "fashion-mnist", "cifar", "cifar-10", "cifar-100",
    "imagenet", "coco", "ms coco", "pascal voc", "celeba", "lfw",
    "svhn", "stl-10", "caltech", "oxford flowers",
    # NLP
    "squad", "squad 2.0", "glue", "superglue", "mrpc", "sst-2",
    "mnli", "qnli", "wnli", "rte", "cola", "sts-b",
    "conll", "conll-2003", "ontonotes", "wikitext",
    "imdb", "yelp", "amazon reviews", "ag news",
    "penn treebank", "wmt", "europarl", "opus",
    # Biomedical
    "pubmed", "mimic", "mimic-iii", "chestx-ray", "isic",
    # Other
    "uci", "kaggle", "openml",
}

KNOWN_TASKS = {
    "classification", "image classification", "text classification",
    "object detection", "semantic segmentation", "instance segmentation",
    "sentiment analysis", "named entity recognition", "ner",
    "machine translation", "question answering", "summarization",
    "text generation", "language modeling", "speech recognition",
    "image generation", "image segmentation", "pose estimation",
    "relation extraction", "information extraction", "entity linking",
    "topic modeling", "clustering", "regression", "anomaly detection",
    "recommendation", "retrieval", "information retrieval",
    "medical diagnosis", "drug discovery", "clinical prediction",
    "autonomous driving", "robot navigation",
}

KNOWN_FIELDS = {
    "natural language processing", "nlp", "computer vision",
    "machine learning", "deep learning", "artificial intelligence",
    "reinforcement learning", "robotics", "bioinformatics",
    "computational biology", "medical imaging", "healthcare",
    "speech processing", "signal processing", "graph learning",
    "knowledge graphs", "information retrieval", "data mining",
    "neural architecture search", "explainability", "fairness",
}

KNOWN_METRICS = {
    "accuracy", "precision", "recall", "f1", "f1-score", "f1 score",
    "auc", "roc", "roc-auc", "map", "mean average precision",
    "bleu", "rouge", "meteor", "cider", "perplexity",
    "mse", "rmse", "mae", "r-squared", "r2",
    "iou", "dice", "ap", "top-1", "top-5",
    "exact match", "em",
}

# Words that should not be treated as standalone entities
_STOP_ENTITIES = {
    "paper", "papers", "method", "methods", "model", "models",
    "result", "results", "approach", "approaches", "system",
    "the", "and", "for", "with", "that", "this", "from",
    "using", "used", "use", "which", "does", "how", "what",
    "where", "when", "between", "about", "compare", "compared",
    "comparison", "find", "search", "list", "show", "claim",
}


def extract_entities(question: str) -> ExtractedEntities:
    """
    Extract typed scientific entities from a natural language question.

    Strategy:
      1. Match against known dictionaries (highest confidence)
      2. Extract capitalised terms / acronyms as candidate methods
      3. Extract terms following contextual cues ("on <dataset>", "using <method>")

    Args:
        question: The user's natural language question.

    Returns:
        ExtractedEntities with populated typed entity lists.
    """
    entities = ExtractedEntities()
    q_lower = question.lower()

    # --- Phase 1: Dictionary matching ---
    entities.methods = _match_known(q_lower, KNOWN_METHODS)
    entities.datasets = _match_known(q_lower, KNOWN_DATASETS)
    entities.tasks = _match_known(q_lower, KNOWN_TASKS)
    entities.fields = _match_known(q_lower, KNOWN_FIELDS)
    entities.metrics = _match_known(q_lower, KNOWN_METRICS)

    # --- Phase 2: Acronym / capitalised term extraction ---
    # Captures uppercase terms like CNN, BERT, SVM, LSTM, etc.
    acronyms = re.findall(r"\b([A-Z][A-Z0-9]{1,10}(?:-[A-Z0-9]+)?)\b", question)
    for acr in acronyms:
        acr_lower = acr.lower()
        if acr_lower in _STOP_ENTITIES:
            continue
        # If already matched, skip
        if acr_lower in {e.lower() for e in entities.all_entities()}:
            continue
        # Heuristic: unrecognized acronyms are likely methods
        entities.methods.append(acr)

    # --- Phase 3: Contextual extraction ---
    # "on <X>" often indicates a dataset
    on_matches = re.findall(r"\bon\s+([A-Z][A-Za-z0-9-]+)", question)
    for match in on_matches:
        if match.lower() not in {e.lower() for e in entities.all_entities()}:
            if match.lower() not in _STOP_ENTITIES:
                entities.datasets.append(match)

    # "for <X>" often indicates a task
    for_matches = re.findall(r"\bfor\s+([a-z][a-z ]{3,30}?)(?:\?|$|,|\band\b)", q_lower)
    for match in for_matches:
        match = match.strip()
        if match not in {e.lower() for e in entities.all_entities()}:
            if match not in _STOP_ENTITIES and len(match) > 3:
                entities.tasks.append(match)

    # Deduplicate while preserving order
    entities.methods = _dedupe(entities.methods)
    entities.datasets = _dedupe(entities.datasets)
    entities.tasks = _dedupe(entities.tasks)
    entities.fields = _dedupe(entities.fields)
    entities.metrics = _dedupe(entities.metrics)

    return entities


def _match_known(text: str, known_set: set[str]) -> list[str]:
    """
    Find all known entities that appear in the text.

    Uses word-boundary matching with optional plural suffix (``s?``)
    so that both "neural network" and "neural networks" match the
    canonical form ``neural network``.

    Returns entities in their canonical (lowercase) form.
    """
    found = []
    for entity in sorted(known_set, key=len, reverse=True):
        # Longer entities first to avoid substring issues (e.g., "cifar-10" before "cifar")
        # Allow an optional trailing 's' so plurals like "neural networks" match
        pattern = r"\b" + re.escape(entity) + r"s?\b"
        if re.search(pattern, text, re.IGNORECASE):
            # Don't add if a longer form was already matched
            if not any(entity.lower() in f.lower() and entity != f for f in found):
                found.append(entity)
    return found


def _dedupe(items: list[str]) -> list[str]:
    """Deduplicate a list preserving order, case-insensitive."""
    seen: set[str] = set()
    result = []
    for item in items:
        key = item.lower()
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


# ── Keyword extraction fallback ─────────────────────────────────
# Used when dictionary-based extraction returns nothing.

_QUESTION_STOP_WORDS = {
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

    This is a **fallback** for when dictionary-based entity extraction
    returns empty results.  It groups consecutive non-stop-words into
    phrases (e.g. "convolutional neural networks" stays as one phrase)
    and returns them for use in broad SPARQL searches.

    Args:
        question: The original user question.

    Returns:
        List of keyword phrases, may be empty.
    """
    cleaned = re.sub(r"[^\w\s-]", "", question.lower())
    words = cleaned.split()

    # Group consecutive significant words into phrases
    keywords: list[str] = []
    current_phrase: list[str] = []

    for word in words:
        if word not in _QUESTION_STOP_WORDS and len(word) > 2:
            current_phrase.append(word)
        else:
            if current_phrase:
                keywords.append(" ".join(current_phrase))
                current_phrase = []
    if current_phrase:
        keywords.append(" ".join(current_phrase))

    return keywords
