"""
Entity normalization: expands extracted entities into variant forms
to improve SPARQL recall.

Scientific entities appear in many surface forms:
    CNN  →  convolutional neural network, DCNN, deep convolutional network
    MNIST →  benchmark MNIST, sequential MNIST

This module provides a simple dictionary-based expansion.
No ML models are needed — the synonyms are curated from common usage.

Usage:
    from backend.rag.entity_normalization import expand_entities

    entities = extract_entities("Which papers use CNN on MNIST?")
    expanded = expand_entities(entities)
    # expanded.methods = ["CNN", "convolutional neural network", "DCNN", ...]
"""

from __future__ import annotations

from dataclasses import dataclass, field

from backend.rag.entity_extractor import ExtractedEntities


# ── Synonym dictionaries ────────────────────────────────────────
# Keys are canonical (lowercase) forms.  Values are alternative forms
# that should also be searched in the knowledge graph.

METHOD_SYNONYMS: dict[str, list[str]] = {
    "cnn": [
        "convolutional neural network",
        "dcnn",
        "deep convolutional network",
        "convnet",
    ],
    "convolutional neural network": ["cnn", "dcnn", "convnet"],
    "rnn": ["recurrent neural network"],
    "recurrent neural network": ["rnn"],
    "lstm": ["long short-term memory", "long short term memory"],
    "long short-term memory": ["lstm"],
    "gru": ["gated recurrent unit"],
    "gated recurrent unit": ["gru"],
    "bert": ["bidirectional encoder representations from transformers"],
    "gpt": ["generative pre-trained transformer"],
    "gpt-2": ["gpt2", "generative pre-trained transformer 2"],
    "gpt-3": ["gpt3", "generative pre-trained transformer 3"],
    "gpt-4": ["gpt4"],
    "svm": ["support vector machine", "support vector machines"],
    "support vector machine": ["svm"],
    "random forest": ["rf", "random forests"],
    "decision tree": ["decision trees", "dt"],
    "knn": ["k-nearest neighbor", "k-nearest neighbours", "k nearest neighbor"],
    "k-nearest neighbor": ["knn"],
    "xgboost": ["extreme gradient boosting", "xgb"],
    "lightgbm": ["light gradient boosting machine", "lgbm"],
    "gradient boosting": ["gbdt", "gradient boosted decision tree"],
    "naive bayes": ["naïve bayes", "nb"],
    "logistic regression": ["logreg"],
    "resnet": ["residual network", "residual neural network"],
    "vgg": ["vggnet"],
    "yolo": ["you only look once"],
    "gan": ["generative adversarial network", "generative adversarial networks"],
    "generative adversarial network": ["gan"],
    "vae": ["variational autoencoder", "variational auto-encoder"],
    "variational autoencoder": ["vae"],
    "transformer": ["transformers", "transformer model"],
    "transfer learning": ["tl", "domain adaptation"],
    "reinforcement learning": ["rl"],
    "deep learning": ["dl"],
    "machine learning": ["ml"],
    "graph neural network": ["gnn", "graph neural networks"],
    "gnn": ["graph neural network"],
    "pca": ["principal component analysis"],
    "word2vec": ["word 2 vec", "word-to-vec"],
    "attention mechanism": ["attention", "self-attention", "self attention"],
}

DATASET_SYNONYMS: dict[str, list[str]] = {
    "mnist": ["benchmark mnist", "sequential mnist", "handwritten mnist"],
    "fashion-mnist": ["fashion mnist", "fmnist"],
    "cifar-10": ["cifar10", "cifar 10"],
    "cifar-100": ["cifar100", "cifar 100"],
    "cifar": ["cifar-10", "cifar-100"],
    "imagenet": ["ilsvrc", "imagenet-1k", "imagenet 1k"],
    "coco": ["ms coco", "ms-coco", "microsoft coco"],
    "ms coco": ["coco", "ms-coco"],
    "squad": ["stanford question answering dataset", "squad 1.1"],
    "squad 2.0": ["squad2", "squad v2"],
    "glue": ["general language understanding evaluation"],
    "superglue": ["super glue", "super-glue"],
    "conll": ["conll-2003", "conll 2003"],
    "conll-2003": ["conll", "conll2003"],
    "imdb": ["imdb reviews", "imdb movie reviews"],
    "penn treebank": ["ptb"],
    "wikitext": ["wikitext-2", "wikitext-103"],
    "wmt": ["workshop on machine translation"],
    "pascal voc": ["voc", "pascal visual object classes"],
    "svhn": ["street view house numbers"],
    "mimic": ["mimic-iii", "mimic iii", "mimic-3"],
    "mimic-iii": ["mimic", "mimic iii"],
    "pubmed": ["pub med"],
}


@dataclass
class ExpandedEntities:
    """
    Holds both the original canonical entities and their expanded variants.

    Attributes:
        methods:  original extracted methods
        datasets: original extracted datasets
        method_variants:  {canonical → [variant1, variant2, ...]}
        dataset_variants: {canonical → [variant1, variant2, ...]}
        tasks / fields / metrics: passed through unchanged
    """

    methods: list[str] = field(default_factory=list)
    datasets: list[str] = field(default_factory=list)
    tasks: list[str] = field(default_factory=list)
    fields: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    method_variants: dict[str, list[str]] = field(default_factory=dict)
    dataset_variants: dict[str, list[str]] = field(default_factory=dict)

    def all_method_forms(self, method: str) -> list[str]:
        """Return the canonical form + all variants for a method."""
        variants = self.method_variants.get(method.lower(), [])
        return [method] + variants

    def all_dataset_forms(self, dataset: str) -> list[str]:
        """Return the canonical form + all variants for a dataset."""
        variants = self.dataset_variants.get(dataset.lower(), [])
        return [dataset] + variants

    def all_entities(self) -> list[str]:
        """Return all original entities as a flat list."""
        return self.methods + self.datasets + self.tasks + self.fields + self.metrics

    def to_dict(self) -> dict:
        return {
            "methods": self.methods,
            "datasets": self.datasets,
            "tasks": self.tasks,
            "fields": self.fields,
            "metrics": self.metrics,
            "method_variants": self.method_variants,
            "dataset_variants": self.dataset_variants,
        }


def expand_entities(entities: ExtractedEntities) -> ExpandedEntities:
    """
    Expand extracted entities with their known synonym variants.

    For each method/dataset, looks up the synonym dictionary and attaches
    all known alternative surface forms.  Tasks, fields, and metrics are
    passed through unchanged.

    Args:
        entities: Raw entities from the entity extractor.

    Returns:
        ExpandedEntities with variant mappings populated.
    """
    expanded = ExpandedEntities(
        methods=list(entities.methods),
        datasets=list(entities.datasets),
        tasks=list(entities.tasks),
        fields=list(entities.fields),
        metrics=list(entities.metrics),
    )

    for method in entities.methods:
        key = method.lower()
        if key in METHOD_SYNONYMS:
            expanded.method_variants[key] = METHOD_SYNONYMS[key]

    for dataset in entities.datasets:
        key = dataset.lower()
        if key in DATASET_SYNONYMS:
            expanded.dataset_variants[key] = DATASET_SYNONYMS[key]

    return expanded


def get_method_variants(method: str) -> list[str]:
    """Return synonym variants for a method (empty list if unknown)."""
    return METHOD_SYNONYMS.get(method.lower(), [])


def get_dataset_variants(dataset: str) -> list[str]:
    """Return synonym variants for a dataset (empty list if unknown)."""
    return DATASET_SYNONYMS.get(dataset.lower(), [])
