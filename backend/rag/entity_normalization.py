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
    # ── Convolutional networks ───────────────────────────────────────
    "cnn": ["convolutional neural network", "dcnn", "deep convolutional network", "convnet"],
    "convolutional neural network": ["cnn", "dcnn", "convnet"],

    # ── Recurrent networks ───────────────────────────────────────────
    "rnn": ["recurrent neural network"],
    "recurrent neural network": ["rnn"],
    "lstm": ["long short-term memory", "long short term memory"],
    "long short-term memory": ["lstm"],
    "gru": ["gated recurrent unit"],
    "gated recurrent unit": ["gru"],

    # ── Transformer-based language models ────────────────────────────
    "transformer": ["transformers", "transformer model", "transformer architecture"],
    "bert": ["bidirectional encoder representations from transformers", "bert-base", "bert-large"],
    "bidirectional encoder representations from transformers": ["bert"],
    "roberta": ["robustly optimized bert", "roberta-base", "roberta-large"],
    "distilbert": ["distilled bert", "distil-bert"],
    "albert": ["a lite bert", "albert-base", "albert-large"],
    "electra": ["efficiently learning an encoder that classifies token replacements accurately"],
    "xlnet": ["xl-net", "autoregressive language model"],
    "t5": ["text-to-text transfer transformer", "t5-base", "t5-large", "t5-small"],
    "text-to-text transfer transformer": ["t5"],
    "bart": ["denoising sequence-to-sequence pre-training", "bart-base", "bart-large"],
    "gpt": ["generative pre-trained transformer", "gpt-1"],
    "gpt-2": ["gpt2", "generative pre-trained transformer 2"],
    "gpt-3": ["gpt3", "generative pre-trained transformer 3"],
    "gpt-4": ["gpt4"],
    "llama": ["llama 1", "meta llama", "large language model meta ai"],
    "llama 2": ["llama2", "llama-2"],
    "llama 3": ["llama3", "llama-3"],
    "mistral": ["mistral 7b", "mistral-7b"],
    "gemma": ["gemma 2", "google gemma"],
    "falcon": ["falcon 40b", "falcon-40b"],

    # ── Vision transformers ──────────────────────────────────────────
    "vit": ["vision transformer", "vision transformers"],
    "vision transformer": ["vit"],
    "deit": ["data-efficient image transformers", "data efficient image transformer"],
    "swin": ["swin transformer", "shifted window transformer"],
    "swin transformer": ["swin"],
    "clip": ["contrastive language-image pretraining", "contrastive language image pretraining"],
    "contrastive language-image pretraining": ["clip"],

    # ── CNN architectures ────────────────────────────────────────────
    "resnet": ["residual network", "residual neural network", "residual network architecture"],
    "resnet-50": ["resnet50"],
    "resnet-101": ["resnet101"],
    "resnet-152": ["resnet152"],
    "vgg": ["vggnet", "vgg network"],
    "alexnet": ["alex net"],
    "inception": ["googlenet", "inception network", "inception v3"],
    "mobilenet": ["mobile net", "mobilenet v2", "mobilenet v3"],
    "efficientnet": ["efficient net", "efficientnet-b0", "efficientnet-b7"],
    "densenet": ["dense network", "densely connected network"],
    "squeezenet": ["squeeze net"],

    # ── Object detection ─────────────────────────────────────────────
    "yolo": ["you only look once", "yolov3", "yolov5", "yolov8"],
    "faster rcnn": ["faster r-cnn", "faster region-based cnn"],
    "faster r-cnn": ["faster rcnn"],
    "mask rcnn": ["mask r-cnn"],
    "mask r-cnn": ["mask rcnn"],
    "ssd": ["single shot multibox detector", "single shot detector"],
    "retinanet": ["retina net", "focal loss"],

    # ── Segmentation ─────────────────────────────────────────────────
    "unet": ["u-net"],
    "u-net": ["unet"],
    "deeplab": ["deeplab v3", "deeplab v3+", "atrous convolution"],

    # ── Generative models ────────────────────────────────────────────
    "gan": ["generative adversarial network", "generative adversarial networks"],
    "generative adversarial network": ["gan"],
    "vae": ["variational autoencoder", "variational auto-encoder"],
    "variational autoencoder": ["vae"],
    "diffusion model": ["diffusion models", "ddpm", "denoising diffusion probabilistic model"],
    "ddpm": ["denoising diffusion probabilistic model", "diffusion model"],
    "dall-e": ["dalle", "dall e"],
    "stable diffusion": ["latent diffusion model", "ldm"],

    # ── Contrastive / self-supervised ────────────────────────────────
    "simclr": ["simple contrastive learning of representations"],
    "moco": ["momentum contrast"],
    "contrastive learning": ["self-supervised contrastive learning"],

    # ── Classical ML ─────────────────────────────────────────────────
    "svm": ["support vector machine", "support vector machines", "support vector classification"],
    "support vector machine": ["svm"],
    "random forest": ["rf", "random forests"],
    "decision tree": ["decision trees", "dt", "cart"],
    "knn": ["k-nearest neighbor", "k-nearest neighbours", "k nearest neighbor"],
    "k-nearest neighbor": ["knn"],
    "xgboost": ["extreme gradient boosting", "xgb"],
    "lightgbm": ["light gradient boosting machine", "lgbm"],
    "catboost": ["categorical boosting"],
    "gradient boosting": ["gbdt", "gradient boosted decision tree", "gradient boosted trees"],
    "adaboost": ["adaptive boosting"],
    "naive bayes": ["naïve bayes", "nb", "gaussian naive bayes"],
    "logistic regression": ["logit regression", "logreg"],
    "linear regression": ["ordinary least squares", "ols"],

    # ── Clustering ───────────────────────────────────────────────────
    "k-means": ["kmeans", "k means clustering"],
    "kmeans": ["k-means", "k means clustering"],
    "dbscan": ["density-based spatial clustering of applications with noise"],
    "spectral clustering": ["normalized cuts"],

    # ── Dimensionality reduction ─────────────────────────────────────
    "pca": ["principal component analysis"],
    "principal component analysis": ["pca"],
    "t-sne": ["tsne", "t distributed stochastic neighbor embedding"],
    "umap": ["uniform manifold approximation and projection"],

    # ── Reinforcement learning ───────────────────────────────────────
    "reinforcement learning": ["rl"],
    "dqn": ["deep q-network", "deep q network"],
    "deep q-network": ["dqn"],
    "ppo": ["proximal policy optimization"],
    "proximal policy optimization": ["ppo"],
    "a3c": ["asynchronous advantage actor-critic"],
    "ddpg": ["deep deterministic policy gradient"],
    "sac": ["soft actor-critic"],
    "td3": ["twin delayed deep deterministic policy gradient"],

    # ── Embeddings / representations ─────────────────────────────────
    "word2vec": ["word 2 vec", "word-to-vec", "word embeddings"],
    "glove": ["global vectors for word representation", "glove embeddings"],
    "fasttext": ["fast text"],
    "elmo": ["embeddings from language models"],

    # ── Attention ────────────────────────────────────────────────────
    "attention mechanism": ["attention", "self-attention", "self attention", "multi-head attention"],

    # ── General paradigms ────────────────────────────────────────────
    "transfer learning": ["fine-tuning", "pretrain and fine-tune"],
    "deep learning": ["dl"],
    "machine learning": ["ml"],
    "graph neural network": ["gnn", "graph neural networks"],
    "gnn": ["graph neural network"],
    "federated learning": ["federated ml"],
    "meta-learning": ["learning to learn", "few-shot meta-learning"],
    "few-shot learning": ["few shot learning", "low-shot learning"],
    "zero-shot learning": ["zero shot learning", "zero-shot transfer"],
    "contrastive learning": ["contrastive self-supervised learning"],
    "knowledge distillation": ["model distillation", "model compression"],
    "neural architecture search": ["nas", "automated machine learning", "automl"],
    "nas": ["neural architecture search"],
    "automl": ["automated machine learning", "neural architecture search"],
}

DATASET_SYNONYMS: dict[str, list[str]] = {
    # ── Handwritten / digits ─────────────────────────────────────────
    "mnist": ["benchmark mnist", "sequential mnist", "handwritten mnist", "le cun mnist"],
    "fashion-mnist": ["fashion mnist", "fmnist", "zalando fashion mnist"],
    "svhn": ["street view house numbers"],

    # ── CIFAR ────────────────────────────────────────────────────────
    "cifar-10": ["cifar10", "cifar 10"],
    "cifar-100": ["cifar100", "cifar 100"],
    "cifar": ["cifar-10", "cifar-100", "cifar10", "cifar100"],

    # ── Large-scale vision ───────────────────────────────────────────
    "imagenet": ["ilsvrc", "imagenet-1k", "imagenet 1k", "imagenet-21k", "imagenet large scale visual recognition challenge"],
    "ilsvrc": ["imagenet"],
    "coco": ["ms coco", "ms-coco", "microsoft coco", "common objects in context"],
    "ms coco": ["coco", "ms-coco", "microsoft coco"],
    "open images": ["open images v4", "open images v6", "openimages"],
    "pascal voc": ["voc", "pascal visual object classes", "voc 2007", "voc 2012"],
    "lsun": ["large scale scene understanding"],
    "celeba": ["large-scale celeb faces attributes", "celeb-a"],
    "stl-10": ["stl10"],
    "oxford flowers": ["flowers 102", "102 category flower dataset"],
    "oxford pets": ["oxford-iiit pet dataset"],

    # ── Autonomous driving / 3D ──────────────────────────────────────
    "kitti": ["kitti vision benchmark", "karlsruhe institute of technology and toyota technological institute"],
    "nuscenes": ["nu-scenes", "nu scenes"],
    "waymo": ["waymo open dataset"],

    # ── NLP benchmarks ───────────────────────────────────────────────
    "glue": ["general language understanding evaluation", "glue benchmark"],
    "superglue": ["super glue", "super-glue", "superglue benchmark"],
    "squad": ["stanford question answering dataset", "squad 1.1", "squad v1"],
    "squad 2.0": ["squad2", "squad v2", "stanford question answering dataset 2"],
    "snli": ["stanford natural language inference", "stanford nli"],
    "stanford natural language inference": ["snli"],
    "mnli": ["multi-genre natural language inference", "multinli"],
    "multi-genre natural language inference": ["mnli", "multinli"],
    "sst-2": ["stanford sentiment treebank", "sst2", "sst 2"],
    "sst-5": ["sst5", "sst 5", "fine-grained sentiment treebank"],

    # ── Question answering ───────────────────────────────────────────
    "natural questions": ["nq", "google natural questions"],
    "nq": ["natural questions"],
    "triviaqa": ["trivia qa"],
    "hotpotqa": ["hotpot qa", "hotpot question answering"],
    "ms marco": ["microsoft machine reading comprehension", "msmarco"],
    "fever": ["fact extraction and verification"],

    # ── Named entity recognition ─────────────────────────────────────
    "conll": ["conll-2003", "conll 2003", "conll2003"],
    "conll-2003": ["conll", "conll2003"],
    "ontonotes": ["ontonotes 5.0"],

    # ── Sentiment / classification ───────────────────────────────────
    "imdb": ["imdb reviews", "imdb movie reviews", "imdb sentiment"],
    "ag news": ["ag's news", "ag news corpus"],
    "yelp": ["yelp reviews", "yelp polarity"],

    # ── Language modelling ───────────────────────────────────────────
    "penn treebank": ["ptb", "ptb-word"],
    "wikitext": ["wikitext-2", "wikitext-103", "wiki text"],
    "bookcorpus": ["books corpus", "toronto books corpus"],
    "common crawl": ["commoncrawl", "cc100"],

    # ── Machine translation ──────────────────────────────────────────
    "wmt": ["workshop on machine translation", "wmt14", "wmt16", "wmt17", "wmt19"],
    "wmt14": ["wmt 2014", "workshop on machine translation 2014"],
    "europarl": ["european parliament proceedings"],
    "opus": ["opus corpus"],

    # ── Speech ───────────────────────────────────────────────────────
    "librispeech": ["libri speech", "librispeech asr"],
    "voxceleb": ["vox celeb", "voxceleb1", "voxceleb2"],
    "commonvoice": ["common voice", "mozilla common voice"],

    # ── Biomedical / clinical ────────────────────────────────────────
    "mimic": ["mimic-iii", "mimic iii", "mimic-3"],
    "mimic-iii": ["mimic", "mimic iii"],
    "pubmed": ["pub med", "pubmed central", "medline"],
    "chestx-ray": ["chestxray", "nih chest x-ray", "chestx-ray14"],
    "isic": ["isic skin lesion", "skin lesion challenge"],

    # ── Multimodal / image-text ──────────────────────────────────────
    "laion": ["laion-400m", "laion-5b", "laion 400m", "laion 5b"],
    "cc3m": ["conceptual captions 3m", "conceptual captions"],
    "cc12m": ["conceptual captions 12m"],

    # ── Misc ─────────────────────────────────────────────────────────
    "uci": ["uci machine learning repository", "uci repository"],
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
