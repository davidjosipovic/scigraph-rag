"""
Abstract base class for LLM clients.

All LLM provider implementations (Ollama, OpenAI, Anthropic) must implement
this interface so the pipeline can swap providers without changing any other
code.

Also contains shared prompt templates used by all providers — the templates
are provider-agnostic since they only define the output structure, not the
model's internals.
"""

from abc import ABC, abstractmethod


# ── System Prompt ────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a scientific research assistant that answers questions based on information from the Open Research Knowledge Graph (ORKG).

Your responses must:
1. Be grounded ONLY in the provided knowledge graph context
2. Reference papers by their EXACT title as shown in the context (e.g. "According to the paper Effective Handwritten Digit Recognition...")
3. NEVER invent citation numbers like [1], [2] — always use the paper title
4. NEVER invent paper titles, authors, or results that are not in the context
5. Clearly distinguish between what the evidence shows and your interpretation
6. If the evidence is insufficient, say so explicitly
7. Be concise but thorough

Do NOT:
- Make up papers, authors, or results not in the provided context
- Use numeric citations like [1], [2], [3]
- Use resource IDs like (R213192) or URIs — reference papers by title only
- Fabricate claims or findings
- Guess answers when evidence is missing
- Include papers that are not listed in the context

Only use information from the provided context.
When referencing a paper, always use its full title."""


# ── Prompt Templates (one per query type) ────────────────────────

QA_PROMPT_TEMPLATE = """Based on the following knowledge graph context, answer the user's question.

{context}

Question: {question}

Provide a clear, well-structured answer with these sections:
1. **Overview**: Brief answer to the question
2. **Relevant Papers**: Describe the most relevant papers, referencing each by its exact title
3. **Methods Used**: List methods mentioned in the retrieved papers
4. **Datasets Used**: List datasets mentioned in the retrieved papers
5. **Sources**: List all paper titles you referenced

IMPORTANT: Do NOT use numeric citations like [1] or [2]. Always reference papers by their exact title from the context above.
If the retrieved information is insufficient, state what is known and what is missing."""

CLAIM_VERIFICATION_TEMPLATE = """Based on the following evidence from the knowledge graph, verify the user's claim.

{context}

Claim to verify: {question}

Analyze the evidence and respond with:
1. **Verdict**: SUPPORTED / REFUTED / INSUFFICIENT EVIDENCE
2. **Evidence**: Summarize the relevant findings, referencing each paper by its exact title
3. **Methods**: Methods mentioned in the evidence
4. **Datasets**: Datasets mentioned in the evidence
5. **Sources**: List all paper titles you referenced

IMPORTANT: Do NOT use numeric citations like [1] or [2]. Always reference papers by their exact title.
Only base your verdict on the evidence provided. Do not fabricate results."""

TOPIC_SEARCH_TEMPLATE = """Based on the following papers retrieved from the knowledge graph, provide a summary of the research topic.

{context}

Topic query: {question}

Provide:
1. **Overview**: A brief summary of the research area based on the retrieved papers
2. **Relevant Papers**: Highlight the most relevant papers, referencing each by its exact title
3. **Methods Used**: Methods or approaches found in these papers
4. **Datasets Used**: Datasets found in these papers
5. **Sources**: List all paper titles you referenced

IMPORTANT: Do NOT use numeric citations like [1] or [2]. Always reference papers by their exact title from the context above."""

METHOD_COMPARISON_TEMPLATE = """Based on the following papers retrieved from the knowledge graph, compare the methods mentioned in the question.

{context}

Question: {question}

Provide:
1. **Overview**: Brief summary of the comparison
2. **Relevant Papers**: Papers that compare or use these methods, referenced by exact title
3. **Methods Used**: The methods being compared and how they are used
4. **Datasets Used**: Datasets the methods were evaluated on
5. **Sources**: List all paper titles you referenced

IMPORTANT: Do NOT use numeric citations like [1] or [2]. Always reference papers by their exact title.
Only report findings present in the evidence. If comparisons are limited, state this clearly."""

DATASET_SEARCH_TEMPLATE = """Based on the following papers retrieved from the knowledge graph, answer the question about datasets.

{context}

Question: {question}

Provide:
1. **Overview**: Brief answer about the datasets
2. **Relevant Papers**: Papers using these datasets, referenced by exact title
3. **Methods Used**: Methods applied to these datasets
4. **Datasets Used**: The datasets found and any details about them
5. **Sources**: List all paper titles you referenced

IMPORTANT: Do NOT use numeric citations like [1] or [2]. Always reference papers by their exact title from the context above."""

METHOD_USAGE_TEMPLATE = """Based on the following papers retrieved from the knowledge graph, describe how the mentioned method(s) are used.

{context}

Question: {question}

Provide:
1. **Overview**: Brief description of how the method(s) are used
2. **Relevant Papers**: Papers using these methods, referenced by exact title
3. **Methods Used**: The methods and how/where they are applied
4. **Datasets Used**: Datasets the methods were applied to
5. **Sources**: List all paper titles you referenced

IMPORTANT: Do NOT use numeric citations like [1] or [2]. Always reference papers by their exact title from the context above."""

PAPER_LOOKUP_TEMPLATE = """Based on the following information from the knowledge graph, provide details about the requested paper(s).

{context}

Question: {question}

Provide:
1. **Overview**: Summary of the paper(s) found
2. **Relevant Papers**: Details including title, DOI, research field
3. **Methods Used**: Methods described in the contributions
4. **Datasets Used**: Datasets referenced
5. **Sources**: List all paper titles you referenced

IMPORTANT: Do NOT use numeric citations like [1] or [2]. Always reference papers by their exact title from the context above."""


def get_prompt_template(query_type: str) -> str:
    """Return the appropriate prompt template for the given query type."""
    templates = {
        "claim_verification": CLAIM_VERIFICATION_TEMPLATE,
        "topic_search": TOPIC_SEARCH_TEMPLATE,
        "method_comparison": METHOD_COMPARISON_TEMPLATE,
        "dataset_search": DATASET_SEARCH_TEMPLATE,
        "method_usage": METHOD_USAGE_TEMPLATE,
        "paper_lookup": PAPER_LOOKUP_TEMPLATE,
    }
    return templates.get(query_type, QA_PROMPT_TEMPLATE)


# ── Abstract base class ──────────────────────────────────────────

class BaseLLMClient(ABC):
    """
    Abstract interface for LLM provider clients.

    All providers (Ollama, OpenAI, Anthropic) implement this interface
    so the RAG pipeline can swap the generation model without any other
    code changes.

    Subclasses must implement:
        generate(prompt, system, temperature, max_tokens) -> str
        is_available() -> bool

    Subclasses must set:
        self.model: str   — human-readable model identifier
    """

    model: str

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        """
        Generate a text completion.

        Args:
            prompt:      The user/context prompt.
            system:      Optional system prompt override.
            temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative).
            max_tokens:  Maximum tokens to generate.

        Returns:
            Generated text string. On error, returns an error message string
            rather than raising — so the pipeline always gets a response.
        """

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the provider is reachable and the model is usable.

        Returns:
            True if the provider is ready, False otherwise.
        """
