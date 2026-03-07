"""
Pydantic models for API request/response validation.

The response model includes full pipeline transparency:
  - entities extracted from the question
  - SPARQL queries used for retrieval
  - retrieval strategies employed
  - the generated answer
  - cited sources with metadata
"""

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """Request body for the /ask endpoint."""

    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="Natural language question about scientific papers.",
        examples=[
            "Which papers compare CNN and SVM?",
            "Does paper X claim that CNN outperforms SVM on MNIST?",
            "Where is reinforcement learning used in medical diagnosis?",
        ],
    )


class ExtractedEntitiesResponse(BaseModel):
    """Entities extracted from the user question."""

    methods: list[str] = Field(default_factory=list, description="ML methods/algorithms")
    datasets: list[str] = Field(default_factory=list, description="Benchmark datasets")
    tasks: list[str] = Field(default_factory=list, description="Research tasks/problems")
    fields: list[str] = Field(default_factory=list, description="Research fields")
    metrics: list[str] = Field(default_factory=list, description="Evaluation metrics")


class Source(BaseModel):
    """A single cited source from the Knowledge Graph."""

    title: str = Field(description="Paper title")
    uri: str = Field(description="ORKG resource URI")
    doi: str = Field(default="N/A", description="Paper DOI if available")
    methods: list[str] = Field(default_factory=list, description="Methods found in this paper")
    datasets: list[str] = Field(default_factory=list, description="Datasets found in this paper")


class AskResponse(BaseModel):
    """
    Full structured response from the GraphRAG pipeline.

    Includes all pipeline artifacts for transparency and traceability.
    """

    question: str = Field(description="The original question")
    query_type: str = Field(
        description=(
            "Detected query type: topic_search | method_comparison | "
            "dataset_search | claim_verification | method_usage | paper_lookup"
        )
    )
    entities: ExtractedEntitiesResponse = Field(
        description="Scientific entities extracted from the question"
    )
    sparql_queries: list[str] = Field(
        default_factory=list,
        description="SPARQL queries executed for KG retrieval",
    )
    strategies_used: list[str] = Field(
        default_factory=list,
        description="Retrieval strategies employed (e.g., method(CNN), field(NLP))",
    )
    answer: str = Field(description="LLM-generated answer grounded in KG context")
    sources: list[Source] = Field(
        default_factory=list,
        description="Cited papers from the Knowledge Graph",
    )
    kg_results_count: int = Field(
        default=0,
        description="Total results retrieved from the Knowledge Graph",
    )


class HealthResponse(BaseModel):
    """Response body for the /health endpoint."""

    llm: dict = Field(description="LLM component status")
    sparql: dict = Field(description="SPARQL endpoint status")
    pipeline: str = Field(description="Overall pipeline status: ready | degraded")
