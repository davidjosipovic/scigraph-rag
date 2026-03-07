"""
FastAPI route definitions.

Endpoints:
  POST /ask     — Ask a question about scientific papers
  GET  /health  — Check system component health
"""

from fastapi import APIRouter, HTTPException
from loguru import logger

from backend.api.schemas import (
    AskRequest,
    AskResponse,
    ExtractedEntitiesResponse,
    HealthResponse,
    Source,
)
from backend.rag.pipeline import RAGPipeline

router = APIRouter()

# Pipeline instance (initialized once, reused across requests)
_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    """Lazy-initialize and return the RAG pipeline singleton."""
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline


@router.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest) -> AskResponse:
    """
    Ask a question about scientific papers.

    The system will:
    1. Classify the question type (6 types)
    2. Extract scientific entities (methods, datasets, tasks, fields, metrics)
    3. Execute multi-strategy SPARQL retrieval in parallel (asyncio.gather)
    4. Rank, hard-filter, and truncate results to top 8 papers
    5. Build structured per-paper context with relevance scores
    6. Generate an answer using a local LLM (Ollama)
    7. Return the answer with cited sources and full pipeline transparency
    """
    logger.info(f"Received question: {request.question}")

    try:
        pipeline = get_pipeline()
        result = await pipeline.ask(request.question)

        sources = [Source(**s) for s in result.get("sources", [])]
        entities = ExtractedEntitiesResponse(**result.get("entities", {}))

        return AskResponse(
            question=result["question"],
            query_type=result["query_type"],
            entities=entities,
            sparql_queries=result.get("sparql_queries", []),
            strategies_used=result.get("strategies_used", []),
            answer=result["answer"],
            sources=sources,
            kg_results_count=result.get("kg_results_count", 0),
        )

    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while processing your question: {str(e)}",
        )


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """
    Check the health of all system components.

    Returns the status of the LLM, SPARQL endpoint, and overall pipeline.
    """
    pipeline = get_pipeline()
    status = pipeline.health_check()
    return HealthResponse(**status)
