"""
FastAPI application entry point.

Run with:
    uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from backend.config import settings
from backend.api.routes import router


# Configure loguru
logger.add(
    "logs/kg_rag_{time}.log",
    rotation="10 MB",
    retention="7 days",
    level=settings.log_level,
)

# Create the FastAPI application
app = FastAPI(
    title="KG-RAG: Knowledge Graph RAG for Scientific Papers",
    description=(
        "A GraphRAG pipeline over the Open Research Knowledge Graph (ORKG) "
        "for answering questions, verifying claims, and searching topics "
        "in scientific literature."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware — origins controlled via CORS_ORIGINS env var.
# Default allows common local dev ports (3000, 8080, 5173).
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

# Register routes
app.include_router(router)


@app.get("/")
def root():
    """Root endpoint — basic system info."""
    return {
        "system": "KG-RAG",
        "version": "0.1.0",
        "description": "GraphRAG pipeline over ORKG for scientific papers",
        "endpoints": {
            "ask": "POST /ask — Ask a question about scientific papers",
            "health": "GET /health — Check system component health",
            "docs": "GET /docs — Interactive API documentation",
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
    )
