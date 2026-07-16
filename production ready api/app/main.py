from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from langsmith import traceable
from app.agent import ProductionAgent
from app.cache import ResponseCache, redis_client
from app.monitoring import MetricsCollector, RequestTimer, get_logger
from app.config import get_settings
from app.models import (
    ChatRequest,
    ChatResponse,
    HealthCheckResponse,
    MetricsResponse,
    ErrorResponse,
)
from app.security import SecurityLayer
from dotenv import load_dotenv
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler

load_dotenv(".env.prod")  # Load environment variables from .env.prod
logger = get_logger()
settings = get_settings()

secutity = None
metrics = None
cache = None
agent = None


# ====lifespan (startup and shutdown events)====
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    initialize all components on startup and cleanup on shutdown.
    this is modern fastapi pattern (replaces @app.on_event("startup") and @app.on_event("shutdown"))
    """

    logger.info(
        "Starting up the Production API",
        extra={
            "extra_metadata": {
                "environment": settings.app_env,
                "primary_model": settings.primary_model,
                "tracing_enabled": settings.langsmith_tracing,
            }
        },
    )

    # initialize components
    secutity = SecurityLayer()
    metrics = MetricsCollector()
    cache = ResponseCache(redis_client, ttl=settings.cache_ttl_seconds)
    agent = ProductionAgent()
    logger.info("components initialized successfully ready to serve requests")

    yield  # this is where the application runs

    # shutdown cleanup
    logger.info(
        "Shutting down the Production API",
        extra={"extra_metadata": {"metrics": metrics.get_metrics()}},
    )


# Rate limiter setup
limiter = Limiter(key_func=get_remote_address)

# FastAPI app initialization
app = FastAPI(
    lifespan=lifespan,
    title="Production RAG API",
    version="1.0.0",
    description="A production-ready RAG API with security, monitoring, and caching features.",
)
app.state.limiter = limiter

# ===========================
# ENDPOINTS
# ===========================


@app.post("/chat", response_model=ChatResponse)
@limiter.limit(settings.rate_limit)
@traceable(project_name="chat_endpoint")
async def chat_endpoint(request: Request, chat_request: ChatRequest):
    """
    Main flow:
    1-security checks on input(prompt injection, PII)
    2-cache lookup
    3-langgraph agent invoke(if cahe miss)
    4-Output validation (PII, harmful content)
    5-Cache store
    6-return response
    """
    with RequestTimer() as timer:
        secutity_notes = []

    # 1: Security checks on input
    is_allowed, cleaned_input, notes = secutity.check_input(chat_request.message)
    secutity_notes.extend(notes)
    if not is_allowed:
        logger.warning(
            "Input rejected due to security concerns",
            extra={
                "extra_metadata": {
                    "reason": notes,
                    "thread_id": getattr(chat_request, "thread_id", None),
                }
            },
        )
        raise HTTPException(
            status_code=400, detail="Input rejected due to security concerns"
        )
    # 2: Cache lookup
    try:
        cached_response = cache.get(cleaned_input)
    except Exception as cache_error:
        logger.warning(
            "Cache lookup failed, continuing without cache",
            extra={
                "extra_metadata": {
                    "error": str(cache_error),
                    "thread_id": getattr(chat_request, "thread_id", None),
                }
            },
        )
        cached_response = None
    if cached_response:
        metrics.record_request(
            latency_ms=timer.elapsed_ms,
            error=False,
            cache_hit=True,
            input_tokens=0,
            output_tokens=0,
        )
        logger.info(
            "Cache hit for input",
            extra={
                "extra_metadata": {
                    "thread_id": getattr(chat_request, "thread_id", None)
                }
            },
        )
        return ChatResponse(
            response=cached_response,
            model_used="cache",
            cache_hit=True,
            processing_time_ms=0,
            thread_id=getattr(chat_request, "thread_id", None),
        )
    # 3: LangGraph agent invoke (if cache miss)
    try:
        agent_response = agent.invoke(cleaned_input)
    except Exception as e:
        logger.error(
            "Error invoking agent",
            extra={
                "extra_metadata": {
                    "error": str(e),
                    "thread_id": getattr(chat_request, "thread_id", None),
                }
            },
        )
        metrics.record_request(
            latency_ms=timer.elapsed_ms,
            error=True,
            cache_hit=False,
            input_tokens=0,
            output_tokens=0,
        )
        raise HTTPException(status_code=500, detail="Error processing request")

    # 4: Output validation (PII, harmful content)
    cleaned_output, output_notes = secutity.validate_output(agent_response["response"])
    secutity_notes.extend(output_notes)
    # 5: Cache store
    try:
        cache.set(cleaned_input, cleaned_output)
    except Exception as cache_error:
        logger.warning(
            "Cache store failed, returning response without caching",
            extra={
                "extra_metadata": {
                    "error": str(cache_error),
                    "thread_id": getattr(chat_request, "thread_id", None),
                }
            },
        )

    # 6:log and record metrics
    input_tokens = int(len(cleaned_input.split()) * 1.3)  # Estimate token count
    output_tokens = int(len(cleaned_output.split()) * 1.3)  # Estimate token count
    metrics.record_request(
        latency_ms=timer.elapsed_ms,
        error=False,
        cache_hit=False,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
    return ChatResponse(
        response=cleaned_output,
        model_used=settings.primary_model,
        processing_time_ms=round(timer.elapsed_ms, 2),
        cached=False,
        thread_id=getattr(chat_request, "thread_id", None),
    )


@app.get("/health", response_model=HealthCheckResponse)
def health_check():
    """
    Health check endpoint to verify the API is running.
    """
    settings = get_settings()
    checks = {
        "agent": agent is not None,
        "cache": cache is not None,
        "security": secutity is not None,
    }
    all_healthy = all(checks.values())
    return HealthCheckResponse(
        status="Healthy" if all_healthy else "Error",
        environment=settings.app_env,
        checks=checks,
    )


# @app.get("/metrics", response_model=MetricsResponse)
# def get_metrics():
#     """
#     Returns collected metrics for monitoring purposes.
#     """
#     return MetricsResponse(metrics=get_metrics())


# @app.get("/cache_stats")
# def get_cache_stats():
#     """
#     Returns cache statistics for monitoring purposes.
#     """
#     return cache.get_stats()
