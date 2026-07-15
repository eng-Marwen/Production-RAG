from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from langsmith import traceable
from agent import ProductionAgent
from cache import ResponseCache
from monitoring import MetricsCollector, RequestTimer, get_logger
from config import get_settings
from models import (
    ChatRequest,
    ChatResponse,
    HealthCheckResponse,
    MetricsResponse,
    ErrorResponse,
)
from security import SecurityLayer
from dotenv import load_dotenv
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi import _rate_limit_exceeded_handler

load_dotenv(".env.prod")  # Load environment variables from .env.prod
logger=get_logger()
settings=get_settings()

# ====lifespan (startup and shutdown events)====
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    initialize all components on startup and cleanup on shutdown.
    this is modern fastapi pattern (replaces @app.on_event("startup") and @app.on_event("shutdown"))
    """
    global secutity, metrics, cache, agent

    logger.info("Starting up the Production API",extra={"extra_metadata":{
        "environment": settings.app_env,
        "primary_model": settings.primary_model,
        "tracing_enabled": settings.langsmith_tracing
    }})

    #initialize components
    secutity=SecurityLayer()
    metrics=MetricsCollector()
    cache=ResponseCache(ttl_seconds=settings.cache_ttl_seconds)
    agent=ProductionAgent()
    logger.info("components initialized successfully ready to serve requests")

    yield  # this is where the application runs

    #shutdown cleanup
    logger.info("Shutting down the Production API", extra={"extra_metadata":{"metrics":metrics.get_metrics()}})

#Rate limiter setup
limiter=Limiter(key_func=get_remote_address)

#FastAPI app initialization
app=FastAPI(
    lifespan=lifespan,
    title="Production RAG API", 
    version="1.0.0", 
    description="A production-ready RAG API with security, monitoring, and caching features."
)
app.state.limiter=limiter

#===========================
# ENDPOINTS
#===========================

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
        secutity_notes=[]

    #1: Security checks on input
    is_allowed, cleaned_input, notes = secutity.check_input(chat_request)
    secutity_notes.extend(notes)
    if not is_allowed:
        logger.warning(
            "Input rejected due to security concerns",
            extra={"extra_metadata":{"reason":notes, "thread_id": getattr(chat_request, 'thread_id', None)}}
        )
        raise HTTPException(status_code=400, detail="Input rejected due to security concerns")
    #2: Cache lookup
    cached_response=cache.get(cleaned_input)
    if cached_response:
        metrics.record_request(
            latency_ms=0,
            error=False,
            cache_hit=True
        )
        logger.info(
            "Cache hit for input",
            extra={"extra_metadata":{"thread_id": getattr(chat_request, 'thread_id', None)}}
        )
        return ChatResponse(
            response=cached_response, 
            model_used='cache', 
            cache_hit=True, 
            processing_time_ms=0, 
            security_notes=secutity_notes
        )
