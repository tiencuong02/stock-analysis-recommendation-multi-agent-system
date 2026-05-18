from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncio
import sys
import logging

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from asgi_correlation_id import CorrelationIdMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from app.core.config import settings
from app.core.logging_config import setup_logging
from app.api.router import router as api_router
from app.db.mongodb import connect_to_mongo, close_mongo_connection, get_db
from app.db.redis import connect_to_redis, close_redis_connection
from app.api.kafka_producer import KafkaProducerService

# Setup centralized logging with Correlation ID
setup_logging()
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up services...")
    
    # Mongo
    try:
        await asyncio.wait_for(connect_to_mongo(), timeout=10.0)
    except Exception as e:
        logger.error(f"MongoDB startup failed: {e}")
        
    # Redis
    try:
        await asyncio.wait_for(connect_to_redis(), timeout=10.0)
    except Exception as e:
        logger.error(f"Redis startup failed: {e}")
        
    # Kafka
    try:
        await asyncio.wait_for(KafkaProducerService.get_producer(), timeout=10.0)
    except Exception as e:
        logger.warning(f"Kafka connection skipped during startup: {e}")
    
    # Initialize Data
    try:
        from app.repositories.user_repository import UserRepository
        from app.repositories.quote_repository import QuoteRepository
        from app.services.quote_service import QuoteService
        db = get_db()
        if db is not None:
            user_repo = UserRepository(db)
            await user_repo.init_default_users()
            
            quote_repo = QuoteRepository(db)
            quote_service = QuoteService(quote_repo)
            await quote_service.seed_quotes()
            logger.info("Default users and quotes initialized.")
    except Exception as e:
        logger.error(f"Data initialization failed: {e}")

    # Rate Limiter (Redis-backed nếu có, InMemory fallback)
    try:
        from app.db.redis import get_redis
        from app.services.redis_rate_limiter import make_rate_limiter
        app.state.rate_limiter = make_rate_limiter(get_redis())
    except Exception as e:
        logger.warning(f"Rate limiter init failed: {e}")
        from app.services.redis_rate_limiter import InMemoryRateLimiter
        app.state.rate_limiter = InMemoryRateLimiter()

    # RAG Services (singleton - load embedding model 1 lần duy nhất)
    try:
        from app.services.rag.vector_store import VectorStoreService
        from app.services.rag.rag_pipeline import RAGPipelineService
        from app.services.ticker_context_cache import TickerContextCache
        logger.info("Initializing RAG services (singleton)...")
        app.state.vector_store = VectorStoreService()
        app.state.rag_pipeline = RAGPipelineService(app.state.vector_store)
        app.state.rag_pipeline._prewarm()

        # Ticker context cache — inject Redis client (None-safe nếu Redis không có)
        from app.db.redis import get_redis
        app.state.ticker_cache = TickerContextCache(redis_client=get_redis())
        app.state.rag_pipeline.set_ticker_cache(app.state.ticker_cache)

        vs = app.state.vector_store
        rp = app.state.rag_pipeline
        logger.info(
            f"RAG init status — "
            f"embeddings={'OK' if vs.embeddings else 'FAIL'} | "
            f"qdrant_client={'OK' if vs._client else 'FAIL'} | "
            f"qdrant_store={'OK' if vs._store else 'FAIL'} | "
            f"cross_encoder={'OK' if vs._cross_encoder else 'WARN(disabled)'} | "
            f"llm={'OK' if rp.llm else 'FAIL'}"
        )
        if not rp.llm:
            logger.error(
                "RAG pipeline LLM is None — chatbot will return 'chưa sẵn sàng'. "
                "Check Ollama is running and model is pulled: "
                f"ollama pull {settings.OLLAMA_MODEL}"
            )
        if not vs._store:
            logger.error(
                f"Qdrant store is None — vector search disabled. "
                f"Ensure Qdrant is running at {settings.QDRANT_URL}"
            )
        logger.info("RAG services initialized successfully.")
    except Exception as e:
        import traceback as _tb
        logger.error(f"RAG services initialization failed: {e}\n{_tb.format_exc()}")
        app.state.vector_store = None
        app.state.rag_pipeline = None

    logger.info("All services startup complete.")
    yield
    
    # Shutdown
    logger.info("Shutting down services...")
    await close_mongo_connection()
    await close_redis_connection()
    await KafkaProducerService.stop_producer()

from app.core.exceptions.app_exceptions import (
    BaseAppException, 
    app_exception_handler, 
    generic_exception_handler
)

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Production-grade Multi-Agent Stock Analysis API",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    redirect_slashes=True
)

# Exception Handlers
app.add_exception_handler(BaseAppException, app_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)

# 1. Correlation ID Middleware (Dùng để truy vết request qua logs)
app.add_middleware(CorrelationIdMiddleware)

# 2. Prometheus Metrics (Tự động đo lường hiệu năng API)
Instrumentator().instrument(app).expose(app)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_origin_regex=r"https://.*\.ngrok-free\.dev",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get("/")
async def root():
    return {
        "message": f"Welcome to {settings.PROJECT_NAME}",
        "version": settings.VERSION,
        "docs": "/docs"
    }

import datetime

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
    }

@app.get("/health/rag")
async def rag_health_check():
    """Chẩn đoán trạng thái từng thành phần RAG — dùng khi chatbot báo 'chưa sẵn sàng'."""
    vs = getattr(app.state, "vector_store", None)
    rp = getattr(app.state, "rag_pipeline", None)
    return {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "rag_pipeline_initialized": rp is not None,
        "llm_ready":                rp.llm is not None if rp else False,
        "embeddings_ready":         vs.embeddings is not None if vs else False,
        "qdrant_client_ready":      vs._client is not None if vs else False,
        "qdrant_store_ready":       vs._store is not None if vs else False,
        "cross_encoder_ready":      vs._cross_encoder is not None if vs else False,
        "chatbot_will_work":        (rp is not None and rp.llm is not None),
        "vector_search_will_work":  (vs is not None and vs._store is not None),
    }
