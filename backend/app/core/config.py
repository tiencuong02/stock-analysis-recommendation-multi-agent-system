from pydantic_settings import BaseSettings
from typing import Optional, List

class Settings(BaseSettings):
    PROJECT_NAME: str = "Multi-Agent Stock Analysis Platform"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    # CORS
    BACKEND_CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://127.0.0.1:3000", "http://0.0.0.0:3000"]

    # MongoDB
    MONGO_URI: str = "mongodb://localhost:27017/stockdb"

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"

    # Kafka
    KAFKA_BROKER_URL: str = "localhost:9092"
    KAFKA_TOPIC: str = "stock_analysis_tasks"

    # Backend Security
    SECRET_KEY: str = "YOUR_SUPER_SECRET_KEY_DONT_USE_THIS_IN_PROD"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7

    # Default User passwords for DB Seeding — MUST override via .env in production
    ADMIN_PASSWORD: str = ""
    USER_PASSWORD: str = ""

    # Market Data APIs (on-premise exception: external data feeds)
    ALPHA_VANTAGE_API_KEY: Optional[str] = None

    # Qdrant — on-premise vector database
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_COLLECTION_NAME: str = "stock_reports"

    # Ollama — on-premise LLM server
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen2.5:7b-instruct-q4_K_M"

    # Embeddings (local HuggingFace model — no API key needed)
    EMBEDDING_MODEL_NAME: str = "paraphrase-multilingual-MiniLM-L12-v2"
    EMBEDDING_DIMENSION: int = 384

    class Config:
        case_sensitive = True
        env_file = ".env"
        extra = "ignore"

settings = Settings()
