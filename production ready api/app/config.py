"""
Centralized configuration.
Loads application settings from environment variables,
validates them with Pydantic, and provides one shared
configuration object for the entire application.instead of each time os.getenv.
"""

from pydantic_settings import BaseSettings   # Base class for loading and validating environment variables.
from functools import lru_cache     # Caches the Settings object so it is created only once.

class Settings(BaseSettings):
    """Application configuration loaded from .env.prod."""

    # LLM Configuration
    groq_api_key: str
    gemini_api_key: str
    primary_model: str = "llama-3.3-70b-versatile"
    fallback_model: str = "llama-3.3-70b-versatile" #standby

    # LangSmith Configuration
    langsmith_api_key: str
    langsmith_tracing_v2: bool = True
    langsmith_project: str = "production-ready-api"

    # Application Configuration
    app_env: str = "development"
    log_level: str = "INFO"
    rate_limit: str = "20/minute"
    cache_ttl_seconds: int = 300
    max_retries: int = 3
    # Pydantic Configuration

    model_config = {
        "env_file": ".env.prod",   # Load environment variables from .env.prod.
        "extra": "ignore"          # Ignore unknown variables found in the .env file.
    }

    @property
    # Allows this method to be accessed like a variable (no parentheses needed).
    def is_production(self) -> bool:
        # Returns True if the application is running in production.
        return self.app_env == "production"


@lru_cache()
# Cache the Settings object so it is created only once.
def get_settings() -> Settings:
    """Return the shared application settings."""
    return Settings()   # Load, validate, and return the configuration.

"""lru = last recently used cache. if we call get_settings() multiple times, 
it will return the same instance(cached).

"""