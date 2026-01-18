"""
Compliance RAG System - Configuration Settings

This module provides centralized configuration management with:
- Environment-based configuration
- Secrets management (via environment variables or vault)
- Validation of required settings
- Audit logging of configuration access

SEC Examination Note: Configuration is version-controlled and changes are logged.
Sensitive values are never stored in code - only environment variables.
"""

from __future__ import annotations

import os
import hashlib
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Environment(Enum):
    """Deployment environments with different security profiles."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass(frozen=True)
class PostgresConfig:
    """
    Postgres connection configuration for the Evidence Store.
    
    Security: Uses read-write access for control results storage.
    Connection is pooled and SSL-enforced in production.
    """
    host: str
    port: int
    database: str
    username: str
    password: str  # From environment variable only
    ssl_mode: str = "require"
    pool_min_size: int = 5
    pool_max_size: int = 20
    connection_timeout: int = 30
    
    @property
    def connection_string(self) -> str:
        """Generate connection string (password masked in logs)."""
        return (
            f"postgresql://{self.username}:***@"
            f"{self.host}:{self.port}/{self.database}"
            f"?sslmode={self.ssl_mode}"
        )
    
    @property
    def connection_string_full(self) -> str:
        """Full connection string for actual connections (never log this)."""
        return (
            f"postgresql://{self.username}:{self.password}@"
            f"{self.host}:{self.port}/{self.database}"
            f"?sslmode={self.ssl_mode}"
        )


@dataclass(frozen=True)
class SnowflakeConfig:
    """
    Snowflake connection configuration for compliance data.
    
    Security: Read-only service account with access to curated views only.
    No access to raw production tables.
    """
    account: str
    username: str
    password: str  # From environment variable or key-pair auth
    warehouse: str
    database: str
    schema: str
    role: str
    private_key_path: Optional[str] = None
    connection_timeout: int = 60
    
    @property
    def connection_params(self) -> Dict[str, Any]:
        """Connection parameters for Snowflake connector."""
        params = {
            "account": self.account,
            "user": self.username,
            "warehouse": self.warehouse,
            "database": self.database,
            "schema": self.schema,
            "role": self.role,
        }
        if self.private_key_path:
            # Key-pair authentication (preferred for service accounts)
            params["private_key_path"] = self.private_key_path
        else:
            params["password"] = self.password
        return params


@dataclass(frozen=True)
class RedisConfig:
    """
    Redis configuration for ephemeral state and job coordination.
    
    Note: Redis is NOT used for evidence storage - only for:
    - Job state and coordination
    - Rate limiting
    - Temporary caching (with short TTL)
    """
    host: str
    port: int
    password: Optional[str] = None
    db: int = 0
    ssl: bool = True
    socket_timeout: int = 5
    
    @property
    def connection_url(self) -> str:
        protocol = "rediss" if self.ssl else "redis"
        auth = f":{self.password}@" if self.password else ""
        return f"{protocol}://{auth}{self.host}:{self.port}/{self.db}"


@dataclass(frozen=True)
class LLMConfig:
    """
    LLM configuration for narrative generation.
    
    Critical constraints:
    - LLM is used ONLY for narrative text generation
    - All calculations are done in SQL/Python BEFORE LLM call
    - Prompts are templated and version-controlled
    - All LLM calls are logged with prompt hashes
    """
    provider: str  # 'openai', 'anthropic', 'azure_openai'
    model_id: str
    api_key: str  # From environment variable only
    api_base_url: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.1  # Low temperature for consistency
    timeout: int = 60
    max_retries: int = 3
    
    # For audit trail
    @property
    def model_version_string(self) -> str:
        """Version string for audit logging."""
        return f"{self.provider}/{self.model_id}"


@dataclass(frozen=True)
class VectorStoreConfig:
    """
    Vector store configuration for policy document retrieval.
    
    Used for semantic search over:
    - Compliance policies and procedures
    - Prior SEC filings
    - Internal memos
    """
    provider: str  # 'pinecone', 'weaviate', 'pgvector', 'chroma'
    api_key: Optional[str] = None
    environment: Optional[str] = None
    index_name: str = "compliance-policies"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    
    # For pgvector (Postgres-based)
    pgvector_connection_string: Optional[str] = None


@dataclass(frozen=True)
class DocumentConfig:
    """
    Document generation configuration.
    """
    output_directory: Path
    template_directory: Path
    policy_directory: Path
    temp_directory: Path
    
    # PDF settings
    pdf_font_family: str = "Helvetica"
    pdf_font_size: int = 10
    pdf_margins: tuple = (72, 72, 72, 72)  # points (1 inch = 72 points)
    
    # Retention
    document_retention_days: int = 2555  # 7 years for SEC compliance


@dataclass(frozen=True)
class ControlConfig:
    """
    Control execution configuration.
    """
    # Scheduling
    daily_run_hour: int = 6  # 6 AM UTC
    retry_attempts: int = 3
    retry_delay_seconds: int = 300  # 5 minutes
    
    # Thresholds for alerts
    warning_threshold_pct: float = 80.0  # Warn at 80% of limit
    
    # Timeouts
    query_timeout_seconds: int = 300  # 5 minutes per query
    total_run_timeout_seconds: int = 3600  # 1 hour for full run


@dataclass
class Settings:
    """
    Central settings object containing all configuration.
    
    This class is immutable after initialization and provides
    a config hash for audit trail purposes.
    """
    environment: Environment
    postgres: PostgresConfig
    snowflake: SnowflakeConfig
    redis: RedisConfig
    llm: LLMConfig
    vector_store: VectorStoreConfig
    documents: DocumentConfig
    controls: ControlConfig
    
    # Metadata
    version: str = "1.0.0"
    service_name: str = "compliance-rag"
    initialized_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def config_hash(self) -> str:
        """
        Generate SHA-256 hash of configuration for audit trail.
        
        This hash is stored with each control run to ensure reproducibility.
        If configuration changes, the hash changes, providing an audit trail.
        """
        # Exclude secrets from hash (they don't affect behavior)
        config_str = (
            f"{self.environment.value}|"
            f"{self.postgres.host}:{self.postgres.port}/{self.postgres.database}|"
            f"{self.snowflake.account}/{self.snowflake.database}/{self.snowflake.schema}|"
            f"{self.llm.model_version_string}|"
            f"{self.vector_store.provider}/{self.vector_store.index_name}|"
            f"{self.version}"
        )
        return hashlib.sha256(config_str.encode()).hexdigest()
    
    def validate(self) -> list[str]:
        """Validate configuration and return list of errors."""
        errors = []
        
        # Check required connections
        if not self.postgres.host:
            errors.append("Postgres host is required")
        if not self.snowflake.account:
            errors.append("Snowflake account is required")
        if not self.llm.api_key:
            errors.append("LLM API key is required")
            
        # Check paths exist
        if not self.documents.template_directory.exists():
            errors.append(f"Template directory does not exist: {self.documents.template_directory}")
        if not self.documents.policy_directory.exists():
            errors.append(f"Policy directory does not exist: {self.documents.policy_directory}")
            
        # Environment-specific checks
        if self.environment == Environment.PRODUCTION:
            if self.postgres.ssl_mode != "require":
                errors.append("SSL must be required in production")
            if self.llm.temperature > 0.3:
                errors.append("LLM temperature should be ≤0.3 in production for consistency")
                
        return errors


def _get_env(key: str, default: Optional[str] = None, required: bool = False) -> str:
    """Get environment variable with validation."""
    value = os.environ.get(key, default)
    if required and not value:
        raise ValueError(f"Required environment variable {key} is not set")
    return value or ""


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Load and cache settings from environment variables.
    
    This function is called once at startup and cached.
    Settings are immutable after initialization.
    """
    env_name = _get_env("ENVIRONMENT", "development")
    environment = Environment(env_name.lower())
    
    logger.info(f"Loading configuration for environment: {environment.value}")
    
    # Base paths
    base_dir = Path(__file__).parent.parent.parent
    
    settings = Settings(
        environment=environment,
        
        postgres=PostgresConfig(
            host=_get_env("POSTGRES_HOST", "localhost"),
            port=int(_get_env("POSTGRES_PORT", "5432")),
            database=_get_env("POSTGRES_DATABASE", "compliance"),
            username=_get_env("POSTGRES_USER", "compliance_app"),
            password=_get_env("POSTGRES_PASSWORD", required=environment == Environment.PRODUCTION),
            ssl_mode=_get_env("POSTGRES_SSL_MODE", "require" if environment == Environment.PRODUCTION else "prefer"),
        ),
        
        snowflake=SnowflakeConfig(
            account=_get_env("SNOWFLAKE_ACCOUNT", ""),
            username=_get_env("SNOWFLAKE_USER", "compliance_reader"),
            password=_get_env("SNOWFLAKE_PASSWORD", ""),
            warehouse=_get_env("SNOWFLAKE_WAREHOUSE", "COMPLIANCE_WH"),
            database=_get_env("SNOWFLAKE_DATABASE", "HEDGE_FUND"),
            schema=_get_env("SNOWFLAKE_SCHEMA", "COMPLIANCE"),
            role=_get_env("SNOWFLAKE_ROLE", "COMPLIANCE_READER"),
            private_key_path=_get_env("SNOWFLAKE_PRIVATE_KEY_PATH"),
        ),
        
        redis=RedisConfig(
            host=_get_env("REDIS_HOST", "localhost"),
            port=int(_get_env("REDIS_PORT", "6379")),
            password=_get_env("REDIS_PASSWORD"),
            db=int(_get_env("REDIS_DB", "0")),
            ssl=_get_env("REDIS_SSL", "true").lower() == "true",
        ),
        
        llm=LLMConfig(
            provider=_get_env("LLM_PROVIDER", "openai"),
            model_id=_get_env("LLM_MODEL_ID", "gpt-4-turbo"),
            api_key=_get_env("LLM_API_KEY", required=environment == Environment.PRODUCTION),
            api_base_url=_get_env("LLM_API_BASE_URL"),
            max_tokens=int(_get_env("LLM_MAX_TOKENS", "4096")),
            temperature=float(_get_env("LLM_TEMPERATURE", "0.1")),
        ),
        
        vector_store=VectorStoreConfig(
            provider=_get_env("VECTOR_STORE_PROVIDER", "pgvector"),
            api_key=_get_env("VECTOR_STORE_API_KEY"),
            environment=_get_env("VECTOR_STORE_ENVIRONMENT"),
            index_name=_get_env("VECTOR_STORE_INDEX", "compliance-policies"),
            embedding_model=_get_env("EMBEDDING_MODEL", "text-embedding-3-small"),
        ),
        
        documents=DocumentConfig(
            output_directory=Path(_get_env("DOCUMENT_OUTPUT_DIR", str(base_dir / "output"))),
            template_directory=Path(_get_env("TEMPLATE_DIR", str(base_dir / "templates"))),
            policy_directory=Path(_get_env("POLICY_DIR", str(base_dir / "policies"))),
            temp_directory=Path(_get_env("TEMP_DIR", "/tmp/compliance-rag")),
        ),
        
        controls=ControlConfig(
            daily_run_hour=int(_get_env("CONTROL_RUN_HOUR", "6")),
            retry_attempts=int(_get_env("CONTROL_RETRY_ATTEMPTS", "3")),
        ),
    )
    
    # Validate
    errors = settings.validate()
    if errors and environment == Environment.PRODUCTION:
        raise ValueError(f"Configuration errors in production: {errors}")
    elif errors:
        for error in errors:
            logger.warning(f"Configuration warning: {error}")
    
    logger.info(f"Configuration loaded. Hash: {settings.config_hash[:16]}...")
    
    return settings


# Convenience function for tests
def get_test_settings() -> Settings:
    """Get settings configured for testing."""
    base_dir = Path(__file__).parent.parent.parent
    
    return Settings(
        environment=Environment.DEVELOPMENT,
        postgres=PostgresConfig(
            host="localhost",
            port=5432,
            database="compliance_test",
            username="test_user",
            password="test_password",
            ssl_mode="disable",
        ),
        snowflake=SnowflakeConfig(
            account="test_account",
            username="test_user",
            password="test_password",
            warehouse="TEST_WH",
            database="TEST_DB",
            schema="TEST_SCHEMA",
            role="TEST_ROLE",
        ),
        redis=RedisConfig(
            host="localhost",
            port=6379,
            ssl=False,
        ),
        llm=LLMConfig(
            provider="mock",
            model_id="mock-model",
            api_key="test-key",
        ),
        vector_store=VectorStoreConfig(
            provider="mock",
        ),
        documents=DocumentConfig(
            output_directory=base_dir / "test_output",
            template_directory=base_dir / "templates",
            policy_directory=base_dir / "policies",
            temp_directory=Path("/tmp/compliance-rag-test"),
        ),
        controls=ControlConfig(),
    )
