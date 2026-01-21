# Copilot Instructions: Compliance RAG System

## Architecture Overview

This is a **SEC-compliant Retrieval-Augmented Generation (RAG) system** for automated hedge fund compliance reporting. The core design principle: **LLMs generate prose, never calculations**.

### Component Flow
```
PostgreSQL (data/vectors) → Control Runner (deterministic) → Evidence Store (audit trail)
                                     ↓
                        RAG Retrieval → Safety Layer → Narrative Generator (LLM) → Document Builder (PDF)
```

### Key Directories
- `src/control_runner/` - Deterministic SQL-based compliance checks
- `src/evidence_store/` - Immutable audit trail with hash chains
- `src/rag/` - pgvector embeddings + semantic search + **safety verification**
- `src/narrative/` - LLM text generation with mandatory citations
- `src/agent/` - ReAct-style investigation agent with tool calling
- `src/orchestrator.py` - Main entry point coordinating all components
- `policies/` - Markdown policy documents (chunked and embedded for RAG)

## Critical Constraints

1. **Never put calculations in LLM prompts** - All numeric thresholds are evaluated in `src/control_runner/controls.py` via `ControlDefinition.evaluate_threshold()`
2. **All LLM outputs require citations** - See `GeneratedNarrative.citations` in `src/narrative/generator.py`
3. **Evidence is immutable** - Once recorded in `EvidenceStore`, data cannot be modified
4. **Everything is hashed** - Control SQL queries, prompts, documents all have SHA-256 hashes for audit
5. **No fabricated numbers** - `HallucinationDetector` verifies every number against source data
6. **Human-in-the-loop for high risk** - Outputs with `RiskLevel.HIGH` or `CRITICAL` require review

## Safety Architecture (Billion-Dollar AUM)

The `src/rag/safety.py` module provides defense-in-depth:

```python
from src.rag import SafeRAGPipeline, PRODUCTION_THRESHOLDS

safe_rag = SafeRAGPipeline(retriever, generator, evidence_store, PRODUCTION_THRESHOLDS)
result = safe_rag.generate_safe_narrative(control_result)

if result.requires_human_review:
    escalate_to_compliance_officer(result)
elif not result.success:
    log_rejection(result.rejection_reason)
```

### Safety Components
| Component | Purpose | Location |
|-----------|---------|----------|
| `HallucinationDetector` | Verify facts against retrieved context | `src/rag/safety.py` |
| `OutputValidator` | Check for prohibited phrases, missing citations | `src/rag/safety.py` |
| `CircuitBreaker` | Fail fast on cascading failures | `src/rag/safety.py` |
| `GracefulDegradation` | Fallback to templates when RAG fails | `src/rag/safety.py` |
| `ProductionMonitor` | Real-time latency, error rates, drift detection | `src/rag/evaluation.py` |
| `RegressionTester` | Golden dataset testing before deployment | `src/rag/evaluation.py` |

### Safety Thresholds
```python
PRODUCTION_THRESHOLDS = SafetyThresholds(
    min_retrieval_confidence=0.60,    # Reject if below
    human_review_confidence=0.80,     # Flag for review if below
    max_unverified_fact_ratio=0.05,   # Only 5% unverified facts allowed
    allow_calculated_numbers=False,   # Strict: no LLM math
)
```

## Development Setup

```powershell
# Start infrastructure (PostgreSQL + pgvector on port 5432, Redis)
docker-compose up -d
# Credentials: compliance_user / compliance_dev_password_123

# Embed policy documents (required before RAG works)
python -m src.rag.embedder

# Run full pipeline with sample data
$env:LLM_PROVIDER = "lmstudio"  # or: anthropic, openai, ollama
python run_database_pipeline.py --setup-sample
```

## LLM Configuration

Configure via environment variables (see `src/integration/llm_config.py`):
- `LLM_PROVIDER`: `lmstudio`, `anthropic`, `openai`, `ollama`, `vllm`, `mock`
- `LLM_MODEL`: Model identifier (defaults vary by provider)
- `LLM_API_BASE`: For local LLMs (e.g., `http://localhost:1234/v1`)
- `LLM_ANONYMIZE=true`: Scrubs sensitive data before API calls

## Testing

```powershell
pytest -m unit                  # Unit tests (no DB required)
pytest -m integration           # Integration tests (requires docker-compose up)
python prove_system.py          # Prove RAG/Agent are real (not mocked)
```

### Regression Testing (Before Deployment)
```python
from src.rag import RegressionTester, get_default_golden_dataset

tester = RegressionTester(retriever, golden_dataset=get_default_golden_dataset())
result = tester.run_full_suite()

if not result.passed:
    raise Exception(f"Deployment blocked: {result.regressions}")
```

### Evaluation Framework
Ground-truth tests in `tests/eval/` ensure retrieval quality:
```powershell
# Run evaluation suite
python -m tests.eval.run_evaluation

# Run pytest tests
pytest tests/eval/ -v
```

The evaluation dataset (`tests/eval/__init__.py`) contains 18 compliance Q&A pairs:
- **Concentration limits** - Single security 5%, sector 25%
- **Liquidity requirements** - 15% liquid assets minimum
- **Exposure limits** - 200% gross, ±100% net
- **Exception handling** - Escalation and approval workflows
- **Regulatory** - SEC Form ADV, PF requirements

## Key Patterns

### Adding a New Control
Define in `src/control_runner/controls.py` using `ControlDefinition`:
```python
ControlDefinition(
    control_code="CONC_SECTOR_001",
    control_name="Sector Concentration",
    category=ControlCategory.CONCENTRATION,
    computation_sql="SELECT SUM(market_value)/nav AS calculated_value FROM ...",
    threshold_value=0.30,
    threshold_operator=ThresholdOperator.GTE,  # Fail if >= 30%
)
```

### Database Adapter Pattern
Client integrations map their schema in `run_database_pipeline.py`:
```python
TABLE_MAPPINGS = {"positions_table": "fund_positions", ...}
COLUMN_MAPPINGS = {"security_id": "client_sec_id", ...}
```

### Snowflake Integration
For institutional hedge funds using Snowflake as their data warehouse:

```python
from src.integration import SnowflakeAdapter, SnowflakeConfig, SnowflakeViewConfig

# Option 1: Load from environment variables
config = SnowflakeConfig.from_env()

# Option 2: Explicit configuration
config = SnowflakeConfig(
    account="xy12345.us-east-1",
    user="compliance_svc",
    password="...",  # Or use private_key_path for key-pair auth
    warehouse="COMPLIANCE_WH",
    database="HEDGE_FUND_DATA",
    schema="COMPLIANCE",
    role="COMPLIANCE_READER",
)

# Customize view/column mappings for client's schema
view_config = SnowflakeViewConfig(
    positions_view="V_POSITIONS_CURRENT",
    controls_view="V_CONTROL_RESULTS",
    nav_view="V_NAV_DAILY",
    position_columns={"market_value": "MV_USD", ...},
)

adapter = SnowflakeAdapter(config, view_config)
snapshot = adapter.get_snapshot(date.today())
```

**Environment Variables:**
- `SNOWFLAKE_ACCOUNT` - Account identifier (e.g., xy12345.us-east-1)
- `SNOWFLAKE_USER` - Service account username
- `SNOWFLAKE_PASSWORD` - Password (or use key-pair auth)
- `SNOWFLAKE_WAREHOUSE`, `SNOWFLAKE_DATABASE`, `SNOWFLAKE_SCHEMA`
- `SNOWFLAKE_PRIVATE_KEY_PATH` - For production key-pair authentication

**Requires:** `pip install snowflake-connector-python`

### RAG Retrieval (SOTA Features)
Uses hybrid search in `src/rag/retriever.py`:
- **BM25 + Dense vectors** - Hybrid search with configurable weights
- **Cross-encoder reranking** - `ms-marco-MiniLM` for precision
- **Query rewriting** - LLM-powered query expansion
- **Multi-hop retrieval** - policy → exception → precedent chains
- **Confidence calibration** - Multi-factor confidence scoring

## Observability & Tracing

The `src/observability/` module provides comprehensive instrumentation:

```python
from src.observability import Tracer, SpanKind, trace_operation, rag_logger

tracer = Tracer(service_name="compliance-rag")

with tracer.start_trace("rag_pipeline") as trace:
    with tracer.start_span("retrieval", SpanKind.RETRIEVAL) as span:
        result = retriever.retrieve(query)
        span.set_attribute("chunk_count", len(result.chunks))
```

### Observability Components
| Component | Purpose | Location |
|-----------|---------|----------|
| `Tracer` | Span-based tracing (OpenTelemetry-compatible) | `src/observability/__init__.py` |
| `MetricsCollector` | Prometheus-style metrics (latency P50/P95/P99) | `src/observability/__init__.py` |
| `RAGLogger` | Structured logging for retrieval, LLM, validation | `src/observability/__init__.py` |
| `QueryCache` | Multi-tier caching (embeddings, results, LLM) | `src/observability/caching.py` |
| `BatchEmbedder` | Efficient batch embedding with caching | `src/observability/caching.py` |
| `ContextWindowOptimizer` | Dynamic chunk selection for context limits | `src/observability/caching.py` |

### Caching Configuration
```python
from src.observability import QueryCache, CacheConfig

cache = QueryCache(CacheConfig(
    embedding_cache_size=10000,      # Cache up to 10K embeddings
    result_cache_ttl_seconds=3600,   # Results expire after 1 hour
    semantic_cache_enabled=False,    # Enable for near-duplicate matching
))

# After data refresh, invalidate result cache
cache.invalidate_results()
```

### Context Window Optimization
```python
from src.observability import ContextWindowOptimizer, ContextWindowConfig

optimizer = ContextWindowOptimizer(ContextWindowConfig(
    max_tokens=4096,      # LLM context limit
    reserved_tokens=1024, # For prompt + response
    min_chunks=2,         # Always include at least 2
    max_chunks=10,        # Never more than 10
))

optimized_chunks = optimizer.optimize_chunks(chunks, query)

## Daily Workflow (Airflow DAG)

The `dags/daily_compliance_dag.py` defines the **production execution order** (runs 6 AM ET):
```
sync_from_snowflake → run_compliance_controls → generate_narratives → build_workpaper
        ↓                      ↓                       ↓                    ↓
   Pull positions         Process results         RAG + LLM           Create PDF
   from client OMS        (trust client calcs)    commentary          with citations
```

Each task passes data via XCom (`snapshot_id`, `position_count`, etc.). For local DAG testing, use `docker-compose.full.yml` which includes Airflow.

### Agent Investigation Pattern
The `src/agent/investigator.py` uses ReAct-style tool calling:
```python
# Tools are defined with: name, description, parameters, function
# Agent steps through: Thought → ToolCall → Observation → repeat
# Investigation produces: findings, root_cause, recommendations, evidence
```

## Data Store Architecture

This system uses a **polyglot persistence** architecture optimized for institutional hedge fund compliance:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           DATA STORE ARCHITECTURE                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   CLIENT SIDE                    OUR SYSTEM                                     │
│   ════════════                   ══════════                                     │
│                                                                                  │
│   ┌──────────────┐    read-only    ┌──────────────┐   ETL    ┌──────────────┐  │
│   │  Snowflake   │ ───────────────→│  PostgreSQL  │ ───────→│  ClickHouse  │  │
│   │  (client DW) │                 │  + pgvector  │         │  (analytics) │  │
│   │              │                 │              │         │              │  │
│   │ • Positions  │                 │ • Today's    │         │ • 10 years   │  │
│   │ • NAV        │                 │   data       │         │   history    │  │
│   │ • Trades     │                 │ • Vectors    │         │ • Trends     │  │
│   │ • Holdings   │                 │ • Audit log  │         │ • Form PF    │  │
│   └──────────────┘                 └──────────────┘         └──────────────┘  │
│                                           │                                     │
│                                           ↓                                     │
│                                    ┌──────────────┐                            │
│                                    │    Redis     │                            │
│                                    │   (cache)    │                            │
│                                    │              │                            │
│                                    │ • Embeddings │                            │
│                                    │ • Results    │                            │
│                                    │ • Alerts     │                            │
│                                    └──────────────┘                            │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Component Roles

| Store | Purpose | Retention | Query Pattern |
|-------|---------|-----------|---------------|
| **Snowflake** | Client's data warehouse (read-only) | Theirs | Daily sync |
| **PostgreSQL** | Operational data + vectors + audit | 90 days | OLTP + vector search |
| **ClickHouse** | Historical analytics + Form PF | 10 years | OLAP aggregations |
| **Redis** | Ephemeral cache + pub/sub | TTL-based | Sub-ms lookups |

### ClickHouse Analytics Integration

For time-series analytics and regulatory reporting:

```python
from src.integration import get_clickhouse_analytics, ClickHouseConfig

# Option 1: Auto-configure from environment
analytics = get_clickhouse_analytics()

# Option 2: Explicit configuration
config = ClickHouseConfig(
    host="clickhouse.internal",
    port=9000,
    database="compliance_analytics",
    user="compliance_svc",
    password="...",
)
analytics = ClickHouseAnalytics(config)

# Historical trend analysis
trend = analytics.get_control_trend(
    control_id="CONC_SECTOR_001",
    start_date=date(2021, 1, 1),
)
print(f"5-year avg: {trend.statistics.avg:.2%}")
print(f"Breaches: {trend.statistics.breach_count}")

# Breach statistics by month
stats = analytics.get_breach_statistics(group_by="month")
for s in stats:
    print(f"{s.period}: {s.breach_count} breaches ({s.breach_rate:.1%})")

# SEC Form PF quarterly data
pf_data = analytics.get_form_pf_data(
    reporting_period_start=date(2025, 10, 1),
    reporting_period_end=date(2025, 12, 31),
)
```

**Environment Variables:**
- `CLICKHOUSE_HOST` - Hostname (default: localhost)
- `CLICKHOUSE_PORT` - Native port (default: 9000)
- `CLICKHOUSE_HTTP_PORT` - HTTP port (default: 8123)
- `CLICKHOUSE_DATABASE` - Database name
- `CLICKHOUSE_USER`, `CLICKHOUSE_PASSWORD` - Credentials
- `CLICKHOUSE_DRIVER` - `native` or `http` (default: native)

**Requires:** `pip install clickhouse-connect` (or `clickhouse-driver` for native)

### Full Stack Docker Compose

Start all data stores with `docker-compose.full.yml`:
```powershell
docker-compose -f docker-compose.full.yml up -d

# Services started:
# - postgres (5432) - PostgreSQL + pgvector
# - redis (6379) - Cache + pub/sub
# - clickhouse (8123/9000) - Analytics engine
# - airflow (8080) - DAG orchestration
```

## Production vs Development

| Environment | Data Source | Vector Store | Analytics | Port |
|-------------|-------------|--------------|-----------|------|
| **Production** | Snowflake | PostgreSQL + pgvector | ClickHouse | varies |
| **Development** | PostgreSQL | Same PostgreSQL | Mock/ClickHouse | 5432 |

Production requires environment variables:
- **Snowflake**: `SNOWFLAKE_ACCOUNT`, `SNOWFLAKE_USER`, `SNOWFLAKE_PASSWORD`
- **ClickHouse**: `CLICKHOUSE_HOST`, `CLICKHOUSE_USER`, `CLICKHOUSE_PASSWORD`

When not set, the system falls back to `MockAdapter` / `MockClickHouseAnalytics`.

## Policy Document Format

Policies in `policies/` must include metadata headers for proper RAG chunking:
```markdown
# Policy Title
## Document ID: POL-XXX-001
## Effective Date: January 1, 2024
---
## 1. Section Name
**Key Term: Definition or limit value**
```

The embedder maps documents to control types via `DOCUMENT_CONTROL_MAPPING` in `src/rag/embedder.py`.

## File Conventions

- **Entry points**: `run_*.py`, `prove_system.py`, `quick_start.py`
- **Config**: `src/config/settings.py` (frozen dataclasses)
- **Type checking**: `pyrightconfig.json` with `typeCheckingMode: "basic"`
- **Imports**: Scripts add `sys.path.insert(0, str(Path(__file__).parent / 'src'))` at top
- **Output**: Generated files go to `output/` with pattern `{type}_{date}.{ext}`

## Verification Checklist

```powershell
# Basic stack
docker-compose ps                              # postgres, redis healthy?
python prove_system.py                         # embeddings + vector search working?
python run_database_pipeline.py --setup-sample # full pipeline produces PDF?

# Integration demos (mock mode, no external dependencies)
python run_snowflake_demo.py --mock            # Snowflake adapter works?
python run_clickhouse_demo.py --mock           # ClickHouse analytics works?

# Full stack (requires docker-compose.full.yml)
docker-compose -f docker-compose.full.yml up -d
python run_clickhouse_demo.py                  # Real ClickHouse queries?
```

### Troubleshooting
- **Empty PDF**: Check `LLM_PROVIDER` is set correctly
- **No embeddings**: Run `python -m src.rag.embedder`
- **Missing citations**: Verify `policy_chunks` table has data
- **"No data found"**: Use `--setup-sample` flag
- **ClickHouse connection failed**: Check `CLICKHOUSE_HOST` and ports 8123/9000

- Entry points: `run_*.py`, `prove_system.py`, `quick_start.py`
- Config: `src/config/settings.py` (frozen dataclasses)
- Type checking: `pyrightconfig.json` with `typeCheckingMode: "basic"`
- Imports: Add `src/` to path (`sys.path.insert(0, ...)`) at script top
