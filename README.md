# Compliance RAG System for SEC-Registered Hedge Funds

A production-grade Retrieval-Augmented Generation (RAG) system designed to automate daily compliance reporting and SEC filing workpapers for hedge funds with $1–5B AUM.

## Overview

This system provides automated compliance documentation with three core guarantees:

1. **No Hallucinations** - Deterministic calculations with full audit trails
2. **Traceable** - Every output traces back to source data with evidence hashes
3. **Reproducible** - Same inputs always produce identical outputs

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Compliance RAG System                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐       │
│  │   Control Runner │    │  Evidence Store  │    │ Retrieval Layer  │       │
│  │  (Deterministic) │───▶│  (Audit Trail)   │◀───│   (Hybrid RAG)   │       │
│  │                  │    │                  │    │                  │       │
│  │ • SQL execution  │    │ • Hash chains    │    │ • SQL-first      │       │
│  │ • Threshold eval │    │ • Queryable logs │    │ • Lexical match  │       │
│  │ • Exception flag │    │ • Tamper-proof   │    │ • Vector search  │       │
│  └──────────────────┘    └──────────────────┘    └──────────────────┘       │
│           │                      │                       │                   │
│           ▼                      ▼                       ▼                   │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                      Narrative Generator                              │   │
│  │              (LLM-Assisted with Evidence Binding)                     │   │
│  │                                                                       │   │
│  │  • Template-driven structure     • Mandatory citation validation     │   │
│  │  • Deterministic formatting      • "Insufficient evidence" fallback  │   │
│  │  • No calculations in LLM        • Full prompt logging               │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│                                    ▼                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                       Document Builder                                │   │
│  │                    (PDF with Locked Structure)                        │   │
│  │                                                                       │   │
│  │  • Fixed section layout          • Deterministic table rendering     │   │
│  │  • SHA-256 document hashing      • Full evidence appendix            │   │
│  │  • Version control               • Print-ready formatting            │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Data Infrastructure

### Databases

| Database | Purpose | Access |
|----------|---------|--------|
| **Snowflake** | Daily position/exposure snapshots | Read-only |
| **PostgreSQL** | Workflow & audit trail | Read-write |
| **Redis** | Ephemeral alerts & job state | Read-write |

### Key Data Types

- **Positions**: Daily holdings with quantities and marks
- **Exposures**: Long/short/net by asset class, geography, sector
- **Liquidity Buckets**: T+1 through T+365+ redemption capacity
- **Counterparty Exposure**: Prime broker and OTC counterparty limits
- **PnL**: Daily attribution for NAV reconciliation

## Components

### 1. Control Runner (`src/control_runner/`)

Executes deterministic compliance controls against Snowflake views.

```python
from compliance_rag import ControlRunner, ControlRunContext

runner = ControlRunner(snowflake_conn, postgres_conn)

context = ControlRunContext(
    run_date=date.today(),
    fund_ids=["FUND-001"],
    control_ids=None,  # Run all active controls
)

results = await runner.run_all_controls(context)
```

**Built-in Controls:**
- Position limits (gross/net exposure)
- Concentration limits (single name, sector)
- Liquidity monitoring (bucket thresholds)
- Counterparty exposure (prime broker limits)
- NAV reconciliation (fund admin vs internal)

### 2. Evidence Store (`src/evidence_store/`)

Maintains queryable audit trail for all compliance evidence.

```python
from compliance_rag import EvidenceStore

store = EvidenceStore(postgres_conn)

# Record control result with evidence
await store.record_control_result(
    control_result=result,
    evidence_hash=data_hash,
)

# Retrieve evidence for SEC examination
evidence = await store.get_evidence_for_date(
    run_date=date.today(),
    fund_id="FUND-001",
)
```

### 3. Retrieval Layer (`src/retrieval/`)

Hybrid RAG with 3-tier retrieval strategy:

```python
from compliance_rag import HybridRetriever, RetrievalContext

retriever = HybridRetriever(
    snowflake_conn=snowflake_conn,
    postgres_conn=postgres_conn,
    vector_store=pinecone_client,
)

context = RetrievalContext(
    query="position limit breach for FUND-001",
    retrieval_date=date.today(),
    user_permissions=["compliance", "risk"],
)

documents = await retriever.retrieve(context)
```

**Retrieval Priority:**
1. **SQL-first**: Structured queries against Snowflake views
2. **Lexical**: Exact keyword matching in policy documents
3. **Vector**: Semantic search for natural language context

### 4. Narrative Generator (`src/narrative/`)

LLM-assisted text generation with mandatory evidence binding:

```python
from compliance_rag import NarrativeGenerator, NarrativeType

generator = NarrativeGenerator(
    llm_client=openai_client,
    evidence_store=evidence_store,
)

narrative = await generator.generate(
    narrative_type=NarrativeType.EXCEPTION_COMMENTARY,
    evidence=evidence_documents,
    template="exception_commentary.jinja2",
)

# Validates all citations before returning
assert all(c.evidence_hash in evidence_hashes for c in narrative.citations)
```

**Key Rules:**
- All numerical values come from control results (not LLM)
- Every claim must have inline citation `[Evidence: {hash}]`
- "Insufficient evidence" returned if citation validation fails

### 5. Document Builder (`src/document_builder/`)

PDF generation with locked structure:

```python
from compliance_rag import DocumentBuilder, DocumentType

builder = DocumentBuilder(
    template_dir="templates/",
    output_dir="output/",
)

document = await builder.build(
    document_type=DocumentType.DAILY_COMPLIANCE_PACK,
    sections=[
        executive_summary,
        control_results_table,
        exception_details,
        evidence_appendix,
    ],
)

# Document hash for tamper detection
print(f"Document hash: {document.sha256_hash}")
```

## Output Documents

### Daily Compliance Pack

Generated every trading day with:

1. **Executive Summary** - Status overview with exception count
2. **Control Results Table** - All controls with pass/fail/warning
3. **Exception Details** - Each breach with commentary
4. **Position Snapshot** - Top positions by exposure
5. **Evidence Appendix** - All source data hashes

### SEC Filing Workpapers

Support packages for regulatory filings:

| Filing | Content |
|--------|---------|
| **Form PF** | Quarterly risk metrics with calculations |
| **13F** | Holdings snapshot with manager certification |
| **Form ADV** | Disclosure updates with evidence |

## SEC Examination Readiness

This system is designed for SEC examination. Key features:

### Prompt Logging

Every LLM interaction is logged:

```json
{
  "prompt_id": "uuid",
  "timestamp": "2024-01-15T09:30:00Z",
  "model": "gpt-4-turbo",
  "prompt": "...",
  "response": "...",
  "citations_validated": true,
  "evidence_hashes": ["abc123", "def456"]
}
```

### Evidence Chain

All outputs trace back to source:

```
Document → Section → Narrative → Evidence → Source Query → Snowflake View
```

### Reproducibility

Given the same:
- Run date
- Fund IDs
- Control definitions
- Source data snapshot

The system produces identical outputs.

## Installation

### Quick Start with Docker (Recommended)

The fastest way to get started is using Docker Compose, which sets up PostgreSQL and Redis automatically:

```bash
# Clone repository
git clone https://github.com/your-org/compliance-rag.git
cd compliance-rag

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env

# Start PostgreSQL and Redis with Docker
docker-compose up -d

# Verify services are running
docker-compose ps
```

**Services started by Docker:**

| Service | Container Name | Port | Purpose |
|---------|---------------|------|---------|
| PostgreSQL | `compliance_postgres` | `5432` | Workflow & audit database |
| Redis | `compliance_redis` | `6379` | Ephemeral alerts cache |
| pgAdmin | `compliance_pgadmin` | `5050` | Web UI for PostgreSQL (optional) |

**Docker Commands Reference:**

```bash
# Start all services
docker-compose up -d

# Stop all services
docker-compose down

# View logs
docker-compose logs -f

# View logs for specific service
docker-compose logs -f postgres

# Restart services
docker-compose restart

# Remove all data (fresh start)
docker-compose down -v
```

### Run Transparent Demo

After starting Docker, run the transparent demo to verify everything works:

```bash
python run_transparent.py
```

This will show you:
- ✅ Environment check (what's configured)
- ✅ Control definitions loaded
- ✅ Mock position data
- ✅ Compliance control execution
- ✅ Evidence hashing
- ✅ Example LLM prompts
- ✅ Document structure

### LLM Configuration (Optional but Recommended)

The system generates AI-powered compliance narratives when an LLM API key is configured. Without an API key, it falls back to template-based narratives.

**Option 1: Anthropic Claude (Recommended for Compliance)**
```bash
# Windows
set ANTHROPIC_API_KEY=sk-ant-api03-...

# Linux/Mac
export ANTHROPIC_API_KEY=sk-ant-api03-...
```

**Option 2: OpenAI GPT-4**
```bash
# Windows
set OPENAI_API_KEY=sk-...
set LLM_PROVIDER=openai

# Linux/Mac
export OPENAI_API_KEY=sk-...
export LLM_PROVIDER=openai
```

**Option 3: Local LLM (Ollama)**
```bash
# First install Ollama: https://ollama.ai
ollama serve
ollama pull llama3.1:70b

# Then run with:
set LLM_PROVIDER=ollama
```

**Test LLM Integration:**
```bash
python test_llm_integration.py
```

**Run Full Pipeline with LLM:**
```bash
python run_database_pipeline.py --setup-sample
```

The system features **automatic data anonymization** - sensitive data (tickers, large dollar amounts) is anonymized before being sent to external LLMs and restored in the response.

### Manual Installation (Without Docker)

If you prefer to set up databases manually:

```bash
# Clone repository
git clone https://github.com/your-org/compliance-rag.git
cd compliance-rag

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your database credentials

# Run database migrations
python -m alembic upgrade head
```

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
# ============================================
# PostgreSQL (Required)
# ============================================
# Using Docker (default):
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=compliance
POSTGRES_USER=compliance_user
POSTGRES_PASSWORD=compliance_dev_password_123

# ============================================
# Snowflake (Optional - uses mock data if not set)
# ============================================
SNOWFLAKE_ACCOUNT=your_account
SNOWFLAKE_USER=service_account
SNOWFLAKE_PASSWORD=your_password
SNOWFLAKE_WAREHOUSE=COMPLIANCE_WH
SNOWFLAKE_DATABASE=COMPLIANCE_DB

# ============================================
# OpenAI (Optional - shows prompts if not set)
# ============================================
OPENAI_API_KEY=sk-your-key-here
LLM_MODEL=gpt-4-turbo
LLM_TEMPERATURE=0.0  # Deterministic output

# ============================================
# Redis (Optional)
# ============================================
# Using Docker (default):
REDIS_HOST=localhost
REDIS_PORT=6379

# ============================================
# Vector Store (Optional)
# ============================================
PINECONE_API_KEY=your-key
PINECONE_ENVIRONMENT=us-east-1-aws
PINECONE_INDEX=compliance-docs

# ============================================
# Output Directories
# ============================================
OUTPUT_DIR=./output
TEMPLATE_DIR=./templates
```

### Minimal Configuration (Demo Mode)

For a quick demo with mock data, you only need Docker running:

```bash
# Start Docker services
docker-compose up -d

# Run demo (uses mock Snowflake data, shows LLM prompts)
python run_transparent.py
```

### Production Configuration

For production, configure all services:

| Service | Required | Notes |
|---------|----------|-------|
| PostgreSQL | ✅ Yes | Use managed service (RDS, Cloud SQL) |
| Snowflake | ✅ Yes | Read-only service account |
| OpenAI | ✅ Yes | For narrative generation |
| Redis | Optional | For alerts caching |
| Pinecone | Optional | For semantic search |

## Usage

### Daily Compliance Run

```python
import asyncio
from datetime import date
from compliance_rag import ComplianceOrchestrator

async def run_daily():
    orchestrator = ComplianceOrchestrator.from_env()
    
    result = await orchestrator.run_daily_compliance(
        run_date=date.today(),
        fund_ids=["FUND-001", "FUND-002", "FUND-003"],
    )
    
    print(f"Status: {result.status}")
    print(f"Exceptions: {result.exception_count}")
    print(f"Document: {result.document_path}")
    print(f"Hash: {result.document_hash}")

asyncio.run(run_daily())
```

### Filing Workpapers

```python
from compliance_rag import DocumentType

result = await orchestrator.generate_filing_workpapers(
    document_type=DocumentType.FORM_PF,
    as_of_date=date(2024, 3, 31),  # Quarter end
    fund_ids=["FUND-001"],
)
```

### API Server

```bash
# Start FastAPI server
uvicorn compliance_rag.api:app --host 0.0.0.0 --port 8000

# Endpoints:
# POST /daily-compliance - Run daily compliance
# POST /filing-workpapers - Generate filing support
# GET /evidence/{hash} - Retrieve evidence by hash
# GET /audit-log/{run_id} - Get audit trail
```

## Testing

```bash
# Run all tests
pytest tests/

# With coverage
pytest tests/ --cov=compliance_rag --cov-report=html

# Integration tests (requires database)
pytest tests/integration/ --run-integration
```

## Project Structure

```
compliance-rag/
├── schemas/
│   ├── postgres_schema.sql    # Workflow & audit tables
│   └── snowflake_views.sql    # Position/exposure views
├── src/
│   ├── __init__.py           # Package exports
│   ├── config/               # Configuration management
│   │   ├── __init__.py
│   │   └── settings.py       # Pydantic settings
│   ├── control_runner/       # Deterministic controls
│   │   ├── __init__.py
│   │   ├── controls.py       # Control definitions
│   │   └── runner.py         # Execution engine
│   ├── evidence_store/       # Audit trail
│   │   ├── __init__.py
│   │   └── store.py          # Evidence management
│   ├── retrieval/            # Hybrid RAG
│   │   ├── __init__.py
│   │   └── retriever.py      # 3-tier retrieval
│   ├── narrative/            # Text generation
│   │   ├── __init__.py
│   │   └── generator.py      # LLM with citations
│   ├── document_builder/     # PDF generation
│   │   ├── __init__.py
│   │   └── builder.py        # Document assembly
│   └── orchestrator.py       # Main coordinator
├── templates/                 # Document templates
│   ├── daily_pack.jinja2
│   ├── form_pf.jinja2
│   └── exception_commentary.jinja2
├── policies/                  # Compliance policies
│   └── investment_guidelines.md
├── tests/                     # Test suite
│   ├── unit/
│   └── integration/
├── requirements.txt
├── .env.example
└── README.md
```

## License

Proprietary - Internal use only.

## Support

Contact: compliance-tech@yourfund.com
