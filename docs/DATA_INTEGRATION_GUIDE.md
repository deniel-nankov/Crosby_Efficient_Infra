# Data Integration Guide: High-Quality Dataset Pipeline

## Overview

This guide walks you through integrating your data sources into the Compliance RAG system with **pristine data quality** as the top priority. Every step includes validation checkpoints.

---

## Architecture: Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DATA SOURCES                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                   │
│  │  SNOWFLAKE   │    │   POLICY     │    │   VECTOR     │                   │
│  │  (Positions, │    │  DOCUMENTS   │    │   STORE      │                   │
│  │   Prices)    │    │  (Markdown,  │    │  (Embeddings)│                   │
│  └──────┬───────┘    │   PDF)       │    └──────┬───────┘                   │
│         │            └──────┬───────┘           │                           │
│         │                   │                   │                           │
│         ▼                   ▼                   ▼                           │
│  ┌──────────────────────────────────────────────────────────────────┐      │
│  │                    DATA VALIDATION LAYER                          │      │
│  │  • Schema validation    • Completeness checks                     │      │
│  │  • Data type validation • Referential integrity                   │      │
│  │  • Business rule checks • Anomaly detection                       │      │
│  └────────────────────────────┬─────────────────────────────────────┘      │
│                               │                                             │
│                               ▼                                             │
│  ┌──────────────────────────────────────────────────────────────────┐      │
│  │                    POSTGRESQL (Evidence Store)                    │      │
│  │  • control_runs          • control_results                        │      │
│  │  • control_definitions   • audit_log                              │      │
│  │  • positions_current     • policy_chunks                          │      │
│  └────────────────────────────┬─────────────────────────────────────┘      │
│                               │                                             │
│                               ▼                                             │
│  ┌──────────────────────────────────────────────────────────────────┐      │
│  │                    RAG RETRIEVAL LAYER                            │      │
│  │  1. Structured SQL → Control results, metrics                     │      │
│  │  2. Lexical Search → Policy text, regulations                     │      │
│  │  3. Vector Search  → Semantic similarity                          │      │
│  └──────────────────────────────────────────────────────────────────┘      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Step-by-Step Integration

### STEP 1: Configure Environment

**File: `.env`**

```bash
# PostgreSQL (required)
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=compliance
POSTGRES_USER=compliance_user
POSTGRES_PASSWORD=compliance_pass

# Snowflake (required for production, mock available for testing)
SNOWFLAKE_ACCOUNT=your_account.region
SNOWFLAKE_USER=compliance_svc
SNOWFLAKE_PASSWORD=your_password
SNOWFLAKE_WAREHOUSE=COMPLIANCE_WH
SNOWFLAKE_DATABASE=HEDGE_FUND_DATA
SNOWFLAKE_SCHEMA=compliance

# OpenAI (optional, for embeddings)
OPENAI_API_KEY=sk-your-key

# Redis (for caching)
REDIS_HOST=localhost
REDIS_PORT=6379
```

### STEP 2: Deploy Database Schema

```bash
# Start Docker services
docker-compose up -d

# Apply schema (tables already exist if you followed setup)
docker exec -i compliance_postgres psql -U compliance_user -d compliance < schemas/postgres_schema.sql
```

### STEP 3: Deploy Snowflake Views (Production)

Execute `schemas/snowflake_views.sql` in your Snowflake environment:

```sql
USE DATABASE HEDGE_FUND_DATA;
USE SCHEMA compliance;

-- Run the snowflake_views.sql script
-- This creates curated, read-only views for compliance
```

### STEP 4: Run Integration Demo

```bash
# Test with mock data first
python run_data_integration.py

# This will:
# 1. Extract sample positions
# 2. Validate data quality (must score ≥95%)
# 3. Load to PostgreSQL
# 4. Ingest policy documents
# 5. Validate control definitions
```

### STEP 5: Seed Control Definitions

Control definitions tell the system what to check. Use the sample controls or customize:

```python
from src.data_quality import DataIntegrationOrchestrator, get_sample_controls

# Connect to PostgreSQL
orchestrator = DataIntegrationOrchestrator(postgres_conn=conn)

# Seed the 10 sample controls
controls = get_sample_controls()
inserted, reports = orchestrator.seed_control_definitions(controls)
print(f"Seeded {inserted} controls")
```

**Sample Controls Included:**
| Code | Name | Threshold |
|------|------|-----------|
| CONC_ISSUER_001 | Single Issuer Concentration | ≤10% NAV |
| CONC_SECTOR_001 | Sector Concentration | ≤30% NAV |
| EXP_GROSS_001 | Gross Exposure Limit | ≤200% NAV |
| EXP_NET_LONG_001 | Net Long Exposure | ≤100% NAV |
| LIQ_T1_001 | T+1 Liquidity | ≥10% NAV |
| LIQ_T7_001 | T+7 Liquidity | ≥40% NAV |
| CASH_MIN_001 | Minimum Cash | ≥2% NAV |

### STEP 6: Add Your Policy Documents

Place Markdown policy files in `policies/` directory:

```markdown
# Investment Guidelines

## Fund: Master Fund LP
## Effective Date: January 1, 2024
## Version: 3.2

## 1. Position Limits
### 1.1 Gross Exposure
- **Maximum Gross Exposure**: 200% of NAV
...
```

The system will:
1. Parse sections semantically (not arbitrary splits)
2. Extract compliance keywords
3. Create searchable chunks with citations
4. Validate document quality

---

## Data Quality Framework

### Quality Dimensions (DAMA Standard)

| Dimension | Weight | What It Checks |
|-----------|--------|----------------|
| **Completeness** | 20% | All required fields populated |
| **Accuracy** | 25% | Market values match qty × price (±5%) |
| **Consistency** | 15% | Long/short signs match quantities |
| **Timeliness** | 15% | Prices not stale (>2 days) |
| **Uniqueness** | 15% | No duplicate positions |
| **Validity** | 10% | Valid asset classes, ISIN formats |

### Quality Gate

Data must score **≥95%** to pass the quality gate. If rejected:

```
❌ DATA QUALITY GATE: FAILED
   Reason: Critical issues found: 3
   
   Critical Issues:
   - position_id: Required fields missing
   - security_identifiers: No ISIN/CUSIP/ticker found
   - quantity: Long/short signs inconsistent
```

### Validation Rules

**Position Data:**
- `POS_COMP_001`: All required fields present
- `POS_COMP_002`: Security identifier present (ISIN/CUSIP/SEDOL/ticker)
- `POS_ACC_001`: Market value ≈ quantity × price
- `POS_ACC_002`: Market value < $1B per position
- `POS_CON_001`: Quantity sign matches long/short
- `POS_TIM_001`: Price date within 2 business days

**Control Definitions:**
- Valid category (concentration, exposure, liquidity, etc.)
- Valid operator (gt, gte, lt, lte, eq)
- Numeric threshold value
- Valid frequency (daily, weekly, monthly)

---

## Production Deployment

### Daily Sync Schedule

```python
# Run daily at 6 AM UTC
from src.data_quality import DataIntegrationOrchestrator

orchestrator = DataIntegrationOrchestrator(
    postgres_conn=conn,
    quality_threshold=95.0,  # Reject below 95%
)

# Full daily sync
run = orchestrator.run_daily_sync(
    policy_dir=Path('policies/'),
)

if run.status == 'completed':
    print(f"✅ Sync complete: {run.position_count} positions")
elif run.status == 'rejected':
    print(f"❌ Quality gate failed: {run.error_message}")
    # Alert compliance team
```

### Monitoring

Check quality trends in PostgreSQL:

```sql
-- Recent integration runs
SELECT run_id, started_at, status, position_count, 
       position_quality_score, error_message
FROM integration_runs
ORDER BY started_at DESC
LIMIT 10;

-- Quality score trend
SELECT DATE(validated_at), dataset_name, 
       AVG(overall_score) as avg_score
FROM data_quality_reports
GROUP BY DATE(validated_at), dataset_name
ORDER BY DATE(validated_at) DESC;
```

---

## Troubleshooting

### "Snowflake not configured"
- Check `.env` has SNOWFLAKE_* variables
- Verify network connectivity to Snowflake
- System will use mock data if not configured

### "Quality gate failed"
- Check the quality report for critical issues
- Fix source data in Snowflake
- Re-run validation

### "Position staging not found"
- Run schema migration: `docker exec -i compliance_postgres psql ...`
- Tables created in `schemas/postgres_schema.sql`

---

## Files Created

| File | Purpose |
|------|---------|
| `src/data_quality/validators.py` | Quality validation rules |
| `src/data_quality/snowflake_connector.py` | Snowflake data extraction |
| `src/data_quality/policy_ingestion.py` | Policy chunking pipeline |
| `src/data_quality/orchestrator.py` | Main integration coordinator |
| `run_data_integration.py` | Demo/execution script |

---

## Next Steps

After completing integration:

1. **Run Compliance Controls**: `python run_transparent.py`
2. **Generate Reports**: Use the document_builder
3. **Set up Monitoring**: Track quality scores daily
4. **Add More Policies**: Expand `policies/` directory
5. **Configure Alerts**: Alert on quality < 95%
