# Data Input Guide - No Snowflake Required!

Since you were rejected from Snowflake, here's how to input your data using **PostgreSQL** (FREE).

## Architecture (Updated)

```
┌─────────────────────────────────────────────────────────────────┐
│                    YOUR EXISTING SYSTEMS                        │
│     Bloomberg Terminal  │  Eze OMS  │  Geneva  │  Excel         │
└────────────┬────────────────┬───────────┬───────────┬───────────┘
             │                │           │           │
             │   Export as CSV files (daily)          │
             │                │           │           │
             ▼                ▼           ▼           ▼
┌─────────────────────────────────────────────────────────────────┐
│                      CSV FILES (data/)                          │
│     positions.csv    │   controls.csv    │   nav.csv            │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                    python quick_start.py
                    (or automated via cron)
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                    POSTGRESQL (Docker)                          │
│   fund_positions  │  fund_control_results  │  fund_nav          │
│                   │                        │                     │
│   + pgvector for policy embeddings                              │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                    COMPLIANCE RAG SYSTEM                        │
│                                                                 │
│   1. Read positions & control results from PostgreSQL           │
│   2. Retrieve relevant policies (RAG)                           │
│   3. Generate narratives with citations                         │
│   4. Output compliance workpapers                               │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start (3 Steps)

### Step 1: Start PostgreSQL
```bash
docker-compose up -d
```

### Step 2: Load sample data and test
```bash
python quick_start.py
```

### Step 3: Replace with your real data

**Option A: Use CSVs (Easiest)**
1. Export positions from Bloomberg/Eze to `data/positions.csv`
2. Export controls from your system to `data/controls.csv`
3. Run the load script

**Option B: Direct database insert**
Connect to PostgreSQL and insert directly:
```bash
docker exec -it compliance_postgres psql -U compliance_user -d compliance
```

## CSV File Formats

### positions.csv
```csv
security_id,ticker,security_name,quantity,market_value,currency,sector,issuer,asset_class
SEC001,AAPL,Apple Inc,100000,15000000.00,USD,Technology,Apple Inc,equity
```

### controls.csv
```csv
control_id,control_name,control_type,calculated_value,threshold,threshold_operator,status,breach_amount
CONC_ISSUER_001,Single Issuer Concentration,concentration,7.5,10.0,lte,pass,
```

## Why PostgreSQL Instead of Snowflake?

| Feature | Snowflake | PostgreSQL |
|---------|-----------|------------|
| Cost | $$$ (pay per compute) | FREE |
| Setup | Cloud account needed | `docker-compose up` |
| Performance | Excellent for big data | Excellent for < 1M rows |
| pgvector | ❌ | ✅ Built-in |
| Your data size? | Overkill | Perfect fit |

For a hedge fund with < 10,000 positions and < 100 daily controls, PostgreSQL is:
- Faster to query (no cloud latency)
- Free forever
- Keeps data local (data privacy)
- Already in the Docker stack

## Files Changed

- `docker-compose.yml` - Now uses `pgvector/pgvector:pg16`
- `src/integration/postgres_adapter.py` - New adapter (replaces Snowflake)
- `data/sample/positions.csv` - Sample position data
- `data/sample/controls.csv` - Sample control data
- `quick_start.py` - One-command setup

## Next Steps

1. Run `docker-compose up -d`
2. Run `python quick_start.py`
3. See the compliance report with sample data
4. Replace sample data with your real data
