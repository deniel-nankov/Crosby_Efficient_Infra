# 🔍 Compliance RAG System - Complete Architecture & Validation Guide

**Document Version:** 1.0  
**Last Updated:** January 18, 2026  
**Purpose:** Eliminate "black box" confusion - understand EXACTLY where data flows and what each component does.

---

## 📋 Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Architecture Diagram](#2-system-architecture-diagram)
3. [Component Inventory](#3-component-inventory)
4. [Data Flow Walkthrough](#4-data-flow-walkthrough)
5. [Validation Checkpoints](#5-validation-checkpoints)
6. [Tool Integration Map](#6-tool-integration-map)
7. [Step-by-Step Pipeline Execution](#7-step-by-step-pipeline-execution)
8. [Troubleshooting Guide](#8-troubleshooting-guide)

---

## 1. Executive Summary

### What This System Does (Plain English)
This is a **Compliance Report Generator** for hedge funds. It takes your portfolio data (positions, trades) and policy documents, then generates professionally written compliance reports with proper citations - like having a compliance analyst who never sleeps.

### The Key Innovation
**Separation of Concerns:**
- **DETERMINISTIC** calculations (math) → Done by SQL/Python, NOT the AI
- **NARRATIVE** generation (prose) → Done by the AI (LLM), but ONLY using pre-verified facts

This means the SEC can audit every number because they come from your own systems, not from AI hallucinations.

---

## 2. System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        YOUR DATA SOURCES (INPUT LAYER)                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐          │
│  │  CSV Files       │  │  Bloomberg/Eze   │  │  Policy Docs     │          │
│  │  (positions_     │  │  (if connected)  │  │  (policies/*.md) │          │
│  │   YYYYMMDD.csv)  │  │                  │  │                  │          │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘          │
│           │                     │                     │                     │
│           ▼                     ▼                     ▼                     │
│  ┌──────────────────────────────────────────────────────────────┐          │
│  │              CLIENT ADAPTER (client_adapter.py)              │          │
│  │  ─────────────────────────────────────────────────────────── │          │
│  │  • Reads CSV files OR connects to live systems               │          │
│  │  • Normalizes data into standard format (Position, Control)  │          │
│  │  • Creates DataSnapshot object                               │          │
│  │                                                               │          │
│  │  📁 File: src/integration/client_adapter.py                   │          │
│  │  🔧 Key Classes: CSVAdapter, DataSnapshot, Position           │          │
│  └───────────────────────────────┬──────────────────────────────┘          │
│                                  │                                          │
└──────────────────────────────────┼──────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DATA STORAGE LAYER                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                    POSTGRESQL DATABASE                              │    │
│  │                    (Docker Container)                               │    │
│  │  ──────────────────────────────────────────────────────────────── │    │
│  │                                                                    │    │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐   │    │
│  │  │ fund_positions  │  │ fund_control_   │  │ fund_nav        │   │    │
│  │  │                 │  │ results         │  │                 │   │    │
│  │  │ • security_id   │  │ • control_id    │  │ • nav           │   │    │
│  │  │ • ticker        │  │ • calculated_   │  │ • as_of_date    │   │    │
│  │  │ • market_value  │  │   value         │  │                 │   │    │
│  │  │ • sector        │  │ • threshold     │  │                 │   │    │
│  │  │ • as_of_date    │  │ • status        │  │                 │   │    │
│  │  └─────────────────┘  └─────────────────┘  └─────────────────┘   │    │
│  │                                                                    │    │
│  │  📁 Schema: schemas/postgres_schema.sql                           │    │
│  │  📁 Adapter: src/integration/postgres_adapter.py                   │    │
│  │  🔧 Class: PostgresDataSource                                      │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                    VECTOR STORE (pgvector)                          │    │
│  │  ──────────────────────────────────────────────────────────────── │    │
│  │  • Stores policy document embeddings                              │    │
│  │  • Enables semantic search ("find policies about concentration")   │    │
│  │  • Uses 1536-dimension vectors (OpenAI embedding format)           │    │
│  │                                                                    │    │
│  │  📁 Part of PostgreSQL with pgvector extension                     │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PROCESSING LAYER                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │  STEP 1: CONTROL RUNNER (Deterministic - NO AI)                    │    │
│  │  ──────────────────────────────────────────────────────────────── │    │
│  │                                                                    │    │
│  │  What it does:                                                     │    │
│  │  • Executes SQL queries to calculate compliance metrics            │    │
│  │  • Compares results against thresholds (e.g., "is 8.5% > 5%?")    │    │
│  │  • Records PASS/FAIL/WARNING status                                │    │
│  │  • Creates exceptions for failures                                 │    │
│  │                                                                    │    │
│  │  PLAIN ENGLISH: This is the calculator. It does pure math.         │    │
│  │  If your top position is 8.5% and limit is 5%, it says "BREACH".   │    │
│  │  No AI involved - just arithmetic.                                 │    │
│  │                                                                    │    │
│  │  📁 File: src/control_runner/runner.py                             │    │
│  │  📁 File: src/control_runner/controls.py                           │    │
│  │  🔧 Classes: ControlRunner, ControlExecutionResult                 │    │
│  │                                                                    │    │
│  │  Input:  DataSnapshot (positions, NAV)                             │    │
│  │  Output: List[ControlExecutionResult] with status, breach_amount   │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                   │                                         │
│                                   ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │  STEP 2: EVIDENCE STORE (Audit Trail)                              │    │
│  │  ──────────────────────────────────────────────────────────────── │    │
│  │                                                                    │    │
│  │  What it does:                                                     │    │
│  │  • Saves control results to PostgreSQL                             │    │
│  │  • Records timestamps, hashes, execution metadata                  │    │
│  │  • Creates immutable audit trail for SEC                           │    │
│  │  • Links results to specific data snapshots                        │    │
│  │                                                                    │    │
│  │  PLAIN ENGLISH: This is your compliance filing cabinet.            │    │
│  │  Every calculation is logged with a timestamp so regulators        │    │
│  │  can trace back: "What data did you use? When did you run it?"     │    │
│  │                                                                    │    │
│  │  📁 File: src/evidence_store/store.py                              │    │
│  │  🔧 Class: EvidenceStore                                           │    │
│  │                                                                    │    │
│  │  Input:  ControlExecutionResult                                    │    │
│  │  Output: Persisted records in control_runs, control_results,       │    │
│  │          exceptions tables                                         │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                   │                                         │
│                                   ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │  STEP 3: RETRIEVER (Hybrid RAG Search)                             │    │
│  │  ──────────────────────────────────────────────────────────────── │    │
│  │                                                                    │    │
│  │  What it does:                                                     │    │
│  │  • THREE-TIER search strategy:                                     │    │
│  │    1. STRUCTURED: SQL queries for control results/exceptions       │    │
│  │    2. LEXICAL: Full-text search for exact policy matches           │    │
│  │    3. VECTOR: Semantic search for related policies                 │    │
│  │                                                                    │    │
│  │  PLAIN ENGLISH: This is your research assistant.                   │    │
│  │  "Hey, we have a concentration breach. What does our policy say?"  │    │
│  │  It finds the relevant policy sections and control history.        │    │
│  │                                                                    │    │
│  │  📁 File: src/retrieval/retriever.py                               │    │
│  │  🔧 Class: HybridRetriever                                         │    │
│  │                                                                    │    │
│  │  Input:  Query ("concentration breach in Energy sector")           │    │
│  │  Output: RetrievalContext with:                                    │    │
│  │          - structured_results (control data)                       │    │
│  │          - lexical_results (exact policy matches)                  │    │
│  │          - vector_results (semantically similar policies)          │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                   │                                         │
│                                   ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │  STEP 4: NARRATIVE GENERATOR (LLM - This is where AI lives)        │    │
│  │  ──────────────────────────────────────────────────────────────── │    │
│  │                                                                    │    │
│  │  What it does:                                                     │    │
│  │  • Takes retrieved context (facts + policies)                      │    │
│  │  • Sends to LLM with strict prompt template                        │    │
│  │  • LLM generates human-readable prose WITH CITATIONS               │    │
│  │  • Validates that output includes proper citations                 │    │
│  │                                                                    │    │
│  │  CRITICAL CONSTRAINTS (hardcoded):                                 │    │
│  │  ❌ LLM CANNOT do calculations                                     │    │
│  │  ❌ LLM CANNOT invent facts                                        │    │
│  │  ✅ LLM CAN ONLY rephrase provided evidence into prose             │    │
│  │  ✅ Every sentence must cite its source                            │    │
│  │                                                                    │    │
│  │  PLAIN ENGLISH: This is your writing assistant.                    │    │
│  │  You give it: "AAPL is 8.5%, limit is 5%, breach is 3.5%"          │    │
│  │  It writes: "Apple Inc. position exceeded the 5% single-issuer     │    │
│  │  limit by 3.5 percentage points [Control: CONC_001, Result: FAIL]" │    │
│  │                                                                    │    │
│  │  📁 File: src/narrative/generator.py                               │    │
│  │  📁 File: src/integration/llm_config.py                            │    │
│  │  🔧 Classes: NarrativeGenerator, LLMClient                         │    │
│  │                                                                    │    │
│  │  LLM Options:                                                      │    │
│  │  • Ollama (LOCAL) - llama3.1:70b - Data never leaves your server  │    │
│  │  • Claude (CLOUD) - claude-sonnet-4-20250514 - Better quality            │    │
│  │  • OpenAI (CLOUD) - gpt-4o - Alternative                           │    │
│  │                                                                    │    │
│  │  Input:  RetrievalContext + PromptTemplate                         │    │
│  │  Output: GeneratedNarrative with citations                         │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                   │                                         │
│                                   ▼                                         │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │  STEP 5: DOCUMENT BUILDER (Output Generation)                      │    │
│  │  ──────────────────────────────────────────────────────────────── │    │
│  │                                                                    │    │
│  │  What it does:                                                     │    │
│  │  • Takes narrative + data + tables                                 │    │
│  │  • Renders into professional PDF document                          │    │
│  │  • Includes all citations and evidence references                  │    │
│  │                                                                    │    │
│  │  PLAIN ENGLISH: This is your document formatter.                   │    │
│  │  Takes all the pieces and makes a pretty PDF for your CCO.         │    │
│  │                                                                    │    │
│  │  📁 File: src/document_builder/                                    │    │
│  │  🔧 Class: DocumentBuilder                                         │    │
│  │                                                                    │    │
│  │  Input:  Narrative + ControlResults + PolicyCitations              │    │
│  │  Output: PDF file in output/ directory                             │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ORCHESTRATION LAYER                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                    ORCHESTRATOR (orchestrator.py)                   │    │
│  │  ──────────────────────────────────────────────────────────────── │    │
│  │                                                                    │    │
│  │  What it does:                                                     │    │
│  │  • Coordinates ALL the above components                            │    │
│  │  • Ensures correct execution order                                 │    │
│  │  • Handles errors and retries                                      │    │
│  │  • Provides single entry point: run_daily_compliance()             │    │
│  │                                                                    │    │
│  │  PLAIN ENGLISH: This is the conductor of the orchestra.            │    │
│  │  You call ONE function, it handles the rest.                       │    │
│  │                                                                    │    │
│  │  📁 File: src/orchestrator.py                                      │    │
│  │  🔧 Class: ComplianceOrchestrator                                  │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │                    RAG PIPELINE (rag_pipeline.py)                   │    │
│  │  ──────────────────────────────────────────────────────────────── │    │
│  │                                                                    │    │
│  │  What it does:                                                     │    │
│  │  • Simplified pipeline for standalone RAG                          │    │
│  │  • Takes DataSnapshot → Returns ComplianceReport                   │    │
│  │  • Good for demos and testing                                      │    │
│  │                                                                    │    │
│  │  📁 File: src/integration/rag_pipeline.py                          │    │
│  │  🔧 Class: ComplianceRAGPipeline                                   │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Component Inventory

### Complete File Map

| Layer | File | Purpose | Key Class/Function |
|-------|------|---------|-------------------|
| **Config** | `src/config/settings.py` | Environment configuration | `Settings`, `get_settings()` |
| **Input** | `src/integration/client_adapter.py` | Read from CSV/Bloomberg/Eze | `CSVAdapter`, `DataSnapshot` |
| **Input** | `src/integration/postgres_adapter.py` | PostgreSQL data source | `PostgresDataSource` |
| **Processing** | `src/control_runner/runner.py` | Execute compliance checks | `ControlRunner` |
| **Processing** | `src/control_runner/controls.py` | Control definitions | `ControlDefinition` |
| **Storage** | `src/evidence_store/store.py` | Audit trail persistence | `EvidenceStore` |
| **Retrieval** | `src/retrieval/retriever.py` | Hybrid RAG search | `HybridRetriever` |
| **Generation** | `src/narrative/generator.py` | LLM narrative generation | `NarrativeGenerator` |
| **Generation** | `src/integration/llm_config.py` | LLM client setup | `LLMClient`, `OllamaClient` |
| **Output** | `src/document_builder/` | PDF generation | `DocumentBuilder` |
| **Orchestration** | `src/orchestrator.py` | Main coordinator | `ComplianceOrchestrator` |
| **Orchestration** | `src/integration/rag_pipeline.py` | Simplified RAG flow | `ComplianceRAGPipeline` |

### External Dependencies

| Service | Purpose | Where Used | Docker Container |
|---------|---------|------------|-----------------|
| **PostgreSQL** | Data storage + audit trail | `postgres_adapter.py`, `store.py` | `postgres` |
| **pgvector** | Vector embeddings for RAG | `retriever.py` | Part of `postgres` |
| **Ollama** | Local LLM inference | `llm_config.py` | `ollama` |
| **Airflow** | Job scheduling (optional) | `dags/` | `airflow-*` |

---

## 4. Data Flow Walkthrough

### Complete Data Journey (With Timestamps)

```
TIME    ACTION                                      LOCATION
─────   ─────────────────────────────────────────   ──────────────────────────────
T+0ms   Load positions CSV                          client_adapter.py:get_positions()
        └── data/commodity_positions_20260117.csv
        
T+50ms  Load control results CSV                    client_adapter.py:get_control_results()
        └── data/commodity_controls_20260117.csv
        
T+100ms Create DataSnapshot object                  client_adapter.py:get_snapshot()
        └── Contains: 28 positions, 20 controls, $2B NAV
        
T+150ms Execute control checks                      control_runner/runner.py:run_daily_controls()
        └── For each control:
            ├── SQL query against PostgreSQL
            ├── Compare value vs threshold
            └── Return PASS/FAIL/WARNING
            
T+500ms Store results in Evidence Store             evidence_store/store.py:record_control_result()
        └── PostgreSQL tables:
            ├── control_runs (run metadata)
            ├── control_results (each check)
            └── exceptions (failures)
            
T+600ms Retrieve relevant context                   retrieval/retriever.py:retrieve_for_daily_pack()
        └── Three-tier search:
            ├── SQL: Get today's control results
            ├── Lexical: Find exact policy matches
            └── Vector: Find similar policies
            
T+800ms Load policy documents                       retrieval/retriever.py
        └── policies/
            ├── commodity_trading.md
            ├── concentration_limits.md
            └── exposure_limits.md
            
T+1000ms Generate narrative                         narrative/generator.py:generate_daily_summary()
         └── Send to LLM:
             ├── System prompt (strict rules)
             ├── User prompt (template + data)
             └── Retrieved context (facts + policies)
             
T+3000ms Receive LLM response                       narrative/generator.py
         └── Validate:
             ├── Has citations? ✓
             ├── No hallucinated numbers? ✓
             └── Follows template? ✓
             
T+3200ms Build PDF document                         document_builder/:build_daily_compliance_pack()
         └── output/compliance_report_20260118.pdf
```

---

## 5. Validation Checkpoints

### Pre-Flight Checks (Run Before Pipeline)

```bash
# ═══════════════════════════════════════════════════════════════════
# CHECKPOINT 1: Docker Services
# ═══════════════════════════════════════════════════════════════════
docker-compose ps

# Expected output:
# compliance_postgres    Up    0.0.0.0:5432->5432/tcp
# compliance_ollama      Up    0.0.0.0:11434->11434/tcp (optional)

# PLAIN ENGLISH: Are your databases running?
```

```bash
# ═══════════════════════════════════════════════════════════════════
# CHECKPOINT 2: Database Connection
# ═══════════════════════════════════════════════════════════════════
docker exec -it compliance_postgres psql -U compliance_user -d compliance -c "\dt"

# Expected: List of tables (fund_positions, fund_control_results, etc.)

# PLAIN ENGLISH: Can we talk to the database?
```

```bash
# ═══════════════════════════════════════════════════════════════════
# CHECKPOINT 3: Data Files Exist
# ═══════════════════════════════════════════════════════════════════
ls -la data/*.csv

# Expected:
# commodity_positions_20260117.csv
# commodity_controls_20260117.csv

# PLAIN ENGLISH: Do we have data to process?
```

```bash
# ═══════════════════════════════════════════════════════════════════
# CHECKPOINT 4: Policy Files Exist
# ═══════════════════════════════════════════════════════════════════
ls -la policies/*.md

# Expected:
# commodity_trading.md
# concentration_limits.md
# exposure_limits.md
# etc.

# PLAIN ENGLISH: Do we have policies for the LLM to cite?
```

```bash
# ═══════════════════════════════════════════════════════════════════
# CHECKPOINT 5: LLM Available (if using AI narrative)
# ═══════════════════════════════════════════════════════════════════
curl http://localhost:11434/api/tags

# Expected: List of available Ollama models

# OR for mock mode (no LLM needed):
export LLM_PROVIDER=mock

# PLAIN ENGLISH: Is the AI brain online?
```

---

## 6. Tool Integration Map

### Which Tool Does What?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TOOLS & RESPONSIBILITIES                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────┐                                                    │
│  │      POSTGRES       │                                                    │
│  │   (Data Storage)    │                                                    │
│  ├─────────────────────┤                                                    │
│  │ • Store positions   │◄──── load_real_commodities.py writes here         │
│  │ • Store controls    │                                                    │
│  │ • Store audit logs  │◄──── evidence_store/store.py writes here          │
│  │ • Store vectors     │◄──── retriever.py reads from here                 │
│  └─────────────────────┘                                                    │
│            ▲                                                                │
│            │ SQL queries                                                    │
│            │                                                                │
│  ┌─────────┴───────────┐                                                    │
│  │   PYTHON SCRIPTS    │                                                    │
│  │  (Processing Logic) │                                                    │
│  ├─────────────────────┤                                                    │
│  │ quick_start.py      │ → Load sample data, test connection               │
│  │ demo_data.py        │ → Load CSV data, show analysis                    │
│  │ load_real_          │ → Download real prices, generate positions        │
│  │   commodities.py    │                                                    │
│  │ run_demo.py         │ → Full pipeline execution                         │
│  └─────────────────────┘                                                    │
│            │                                                                │
│            │ Python calls                                                   │
│            ▼                                                                │
│  ┌─────────────────────┐                                                    │
│  │       OLLAMA        │                                                    │
│  │    (Local LLM)      │                                                    │
│  ├─────────────────────┤                                                    │
│  │ • Generate prose    │◄──── narrative/generator.py calls via API         │
│  │ • Follow prompts    │                                                    │
│  │ • Return citations  │                                                    │
│  │                     │                                                    │
│  │ Model: llama3.1:8b  │                                                    │
│  │ or llama3.1:70b     │                                                    │
│  └─────────────────────┘                                                    │
│                                                                             │
│  ┌─────────────────────┐                                                    │
│  │       DOCKER        │                                                    │
│  │   (Infrastructure)  │                                                    │
│  ├─────────────────────┤                                                    │
│  │ docker-compose.yml  │ → Defines all services                            │
│  │ • postgres          │ → Port 5432                                       │
│  │ • ollama            │ → Port 11434                                      │
│  │ • airflow (opt)     │ → Port 8080                                       │
│  └─────────────────────┘                                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data File Mapping

```
INPUT FILES (Your Data)                    PROCESSING                          OUTPUT
───────────────────────                    ──────────                          ──────

data/
├── commodity_positions_20260117.csv  ───► CSVAdapter.get_positions() ───► PostgreSQL
│   (28 positions, real prices)            
│                                          
├── commodity_controls_20260117.csv  ────► CSVAdapter.get_control_results() ─► PostgreSQL
│   (20 control checks)                    
│                                          
└── real_commodities/                      
    ├── brent_oil_daily.csv          ────► load_real_commodities.py ───► Prices used
    ├── wti_oil_daily.csv                  (Downloaded from EIA)          in positions
    ├── natural_gas_daily.csv              
    └── gold_monthly.csv             

policies/
├── commodity_trading.md             ────► HybridRetriever ───────────► LLM Context
├── concentration_limits.md                (Full-text + vector search)
├── exposure_limits.md               
├── liquidity_policy.md              
└── sec_compliance.md                

                                           ▼
                                    
                                    NarrativeGenerator
                                    (LLM writes prose)
                                           
                                           ▼
                                           
                                    output/
                                    └── compliance_report_20260118.pdf
```

---

## 7. Step-by-Step Pipeline Execution

### Option A: Quick Test (No LLM)

```bash
# Step 1: Start database
docker-compose up -d postgres

# Step 2: Load sample data
python quick_start.py

# What happens:
# 1. Connects to PostgreSQL
# 2. Creates tables
# 3. Loads 28 sample positions
# 4. Loads 20 control results
# 5. Prints summary report (no AI)
```

### Option B: Full Demo (With Analysis)

```bash
# Step 1: Ensure data files exist
ls data/commodity_positions_*.csv

# Step 2: Run demo script
python demo_data.py

# What happens:
# 1. Loads CSV files
# 2. Analyzes positions (sector breakdown, top holdings)
# 3. Shows control results (pass/fail/warning)
# 4. Generates sample narratives (simulated)
```

### Option C: Full RAG Pipeline (With LLM)

```bash
# Step 1: Start all services
docker-compose up -d

# Step 2: Pull LLM model (first time only)
docker exec -it ollama ollama pull llama3.1:8b

# Step 3: Set environment
export LLM_PROVIDER=ollama
export LLM_MODEL=llama3.1:8b

# Step 4: Run full pipeline
python run_demo.py

# What happens:
# 1. Load positions + controls
# 2. Execute compliance checks
# 3. Store in Evidence Store
# 4. Retrieve relevant policies (RAG)
# 5. Generate narrative (LLM)
# 6. Build PDF document
```

---

## 8. Troubleshooting Guide

### Common Issues and Solutions

#### Issue: "Cannot connect to PostgreSQL"
```
Error: psycopg2.OperationalError: could not connect to server
```

**Diagnosis:**
```bash
docker-compose ps  # Is postgres running?
docker logs postgres  # Any errors?
```

**Solution:**
```bash
docker-compose down
docker-compose up -d postgres
sleep 5  # Wait for startup
python quick_start.py
```

**Plain English:** The database container crashed or hasn't started. Restart it.

---

#### Issue: "No positions found for date"
```
Warning: Position file not found: data/positions_20260118.csv
```

**Diagnosis:**
```bash
ls -la data/*.csv  # What files exist?
```

**Solution:**
```bash
# Use the commodities data we generated:
cp data/commodity_positions_20260117.csv data/positions_20260118.csv
cp data/commodity_controls_20260117.csv data/controls_20260118.csv
```

**Plain English:** The system looks for files named with today's date. Rename your files or use the right date.

---

#### Issue: "LLM timeout" or "Connection refused to Ollama"
```
Error: Failed to connect to localhost:11434
```

**Diagnosis:**
```bash
curl http://localhost:11434/api/tags  # Is Ollama running?
docker logs ollama  # Any errors?
```

**Solution:**
```bash
# Start Ollama
docker-compose up -d ollama

# Pull model
docker exec -it ollama ollama pull llama3.1:8b

# Test
curl http://localhost:11434/api/generate -d '{"model":"llama3.1:8b","prompt":"Hello"}'
```

**Or use mock mode (no LLM needed):**
```bash
export LLM_PROVIDER=mock
python run_demo.py
```

**Plain English:** The AI server isn't running or doesn't have the model downloaded.

---

#### Issue: "No citations in generated narrative"
```
ValidationError: Narrative missing required citations
```

**Diagnosis:** LLM didn't follow the prompt instructions.

**Solution:**
1. Check `policies/` has relevant documents
2. Try a larger model: `llama3.1:70b` instead of `8b`
3. Check prompt in `src/narrative/generator.py`

**Plain English:** The AI didn't cite its sources. Either policies are missing or the model is too small to follow complex instructions.

---

## 9. Summary: The 30-Second Explanation

```
┌─────────────────────────────────────────────────────────────────┐
│                    HOW THIS SYSTEM WORKS                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. YOUR DATA (CSV) → 2. MATH (Python) → 3. FACTS (PostgreSQL) │
│                            │                                    │
│                            │ Numbers are calculated here        │
│                            │ (8.5% > 5% = BREACH)              │
│                            │ NO AI INVOLVED                     │
│                            ▼                                    │
│  4. SEARCH (RAG) ────────────────────────────────────────────► │
│     Find relevant                                               │
│     policies                                                    │
│            │                                                    │
│            ▼                                                    │
│  5. AI WRITES ──► 6. PDF OUTPUT                                │
│     "Based on policy XYZ,    compliance_report.pdf             │
│     the position exceeded                                       │
│     limits by 3.5%"                                            │
│                                                                 │
│  ═══════════════════════════════════════════════════════════   │
│                                                                 │
│  KEY INSIGHT: AI only writes PROSE, never does MATH            │
│  Every number comes from your systems, not AI hallucination    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 10. Files Quick Reference

```
Crosby_Efficient_Infra/
├── data/
│   ├── commodity_positions_20260117.csv    ← Your position data
│   ├── commodity_controls_20260117.csv     ← Your control results
│   └── real_commodities/                   ← Real price data from EIA/CFTC
│
├── policies/
│   ├── commodity_trading.md                ← RAG will cite this
│   ├── concentration_limits.md             
│   └── exposure_limits.md                  
│
├── src/
│   ├── integration/
│   │   ├── client_adapter.py               ← Reads your CSV files
│   │   ├── postgres_adapter.py             ← Talks to PostgreSQL
│   │   ├── llm_config.py                   ← Configures Ollama/Claude
│   │   └── rag_pipeline.py                 ← Main RAG logic
│   │
│   ├── control_runner/
│   │   └── runner.py                       ← Executes compliance checks
│   │
│   ├── retrieval/
│   │   └── retriever.py                    ← Finds relevant policies
│   │
│   ├── narrative/
│   │   └── generator.py                    ← LLM generates text
│   │
│   └── orchestrator.py                     ← Coordinates everything
│
├── quick_start.py                          ← Run this first
├── demo_data.py                            ← Analyze your data
├── run_demo.py                             ← Full pipeline test
└── docker-compose.yml                      ← Start all services
```

---

**Document End**

*For questions: Review the source files referenced in each section. Every component is documented inline.*
