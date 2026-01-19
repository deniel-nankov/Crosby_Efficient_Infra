#!/usr/bin/env python3
"""
=============================================================================
DATABASE-BACKED COMPLIANCE PIPELINE WITH TRUE RAG
=============================================================================

This script demonstrates the PRODUCTION workflow:
1. Connect to PostgreSQL database (simulating client's OMS/PMS database)
2. Load position and control data from database tables
3. Retrieve relevant policy context using RAG (vector similarity search)
4. Generate LLM-powered narratives grounded in actual policy text
5. Output PDF workpapers with full audit trail and citations

FOR REAL CLIENT DEPLOYMENT:
- Replace the connection string with client's database
- Map their table names/columns to the adapter
- Set LLM_PROVIDER and API key environment variables
- Run embed_policies.py first to populate vector store
- Deploy Docker stack in their infrastructure

RAG CONFIGURATION:
    First, embed policy documents:
        python -m src.rag.embedder
    
    Then run pipeline:
        LLM_PROVIDER=lmstudio python run_database_pipeline.py

LLM CONFIGURATION:
    Set environment variables:
        LLM_PROVIDER=lmstudio     # or: anthropic, openai, ollama, vllm
        ANTHROPIC_API_KEY=sk-...  # for Anthropic Claude
        OPENAI_API_KEY=sk-...     # for OpenAI GPT-4
    
    The system will:
    - Retrieve relevant policy sections using vector search
    - Ground LLM generation in actual policy text
    - Include proper citations with document references
    - Fall back to templates if no LLM/RAG available

=============================================================================
"""

import sys
import os
from datetime import date, datetime, timezone
from pathlib import Path
from decimal import Decimal

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

# Import LLM infrastructure
from integration.llm_config import get_compliance_llm, LLMConfig, LLMProvider

# =============================================================================
# CONFIGURATION - WHAT A CLIENT WOULD CUSTOMIZE
# =============================================================================

# In production, these come from environment variables or config file
DATABASE_CONFIG = {
    # Client would change these to their database
    "host": os.environ.get("DB_HOST", "localhost"),
    "port": int(os.environ.get("DB_PORT", "5433")),  # Dedicated compliance-postgres container
    "database": os.environ.get("DB_NAME", "compliance"),
    "user": os.environ.get("DB_USER", "compliance_user"),
    "password": os.environ.get("DB_PASSWORD", "compliance_dev_password_123"),
}

# Client's table mappings - customize to match their schema
TABLE_MAPPINGS = {
    "positions_table": "fund_positions",      # Their positions table name
    "controls_table": "fund_control_results", # Their control results table  
    "nav_table": "fund_nav",                  # Their NAV table
}

# Client's column mappings - if their columns have different names
COLUMN_MAPPINGS = {
    # Standard name: Client's column name
    "security_id": "security_id",
    "ticker": "ticker", 
    "security_name": "security_name",
    "quantity": "quantity",
    "market_value": "market_value",
    "sector": "sector",
    "issuer": "issuer",
    # etc.
}


def check_database_connection():
    """Verify database connectivity before running pipeline."""
    try:
        import psycopg2
        conn = psycopg2.connect(
            host=DATABASE_CONFIG["host"],
            port=DATABASE_CONFIG["port"],
            database=DATABASE_CONFIG["database"],
            user=DATABASE_CONFIG["user"],
            password=DATABASE_CONFIG["password"],
        )
        cursor = conn.cursor()
        cursor.execute("SELECT version()")
        version = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        return True, version
    except Exception as e:
        return False, str(e)


def setup_sample_data(as_of_date: date):
    """
    Load sample data into database for testing.
    
    IN PRODUCTION: This step is replaced by client's ETL process
    that loads their real data from Bloomberg/Eze/Geneva into these tables.
    """
    from integration.postgres_adapter import PostgresDataSource, PostgresConfig
    
    config = PostgresConfig(
        host=DATABASE_CONFIG["host"],
        port=DATABASE_CONFIG["port"],
        database=DATABASE_CONFIG["database"],
        user=DATABASE_CONFIG["user"],
        password=DATABASE_CONFIG["password"],
    )
    
    source = PostgresDataSource(config)
    source.create_data_tables()
    source.load_sample_data(as_of_date)
    source.close()
    
    print(f"  Sample data loaded for {as_of_date}")


def run_database_pipeline(as_of_date: date, output_dir: Path):
    """
    Execute the full database-backed compliance pipeline.
    
    This is what runs in production every day.
    """
    from integration.postgres_adapter import PostgresAdapter, PostgresConfig
    from document_builder import DocumentBuilder
    from document_builder.professional_pdf import ProfessionalCompliancePDF
    
    # =========================================================================
    # STEP 1: Connect to Database and Get Data
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: Connecting to Database")
    print("=" * 70)
    
    config = PostgresConfig(
        host=DATABASE_CONFIG["host"],
        port=DATABASE_CONFIG["port"],
        database=DATABASE_CONFIG["database"],
        user=DATABASE_CONFIG["user"],
        password=DATABASE_CONFIG["password"],
    )
    
    adapter = PostgresAdapter(config)
    snapshot = adapter.get_snapshot(as_of_date=as_of_date)
    
    print(f"\n  Database: {DATABASE_CONFIG['host']}:{DATABASE_CONFIG['port']}/{DATABASE_CONFIG['database']}")
    print(f"  As-of Date: {as_of_date}")
    print(f"  Snapshot ID: {snapshot.snapshot_id}")
    print(f"  Positions loaded: {len(snapshot.positions)}")
    print(f"  Control results loaded: {len(snapshot.control_results)}")
    print(f"  NAV: ${snapshot.nav:,.0f}")
    
    # =========================================================================
    # STEP 2: Analyze Control Results
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: Control Results Analysis")
    print("=" * 70)
    
    passed = [c for c in snapshot.control_results if c.status == "pass"]
    warnings = [c for c in snapshot.control_results if c.status == "warning"]
    failed = [c for c in snapshot.control_results if c.status == "fail"]
    
    print(f"\n  Total Controls: {len(snapshot.control_results)}")
    print(f"  Passed: {len(passed)}")
    print(f"  Warnings: {len(warnings)}")
    print(f"  Failed: {len(failed)}")
    
    print("\n  Control Details:")
    for ctrl in snapshot.control_results:
        status_icon = {"pass": "[PASS]", "warning": "[WARN]", "fail": "[FAIL]"}.get(ctrl.status, "[????]")
        print(f"    {status_icon} {ctrl.control_id}: {ctrl.calculated_value}% vs {ctrl.threshold}% ({ctrl.threshold_operator})")
    
    # =========================================================================
    # STEP 3: Generate Narrative (LLM-powered or Template fallback)
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: Generating Compliance Narrative (RAG + LLM)")
    print("=" * 70)
    
    # Check for LLM availability and determine provider
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    llm_provider = os.environ.get("LLM_PROVIDER", "").lower()
    
    llm_available = bool(anthropic_key or openai_key or llm_provider in ("ollama", "vllm", "lmstudio"))
    
    # Check for RAG availability
    rag_context = None
    if llm_available:
        print("\n  Checking RAG (policy retrieval) availability...")
        retriever = get_rag_retriever()
        if retriever:
            print("  ✓ RAG enabled - retrieving relevant policy sections...")
            retrieved = retriever.retrieve_for_controls(snapshot.control_results)
            if retrieved.chunks:
                print(f"    Retrieved {len(retrieved.chunks)} policy chunks (~{retrieved.total_tokens_estimate} tokens)")
                for chunk in retrieved.chunks[:3]:  # Show first 3
                    similarity = getattr(chunk, 'similarity', 0)
                    print(f"    - {chunk.document_name} | {chunk.section_title} ({similarity:.0%} match)")
                rag_context = retrieved.to_prompt_context()
            else:
                print("    No relevant policy chunks found")
        else:
            print("  ⚠ RAG not available - run 'python -m src.rag.embedder' to enable")
    
    if llm_available:
        if anthropic_key:
            provider_info = "Anthropic Claude"
        elif openai_key:
            provider_info = "OpenAI GPT-4"
        elif llm_provider == "lmstudio":
            provider_info = "LM Studio (local)"
        elif llm_provider == "ollama":
            provider_info = "Ollama (local)"
        elif llm_provider == "vllm":
            provider_info = "vLLM (local)"
        else:
            provider_info = "configured LLM"
        
        print(f"\n  ✓ LLM: {provider_info}")
        rag_status = "with RAG context" if rag_context else "without RAG"
        print(f"  Generating AI-powered narrative ({rag_status})...")
        narrative = generate_llm_narrative(snapshot, rag_context)
    else:
        print("\n  ⚠ No LLM configured - using template-based narrative")
        print("  Set LLM_PROVIDER=lmstudio for AI-powered narratives")
        narrative = generate_template_narrative(snapshot)
    
    print("\n  Narrative preview (first 500 chars):")
    print(f"  {narrative[:500]}...")
    
    # =========================================================================
    # STEP 4: Build PDF Report
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 4: Generating PDF Report")
    print("=" * 70)
    
    # Prepare data structures for document builder
    run_summary = {
        "run_id": f"RUN-{snapshot.as_of_date.isoformat()}-001",
        "run_code": f"DAILY-{snapshot.as_of_date.isoformat()}-001",
        "snowflake_snapshot_id": snapshot.snapshot_id,
        "snowflake_snapshot_ts": snapshot.extracted_at.isoformat(),
        "total_controls": len(snapshot.control_results),
        "controls_passed": len(passed),
        "controls_failed": len(failed),
        "controls_warning": len(warnings),
        "config_hash": snapshot.data_hash,
        "executor_service": "database-pipeline-v1.0",
        "executor_version": "1.0.0",
        "run_timestamp_start": datetime.now(timezone.utc).isoformat(),
        "run_timestamp_end": datetime.now(timezone.utc).isoformat(),
    }
    
    control_results_dict = [
        {
            "result_id": f"CR-{i+1:03d}",
            "control_code": c.control_id,
            "control_name": c.control_name,
            "control_category": c.control_type,
            "calculated_value": float(c.calculated_value),
            "threshold_value": float(c.threshold),
            "threshold_operator": c.threshold_operator,
            "result_status": c.status,
            "breach_amount": float(c.breach_amount) if c.breach_amount else None,
        }
        for i, c in enumerate(snapshot.control_results)
    ]
    
    exceptions = [
        {
            "exception_id": f"EX-{i+1:03d}",
            "exception_code": f"EXC-{snapshot.as_of_date.isoformat()}-{i+1:03d}",
            "control_code": c.control_id,
            "severity": "critical" if c.status == "fail" else "high",
            "title": f"{c.control_name} - {'Breach' if c.status == 'fail' else 'Warning'}",
            "status": "open",
            "due_date": str(snapshot.as_of_date),
        }
        for i, c in enumerate(snapshot.control_results) if c.status != "pass"
    ]
    
    narrative_metadata = {
        "narrative_id": f"NAR-{snapshot.as_of_date.isoformat()}-001",
        "model_id": "llm" if llm_available else "template",
        "model_version": "1.0",
        "tokens_used": 0,
    }
    
    # Build Professional PDF
    output_dir.mkdir(exist_ok=True)
    
    # Convert positions and controls to format expected by professional PDF
    # Include all relevant data fields from the source
    positions_for_pdf = [
        {
            "security_id": p.security_id,
            "ticker": p.ticker,
            "security_name": p.security_name,
            "quantity": float(p.quantity),
            "market_value": float(p.market_value),
            "sector": p.sector or p.asset_class,
            "issuer": p.issuer,
            "asset_class": p.asset_class,
        }
        for p in snapshot.positions
    ]
    
    controls_for_pdf = [
        {
            "control_id": c.control_id,
            "control_name": c.control_name,
            "control_type": c.control_type,  # Category: concentration, liquidity, etc.
            "current_value": float(c.calculated_value) / 100,  # Convert to decimal for %
            "threshold": f"{c.threshold_operator} {c.threshold}%",
            "threshold_value": float(c.threshold),
            "status": c.status,
            "breach_amount": float(c.breach_amount) if c.breach_amount else None,
        }
        for c in snapshot.control_results
    ]
    
    # Generate professional PDF
    pdf_builder = ProfessionalCompliancePDF(
        fund_name="Crosby Capital Management",
        fund_id="CCM-2026-001",
        confidentiality="CONFIDENTIAL - FOR INTERNAL USE ONLY"
    )
    
    pdf_path = output_dir / f"db_compliance_report_{snapshot.as_of_date.isoformat()}.pdf"
    pdf_bytes = pdf_builder.generate_daily_compliance_report(
        report_date=snapshot.as_of_date,
        nav=float(snapshot.nav),
        positions=positions_for_pdf,
        control_results=controls_for_pdf,
        narrative=narrative,
        snapshot_id=snapshot.snapshot_id,
        document_id=f"DCP-{snapshot.as_of_date.isoformat()}-001",
    )
    
    # Save PDF
    with open(pdf_path, 'wb') as f:
        f.write(pdf_bytes)
    
    import hashlib
    doc_hash = hashlib.sha256(pdf_bytes).hexdigest()
    
    print(f"\n  Professional PDF Generated: {pdf_path}")
    print(f"  Size: {len(pdf_bytes):,} bytes")
    print(f"  Document Hash: {doc_hash[:16]}...")
    print(f"  Style: Institutional Finance Grade")
    
    # =========================================================================
    # STEP 5: Audit Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("STEP 5: Audit Trail Summary")
    print("=" * 70)
    
    document_id = f"DCP-{snapshot.as_of_date.isoformat()}-001"
    print(f"""
  Document ID: {document_id}
  Document Code: {document_id}
  Generated At: {datetime.now(timezone.utc).isoformat()}
  Data Source: {snapshot.source_system}
  Snapshot ID: {snapshot.snapshot_id}
  Data Hash: {snapshot.data_hash}
  Document Hash: {doc_hash[:32]}...
  PDF Format: Professional Finance Grade
""")
    
    # Cleanup
    adapter.close()
    
    return pdf_path


def generate_template_narrative(snapshot) -> str:
    """Generate narrative using templates (no LLM required)."""
    
    passed = sum(1 for c in snapshot.control_results if c.status == "pass")
    warnings = [c for c in snapshot.control_results if c.status == "warning"]
    failed = [c for c in snapshot.control_results if c.status == "fail"]
    
    narrative = f"""Daily Compliance Review - {snapshot.as_of_date}

Executive Summary:
Today's compliance review tested {len(snapshot.control_results)} controls against the fund's 
investment guidelines and regulatory requirements. {passed} controls passed within 
acceptable thresholds.
"""
    
    if warnings:
        narrative += f"\nWarning Items ({len(warnings)}):\n"
        for w in warnings:
            narrative += f"""
- {w.control_name}: Current value of {w.calculated_value}% is approaching the 
  {w.threshold}% threshold. Monitoring recommended.
  [Policy: investment_guidelines.md | Section: {w.control_type}]
"""
    
    if failed:
        narrative += f"\nBreaches Requiring Action ({len(failed)}):\n"
        for f in failed:
            narrative += f"""
- {f.control_name}: Current value of {f.calculated_value}% exceeds the 
  {f.threshold}% limit by {f.breach_amount}%. Immediate remediation required.
  [Policy: investment_guidelines.md | Section: {f.control_type}]
"""
    
    if not warnings and not failed:
        narrative += "\nAll controls passed. No exceptions to report.\n"
    
    narrative += f"""
Data Source: {snapshot.source_system}
Snapshot ID: {snapshot.snapshot_id}
NAV as of date: ${snapshot.nav:,.0f}
"""
    
    return narrative


def get_rag_retriever():
    """Initialize RAG retriever if available."""
    try:
        from rag.vector_store import VectorStore
        from rag.embedder import LocalEmbedder
        from rag.retriever import RAGRetriever
        
        vector_store = VectorStore(DATABASE_CONFIG)
        embedder = LocalEmbedder()
        retriever = RAGRetriever(vector_store, embedder)
        
        if retriever.is_available():
            return retriever
        return None
    except Exception as e:
        print(f"    Note: RAG not available ({e})")
        return None


def generate_llm_narrative(snapshot, rag_context: str = None) -> str:
    """
    Generate narrative using LLM with RAG-retrieved policy context.
    
    This function:
    1. Receives pre-retrieved policy context from RAG
    2. Prepares control data in a structured format
    3. Sends context + data to the LLM
    4. Receives narrative grounded in actual policy text
    
    The LLM is prompted with SEC-compliant system instructions to:
    - Only use facts from provided evidence AND retrieved policies
    - Include citations referencing the actual policy documents
    - Use professional compliance language
    """
    try:
        llm = get_compliance_llm()
        
        # Build context for LLM
        passed = [c for c in snapshot.control_results if c.status == "pass"]
        warnings = [c for c in snapshot.control_results if c.status == "warning"]
        failed = [c for c in snapshot.control_results if c.status == "fail"]
        
        # Build control summary
        control_summary = f"""
COMPLIANCE CONTROL RESULTS - {snapshot.as_of_date}
================================================================================

Total Controls: {len(snapshot.control_results)}
Passed: {len(passed)}
Warnings: {len(warnings)}
Failures: {len(failed)}

NAV: ${snapshot.nav:,.0f}
Total Positions: {len(snapshot.positions)}
Data Source: {snapshot.source_system}
Snapshot ID: {snapshot.snapshot_id}
"""
        
        if failed:
            control_summary += "\n\nFAILED CONTROLS (require immediate action):\n"
            for f in failed:
                control_summary += f"""
- Control: {f.control_name}
  Type: {f.control_type}
  Calculated Value: {f.calculated_value}%
  Threshold: {f.threshold}%
  Breach Amount: {f.breach_amount}%
  Status: FAIL
"""
        
        if warnings:
            control_summary += "\n\nWARNING CONTROLS (approaching thresholds):\n"
            for w in warnings:
                control_summary += f"""
- Control: {w.control_name}
  Type: {w.control_type}
  Calculated Value: {w.calculated_value}%
  Threshold: {w.threshold}%
  Distance to Threshold: {w.threshold - w.calculated_value:.2f}%
  Status: WARNING
"""
        
        if not warnings and not failed:
            control_summary += "\n\nAll controls passed within acceptable thresholds.\n"
        
        # SEC-compliant system prompt
        system_prompt = """You are a compliance documentation assistant for an SEC-registered hedge fund.
Your role is to generate clear, accurate, and well-cited compliance narratives.

CRITICAL RULES:
1. ONLY use information from the CONTROL RESULTS and POLICY CONTEXT provided
2. NEVER invent, assume, or hallucinate any facts, numbers, or statements
3. Include inline citations for EVERY factual statement referencing the source policy
4. Use the exact policy document names and sections from the retrieved context
5. DO NOT perform any calculations - all numbers must come from the evidence
6. Be concise but complete - 2-3 paragraphs maximum

Your output will be reviewed by the Chief Compliance Officer and may be examined by SEC regulators.
Accuracy and traceability are paramount."""

        # User prompt with RAG context
        if rag_context:
            user_prompt = f"""Generate a daily compliance summary narrative based on the control results AND the retrieved policy context.

{control_summary}

{rag_context}

INSTRUCTIONS:
1. Start with an executive summary of today's compliance status
2. For each warning or breach, reference the SPECIFIC policy section from the retrieved context
3. Use citations in format [Policy: document_name.md | Section: section_title]
4. End with the data source and snapshot information for audit trail

Generate the professional compliance narrative grounded in the policy documents:"""
        else:
            user_prompt = f"""Generate a daily compliance summary narrative based on the following control run results.

{control_summary}

INSTRUCTIONS:
1. Start with an executive summary of today's compliance status
2. Highlight any breaches or warnings with their severity
3. Reference the applicable policy for each issue
4. End with the data source and snapshot information for audit trail
5. Use citations in format [Policy: investment_guidelines.md | Section: control_type]

Generate the professional compliance narrative now:"""

        # Generate with anonymization (data is auto-anonymized, then restored)
        narrative = llm.generate(user_prompt, system_prompt)
        
        # Add model attribution for audit trail
        narrative += f"\n\n---\n[Generated by {llm.model_id} | {datetime.now(timezone.utc).isoformat()}]"
        
        return narrative
        
    except Exception as e:
        # Fall back to template if LLM fails
        print(f"⚠ LLM generation failed: {e}")
        print("  Falling back to template-based narrative...")
        return generate_template_narrative(snapshot) + "\n\n[Template-based: LLM unavailable]"


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Database-Backed Compliance Pipeline")
    parser.add_argument("--date", type=str, help="Report date (YYYY-MM-DD)", default=None)
    parser.add_argument("--setup-sample", action="store_true", help="Load sample data first")
    parser.add_argument("--output", type=str, help="Output directory", default="output")
    
    args = parser.parse_args()
    
    # Determine date
    report_date = date.fromisoformat(args.date) if args.date else date.today()
    output_dir = Path(args.output)
    
    print("=" * 70)
    print("DATABASE-BACKED COMPLIANCE PIPELINE")
    print("=" * 70)
    print(f"\nReport Date: {report_date}")
    print(f"Output Directory: {output_dir.absolute()}")
    
    # Check database connection
    print("\nChecking database connection...")
    connected, info = check_database_connection()
    
    if not connected:
        print(f"\n[ERROR] Cannot connect to database: {info}")
        print("\nTo start the database, run:")
        print("  docker-compose up -d postgres")
        print("\nOr configure these environment variables:")
        print("  DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD")
        return 1
    
    print(f"  Connected! PostgreSQL version: {info[:50]}...")
    
    # Optionally load sample data
    if args.setup_sample:
        print("\nLoading sample data...")
        setup_sample_data(report_date)
    
    # Run pipeline
    try:
        pdf_path = run_database_pipeline(report_date, output_dir)
        
        print("\n" + "=" * 70)
        print("PIPELINE COMPLETE")
        print("=" * 70)
        print(f"\nOutput: {pdf_path}")
        print("\nThis pipeline can be scheduled via:")
        print("  - Airflow (see dags/daily_compliance_dag.py)")
        print("  - Cron job")
        print("  - Windows Task Scheduler")
        print("  - Any orchestration tool")
        
        return 0
        
    except ValueError as e:
        print(f"\n[ERROR] {e}")
        print("\nNo data found for this date. Run with --setup-sample to load test data:")
        print(f"  python run_database_pipeline.py --date {report_date} --setup-sample")
        return 1


if __name__ == "__main__":
    sys.exit(main())
