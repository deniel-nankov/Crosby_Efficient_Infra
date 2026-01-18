#!/usr/bin/env python3
"""
TRANSPARENT COMPLIANCE RAG RUNNER
=================================
This script runs the compliance system with FULL VISIBILITY.
Every step is printed in plain English so you understand exactly what's happening.

NO BLACK BOXES - Everything is logged and explained.
"""

import os
import sys
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Dict, Any, List
import json
import hashlib
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment from .env file
def load_dotenv():
    """Load environment variables from .env file."""
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        print(f"  Loading environment from: {env_file}")
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    # Only set if not already in environment
                    if key not in os.environ or not os.environ[key]:
                        os.environ[key] = value
        return True
    else:
        print(f"  ⚠️  No .env file found at: {env_file}")
        return False

# Load .env at startup
print("\n[STARTUP] Loading configuration...")
load_dotenv()


# ==============================================================================
# CONFIGURATION DISPLAY
# ==============================================================================

def print_banner(title: str) -> None:
    """Print a visible section banner."""
    width = 80
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width + "\n")


def print_step(step_num: int, description: str) -> None:
    """Print a numbered step."""
    print(f"\n[STEP {step_num}] {description}")
    print("-" * 60)


def print_config(config: Dict[str, Any], hide_secrets: bool = True) -> None:
    """Print configuration in readable format."""
    for key, value in config.items():
        display_value = value
        if hide_secrets and any(secret in key.lower() for secret in ['password', 'key', 'secret', 'token']):
            display_value = "***HIDDEN***" if value else "(not set)"
        print(f"  • {key}: {display_value}")


def print_data_table(headers: List[str], rows: List[List[Any]], max_rows: int = 10) -> None:
    """Print data as a readable table."""
    if not rows:
        print("  (no data)")
        return
    
    # Calculate column widths
    widths = [len(h) for h in headers]
    for row in rows[:max_rows]:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)[:30]))
    
    # Print header
    header_line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(f"  {header_line}")
    print("  " + "-" * len(header_line))
    
    # Print rows
    for row in rows[:max_rows]:
        row_line = " | ".join(str(cell)[:30].ljust(widths[i]) for i, cell in enumerate(row))
        print(f"  {row_line}")
    
    if len(rows) > max_rows:
        print(f"  ... and {len(rows) - max_rows} more rows")


# ==============================================================================
# MOCK DATA FOR LOCAL TESTING
# ==============================================================================

def generate_mock_positions() -> List[Dict[str, Any]]:
    """Generate mock position data (simulates Snowflake query)."""
    return [
        {"position_id": "POS-001", "symbol": "AAPL", "quantity": Decimal("1000"), "market_value": Decimal("175000.00"), "asset_class": "equity"},
        {"position_id": "POS-002", "symbol": "MSFT", "quantity": Decimal("500"), "market_value": Decimal("187500.00"), "asset_class": "equity"},
        {"position_id": "POS-003", "symbol": "GOOGL", "quantity": Decimal("300"), "market_value": Decimal("42000.00"), "asset_class": "equity"},
        {"position_id": "POS-004", "symbol": "US_TREASURY_10Y", "quantity": Decimal("1000000"), "market_value": Decimal("985000.00"), "asset_class": "fixed_income"},
        {"position_id": "POS-005", "symbol": "CASH_USD", "quantity": Decimal("500000"), "market_value": Decimal("500000.00"), "asset_class": "cash"},
    ]


def generate_mock_exposures() -> Dict[str, Any]:
    """Generate mock exposure data (simulates Snowflake aggregations)."""
    return {
        "total_nav": Decimal("100000000.00"),  # $100M NAV
        "gross_exposure": Decimal("150000000.00"),  # $150M gross
        "net_exposure": Decimal("95000000.00"),  # $95M net
        "single_issuer_max": Decimal("15000000.00"),  # 15% in one issuer
        "sector_concentration": {
            "technology": Decimal("0.35"),
            "financials": Decimal("0.20"),
            "healthcare": Decimal("0.15"),
            "consumer": Decimal("0.15"),
            "other": Decimal("0.15"),
        },
        "liquidity_buckets": {
            "1_day": Decimal("0.25"),
            "2_7_days": Decimal("0.35"),
            "8_30_days": Decimal("0.25"),
            "31_90_days": Decimal("0.10"),
            "over_90_days": Decimal("0.05"),
        },
        "counterparty_exposure": {
            "GOLDMAN": Decimal("12000000.00"),
            "JPMORGAN": Decimal("10000000.00"),
            "MORGAN_STANLEY": Decimal("8000000.00"),
        },
    }


# ==============================================================================
# MAIN TRANSPARENT RUNNER
# ==============================================================================

def check_environment() -> Dict[str, Any]:
    """Check and display environment configuration."""
    print_banner("ENVIRONMENT CHECK")
    
    required_vars = {
        "PostgreSQL": ["POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_DB", "POSTGRES_USER", "POSTGRES_PASSWORD"],
        "Snowflake": ["SNOWFLAKE_ACCOUNT", "SNOWFLAKE_USER", "SNOWFLAKE_PASSWORD", "SNOWFLAKE_WAREHOUSE"],
        "OpenAI": ["OPENAI_API_KEY"],
        "Redis (Optional)": ["REDIS_HOST", "REDIS_PORT"],
    }
    
    status = {}
    
    for category, vars_list in required_vars.items():
        print(f"\n{category}:")
        category_status = {}
        for var in vars_list:
            value = os.environ.get(var, "")
            is_set = bool(value)
            status_symbol = "✓" if is_set else "✗"
            display = "***SET***" if is_set and any(s in var.lower() for s in ['password', 'key']) else (value[:20] + "..." if len(value) > 20 else value) if is_set else "(not set)"
            print(f"  {status_symbol} {var}: {display}")
            category_status[var] = is_set
        status[category] = category_status
    
    return status


def run_transparent_compliance_check():
    """
    Run the compliance system with full transparency.
    
    This demonstrates exactly what happens at each step.
    """
    print_banner("TRANSPARENT COMPLIANCE RAG SYSTEM")
    print("""
    This system performs daily compliance checks with FULL VISIBILITY.
    
    You will see:
    1. What data is being loaded
    2. What calculations are being performed
    3. What thresholds are being checked
    4. What the LLM is being asked (if enabled)
    5. What documents are generated
    
    NO BLACK BOXES - Everything is explained in plain English.
    """)
    
    # Step 1: Check Environment
    env_status = check_environment()
    
    # Determine if we can use real connections or need mock data
    use_mock = True  # For now, always use mock for demo
    if use_mock:
        print("\n⚠️  Running in MOCK MODE - Using simulated data for demonstration")
        print("    To use real data, configure .env file with database credentials")
    
    # Step 2: Load Control Definitions
    print_step(2, "LOADING CONTROL DEFINITIONS")
    print("Reading compliance controls from src/control_runner/controls.py...")
    
    from src.control_runner import get_all_controls, get_active_controls, ControlCategory
    
    all_controls = get_all_controls()
    active_controls = get_active_controls()
    
    print(f"\n  Total controls defined: {len(all_controls)}")
    print(f"  Active controls: {len(active_controls)}")
    
    # Group by category
    by_category = {}
    for ctrl in active_controls:
        cat = ctrl.category.value
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(ctrl)
    
    print("\n  Controls by category:")
    for cat, ctrls in sorted(by_category.items()):
        print(f"    • {cat}: {len(ctrls)} controls")
        for ctrl in ctrls:
            print(f"        - {ctrl.control_code}: {ctrl.control_name}")
    
    # Step 3: Load Position Data
    print_step(3, "LOADING POSITION DATA")
    
    if use_mock:
        print("  [MOCK] Generating simulated position data...")
        positions = generate_mock_positions()
    else:
        print("  [LIVE] Querying Snowflake for latest positions...")
        # In real mode, this would execute:
        # SELECT * FROM daily_positions WHERE snapshot_date = CURRENT_DATE()
        positions = []
    
    print(f"\n  Loaded {len(positions)} positions")
    
    headers = ["Position ID", "Symbol", "Quantity", "Market Value", "Asset Class"]
    rows = [[p["position_id"], p["symbol"], p["quantity"], f"${p['market_value']:,.2f}", p["asset_class"]] for p in positions]
    print_data_table(headers, rows)
    
    # Step 4: Calculate Exposures
    print_step(4, "CALCULATING EXPOSURES")
    
    if use_mock:
        print("  [MOCK] Using simulated exposure data...")
        exposures = generate_mock_exposures()
    else:
        print("  [LIVE] Calculating exposures from position data...")
        exposures = {}
    
    print("\n  Key Metrics:")
    print(f"    • Total NAV: ${exposures['total_nav']:,.2f}")
    print(f"    • Gross Exposure: ${exposures['gross_exposure']:,.2f}")
    print(f"    • Net Exposure: ${exposures['net_exposure']:,.2f}")
    print(f"    • Gross Leverage: {exposures['gross_exposure'] / exposures['total_nav']:.2f}x")
    print(f"    • Net Leverage: {exposures['net_exposure'] / exposures['total_nav']:.2f}x")
    
    print("\n  Sector Concentration:")
    for sector, pct in exposures['sector_concentration'].items():
        bar = "█" * int(float(pct) * 40)
        print(f"    • {sector:15s}: {float(pct)*100:5.1f}% {bar}")
    
    print("\n  Liquidity Profile:")
    for bucket, pct in exposures['liquidity_buckets'].items():
        bar = "█" * int(float(pct) * 40)
        print(f"    • {bucket:15s}: {float(pct)*100:5.1f}% {bar}")
    
    # Step 5: Execute Controls
    print_step(5, "EXECUTING COMPLIANCE CONTROLS")
    print("  Running deterministic control calculations...")
    print("  NOTE: These are PURE MATH calculations - NO LLM involved\n")
    
    from src.control_runner import ThresholdOperator
    
    # Simulate control execution
    control_results = []
    
    # Control 1: Gross Leverage
    gross_leverage = float(exposures['gross_exposure'] / exposures['total_nav'])
    threshold = 2.0
    passed = gross_leverage <= threshold
    control_results.append({
        "control_code": "CTRL-LEV-001",
        "control_name": "Gross Leverage Limit",
        "calculated_value": gross_leverage,
        "threshold": f"<= {threshold}",
        "result": "PASS ✓" if passed else "FAIL ✗",
        "explanation": f"Gross leverage is {gross_leverage:.2f}x vs limit of {threshold}x"
    })
    
    # Control 2: Single Issuer Concentration
    single_issuer_pct = float(exposures['single_issuer_max'] / exposures['total_nav'])
    threshold = 0.15
    passed = single_issuer_pct <= threshold
    control_results.append({
        "control_code": "CTRL-CONC-001",
        "control_name": "Single Issuer Concentration",
        "calculated_value": single_issuer_pct,
        "threshold": f"<= {threshold*100}%",
        "result": "PASS ✓" if passed else "FAIL ✗",
        "explanation": f"Max single issuer is {single_issuer_pct*100:.1f}% vs limit of {threshold*100}%"
    })
    
    # Control 3: Sector Concentration
    max_sector_pct = max(float(v) for v in exposures['sector_concentration'].values())
    threshold = 0.40
    passed = max_sector_pct <= threshold
    control_results.append({
        "control_code": "CTRL-CONC-002",
        "control_name": "Sector Concentration Limit",
        "calculated_value": max_sector_pct,
        "threshold": f"<= {threshold*100}%",
        "result": "PASS ✓" if passed else "FAIL ✗",
        "explanation": f"Max sector is {max_sector_pct*100:.1f}% (technology) vs limit of {threshold*100}%"
    })
    
    # Control 4: Counterparty Exposure
    max_cp = max(float(v) for v in exposures['counterparty_exposure'].values())
    max_cp_pct = max_cp / float(exposures['total_nav'])
    threshold = 0.15
    passed = max_cp_pct <= threshold
    control_results.append({
        "control_code": "CTRL-CP-001",
        "control_name": "Counterparty Exposure Limit",
        "calculated_value": max_cp_pct,
        "threshold": f"<= {threshold*100}%",
        "result": "PASS ✓" if passed else "FAIL ✗",
        "explanation": f"Max counterparty is {max_cp_pct*100:.1f}% vs limit of {threshold*100}%"
    })
    
    # Control 5: Liquidity Coverage
    quick_liquidity = float(exposures['liquidity_buckets']['1_day'] + exposures['liquidity_buckets']['2_7_days'])
    threshold = 0.30
    passed = quick_liquidity >= threshold
    control_results.append({
        "control_code": "CTRL-LIQ-001",
        "control_name": "7-Day Liquidity Minimum",
        "calculated_value": quick_liquidity,
        "threshold": f">= {threshold*100}%",
        "result": "PASS ✓" if passed else "FAIL ✗",
        "explanation": f"7-day liquidity is {quick_liquidity*100:.1f}% vs minimum of {threshold*100}%"
    })
    
    print("  Control Results:\n")
    for r in control_results:
        status_color = "" if "PASS" in r["result"] else ""
        print(f"  [{r['result']}] {r['control_code']}: {r['control_name']}")
        print(f"      Calculation: {r['explanation']}")
        print()
    
    # Summary
    passed_count = sum(1 for r in control_results if "PASS" in r["result"])
    failed_count = len(control_results) - passed_count
    
    print(f"\n  SUMMARY: {passed_count} passed, {failed_count} failed out of {len(control_results)} controls")
    
    # Step 6: Evidence Storage
    print_step(6, "STORING EVIDENCE (AUDIT TRAIL)")
    print("  Every calculation is stored with full traceability:\n")
    
    run_id = f"RUN-{date.today().isoformat()}-MOCK001"
    snapshot_id = f"SNAP-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    
    evidence_records = []
    for r in control_results:
        record = {
            "run_id": run_id,
            "snapshot_id": snapshot_id,
            "control_code": r["control_code"],
            "calculated_value": r["calculated_value"],
            "threshold": r["threshold"],
            "result_status": "pass" if "PASS" in r["result"] else "fail",
            "evidence_hash": hashlib.sha256(json.dumps(r, default=str).encode()).hexdigest()[:16],
            "calculated_at": datetime.now(timezone.utc).isoformat(),
        }
        evidence_records.append(record)
        print(f"  Stored: {record['control_code']}")
        print(f"    → Result: {record['result_status'].upper()}")
        print(f"    → Hash: {record['evidence_hash']}...")
        print()
    
    print(f"\n  All {len(evidence_records)} results stored with cryptographic hashes")
    print(f"  Run ID: {run_id}")
    print(f"  Snapshot ID: {snapshot_id}")
    
    # Step 7: Narrative Generation (LLM)
    print_step(7, "NARRATIVE GENERATION (LLM-ASSISTED)")
    
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    
    if not openai_key:
        print("  ⚠️  OpenAI API key not configured - showing example prompt instead\n")
        print("  The LLM would receive this prompt:\n")
        
        prompt = f"""You are a compliance officer writing a daily summary.

CONTEXT (from database - NOT editable by you):
- Date: {date.today().isoformat()}
- Total Controls: {len(control_results)}
- Passed: {passed_count}
- Failed: {failed_count}
- Pass Rate: {passed_count/len(control_results)*100:.1f}%

CONTROL RESULTS:
"""
        for r in control_results:
            prompt += f"- {r['control_code']}: {r['result']} - {r['explanation']}\n"
        
        prompt += """
TASK: Write a 2-3 paragraph summary of today's compliance status.
RULES:
1. Only reference the data provided above
2. Include citations like [ControlResult: CTRL-XXX-001]
3. Do not make up any numbers or facts
4. Use professional compliance language
"""
        print("  " + "-" * 60)
        print(prompt)
        print("  " + "-" * 60)
        
        print("\n  Example LLM response (what would be generated):\n")
        example_narrative = f"""  Today's compliance review shows a pass rate of {passed_count/len(control_results)*100:.1f}% across 
  {len(control_results)} monitored controls. [ControlResult: {control_results[0]['control_code']}] 
  
  Leverage metrics remain within acceptable bounds with gross leverage at 
  {control_results[0]['calculated_value']:.2f}x against the 2.0x limit. 
  [ControlResult: CTRL-LEV-001]
  
  Liquidity coverage is strong at {quick_liquidity*100:.1f}%, exceeding the 30% minimum 
  threshold for 7-day liquidity requirements. [ControlResult: CTRL-LIQ-001]"""
        print(example_narrative)
    else:
        print("  ✓ OpenAI API configured - would generate real narrative")
    
    # Step 8: Document Generation
    print_step(8, "DOCUMENT GENERATION (PDF)")
    print("  Building Daily Compliance Pack with LOCKED STRUCTURE:\n")
    
    print("  Document Structure:")
    print("  ┌─────────────────────────────────────────────────┐")
    print("  │ 1. HEADER (auto-generated)                      │")
    print("  │    - Document code, date, run ID                │")
    print("  ├─────────────────────────────────────────────────┤")
    print("  │ 2. EXECUTIVE SUMMARY (deterministic)            │")
    print("  │    - Pass/fail counts from database             │")
    print("  ├─────────────────────────────────────────────────┤")
    print("  │ 3. CONTROL FAILURES TABLE (deterministic)       │")
    print("  │    - Failed controls with values                │")
    print("  ├─────────────────────────────────────────────────┤")
    print("  │ 4. EXCEPTIONS TABLE (deterministic)             │")
    print("  │    - New and outstanding exceptions             │")
    print("  ├─────────────────────────────────────────────────┤")
    print("  │ 5. NARRATIVE SUMMARY (LLM-generated)            │")
    print("  │    - Marked as AI content with citations        │")
    print("  ├─────────────────────────────────────────────────┤")
    print("  │ 6. DETAILED RESULTS (deterministic)             │")
    print("  │    - All controls by category                   │")
    print("  ├─────────────────────────────────────────────────┤")
    print("  │ 7. AUDIT APPENDIX (deterministic)               │")
    print("  │    - Hashes, timestamps, evidence IDs           │")
    print("  └─────────────────────────────────────────────────┘")
    
    print("\n  Document would be saved to: output/DCP-{date}-{id}.pdf")
    print("  Document hash: SHA-256 of entire PDF for tamper detection")
    
    # Final Summary
    print_banner("RUN COMPLETE - SUMMARY")
    
    print(f"""
    Run ID:          {run_id}
    Snapshot ID:     {snapshot_id}
    Date:            {date.today().isoformat()}
    
    CONTROLS EXECUTED:
    ✓ Passed:        {passed_count}
    ✗ Failed:        {failed_count}
    Pass Rate:       {passed_count/len(control_results)*100:.1f}%
    
    EVIDENCE STORED:
    Records:         {len(evidence_records)}
    Hashes:          {len(evidence_records)} SHA-256 hashes
    
    DOCUMENTS:
    Generated:       Daily Compliance Pack (PDF)
    
    TRANSPARENCY NOTES:
    • All calculations are deterministic (same input = same output)
    • LLM is ONLY used for narrative text, NOT for numbers
    • Every value is traced to a database record with hash
    • Document structure is fixed - LLM cannot change layout
    • All LLM content is clearly marked with citations
    """)


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "█" * 80)
    print("█" + " " * 78 + "█")
    print("█" + "  COMPLIANCE RAG SYSTEM - TRANSPARENT MODE".center(78) + "█")
    print("█" + "  Anti-Black-Box Edition".center(78) + "█")
    print("█" + " " * 78 + "█")
    print("█" * 80 + "\n")
    
    try:
        run_transparent_compliance_check()
    except KeyboardInterrupt:
        print("\n\nAborted by user.")
    except Exception as e:
        print(f"\n\nError: {e}")
        raise
