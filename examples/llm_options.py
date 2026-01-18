#!/usr/bin/env python3
"""
LLM Configuration Examples for Compliance RAG System

This script shows how to configure different LLM backends.
"""

print("""
================================================================================
LLM OPTIONS FOR $2B HEDGE FUND COMPLIANCE
================================================================================

RECOMMENDATION: Cloud API (Claude) with Data Anonymization
─────────────────────────────────────────────────────────────────────────────────

Why Cloud API for a $2B Fund:
✓ Superior prose quality for compliance documentation
✓ No GPU infrastructure costs ($50K+ for 70B model)
✓ Anthropic/OpenAI are SOC 2 Type II certified
✓ Data anonymization removes sensitive info BEFORE API call

What Gets Anonymized:
• Security tickers: AAPL → [SECURITY_1]
• Large dollar amounts: $2,500,000,000 → [AMOUNT_1]

What Does NOT Get Anonymized (needed for narrative):
• Percentages (28% concentration)
• Control types (liquidity, concentration)
• Policy text (already internal documents)

================================================================================
OPTION 1: CLAUDE API (Recommended)
================================================================================

Setup:
    export LLM_PROVIDER=anthropic
    export ANTHROPIC_API_KEY=sk-ant-api03-...
    export LLM_MODEL=claude-sonnet-4-20250514

Python:
    from src.integration import get_compliance_llm
    
    llm = get_compliance_llm()
    narrative = llm.generate(
        "Generate compliance narrative for sector concentration at 28%..."
    )

Cost Estimate:
    • ~$0.003 per 1K input tokens
    • ~$0.015 per 1K output tokens
    • Daily compliance run (~50 narratives): ~$0.50/day = ~$180/year

================================================================================
OPTION 2: OPENAI GPT-4 API
================================================================================

Setup:
    export LLM_PROVIDER=openai
    export OPENAI_API_KEY=sk-...
    export LLM_MODEL=gpt-4o

Python:
    from src.integration import LLMConfig, LLMProvider, ComplianceLLM
    
    config = LLMConfig(
        provider=LLMProvider.OPENAI,
        api_key="sk-...",
        model_id="gpt-4o",
        anonymize_data=True,
    )
    llm = ComplianceLLM(config)

================================================================================
OPTION 3: LOCAL LLM VIA OLLAMA (Maximum Privacy)
================================================================================

When to use:
• Regulatory requirement for on-premise data processing
• Extremely sensitive client data
• Air-gapped environment

Setup:
    # 1. Install Ollama
    curl -fsSL https://ollama.ai/install.sh | sh
    
    # 2. Pull model (requires 40GB+ RAM for 70B)
    ollama pull llama3.1:70b
    
    # 3. Configure
    export LLM_PROVIDER=ollama
    export LLM_MODEL=llama3.1:70b
    export LLM_API_BASE=http://localhost:11434

Hardware Requirements:
    • LLaMA 3.1 70B: 2x A100 80GB or 4x A100 40GB (~$50K)
    • LLaMA 3.1 8B: 1x RTX 4090 24GB (~$2K) - Lower quality

Python:
    from src.integration import LLMConfig, LLMProvider, ComplianceLLM
    
    config = LLMConfig(
        provider=LLMProvider.OLLAMA,
        model_id="llama3.1:70b",
        api_base="http://localhost:11434",
        anonymize_data=False,  # Not needed for local
    )
    llm = ComplianceLLM(config)

================================================================================
OPTION 4: LOCAL LLM VIA vLLM (High Performance)
================================================================================

Setup:
    # 1. Install vLLM
    pip install vllm
    
    # 2. Start server
    python -m vllm.entrypoints.openai.api_server \\
        --model meta-llama/Meta-Llama-3.1-70B-Instruct \\
        --tensor-parallel-size 2
    
    # 3. Configure
    export LLM_PROVIDER=vllm
    export LLM_MODEL=meta-llama/Meta-Llama-3.1-70B-Instruct
    export LLM_API_BASE=http://localhost:8000

================================================================================
SECURITY COMPARISON
================================================================================

                        Cloud API       Local LLM
                        (Claude)        (Ollama/vLLM)
─────────────────────────────────────────────────────
Data leaves network     Yes*            No
SOC 2 certified         Yes             N/A
SEC 17a-4 compliant     Via attestation Your responsibility
GPU costs               $0              $50K+
Quality                 Excellent       Good (70B) / Fair (8B)
Latency                 ~2s             ~5-10s (depending on HW)
Maintenance             None            ML Ops required

* With anonymization, only percentages and policy text leave network

================================================================================
MY RECOMMENDATION FOR A $2B FUND
================================================================================

USE CLAUDE API because:

1. Your compliance data is already partially public (SEC filings)
2. We anonymize tickers and dollar amounts before API calls
3. The quality difference is significant for regulatory documentation
4. Cost is trivial ($180/year vs $50K+ for GPU infrastructure)
5. Anthropic is SOC 2 certified and has enterprise agreements

If you have a SPECIFIC regulatory requirement for on-premise processing,
then go with Ollama + LLaMA 3.1 70B on GPU infrastructure.

================================================================================
""")

# Demo the anonymizer
print("DEMO: Data Anonymization")
print("─" * 80)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.integration.llm_config import DataAnonymizer

anonymizer = DataAnonymizer()

sample_prompt = """
Generate a compliance narrative for the following:

Control: Sector Concentration - Technology
Current Value: 28% of NAV
Threshold: 30%
Status: WARNING

Top Holdings in Tech Sector:
- AAPL: $150,000,000 (7.5%)
- MSFT: $120,000,000 (6.0%)
- NVDA: $95,000,000 (4.75%)
- GOOGL: $85,000,000 (4.25%)

Fund NAV: $2,000,000,000
"""

result = anonymizer.anonymize(sample_prompt)

print("\nORIGINAL PROMPT:")
print(sample_prompt)

print("\nANONYMIZED PROMPT (what gets sent to API):")
print(result.anonymized_text)

print("\nMAPPING (stored locally for de-anonymization):")
for original, anon in result.mapping.items():
    print(f"  {anon} → {original}")

print("\n" + "─" * 80)
print("Notice: Percentages and control names are NOT anonymized (needed for narrative)")
print("─" * 80)
