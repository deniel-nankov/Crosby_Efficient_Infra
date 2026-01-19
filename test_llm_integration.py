#!/usr/bin/env python3
"""
=============================================================================
LLM INTEGRATION TEST
=============================================================================

Quick test to verify LLM connectivity before running full pipeline.

USAGE:
    # With Anthropic Claude (recommended):
    set ANTHROPIC_API_KEY=sk-ant-...
    python test_llm_integration.py
    
    # With OpenAI GPT-4:
    set OPENAI_API_KEY=sk-...
    set LLM_PROVIDER=openai
    python test_llm_integration.py

    # With local Ollama:
    # First: ollama serve && ollama pull llama3.1:70b
    set LLM_PROVIDER=ollama
    python test_llm_integration.py

=============================================================================
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def main():
    print("=" * 70)
    print("LLM INTEGRATION TEST")
    print("=" * 70)
    
    # Check environment
    print("\n1. Checking environment variables...")
    
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")
    llm_provider = os.environ.get("LLM_PROVIDER", "").lower()
    
    if anthropic_key:
        print(f"   ✓ ANTHROPIC_API_KEY found: {anthropic_key[:12]}...")
        expected_provider = "anthropic"
    elif openai_key:
        print(f"   ✓ OPENAI_API_KEY found: {openai_key[:12]}...")
        expected_provider = "openai"
    elif llm_provider in ("ollama", "vllm"):
        print(f"   ✓ Local LLM provider: {llm_provider}")
        expected_provider = llm_provider
    else:
        print("   ✗ No API key or provider configured!")
        print("\n   To configure:")
        print("     set ANTHROPIC_API_KEY=sk-ant-...")
        print("   OR")
        print("     set OPENAI_API_KEY=sk-...")
        print("   OR")
        print("     set LLM_PROVIDER=ollama")
        return 1
    
    # Import and test
    print("\n2. Initializing LLM client...")
    try:
        from integration.llm_config import get_compliance_llm, LLMConfig
        
        llm = get_compliance_llm()
        print(f"   ✓ LLM client created: {llm.model_id}")
        
    except ImportError as e:
        print(f"   ✗ Import error: {e}")
        print("   Try: pip install anthropic openai")
        return 1
    except Exception as e:
        print(f"   ✗ Client creation failed: {e}")
        return 1
    
    # Test generation
    print("\n3. Testing LLM generation...")
    
    test_prompt = """Generate a brief 2-sentence compliance summary for the following:
    
Control: Sector Concentration - Technology
Status: WARNING  
Value: 28%
Threshold: 30%

Include a citation in format [Policy: investment_guidelines.md | Section: concentration]."""

    system_prompt = "You are a compliance documentation assistant. Be brief and professional."
    
    try:
        print("   Sending test prompt...")
        response = llm.generate(test_prompt, system_prompt)
        print(f"   ✓ Response received ({len(response)} chars)")
        print("\n   --- LLM Response ---")
        print(f"   {response}")
        print("   --- End Response ---")
        
    except Exception as e:
        print(f"   ✗ Generation failed: {e}")
        return 1
    
    # Success
    print("\n" + "=" * 70)
    print("LLM INTEGRATION: SUCCESS ✓")
    print("=" * 70)
    print("\nYou can now run the full pipeline with LLM-powered narratives:")
    print("  python run_database_pipeline.py --setup-sample")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
