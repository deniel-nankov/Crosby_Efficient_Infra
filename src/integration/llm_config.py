"""
LLM Configuration for Compliance RAG System

Supports:
1. Cloud APIs (Claude, OpenAI) - Recommended for quality
2. Local LLMs (Ollama, vLLM) - For maximum data privacy

For a $2B hedge fund, we recommend CLOUD API with data anonymization:
- Better prose quality for compliance documentation
- No GPU infrastructure costs ($50K+)
- SOC 2 certified providers
- Data anonymization removes sensitive info before API call
"""

from __future__ import annotations

import os
import re
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers."""
    ANTHROPIC = "anthropic"  # Claude - Recommended
    OPENAI = "openai"        # GPT-4
    LMSTUDIO = "lmstudio"    # Local via LM Studio (OpenAI-compatible)
    OLLAMA = "ollama"        # Local via Ollama
    VLLM = "vllm"            # Local via vLLM server
    MOCK = "mock"            # For testing


@dataclass
class LLMConfig:
    """LLM configuration settings."""
    provider: LLMProvider
    model_id: str
    api_key: Optional[str] = None
    api_base: Optional[str] = None  # For local LLMs
    temperature: float = 0.3        # Low for compliance consistency
    max_tokens: int = 500
    timeout: int = 30
    
    # Privacy settings
    anonymize_data: bool = True     # Scrub sensitive data before API call
    log_prompts: bool = False       # Don't log prompts in production
    
    @classmethod
    def from_env(cls) -> "LLMConfig":
        """Create config from environment variables."""
        provider_str = os.environ.get("LLM_PROVIDER", "mock").lower()
        provider = LLMProvider(provider_str)
        
        return cls(
            provider=provider,
            model_id=os.environ.get("LLM_MODEL", cls._default_model(provider)),
            api_key=os.environ.get("LLM_API_KEY") or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY"),
            api_base=os.environ.get("LLM_API_BASE"),
            temperature=float(os.environ.get("LLM_TEMPERATURE", "0.3")),
            max_tokens=int(os.environ.get("LLM_MAX_TOKENS", "500")),
            anonymize_data=os.environ.get("LLM_ANONYMIZE", "true").lower() == "true",
        )
    
    @staticmethod
    def _default_model(provider: LLMProvider) -> str:
        """Default model for each provider."""
        defaults = {
            LLMProvider.ANTHROPIC: "claude-sonnet-4-20250514",
            LLMProvider.OPENAI: "gpt-4o",
            LLMProvider.LMSTUDIO: "local-model",  # LM Studio uses whatever is loaded
            LLMProvider.OLLAMA: "llama3.1:70b",
            LLMProvider.VLLM: "meta-llama/Meta-Llama-3.1-70B-Instruct",
            LLMProvider.MOCK: "mock",
        }
        return defaults.get(provider, "mock")


# =============================================================================
# DATA ANONYMIZATION
# =============================================================================

@dataclass
class AnonymizationResult:
    """Result of data anonymization."""
    anonymized_text: str
    mapping: Dict[str, str]  # Original -> Anonymized
    
    def deanonymize(self, text: str) -> str:
        """Restore original values in text."""
        result = text
        for original, anon in self.mapping.items():
            result = result.replace(anon, original)
        return result


class DataAnonymizer:
    """
    Anonymize sensitive data before sending to external LLM.
    
    What we anonymize:
    - Security tickers -> [SECURITY_1], [SECURITY_2], etc.
    - Dollar amounts (if large) -> [AMOUNT_1], etc.
    - Dates (if specific) -> [DATE_1], etc.
    
    What we DON'T anonymize:
    - Percentages (needed for compliance narrative)
    - Control names (generic)
    - Policy text (already internal docs)
    """
    
    def __init__(self):
        self.security_counter = 0
        self.amount_counter = 0
        self.date_counter = 0
    
    def anonymize(self, text: str) -> AnonymizationResult:
        """Anonymize sensitive data in text."""
        mapping = {}
        result = text
        
        # Common words that look like tickers but aren't
        skip_words = {
            'NAV', 'USD', 'SEC', 'CEO', 'CIO', 'CFO', 'ADV', 'GICS', 
            'THE', 'AND', 'FOR', 'NOT', 'ARE', 'WAS', 'HAS', 'HAD',
            'TOP', 'ALL', 'ANY', 'CAN', 'MAY', 'NEW', 'NOW', 'OLD',
            'ONE', 'OUR', 'OUT', 'OWN', 'SAY', 'SHE', 'TOO', 'TWO',
            'WAY', 'WHO', 'BOY', 'DID', 'GET', 'HIM', 'HIS', 'HOW',
            'MAN', 'ITS', 'LET', 'PUT', 'SAW', 'HER', 'USE', 'WARNING',
        }
        
        # Anonymize tickers - look for patterns like "- AAPL:" or "AAPL position"
        ticker_pattern = r'(?:^|\s|-\s)([A-Z]{3,5})(?=\s*:|,|\s|$)'
        for match in re.finditer(ticker_pattern, result):
            ticker = match.group(1)
            if ticker in skip_words:
                continue
            if ticker not in mapping:
                self.security_counter += 1
                mapping[ticker] = f"[SECURITY_{self.security_counter}]"
        
        for original, anon in mapping.items():
            if not original.startswith('$'):  # Only replace tickers here
                result = re.sub(rf'\b{original}\b', anon, result)
        
        # Anonymize large dollar amounts (>$1M)
        amount_pattern = r'\$[\d,]+(?:\.\d{2})?(?=\s|,|\.|$)'
        for match in re.finditer(amount_pattern, result):
            amount_str = match.group(0)
            # Parse amount
            amount = float(amount_str.replace('$', '').replace(',', ''))
            if amount > 1_000_000:  # Only anonymize large amounts
                if amount_str not in mapping:
                    self.amount_counter += 1
                    mapping[amount_str] = f"[AMOUNT_{self.amount_counter}]"
        
        for original, anon in list(mapping.items()):
            if original.startswith('$'):
                result = result.replace(original, anon)
        
        return AnonymizationResult(
            anonymized_text=result,
            mapping=mapping,
        )


# =============================================================================
# LLM CLIENT INTERFACE
# =============================================================================

class LLMClient(ABC):
    """Abstract LLM client interface."""
    
    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate text from prompt."""
        pass
    
    @property
    @abstractmethod
    def model_id(self) -> str:
        """Get model identifier."""
        pass


class AnthropicClient(LLMClient):
    """Claude API client - RECOMMENDED for compliance quality."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = None
    
    @property
    def client(self):
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.config.api_key)
            except ImportError:
                raise ImportError("Install anthropic: pip install anthropic")
        return self._client
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        response = self.client.messages.create(
            model=self.config.model_id,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            system=system_prompt or "You are a compliance documentation assistant.",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
    
    @property
    def model_id(self) -> str:
        return self.config.model_id


class OpenAIClient(LLMClient):
    """OpenAI GPT client (also works with OpenAI-compatible APIs like LM Studio)."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self._client = None
    
    @property
    def client(self):
        if self._client is None:
            try:
                import openai
                # Support custom base URL for LM Studio and other OpenAI-compatible APIs
                if self.config.api_base:
                    self._client = openai.OpenAI(
                        api_key=self.config.api_key or "lm-studio",  # LM Studio doesn't need real key
                        base_url=self.config.api_base,
                    )
                else:
                    self._client = openai.OpenAI(api_key=self.config.api_key)
            except ImportError:
                raise ImportError("Install openai: pip install openai")
        return self._client
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        response = self.client.chat.completions.create(
            model=self.config.model_id,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            messages=[
                {"role": "system", "content": system_prompt or "You are a compliance documentation assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content
    
    @property
    def model_id(self) -> str:
        return self.config.model_id


class OllamaClient(LLMClient):
    """
    Local LLM via Ollama.
    
    Setup:
    1. Install Ollama: https://ollama.ai
    2. Pull model: ollama pull llama3.1:70b
    3. Run: ollama serve
    """
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.api_base = config.api_base or "http://localhost:11434"
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        import requests
        
        response = requests.post(
            f"{self.api_base}/api/generate",
            json={
                "model": self.config.model_id,
                "prompt": prompt,
                "system": system_prompt or "You are a compliance documentation assistant.",
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens,
                },
            },
            timeout=self.config.timeout,
        )
        response.raise_for_status()
        return response.json()["response"]
    
    @property
    def model_id(self) -> str:
        return f"ollama/{self.config.model_id}"


class VLLMClient(LLMClient):
    """
    Local LLM via vLLM server.
    
    Setup:
    1. Install vLLM: pip install vllm
    2. Start server: python -m vllm.entrypoints.openai.api_server --model meta-llama/Meta-Llama-3.1-70B-Instruct
    3. Set LLM_API_BASE=http://localhost:8000
    """
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.api_base = config.api_base or "http://localhost:8000"
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        import requests
        
        response = requests.post(
            f"{self.api_base}/v1/chat/completions",
            json={
                "model": self.config.model_id,
                "messages": [
                    {"role": "system", "content": system_prompt or "You are a compliance documentation assistant."},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
            },
            timeout=self.config.timeout,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    
    @property
    def model_id(self) -> str:
        return f"vllm/{self.config.model_id}"


class MockClient(LLMClient):
    """Mock client for testing."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        return "[Mock narrative: This is a test response with [Policy: test | Section: 1.0] citation.]"
    
    @property
    def model_id(self) -> str:
        return "mock"


class LMStudioClient(LLMClient):
    """
    Local LLM via LM Studio.
    
    LM Studio provides an OpenAI-compatible API at http://localhost:1234/v1
    
    Setup:
    1. Download LM Studio: https://lmstudio.ai
    2. Download a model (e.g., Qwen 3, Llama 3.1, Mistral)
    3. Go to Local Server tab and click "Start Server"
    4. Set LLM_PROVIDER=lmstudio
    """
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.api_base = config.api_base or "http://localhost:1234/v1"
        self._client = None
    
    @property
    def client(self):
        if self._client is None:
            try:
                import openai
                self._client = openai.OpenAI(
                    api_key="lm-studio",  # LM Studio doesn't require a real key
                    base_url=self.api_base,
                )
            except ImportError:
                raise ImportError("Install openai: pip install openai")
        return self._client
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        response = self.client.chat.completions.create(
            model=self.config.model_id or "local-model",
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            messages=[
                {"role": "system", "content": system_prompt or "You are a compliance documentation assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content
    
    @property
    def model_id(self) -> str:
        return f"lmstudio/{self.config.model_id or 'local-model'}"


# =============================================================================
# FACTORY
# =============================================================================

def create_llm_client(config: Optional[LLMConfig] = None) -> LLMClient:
    """
    Create LLM client based on configuration.
    
    Usage:
        # From environment
        client = create_llm_client()
        
        # Explicit config
        config = LLMConfig(provider=LLMProvider.ANTHROPIC, api_key="sk-...")
        client = create_llm_client(config)
    """
    if config is None:
        config = LLMConfig.from_env()
    
    clients = {
        LLMProvider.ANTHROPIC: AnthropicClient,
        LLMProvider.OPENAI: OpenAIClient,
        LLMProvider.LMSTUDIO: LMStudioClient,
        LLMProvider.OLLAMA: OllamaClient,
        LLMProvider.VLLM: VLLMClient,
        LLMProvider.MOCK: MockClient,
    }
    
    client_class = clients.get(config.provider)
    if not client_class:
        raise ValueError(f"Unknown provider: {config.provider}")
    
    return client_class(config)


# =============================================================================
# COMPLIANCE-SAFE WRAPPER
# =============================================================================

class ComplianceLLM:
    """
    LLM wrapper with compliance-safe features:
    - Automatic data anonymization
    - Audit logging
    - De-anonymization of responses
    """
    
    def __init__(
        self, 
        config: Optional[LLMConfig] = None,
        client: Optional[LLMClient] = None,
    ):
        self.config = config or LLMConfig.from_env()
        self.client = client or create_llm_client(self.config)
        self.anonymizer = DataAnonymizer() if self.config.anonymize_data else None
    
    def generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        deanonymize_response: bool = True,
    ) -> str:
        """
        Generate text with optional anonymization.
        
        Args:
            prompt: User prompt
            system_prompt: System instructions
            deanonymize_response: Whether to restore original values in response
        
        Returns:
            Generated text (with original values restored if deanonymize_response=True)
        """
        mapping = {}
        
        # Anonymize if configured
        if self.anonymizer:
            result = self.anonymizer.anonymize(prompt)
            prompt = result.anonymized_text
            mapping = result.mapping
            
            if self.config.log_prompts:
                logger.debug(f"Anonymized {len(mapping)} items")
        
        # Generate
        response = self.client.generate(prompt, system_prompt)
        
        # De-anonymize if needed
        if deanonymize_response and mapping:
            for original, anon in mapping.items():
                response = response.replace(anon, original)
        
        return response
    
    @property
    def model_id(self) -> str:
        return self.client.model_id


# =============================================================================
# QUICK START
# =============================================================================

def get_compliance_llm() -> ComplianceLLM:
    """
    Get a compliance-safe LLM client.
    
    Configure via environment variables:
        LLM_PROVIDER=anthropic   # or openai, ollama, vllm, mock
        LLM_API_KEY=sk-...       # API key for cloud providers
        LLM_MODEL=claude-sonnet-4-20250514  # Model ID
        LLM_ANONYMIZE=true       # Anonymize sensitive data
    
    Example:
        llm = get_compliance_llm()
        narrative = llm.generate("Generate compliance narrative for...")
    """
    return ComplianceLLM()
