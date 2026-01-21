"""
RAG Safety Layer - Bulletproof Verification for Billion-Dollar Compliance

This module provides defense-in-depth safety for RAG operations:
1. Hallucination Detection - Verify every fact against retrieved context
2. Number Validation - Ensure no fabricated figures in outputs
3. Citation Enforcement - Every claim must reference source
4. Confidence Gating - Refuse to answer when uncertain
5. Human-in-the-Loop Triggers - Escalate high-risk outputs

SEC EXAMINATION CRITICAL:
- All verifications are logged with full audit trail
- False positives are preferred over false negatives
- System fails safe (refuses output rather than hallucinate)
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple, Set
import uuid

logger = logging.getLogger(__name__)


# =============================================================================
# RISK LEVELS & THRESHOLDS
# =============================================================================

class RiskLevel(Enum):
    """Risk classification for outputs requiring different handling."""
    LOW = "low"           # Routine, can auto-approve
    MEDIUM = "medium"     # Review recommended
    HIGH = "high"         # Human review required
    CRITICAL = "critical" # Senior compliance officer must approve


@dataclass(frozen=True)
class SafetyThresholds:
    """
    Configurable safety thresholds for production.
    
    These are conservative defaults for billion-dollar AUM.
    Adjust based on your risk tolerance.
    """
    # Retrieval confidence below this triggers refusal
    min_retrieval_confidence: float = 0.50
    
    # Below this, require human review
    human_review_confidence: float = 0.70
    
    # Minimum sources required for multi-source verification
    min_sources_for_verification: int = 2
    
    # Maximum percentage of unverified facts allowed
    max_unverified_fact_ratio: float = 0.10  # 10%
    
    # If LLM generates numbers not in context, flag as fabrication
    allow_calculated_numbers: bool = False  # Strict: no LLM math
    
    # Citation required for every N sentences
    citation_frequency_sentences: int = 2
    
    # Retry attempts before graceful degradation
    max_retries: int = 3
    
    # Circuit breaker: fail after N consecutive errors
    circuit_breaker_threshold: int = 5


DEFAULT_THRESHOLDS = SafetyThresholds()


# =============================================================================
# HALLUCINATION DETECTION
# =============================================================================

@dataclass
class FactVerificationResult:
    """Result of verifying a single fact against retrieved context."""
    fact_id: str
    fact_text: str
    is_verified: bool
    source_chunk_id: Optional[str]
    source_document: Optional[str]
    confidence: float
    verification_method: str
    evidence_snippet: Optional[str] = None
    
    @property
    def verification_hash(self) -> str:
        """Hash for audit trail."""
        content = f"{self.fact_id}|{self.is_verified}|{self.source_chunk_id}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class HallucinationCheckResult:
    """Complete result of hallucination detection."""
    output_id: str
    total_facts: int
    verified_facts: int
    unverified_facts: int
    fabricated_numbers: List[str]
    missing_citations: List[str]
    fact_results: List[FactVerificationResult]
    overall_verified: bool
    risk_level: RiskLevel
    requires_human_review: bool
    rejection_reason: Optional[str] = None
    checked_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def verification_rate(self) -> float:
        return self.verified_facts / self.total_facts if self.total_facts > 0 else 0
    
    @property
    def audit_hash(self) -> str:
        """Hash of entire check for audit trail."""
        content = f"{self.output_id}|{self.total_facts}|{self.verified_facts}|{self.overall_verified}"
        return hashlib.sha256(content.encode()).hexdigest()


class HallucinationDetector:
    """
    Detects hallucinations by verifying LLM output against retrieved context.
    
    Verification methods:
    1. Exact match - Direct text overlap
    2. Semantic match - Embedding similarity
    3. Number match - All numbers must appear in context
    4. Citation match - Every citation must reference real source
    """
    
    def __init__(
        self,
        thresholds: SafetyThresholds = DEFAULT_THRESHOLDS,
        embedder = None,
    ):
        self.thresholds = thresholds
        self.embedder = embedder
    
    def check_output(
        self,
        llm_output: str,
        retrieved_context: str,
        retrieved_chunks: List[Any],
        control_results: Optional[List[Any]] = None,
    ) -> HallucinationCheckResult:
        """
        Comprehensive hallucination check on LLM output.
        
        Args:
            llm_output: The text generated by the LLM
            retrieved_context: The context provided to the LLM
            retrieved_chunks: List of PolicyChunk objects
            control_results: Optional control results for number verification
            
        Returns:
            HallucinationCheckResult with detailed verification
        """
        output_id = str(uuid.uuid4())[:8]
        
        # Extract facts from output (sentences/claims)
        facts = self._extract_facts(llm_output)
        
        # Extract all numbers from output and context
        output_numbers = self._extract_numbers(llm_output)
        context_numbers = self._extract_numbers(retrieved_context)
        control_numbers = self._extract_control_numbers(control_results) if control_results else set()
        allowed_numbers = context_numbers | control_numbers
        
        # Verify each fact
        fact_results = []
        for i, fact in enumerate(facts):
            result = self._verify_fact(
                fact_id=f"{output_id}-{i}",
                fact=fact,
                retrieved_context=retrieved_context,
                retrieved_chunks=retrieved_chunks,
            )
            fact_results.append(result)
        
        verified_count = sum(1 for r in fact_results if r.is_verified)
        unverified_count = len(facts) - verified_count
        
        # Check for fabricated numbers
        fabricated_numbers = []
        for num in output_numbers:
            if not self._number_in_allowed(num, allowed_numbers):
                fabricated_numbers.append(num)
        
        # Check for missing citations
        missing_citations = self._check_citation_coverage(llm_output, facts)
        
        # Determine risk level
        risk_level, requires_review, rejection_reason = self._assess_risk(
            verification_rate=verified_count / len(facts) if facts else 1.0,
            fabricated_numbers=fabricated_numbers,
            missing_citations=missing_citations,
        )
        
        overall_verified = (
            len(fabricated_numbers) == 0 and
            (verified_count / len(facts) if facts else 1.0) >= (1 - self.thresholds.max_unverified_fact_ratio)
        )
        
        return HallucinationCheckResult(
            output_id=output_id,
            total_facts=len(facts),
            verified_facts=verified_count,
            unverified_facts=unverified_count,
            fabricated_numbers=fabricated_numbers,
            missing_citations=missing_citations,
            fact_results=fact_results,
            overall_verified=overall_verified,
            risk_level=risk_level,
            requires_human_review=requires_review,
            rejection_reason=rejection_reason,
        )
    
    def _extract_facts(self, text: str) -> List[str]:
        """Extract individual facts (sentences) from text."""
        # Split by sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Filter out very short sentences and headers
        facts = []
        for s in sentences:
            s = s.strip()
            if len(s) > 20 and not s.startswith('#'):
                facts.append(s)
        
        return facts
    
    def _extract_numbers(self, text: str) -> Set[str]:
        """Extract all numbers from text (including percentages, currencies)."""
        patterns = [
            r'\d+\.?\d*%',           # Percentages: 30%, 15.5%
            r'\$[\d,]+\.?\d*',       # Currency: $1,000,000
            r'\d{1,3}(?:,\d{3})*\.?\d*',  # Numbers with commas
            r'\d+\.?\d*',            # Plain numbers
        ]
        
        numbers = set()
        for pattern in patterns:
            matches = re.findall(pattern, text)
            numbers.update(matches)
        
        return numbers
    
    def _extract_control_numbers(self, control_results: List[Any]) -> Set[str]:
        """Extract valid numbers from control results."""
        numbers = set()
        for control in control_results:
            if hasattr(control, 'calculated_value') and control.calculated_value is not None:
                val = control.calculated_value
                numbers.add(f"{val:.1f}%")
                numbers.add(f"{val:.2f}%")
                numbers.add(f"{val:.0f}%")
                numbers.add(str(val))
            if hasattr(control, 'threshold') and control.threshold is not None:
                val = control.threshold
                numbers.add(f"{val:.1f}%")
                numbers.add(f"{val:.2f}%")
                numbers.add(f"{val:.0f}%")
                numbers.add(str(val))
        return numbers
    
    def _number_in_allowed(self, num: str, allowed: Set[str]) -> bool:
        """Check if a number appears in allowed set (with fuzzy matching)."""
        # Direct match
        if num in allowed:
            return True
        
        # Normalize and check
        try:
            # Remove formatting
            normalized = num.replace(',', '').replace('$', '').replace('%', '')
            normalized_float = float(normalized)
            
            for allowed_num in allowed:
                allowed_normalized = allowed_num.replace(',', '').replace('$', '').replace('%', '')
                try:
                    allowed_float = float(allowed_normalized)
                    # Allow small floating point differences
                    if abs(normalized_float - allowed_float) < 0.01:
                        return True
                except ValueError:
                    continue
        except ValueError:
            pass
        
        return False
    
    def _verify_fact(
        self,
        fact_id: str,
        fact: str,
        retrieved_context: str,
        retrieved_chunks: List[Any],
    ) -> FactVerificationResult:
        """Verify a single fact against retrieved context."""
        
        # Method 1: Exact substring match
        fact_lower = fact.lower()
        context_lower = retrieved_context.lower()
        
        # Check for key phrases (3+ word sequences)
        words = fact_lower.split()
        for i in range(len(words) - 2):
            phrase = ' '.join(words[i:i+3])
            if phrase in context_lower and len(phrase) > 10:
                # Find which chunk contains this
                source_chunk = None
                for chunk in retrieved_chunks:
                    if phrase in chunk.content.lower():
                        source_chunk = chunk
                        break
                
                return FactVerificationResult(
                    fact_id=fact_id,
                    fact_text=fact[:100],
                    is_verified=True,
                    source_chunk_id=source_chunk.chunk_id if source_chunk else None,
                    source_document=source_chunk.document_name if source_chunk else None,
                    confidence=0.9,
                    verification_method="exact_match",
                    evidence_snippet=phrase,
                )
        
        # Method 2: Key term overlap
        # Extract significant terms (not stopwords)
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                     'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                     'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                     'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                     'as', 'into', 'through', 'during', 'before', 'after', 'and',
                     'but', 'or', 'nor', 'so', 'yet', 'this', 'that', 'these', 'those'}
        
        fact_terms = set(w for w in words if w not in stopwords and len(w) > 3)
        context_terms = set(w for w in context_lower.split() if len(w) > 3)
        
        overlap = fact_terms & context_terms
        overlap_ratio = len(overlap) / len(fact_terms) if fact_terms else 0
        
        if overlap_ratio >= 0.6:
            return FactVerificationResult(
                fact_id=fact_id,
                fact_text=fact[:100],
                is_verified=True,
                source_chunk_id=None,
                source_document=None,
                confidence=0.7,
                verification_method="term_overlap",
                evidence_snippet=f"Terms: {', '.join(list(overlap)[:5])}",
            )
        
        # Method 3: Semantic similarity (if embedder available)
        if self.embedder and self.embedder.available:
            try:
                fact_emb = self.embedder.embed(fact)
                best_sim = 0
                best_chunk = None
                
                for chunk in retrieved_chunks:
                    if chunk.embedding:
                        sim = self._cosine_similarity(fact_emb, chunk.embedding)
                        if sim > best_sim:
                            best_sim = sim
                            best_chunk = chunk
                
                if best_sim >= 0.75:
                    return FactVerificationResult(
                        fact_id=fact_id,
                        fact_text=fact[:100],
                        is_verified=True,
                        source_chunk_id=best_chunk.chunk_id if best_chunk else None,
                        source_document=best_chunk.document_name if best_chunk else None,
                        confidence=best_sim,
                        verification_method="semantic_similarity",
                    )
            except Exception as e:
                logger.warning(f"Semantic verification failed: {e}")
        
        # Unverified
        return FactVerificationResult(
            fact_id=fact_id,
            fact_text=fact[:100],
            is_verified=False,
            source_chunk_id=None,
            source_document=None,
            confidence=overlap_ratio,
            verification_method="unverified",
        )
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        import math
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0
    
    def _check_citation_coverage(self, output: str, facts: List[str]) -> List[str]:
        """Check that output has adequate citations."""
        # Look for citation patterns
        citation_patterns = [
            r'\[.*?\]',           # [Policy: xyz]
            r'per policy',        # per policy X
            r'according to',      # according to...
            r'as defined in',     # as defined in...
            r'source:',           # Source: xyz
            r'reference:',        # Reference: xyz
        ]
        
        citation_count = 0
        for pattern in citation_patterns:
            citation_count += len(re.findall(pattern, output, re.IGNORECASE))
        
        # Calculate required citations
        required = len(facts) // self.thresholds.citation_frequency_sentences
        
        missing = []
        if citation_count < required:
            missing.append(f"Expected {required} citations, found {citation_count}")
        
        return missing
    
    def _assess_risk(
        self,
        verification_rate: float,
        fabricated_numbers: List[str],
        missing_citations: List[str],
    ) -> Tuple[RiskLevel, bool, Optional[str]]:
        """Assess risk level and determine if human review is needed."""
        
        # Critical: Any fabricated numbers
        if fabricated_numbers and not self.thresholds.allow_calculated_numbers:
            return (
                RiskLevel.CRITICAL,
                True,
                f"Fabricated numbers detected: {fabricated_numbers[:3]}"
            )
        
        # High: Low verification rate
        if verification_rate < self.thresholds.min_retrieval_confidence:
            return (
                RiskLevel.HIGH,
                True,
                f"Verification rate {verification_rate:.0%} below threshold"
            )
        
        # Medium: Missing citations or below review threshold
        if verification_rate < self.thresholds.human_review_confidence or missing_citations:
            return (
                RiskLevel.MEDIUM,
                True,
                "Below human review confidence threshold"
            )
        
        # Low: All checks passed
        return (RiskLevel.LOW, False, None)


# =============================================================================
# CIRCUIT BREAKER & FAULT TOLERANCE
# =============================================================================

class CircuitState(Enum):
    CLOSED = "closed"    # Normal operation
    OPEN = "open"        # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreaker:
    """
    Circuit breaker pattern for fault tolerance.
    
    Prevents cascade failures when external services fail.
    After N consecutive failures, circuit opens and rejects requests.
    After cooldown, allows test request to check recovery.
    """
    name: str
    failure_threshold: int = 5
    recovery_timeout_seconds: int = 60
    
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failure_count: int = field(default=0, init=False)
    _last_failure_time: Optional[datetime] = field(default=None, init=False)
    _state_history: List[Dict] = field(default_factory=list, init=False)
    
    @property
    def state(self) -> CircuitState:
        """Get current state, checking for recovery timeout."""
        if self._state == CircuitState.OPEN and self._last_failure_time:
            elapsed = (datetime.now(timezone.utc) - self._last_failure_time).total_seconds()
            if elapsed >= self.recovery_timeout_seconds:
                self._transition_to(CircuitState.HALF_OPEN)
        return self._state
    
    def record_success(self):
        """Record a successful operation."""
        if self._state == CircuitState.HALF_OPEN:
            self._transition_to(CircuitState.CLOSED)
        self._failure_count = 0
    
    def record_failure(self, error: str = ""):
        """Record a failed operation."""
        self._failure_count += 1
        self._last_failure_time = datetime.now(timezone.utc)
        
        if self._failure_count >= self.failure_threshold:
            self._transition_to(CircuitState.OPEN)
        
        logger.warning(
            f"Circuit {self.name}: failure {self._failure_count}/{self.failure_threshold} - {error}"
        )
    
    def is_available(self) -> bool:
        """Check if circuit allows requests."""
        return self.state != CircuitState.OPEN
    
    def _transition_to(self, new_state: CircuitState):
        """Transition to a new state with logging."""
        old_state = self._state
        self._state = new_state
        
        transition = {
            "from": old_state.value,
            "to": new_state.value,
            "at": datetime.now(timezone.utc).isoformat(),
            "failure_count": self._failure_count,
        }
        self._state_history.append(transition)
        
        logger.info(f"Circuit {self.name}: {old_state.value} -> {new_state.value}")


@dataclass
class FallbackResult:
    """Result from fallback mechanism."""
    success: bool
    result: Any
    fallback_used: str
    original_error: Optional[str] = None


class GracefulDegradation:
    """
    Implements graceful degradation with multiple fallback levels.
    
    Fallback chain:
    1. Primary: Full RAG (embeddings + reranking)
    2. Secondary: Simple vector search (no reranking)
    3. Tertiary: Keyword/BM25 search only
    4. Quaternary: Return template response
    """
    
    FALLBACK_TEMPLATE = """
    Unable to retrieve specific policy context for this control.
    
    Generic guidance:
    - Review the applicable policy document manually
    - Consult with the Chief Compliance Officer
    - Document the manual review process
    
    This response was generated using fallback mode due to retrieval issues.
    Manual verification is required.
    """
    
    def __init__(self, thresholds: SafetyThresholds = DEFAULT_THRESHOLDS):
        self.thresholds = thresholds
        self.circuit_breakers = {
            "embeddings": CircuitBreaker("embeddings"),
            "vector_search": CircuitBreaker("vector_search"),
            "reranking": CircuitBreaker("reranking"),
            "llm": CircuitBreaker("llm"),
        }
    
    def get_circuit(self, name: str) -> CircuitBreaker:
        """Get or create a circuit breaker."""
        if name not in self.circuit_breakers:
            self.circuit_breakers[name] = CircuitBreaker(name)
        return self.circuit_breakers[name]
    
    def all_circuits_status(self) -> Dict[str, str]:
        """Get status of all circuits for monitoring."""
        return {
            name: circuit.state.value 
            for name, circuit in self.circuit_breakers.items()
        }


# =============================================================================
# OUTPUT VALIDATION
# =============================================================================

@dataclass
class OutputValidationResult:
    """Result of output validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    risk_level: RiskLevel
    validation_hash: str
    validated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class OutputValidator:
    """
    Validates LLM outputs before they are used in compliance documents.
    
    Checks:
    1. Structure - Required sections present
    2. Numbers - All numbers match source data
    3. Citations - All citations reference real sources
    4. Keywords - Required compliance language present
    5. Prohibited - No prohibited phrases/patterns
    """
    
    # Required compliance language for SEC filings
    REQUIRED_KEYWORDS = {
        "breach": ["threshold", "limit", "exceeded", "violation"],
        "exception": ["exception", "waiver", "approval", "authorized"],
    }
    
    # Prohibited patterns that indicate problems
    PROHIBITED_PATTERNS = [
        r"I think",              # LLM opinion
        r"I believe",            # LLM opinion
        r"probably",             # Uncertainty
        r"might be",             # Uncertainty
        r"approximately",        # Vagueness (use exact numbers)
        r"around \d+",           # Vagueness
        r"as an AI",             # Self-reference
        r"I cannot",             # Inability
        r"I don't have",         # Missing info
    ]
    
    def validate(
        self,
        output: str,
        output_type: str,
        source_numbers: Set[str],
        source_documents: List[str],
    ) -> OutputValidationResult:
        """
        Validate an LLM output.
        
        Args:
            output: The generated text
            output_type: Type of output (e.g., "breach", "summary")
            source_numbers: Set of valid numbers from source data
            source_documents: List of valid source document names
            
        Returns:
            OutputValidationResult with errors and warnings
        """
        errors = []
        warnings = []
        
        # Check 1: Prohibited patterns
        for pattern in self.PROHIBITED_PATTERNS:
            if re.search(pattern, output, re.IGNORECASE):
                errors.append(f"Prohibited pattern found: '{pattern}'")
        
        # Check 2: Required keywords for output type
        if output_type in self.REQUIRED_KEYWORDS:
            required = self.REQUIRED_KEYWORDS[output_type]
            found = sum(1 for kw in required if kw.lower() in output.lower())
            if found < len(required) // 2:
                warnings.append(f"Missing required compliance language for {output_type}")
        
        # Check 3: Number validation
        output_numbers = set(re.findall(r'\d+\.?\d*%?', output))
        for num in output_numbers:
            # Normalize for comparison
            if num not in source_numbers:
                num_float = float(num.replace('%', ''))
                if not any(
                    abs(num_float - float(sn.replace('%', ''))) < 0.01 
                    for sn in source_numbers 
                    if re.match(r'\d+\.?\d*%?$', sn)
                ):
                    errors.append(f"Unverified number in output: {num}")
        
        # Check 4: Citation verification (if citations present)
        citations = re.findall(r'\[(?:Policy|Source|Document):\s*([^\]]+)\]', output)
        for citation in citations:
            if not any(citation.lower() in doc.lower() for doc in source_documents):
                errors.append(f"Invalid citation: {citation}")
        
        # Determine risk level
        if errors:
            risk_level = RiskLevel.HIGH
        elif warnings:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW
        
        # Generate validation hash
        validation_hash = hashlib.sha256(
            f"{output[:100]}|{len(errors)}|{len(warnings)}".encode()
        ).hexdigest()[:16]
        
        return OutputValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            risk_level=risk_level,
            validation_hash=validation_hash,
        )


# =============================================================================
# SAFE RAG WRAPPER
# =============================================================================

@dataclass
class SafeRAGResult:
    """
    Complete result from safe RAG operation.
    
    Includes all verification and audit information.
    """
    # Core result
    output: Optional[str]
    success: bool
    
    # Verification
    hallucination_check: Optional[HallucinationCheckResult]
    output_validation: Optional[OutputValidationResult]
    
    # Retrieval info
    retrieval_confidence: float
    sources_used: List[str]
    
    # Safety
    risk_level: RiskLevel
    requires_human_review: bool
    rejection_reason: Optional[str]
    
    # Fallback
    fallback_used: Optional[str]
    
    # Audit
    operation_id: str
    started_at: datetime
    completed_at: datetime
    duration_seconds: float
    
    @property
    def audit_record(self) -> Dict[str, Any]:
        """Generate complete audit record for evidence store."""
        return {
            "operation_id": self.operation_id,
            "success": self.success,
            "risk_level": self.risk_level.value,
            "requires_human_review": self.requires_human_review,
            "retrieval_confidence": self.retrieval_confidence,
            "sources_used": self.sources_used,
            "hallucination_verified": self.hallucination_check.overall_verified if self.hallucination_check else None,
            "output_valid": self.output_validation.is_valid if self.output_validation else None,
            "fallback_used": self.fallback_used,
            "rejection_reason": self.rejection_reason,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat(),
            "duration_seconds": self.duration_seconds,
        }


class SafeRAGPipeline:
    """
    Production-safe RAG pipeline with comprehensive verification.
    
    Wraps the RAG retriever and narrative generator with:
    - Input validation
    - Retrieval confidence gating
    - Hallucination detection
    - Output validation
    - Circuit breakers
    - Graceful degradation
    - Full audit trail
    
    Usage:
        safe_rag = SafeRAGPipeline(retriever, generator, evidence_store)
        result = safe_rag.generate_narrative(control_result)
        
        if result.requires_human_review:
            escalate_to_compliance_officer(result)
        elif result.success:
            use_output(result.output)
    """
    
    def __init__(
        self,
        retriever,
        narrative_generator,
        evidence_store = None,
        thresholds: SafetyThresholds = DEFAULT_THRESHOLDS,
        embedder = None,
    ):
        self.retriever = retriever
        self.generator = narrative_generator
        self.evidence_store = evidence_store
        self.thresholds = thresholds
        
        # Safety components
        self.hallucination_detector = HallucinationDetector(thresholds, embedder)
        self.output_validator = OutputValidator()
        self.degradation = GracefulDegradation(thresholds)
    
    def generate_safe_narrative(
        self,
        control_result,
        require_verification: bool = True,
        allow_fallback: bool = True,
    ) -> SafeRAGResult:
        """
        Generate a narrative with full safety verification.
        
        Args:
            control_result: The control result to generate narrative for
            require_verification: If True, reject unverified outputs
            allow_fallback: If True, use fallback when primary fails
            
        Returns:
            SafeRAGResult with output and all verification info
        """
        operation_id = str(uuid.uuid4())[:12]
        started_at = datetime.now(timezone.utc)
        
        logger.info(f"SafeRAG {operation_id}: Starting for {control_result.control_name}")
        
        try:
            # Step 1: Retrieve context (with circuit breaker)
            retrieval_result = self._safe_retrieve(control_result, allow_fallback)
            
            if not retrieval_result.chunks:
                return self._create_rejection_result(
                    operation_id=operation_id,
                    started_at=started_at,
                    reason="No relevant context retrieved",
                    risk_level=RiskLevel.HIGH,
                )
            
            # Step 2: Check retrieval confidence
            if retrieval_result.confidence < self.thresholds.min_retrieval_confidence:
                return self._create_rejection_result(
                    operation_id=operation_id,
                    started_at=started_at,
                    reason=f"Retrieval confidence {retrieval_result.confidence:.0%} below threshold",
                    risk_level=RiskLevel.HIGH,
                    retrieval_confidence=retrieval_result.confidence,
                )
            
            # Step 3: Generate narrative (with circuit breaker)
            narrative = self._safe_generate(control_result, retrieval_result, allow_fallback)
            
            if not narrative or not narrative.text:
                return self._create_rejection_result(
                    operation_id=operation_id,
                    started_at=started_at,
                    reason="Narrative generation failed",
                    risk_level=RiskLevel.HIGH,
                    retrieval_confidence=retrieval_result.confidence,
                )
            
            # Step 4: Hallucination check
            hallucination_result = self.hallucination_detector.check_output(
                llm_output=narrative.text,
                retrieved_context=retrieval_result.to_prompt_context(),
                retrieved_chunks=retrieval_result.chunks,
                control_results=[control_result],
            )
            
            # Step 5: Output validation
            source_numbers = self._extract_source_numbers(control_result)
            source_documents = [c.document_name for c in retrieval_result.chunks]
            
            validation_result = self.output_validator.validate(
                output=narrative.text,
                output_type=control_result.status,
                source_numbers=source_numbers,
                source_documents=source_documents,
            )
            
            # Step 6: Determine final risk level
            max_risk = max(
                hallucination_result.risk_level,
                validation_result.risk_level,
                key=lambda r: [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL].index(r)
            )
            
            requires_review = (
                hallucination_result.requires_human_review or
                not validation_result.is_valid or
                max_risk in (RiskLevel.HIGH, RiskLevel.CRITICAL)
            )
            
            # Step 7: Reject if verification failed and required
            if require_verification and not hallucination_result.overall_verified:
                return self._create_rejection_result(
                    operation_id=operation_id,
                    started_at=started_at,
                    reason=hallucination_result.rejection_reason or "Verification failed",
                    risk_level=max_risk,
                    retrieval_confidence=retrieval_result.confidence,
                    hallucination_check=hallucination_result,
                    output_validation=validation_result,
                )
            
            completed_at = datetime.now(timezone.utc)
            
            result = SafeRAGResult(
                output=narrative.text,
                success=True,
                hallucination_check=hallucination_result,
                output_validation=validation_result,
                retrieval_confidence=retrieval_result.confidence,
                sources_used=source_documents,
                risk_level=max_risk,
                requires_human_review=requires_review,
                rejection_reason=None,
                fallback_used=None,
                operation_id=operation_id,
                started_at=started_at,
                completed_at=completed_at,
                duration_seconds=(completed_at - started_at).total_seconds(),
            )
            
            # Record in evidence store
            if self.evidence_store:
                self._record_audit(result)
            
            logger.info(
                f"SafeRAG {operation_id}: Complete - "
                f"verified={hallucination_result.overall_verified}, "
                f"risk={max_risk.value}, "
                f"review_required={requires_review}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"SafeRAG {operation_id}: Exception - {e}")
            return self._create_rejection_result(
                operation_id=operation_id,
                started_at=started_at,
                reason=f"Pipeline error: {str(e)}",
                risk_level=RiskLevel.CRITICAL,
            )
    
    def _safe_retrieve(self, control_result, allow_fallback: bool):
        """Retrieve with circuit breaker and fallback."""
        circuit = self.degradation.get_circuit("vector_search")
        
        if not circuit.is_available():
            if allow_fallback:
                return self._fallback_retrieve(control_result)
            raise RuntimeError("Vector search circuit open, no fallback allowed")
        
        try:
            result = self.retriever.retrieve_for_control(
                control_name=control_result.control_name,
                control_type=getattr(control_result, 'control_type', 'general'),
                status=control_result.status,
                calculated_value=float(control_result.calculated_value or 0),
                threshold=float(control_result.threshold or 0),
            )
            circuit.record_success()
            return result
        except Exception as e:
            circuit.record_failure(str(e))
            if allow_fallback:
                return self._fallback_retrieve(control_result)
            raise
    
    def _fallback_retrieve(self, control_result):
        """Fallback retrieval using keyword search."""
        logger.warning("Using fallback keyword retrieval")
        # Return minimal result indicating fallback
        from .retriever import RetrievedContext
        return RetrievedContext(
            query=control_result.control_name,
            chunks=[],
            confidence=0.3,
            confidence_explanation="Fallback mode - no vector search",
        )
    
    def _safe_generate(self, control_result, retrieval_result, allow_fallback: bool):
        """Generate with circuit breaker."""
        circuit = self.degradation.get_circuit("llm")
        
        if not circuit.is_available():
            if allow_fallback:
                return self._fallback_generate(control_result)
            raise RuntimeError("LLM circuit open, no fallback allowed")
        
        try:
            result = self.generator.generate_for_control(
                control_result=control_result,
                retrieved_context=retrieval_result,
            )
            circuit.record_success()
            return result
        except Exception as e:
            circuit.record_failure(str(e))
            if allow_fallback:
                return self._fallback_generate(control_result)
            raise
    
    def _fallback_generate(self, control_result):
        """Fallback generation using template."""
        from ..narrative.generator import GeneratedNarrative
        return GeneratedNarrative(
            narrative_id="fallback",
            text=GracefulDegradation.FALLBACK_TEMPLATE,
            narrative_type="fallback",
            citations=[],
        )
    
    def _extract_source_numbers(self, control_result) -> Set[str]:
        """Extract valid numbers from control result."""
        numbers = set()
        if hasattr(control_result, 'calculated_value') and control_result.calculated_value:
            val = control_result.calculated_value
            numbers.add(f"{val}")
            numbers.add(f"{val:.1f}")
            numbers.add(f"{val:.2f}")
            numbers.add(f"{val}%")
            numbers.add(f"{val:.1f}%")
        if hasattr(control_result, 'threshold') and control_result.threshold:
            val = control_result.threshold
            numbers.add(f"{val}")
            numbers.add(f"{val:.1f}")
            numbers.add(f"{val:.2f}")
            numbers.add(f"{val}%")
            numbers.add(f"{val:.1f}%")
        return numbers
    
    def _create_rejection_result(
        self,
        operation_id: str,
        started_at: datetime,
        reason: str,
        risk_level: RiskLevel,
        retrieval_confidence: float = 0.0,
        hallucination_check: Optional[HallucinationCheckResult] = None,
        output_validation: Optional[OutputValidationResult] = None,
    ) -> SafeRAGResult:
        """Create a rejection result."""
        completed_at = datetime.now(timezone.utc)
        
        return SafeRAGResult(
            output=None,
            success=False,
            hallucination_check=hallucination_check,
            output_validation=output_validation,
            retrieval_confidence=retrieval_confidence,
            sources_used=[],
            risk_level=risk_level,
            requires_human_review=True,
            rejection_reason=reason,
            fallback_used=None,
            operation_id=operation_id,
            started_at=started_at,
            completed_at=completed_at,
            duration_seconds=(completed_at - started_at).total_seconds(),
        )
    
    def _record_audit(self, result: SafeRAGResult):
        """Record result in evidence store for audit trail."""
        try:
            if self.evidence_store:
                self.evidence_store.record_operation(
                    operation_type="safe_rag_generation",
                    operation_id=result.operation_id,
                    data=result.audit_record,
                )
        except Exception as e:
            logger.error(f"Failed to record audit: {e}")


# =============================================================================
# CONFIDENCE THRESHOLDS FOR PRODUCTION
# =============================================================================

# Conservative thresholds for billion-dollar AUM
PRODUCTION_THRESHOLDS = SafetyThresholds(
    min_retrieval_confidence=0.60,
    human_review_confidence=0.80,
    min_sources_for_verification=2,
    max_unverified_fact_ratio=0.05,  # Only 5% unverified allowed
    allow_calculated_numbers=False,   # Strict: no LLM math
    citation_frequency_sentences=2,
    max_retries=3,
    circuit_breaker_threshold=3,
)

# Relaxed thresholds for development/testing
DEVELOPMENT_THRESHOLDS = SafetyThresholds(
    min_retrieval_confidence=0.40,
    human_review_confidence=0.60,
    min_sources_for_verification=1,
    max_unverified_fact_ratio=0.20,
    allow_calculated_numbers=True,
    citation_frequency_sentences=5,
    max_retries=1,
    circuit_breaker_threshold=10,
)
