"""
VERAFI-Style Evaluation for Compliance RAG

Implements evaluation methodology from the VERAFI paper:
"Verified Agentic Financial Intelligence: An Agentic AI Framework
for Grounded Retrieval-Augmented Generation in Financial Services"

VERAFI achieves 94.7% accuracy on FinanceBench vs 52.4% baseline RAG by:
1. Pre-generation policy injection (not post-hoc filtering)
2. Neurosymbolic policy layer with formal verification
3. Agentic tool use (calculator, retriever, validator)
4. Multi-hop reasoning with explicit evidence chains

This module evaluates your RAG against these VERAFI standards.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
from enum import Enum
import math


class VERAFIMetric(Enum):
    """VERAFI evaluation dimensions."""
    GROUNDED_ACCURACY = "grounded_accuracy"  # Answers backed by evidence
    CALCULATION_ACCURACY = "calculation_accuracy"  # Math is correct
    POLICY_COMPLIANCE = "policy_compliance"  # Follows regulatory policies
    EVIDENCE_QUALITY = "evidence_quality"  # Citation quality
    HALLUCINATION_FREE = "hallucination_free"  # No fabricated facts


@dataclass
class VERAFIResult:
    """Result of VERAFI-style evaluation."""
    # Overall scores (0-1)
    grounded_accuracy: float
    calculation_accuracy: float
    policy_compliance: float
    evidence_quality: float
    hallucination_rate: float  # Lower is better
    
    # Detailed breakdown
    total_claims: int
    verified_claims: int
    total_calculations: int
    correct_calculations: int
    policy_violations: List[str]
    hallucinated_facts: List[str]
    
    # VERAFI benchmark comparison
    verafi_target: float = 0.947  # 94.7% from paper
    baseline_rag: float = 0.524  # 52.4% from paper
    
    @property
    def overall_accuracy(self) -> float:
        """Weighted overall accuracy."""
        return (
            0.35 * self.grounded_accuracy +
            0.25 * self.calculation_accuracy +
            0.20 * self.policy_compliance +
            0.10 * self.evidence_quality +
            0.10 * (1 - self.hallucination_rate)
        )
    
    @property
    def beats_baseline(self) -> bool:
        """Does this system beat baseline RAG?"""
        return self.overall_accuracy > self.baseline_rag
    
    @property
    def meets_verafi(self) -> bool:
        """Does this system meet VERAFI benchmark?"""
        return self.overall_accuracy >= self.verafi_target
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "grounded_accuracy": self.grounded_accuracy,
            "calculation_accuracy": self.calculation_accuracy,
            "policy_compliance": self.policy_compliance,
            "evidence_quality": self.evidence_quality,
            "hallucination_rate": self.hallucination_rate,
            "overall_accuracy": self.overall_accuracy,
            "beats_baseline": self.beats_baseline,
            "meets_verafi": self.meets_verafi,
            "total_claims": self.total_claims,
            "verified_claims": self.verified_claims,
            "policy_violations": self.policy_violations,
            "hallucinated_facts": self.hallucinated_facts,
        }


class VERAFIEvaluator:
    """
    Evaluate RAG outputs against VERAFI standards.
    
    Implements the evaluation methodology from the VERAFI paper
    to measure if your compliance RAG achieves state-of-the-art accuracy.
    """
    
    def __init__(
        self,
        policy_rules: Optional[Dict[str, Any]] = None,
        tolerance_pct: float = 0.05,
    ):
        """
        Initialize evaluator.
        
        Args:
            policy_rules: Dict mapping policy names to rules/limits
            tolerance_pct: Tolerance for numeric comparisons (default 5%)
        """
        self.tolerance_pct = tolerance_pct
        
        # Default compliance policy rules
        self.policy_rules = policy_rules or {
            # Concentration limits
            "single_security_limit": 0.05,
            "sector_limit": 0.25,
            "issuer_limit": 0.10,
            
            # Exposure limits
            "gross_exposure_max": 2.00,
            "net_exposure_min": -1.00,
            "net_exposure_max": 1.00,
            
            # Liquidity
            "min_liquidity": 0.15,
            
            # Risk
            "var_95_limit": 0.02,
            "var_99_limit": 0.03,
        }
    
    def evaluate(
        self,
        question: str,
        answer: str,
        context: str,
        ground_truth: str,
        expected_values: Optional[Dict[str, float]] = None,
        relevant_policies: Optional[List[str]] = None,
    ) -> VERAFIResult:
        """
        Evaluate a single Q&A against VERAFI standards.
        
        Args:
            question: User question
            answer: Generated answer
            context: Retrieved context
            ground_truth: Expected answer
            expected_values: Dict of expected numeric values
            relevant_policies: Policy rules that should be checked
            
        Returns:
            VERAFIResult with all metrics
        """
        # 1. Grounded Accuracy - Are claims backed by evidence?
        grounded, total_claims, verified_claims = self._evaluate_grounding(
            answer, context
        )
        
        # 2. Calculation Accuracy - Is math correct?
        calc_acc, total_calcs, correct_calcs = self._evaluate_calculations(
            answer, expected_values or {}
        )
        
        # 3. Policy Compliance - Are regulatory rules followed?
        policy_score, violations = self._evaluate_policy_compliance(
            answer, relevant_policies or []
        )
        
        # 4. Evidence Quality - Are citations proper?
        evidence_quality = self._evaluate_evidence_quality(answer, context)
        
        # 5. Hallucination Detection
        halluc_rate, halluc_facts = self._detect_hallucinations(
            answer, context, ground_truth
        )
        
        return VERAFIResult(
            grounded_accuracy=grounded,
            calculation_accuracy=calc_acc,
            policy_compliance=policy_score,
            evidence_quality=evidence_quality,
            hallucination_rate=halluc_rate,
            total_claims=total_claims,
            verified_claims=verified_claims,
            total_calculations=total_calcs,
            correct_calculations=correct_calcs,
            policy_violations=violations,
            hallucinated_facts=halluc_facts,
        )
    
    def evaluate_batch(
        self,
        test_cases: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Evaluate a batch of test cases.
        
        Args:
            test_cases: List of dicts with question, answer, context, ground_truth
            
        Returns:
            Aggregate statistics
        """
        results = []
        for case in test_cases:
            result = self.evaluate(
                question=case["question"],
                answer=case["answer"],
                context=case["context"],
                ground_truth=case["ground_truth"],
                expected_values=case.get("expected_values"),
                relevant_policies=case.get("relevant_policies"),
            )
            results.append(result)
        
        # Aggregate
        n = len(results)
        return {
            "count": n,
            "mean_grounded_accuracy": sum(r.grounded_accuracy for r in results) / n,
            "mean_calculation_accuracy": sum(r.calculation_accuracy for r in results) / n,
            "mean_policy_compliance": sum(r.policy_compliance for r in results) / n,
            "mean_evidence_quality": sum(r.evidence_quality for r in results) / n,
            "mean_hallucination_rate": sum(r.hallucination_rate for r in results) / n,
            "mean_overall_accuracy": sum(r.overall_accuracy for r in results) / n,
            "beats_baseline_count": sum(1 for r in results if r.beats_baseline),
            "meets_verafi_count": sum(1 for r in results if r.meets_verafi),
            "beats_baseline_rate": sum(1 for r in results if r.beats_baseline) / n,
            "meets_verafi_rate": sum(1 for r in results if r.meets_verafi) / n,
        }
    
    # === Evaluation Methods ===
    
    def _evaluate_grounding(
        self,
        answer: str,
        context: str,
    ) -> Tuple[float, int, int]:
        """
        Check if answer claims are grounded in context.
        
        Returns: (score, total_claims, verified_claims)
        """
        # Extract claims from answer
        claims = self._extract_claims(answer)
        
        if not claims:
            return 1.0, 0, 0
        
        context_lower = context.lower()
        verified = 0
        
        for claim in claims:
            if self._claim_grounded(claim, context_lower):
                verified += 1
        
        return verified / len(claims), len(claims), verified
    
    def _evaluate_calculations(
        self,
        answer: str,
        expected_values: Dict[str, float],
    ) -> Tuple[float, int, int]:
        """
        Check if calculations in answer are correct.
        
        Returns: (score, total_calcs, correct_calcs)
        """
        if not expected_values:
            return 1.0, 0, 0
        
        answer_numbers = self._extract_numbers(answer)
        correct = 0
        
        for name, expected in expected_values.items():
            for num in answer_numbers:
                if abs(num - expected) / (expected + 1e-10) < self.tolerance_pct:
                    correct += 1
                    break
        
        return correct / len(expected_values), len(expected_values), correct
    
    def _evaluate_policy_compliance(
        self,
        answer: str,
        relevant_policies: List[str],
    ) -> Tuple[float, List[str]]:
        """
        Check if answer correctly interprets policy rules.
        
        Returns: (score, violations)
        """
        if not relevant_policies:
            return 1.0, []
        
        violations = []
        answer_lower = answer.lower()
        
        for policy in relevant_policies:
            if policy in self.policy_rules:
                limit = self.policy_rules[policy]
                
                # Check if answer mentions the correct limit
                if not self._mentions_value(answer_lower, limit):
                    violations.append(f"Missing or incorrect {policy}: {limit}")
        
        score = 1 - (len(violations) / len(relevant_policies))
        return max(0, score), violations
    
    def _evaluate_evidence_quality(
        self,
        answer: str,
        context: str,
    ) -> float:
        """
        Evaluate quality of evidence citations.
        
        Checks for:
        - Explicit citations/references
        - Specific numbers tied to sources
        - Clear attribution
        """
        score = 0.0
        
        # Check for citation markers
        citation_patterns = [
            r'according to',
            r'per (?:the )?policy',
            r'as stated in',
            r'from (?:the )?',
            r'based on',
            r'source:',
            r'\[.*?\]',  # Bracket citations
        ]
        
        for pattern in citation_patterns:
            if re.search(pattern, answer, re.IGNORECASE):
                score += 0.15
        
        # Check for specific number citations
        numbers_in_answer = self._extract_numbers(answer)
        numbers_in_context = self._extract_numbers(context)
        
        if numbers_in_answer:
            matched = sum(
                1 for n in numbers_in_answer 
                if any(abs(n - c) / (c + 1e-10) < 0.01 for c in numbers_in_context)
            )
            score += 0.3 * (matched / len(numbers_in_answer))
        
        # Check for document references
        doc_refs = re.findall(r'(?:policy|document|report|file)[\s:]+\w+', answer, re.IGNORECASE)
        if doc_refs:
            score += 0.2
        
        return min(1.0, score)
    
    def _detect_hallucinations(
        self,
        answer: str,
        context: str,
        ground_truth: str,
    ) -> Tuple[float, List[str]]:
        """
        Detect hallucinated facts not in context or ground truth.
        
        Returns: (hallucination_rate, list_of_hallucinated_facts)
        """
        hallucinated = []
        
        # Extract claims from answer
        claims = self._extract_claims(answer)
        
        if not claims:
            return 0.0, []
        
        context_lower = context.lower()
        truth_lower = ground_truth.lower()
        
        for claim in claims:
            # Check if claim is grounded
            if not self._claim_grounded(claim, context_lower):
                # Also check ground truth
                if not self._claim_grounded(claim, truth_lower):
                    hallucinated.append(claim)
        
        # Also check for fabricated numbers
        answer_numbers = self._extract_numbers(answer)
        context_numbers = self._extract_numbers(context)
        truth_numbers = self._extract_numbers(ground_truth)
        valid_numbers = set(context_numbers) | set(truth_numbers)
        
        for num in answer_numbers:
            is_valid = any(
                abs(num - v) / (v + 1e-10) < 0.05 
                for v in valid_numbers
            )
            if not is_valid and num > 1:  # Ignore small numbers
                hallucinated.append(f"Unsupported number: {num}")
        
        rate = len(hallucinated) / (len(claims) + len(answer_numbers)) if claims or answer_numbers else 0
        return min(1.0, rate), hallucinated
    
    # === Helper Methods ===
    
    def _extract_claims(self, text: str) -> List[str]:
        """Extract verifiable claims from text."""
        sentences = re.split(r'[.!?]\s+', text)
        claims = []
        
        for sent in sentences:
            sent = sent.strip()
            # Claims are sentences with assertions
            if len(sent) > 10 and any(w in sent.lower() for w in ['is', 'are', 'was', 'has', 'exceeds', 'within']):
                claims.append(sent)
        
        return claims
    
    def _claim_grounded(self, claim: str, context: str) -> bool:
        """Check if a claim is grounded in context."""
        # Extract key terms
        words = re.findall(r'\b[a-z]{3,}\b', claim.lower())
        stopwords = {'the', 'and', 'for', 'are', 'but', 'not', 'this', 'that', 'with', 'from'}
        terms = [w for w in words if w not in stopwords]
        
        if not terms:
            return True
        
        # Check term coverage
        found = sum(1 for t in terms if t in context)
        return found / len(terms) >= 0.5
    
    def _extract_numbers(self, text: str) -> List[float]:
        """Extract numeric values from text."""
        numbers = []
        
        # Percentages
        for match in re.findall(r'(\d+(?:\.\d+)?)\s*%', text):
            numbers.append(float(match))
        
        # Millions/billions
        for match in re.findall(r'\$?([\d,]+(?:\.\d+)?)\s*(?:million|M)', text, re.IGNORECASE):
            numbers.append(float(match.replace(',', '')) * 1_000_000)
        
        for match in re.findall(r'\$?([\d,]+(?:\.\d+)?)\s*(?:billion|B)', text, re.IGNORECASE):
            numbers.append(float(match.replace(',', '')) * 1_000_000_000)
        
        # Plain numbers
        for match in re.findall(r'\$?([\d,]+(?:\.\d+)?)', text):
            try:
                num = float(match.replace(',', ''))
                if num not in numbers:
                    numbers.append(num)
            except:
                continue
        
        return numbers
    
    def _mentions_value(self, text: str, value: float) -> bool:
        """Check if text mentions a value (with tolerance)."""
        numbers = self._extract_numbers(text)
        
        # Also check percentage representations
        if value < 1:
            pct_value = value * 100
            numbers_pct = [n for n in numbers if n < 100]
            if any(abs(n - pct_value) < 1 for n in numbers_pct):
                return True
        
        return any(abs(n - value) / (value + 1e-10) < self.tolerance_pct for n in numbers)


# Convenience function
def evaluate_verafi(
    question: str,
    answer: str,
    context: str,
    ground_truth: str,
    expected_values: Optional[Dict[str, float]] = None,
) -> VERAFIResult:
    """
    Quick VERAFI evaluation.
    
    Args:
        question: User question
        answer: Generated answer
        context: Retrieved context
        ground_truth: Expected answer
        expected_values: Expected numeric values
        
    Returns:
        VERAFIResult with all metrics
    """
    evaluator = VERAFIEvaluator()
    return evaluator.evaluate(
        question=question,
        answer=answer,
        context=context,
        ground_truth=ground_truth,
        expected_values=expected_values,
    )
