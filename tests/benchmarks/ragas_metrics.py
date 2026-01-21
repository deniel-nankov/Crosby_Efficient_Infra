"""
RAGAS-Style Metrics Implementation

Implements metrics from the RAGAS framework (Retrieval Augmented Generation Assessment):
- Context Precision: Are retrieved docs relevant to the question?
- Context Recall: Does context contain info needed to answer?
- Faithfulness: Is the answer faithful to the retrieved context?
- Answer Relevancy: Is the answer relevant to the question?
- Answer Correctness: Is the answer factually correct?

Reference: https://arxiv.org/abs/2309.15217
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import Counter
import math


@dataclass
class RAGASScores:
    """RAGAS metric scores for a single query."""
    context_precision: float  # Relevant docs / retrieved docs
    context_recall: float  # Relevant facts retrieved / all relevant facts
    faithfulness: float  # Supported claims / all claims
    answer_relevancy: float  # Answer addresses question
    answer_correctness: float  # Factually correct
    
    @property
    def harmonic_mean(self) -> float:
        """F1-style harmonic mean of all scores."""
        scores = [
            self.context_precision,
            self.context_recall,
            self.faithfulness,
            self.answer_relevancy,
            self.answer_correctness,
        ]
        non_zero = [s for s in scores if s > 0]
        if not non_zero:
            return 0
        return len(non_zero) / sum(1/s for s in non_zero)
    
    @property
    def arithmetic_mean(self) -> float:
        """Simple average of all scores."""
        return (
            self.context_precision +
            self.context_recall +
            self.faithfulness +
            self.answer_relevancy +
            self.answer_correctness
        ) / 5


class RAGASMetrics:
    """
    Calculate RAGAS metrics for RAG system evaluation.
    
    These are simplified, rule-based approximations of the LLM-based
    RAGAS metrics. For production evaluation, use the full RAGAS library.
    """
    
    def __init__(self, domain_terms: Optional[Set[str]] = None):
        """
        Initialize metrics calculator.
        
        Args:
            domain_terms: Set of domain-specific terms for relevancy scoring
        """
        self.domain_terms = domain_terms or {
            # Compliance domain terms
            "concentration", "exposure", "nav", "aum", "liquidity",
            "threshold", "limit", "breach", "compliance", "sec",
            "regulatory", "portfolio", "position", "risk",
            "exception", "approval", "escalation", "policy",
        }
    
    def context_precision(
        self,
        question: str,
        retrieved_docs: List[str],
        ground_truth_docs: Optional[List[str]] = None,
    ) -> float:
        """
        Calculate context precision.
        
        Measures: What fraction of retrieved docs are relevant?
        
        Args:
            question: The user query
            retrieved_docs: List of retrieved document texts
            ground_truth_docs: Optional ground truth relevant docs
            
        Returns:
            Precision score 0-1
        """
        if not retrieved_docs:
            return 0.0
        
        # Extract key terms from question
        question_terms = self._extract_terms(question)
        
        relevant_count = 0
        for doc in retrieved_docs:
            doc_terms = self._extract_terms(doc)
            
            # Check term overlap
            overlap = len(question_terms & doc_terms)
            
            # Also check for domain relevance
            domain_overlap = len(doc_terms & self.domain_terms)
            
            # Consider relevant if significant term overlap
            if overlap >= 2 or (overlap >= 1 and domain_overlap >= 2):
                relevant_count += 1
        
        return relevant_count / len(retrieved_docs)
    
    def context_recall(
        self,
        answer: str,
        context: str,
        ground_truth: Optional[str] = None,
    ) -> float:
        """
        Calculate context recall.
        
        Measures: Does the context contain the information needed?
        
        Args:
            answer: The ground truth answer
            context: Retrieved context
            ground_truth: Optional explicit ground truth
            
        Returns:
            Recall score 0-1
        """
        reference = ground_truth or answer
        
        # Extract facts (sentences/clauses) from reference
        reference_facts = self._extract_facts(reference)
        
        if not reference_facts:
            return 1.0  # No facts to recall
        
        # Check how many facts are supported by context
        supported = 0
        context_lower = context.lower()
        
        for fact in reference_facts:
            # Extract key terms from fact
            fact_terms = self._extract_terms(fact)
            
            # Check if terms appear in context
            found = sum(1 for t in fact_terms if t in context_lower)
            
            if found >= len(fact_terms) * 0.6:  # 60% term coverage
                supported += 1
        
        return supported / len(reference_facts)
    
    def faithfulness(
        self,
        answer: str,
        context: str,
    ) -> float:
        """
        Calculate faithfulness score.
        
        Measures: Is every claim in the answer supported by the context?
        
        Args:
            answer: Generated answer
            context: Retrieved context
            
        Returns:
            Faithfulness score 0-1 (1 = fully faithful)
        """
        # Extract claims from answer
        claims = self._extract_claims(answer)
        
        if not claims:
            return 1.0  # No claims to verify
        
        # Verify each claim against context
        supported = 0
        context_lower = context.lower()
        
        for claim in claims:
            if self._claim_supported(claim, context_lower):
                supported += 1
        
        return supported / len(claims)
    
    def answer_relevancy(
        self,
        question: str,
        answer: str,
    ) -> float:
        """
        Calculate answer relevancy.
        
        Measures: Does the answer address the question?
        
        Args:
            question: User question
            answer: Generated answer
            
        Returns:
            Relevancy score 0-1
        """
        if not answer.strip():
            return 0.0
        
        # Extract question terms
        q_terms = self._extract_terms(question)
        
        # Extract answer terms
        a_terms = self._extract_terms(answer)
        
        if not q_terms:
            return 1.0  # No specific terms to match
        
        # Calculate term overlap ratio
        overlap = len(q_terms & a_terms)
        coverage = overlap / len(q_terms)
        
        # Check for question-answer coherence
        q_type = self._identify_question_type(question)
        coherence = self._check_answer_coherence(answer, q_type)
        
        # Combine scores
        return 0.5 * coverage + 0.5 * coherence
    
    def answer_correctness(
        self,
        answer: str,
        ground_truth: str,
        expected_values: Optional[Dict[str, float]] = None,
    ) -> float:
        """
        Calculate answer correctness.
        
        Measures: Is the answer factually correct?
        
        Args:
            answer: Generated answer
            ground_truth: Ground truth answer
            expected_values: Optional dict of expected numeric values
            
        Returns:
            Correctness score 0-1
        """
        scores = []
        
        # 1. Semantic similarity via term overlap
        a_terms = self._extract_terms(answer)
        gt_terms = self._extract_terms(ground_truth)
        
        if a_terms and gt_terms:
            overlap = len(a_terms & gt_terms)
            union = len(a_terms | gt_terms)
            jaccard = overlap / union if union > 0 else 0
            scores.append(jaccard)
        
        # 2. Numeric accuracy
        if expected_values:
            correct_nums = 0
            for name, expected in expected_values.items():
                # Extract numbers from answer
                numbers = self._extract_numbers(answer)
                
                for num in numbers:
                    # 5% tolerance
                    if abs(num - expected) / (expected + 1e-10) < 0.05:
                        correct_nums += 1
                        break
            
            if expected_values:
                scores.append(correct_nums / len(expected_values))
        
        # 3. Key fact presence
        gt_facts = self._extract_facts(ground_truth)
        if gt_facts:
            found = 0
            answer_lower = answer.lower()
            for fact in gt_facts:
                fact_terms = self._extract_terms(fact)
                if all(t in answer_lower for t in list(fact_terms)[:3]):
                    found += 1
            scores.append(found / len(gt_facts))
        
        return sum(scores) / len(scores) if scores else 0
    
    def calculate_all(
        self,
        question: str,
        answer: str,
        context: str,
        ground_truth: str,
        retrieved_docs: Optional[List[str]] = None,
        expected_values: Optional[Dict[str, float]] = None,
    ) -> RAGASScores:
        """Calculate all RAGAS metrics at once."""
        retrieved = retrieved_docs or [context]
        
        return RAGASScores(
            context_precision=self.context_precision(question, retrieved),
            context_recall=self.context_recall(answer, context, ground_truth),
            faithfulness=self.faithfulness(answer, context),
            answer_relevancy=self.answer_relevancy(question, answer),
            answer_correctness=self.answer_correctness(answer, ground_truth, expected_values),
        )
    
    # === Helper Methods ===
    
    def _extract_terms(self, text: str) -> Set[str]:
        """Extract meaningful terms from text."""
        # Lowercase and tokenize
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        
        # Remove common stopwords
        stopwords = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all',
            'can', 'had', 'her', 'was', 'one', 'our', 'out', 'has',
            'have', 'been', 'were', 'will', 'when', 'who', 'what',
            'this', 'that', 'with', 'from', 'they', 'which', 'their',
            'about', 'would', 'there', 'could', 'other', 'into',
            'than', 'then', 'these', 'some', 'only', 'should', 'such',
        }
        
        return {w for w in words if w not in stopwords}
    
    def _extract_facts(self, text: str) -> List[str]:
        """Extract factual statements from text."""
        # Split into sentences
        sentences = re.split(r'[.!?]\s+', text)
        
        # Filter to meaningful sentences
        facts = []
        for sent in sentences:
            sent = sent.strip()
            if len(sent) > 15 and len(sent.split()) >= 3:
                facts.append(sent)
        
        return facts
    
    def _extract_claims(self, text: str) -> List[str]:
        """Extract verifiable claims from answer."""
        claims = []
        
        # Sentences with numbers are claims
        sentences = re.split(r'[.!?]\s+', text)
        for sent in sentences:
            if re.search(r'\d+', sent):
                claims.append(sent.strip())
        
        # Sentences with definitive statements
        for sent in sentences:
            if any(word in sent.lower() for word in ['is', 'are', 'must', 'shall', 'requires']):
                if sent.strip() not in claims and len(sent.split()) >= 4:
                    claims.append(sent.strip())
        
        return claims
    
    def _claim_supported(self, claim: str, context: str) -> bool:
        """Check if a claim is supported by context."""
        # Extract key terms from claim
        claim_terms = self._extract_terms(claim)
        
        # Extract numbers from claim
        claim_numbers = self._extract_numbers(claim)
        
        # Check term presence
        term_matches = sum(1 for t in claim_terms if t in context)
        term_ratio = term_matches / len(claim_terms) if claim_terms else 1
        
        # Check number presence (exact or close)
        number_supported = True
        for num in claim_numbers:
            # Look for the number or close variants
            found = False
            context_numbers = self._extract_numbers(context)
            for ctx_num in context_numbers:
                if abs(num - ctx_num) / (num + 1e-10) < 0.01:  # 1% tolerance
                    found = True
                    break
            if not found:
                number_supported = False
        
        return term_ratio >= 0.5 and number_supported
    
    def _extract_numbers(self, text: str) -> List[float]:
        """Extract numeric values from text."""
        numbers = []
        
        # Match various number formats
        patterns = [
            r'\$?([\d,]+(?:\.\d+)?)\s*(?:million|M)',  # Millions
            r'\$?([\d,]+(?:\.\d+)?)\s*(?:billion|B)',  # Billions
            r'(\d+(?:\.\d+)?)\s*%',  # Percentages
            r'\$?([\d,]+(?:\.\d+)?)',  # Plain numbers
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                try:
                    num = float(match.replace(',', ''))
                    # Adjust for scale
                    if 'million' in text.lower() or 'M' in text:
                        num *= 1_000_000
                    elif 'billion' in text.lower() or 'B' in text:
                        num *= 1_000_000_000
                    numbers.append(num)
                except:
                    continue
        
        return numbers
    
    def _identify_question_type(self, question: str) -> str:
        """Identify the type of question."""
        q_lower = question.lower()
        
        if q_lower.startswith(('what is', 'what are', "what's")):
            return 'definition'
        elif q_lower.startswith(('how much', 'how many')):
            return 'quantity'
        elif q_lower.startswith(('how do', 'how should', 'how can')):
            return 'procedure'
        elif q_lower.startswith(('why', 'what caused')):
            return 'explanation'
        elif q_lower.startswith(('is ', 'are ', 'does ', 'do ')):
            return 'yes_no'
        else:
            return 'general'
    
    def _check_answer_coherence(self, answer: str, q_type: str) -> float:
        """Check if answer style matches question type."""
        a_lower = answer.lower()
        
        if q_type == 'definition':
            # Should have "is" or "refers to" or similar
            if any(p in a_lower for p in ['is ', 'are ', 'refers to', 'means']):
                return 1.0
            return 0.5
        
        elif q_type == 'quantity':
            # Should have numbers
            if re.search(r'\d+', answer):
                return 1.0
            return 0.3
        
        elif q_type == 'procedure':
            # Should have steps or action words
            if any(w in a_lower for w in ['first', 'then', 'should', 'must', 'need to']):
                return 1.0
            return 0.5
        
        elif q_type == 'yes_no':
            # Should start with yes/no or equivalent
            if any(w in a_lower[:50] for w in ['yes', 'no', 'correct', 'incorrect']):
                return 1.0
            return 0.5
        
        return 0.7  # Default for general questions


# Convenience function
def calculate_ragas_scores(
    question: str,
    answer: str,
    context: str,
    ground_truth: str,
    expected_values: Optional[Dict[str, float]] = None,
) -> RAGASScores:
    """
    Quick calculation of all RAGAS metrics.
    
    Args:
        question: User question
        answer: Generated answer  
        context: Retrieved context (concatenated)
        ground_truth: Expected answer
        expected_values: Optional dict of expected numeric values
        
    Returns:
        RAGASScores dataclass
    """
    metrics = RAGASMetrics()
    return metrics.calculate_all(
        question=question,
        answer=answer,
        context=context,
        ground_truth=ground_truth,
        expected_values=expected_values,
    )
