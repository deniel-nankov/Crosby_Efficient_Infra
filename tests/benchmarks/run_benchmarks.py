"""
Benchmark Runner - Execute comprehensive RAG benchmarks

This runner tests your RAG system against industry-standard benchmarks:
- Retrieval Quality
- Noise Robustness
- Negative Rejection
- Counterfactual Robustness
- Information Integration
- Mathematical Accuracy
- Temporal Consistency
- Faithfulness

Usage:
    python -m tests.benchmarks.run_benchmarks
    python -m tests.benchmarks.run_benchmarks --category mathematical_accuracy
    python -m tests.benchmarks.run_benchmarks --output report.json
"""

import sys
import json
import argparse
import time
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
import uuid

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tests.benchmarks import (
    BenchmarkCase,
    BenchmarkResult,
    BenchmarkReport,
    BenchmarkCategory,
    Difficulty,
    BENCHMARK_DATASET,
)


class BenchmarkRunner:
    """
    Runs benchmark suite against a RAG system.
    
    Evaluates:
    1. Retrieval quality (precision, recall, hit rate)
    2. Generation quality (faithfulness, correctness)
    3. Robustness (noise, negatives, counterfactuals)
    4. Accuracy (mathematical, temporal)
    """
    
    def __init__(
        self,
        retriever,
        generator=None,
        verbose: bool = True,
    ):
        """
        Initialize benchmark runner.
        
        Args:
            retriever: RAGRetriever instance
            generator: Optional NarrativeGenerator for full pipeline testing
            verbose: Print progress
        """
        self.retriever = retriever
        self.generator = generator
        self.verbose = verbose
    
    def run(
        self,
        dataset: Optional[List[BenchmarkCase]] = None,
        category_filter: Optional[BenchmarkCategory] = None,
        difficulty_filter: Optional[Difficulty] = None,
    ) -> BenchmarkReport:
        """
        Run benchmark suite.
        
        Args:
            dataset: Benchmark cases (defaults to BENCHMARK_DATASET)
            category_filter: Only run specific category
            difficulty_filter: Only run specific difficulty
            
        Returns:
            BenchmarkReport with all results
        """
        dataset = dataset or BENCHMARK_DATASET
        
        if category_filter:
            dataset = [d for d in dataset if d.category == category_filter]
        
        if difficulty_filter:
            dataset = [d for d in dataset if d.difficulty == difficulty_filter]
        
        report_id = f"bench_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        results: List[BenchmarkResult] = []
        
        if self.verbose:
            print(f"{'=' * 70}")
            print(f"RAG BENCHMARK SUITE")
            print(f"{'=' * 70}")
            print(f"Report ID: {report_id}")
            print(f"Test Cases: {len(dataset)}")
            print(f"Categories: {len(set(d.category for d in dataset))}")
            print(f"{'=' * 70}")
            print()
        
        for i, case in enumerate(dataset):
            if self.verbose:
                print(f"[{i+1}/{len(dataset)}] {case.id} ({case.category.value})...", end=" ")
            
            result = self._run_case(case)
            results.append(result)
            
            if self.verbose:
                status = "✅" if result.passed else "❌"
                print(f"{status} ({result.latency_ms:.0f}ms)")
                if not result.passed and result.failure_reasons:
                    for reason in result.failure_reasons[:2]:
                        print(f"    └─ {reason}")
        
        # Build report
        report = self._build_report(report_id, results)
        
        if self.verbose:
            print()
            print(report.to_markdown())
        
        return report
    
    def _run_case(self, case: BenchmarkCase) -> BenchmarkResult:
        """Run a single benchmark case."""
        start = time.time()
        failure_reasons = []
        
        # Run retrieval
        try:
            retrieval_result = self.retriever.retrieve_for_control(
                control_name=case.query,
                control_type=case.category.value,
                status="pass",
                calculated_value=0.0,
                threshold=0.0,
                limit=5,
            )
            chunks = retrieval_result.chunks if hasattr(retrieval_result, 'chunks') else []
            retrieved_docs = [c.document_name for c in chunks]
        except Exception as e:
            retrieved_docs = []
            failure_reasons.append(f"Retrieval error: {str(e)[:50]}")
        
        latency_ms = (time.time() - start) * 1000
        
        # Calculate retrieval metrics
        expected_docs = set(case.context_documents)
        retrieved_set = set(retrieved_docs)
        hits = retrieved_set & expected_docs
        
        retrieval_precision = len(hits) / len(retrieved_docs) if retrieved_docs else 0
        retrieval_recall = len(hits) / len(expected_docs) if expected_docs else 1.0  # If no expected, recall=1
        retrieval_hit = len(hits) > 0 or len(expected_docs) == 0
        
        # Generate answer if generator available
        generated_answer = None
        if self.generator and chunks:
            try:
                # This would use the narrative generator
                # For now, we'll use the retrieved context as proxy
                generated_answer = "\n".join(c.content[:200] for c in chunks[:3])
            except Exception as e:
                failure_reasons.append(f"Generation error: {str(e)[:50]}")
        
        # Check expected content
        contains_expected = []
        answer_text = generated_answer or "\n".join(c.content for c in chunks) if chunks else ""
        answer_lower = answer_text.lower()
        
        for term in case.expected_answer_contains:
            found = term.lower() in answer_lower
            contains_expected.append((term, found))
            if not found:
                failure_reasons.append(f"Missing expected term: '{term}'")
        
        # Check unexpected content (noise robustness)
        contains_unexpected = []
        for term in case.expected_answer_not_contains:
            found = term.lower() in answer_lower
            contains_unexpected.append((term, found))
            if found:
                failure_reasons.append(f"Found unexpected term (noise): '{term}'")
        
        # Check negative rejection
        refused_correctly = None
        if case.should_refuse:
            # Check if system refused to answer
            refusal_indicators = ["cannot", "unable", "no information", "not available", "don't have"]
            refused = any(ind in answer_lower for ind in refusal_indicators)
            refused_correctly = refused
            if not refused:
                failure_reasons.append("Should have refused to answer but didn't")
        
        # Check mathematical accuracy
        numeric_accuracy = None
        if case.expected_numeric_values:
            numeric_accuracy = {}
            for name, expected in case.expected_numeric_values.items():
                # Try to extract numbers from answer
                numbers = re.findall(r'\$?([\d,]+(?:\.\d+)?)\s*(?:million|M|m)?', answer_text)
                actual = None
                correct = False
                
                for num_str in numbers:
                    try:
                        num = float(num_str.replace(',', ''))
                        # Check if it's in millions
                        if 'million' in answer_lower or 'M' in answer_text:
                            num *= 1_000_000
                        if abs(num - expected) / expected < 0.05:  # 5% tolerance
                            actual = num
                            correct = True
                            break
                    except:
                        continue
                
                numeric_accuracy[name] = {"expected": expected, "actual": actual, "correct": correct}
                if not correct:
                    failure_reasons.append(f"Math error: {name} expected {expected:,.0f}, got {actual}")
        
        # Check counterfactual robustness
        resisted_counterfactual = None
        if case.counterfactual_claim:
            # Check if the correct answer is present, not the counterfactual
            resisted_counterfactual = all(
                term.lower() in answer_lower for term in case.expected_answer_contains
            ) and all(
                term.lower() not in answer_lower for term in case.expected_answer_not_contains
            )
            if not resisted_counterfactual:
                failure_reasons.append("Failed counterfactual robustness check")
        
        # Determine pass/fail
        passed = len(failure_reasons) == 0
        if not passed and not failure_reasons:
            # Default failures
            if not retrieval_hit and expected_docs:
                failure_reasons.append("Retrieval miss")
        
        return BenchmarkResult(
            case_id=case.id,
            category=case.category.value,
            difficulty=case.difficulty.value,
            retrieved_docs=retrieved_docs,
            retrieval_precision=retrieval_precision,
            retrieval_recall=retrieval_recall,
            retrieval_hit=retrieval_hit,
            generated_answer=generated_answer,
            contains_expected=contains_expected,
            contains_unexpected=contains_unexpected,
            numeric_accuracy=numeric_accuracy,
            refused_correctly=refused_correctly,
            resisted_counterfactual=resisted_counterfactual,
            passed=passed,
            failure_reasons=failure_reasons,
            latency_ms=latency_ms,
        )
    
    def _build_report(
        self,
        report_id: str,
        results: List[BenchmarkResult],
    ) -> BenchmarkReport:
        """Build comprehensive benchmark report."""
        
        passed = sum(1 for r in results if r.passed)
        failed = len(results) - passed
        
        # By category
        category_groups = defaultdict(list)
        for r in results:
            category_groups[r.category].append(r)
        
        category_scores = {}
        for cat, cat_results in category_groups.items():
            cat_passed = sum(1 for r in cat_results if r.passed)
            category_scores[cat] = {
                "score": cat_passed / len(cat_results) if cat_results else 0,
                "passed": cat_passed,
                "total": len(cat_results),
            }
        
        # By difficulty
        difficulty_groups = defaultdict(list)
        for r in results:
            difficulty_groups[r.difficulty].append(r)
        
        difficulty_scores = {}
        for diff, diff_results in difficulty_groups.items():
            diff_passed = sum(1 for r in diff_results if r.passed)
            difficulty_scores[diff] = {
                "score": diff_passed / len(diff_results) if diff_results else 0,
                "passed": diff_passed,
                "total": len(diff_results),
            }
        
        # Aggregate metrics
        mean_precision = sum(r.retrieval_precision for r in results) / len(results) if results else 0
        mean_recall = sum(r.retrieval_recall for r in results) / len(results) if results else 0
        
        # Calculate specialized metrics
        math_results = [r for r in results if r.numeric_accuracy is not None]
        if math_results:
            math_correct = sum(
                1 for r in math_results 
                if all(v["correct"] for v in r.numeric_accuracy.values())
            )
            math_accuracy = math_correct / len(math_results)
        else:
            math_accuracy = 1.0  # No math tests
        
        neg_results = [r for r in results if r.refused_correctly is not None]
        neg_rejection_rate = (
            sum(1 for r in neg_results if r.refused_correctly) / len(neg_results)
            if neg_results else 1.0
        )
        
        noise_results = [r for r in results if r.contains_unexpected]
        if noise_results:
            noise_robust = sum(
                1 for r in noise_results 
                if not any(found for _, found in r.contains_unexpected)
            )
            noise_robustness = noise_robust / len(noise_results)
        else:
            noise_robustness = 1.0
        
        # Faithfulness (proxy: expected terms found)
        faith_scores = []
        for r in results:
            if r.contains_expected:
                found = sum(1 for _, f in r.contains_expected if f)
                faith_scores.append(found / len(r.contains_expected))
        mean_faithfulness = sum(faith_scores) / len(faith_scores) if faith_scores else 0
        
        return BenchmarkReport(
            report_id=report_id,
            run_at=datetime.now(),
            total_cases=len(results),
            passed=passed,
            failed=failed,
            overall_score=passed / len(results) if results else 0,
            category_scores=category_scores,
            difficulty_scores=difficulty_scores,
            results=results,
            mean_retrieval_precision=mean_precision,
            mean_retrieval_recall=mean_recall,
            mean_faithfulness=mean_faithfulness,
            mathematical_accuracy_rate=math_accuracy,
            negative_rejection_rate=neg_rejection_rate,
            noise_robustness_rate=noise_robustness,
        )


def main():
    """Run benchmarks from command line."""
    parser = argparse.ArgumentParser(description="Run RAG benchmark suite")
    parser.add_argument("--category", type=str, help="Filter by category")
    parser.add_argument("--difficulty", type=str, choices=["easy", "medium", "hard"])
    parser.add_argument("--output", type=str, help="Output file for JSON report")
    parser.add_argument("--markdown", type=str, help="Output file for markdown report")
    args = parser.parse_args()
    
    # Import RAG components
    try:
        from src.rag import RAGRetriever, VectorStore, LocalEmbedder
        from src.config import get_settings
        
        settings = get_settings()
        
        # Initialize retriever
        embedder = LocalEmbedder()
        vector_store = VectorStore(settings.database_url)
        retriever = RAGRetriever(vector_store, embedder)
        
        print("RAG system initialized successfully")
        print()
        
    except Exception as e:
        print(f"Error initializing RAG system: {e}")
        print("Running with mock retriever...")
        
        # Mock retriever for testing
        class MockRetriever:
            def retrieve_for_control(self, **kwargs):
                class MockResult:
                    chunks = []
                return MockResult()
        
        retriever = MockRetriever()
    
    # Convert category filter
    category_filter = None
    if args.category:
        try:
            category_filter = BenchmarkCategory(args.category)
        except ValueError:
            print(f"Unknown category: {args.category}")
            print(f"Available: {[c.value for c in BenchmarkCategory]}")
            return
    
    # Convert difficulty filter
    difficulty_filter = None
    if args.difficulty:
        difficulty_filter = Difficulty(args.difficulty)
    
    # Run benchmarks
    runner = BenchmarkRunner(retriever, verbose=True)
    report = runner.run(
        category_filter=category_filter,
        difficulty_filter=difficulty_filter,
    )
    
    # Save outputs
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"\nJSON report saved to: {args.output}")
    
    if args.markdown:
        with open(args.markdown, 'w') as f:
            f.write(report.to_markdown())
        print(f"Markdown report saved to: {args.markdown}")
    
    # Exit code based on pass rate
    if report.overall_score < 0.7:
        print(f"\n⚠️  WARNING: Pass rate {report.overall_score:.1%} below 70% threshold")
        sys.exit(1)
    else:
        print(f"\n✅ Benchmark passed with {report.overall_score:.1%} score")
        sys.exit(0)


if __name__ == "__main__":
    main()
