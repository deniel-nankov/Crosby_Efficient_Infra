"""
Evaluation Runner - Generate comprehensive evaluation reports

Usage:
    python -m tests.eval.run_evaluation
    python -m tests.eval.run_evaluation --output report.json
    python -m tests.eval.run_evaluation --category concentration
"""

import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from collections import defaultdict
import uuid

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from . import (
    COMPLIANCE_EVAL_DATASET,
    EvaluationResult,
    EvaluationReport,
)


def run_evaluation(
    retriever,
    dataset: Optional[List[Dict]] = None,
    category_filter: Optional[str] = None,
    verbose: bool = True,
) -> EvaluationReport:
    """
    Run full evaluation and generate report.
    
    Args:
        retriever: RAGRetriever instance
        dataset: Evaluation dataset (defaults to COMPLIANCE_EVAL_DATASET)
        category_filter: Optional category to filter
        verbose: Print progress
        
    Returns:
        EvaluationReport with all results
    """
    dataset = dataset or COMPLIANCE_EVAL_DATASET
    
    if category_filter:
        dataset = [d for d in dataset if d["category"] == category_filter]
    
    report_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    results: List[EvaluationResult] = []
    
    if verbose:
        print(f"Starting evaluation {report_id} with {len(dataset)} queries...")
        print()
    
    for i, test_case in enumerate(dataset):
        query_id = test_case["id"]
        query = test_case["query"]
        expected_docs = set(test_case["relevant_documents"])
        
        if verbose:
            print(f"[{i+1}/{len(dataset)}] {query_id}: {query[:50]}...", end=" ")
        
        # Run retrieval
        start = time.time()
        try:
            result = retriever.retrieve_for_control(
                control_name=query,
                control_type=test_case["category"],
                status="pass",
                calculated_value=0.0,
                threshold=0.0,
                limit=5,
            )
            latency_ms = (time.time() - start) * 1000
            
            retrieved_docs = [c.document_name for c in result.chunks]
            retrieved_set = set(retrieved_docs)
            
            # Calculate metrics
            hits = retrieved_set & expected_docs
            retrieval_hit = len(hits) > 0
            precision = len(hits) / len(retrieved_docs) if retrieved_docs else 0
            recall = len(hits) / len(expected_docs) if expected_docs else 0
            
            # Determine pass/fail
            passed = retrieval_hit
            failure_reason = None if passed else f"Expected {expected_docs}, got {retrieved_set}"
            
            eval_result = EvaluationResult(
                query_id=query_id,
                query=query,
                category=test_case["category"],
                difficulty=test_case["difficulty"],
                retrieved_documents=retrieved_docs,
                expected_documents=list(expected_docs),
                retrieval_hit=retrieval_hit,
                retrieval_precision=precision,
                retrieval_recall=recall,
                retrieval_latency_ms=latency_ms,
                passed=passed,
                failure_reason=failure_reason,
            )
            
        except Exception as e:
            latency_ms = (time.time() - start) * 1000
            eval_result = EvaluationResult(
                query_id=query_id,
                query=query,
                category=test_case["category"],
                difficulty=test_case["difficulty"],
                retrieved_documents=[],
                expected_documents=list(expected_docs),
                retrieval_hit=False,
                retrieval_precision=0.0,
                retrieval_recall=0.0,
                retrieval_latency_ms=latency_ms,
                passed=False,
                failure_reason=f"Error: {str(e)}",
            )
        
        results.append(eval_result)
        
        if verbose:
            status = "✓" if eval_result.passed else "✗"
            print(f"{status} ({latency_ms:.0f}ms)")
    
    # Aggregate results
    passed_count = sum(1 for r in results if r.passed)
    failed_count = len(results) - passed_count
    
    # By category
    category_results = defaultdict(lambda: {"passed": 0, "total": 0})
    for r in results:
        category_results[r.category]["total"] += 1
        if r.passed:
            category_results[r.category]["passed"] += 1
    
    category_metrics = {}
    for cat, data in category_results.items():
        category_metrics[cat] = {
            "pass_rate": data["passed"] / data["total"] if data["total"] > 0 else 0,
            "count": data["total"],
        }
    
    # By difficulty
    difficulty_results = defaultdict(lambda: {"passed": 0, "total": 0})
    for r in results:
        difficulty_results[r.difficulty]["total"] += 1
        if r.passed:
            difficulty_results[r.difficulty]["passed"] += 1
    
    difficulty_metrics = {}
    for diff, data in difficulty_results.items():
        difficulty_metrics[diff] = {
            "pass_rate": data["passed"] / data["total"] if data["total"] > 0 else 0,
            "count": data["total"],
        }
    
    # Aggregate metrics
    mean_precision = sum(r.retrieval_precision for r in results) / len(results)
    mean_recall = sum(r.retrieval_recall for r in results) / len(results)
    mean_latency = sum(r.retrieval_latency_ms for r in results) / len(results)
    
    report = EvaluationReport(
        report_id=report_id,
        run_at=datetime.now(),
        total_queries=len(results),
        passed=passed_count,
        failed=failed_count,
        pass_rate=passed_count / len(results) if results else 0,
        category_results=category_metrics,
        difficulty_results=difficulty_metrics,
        results=results,
        mean_retrieval_precision=mean_precision,
        mean_retrieval_recall=mean_recall,
        mean_retrieval_latency_ms=mean_latency,
        mean_answer_faithfulness=0.0,  # TODO: implement generation evaluation
    )
    
    if verbose:
        print()
        print("=" * 60)
        print(f"EVALUATION COMPLETE: {report_id}")
        print("=" * 60)
        print(f"Pass Rate: {report.pass_rate:.1%} ({passed_count}/{len(results)})")
        print(f"Mean Precision: {mean_precision:.1%}")
        print(f"Mean Recall: {mean_recall:.1%}")
        print(f"Mean Latency: {mean_latency:.0f}ms")
        print()
        print("By Category:")
        for cat, metrics in category_metrics.items():
            print(f"  {cat}: {metrics['pass_rate']:.1%} ({int(metrics['count'])} queries)")
        print()
        print("By Difficulty:")
        for diff, metrics in difficulty_metrics.items():
            print(f"  {diff}: {metrics['pass_rate']:.1%} ({int(metrics['count'])} queries)")
    
    return report


def main():
    parser = argparse.ArgumentParser(description="Run RAG evaluation")
    parser.add_argument("--output", "-o", type=Path, help="Output file for report (JSON)")
    parser.add_argument("--markdown", "-m", type=Path, help="Output file for report (Markdown)")
    parser.add_argument("--category", "-c", type=str, help="Filter by category")
    parser.add_argument("--quiet", "-q", action="store_true", help="Quiet mode")
    
    args = parser.parse_args()
    
    # Initialize retriever
    try:
        from rag import RAGRetriever, VectorStore, LocalEmbedder
        
        connection_params = {
            "host": "localhost",
            "port": 5432,
            "database": "compliance",
            "user": "compliance_user",
            "password": "compliance_dev_password_123",
        }
        
        vector_store = VectorStore(connection_params)
        embedder = LocalEmbedder()
        
        if not embedder.available:
            print("ERROR: Embedding service not available")
            print("Please start LM Studio with an embedding model loaded.")
            sys.exit(1)
        
        retriever = RAGRetriever(
            vector_store=vector_store,
            embedder=embedder,
            use_hybrid=True,
            use_reranking=False,
        )
    except Exception as e:
        print(f"ERROR: Could not initialize retriever: {e}")
        sys.exit(1)
    
    # Run evaluation
    report = run_evaluation(
        retriever=retriever,
        category_filter=args.category,
        verbose=not args.quiet,
    )
    
    # Save outputs
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"Saved JSON report to {args.output}")
    
    if args.markdown:
        with open(args.markdown, 'w') as f:
            f.write(report.to_markdown())
        print(f"Saved Markdown report to {args.markdown}")
    
    # Exit with error if pass rate too low
    if report.pass_rate < 0.70:
        print(f"\nWARNING: Pass rate {report.pass_rate:.1%} is below 70% threshold")
        sys.exit(1)


if __name__ == "__main__":
    main()
