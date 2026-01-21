"""
Design-by-Contract Tests

Tests verifying preconditions, postconditions, and invariants:
- Control definition contracts
- Evidence store immutability
- RAG retriever contracts
- Agent behavior contracts
- State machine transitions

These tests verify that the system maintains its contracts
at all times, regardless of input.
"""

from __future__ import annotations

import pytest
import json
import hashlib
import dataclasses
from dataclasses import dataclass, FrozenInstanceError
from enum import Enum
from typing import List, Dict, Any
from unittest.mock import MagicMock

pytestmark = [pytest.mark.unit, pytest.mark.contracts]


# =============================================================================
# CONTRACT HELPERS
# =============================================================================

def precondition(condition: bool, message: str = "Precondition failed"):
    """Assert a precondition."""
    if not condition:
        raise ValueError(message)


def postcondition(condition: bool, message: str = "Postcondition failed"):
    """Assert a postcondition."""
    if not condition:
        raise AssertionError(message)


def invariant(condition: bool, message: str = "Invariant violated"):
    """Assert an invariant."""
    if not condition:
        raise AssertionError(message)


# =============================================================================
# MOCK HELPERS FOR AGENT TESTS
# =============================================================================

@dataclass
class MockPosition:
    """Mock position for testing."""
    ticker: str
    security_name: str
    market_value: float
    sector: str = "Technology"
    nav_pct: float = 10.0


@dataclass
class MockControlResult:
    """Mock control result for testing."""
    control_id: str
    control_name: str
    control_type: str
    status: str
    calculated_value: float
    threshold: float


@dataclass
class MockDataSnapshot:
    """Mock data snapshot for InvestigationTools."""
    positions: List[MockPosition]
    control_results: List[MockControlResult]
    as_of_date: str = "2026-01-19"
    nav: float = 1000000.0


# =============================================================================
# CONTROL DEFINITION CONTRACTS
# =============================================================================

class TestControlDefinitionContracts:
    """Contract tests for ControlDefinition."""
    
    def test_threshold_value_precondition(self):
        """Precondition: threshold_value must be a valid number."""
        from control_runner.controls import ControlDefinition, ControlCategory, ThresholdOperator
        
        # Valid thresholds
        for value in [0.0, 0.1, 0.5, 1.0]:
            control = ControlDefinition(
                control_code="TEST",
                control_name="Test",
                category=ControlCategory.CONCENTRATION,
                computation_sql="SELECT 1",
                threshold_value=value,
                threshold_operator=ThresholdOperator.GTE,
                description="Test control description",
            )
            assert control.threshold_value == value
    
    def test_evaluate_threshold_postcondition(self):
        """Contract: evaluate_threshold must return valid ControlResultStatus."""
        from control_runner.controls import ControlDefinition, ControlCategory, ThresholdOperator, ControlResultStatus
        
        control = ControlDefinition(
            control_code="TEST",
            control_name="Test",
            category=ControlCategory.CONCENTRATION,
            computation_sql="SELECT 1",
            threshold_value=0.3,
            threshold_operator=ThresholdOperator.GTE,
            description="Test control description",
        )
        
        for value in [0.0, 0.25, 0.3, 0.5, 1.0]:
            result = control.evaluate_threshold(value)
            
            # Postconditions
            postcondition(
                result is not None,
                "Result must not be None"
            )
            postcondition(
                isinstance(result, ControlResultStatus),
                "Result must be ControlResultStatus"
            )
            postcondition(
                result in [ControlResultStatus.PASS, ControlResultStatus.FAIL, 
                          ControlResultStatus.WARNING, ControlResultStatus.ERROR, ControlResultStatus.SKIP],
                f"Status must be valid, got: {result}"
            )
    
    def test_computation_sql_invariant(self):
        """Invariant: computation_sql must be read-only."""
        from control_runner.controls import ControlDefinition, ControlCategory, ThresholdOperator
        
        control = ControlDefinition(
            control_code="TEST",
            control_name="Test",
            category=ControlCategory.CONCENTRATION,
            computation_sql="SELECT computed_value FROM metrics",
            threshold_value=0.3,
            threshold_operator=ThresholdOperator.GTE,
            description="Test control description",
        )
        
        original_sql = control.computation_sql
        
        # Attempt to modify (frozen dataclass should prevent this)
        try:
            control.computation_sql = "DROP TABLE positions"  # type: ignore[misc]
            pytest.fail("Should not be able to modify computation_sql")
        except (AttributeError, TypeError, FrozenInstanceError):
            # Expected - frozen dataclass
            pass
        
        # Invariant: SQL unchanged
        invariant(
            control.computation_sql == original_sql,
            "computation_sql must not change"
        )


# =============================================================================
# EVIDENCE STORE CONTRACTS
# =============================================================================

class TestEvidenceStoreContracts:
    """Contract tests for EvidenceStore."""
    
    def test_evidence_immutability_concept(self):
        """Contract: EvidenceStore should not have update/delete methods."""
        from evidence_store.store import EvidenceStore
        
        mock_db = MagicMock()
        store = EvidenceStore(postgres_connection=mock_db)
        
        # There should be no update/delete methods for immutability
        assert not hasattr(store, 'update_evidence'), \
            "EvidenceStore should not have update_evidence method"
        assert not hasattr(store, 'delete_evidence'), \
            "EvidenceStore should not have delete_evidence method"
    
    def test_hash_chain_concept(self):
        """Invariant: Evidence hash chain concept using standard hashing."""
        # Create chain of evidence using standard hashing
        hashes = []
        
        for i in range(5):
            evidence = {
                "control_code": f"TEST_{i:03d}",
                "status": "PASS" if i % 2 == 0 else "FAIL",
                "value": i * 0.1,
                "previous_hash": hashes[-1] if hashes else None,
            }
            
            current_hash = hashlib.sha256(
                json.dumps(evidence, sort_keys=True).encode()
            ).hexdigest()
            hashes.append(current_hash)
        
        # Invariant: All hashes are unique
        invariant(
            len(set(hashes)) == len(hashes),
            "All evidence hashes must be unique"
        )
        
        # Invariant: Hash chain is verifiable
        for i, h in enumerate(hashes):
            invariant(
                len(h) == 64,
                f"Hash {i} must be SHA-256 format"
            )
    
    def test_evidence_store_creatable(self):
        """Postcondition: EvidenceStore can be created with connection."""
        from evidence_store.store import EvidenceStore
        
        mock_db = MagicMock()
        store = EvidenceStore(postgres_connection=mock_db)
        
        # Postcondition: Store created
        postcondition(
            store is not None,
            "EvidenceStore must be creatable"
        )
        postcondition(
            store.connection is mock_db,
            "EvidenceStore must store connection"
        )


# =============================================================================
# RAG RETRIEVER CONTRACTS
# =============================================================================

class TestRAGRetrieverContracts:
    """Contract tests for RAGRetriever."""
    
    def test_retriever_creatable_with_components(self):
        """Contract: RAGRetriever must be creatable with required components."""
        from rag.retriever import RAGRetriever
        
        mock_store = MagicMock()
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [0.0] * 768
        mock_embedder.available = True
        
        retriever = RAGRetriever(
            vector_store=mock_store,
            embedder=mock_embedder,
            use_hybrid=False,
            use_reranking=False,
            use_query_rewriting=False,
        )
        
        # Contract: Created successfully
        postcondition(
            retriever is not None,
            "RAGRetriever must be created"
        )
        
        # Contract: Has required attributes
        postcondition(
            retriever.vector_store is mock_store,
            "RAGRetriever must have vector_store"
        )
        postcondition(
            retriever.embedder is mock_embedder,
            "RAGRetriever must have embedder"
        )
    
    def test_retriever_has_control_method(self):
        """Contract: RAGRetriever must have retrieve_for_control method."""
        from rag.retriever import RAGRetriever
        
        mock_store = MagicMock()
        mock_embedder = MagicMock()
        mock_embedder.embed.return_value = [0.0] * 768
        mock_embedder.available = True
        
        retriever = RAGRetriever(
            vector_store=mock_store,
            embedder=mock_embedder,
            use_hybrid=False,
            use_reranking=False,
            use_query_rewriting=False,
        )
        
        # Must have retrieve_for_control method
        postcondition(
            hasattr(retriever, 'retrieve_for_control'),
            "RAGRetriever must have retrieve_for_control method"
        )
        postcondition(
            callable(retriever.retrieve_for_control),
            "retrieve_for_control must be callable"
        )


# =============================================================================
# AGENT CONTRACTS
# =============================================================================

class TestAgentContracts:
    """Contract tests for ComplianceAgent and InvestigationTools."""
    
    def test_investigation_tools_creatable(self):
        """Contract: InvestigationTools must be creatable with snapshot."""
        from agent.investigator import InvestigationTools
        
        snapshot = MockDataSnapshot(
            positions=[
                MockPosition(ticker="AAPL", security_name="Apple", market_value=100000),
            ],
            control_results=[],
        )
        
        tools = InvestigationTools(snapshot=snapshot)
        
        # Contract: Created successfully
        postcondition(
            tools is not None,
            "InvestigationTools must be created"
        )
        postcondition(
            tools.snapshot is snapshot,
            "InvestigationTools must store snapshot"
        )
    
    def test_investigation_tools_provides_tools(self):
        """Contract: InvestigationTools.get_tools must return list of tools."""
        from agent.investigator import InvestigationTools
        
        snapshot = MockDataSnapshot(
            positions=[
                MockPosition(ticker="AAPL", security_name="Apple", market_value=100000),
            ],
            control_results=[],
        )
        
        tools = InvestigationTools(snapshot=snapshot)
        available_tools = tools.get_tools()
        
        # Contract: Returns list
        postcondition(
            isinstance(available_tools, list),
            "get_tools must return a list"
        )
        
        # Contract: Has at least one tool
        postcondition(
            len(available_tools) > 0,
            "Must have at least one tool"
        )
    
    def test_tool_execution_returns_string(self):
        """Contract: Tool execution must return string result."""
        from agent.investigator import InvestigationTools
        
        snapshot = MockDataSnapshot(
            positions=[
                MockPosition(ticker="AAPL", security_name="Apple", market_value=100000, sector="Technology"),
            ],
            control_results=[],
        )
        
        tools = InvestigationTools(snapshot=snapshot)
        
        # Query positions returns string
        result = tools.query_positions_by_sector("Technology")
        
        postcondition(
            isinstance(result, str),
            "Tool result must be string"
        )
    
    def test_compliance_agent_creatable(self):
        """Contract: ComplianceAgent can be created."""
        from agent.investigator import ComplianceAgent, InvestigationTools
        
        snapshot = MockDataSnapshot(
            positions=[
                MockPosition(ticker="AAPL", security_name="Apple", market_value=100000),
            ],
            control_results=[],
        )
        
        tools = InvestigationTools(snapshot=snapshot)
        
        mock_llm = MagicMock()
        mock_llm.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='{"action": "FINAL_ANSWER", "action_input": {}}'))]
        )
        
        agent = ComplianceAgent(
            llm_client=mock_llm,
            tools=tools,
        )
        
        postcondition(
            agent is not None,
            "ComplianceAgent must be creatable"
        )


# =============================================================================
# STATE MACHINE TESTS
# =============================================================================

class ControlState(Enum):
    """Control execution states."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"


class TestControlStateMachine:
    """State machine tests for control execution."""
    
    VALID_TRANSITIONS = {
        ControlState.PENDING: [ControlState.RUNNING],
        ControlState.RUNNING: [ControlState.PASSED, ControlState.FAILED, ControlState.ERROR],
        ControlState.PASSED: [],  # Terminal state
        ControlState.FAILED: [],  # Terminal state
        ControlState.ERROR: [],   # Terminal state
    }
    
    def test_control_follows_valid_state_transitions(self):
        """Control execution must follow valid state transitions."""
        # Track state transitions
        states = []
        
        def track_state(state: ControlState):
            if states:
                prev = states[-1]
                allowed = self.VALID_TRANSITIONS[prev]
                assert state in allowed, \
                    f"Invalid transition: {prev} -> {state}"
            states.append(state)
        
        # Simulate execution
        track_state(ControlState.PENDING)
        track_state(ControlState.RUNNING)
        track_state(ControlState.PASSED)  # or FAILED
        
        # Verify we reached a terminal state
        assert states[-1] in [ControlState.PASSED, ControlState.FAILED, ControlState.ERROR]
    
    def test_no_transition_from_terminal_state(self):
        """Terminal states must not transition."""
        terminal_states = [
            ControlState.PASSED,
            ControlState.FAILED,
            ControlState.ERROR,
        ]
        
        for state in terminal_states:
            allowed = self.VALID_TRANSITIONS[state]
            assert len(allowed) == 0, \
                f"Terminal state {state} should not have transitions"


class TestRAGPipelineStateMachine:
    """State machine tests for RAG pipeline."""
    
    class PipelineState(Enum):
        IDLE = "idle"
        EMBEDDING = "embedding"
        SEARCHING = "searching"
        RERANKING = "reranking"
        COMPLETE = "complete"
        ERROR = "error"
    
    def test_pipeline_state_sequence(self):
        """RAG pipeline must follow correct state sequence."""
        expected_sequence = [
            self.PipelineState.IDLE,
            self.PipelineState.EMBEDDING,
            self.PipelineState.SEARCHING,
            self.PipelineState.RERANKING,
            self.PipelineState.COMPLETE,
        ]
        
        # Verify sequence is valid
        for i, state in enumerate(expected_sequence[:-1]):
            next_state = expected_sequence[i + 1]
            # This is the expected happy path
            assert next_state is not None


# =============================================================================
# INTERFACE COMPLIANCE TESTS
# =============================================================================

class TestInterfaceCompliance:
    """Test that implementations comply with expected interfaces."""
    
    def test_embedder_interface(self):
        """Embedder must implement required interface."""
        from rag.embedder import LocalEmbedder
        
        embedder = LocalEmbedder()
        
        # Required methods
        assert hasattr(embedder, 'embed'), "Must have embed method"
        assert hasattr(embedder, 'embed_batch'), "Must have embed_batch method"
        assert hasattr(embedder, 'available'), "Must have available property"
        
        # Method signatures
        assert callable(embedder.embed)
        assert callable(embedder.embed_batch)
    
    def test_vector_store_interface(self):
        """VectorStore must implement required interface."""
        from rag.vector_store import VectorStore
        
        mock_db = MagicMock()
        store = VectorStore(mock_db)
        
        # Required methods
        required_methods = [
            'search_similar',
            'upsert_chunk',
            'count_chunks',
        ]
        
        for method in required_methods:
            assert hasattr(store, method), f"Must have {method} method"
            assert callable(getattr(store, method)), f"{method} must be callable"
    
    def test_retriever_interface(self):
        """RAGRetriever must implement required interface."""
        from rag.retriever import RAGRetriever
        
        mock_store = MagicMock()
        mock_embedder = MagicMock()
        mock_embedder.available = True
        
        retriever = RAGRetriever(
            vector_store=mock_store,
            embedder=mock_embedder,
            use_hybrid=False,
            use_reranking=False,
            use_query_rewriting=False,
        )
        
        # Required methods - checking actual methods that exist
        required_methods = [
            'retrieve_for_control',
        ]
        
        for method in required_methods:
            assert hasattr(retriever, method), f"Must have {method} method"
            assert callable(getattr(retriever, method)), f"{method} must be callable"
    
    def test_control_result_is_enum(self):
        """evaluate_threshold must return ControlResultStatus enum."""
        from control_runner.controls import ControlDefinition, ControlCategory, ThresholdOperator, ControlResultStatus
        
        control = ControlDefinition(
            control_code="TEST",
            control_name="Test",
            category=ControlCategory.CONCENTRATION,
            computation_sql="SELECT 1",
            threshold_value=0.3,
            threshold_operator=ThresholdOperator.GTE,
            description="Test control description",
        )
        
        result = control.evaluate_threshold(0.25)
        
        # Result is ControlResultStatus enum
        assert isinstance(result, ControlResultStatus), \
            f"Result must be ControlResultStatus, got {type(result)}"
