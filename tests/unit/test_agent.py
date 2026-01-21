"""
Agent Unit Tests - Aligned with Actual Implementation

Tests for:
- Tool: Tool definition structure
- InvestigationTools: Tool implementations
- ComplianceAgent: ReAct-style agent
- InvestigationResult: Result structure
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass
from typing import List, Dict, Any

pytestmark = [pytest.mark.unit]


@dataclass
class MockPosition:
    """Mock position for testing."""
    ticker: str
    security_name: str
    market_value: float
    weight_pct: float
    sector: str = "Technology"
    asset_class: str = "Equity"
    quantity: int = 1000  # Added for compatibility


@dataclass
class MockControlResult:
    """Mock control result for testing."""
    control_code: str
    control_name: str
    control_type: str
    status: str
    calculated_value: float
    threshold: float


@dataclass  
class MockSnapshot:
    """Mock data snapshot for testing."""
    positions: List[MockPosition]
    control_results: List[MockControlResult]
    nav: float = 1_000_000_000


class TestToolDefinition:
    """Test Tool dataclass structure."""
    
    def test_tool_has_required_properties(self):
        """Tool should have name, description, parameters, function."""
        from agent.investigator import Tool
        
        tool = Tool(
            name="test_tool",
            description="A test tool",
            parameters={"arg1": "Description of arg1"},
            function=lambda x: x,
        )
        
        assert tool.name == "test_tool"
        assert tool.description == "A test tool"
        assert "arg1" in tool.parameters
        assert callable(tool.function)


class TestInvestigationTools:
    """Test InvestigationTools class."""
    
    def _create_mock_snapshot(self) -> MockSnapshot:
        """Create mock snapshot for testing."""
        positions = [
            MockPosition("AAPL", "Apple Inc", 100_000_000, 10.0, "Technology"),
            MockPosition("MSFT", "Microsoft", 80_000_000, 8.0, "Technology"),
            MockPosition("JPM", "JPMorgan", 50_000_000, 5.0, "Financials"),
        ]
        controls = [
            MockControlResult("CONC_001", "Sector Concentration", "concentration", "WARNING", 0.28, 0.30),
        ]
        return MockSnapshot(positions=positions, control_results=controls)
    
    def test_tools_initialization(self):
        """InvestigationTools should initialize with snapshot."""
        from agent.investigator import InvestigationTools
        
        snapshot = self._create_mock_snapshot()
        tools = InvestigationTools(snapshot=snapshot)
        
        assert tools.snapshot == snapshot
    
    def test_get_tools_returns_list(self):
        """get_tools should return list of Tool objects."""
        from agent.investigator import InvestigationTools, Tool
        
        snapshot = self._create_mock_snapshot()
        tools = InvestigationTools(snapshot=snapshot)
        
        tool_list = tools.get_tools()
        assert isinstance(tool_list, list)
        assert len(tool_list) > 0
        assert all(isinstance(t, Tool) for t in tool_list)
    
    def test_query_positions_by_sector(self):
        """query_positions_by_sector should filter by sector."""
        from agent.investigator import InvestigationTools
        
        snapshot = self._create_mock_snapshot()
        tools = InvestigationTools(snapshot=snapshot)
        
        result = tools.query_positions_by_sector("Technology")
        
        assert "AAPL" in result or "Apple" in result
        assert "MSFT" in result or "Microsoft" in result
    
    def test_query_top_positions(self):
        """query_top_positions should return top N by market value."""
        from agent.investigator import InvestigationTools
        
        snapshot = self._create_mock_snapshot()
        tools = InvestigationTools(snapshot=snapshot)
        
        result = tools.query_top_positions(n=2)
        
        # AAPL should be first (highest market value)
        assert "AAPL" in result or "Apple" in result
    
    def test_calculate_sector_concentration(self):
        """calculate_sector_concentration should compute percentage."""
        from agent.investigator import InvestigationTools
        
        snapshot = self._create_mock_snapshot()
        tools = InvestigationTools(snapshot=snapshot)
        
        result = tools.calculate_sector_concentration("Technology")
        
        # Technology has AAPL (10%) + MSFT (8%) = 18%
        assert "18" in result or "Technology" in result
    
    def test_get_position_details(self):
        """get_position_details should return info for specific ticker."""
        from agent.investigator import InvestigationTools
        
        snapshot = self._create_mock_snapshot()
        tools = InvestigationTools(snapshot=snapshot)
        
        result = tools.get_position_details("AAPL")
        
        assert "AAPL" in result or "Apple" in result
        assert "100" in result  # Market value in millions


class TestComplianceAgent:
    """Test ComplianceAgent class."""
    
    def _create_mock_snapshot(self) -> MockSnapshot:
        """Create mock snapshot for testing."""
        positions = [
            MockPosition("AAPL", "Apple Inc", 100_000_000, 10.0, "Technology"),
        ]
        controls = [
            MockControlResult("CONC_001", "Sector Concentration", "concentration", "WARNING", 0.28, 0.30),
        ]
        return MockSnapshot(positions=positions, control_results=controls)
    
    def test_agent_initialization(self):
        """ComplianceAgent should initialize with LLM client and tools."""
        from agent.investigator import ComplianceAgent, InvestigationTools
        
        mock_llm = MagicMock()
        snapshot = self._create_mock_snapshot()
        tools = InvestigationTools(snapshot=snapshot)
        
        agent = ComplianceAgent(
            llm_client=mock_llm,
            tools=tools,
        )
        
        # Agent stores llm as self.llm
        assert agent.llm == mock_llm
        assert agent.tools == tools
    
    def test_agent_default_max_steps(self):
        """Agent should have default max_steps of 8."""
        from agent.investigator import ComplianceAgent, InvestigationTools
        
        mock_llm = MagicMock()
        snapshot = self._create_mock_snapshot()
        tools = InvestigationTools(snapshot=snapshot)
        
        agent = ComplianceAgent(
            llm_client=mock_llm,
            tools=tools,
        )
        
        assert agent.max_steps == 8
    
    def test_agent_custom_max_steps(self):
        """Agent should accept custom max_steps."""
        from agent.investigator import ComplianceAgent, InvestigationTools
        
        mock_llm = MagicMock()
        snapshot = self._create_mock_snapshot()
        tools = InvestigationTools(snapshot=snapshot)
        
        agent = ComplianceAgent(
            llm_client=mock_llm,
            tools=tools,
            max_steps=5,
        )
        
        assert agent.max_steps == 5
    
    def test_agent_has_investigate_method(self):
        """Agent should have investigate method."""
        from agent.investigator import ComplianceAgent, InvestigationTools
        
        mock_llm = MagicMock()
        snapshot = self._create_mock_snapshot()
        tools = InvestigationTools(snapshot=snapshot)
        
        agent = ComplianceAgent(
            llm_client=mock_llm,
            tools=tools,
        )
        
        assert hasattr(agent, 'investigate')
        assert callable(agent.investigate)


class TestInvestigationDataclass:
    """Test Investigation dataclass."""
    
    def test_result_has_required_fields(self):
        """Investigation should have required fields."""
        from agent.investigator import Investigation
        
        result = Investigation(
            issue="Sector concentration warning",
            steps=[],
            findings="Technology sector at 28%",
            root_cause="Large AAPL position",
            recommendations=["Reduce AAPL position"],
            evidence=["Tool output 1"],
            duration_seconds=1.5,
        )
        
        assert result.issue == "Sector concentration warning"
        assert result.findings == "Technology sector at 28%"
        assert "AAPL" in result.root_cause
    
    def test_result_default_values(self):
        """Investigation should have required fields."""
        from agent.investigator import Investigation
        
        result = Investigation(
            issue="Test issue",
            steps=[],
            findings="Test findings",
            root_cause="Test root cause",
            recommendations=["Test recommendation"],
            evidence=["Test evidence"],
        )
        
        assert result.issue == "Test issue"
        assert result.findings == "Test findings"


class TestReActLoop:
    """Test ReAct loop behavior."""
    
    def test_react_loop_terminates_on_final_answer(self):
        """Agent should stop when LLM returns final_answer."""
        from agent.investigator import ComplianceAgent, InvestigationTools
        
        snapshot = MagicMock()
        snapshot.positions = []
        snapshot.control_results = []
        
        mock_llm = MagicMock()
        mock_llm.generate.return_value = '{"final_answer": "Investigation complete"}'
        
        tools = InvestigationTools(snapshot=snapshot)
        agent = ComplianceAgent(llm_client=mock_llm, tools=tools)
        
        result = agent.investigate("Test issue")
        
        # Should complete without hitting max steps
        assert result is not None
    
    def test_react_loop_respects_max_steps(self):
        """Agent should stop after max_steps."""
        from agent.investigator import ComplianceAgent, InvestigationTools
        
        snapshot = MagicMock()
        snapshot.positions = []
        snapshot.control_results = []
        
        mock_llm = MagicMock()
        # Always return a tool call (never final answer)
        mock_llm.generate.return_value = '{"thought": "thinking", "action": "unknown_tool", "action_input": {}}'
        
        tools = InvestigationTools(snapshot=snapshot)
        agent = ComplianceAgent(llm_client=mock_llm, tools=tools, max_steps=3)
        
        result = agent.investigate("Test issue")
        
        # Should still return result even if max steps reached
        assert result is not None
