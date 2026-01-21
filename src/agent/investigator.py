"""
Compliance Investigation Agent

An agentic LLM that autonomously investigates compliance issues by:
1. Reasoning step-by-step about the problem
2. Calling tools to gather data (query positions, lookup policies, etc.)
3. Chaining observations to discover root causes
4. Producing actionable findings with evidence

This is NOT summarization - the agent actively explores and discovers.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Callable
from decimal import Decimal

logger = logging.getLogger(__name__)


# =============================================================================
# TOOL DEFINITIONS
# =============================================================================

@dataclass
class Tool:
    """A tool the agent can call."""
    name: str
    description: str
    parameters: Dict[str, str]  # param_name -> description
    function: Callable
    
    def to_prompt(self) -> str:
        """Format tool for LLM prompt."""
        params = ", ".join(f"{k}: {v}" for k, v in self.parameters.items())
        return f"- {self.name}({params}): {self.description}"


@dataclass
class ToolCall:
    """A tool call made by the agent."""
    tool_name: str
    arguments: Dict[str, Any]
    result: Optional[str] = None


@dataclass 
class AgentStep:
    """One step in the agent's reasoning chain."""
    thought: str
    tool_call: Optional[ToolCall] = None
    observation: Optional[str] = None


@dataclass
class Investigation:
    """Complete investigation result."""
    issue: str
    steps: List[AgentStep]
    findings: str
    root_cause: str
    recommendations: List[str]
    evidence: List[str]
    duration_seconds: float = 0.0


# =============================================================================
# INVESTIGATION TOOLS
# =============================================================================

class InvestigationTools:
    """Tools available to the compliance investigation agent."""
    
    def __init__(self, snapshot, vector_store=None, embedder=None):
        """
        Initialize with data snapshot.
        
        Args:
            snapshot: DataSnapshot with positions and control_results
            vector_store: Optional VectorStore for policy lookup
            embedder: Optional embedder for semantic search
        """
        self.snapshot = snapshot
        self.vector_store = vector_store
        self.embedder = embedder
        self._build_indexes()
    
    def _build_indexes(self):
        """Build indexes for fast lookups."""
        # Index positions by sector
        self.positions_by_sector = {}
        for p in self.snapshot.positions:
            sector = getattr(p, 'sector', None) or getattr(p, 'asset_class', 'Unknown')
            if sector not in self.positions_by_sector:
                self.positions_by_sector[sector] = []
            self.positions_by_sector[sector].append(p)
        
        # Index positions by ticker
        self.positions_by_ticker = {p.ticker: p for p in self.snapshot.positions}
        
        # Index controls by type
        self.controls_by_type = {}
        for c in self.snapshot.control_results:
            ctype = c.control_type
            if ctype not in self.controls_by_type:
                self.controls_by_type[ctype] = []
            self.controls_by_type[ctype].append(c)
    
    def get_tools(self) -> List[Tool]:
        """Get all available tools."""
        return [
            Tool(
                name="query_positions_by_sector",
                description="Get all positions in a specific sector, sorted by market value",
                parameters={"sector": "Sector name (e.g., 'Technology', 'Healthcare')"},
                function=self.query_positions_by_sector,
            ),
            Tool(
                name="query_top_positions",
                description="Get top N positions by market value across entire portfolio",
                parameters={"n": "Number of positions to return (default 5)"},
                function=self.query_top_positions,
            ),
            Tool(
                name="get_position_details",
                description="Get detailed information about a specific position by ticker",
                parameters={"ticker": "Stock ticker symbol (e.g., 'AAPL')"},
                function=self.get_position_details,
            ),
            Tool(
                name="calculate_sector_concentration",
                description="Calculate concentration percentage for a specific sector",
                parameters={"sector": "Sector name to calculate concentration for"},
                function=self.calculate_sector_concentration,
            ),
            Tool(
                name="get_control_details",
                description="Get details about a specific control by ID or name",
                parameters={"control_id": "Control ID (e.g., 'CONC_SECTOR_001') or partial name"},
                function=self.get_control_details,
            ),
            Tool(
                name="lookup_policy",
                description="Search policy documents for relevant sections",
                parameters={"query": "Search query (e.g., 'sector concentration limit')"},
                function=self.lookup_policy,
            ),
            Tool(
                name="compare_to_threshold",
                description="Calculate how far a value is from its threshold and trend",
                parameters={"control_id": "Control ID to analyze"},
                function=self.compare_to_threshold,
            ),
            Tool(
                name="list_all_sectors",
                description="List all sectors in the portfolio with position counts",
                parameters={},
                function=self.list_all_sectors,
            ),
            Tool(
                name="calculate_what_if",
                description="Calculate impact if a position were sold",
                parameters={"ticker": "Ticker to simulate selling", "percent_to_sell": "Percentage to sell (0-100)"},
                function=self.calculate_what_if,
            ),
        ]
    
    def query_positions_by_sector(self, sector: str) -> str:
        """Get positions in a sector."""
        # Normalize sector name
        sector_lower = sector.lower()
        matching_sector = None
        for s in self.positions_by_sector.keys():
            if sector_lower in s.lower():
                matching_sector = s
                break
        
        if not matching_sector:
            return f"No positions found in sector '{sector}'. Available sectors: {list(self.positions_by_sector.keys())}"
        
        positions = self.positions_by_sector[matching_sector]
        positions_sorted = sorted(positions, key=lambda p: float(p.market_value), reverse=True)
        
        nav = float(self.snapshot.nav)
        result = f"Positions in {matching_sector} sector ({len(positions)} total):\n"
        for p in positions_sorted[:10]:  # Top 10
            mv = float(p.market_value)
            pct = (mv / nav) * 100
            result += f"  - {p.ticker}: ${mv:,.0f} ({pct:.2f}% of NAV)\n"
        
        total_mv = sum(float(p.market_value) for p in positions)
        total_pct = (total_mv / nav) * 100
        result += f"\nTotal {matching_sector}: ${total_mv:,.0f} ({total_pct:.2f}% of NAV)"
        return result
    
    def query_top_positions(self, n: int = 5) -> str:
        """Get top N positions."""
        n = int(n) if n else 5
        positions_sorted = sorted(
            self.snapshot.positions, 
            key=lambda p: float(p.market_value), 
            reverse=True
        )[:n]
        
        nav = float(self.snapshot.nav)
        result = f"Top {n} positions by market value:\n"
        for i, p in enumerate(positions_sorted, 1):
            mv = float(p.market_value)
            pct = (mv / nav) * 100
            sector = getattr(p, 'sector', 'N/A')
            result += f"  {i}. {p.ticker} ({sector}): ${mv:,.0f} ({pct:.2f}% of NAV)\n"
        return result
    
    def get_position_details(self, ticker: str = "") -> str:
        """Get details for a specific position."""
        if not ticker:
            # Return the largest position if no ticker specified
            largest = max(self.snapshot.positions, key=lambda p: float(p.market_value))
            ticker = largest.ticker
        ticker = ticker.upper()
        if ticker not in self.positions_by_ticker:
            return f"Position '{ticker}' not found. Available: {list(self.positions_by_ticker.keys())}"
        
        p = self.positions_by_ticker[ticker]
        nav = float(self.snapshot.nav)
        mv = float(p.market_value)
        pct = (mv / nav) * 100
        
        return f"""Position: {p.ticker}
  Security Name: {p.security_name}
  Quantity: {float(p.quantity):,.0f} shares
  Market Value: ${mv:,.0f}
  % of NAV: {pct:.2f}%
  Sector: {getattr(p, 'sector', 'N/A')}
  Issuer: {getattr(p, 'issuer', 'N/A')}
  Asset Class: {getattr(p, 'asset_class', 'N/A')}"""
    
    def calculate_sector_concentration(self, sector: str) -> str:
        """Calculate sector concentration."""
        sector_lower = sector.lower()
        matching_sector = None
        for s in self.positions_by_sector.keys():
            if sector_lower in s.lower():
                matching_sector = s
                break
        
        if not matching_sector:
            return f"Sector '{sector}' not found."
        
        positions = self.positions_by_sector[matching_sector]
        total_mv = sum(float(p.market_value) for p in positions)
        nav = float(self.snapshot.nav)
        concentration = (total_mv / nav) * 100
        
        # Find the limit from controls
        limit = 30.0  # default
        for c in self.snapshot.control_results:
            if 'sector' in c.control_type.lower() and matching_sector.lower() in c.control_name.lower():
                limit = float(c.threshold)
                break
        
        buffer = limit - concentration
        return f"""{matching_sector} Sector Concentration:
  Total Market Value: ${total_mv:,.0f}
  NAV: ${nav:,.0f}
  Concentration: {concentration:.2f}%
  Limit: {limit:.2f}%
  Buffer to Limit: {buffer:.2f}%
  Status: {'WARNING' if buffer < 5 else 'OK'}"""
    
    def get_control_details(self, control_id: str = "") -> str:
        """Get control details."""
        if not control_id:
            # Return first warning/failed control if no ID specified
            for c in self.snapshot.control_results:
                if c.status.lower() in ('warning', 'fail', 'failed'):
                    control_id = c.control_id
                    break
            if not control_id:
                return "No control_id specified. Please provide a control ID like 'CONC_SECTOR_001'."
        for c in self.snapshot.control_results:
            if control_id.upper() in c.control_id.upper() or control_id.lower() in c.control_name.lower():
                return f"""Control: {c.control_id}
  Name: {c.control_name}
  Type: {c.control_type}
  Calculated Value: {c.calculated_value}%
  Threshold: {c.threshold_operator} {c.threshold}%
  Status: {c.status.upper()}
  Breach Amount: {c.breach_amount}%"""
        
        return f"Control '{control_id}' not found."
    
    def lookup_policy(self, query: str) -> str:
        """Search policy documents."""
        if not self.vector_store or not self.embedder:
            return "Policy search not available (vector store not configured)."
        
        try:
            query_embedding = self.embedder.embed(query)
            chunks = self.vector_store.search_similar(query_embedding, limit=2)
            
            if not chunks:
                return "No relevant policy sections found."
            
            result = "Relevant policy sections:\n"
            for chunk in chunks:
                similarity = getattr(chunk, 'similarity', 0)
                result += f"\n[{chunk.document_name} | {chunk.section_title}] ({similarity:.0%} match)\n"
                # Truncate content
                content = chunk.content[:500] + "..." if len(chunk.content) > 500 else chunk.content
                result += content + "\n"
            return result
        except Exception as e:
            return f"Policy search failed: {e}"
    
    def compare_to_threshold(self, control_id: str) -> str:
        """Analyze control vs threshold."""
        for c in self.snapshot.control_results:
            if control_id.upper() in c.control_id.upper():
                value = float(c.calculated_value)
                threshold = float(c.threshold)
                
                if c.threshold_operator in ('lte', '<='):
                    distance = threshold - value
                    direction = "below" if distance > 0 else "above"
                else:  # gte
                    distance = value - threshold
                    direction = "above" if distance > 0 else "below"
                
                # Estimate risk
                if abs(distance) < 2:
                    risk = "HIGH - Very close to threshold"
                elif abs(distance) < 5:
                    risk = "MEDIUM - Approaching threshold"
                else:
                    risk = "LOW - Comfortable buffer"
                
                return f"""Threshold Analysis: {c.control_id}
  Current: {value:.2f}%
  Threshold: {c.threshold_operator} {threshold:.2f}%
  Distance: {abs(distance):.2f}% {direction} threshold
  Risk Level: {risk}
  Status: {c.status.upper()}"""
        
        return f"Control '{control_id}' not found."
    
    def list_all_sectors(self) -> str:
        """List all sectors."""
        nav = float(self.snapshot.nav)
        result = "Portfolio sectors:\n"
        
        sectors_data = []
        for sector, positions in self.positions_by_sector.items():
            total_mv = sum(float(p.market_value) for p in positions)
            pct = (total_mv / nav) * 100
            sectors_data.append((sector, len(positions), total_mv, pct))
        
        # Sort by concentration
        sectors_data.sort(key=lambda x: x[3], reverse=True)
        
        for sector, count, mv, pct in sectors_data:
            result += f"  - {sector}: {count} positions, ${mv:,.0f} ({pct:.2f}%)\n"
        
        return result
    
    def calculate_what_if(self, ticker: str, percent_to_sell: float = 100) -> str:
        """Simulate selling a position."""
        ticker = ticker.upper()
        percent = float(percent_to_sell)
        
        if ticker not in self.positions_by_ticker:
            return f"Position '{ticker}' not found."
        
        p = self.positions_by_ticker[ticker]
        mv = float(p.market_value)
        sell_amount = mv * (percent / 100)
        sector = getattr(p, 'sector', 'Unknown')
        
        # Calculate current sector concentration
        sector_mv = sum(float(pos.market_value) for pos in self.positions_by_sector.get(sector, []))
        nav = float(self.snapshot.nav)
        current_conc = (sector_mv / nav) * 100
        
        # Calculate new concentration after sale
        new_sector_mv = sector_mv - sell_amount
        new_conc = (new_sector_mv / nav) * 100
        
        return f"""What-If Analysis: Sell {percent:.0f}% of {ticker}
  Current Position: ${mv:,.0f}
  Sell Amount: ${sell_amount:,.0f}
  
  Impact on {sector} Sector:
    Before: {current_conc:.2f}%
    After: {new_conc:.2f}%
    Reduction: {current_conc - new_conc:.2f}%
  
  This would {'cure the warning' if new_conc < 28 else 'reduce but not cure the warning'}."""


# =============================================================================
# AGENT EXECUTOR
# =============================================================================

AGENT_SYSTEM_PROMPT = """You are a Compliance Investigation Agent for an SEC-registered hedge fund.

Your job is to INVESTIGATE compliance issues by:
1. Reasoning step-by-step about what might be causing an issue
2. Using tools to gather evidence
3. Discovering the root cause
4. Providing actionable recommendations

You have access to these tools:
{tools}

INVESTIGATION PROTOCOL:
1. Start with THOUGHT: Reason about what you need to investigate
2. Then ACTION: Call a tool to gather information
3. Then OBSERVATION: (I will provide the tool result)
4. Repeat until you have enough evidence
5. End with FINDINGS: Your conclusions and recommendations

FORMAT YOUR RESPONSE EXACTLY LIKE THIS:
THOUGHT: [Your reasoning about what to investigate next]
ACTION: tool_name(param1="value1", param2="value2")

When you have gathered enough evidence, respond with:
FINDINGS:
Root Cause: [What is causing the issue]
Evidence: [Key facts discovered]
Recommendations: [Specific actions to take]

IMPORTANT:
- Be thorough but efficient (aim for 3-5 tool calls)
- Always cite evidence from tool results
- Suggest SPECIFIC remediation actions with numbers
- If you need more data, call another tool

Begin your investigation."""


class ComplianceAgent:
    """
    Agentic LLM that investigates compliance issues.
    
    Uses a ReAct-style loop:
    1. Reason (THOUGHT)
    2. Act (ACTION - tool call)
    3. Observe (OBSERVATION - tool result)
    4. Repeat until FINDINGS
    """
    
    def __init__(self, llm_client, tools: InvestigationTools, max_steps: int = 8):
        """
        Initialize agent.
        
        Args:
            llm_client: LLM client with generate() method
            tools: InvestigationTools instance
            max_steps: Maximum reasoning steps
        """
        self.llm = llm_client
        self.tools = tools
        self.max_steps = max_steps
        self.tool_map = {t.name: t for t in tools.get_tools()}
    
    def _build_system_prompt(self) -> str:
        """Build system prompt with tool descriptions."""
        tools_desc = "\n".join(t.to_prompt() for t in self.tools.get_tools())
        return AGENT_SYSTEM_PROMPT.format(tools=tools_desc)
    
    def _parse_action(self, text: str) -> Optional[ToolCall]:
        """Parse ACTION from LLM response."""
        # Look for ACTION: tool_name(args)
        action_match = re.search(r'ACTION:\s*(\w+)\((.*?)\)', text, re.DOTALL)
        if not action_match:
            return None
        
        tool_name = action_match.group(1)
        args_str = action_match.group(2)
        
        # Parse arguments
        args = {}
        # Match param="value" or param=value
        for match in re.finditer(r'(\w+)\s*=\s*["\']?([^"\'",)]+)["\']?', args_str):
            args[match.group(1)] = match.group(2).strip()
        
        return ToolCall(tool_name=tool_name, arguments=args)
    
    def _execute_tool(self, tool_call: ToolCall) -> str:
        """Execute a tool call."""
        if tool_call.tool_name not in self.tool_map:
            return f"Error: Unknown tool '{tool_call.tool_name}'"
        
        tool = self.tool_map[tool_call.tool_name]
        try:
            result = tool.function(**tool_call.arguments)
            return result
        except Exception as e:
            return f"Error executing {tool_call.tool_name}: {e}"
    
    def investigate(self, issue: str) -> Investigation:
        """
        Investigate a compliance issue.
        
        Args:
            issue: Description of the issue to investigate
            
        Returns:
            Investigation with steps, findings, and recommendations
        """
        start_time = datetime.now(timezone.utc)
        steps = []
        conversation = []
        
        # Initial prompt
        system_prompt = self._build_system_prompt()
        user_prompt = f"INVESTIGATE THIS ISSUE:\n{issue}\n\nBegin your investigation with THOUGHT:"
        
        for step_num in range(self.max_steps):
            # Get LLM response
            full_prompt = user_prompt
            if conversation:
                full_prompt = "\n\n".join(conversation) + "\n\n" + user_prompt
            
            response = self.llm.generate(full_prompt, system_prompt)
            
            # Check for FINDINGS (end of investigation)
            if "FINDINGS:" in response:
                # Parse findings
                findings_match = re.search(r'FINDINGS:(.*)', response, re.DOTALL)
                findings_text = findings_match.group(1).strip() if findings_match else response
                
                # Extract root cause
                root_cause_match = re.search(r'Root Cause[:\s]+(.*?)(?:Evidence|Recommendation|$)', findings_text, re.DOTALL | re.IGNORECASE)
                root_cause = root_cause_match.group(1).strip() if root_cause_match else "See findings"
                
                # Extract recommendations
                recommendations = []
                rec_match = re.search(r'Recommendation[s]?[:\s]+(.*)', findings_text, re.DOTALL | re.IGNORECASE)
                if rec_match:
                    rec_text = rec_match.group(1)
                    recommendations = [r.strip() for r in re.split(r'\n[-•\d.]+\s*', rec_text) if r.strip()]
                
                duration = (datetime.now(timezone.utc) - start_time).total_seconds()
                
                return Investigation(
                    issue=issue,
                    steps=steps,
                    findings=findings_text,
                    root_cause=root_cause,
                    recommendations=recommendations or ["Review findings and take appropriate action"],
                    evidence=[s.observation for s in steps if s.observation],
                    duration_seconds=duration,
                )
            
            # Parse THOUGHT
            thought_match = re.search(r'THOUGHT[:\s]+(.*?)(?:ACTION|$)', response, re.DOTALL | re.IGNORECASE)
            thought = thought_match.group(1).strip() if thought_match else ""
            
            # Parse ACTION
            tool_call = self._parse_action(response)
            
            step = AgentStep(thought=thought, tool_call=tool_call)
            
            if tool_call:
                # Execute tool
                result = self._execute_tool(tool_call)
                tool_call.result = result
                step.observation = result
                
                # Add to conversation
                conversation.append(f"THOUGHT: {thought}")
                conversation.append(f"ACTION: {tool_call.tool_name}({tool_call.arguments})")
                conversation.append(f"OBSERVATION: {result}")
                
                user_prompt = "Continue your investigation. What's your next THOUGHT?"
            else:
                # No action parsed, might be final response
                conversation.append(response)
                user_prompt = "Please provide either an ACTION to investigate further, or FINDINGS if you have enough evidence."
            
            steps.append(step)
        
        # Max steps reached - generate summary from collected evidence
        duration = (datetime.now(timezone.utc) - start_time).total_seconds()
        evidence = [s.observation for s in steps if s.observation]
        
        # Try to infer root cause from evidence
        root_cause = self._infer_root_cause(issue, evidence)
        recommendations = self._infer_recommendations(issue, evidence)
        
        return Investigation(
            issue=issue,
            steps=steps,
            findings=f"Investigation completed after {len(steps)} steps. Evidence collected from {len(evidence)} tool calls.",
            root_cause=root_cause,
            recommendations=recommendations,
            evidence=evidence,
            duration_seconds=duration,
        )
    
    def _infer_root_cause(self, issue: str, evidence: list) -> str:
        """Infer root cause from collected evidence."""
        evidence_text = "\n".join(evidence)
        
        # Look for concentration patterns
        if "sector" in issue.lower() or "concentration" in issue.lower():
            # Find the largest contributing position
            positions = re.findall(r'(\w+):\s*\$[\d,]+\s*\((\d+\.?\d*)%', evidence_text)
            if positions:
                largest = max(positions, key=lambda x: float(x[1]))
                return f"High concentration driven by {largest[0]} at {largest[1]}% of NAV"
        
        if "liquidity" in issue.lower():
            return "Portfolio liquidity metrics approaching minimum thresholds"
        
        if "issuer" in issue.lower():
            # Find issuer concentration
            positions = re.findall(r'(\w+).*?(\d+\.?\d*)%\s*of\s*NAV', evidence_text)
            if positions:
                largest = max(positions, key=lambda x: float(x[1]))
                return f"Single issuer concentration in {largest[0]} at {largest[1]}% of NAV"
        
        return "Multiple factors contributing - see evidence for details"
    
    def _infer_recommendations(self, issue: str, evidence: list) -> list:
        """Infer recommendations from issue type and evidence."""
        recommendations = []
        
        if "sector" in issue.lower() or "concentration" in issue.lower():
            recommendations.append("Consider reducing largest sector positions")
            recommendations.append("Review sector allocation vs investment policy limits")
            recommendations.append("Add diversification across other sectors")
        elif "liquidity" in issue.lower():
            recommendations.append("Review T+7 liquidity requirements")
            recommendations.append("Consider increasing allocation to liquid instruments")
        elif "issuer" in issue.lower():
            recommendations.append("Consider reducing single issuer exposure")
            recommendations.append("Review issuer concentration limits in policy")
        else:
            recommendations.append("Manual review recommended")
        
        return recommendations
    
    def investigate_control(self, control_result) -> Investigation:
        """Investigate a specific control result."""
        issue = f"""Control {control_result.control_id} ({control_result.control_name}) is in {control_result.status.upper()} status.
Current value: {control_result.calculated_value}%
Threshold: {control_result.threshold_operator} {control_result.threshold}%
Control type: {control_result.control_type}

Investigate:
1. What is causing this control to approach/breach its limit?
2. Which specific positions are contributing most?
3. What specific actions could cure or improve this?"""
        
        return self.investigate(issue)
