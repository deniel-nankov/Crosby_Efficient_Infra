"""
Compliance Investigation Agent Module

Provides an agentic LLM that autonomously investigates compliance issues.
"""

from .investigator import (
    ComplianceAgent,
    InvestigationTools,
    Investigation,
    AgentStep,
    Tool,
    ToolCall,
)

__all__ = [
    "ComplianceAgent",
    "InvestigationTools", 
    "Investigation",
    "AgentStep",
    "Tool",
    "ToolCall",
]
