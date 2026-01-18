"""
Compliance RAG Pipeline - AI-Powered Narrative Generation

This is the main value-add: taking client's existing compliance data
and generating well-cited, auditable narratives using RAG.

Flow:
1. Client's system provides control results (already calculated)
2. We retrieve relevant policies
3. LLM generates narrative with citations
4. Output includes full audit trail

What we DO:
- Policy retrieval (RAG)
- Narrative generation with citations
- Audit trail for SEC examination

What we DON'T DO:
- Re-calculate compliance metrics (that's their system)
- Re-validate position data (already audited)
- Store their data long-term (read and process)
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, date
from typing import Optional, Dict, Any, List
from decimal import Decimal
import uuid

from .client_adapter import DataSnapshot, ControlResult, Position

logger = logging.getLogger(__name__)


@dataclass
class PolicyContext:
    """Relevant policy excerpts for a control."""
    policy_id: str
    section: str
    content: str
    relevance_score: float = 1.0
    
    def to_citation(self) -> str:
        return f"[Policy: {self.policy_id} | Section: {self.section}]"


@dataclass 
class GeneratedNarrative:
    """AI-generated narrative with full audit trail."""
    narrative_id: str
    control_id: str
    
    # Generated content
    content: str
    content_hash: str
    citations: List[str]
    
    # Audit trail
    model_used: str
    prompt_hash: str
    context_hash: str
    generated_at: datetime
    
    # Validation
    has_citations: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "narrative_id": self.narrative_id,
            "control_id": self.control_id,
            "content": self.content,
            "content_hash": self.content_hash,
            "citations": self.citations,
            "model_used": self.model_used,
            "prompt_hash": self.prompt_hash,
            "context_hash": self.context_hash,
            "generated_at": self.generated_at.isoformat(),
            "has_citations": self.has_citations,
        }


@dataclass
class ComplianceReport:
    """Complete compliance report for a date."""
    report_id: str
    as_of_date: date
    
    # Source data (from client system)
    snapshot: DataSnapshot
    
    # Generated narratives
    narratives: List[GeneratedNarrative] = field(default_factory=list)
    
    # Summary
    controls_passed: int = 0
    controls_warning: int = 0
    controls_failed: int = 0
    
    # Audit
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def get_executive_summary(self) -> str:
        """Generate executive summary text."""
        total = self.controls_passed + self.controls_warning + self.controls_failed
        
        if self.controls_failed > 0:
            status = "⚠️ BREACHES DETECTED"
        elif self.controls_warning > 0:
            status = "⚡ WARNINGS"
        else:
            status = "✅ ALL CONTROLS PASSED"
        
        return f"""
Daily Compliance Report - {self.as_of_date}
============================================

Status: {status}

Control Summary:
  Passed:   {self.controls_passed}
  Warning:  {self.controls_warning}
  Failed:   {self.controls_failed}
  Total:    {total}

NAV: ${self.snapshot.nav:,.0f}
Positions: {len(self.snapshot.positions)}
"""


class ComplianceRAGPipeline:
    """
    Main RAG pipeline for compliance narrative generation.
    
    This is the core AI value-add on top of client's existing systems.
    """
    
    def __init__(
        self,
        policy_store=None,
        llm_client=None,
        model_id: str = "gpt-4o",
    ):
        self.policy_store = policy_store
        self.llm_client = llm_client
        self.model_id = model_id
    
    def generate_report(self, snapshot: DataSnapshot) -> ComplianceReport:
        """
        Generate complete compliance report with AI narratives.
        
        Args:
            snapshot: Data snapshot from client's system
            
        Returns:
            ComplianceReport with generated narratives
        """
        report = ComplianceReport(
            report_id=str(uuid.uuid4()),
            as_of_date=snapshot.as_of_date,
            snapshot=snapshot,
        )
        
        # Process each control result
        for control in snapshot.control_results:
            # Track status counts
            if control.status == "pass":
                report.controls_passed += 1
            elif control.status == "warning":
                report.controls_warning += 1
            else:
                report.controls_failed += 1
            
            # Generate narrative for warnings and failures
            if control.status in ("warning", "fail"):
                narrative = self._generate_narrative(control, snapshot)
                report.narratives.append(narrative)
        
        logger.info(
            f"Generated report {report.report_id}: "
            f"{report.controls_passed} passed, "
            f"{report.controls_warning} warnings, "
            f"{report.controls_failed} failed"
        )
        
        return report
    
    def _generate_narrative(
        self, 
        control: ControlResult,
        snapshot: DataSnapshot,
    ) -> GeneratedNarrative:
        """Generate narrative for a single control result."""
        
        # Step 1: Retrieve relevant policies
        policies = self._retrieve_policies(control)
        
        # Step 2: Build prompt
        prompt = self._build_prompt(control, policies, snapshot)
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:16]
        
        # Step 3: Generate narrative
        if self.llm_client:
            content = self._call_llm(prompt)
        else:
            # Deterministic fallback when no LLM configured
            content = self._generate_deterministic(control, policies)
        
        # Step 4: Extract citations
        citations = self._extract_citations(content)
        
        # Step 5: Build context hash for audit
        context_hash = hashlib.sha256(
            json.dumps({
                "control_id": control.control_id,
                "policies": [p.policy_id for p in policies],
            }, sort_keys=True).encode()
        ).hexdigest()[:16]
        
        return GeneratedNarrative(
            narrative_id=str(uuid.uuid4()),
            control_id=control.control_id,
            content=content,
            content_hash=hashlib.sha256(content.encode()).hexdigest()[:16],
            citations=citations,
            model_used=self.model_id if self.llm_client else "deterministic",
            prompt_hash=prompt_hash,
            context_hash=context_hash,
            generated_at=datetime.now(timezone.utc),
            has_citations=len(citations) > 0,
        )
    
    def _retrieve_policies(self, control: ControlResult) -> List[PolicyContext]:
        """Retrieve relevant policy sections for a control."""
        
        # If we have a policy store, search it
        if self.policy_store:
            # Search by control type and name
            chunks = self.policy_store.search_chunks(
                query=f"{control.control_type} {control.control_name}",
                limit=3,
            )
            return [
                PolicyContext(
                    policy_id=chunk.policy_id,
                    section=chunk.section_path,
                    content=chunk.content,
                )
                for chunk in chunks
            ]
        
        # Fallback: return mock policy context
        return self._get_mock_policies(control)
    
    def _get_mock_policies(self, control: ControlResult) -> List[PolicyContext]:
        """Return mock policy context when no store configured."""
        
        policy_map = {
            "concentration": PolicyContext(
                policy_id="investment_guidelines",
                section="2. Concentration Limits",
                content="""
### 2.1 Sector Concentration
- **Maximum Sector Exposure**: 30% of NAV
- **Sectors**: GICS Level 1 classification
- **Exceptions**: Technology sector permitted up to 40% with CIO approval

### 2.3 Issuer Concentration
- **Maximum Single Issuer**: 10% of NAV
- **Definition**: All securities of a single corporate family
""",
            ),
            "liquidity": PolicyContext(
                policy_id="investment_guidelines",
                section="3. Liquidity Requirements",
                content="""
### 3.1 Liquidity Buckets
| Bucket | Timeframe | Minimum |
|--------|-----------|---------|
| T+1 | Same/next day | 10% NAV |
| T+7 | Within 1 week | 40% NAV |
| T+30 | Within 1 month | 60% NAV |

### 3.2 Liquidity Calculation
- Based on 20-day average daily volume (ADV)
- Position liquidation assumes 25% of ADV participation
""",
            ),
            "exposure": PolicyContext(
                policy_id="investment_guidelines",
                section="1. Position Limits",
                content="""
### 1.1 Gross Exposure
- **Maximum Gross Exposure**: 200% of NAV
- **Warning Threshold**: 180% of NAV
- **Calculation**: (Long Market Value + |Short Market Value|) / NAV

### 1.2 Net Exposure
- **Maximum Net Long**: 100% of NAV
- **Maximum Net Short**: -50% of NAV
""",
            ),
        }
        
        policy = policy_map.get(control.control_type)
        return [policy] if policy else []
    
    def _build_prompt(
        self, 
        control: ControlResult, 
        policies: List[PolicyContext],
        snapshot: DataSnapshot,
    ) -> str:
        """Build LLM prompt with evidence and policies."""
        
        policy_text = "\n\n".join([
            f"**{p.to_citation()}**\n{p.content}"
            for p in policies
        ])
        
        return f"""You are a compliance documentation assistant for an SEC-registered hedge fund.
Generate a clear, professional narrative explaining the following control result.

RULES:
1. Use ONLY the information provided below
2. Include citations in [brackets] for every factual statement
3. Be concise but complete
4. Do not invent any numbers or facts

## CONTROL RESULT

Control: {control.control_name} ({control.control_id})
Type: {control.control_type}
Status: {control.status.upper()}

Calculated Value: {control.calculated_value}%
Threshold: {control.threshold}% ({control.threshold_operator})
{f"Breach Amount: {control.breach_amount}%" if control.breach_amount else ""}

As of Date: {control.as_of_date}
Fund NAV: ${snapshot.nav:,.0f}

{f"Details: {json.dumps(control.details)}" if control.details else ""}

## RELEVANT POLICY

{policy_text}

## TASK

Write a 2-3 paragraph narrative that:
1. States the control result clearly
2. Explains the policy requirement with citation
3. Describes the current situation
4. If warning/breach, suggests remediation

Use citations like [Policy: investment_guidelines | Section: X.X]
"""
    
    def _call_llm(self, prompt: str) -> str:
        """Call the LLM API."""
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": "You are a compliance documentation assistant."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=500,
                temperature=0.3,  # Low temperature for consistency
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return f"[Error generating narrative: {e}]"
    
    def _generate_deterministic(
        self, 
        control: ControlResult,
        policies: List[PolicyContext],
    ) -> str:
        """Generate deterministic narrative without LLM."""
        
        policy_citation = policies[0].to_citation() if policies else "[No policy found]"
        
        if control.status == "warning":
            return f"""The {control.control_name} control is currently at {control.calculated_value}%, which is approaching the policy threshold of {control.threshold}%. {policy_citation}

Per the investment guidelines, the fund must maintain {control.control_type} within prescribed limits. The current level represents a warning condition that should be monitored closely.

Recommended action: Review current positions and consider rebalancing to reduce {control.control_type} exposure before reaching the threshold limit."""
        
        elif control.status == "fail":
            return f"""BREACH: The {control.control_name} control has exceeded the policy threshold. Current value: {control.calculated_value}% vs limit: {control.threshold}%. Breach amount: {control.breach_amount}%. {policy_citation}

This represents a violation of the investment guidelines which require {control.control_type} to be maintained within limits. Immediate attention is required.

Required action: Remediation plan must be documented and approved. Positions should be adjusted to bring {control.control_type} back within policy limits."""
        
        else:
            return f"""The {control.control_name} control is passing at {control.calculated_value}%, within the threshold of {control.threshold}%. {policy_citation}"""
    
    def _extract_citations(self, content: str) -> List[str]:
        """Extract citation strings from generated content."""
        import re
        pattern = r'\[(?:Policy|Control|Source)[^\]]+\]'
        return re.findall(pattern, content)
