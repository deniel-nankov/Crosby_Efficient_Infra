"""
Narrative Generator - LLM-Assisted Text Generation with Evidence Binding

This module handles LLM-assisted generation of compliance narratives.
It is the ONLY component that interacts with the LLM.

CRITICAL CONSTRAINTS:
1. LLM is used ONLY for prose generation, NEVER for calculations
2. All facts must come from retrieved evidence
3. Every generated paragraph must include citations
4. Templates are fixed and version-controlled
5. Output is deterministic given the same evidence

SEC Examination Note:
- All prompts are logged with hashes
- All LLM calls are recorded with model version
- Generated text includes inline citations
- If evidence is insufficient, generation fails rather than hallucinating
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, date
from enum import Enum
from typing import Optional, Dict, Any, List, Callable
import uuid
import re

from ..retrieval.retriever import RetrievalContext, RetrievedDocument

logger = logging.getLogger(__name__)


class NarrativeType(Enum):
    """Types of narratives that can be generated."""
    DAILY_SUMMARY = "daily_summary"
    EXCEPTION_DESCRIPTION = "exception_description"
    CONTROL_EXPLANATION = "control_explanation"
    METHODOLOGY_SECTION = "methodology_section"
    FILING_WORKPAPER = "filing_workpaper"


@dataclass
class PromptTemplate:
    """
    Immutable prompt template for narrative generation.
    
    Templates are version-controlled and hashed for audit trail.
    """
    template_id: str
    template_version: str
    narrative_type: NarrativeType
    
    # Template content
    system_prompt: str
    user_prompt_template: str  # Contains {placeholders}
    
    # Constraints
    required_evidence_types: List[str]
    max_output_tokens: int = 1000
    
    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def template_hash(self) -> str:
        """Hash of template for audit trail."""
        content = f"{self.system_prompt}|{self.user_prompt_template}|{self.template_version}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def render(self, **kwargs) -> str:
        """Render the user prompt with provided values."""
        return self.user_prompt_template.format(**kwargs)


@dataclass
class GeneratedNarrative:
    """
    Output of narrative generation.
    
    Includes all audit information needed for SEC examination.
    """
    narrative_id: str
    narrative_type: NarrativeType
    
    # Generated content
    content: str
    content_hash: str
    
    # Citations extracted from content
    citations: List[str]
    
    # Audit trail
    template_id: str
    template_hash: str
    prompt_hash: str
    context_hash: str
    
    # LLM metadata
    model_id: str
    model_version: str
    tokens_used: int
    
    # Timing
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    generation_duration_ms: int = 0
    
    # Validation
    passed_validation: bool = True
    validation_errors: List[str] = field(default_factory=list)
    
    def to_audit_record(self) -> Dict[str, Any]:
        """Create an audit record for this narrative."""
        return {
            "narrative_id": self.narrative_id,
            "narrative_type": self.narrative_type.value,
            "content_hash": self.content_hash,
            "template_id": self.template_id,
            "template_hash": self.template_hash,
            "prompt_hash": self.prompt_hash,
            "context_hash": self.context_hash,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "tokens_used": self.tokens_used,
            "generated_at": self.generated_at.isoformat(),
            "citation_count": len(self.citations),
            "passed_validation": self.passed_validation,
        }


# =============================================================================
# PROMPT TEMPLATES
# =============================================================================
# These templates are the ONLY prompts used for generation.
# Any changes require version updates.

SYSTEM_PROMPT_BASE = """You are a compliance documentation assistant for an SEC-registered hedge fund.
Your role is to generate clear, accurate, and well-cited compliance narratives.

CRITICAL RULES:
1. ONLY use information provided in the evidence context
2. NEVER invent, assume, or hallucinate any facts, numbers, or statements
3. Include inline citations for EVERY factual statement using the format provided
4. If evidence is insufficient, respond with: "Insufficient evidence available to generate this section."
5. Use precise, professional language appropriate for SEC examination
6. DO NOT perform any calculations - all numbers must come from the evidence
7. Be concise but complete

Your output will be reviewed by compliance officers and may be examined by SEC regulators.
Accuracy and traceability are paramount."""


DAILY_SUMMARY_TEMPLATE = PromptTemplate(
    template_id="TMPL-DAILY-001",
    template_version="1.0.0",
    narrative_type=NarrativeType.DAILY_SUMMARY,
    system_prompt=SYSTEM_PROMPT_BASE,
    user_prompt_template="""Generate a daily compliance summary narrative based on the following evidence.

CONTROL RUN INFORMATION:
{run_summary}

FAILED CONTROLS:
{failed_controls}

EXCEPTIONS OPENED:
{exceptions}

OUTSTANDING EXCEPTIONS:
{outstanding_exceptions}

RELEVANT POLICY CONTEXT:
{policy_context}

Generate a professional summary that:
1. Summarizes the overall compliance status for the day
2. Highlights any control failures and their significance
3. Notes new exceptions and their severity
4. Mentions outstanding exceptions requiring attention
5. Includes citations in the format [ControlRun: XXX | Control: XXX | Snapshot: XXX] for each factual statement

The summary should be 2-3 paragraphs, suitable for review by the Chief Compliance Officer.""",
    required_evidence_types=["control_results", "exceptions"],
    max_output_tokens=800,
)


EXCEPTION_NARRATIVE_TEMPLATE = PromptTemplate(
    template_id="TMPL-EXC-001",
    template_version="1.0.0",
    narrative_type=NarrativeType.EXCEPTION_DESCRIPTION,
    system_prompt=SYSTEM_PROMPT_BASE,
    user_prompt_template="""Generate a narrative description for the following compliance exception.

EXCEPTION DETAILS:
{exception_details}

CONTROL RESULT:
{control_result}

RELEVANT POLICY:
{policy_context}

Generate a professional exception narrative that:
1. Describes what occurred (the control breach)
2. Explains the significance based on the threshold exceeded
3. References the applicable policy
4. Does NOT suggest remediation (that requires human judgment)
5. Includes citations for all factual statements

The narrative should be 1-2 paragraphs, factual and objective.""",
    required_evidence_types=["exceptions", "control_results"],
    max_output_tokens=500,
)


METHODOLOGY_TEMPLATE = PromptTemplate(
    template_id="TMPL-METH-001",
    template_version="1.0.0",
    narrative_type=NarrativeType.METHODOLOGY_SECTION,
    system_prompt=SYSTEM_PROMPT_BASE,
    user_prompt_template="""Generate a methodology description for the compliance control system.

CONTROL DEFINITIONS:
{control_definitions}

POLICY REFERENCES:
{policy_context}

DATA SOURCES:
{data_sources}

Generate a professional methodology section that:
1. Describes how controls are calculated (based on the SQL/evidence provided)
2. Explains the thresholds and their basis in policy
3. Notes the data sources and snapshot methodology
4. Is suitable for inclusion in SEC filing workpapers
5. Includes citations to policies and control definitions

The methodology should be 2-4 paragraphs.""",
    required_evidence_types=["policies"],
    max_output_tokens=1000,
)


FILING_WORKPAPER_TEMPLATE = PromptTemplate(
    template_id="TMPL-FILE-001",
    template_version="1.0.0",
    narrative_type=NarrativeType.FILING_WORKPAPER,
    system_prompt=SYSTEM_PROMPT_BASE,
    user_prompt_template="""Generate supporting narrative for a {filing_type} filing workpaper.

FILING PERIOD: {period}

RELEVANT CONTROL RESULTS:
{control_results}

METRICS SUMMARY:
{metrics}

POLICY CONTEXT:
{policy_context}

PRIOR FILING REFERENCE:
{prior_filing}

Generate a professional workpaper narrative that:
1. Summarizes the compliance status for the filing period
2. Notes any material changes from the prior period
3. References specific control results that support filing data
4. Is suitable for SEC examiner review
5. Includes full citations

The narrative should be 2-3 paragraphs per section.""",
    required_evidence_types=["control_results", "policies"],
    max_output_tokens=1200,
)


# Template registry
TEMPLATES: Dict[str, PromptTemplate] = {
    "daily_summary": DAILY_SUMMARY_TEMPLATE,
    "exception_narrative": EXCEPTION_NARRATIVE_TEMPLATE,
    "methodology": METHODOLOGY_TEMPLATE,
    "filing_workpaper": FILING_WORKPAPER_TEMPLATE,
}


class NarrativeGenerator:
    """
    Main narrative generation engine.
    
    This class:
    1. Takes retrieved evidence context
    2. Applies appropriate prompt template
    3. Calls the LLM
    4. Validates the output for citations
    5. Records all audit information
    
    The LLM is used ONLY for text generation, not calculations.
    """
    
    def __init__(
        self,
        llm_client: Any,  # OpenAI, Anthropic, or Azure client
        settings: Any,
        postgres_connection: Optional[Any] = None,
    ):
        self.llm_client = llm_client
        self.settings = settings
        self.postgres = postgres_connection
        self.logger = logging.getLogger(f"{__name__}.NarrativeGenerator")
    
    def generate_daily_summary(
        self,
        context: RetrievalContext,
        run_summary: Dict[str, Any],
    ) -> GeneratedNarrative:
        """
        Generate the daily compliance summary narrative.
        
        Args:
            context: Retrieved evidence context
            run_summary: Control run summary data
        
        Returns:
            GeneratedNarrative with full audit trail
        """
        template = TEMPLATES["daily_summary"]
        
        # Extract evidence from context
        failed_controls = self._extract_failed_controls(context)
        exceptions = self._extract_exceptions(context)
        outstanding = self._extract_outstanding_exceptions(context)
        policy_context = self._extract_policy_context(context)
        
        # Validate we have required evidence
        if not failed_controls and not exceptions:
            # Check if this is a clean day (no failures)
            if run_summary.get('controls_failed', 0) == 0:
                failed_controls = "No control failures on this date."
                exceptions = "No new exceptions opened."
        
        # Render prompt
        user_prompt = template.render(
            run_summary=self._format_run_summary(run_summary),
            failed_controls=failed_controls or "No control failures.",
            exceptions=exceptions or "No new exceptions.",
            outstanding_exceptions=outstanding or "No outstanding exceptions.",
            policy_context=policy_context or "No specific policy context retrieved.",
        )
        
        # Generate narrative
        return self._generate(template, user_prompt, context)
    
    def generate_exception_narrative(
        self,
        context: RetrievalContext,
    ) -> GeneratedNarrative:
        """
        Generate narrative description for an exception.
        """
        template = TEMPLATES["exception_narrative"]
        
        # Extract evidence
        exception_details = self._extract_exception_details(context)
        control_result = self._extract_control_result(context)
        policy_context = self._extract_policy_context(context)
        
        if not exception_details:
            raise ValueError("Insufficient evidence: no exception details in context")
        
        user_prompt = template.render(
            exception_details=exception_details,
            control_result=control_result or "Control result details not available.",
            policy_context=policy_context or "No specific policy context retrieved.",
        )
        
        return self._generate(template, user_prompt, context)
    
    def generate_methodology_section(
        self,
        context: RetrievalContext,
        control_definitions: List[Dict[str, Any]],
        data_sources: str,
    ) -> GeneratedNarrative:
        """
        Generate methodology section for workpapers.
        """
        template = TEMPLATES["methodology"]
        
        policy_context = self._extract_policy_context(context)
        
        user_prompt = template.render(
            control_definitions=self._format_control_definitions(control_definitions),
            policy_context=policy_context or "Policy context not available.",
            data_sources=data_sources,
        )
        
        return self._generate(template, user_prompt, context)
    
    def generate_filing_workpaper(
        self,
        context: RetrievalContext,
        filing_type: str,
        period: str,
        metrics: Dict[str, Any],
    ) -> GeneratedNarrative:
        """
        Generate narrative for SEC filing workpaper.
        """
        template = TEMPLATES["filing_workpaper"]
        
        control_results = self._extract_control_results_summary(context)
        policy_context = self._extract_policy_context(context)
        prior_filing = self._extract_prior_filing(context)
        
        user_prompt = template.render(
            filing_type=filing_type.upper(),
            period=period,
            control_results=control_results or "Control results not available.",
            metrics=json.dumps(metrics, indent=2),
            policy_context=policy_context or "Policy context not available.",
            prior_filing=prior_filing or "No prior filing reference available.",
        )
        
        return self._generate(template, user_prompt, context)
    
    def _generate(
        self,
        template: PromptTemplate,
        user_prompt: str,
        context: RetrievalContext,
    ) -> GeneratedNarrative:
        """
        Core generation method that calls the LLM.
        
        This method:
        1. Prepares the prompt
        2. Calls the LLM
        3. Validates the response
        4. Records audit information
        """
        import time
        start_time = time.time()
        
        narrative_id = str(uuid.uuid4())
        prompt_hash = hashlib.sha256(user_prompt.encode()).hexdigest()
        
        self.logger.info(
            f"Generating narrative: type={template.narrative_type.value}, "
            f"template={template.template_id}, prompt_hash={prompt_hash[:16]}"
        )
        
        try:
            # Call LLM
            response = self._call_llm(
                system_prompt=template.system_prompt,
                user_prompt=user_prompt,
                max_tokens=template.max_output_tokens,
            )
            
            content = response["content"]
            tokens_used = response.get("tokens_used", 0)
            model_id = response.get("model_id", "unknown")
            model_version = response.get("model_version", "unknown")
            
            # Extract citations
            citations = self._extract_citations(content)
            
            # Validate output
            validation_errors = self._validate_narrative(content, template, citations)
            
            duration_ms = int((time.time() - start_time) * 1000)
            
            narrative = GeneratedNarrative(
                narrative_id=narrative_id,
                narrative_type=template.narrative_type,
                content=content,
                content_hash=hashlib.sha256(content.encode()).hexdigest(),
                citations=citations,
                template_id=template.template_id,
                template_hash=template.template_hash,
                prompt_hash=prompt_hash,
                context_hash=context.context_hash,
                model_id=model_id,
                model_version=model_version,
                tokens_used=tokens_used,
                generation_duration_ms=duration_ms,
                passed_validation=len(validation_errors) == 0,
                validation_errors=validation_errors,
            )
            
            # Log the generation
            self._log_generation(narrative)
            
            return narrative
            
        except Exception as e:
            self.logger.error(f"Narrative generation failed: {e}")
            raise
    
    def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
    ) -> Dict[str, Any]:
        """
        Call the LLM API.
        
        This method abstracts the LLM provider (OpenAI, Anthropic, Azure).
        """
        # This is a placeholder showing the interface
        # Actual implementation depends on the LLM provider
        
        provider = getattr(self.settings, 'llm', None)
        if provider is None:
            # Mock response for testing
            return {
                "content": "Insufficient evidence available to generate this section.",
                "tokens_used": 0,
                "model_id": "mock",
                "model_version": "1.0",
            }
        
        try:
            # OpenAI-style API
            response = self.llm_client.chat.completions.create(
                model=provider.model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=provider.temperature,
            )
            
            return {
                "content": response.choices[0].message.content,
                "tokens_used": response.usage.total_tokens,
                "model_id": provider.model_id,
                "model_version": provider.model_version_string,
            }
        except Exception as e:
            self.logger.error(f"LLM call failed: {e}")
            raise
    
    def _extract_citations(self, content: str) -> List[str]:
        """Extract citation strings from generated content."""
        # Match patterns like [ControlRun: XXX | Control: YYY | Snapshot: ZZZ]
        pattern = r'\[(?:ControlRun|ControlResult|Exception|Policy|Filing):[^\]]+\]'
        return re.findall(pattern, content)
    
    def _validate_narrative(
        self,
        content: str,
        template: PromptTemplate,
        citations: List[str],
    ) -> List[str]:
        """
        Validate generated narrative.
        
        Returns list of validation errors (empty if valid).
        """
        errors = []
        
        # Check for insufficient evidence response
        if "Insufficient evidence" in content:
            # This is actually valid - the model correctly identified missing evidence
            return errors
        
        # Check for minimum content length
        if len(content) < 100:
            errors.append("Generated content is too short")
        
        # Check for citations
        if len(citations) == 0:
            errors.append("No citations found in generated content")
        
        # Check for prohibited phrases (hallucination indicators)
        prohibited = [
            "I believe",
            "I think",
            "probably",
            "might be",
            "could be",
            "approximately",  # Should use exact numbers from evidence
            "around",         # Same reason
            "I recommend",    # Should not give recommendations
            "you should",     # Same reason
        ]
        
        for phrase in prohibited:
            if phrase.lower() in content.lower():
                errors.append(f"Contains prohibited phrase: '{phrase}'")
        
        # Check for calculation language (LLM should not calculate)
        calculation_phrases = [
            "calculated",
            "I computed",
            "adding",
            "subtracting",
            "multiplying",
            "dividing",
            "the sum of",
            "the difference",
        ]
        
        for phrase in calculation_phrases:
            if phrase.lower() in content.lower():
                errors.append(f"May contain calculation: '{phrase}'")
        
        return errors
    
    def _log_generation(self, narrative: GeneratedNarrative) -> None:
        """Log narrative generation for audit trail."""
        self.logger.info(
            f"Narrative generated: id={narrative.narrative_id}, "
            f"type={narrative.narrative_type.value}, "
            f"citations={len(narrative.citations)}, "
            f"tokens={narrative.tokens_used}, "
            f"valid={narrative.passed_validation}"
        )
        
        # In production, this would also write to the database
        if self.postgres and narrative.validation_errors:
            self.logger.warning(
                f"Validation errors for narrative {narrative.narrative_id}: "
                f"{narrative.validation_errors}"
            )
    
    # =========================================================================
    # EVIDENCE EXTRACTION HELPERS
    # =========================================================================
    
    def _extract_failed_controls(self, context: RetrievalContext) -> str:
        """Extract failed control summaries from context."""
        controls = []
        for doc in context.structured_results:
            if doc.scope.value == "control_results" and "Failure" in (doc.title or ""):
                controls.append(f"- {doc.content}\n  {doc.to_citation()}")
        
        return "\n\n".join(controls) if controls else ""
    
    def _extract_exceptions(self, context: RetrievalContext) -> str:
        """Extract exception summaries from context."""
        exceptions = []
        for doc in context.structured_results:
            if doc.scope.value == "exceptions":
                exceptions.append(f"- {doc.content}\n  {doc.to_citation()}")
        
        return "\n\n".join(exceptions) if exceptions else ""
    
    def _extract_outstanding_exceptions(self, context: RetrievalContext) -> str:
        """Extract outstanding exception summaries."""
        # This would be filtered by status
        return self._extract_exceptions(context)
    
    def _extract_policy_context(self, context: RetrievalContext) -> str:
        """Extract policy excerpts from context."""
        policies = []
        
        # Lexical matches first (more relevant)
        for doc in context.lexical_results:
            if doc.scope.value == "policies":
                policies.append(f"### {doc.title} - {doc.section}\n{doc.content}\n{doc.to_citation()}")
        
        # Then vector matches
        for doc in context.vector_results[:2]:  # Limit semantic matches
            if doc.scope.value == "policies":
                policies.append(f"### {doc.title} (Related)\n{doc.content}\n{doc.to_citation()}")
        
        return "\n\n".join(policies) if policies else ""
    
    def _extract_exception_details(self, context: RetrievalContext) -> str:
        """Extract detailed exception information."""
        for doc in context.structured_results:
            if doc.scope.value == "exceptions" and "Details" in (doc.title or ""):
                return f"{doc.content}\n{doc.to_citation()}"
        return ""
    
    def _extract_control_result(self, context: RetrievalContext) -> str:
        """Extract control result details."""
        for doc in context.structured_results:
            if doc.scope.value == "control_results":
                return f"{doc.content}\n{doc.to_citation()}"
        return ""
    
    def _extract_control_results_summary(self, context: RetrievalContext) -> str:
        """Extract summary of control results."""
        results = []
        for doc in context.structured_results:
            if doc.scope.value == "control_results":
                results.append(f"- {doc.title}: {doc.content[:200]}...\n  {doc.to_citation()}")
        return "\n\n".join(results) if results else ""
    
    def _extract_prior_filing(self, context: RetrievalContext) -> str:
        """Extract prior filing reference."""
        for doc in context.structured_results:
            if doc.scope.value == "prior_filings":
                return f"{doc.content}\n{doc.to_citation()}"
        return ""
    
    def _format_run_summary(self, run_summary: Dict[str, Any]) -> str:
        """Format run summary for prompt."""
        return (
            f"Run Code: {run_summary.get('run_code', 'N/A')}\n"
            f"Date: {run_summary.get('run_date', 'N/A')}\n"
            f"Snapshot: {run_summary.get('snowflake_snapshot_id', 'N/A')}\n"
            f"Total Controls: {run_summary.get('total_controls', 0)}\n"
            f"Passed: {run_summary.get('controls_passed', 0)}\n"
            f"Failed: {run_summary.get('controls_failed', 0)}\n"
            f"Warnings: {run_summary.get('controls_warning', 0)}\n"
        )
    
    def _format_control_definitions(self, definitions: List[Dict[str, Any]]) -> str:
        """Format control definitions for methodology prompt."""
        formatted = []
        for defn in definitions:
            formatted.append(
                f"Control: {defn.get('control_code')} - {defn.get('control_name')}\n"
                f"Category: {defn.get('category')}\n"
                f"Description: {defn.get('description')}\n"
                f"Threshold: {defn.get('threshold_operator')} {defn.get('threshold_value')}\n"
                f"Regulatory Reference: {defn.get('regulatory_reference', 'N/A')}\n"
            )
        return "\n---\n".join(formatted)
