"""Narrative Generator - LLM-Assisted Text Generation with Evidence Binding."""

from .generator import (
    NarrativeGenerator,
    NarrativeType,
    PromptTemplate,
    GeneratedNarrative,
    TEMPLATES,
    DAILY_SUMMARY_TEMPLATE,
    EXCEPTION_NARRATIVE_TEMPLATE,
    METHODOLOGY_TEMPLATE,
    FILING_WORKPAPER_TEMPLATE,
)

__all__ = [
    "NarrativeGenerator",
    "NarrativeType",
    "PromptTemplate",
    "GeneratedNarrative",
    "TEMPLATES",
    "DAILY_SUMMARY_TEMPLATE",
    "EXCEPTION_NARRATIVE_TEMPLATE",
    "METHODOLOGY_TEMPLATE",
    "FILING_WORKPAPER_TEMPLATE",
]
