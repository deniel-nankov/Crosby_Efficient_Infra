"""Document Builder - PDF Generation with Locked Structure and Audit Trail."""

from .builder import (
    DocumentBuilder,
    DocumentType,
    DocumentSection,
    SectionType,
    DocumentMetadata,
    GeneratedDocument,
)

from .professional_pdf import (
    ProfessionalCompliancePDF,
    generate_professional_pdf,
)

from .institutional_pdf import (
    InstitutionalCompliancePDF,
    FundMetrics,
    ControlResult,
    ExceptionRecord,
    PositionDetail,
    RiskLevel,
    LimitStatus,
    LimitGauge,
    create_sample_fund_metrics,
    create_sample_control_results,
    create_sample_exceptions,
    create_sample_positions,
)

__all__ = [
    # Core builder
    "DocumentBuilder",
    "DocumentType",
    "DocumentSection",
    "SectionType",
    "DocumentMetadata",
    "GeneratedDocument",
    # Professional PDF
    "ProfessionalCompliancePDF",
    "generate_professional_pdf",
    # Institutional PDF (hedge fund grade)
    "InstitutionalCompliancePDF",
    "FundMetrics",
    "ControlResult",
    "ExceptionRecord",
    "PositionDetail",
    "RiskLevel",
    "LimitStatus",
    "LimitGauge",
    "create_sample_fund_metrics",
    "create_sample_control_results",
    "create_sample_exceptions",
    "create_sample_positions",
]

