-- =============================================================================
-- COMPLIANCE RAG SYSTEM - POSTGRES SCHEMA
-- Evidence Store & Audit Trail
-- =============================================================================
-- 
-- Purpose: Provides the audit trail and workflow management for SEC compliance.
-- This schema stores:
--   - Control execution records
--   - Control results with deterministic outputs
--   - Exceptions and breaches
--   - Approvals and attestations
--   - Document generation audit trail
--
-- SEC Examination Note: All tables include immutable audit columns.
-- Records are never deleted - only soft-deleted with status changes.
-- All timestamps are UTC. All IDs are UUIDs for global uniqueness.
-- =============================================================================

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- =============================================================================
-- CORE REFERENCE TABLES
-- =============================================================================

-- Control definitions (what controls exist)
CREATE TABLE control_definitions (
    control_id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    control_code            VARCHAR(50) NOT NULL UNIQUE,  -- e.g., 'CONC_001', 'LIQ_002'
    control_name            VARCHAR(255) NOT NULL,
    control_category        VARCHAR(50) NOT NULL,         -- 'concentration', 'liquidity', 'counterparty', 'exposure'
    description             TEXT NOT NULL,
    threshold_type          VARCHAR(50) NOT NULL,         -- 'percentage', 'absolute', 'ratio'
    threshold_value         NUMERIC(20, 8),
    threshold_operator      VARCHAR(10) NOT NULL,         -- 'lt', 'lte', 'gt', 'gte', 'eq', 'neq'
    frequency               VARCHAR(20) NOT NULL,         -- 'daily', 'weekly', 'monthly', 'quarterly'
    regulatory_reference    TEXT,                         -- e.g., 'Form PF Section 2a'
    policy_document_id      VARCHAR(255),                 -- Reference to policy document
    is_active               BOOLEAN NOT NULL DEFAULT TRUE,
    effective_date          DATE NOT NULL,
    expiration_date         DATE,
    created_at              TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    created_by              VARCHAR(100) NOT NULL,
    updated_at              TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_by              VARCHAR(100) NOT NULL,
    
    CONSTRAINT chk_threshold_operator CHECK (threshold_operator IN ('lt', 'lte', 'gt', 'gte', 'eq', 'neq')),
    CONSTRAINT chk_frequency CHECK (frequency IN ('daily', 'weekly', 'monthly', 'quarterly', 'annual')),
    CONSTRAINT chk_category CHECK (control_category IN ('concentration', 'liquidity', 'counterparty', 'exposure', 'leverage', 'valuation', 'reconciliation', 'regulatory'))
);

-- Approved signatories for compliance actions
CREATE TABLE signatories (
    signatory_id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    employee_id             VARCHAR(50) NOT NULL UNIQUE,
    full_name               VARCHAR(255) NOT NULL,
    title                   VARCHAR(255) NOT NULL,
    email                   VARCHAR(255) NOT NULL,
    approval_level          INTEGER NOT NULL,             -- 1=analyst, 2=manager, 3=CCO, 4=CEO
    can_approve_exceptions  BOOLEAN NOT NULL DEFAULT FALSE,
    can_approve_filings     BOOLEAN NOT NULL DEFAULT FALSE,
    digital_signature_hash  VARCHAR(512),                 -- For document signing
    is_active               BOOLEAN NOT NULL DEFAULT TRUE,
    created_at              TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

-- =============================================================================
-- CONTROL EXECUTION TABLES
-- =============================================================================

-- Each control run (batch execution)
CREATE TABLE control_runs (
    run_id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_code                VARCHAR(100) NOT NULL UNIQUE, -- e.g., '2026-01-15-DAILY-001'
    run_type                VARCHAR(50) NOT NULL,         -- 'scheduled', 'ad-hoc', 'remediation'
    run_date                DATE NOT NULL,
    run_timestamp_start     TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    run_timestamp_end       TIMESTAMP WITH TIME ZONE,
    
    -- Data lineage - critical for reproducibility
    snowflake_snapshot_id   VARCHAR(255) NOT NULL,        -- e.g., 'SNOW_2026_01_15_EOD'
    snowflake_snapshot_ts   TIMESTAMP WITH TIME ZONE NOT NULL,
    postgres_snapshot_lsn   VARCHAR(100),                 -- Postgres Log Sequence Number
    
    -- Execution metadata
    executor_service        VARCHAR(100) NOT NULL,        -- 'control-runner-prod-v2.3.1'
    executor_version        VARCHAR(50) NOT NULL,
    config_hash             VARCHAR(64) NOT NULL,         -- SHA-256 of control config at runtime
    
    -- Results summary (denormalized for fast querying)
    total_controls          INTEGER NOT NULL DEFAULT 0,
    controls_passed         INTEGER NOT NULL DEFAULT 0,
    controls_failed         INTEGER NOT NULL DEFAULT 0,
    controls_warning        INTEGER NOT NULL DEFAULT 0,
    controls_skipped        INTEGER NOT NULL DEFAULT 0,
    
    -- Status
    status                  VARCHAR(20) NOT NULL DEFAULT 'running',
    error_message           TEXT,
    
    created_at              TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    created_by              VARCHAR(100) NOT NULL DEFAULT 'system',
    
    CONSTRAINT chk_run_status CHECK (status IN ('running', 'completed', 'failed', 'cancelled')),
    CONSTRAINT chk_run_type CHECK (run_type IN ('scheduled', 'ad-hoc', 'remediation', 'backfill'))
);

-- Index for common queries
CREATE INDEX idx_control_runs_date ON control_runs(run_date DESC);
CREATE INDEX idx_control_runs_status ON control_runs(status, run_date DESC);

-- Individual control results within a run
CREATE TABLE control_results (
    result_id               UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_id                  UUID NOT NULL REFERENCES control_runs(run_id),
    control_id              UUID NOT NULL REFERENCES control_definitions(control_id),
    
    -- Deterministic calculation outputs
    calculated_value        NUMERIC(20, 8),
    threshold_value         NUMERIC(20, 8) NOT NULL,      -- Snapshot of threshold at execution time
    threshold_operator      VARCHAR(10) NOT NULL,
    
    -- Result
    result_status           VARCHAR(20) NOT NULL,         -- 'pass', 'fail', 'warning', 'skip', 'error'
    breach_amount           NUMERIC(20, 8),               -- If failed, by how much
    breach_percentage       NUMERIC(10, 4),               -- Breach as percentage
    
    -- Evidence linkage
    evidence_query_hash     VARCHAR(64) NOT NULL,         -- SHA-256 of the SQL query used
    evidence_row_count      INTEGER NOT NULL,
    evidence_sample_json    JSONB,                        -- Sample of underlying data (first 10 rows)
    
    -- Computation details
    computation_sql         TEXT NOT NULL,                -- Exact SQL that produced the result
    computation_duration_ms INTEGER NOT NULL,
    
    -- Narrative placeholder (filled by narrative generator)
    narrative_summary       TEXT,
    narrative_generated_at  TIMESTAMP WITH TIME ZONE,
    narrative_model_version VARCHAR(100),
    
    created_at              TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    CONSTRAINT chk_result_status CHECK (result_status IN ('pass', 'fail', 'warning', 'skip', 'error'))
);

-- Indexes for control results
CREATE INDEX idx_control_results_run ON control_results(run_id);
CREATE INDEX idx_control_results_control ON control_results(control_id);
CREATE INDEX idx_control_results_status ON control_results(result_status, created_at DESC);
CREATE INDEX idx_control_results_failed ON control_results(run_id) WHERE result_status = 'fail';

-- =============================================================================
-- EXCEPTION MANAGEMENT
-- =============================================================================

-- Exceptions raised from control failures
CREATE TABLE exceptions (
    exception_id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    exception_code          VARCHAR(50) NOT NULL UNIQUE,  -- e.g., 'EXC-2026-0001'
    result_id               UUID NOT NULL REFERENCES control_results(result_id),
    run_id                  UUID NOT NULL REFERENCES control_runs(run_id),
    control_id              UUID NOT NULL REFERENCES control_definitions(control_id),
    
    -- Exception details
    severity                VARCHAR(20) NOT NULL,         -- 'critical', 'high', 'medium', 'low'
    exception_type          VARCHAR(50) NOT NULL,         -- 'breach', 'anomaly', 'data_quality', 'system'
    title                   VARCHAR(500) NOT NULL,
    description             TEXT NOT NULL,
    
    -- Quantitative details (copied from control_result for audit trail)
    breach_value            NUMERIC(20, 8),
    threshold_value         NUMERIC(20, 8),
    breach_amount           NUMERIC(20, 8),
    
    -- Workflow status
    status                  VARCHAR(30) NOT NULL DEFAULT 'open',
    assigned_to             UUID REFERENCES signatories(signatory_id),
    escalation_level        INTEGER NOT NULL DEFAULT 1,
    
    -- Resolution
    resolution_type         VARCHAR(50),                  -- 'remediated', 'accepted', 'false_positive', 'waived'
    resolution_notes        TEXT,
    resolved_at             TIMESTAMP WITH TIME ZONE,
    resolved_by             UUID REFERENCES signatories(signatory_id),
    
    -- Audit
    opened_at               TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    due_date                DATE NOT NULL,
    last_updated_at         TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    last_updated_by         VARCHAR(100) NOT NULL,
    
    CONSTRAINT chk_exception_severity CHECK (severity IN ('critical', 'high', 'medium', 'low')),
    CONSTRAINT chk_exception_status CHECK (status IN ('open', 'in_review', 'pending_approval', 'approved', 'closed', 'escalated'))
);

-- Indexes for exceptions
CREATE INDEX idx_exceptions_status ON exceptions(status, severity);
CREATE INDEX idx_exceptions_run ON exceptions(run_id);
CREATE INDEX idx_exceptions_open ON exceptions(status, opened_at DESC) WHERE status = 'open';
CREATE INDEX idx_exceptions_due ON exceptions(due_date) WHERE status NOT IN ('closed', 'approved');

-- Exception comments/activity log
CREATE TABLE exception_activities (
    activity_id             UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    exception_id            UUID NOT NULL REFERENCES exceptions(exception_id),
    activity_type           VARCHAR(50) NOT NULL,         -- 'comment', 'status_change', 'assignment', 'escalation'
    activity_description    TEXT NOT NULL,
    previous_value          TEXT,
    new_value               TEXT,
    performed_by            UUID REFERENCES signatories(signatory_id),
    performed_at            TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_exception_activities_exception ON exception_activities(exception_id, performed_at DESC);

-- =============================================================================
-- APPROVALS & ATTESTATIONS
-- =============================================================================

-- Approval records (immutable)
CREATE TABLE approvals (
    approval_id             UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    approval_type           VARCHAR(50) NOT NULL,         -- 'exception', 'document', 'filing', 'control_override'
    reference_type          VARCHAR(50) NOT NULL,         -- 'exception', 'document_generation', 'sec_filing'
    reference_id            UUID NOT NULL,                -- Foreign key to relevant table
    
    -- Approval chain
    approval_level          INTEGER NOT NULL,
    required_level          INTEGER NOT NULL,
    
    -- Approver
    approved_by             UUID NOT NULL REFERENCES signatories(signatory_id),
    approval_status         VARCHAR(20) NOT NULL,         -- 'approved', 'rejected', 'pending'
    
    -- Evidence
    approval_notes          TEXT,
    digital_signature       VARCHAR(512),                 -- Cryptographic signature
    ip_address              VARCHAR(45),
    user_agent              VARCHAR(500),
    
    -- Timestamps
    requested_at            TIMESTAMP WITH TIME ZONE NOT NULL,
    decided_at              TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    CONSTRAINT chk_approval_status CHECK (approval_status IN ('approved', 'rejected', 'pending'))
);

CREATE INDEX idx_approvals_reference ON approvals(reference_type, reference_id);
CREATE INDEX idx_approvals_approver ON approvals(approved_by, decided_at DESC);

-- =============================================================================
-- DOCUMENT GENERATION AUDIT
-- =============================================================================

-- Document generation records
CREATE TABLE document_generations (
    document_id             UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_code           VARCHAR(100) NOT NULL UNIQUE, -- e.g., 'DCP-2026-01-15-001'
    document_type           VARCHAR(50) NOT NULL,         -- 'daily_compliance_pack', 'form_pf_workpaper', '13f_workpaper'
    document_date           DATE NOT NULL,
    
    -- Source data
    run_id                  UUID REFERENCES control_runs(run_id),
    source_snapshot_id      VARCHAR(255) NOT NULL,
    
    -- Generation details
    template_id             VARCHAR(100) NOT NULL,
    template_version        VARCHAR(50) NOT NULL,
    template_hash           VARCHAR(64) NOT NULL,
    
    -- LLM usage (for narrative sections only)
    llm_model_id            VARCHAR(100),
    llm_model_version       VARCHAR(50),
    llm_prompt_hash         VARCHAR(64),                  -- Hash of the prompt template
    llm_tokens_used         INTEGER,
    
    -- Output
    output_format           VARCHAR(20) NOT NULL,         -- 'pdf', 'xlsx', 'json'
    output_file_path        VARCHAR(500) NOT NULL,
    output_file_hash        VARCHAR(64) NOT NULL,         -- SHA-256 of output file
    output_file_size_bytes  BIGINT NOT NULL,
    page_count              INTEGER,
    
    -- Review workflow
    status                  VARCHAR(30) NOT NULL DEFAULT 'generated',
    reviewed_by             UUID REFERENCES signatories(signatory_id),
    reviewed_at             TIMESTAMP WITH TIME ZONE,
    review_notes            TEXT,
    
    -- Approval for distribution
    approved_by             UUID REFERENCES signatories(signatory_id),
    approved_at             TIMESTAMP WITH TIME ZONE,
    
    -- Metadata
    generated_at            TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    generated_by            VARCHAR(100) NOT NULL DEFAULT 'system',
    
    CONSTRAINT chk_doc_status CHECK (status IN ('generated', 'pending_review', 'reviewed', 'approved', 'rejected', 'distributed')),
    CONSTRAINT chk_doc_format CHECK (output_format IN ('pdf', 'xlsx', 'json', 'html'))
);

CREATE INDEX idx_document_generations_type_date ON document_generations(document_type, document_date DESC);
CREATE INDEX idx_document_generations_run ON document_generations(run_id);

-- Document sections (for granular audit of LLM-generated content)
CREATE TABLE document_sections (
    section_id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id             UUID NOT NULL REFERENCES document_generations(document_id),
    section_order           INTEGER NOT NULL,
    section_type            VARCHAR(50) NOT NULL,         -- 'header', 'metrics_table', 'narrative', 'appendix'
    section_name            VARCHAR(255) NOT NULL,
    
    -- Content
    content_type            VARCHAR(20) NOT NULL,         -- 'deterministic', 'llm_generated', 'template'
    content_text            TEXT,
    content_hash            VARCHAR(64) NOT NULL,
    
    -- For LLM-generated sections
    evidence_ids            UUID[],                       -- Array of control_result IDs used as evidence
    prompt_template_id      VARCHAR(100),
    prompt_hash             VARCHAR(64),
    
    -- Citations embedded in this section
    citation_count          INTEGER NOT NULL DEFAULT 0,
    
    created_at              TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_document_sections_document ON document_sections(document_id, section_order);

-- =============================================================================
-- POLICY & PROCEDURE DOCUMENTS
-- =============================================================================

-- Policy document registry
CREATE TABLE policy_documents (
    policy_id               UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    policy_code             VARCHAR(50) NOT NULL UNIQUE,  -- e.g., 'POL-CONC-001'
    title                   VARCHAR(500) NOT NULL,
    category                VARCHAR(100) NOT NULL,
    version                 VARCHAR(20) NOT NULL,
    
    -- Content
    file_path               VARCHAR(500) NOT NULL,
    file_hash               VARCHAR(64) NOT NULL,
    content_text            TEXT,                         -- Extracted text for search
    
    -- Embedding for vector search
    embedding_model         VARCHAR(100),
    embedding_vector_id     VARCHAR(255),                 -- Reference to vector store
    
    -- Lifecycle
    effective_date          DATE NOT NULL,
    expiration_date         DATE,
    status                  VARCHAR(20) NOT NULL DEFAULT 'active',
    
    -- Approvals
    approved_by             UUID REFERENCES signatories(signatory_id),
    approved_at             TIMESTAMP WITH TIME ZONE,
    
    created_at              TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    CONSTRAINT chk_policy_status CHECK (status IN ('draft', 'active', 'superseded', 'archived'))
);

CREATE INDEX idx_policy_documents_category ON policy_documents(category, status);
CREATE INDEX idx_policy_documents_active ON policy_documents(status) WHERE status = 'active';

-- Policy sections (for granular retrieval)
CREATE TABLE policy_sections (
    section_id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    policy_id               UUID NOT NULL REFERENCES policy_documents(policy_id),
    section_number          VARCHAR(50) NOT NULL,         -- e.g., '2.3.1'
    section_title           VARCHAR(500) NOT NULL,
    section_text            TEXT NOT NULL,
    
    -- Vector embedding reference
    embedding_vector_id     VARCHAR(255),
    
    created_at              TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_policy_sections_policy ON policy_sections(policy_id);

-- =============================================================================
-- RECONCILIATION TRACKING
-- =============================================================================

-- Reconciliation breaks
CREATE TABLE reconciliation_breaks (
    break_id                UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    break_code              VARCHAR(50) NOT NULL UNIQUE,
    run_id                  UUID NOT NULL REFERENCES control_runs(run_id),
    
    -- Break details
    reconciliation_type     VARCHAR(50) NOT NULL,         -- 'position', 'cash', 'nav', 'trade'
    source_system           VARCHAR(100) NOT NULL,
    target_system           VARCHAR(100) NOT NULL,
    
    -- Variance
    source_value            NUMERIC(20, 8) NOT NULL,
    target_value            NUMERIC(20, 8) NOT NULL,
    variance_amount         NUMERIC(20, 8) NOT NULL,
    variance_percentage     NUMERIC(10, 4),
    
    -- Identifiers
    security_id             VARCHAR(50),
    account_id              VARCHAR(50),
    
    -- Status
    status                  VARCHAR(30) NOT NULL DEFAULT 'open',
    resolution_notes        TEXT,
    resolved_at             TIMESTAMP WITH TIME ZONE,
    
    created_at              TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    CONSTRAINT chk_break_status CHECK (status IN ('open', 'investigating', 'resolved', 'accepted'))
);

CREATE INDEX idx_recon_breaks_run ON reconciliation_breaks(run_id);
CREATE INDEX idx_recon_breaks_open ON reconciliation_breaks(status) WHERE status = 'open';

-- =============================================================================
-- AUDIT LOG (IMMUTABLE)
-- =============================================================================

-- System-wide audit log
CREATE TABLE audit_log (
    log_id                  BIGSERIAL PRIMARY KEY,
    event_timestamp         TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    event_type              VARCHAR(50) NOT NULL,
    table_name              VARCHAR(100) NOT NULL,
    record_id               UUID NOT NULL,
    action                  VARCHAR(20) NOT NULL,         -- 'insert', 'update', 'delete'
    old_values              JSONB,
    new_values              JSONB,
    changed_by              VARCHAR(100) NOT NULL,
    ip_address              VARCHAR(45),
    session_id              VARCHAR(100)
);

-- Partition by month for performance
CREATE INDEX idx_audit_log_timestamp ON audit_log(event_timestamp DESC);
CREATE INDEX idx_audit_log_table_record ON audit_log(table_name, record_id);

-- =============================================================================
-- VIEWS FOR COMMON QUERIES
-- =============================================================================

-- Daily compliance summary view
CREATE OR REPLACE VIEW v_daily_compliance_summary AS
SELECT 
    cr.run_id,
    cr.run_code,
    cr.run_date,
    cr.snowflake_snapshot_id,
    cr.total_controls,
    cr.controls_passed,
    cr.controls_failed,
    cr.controls_warning,
    ROUND(cr.controls_passed::NUMERIC / NULLIF(cr.total_controls, 0) * 100, 2) AS pass_rate,
    COUNT(DISTINCT e.exception_id) FILTER (WHERE e.status = 'open') AS open_exceptions,
    COUNT(DISTINCT e.exception_id) FILTER (WHERE e.severity = 'critical' AND e.status = 'open') AS critical_exceptions,
    cr.run_timestamp_start,
    cr.run_timestamp_end,
    EXTRACT(EPOCH FROM (cr.run_timestamp_end - cr.run_timestamp_start)) AS duration_seconds
FROM control_runs cr
LEFT JOIN exceptions e ON cr.run_id = e.run_id
WHERE cr.status = 'completed'
GROUP BY cr.run_id, cr.run_code, cr.run_date, cr.snowflake_snapshot_id,
         cr.total_controls, cr.controls_passed, cr.controls_failed, cr.controls_warning,
         cr.run_timestamp_start, cr.run_timestamp_end;

-- Active exceptions dashboard view
CREATE OR REPLACE VIEW v_active_exceptions AS
SELECT 
    e.exception_id,
    e.exception_code,
    e.title,
    e.severity,
    e.status,
    e.opened_at,
    e.due_date,
    CURRENT_DATE - e.due_date AS days_overdue,
    cd.control_code,
    cd.control_name,
    cd.control_category,
    s.full_name AS assigned_to_name,
    cr.run_date
FROM exceptions e
JOIN control_definitions cd ON e.control_id = cd.control_id
JOIN control_runs cr ON e.run_id = cr.run_id
LEFT JOIN signatories s ON e.assigned_to = s.signatory_id
WHERE e.status NOT IN ('closed', 'approved');

-- =============================================================================
-- FUNCTIONS FOR COMMON OPERATIONS
-- =============================================================================

-- Generate exception code
CREATE OR REPLACE FUNCTION generate_exception_code()
RETURNS VARCHAR(50) AS $$
DECLARE
    next_seq INTEGER;
    year_part VARCHAR(4);
BEGIN
    year_part := TO_CHAR(CURRENT_DATE, 'YYYY');
    SELECT COALESCE(MAX(CAST(SUBSTRING(exception_code FROM 10 FOR 4) AS INTEGER)), 0) + 1
    INTO next_seq
    FROM exceptions
    WHERE exception_code LIKE 'EXC-' || year_part || '-%';
    
    RETURN 'EXC-' || year_part || '-' || LPAD(next_seq::TEXT, 4, '0');
END;
$$ LANGUAGE plpgsql;

-- Generate document code
CREATE OR REPLACE FUNCTION generate_document_code(doc_type VARCHAR, doc_date DATE)
RETURNS VARCHAR(100) AS $$
DECLARE
    next_seq INTEGER;
    prefix VARCHAR(10);
BEGIN
    prefix := CASE doc_type
        WHEN 'daily_compliance_pack' THEN 'DCP'
        WHEN 'form_pf_workpaper' THEN 'PF'
        WHEN '13f_workpaper' THEN '13F'
        WHEN 'adv_workpaper' THEN 'ADV'
        ELSE 'DOC'
    END;
    
    SELECT COALESCE(MAX(CAST(SUBSTRING(document_code FROM LENGTH(document_code) - 2) AS INTEGER)), 0) + 1
    INTO next_seq
    FROM document_generations
    WHERE document_type = doc_type AND document_date = doc_date;
    
    RETURN prefix || '-' || TO_CHAR(doc_date, 'YYYY-MM-DD') || '-' || LPAD(next_seq::TEXT, 3, '0');
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- COMMENTS FOR DOCUMENTATION
-- =============================================================================

COMMENT ON TABLE control_runs IS 'Audit trail of each control batch execution. Each record represents a complete run of all active controls against a specific data snapshot.';
COMMENT ON TABLE control_results IS 'Individual control execution results. Contains deterministic calculations and evidence linkage.';
COMMENT ON TABLE exceptions IS 'Compliance exceptions requiring review and resolution. Workflow managed through status transitions.';
COMMENT ON TABLE approvals IS 'Immutable approval records for audit trail. Supports multi-level approval chains.';
COMMENT ON TABLE document_generations IS 'Audit trail of all generated documents including hash verification and LLM usage tracking.';
COMMENT ON COLUMN control_runs.snowflake_snapshot_id IS 'Critical for reproducibility - identifies exact data snapshot used for control execution.';
COMMENT ON COLUMN control_results.computation_sql IS 'Exact SQL query used for calculation - enables reproduction of results.';
COMMENT ON COLUMN document_sections.content_type IS 'Distinguishes between deterministic (SQL/Python computed) and LLM-generated content for audit purposes.';


-- =============================================================================
-- DATA INTEGRATION TABLES
-- =============================================================================

-- Position staging table for data integration
CREATE TABLE IF NOT EXISTS position_staging (
    staging_id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    snapshot_id             VARCHAR(255) NOT NULL,
    snapshot_date           DATE NOT NULL,
    position_count          INTEGER NOT NULL,
    loaded_at               TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    status                  VARCHAR(20) NOT NULL DEFAULT 'loaded',
    quality_score           NUMERIC(5, 2),
    validation_errors       JSONB,
    processed_at            TIMESTAMP WITH TIME ZONE,
    
    CONSTRAINT chk_staging_status CHECK (status IN ('loaded', 'validated', 'processed', 'rejected'))
);

-- Current positions (loaded from Snowflake)
CREATE TABLE IF NOT EXISTS positions_current (
    id                      BIGSERIAL PRIMARY KEY,
    staging_id              UUID REFERENCES position_staging(staging_id),
    position_id             VARCHAR(100) NOT NULL,
    snapshot_id             VARCHAR(255) NOT NULL,
    fund_id                 VARCHAR(100) NOT NULL,
    security_id             VARCHAR(100) NOT NULL,
    ticker                  VARCHAR(50),
    isin                    VARCHAR(12),
    quantity                NUMERIC(20, 8) NOT NULL,
    market_value_usd        NUMERIC(20, 2) NOT NULL,
    currency                VARCHAR(3) NOT NULL DEFAULT 'USD',
    asset_class             VARCHAR(50),
    sector                  VARCHAR(100),
    issuer_id               VARCHAR(100),
    issuer_name             VARCHAR(255),
    price                   NUMERIC(20, 8),
    price_date              DATE,
    is_stale_price          BOOLEAN DEFAULT FALSE,
    created_at              TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    CONSTRAINT uk_positions_current UNIQUE (snapshot_id, position_id)
);

CREATE INDEX IF NOT EXISTS idx_positions_current_snapshot ON positions_current(snapshot_id);
CREATE INDEX IF NOT EXISTS idx_positions_current_fund ON positions_current(fund_id);
CREATE INDEX IF NOT EXISTS idx_positions_current_issuer ON positions_current(issuer_id);

-- Policy documents table
CREATE TABLE IF NOT EXISTS policy_documents (
    id                      BIGSERIAL PRIMARY KEY,
    policy_id               VARCHAR(100) NOT NULL,
    title                   VARCHAR(500) NOT NULL,
    version                 VARCHAR(50) NOT NULL,
    effective_date          DATE NOT NULL,
    expiration_date         DATE,
    content_hash            VARCHAR(64) NOT NULL,
    fund_name               VARCHAR(255),
    category                VARCHAR(100),
    chunk_count             INTEGER NOT NULL DEFAULT 0,
    quality_score           NUMERIC(5, 2),
    created_at              TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    CONSTRAINT uk_policy_version UNIQUE (policy_id, version)
);

-- Policy chunks for RAG retrieval
CREATE TABLE IF NOT EXISTS policy_chunks (
    id                      BIGSERIAL PRIMARY KEY,
    chunk_id                UUID NOT NULL UNIQUE DEFAULT uuid_generate_v4(),
    policy_id               VARCHAR(100) NOT NULL,
    policy_version          VARCHAR(50) NOT NULL,
    content                 TEXT NOT NULL,
    content_hash            VARCHAR(64) NOT NULL,
    section_path            VARCHAR(500) NOT NULL,
    section_level           INTEGER NOT NULL,
    chunk_index             INTEGER NOT NULL,
    effective_date          DATE NOT NULL,
    expiration_date         DATE,
    keywords                JSONB,
    embedding_vector        VECTOR(1536),  -- For OpenAI embeddings (optional)
    created_at              TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at              TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_policy_chunks_policy ON policy_chunks(policy_id, policy_version);
CREATE INDEX IF NOT EXISTS idx_policy_chunks_section ON policy_chunks(section_path);

-- Full-text search index on policy content
CREATE INDEX IF NOT EXISTS idx_policy_chunks_fts ON policy_chunks USING gin(to_tsvector('english', content));

-- Data quality reports
CREATE TABLE IF NOT EXISTS data_quality_reports (
    report_id               UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    dataset_name            VARCHAR(100) NOT NULL,
    record_count            INTEGER NOT NULL,
    validated_at            TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    
    -- Quality scores
    completeness_score      NUMERIC(5, 2),
    accuracy_score          NUMERIC(5, 2),
    consistency_score       NUMERIC(5, 2),
    timeliness_score        NUMERIC(5, 2),
    uniqueness_score        NUMERIC(5, 2),
    validity_score          NUMERIC(5, 2),
    overall_score           NUMERIC(5, 2),
    
    -- Decision
    is_acceptable           BOOLEAN NOT NULL,
    rejection_reason        TEXT,
    
    -- Issues
    issue_count             INTEGER NOT NULL DEFAULT 0,
    critical_issue_count    INTEGER NOT NULL DEFAULT 0,
    issues_json             JSONB,
    
    -- Context
    integration_run_id      UUID,
    snapshot_id             VARCHAR(255)
);

CREATE INDEX IF NOT EXISTS idx_quality_reports_dataset ON data_quality_reports(dataset_name, validated_at);

-- Integration run history
CREATE TABLE IF NOT EXISTS integration_runs (
    run_id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    run_type                VARCHAR(50) NOT NULL,
    started_at              TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    completed_at            TIMESTAMP WITH TIME ZONE,
    
    snapshot_id             VARCHAR(255),
    position_count          INTEGER DEFAULT 0,
    policy_count            INTEGER DEFAULT 0,
    control_count           INTEGER DEFAULT 0,
    
    position_quality_score  NUMERIC(5, 2),
    status                  VARCHAR(20) NOT NULL DEFAULT 'running',
    error_message           TEXT,
    config_hash             VARCHAR(64),
    
    CONSTRAINT chk_integration_status CHECK (status IN ('running', 'completed', 'failed', 'rejected'))
);

CREATE INDEX IF NOT EXISTS idx_integration_runs_status ON integration_runs(status, started_at);

