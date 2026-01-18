-- =============================================================================
-- PostgreSQL Initialization Script
-- =============================================================================
-- Creates tables for:
--   1. Vector store (pgvector) for policy embeddings
--   2. Compliance metadata
--   3. Audit trail
-- =============================================================================

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- =============================================================================
-- POLICY STORE (for RAG)
-- =============================================================================

CREATE TABLE IF NOT EXISTS policies (
    id SERIAL PRIMARY KEY,
    policy_id VARCHAR(100) UNIQUE NOT NULL,
    title VARCHAR(500) NOT NULL,
    source VARCHAR(200),
    version VARCHAR(50),
    effective_date DATE,
    content TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS policy_chunks (
    id SERIAL PRIMARY KEY,
    policy_id VARCHAR(100) REFERENCES policies(policy_id),
    chunk_index INTEGER NOT NULL,
    section_path VARCHAR(500),
    content TEXT NOT NULL,
    embedding vector(1536),  -- OpenAI ada-002 dimension (or 384 for local models)
    token_count INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(policy_id, chunk_index)
);

-- Index for vector similarity search
CREATE INDEX IF NOT EXISTS idx_policy_chunks_embedding 
ON policy_chunks USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- =============================================================================
-- CONTROL DEFINITIONS
-- =============================================================================

CREATE TABLE IF NOT EXISTS control_definitions (
    id SERIAL PRIMARY KEY,
    control_id VARCHAR(100) UNIQUE NOT NULL,
    control_name VARCHAR(500) NOT NULL,
    control_type VARCHAR(100) NOT NULL,
    description TEXT,
    threshold DECIMAL(10,4),
    threshold_operator VARCHAR(10),
    warning_threshold DECIMAL(10,4),
    policy_reference VARCHAR(200),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =============================================================================
-- COMPLIANCE RESULTS
-- =============================================================================

CREATE TABLE IF NOT EXISTS compliance_snapshots (
    id SERIAL PRIMARY KEY,
    snapshot_id VARCHAR(100) UNIQUE NOT NULL,
    as_of_date DATE NOT NULL,
    source_system VARCHAR(200),
    nav DECIMAL(20,2),
    position_count INTEGER,
    control_count INTEGER,
    data_hash VARCHAR(64),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS control_results (
    id SERIAL PRIMARY KEY,
    snapshot_id VARCHAR(100) REFERENCES compliance_snapshots(snapshot_id),
    control_id VARCHAR(100) NOT NULL,
    control_name VARCHAR(500),
    control_type VARCHAR(100),
    calculated_value DECIMAL(10,4),
    threshold DECIMAL(10,4),
    threshold_operator VARCHAR(10),
    status VARCHAR(20) NOT NULL,  -- pass, warning, fail
    breach_amount DECIMAL(10,4),
    details JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    INDEX idx_control_results_snapshot (snapshot_id),
    INDEX idx_control_results_status (status)
);

-- =============================================================================
-- GENERATED NARRATIVES
-- =============================================================================

CREATE TABLE IF NOT EXISTS generated_narratives (
    id SERIAL PRIMARY KEY,
    narrative_id VARCHAR(100) UNIQUE NOT NULL,
    snapshot_id VARCHAR(100) REFERENCES compliance_snapshots(snapshot_id),
    control_id VARCHAR(100) NOT NULL,
    content TEXT NOT NULL,
    content_hash VARCHAR(64) NOT NULL,
    citations TEXT[],
    model_used VARCHAR(100),
    prompt_hash VARCHAR(64),
    context_hash VARCHAR(64),
    generated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    INDEX idx_narratives_snapshot (snapshot_id),
    INDEX idx_narratives_control (control_id)
);

-- =============================================================================
-- AUDIT TRAIL
-- =============================================================================

CREATE TABLE IF NOT EXISTS audit_log (
    id SERIAL PRIMARY KEY,
    event_type VARCHAR(100) NOT NULL,
    entity_type VARCHAR(100),
    entity_id VARCHAR(100),
    user_id VARCHAR(100),
    action VARCHAR(100) NOT NULL,
    details JSONB,
    ip_address INET,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    INDEX idx_audit_log_entity (entity_type, entity_id),
    INDEX idx_audit_log_time (created_at)
);

-- =============================================================================
-- AIRFLOW METADATA (Airflow will create its own tables too)
-- =============================================================================

CREATE TABLE IF NOT EXISTS dag_runs_metadata (
    id SERIAL PRIMARY KEY,
    dag_id VARCHAR(250) NOT NULL,
    run_id VARCHAR(250) NOT NULL,
    execution_date TIMESTAMP WITH TIME ZONE,
    status VARCHAR(50),
    snapshot_id VARCHAR(100),
    records_processed INTEGER,
    errors_count INTEGER,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    
    UNIQUE(dag_id, run_id)
);

-- =============================================================================
-- SNOWFLAKE SYNC TRACKING
-- =============================================================================

CREATE TABLE IF NOT EXISTS snowflake_sync_log (
    id SERIAL PRIMARY KEY,
    sync_id VARCHAR(100) UNIQUE NOT NULL,
    table_name VARCHAR(200) NOT NULL,
    sync_type VARCHAR(50),  -- full, incremental
    records_synced INTEGER,
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    status VARCHAR(50),
    error_message TEXT,
    
    INDEX idx_snowflake_sync_table (table_name),
    INDEX idx_snowflake_sync_time (started_at)
);

-- =============================================================================
-- VIEWS
-- =============================================================================

CREATE OR REPLACE VIEW v_latest_compliance_status AS
SELECT 
    cs.as_of_date,
    cs.nav,
    cr.control_id,
    cr.control_name,
    cr.control_type,
    cr.calculated_value,
    cr.threshold,
    cr.status,
    cr.breach_amount
FROM compliance_snapshots cs
JOIN control_results cr ON cs.snapshot_id = cr.snapshot_id
WHERE cs.as_of_date = (SELECT MAX(as_of_date) FROM compliance_snapshots);

CREATE OR REPLACE VIEW v_breach_history AS
SELECT 
    cs.as_of_date,
    cr.control_id,
    cr.control_name,
    cr.calculated_value,
    cr.threshold,
    cr.breach_amount,
    gn.content as narrative
FROM compliance_snapshots cs
JOIN control_results cr ON cs.snapshot_id = cr.snapshot_id
LEFT JOIN generated_narratives gn ON cr.snapshot_id = gn.snapshot_id AND cr.control_id = gn.control_id
WHERE cr.status IN ('warning', 'fail')
ORDER BY cs.as_of_date DESC;

-- =============================================================================
-- GRANTS
-- =============================================================================

GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO compliance_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO compliance_user;
