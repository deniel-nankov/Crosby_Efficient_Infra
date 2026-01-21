-- ============================================================================
-- CLICKHOUSE SCHEMA INITIALIZATION
-- ============================================================================
-- This script runs automatically when ClickHouse container starts.
-- Creates the compliance analytics database and tables.
-- ============================================================================

-- Create compliance database
CREATE DATABASE IF NOT EXISTS compliance;

-- Use compliance database
USE compliance;

-- ============================================================================
-- CONTROL RESULTS HISTORY
-- ============================================================================
-- Main analytics table for historical compliance control results.
-- Optimized for time-series queries with ReplacingMergeTree.
-- ============================================================================
CREATE TABLE IF NOT EXISTS control_results_history (
    as_of_date Date,
    control_id LowCardinality(String),
    control_name String,
    control_type LowCardinality(String),
    calculated_value Decimal(20, 6),
    threshold Decimal(20, 6),
    threshold_operator LowCardinality(String),
    status LowCardinality(String),
    breach_amount Nullable(Decimal(20, 6)),
    headroom_pct Float64,
    fund_id LowCardinality(String) DEFAULT 'MAIN',
    details String DEFAULT '{}',
    inserted_at DateTime64(3) DEFAULT now64(3),
    data_hash FixedString(32)
)
ENGINE = ReplacingMergeTree(inserted_at)
PARTITION BY toYYYYMM(as_of_date)
ORDER BY (fund_id, control_id, as_of_date)
TTL as_of_date + INTERVAL 10 YEAR
SETTINGS index_granularity = 8192;

-- ============================================================================
-- POSITIONS HISTORY
-- ============================================================================
-- Historical position snapshots for point-in-time analysis.
-- ============================================================================
CREATE TABLE IF NOT EXISTS positions_history (
    as_of_date Date,
    security_id String,
    ticker LowCardinality(String),
    security_name String,
    quantity Decimal(20, 4),
    market_value Decimal(20, 2),
    price Decimal(20, 6),
    currency LowCardinality(String) DEFAULT 'USD',
    sector LowCardinality(String),
    issuer String,
    asset_class LowCardinality(String),
    country LowCardinality(String) DEFAULT 'US',
    weight_pct Float64,
    fund_id LowCardinality(String) DEFAULT 'MAIN',
    inserted_at DateTime64(3) DEFAULT now64(3)
)
ENGINE = ReplacingMergeTree(inserted_at)
PARTITION BY toYYYYMM(as_of_date)
ORDER BY (fund_id, as_of_date, security_id)
TTL as_of_date + INTERVAL 10 YEAR
SETTINGS index_granularity = 8192;

-- ============================================================================
-- NAV HISTORY
-- ============================================================================
-- Daily Net Asset Value history.
-- ============================================================================
CREATE TABLE IF NOT EXISTS nav_history (
    as_of_date Date,
    nav Decimal(20, 2),
    currency LowCardinality(String) DEFAULT 'USD',
    fund_id LowCardinality(String) DEFAULT 'MAIN',
    inserted_at DateTime64(3) DEFAULT now64(3)
)
ENGINE = ReplacingMergeTree(inserted_at)
PARTITION BY toYYYYMM(as_of_date)
ORDER BY (fund_id, as_of_date)
SETTINGS index_granularity = 8192;

-- ============================================================================
-- BREACH EVENTS
-- ============================================================================
-- Dedicated table for breach tracking and analysis.
-- ============================================================================
CREATE TABLE IF NOT EXISTS breach_events (
    event_id UUID DEFAULT generateUUIDv4(),
    occurred_at DateTime64(3),
    as_of_date Date,
    control_id LowCardinality(String),
    control_name String,
    control_type LowCardinality(String),
    severity LowCardinality(String),
    calculated_value Decimal(20, 6),
    threshold Decimal(20, 6),
    breach_amount Decimal(20, 6),
    breach_pct Float64,
    fund_id LowCardinality(String) DEFAULT 'MAIN',
    resolved_at Nullable(DateTime64(3)),
    resolution_notes String DEFAULT ''
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(as_of_date)
ORDER BY (fund_id, occurred_at, control_id)
SETTINGS index_granularity = 8192;

-- ============================================================================
-- MATERIALIZED VIEWS
-- ============================================================================
-- Pre-aggregated views for common queries.
-- ============================================================================

-- Daily compliance summary (auto-updated)
CREATE MATERIALIZED VIEW IF NOT EXISTS daily_compliance_summary
ENGINE = SummingMergeTree()
PARTITION BY toYYYYMM(as_of_date)
ORDER BY (fund_id, as_of_date)
AS SELECT
    as_of_date,
    fund_id,
    count() AS total_controls,
    countIf(status = 'pass') AS passed_controls,
    countIf(status = 'warning') AS warning_controls,
    countIf(status = 'fail') AS failed_controls,
    avg(headroom_pct) AS avg_headroom_pct
FROM control_results_history
GROUP BY as_of_date, fund_id;

-- Monthly breach summary
CREATE MATERIALIZED VIEW IF NOT EXISTS monthly_breach_summary
ENGINE = SummingMergeTree()
PARTITION BY toYear(as_of_date)
ORDER BY (fund_id, toStartOfMonth(as_of_date))
AS SELECT
    toStartOfMonth(as_of_date) AS month,
    fund_id,
    count() AS total_checks,
    countIf(status = 'fail') AS total_breaches,
    countIf(status = 'warning') AS total_warnings
FROM control_results_history
GROUP BY month, fund_id;

-- ============================================================================
-- SAMPLE DATA (for testing)
-- ============================================================================
-- Uncomment to insert sample data for testing

-- INSERT INTO control_results_history (as_of_date, control_id, control_name, control_type, calculated_value, threshold, threshold_operator, status, headroom_pct, data_hash)
-- VALUES 
--     (today(), 'CONC_SECTOR_001', 'Sector Concentration', 'concentration', 28.5, 35.0, 'lte', 'pass', 18.57, 'abc123'),
--     (today(), 'EXP_GROSS_001', 'Gross Exposure', 'exposure', 145.0, 200.0, 'lte', 'pass', 27.5, 'def456');
