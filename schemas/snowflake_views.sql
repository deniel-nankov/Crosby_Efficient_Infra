-- =============================================================================
-- COMPLIANCE RAG SYSTEM - SNOWFLAKE VIEWS
-- Read-Only Curated Views for Compliance Controls
-- =============================================================================
-- 
-- Purpose: Provides curated, read-only views of hedge fund data for compliance.
-- These views are designed to be:
--   - Deterministic (point-in-time snapshots)
--   - Secure (column-level masking where needed)
--   - Optimized for compliance control queries
--
-- Access Pattern: The compliance service account has SELECT-only access.
-- All views include snapshot metadata for reproducibility.
-- =============================================================================

-- =============================================================================
-- SNAPSHOT METADATA
-- =============================================================================

-- View to identify available snapshots
CREATE OR REPLACE VIEW compliance.v_data_snapshots AS
SELECT 
    snapshot_id,
    snapshot_date,
    snapshot_timestamp,
    snapshot_type,          -- 'eod', 'intraday', 'adhoc'
    source_systems,
    record_count,
    validation_status,
    created_at
FROM compliance.data_snapshots
WHERE validation_status = 'valid'
ORDER BY snapshot_date DESC, snapshot_timestamp DESC;

-- =============================================================================
-- POSITION DATA
-- =============================================================================

-- Curated positions view for control calculations
CREATE OR REPLACE VIEW compliance.v_positions AS
SELECT 
    p.position_id,
    p.snapshot_id,
    p.snapshot_date,
    p.account_id,
    p.fund_id,
    p.security_id,
    p.security_type,
    p.ticker,
    p.isin,
    p.cusip,
    p.sedol,
    p.security_description,
    
    -- Position quantities
    p.quantity,
    p.quantity_long,
    p.quantity_short,
    
    -- Market values (USD)
    p.market_value_usd,
    p.market_value_local,
    p.currency,
    p.fx_rate_to_usd,
    
    -- Cost basis
    p.cost_basis_usd,
    p.unrealized_pnl_usd,
    
    -- Classification
    p.asset_class,
    p.sector,
    p.country,
    p.region,
    
    -- Issuer for concentration
    p.issuer_id,
    p.issuer_name,
    p.ultimate_parent_issuer_id,
    
    -- Pricing
    p.price,
    p.price_date,
    p.price_source,
    p.is_stale_price,
    
    -- Metadata
    p.created_at AS position_created_at
FROM positions.positions_daily p
WHERE p.is_current = TRUE;

-- Aggregated position summary by fund
CREATE OR REPLACE VIEW compliance.v_fund_position_summary AS
SELECT 
    snapshot_id,
    snapshot_date,
    fund_id,
    COUNT(DISTINCT security_id) AS distinct_securities,
    COUNT(DISTINCT issuer_id) AS distinct_issuers,
    SUM(market_value_usd) AS total_market_value,
    SUM(CASE WHEN quantity > 0 THEN market_value_usd ELSE 0 END) AS long_market_value,
    SUM(CASE WHEN quantity < 0 THEN ABS(market_value_usd) ELSE 0 END) AS short_market_value,
    SUM(unrealized_pnl_usd) AS total_unrealized_pnl,
    SUM(CASE WHEN is_stale_price THEN market_value_usd ELSE 0 END) AS stale_price_exposure
FROM compliance.v_positions
GROUP BY snapshot_id, snapshot_date, fund_id;

-- =============================================================================
-- CONCENTRATION METRICS
-- =============================================================================

-- Issuer concentration for single-name limits
CREATE OR REPLACE VIEW compliance.v_issuer_concentration AS
WITH fund_nav AS (
    SELECT 
        snapshot_id,
        fund_id,
        SUM(market_value_usd) AS fund_nav
    FROM compliance.v_positions
    GROUP BY snapshot_id, fund_id
)
SELECT 
    p.snapshot_id,
    p.snapshot_date,
    p.fund_id,
    p.issuer_id,
    p.issuer_name,
    p.ultimate_parent_issuer_id,
    SUM(p.market_value_usd) AS issuer_exposure_usd,
    fn.fund_nav,
    ROUND(SUM(p.market_value_usd) / NULLIF(fn.fund_nav, 0) * 100, 4) AS issuer_concentration_pct,
    COUNT(DISTINCT p.security_id) AS distinct_securities,
    ARRAY_AGG(DISTINCT p.security_type) AS security_types
FROM compliance.v_positions p
JOIN fund_nav fn ON p.snapshot_id = fn.snapshot_id AND p.fund_id = fn.fund_id
GROUP BY p.snapshot_id, p.snapshot_date, p.fund_id, p.issuer_id, 
         p.issuer_name, p.ultimate_parent_issuer_id, fn.fund_nav;

-- Sector concentration
CREATE OR REPLACE VIEW compliance.v_sector_concentration AS
WITH fund_nav AS (
    SELECT 
        snapshot_id,
        fund_id,
        SUM(market_value_usd) AS fund_nav
    FROM compliance.v_positions
    GROUP BY snapshot_id, fund_id
)
SELECT 
    p.snapshot_id,
    p.snapshot_date,
    p.fund_id,
    p.sector,
    SUM(p.market_value_usd) AS sector_exposure_usd,
    fn.fund_nav,
    ROUND(SUM(p.market_value_usd) / NULLIF(fn.fund_nav, 0) * 100, 4) AS sector_concentration_pct,
    COUNT(DISTINCT p.security_id) AS distinct_securities,
    COUNT(DISTINCT p.issuer_id) AS distinct_issuers
FROM compliance.v_positions p
JOIN fund_nav fn ON p.snapshot_id = fn.snapshot_id AND p.fund_id = fn.fund_id
GROUP BY p.snapshot_id, p.snapshot_date, p.fund_id, p.sector, fn.fund_nav;

-- Country concentration
CREATE OR REPLACE VIEW compliance.v_country_concentration AS
WITH fund_nav AS (
    SELECT 
        snapshot_id,
        fund_id,
        SUM(market_value_usd) AS fund_nav
    FROM compliance.v_positions
    GROUP BY snapshot_id, fund_id
)
SELECT 
    p.snapshot_id,
    p.snapshot_date,
    p.fund_id,
    p.country,
    p.region,
    SUM(p.market_value_usd) AS country_exposure_usd,
    fn.fund_nav,
    ROUND(SUM(p.market_value_usd) / NULLIF(fn.fund_nav, 0) * 100, 4) AS country_concentration_pct
FROM compliance.v_positions p
JOIN fund_nav fn ON p.snapshot_id = fn.snapshot_id AND p.fund_id = fn.fund_id
GROUP BY p.snapshot_id, p.snapshot_date, p.fund_id, p.country, p.region, fn.fund_nav;

-- =============================================================================
-- COUNTERPARTY EXPOSURE
-- =============================================================================

-- Counterparty exposure summary
CREATE OR REPLACE VIEW compliance.v_counterparty_exposure AS
SELECT 
    ce.snapshot_id,
    ce.snapshot_date,
    ce.fund_id,
    ce.counterparty_id,
    cp.counterparty_name,
    cp.counterparty_type,       -- 'prime_broker', 'otc_dealer', 'clearing_house', 'custodian'
    cp.credit_rating,
    cp.credit_rating_agency,
    
    -- Exposure components
    ce.cash_balance,
    ce.margin_balance,
    ce.securities_lending_exposure,
    ce.swap_mtm_exposure,
    ce.repo_exposure,
    ce.futures_margin,
    ce.options_margin,
    
    -- Aggregates
    ce.gross_exposure,
    ce.net_exposure,
    
    -- Collateral
    ce.collateral_posted,
    ce.collateral_received,
    ce.net_collateral,
    
    -- Risk metrics
    ce.potential_future_exposure,
    ce.exposure_at_default,
    
    -- Limits
    ce.internal_limit,
    ROUND(ce.net_exposure / NULLIF(ce.internal_limit, 0) * 100, 4) AS limit_utilization_pct
FROM counterparty.counterparty_exposure_daily ce
JOIN counterparty.counterparties cp ON ce.counterparty_id = cp.counterparty_id
WHERE ce.is_current = TRUE;

-- =============================================================================
-- LIQUIDITY METRICS
-- =============================================================================

-- Liquidity buckets for Form PF
CREATE OR REPLACE VIEW compliance.v_liquidity_buckets AS
SELECT 
    l.snapshot_id,
    l.snapshot_date,
    l.fund_id,
    l.liquidity_bucket,         -- '1d', '2-7d', '8-30d', '31-90d', '91-180d', '181-365d', '>365d'
    l.bucket_order,
    
    -- Long positions
    l.long_market_value,
    l.long_pct_nav,
    
    -- Short positions  
    l.short_market_value,
    l.short_pct_nav,
    
    -- Combined
    l.net_market_value,
    l.gross_market_value,
    
    -- Security counts
    l.position_count,
    l.distinct_securities
FROM liquidity.liquidity_profile_daily l
WHERE l.is_current = TRUE
ORDER BY l.fund_id, l.bucket_order;

-- Cumulative liquidity (what % can be liquidated in X days)
CREATE OR REPLACE VIEW compliance.v_cumulative_liquidity AS
SELECT 
    snapshot_id,
    snapshot_date,
    fund_id,
    liquidity_bucket,
    bucket_order,
    long_pct_nav,
    SUM(long_pct_nav) OVER (
        PARTITION BY snapshot_id, fund_id 
        ORDER BY bucket_order
    ) AS cumulative_long_pct,
    gross_market_value,
    SUM(gross_market_value) OVER (
        PARTITION BY snapshot_id, fund_id 
        ORDER BY bucket_order
    ) AS cumulative_gross_value
FROM compliance.v_liquidity_buckets;

-- =============================================================================
-- LEVERAGE METRICS
-- =============================================================================

-- Leverage calculations for Form PF
CREATE OR REPLACE VIEW compliance.v_leverage_metrics AS
SELECT 
    lm.snapshot_id,
    lm.snapshot_date,
    lm.fund_id,
    
    -- NAV
    lm.fund_nav,
    
    -- Gross exposure
    lm.gross_long_exposure,
    lm.gross_short_exposure,
    lm.gross_notional,
    
    -- Leverage ratios
    ROUND(lm.gross_notional / NULLIF(lm.fund_nav, 0), 4) AS gross_leverage_ratio,
    ROUND((lm.gross_long_exposure + lm.gross_short_exposure) / NULLIF(lm.fund_nav, 0), 4) AS long_short_leverage,
    ROUND(lm.net_exposure / NULLIF(lm.fund_nav, 0), 4) AS net_leverage_ratio,
    
    -- Borrowing
    lm.total_borrowing,
    ROUND(lm.total_borrowing / NULLIF(lm.fund_nav, 0), 4) AS borrowing_to_nav_ratio,
    
    -- Derivatives notional
    lm.derivatives_gross_notional,
    ROUND(lm.derivatives_gross_notional / NULLIF(lm.fund_nav, 0), 4) AS derivatives_leverage,
    
    -- SEC method (for Form PF)
    lm.gross_asset_value_sec_method,
    ROUND(lm.gross_asset_value_sec_method / NULLIF(lm.fund_nav, 0), 4) AS gav_to_nav_ratio
    
FROM leverage.leverage_metrics_daily lm
WHERE lm.is_current = TRUE;

-- =============================================================================
-- PNL DATA
-- =============================================================================

-- Daily PnL by fund
CREATE OR REPLACE VIEW compliance.v_daily_pnl AS
SELECT 
    pnl.snapshot_id,
    pnl.snapshot_date,
    pnl.fund_id,
    
    -- PnL components
    pnl.realized_pnl,
    pnl.unrealized_pnl,
    pnl.total_pnl,
    
    -- Attribution
    pnl.pnl_from_equities,
    pnl.pnl_from_fixed_income,
    pnl.pnl_from_derivatives,
    pnl.pnl_from_fx,
    pnl.pnl_from_other,
    
    -- Performance
    pnl.daily_return_pct,
    pnl.mtd_return_pct,
    pnl.ytd_return_pct,
    
    -- Benchmarks
    pnl.benchmark_return_pct,
    pnl.excess_return_pct
FROM pnl.pnl_daily pnl
WHERE pnl.is_current = TRUE;

-- =============================================================================
-- 13F HOLDINGS
-- =============================================================================

-- 13F reportable positions
CREATE OR REPLACE VIEW compliance.v_13f_holdings AS
SELECT 
    p.snapshot_id,
    p.snapshot_date,
    p.fund_id,
    p.security_id,
    p.cusip,
    p.issuer_name,
    p.security_description,
    p.security_type,
    p.ticker,
    
    -- Share information
    p.quantity AS shares_principal_amount,
    CASE WHEN p.security_type LIKE '%Option%' THEN 'SH' ELSE 'SH' END AS shares_type,
    
    -- Value
    p.market_value_usd AS value,
    
    -- Voting authority (simplified - would come from separate table)
    p.quantity AS sole_voting_authority,
    0 AS shared_voting_authority,
    0 AS no_voting_authority,
    
    -- Investment discretion
    'SOLE' AS investment_discretion,
    
    -- Manager info
    'Investment Manager' AS other_manager,
    
    -- 13F list check
    s.is_13f_security,
    s.effective_date AS thirteen_f_list_date
    
FROM compliance.v_positions p
LEFT JOIN reference.securities_13f_list s ON p.cusip = s.cusip
WHERE p.quantity > 0  -- Long positions only
  AND p.security_type IN ('Common Stock', 'ETF', 'Preferred Stock', 'Convertible Bond', 'Warrant', 'Option', 'ADR');

-- =============================================================================
-- FORM PF SPECIFIC VIEWS
-- =============================================================================

-- Form PF Question 22 - Geographic exposure
CREATE OR REPLACE VIEW compliance.v_form_pf_geographic AS
SELECT 
    snapshot_id,
    snapshot_date,
    fund_id,
    country,
    region,
    SUM(market_value_usd) AS exposure_usd,
    ROUND(SUM(market_value_usd) / NULLIF(SUM(SUM(market_value_usd)) OVER (PARTITION BY snapshot_id, fund_id), 0) * 100, 2) AS pct_nav
FROM compliance.v_positions
GROUP BY snapshot_id, snapshot_date, fund_id, country, region;

-- Form PF Question 29 - Trading and clearing
CREATE OR REPLACE VIEW compliance.v_form_pf_counterparty_credit AS
SELECT 
    ce.snapshot_id,
    ce.snapshot_date,
    ce.fund_id,
    ce.counterparty_name,
    ce.counterparty_type,
    ce.credit_rating,
    ce.net_exposure,
    ce.collateral_posted,
    ce.net_exposure - ce.collateral_posted AS uncollateralized_exposure
FROM compliance.v_counterparty_exposure ce
WHERE ce.net_exposure > 0
ORDER BY ce.net_exposure DESC;

-- =============================================================================
-- DATA QUALITY VIEWS
-- =============================================================================

-- Stale price report
CREATE OR REPLACE VIEW compliance.v_stale_prices AS
SELECT 
    snapshot_id,
    snapshot_date,
    fund_id,
    security_id,
    ticker,
    security_description,
    market_value_usd,
    price,
    price_date,
    price_source,
    DATEDIFF('day', price_date, snapshot_date) AS days_stale
FROM compliance.v_positions
WHERE is_stale_price = TRUE
ORDER BY market_value_usd DESC;

-- Missing reference data
CREATE OR REPLACE VIEW compliance.v_missing_reference_data AS
SELECT 
    snapshot_id,
    snapshot_date,
    security_id,
    ticker,
    security_description,
    CASE WHEN isin IS NULL THEN 'ISIN' END AS missing_isin,
    CASE WHEN cusip IS NULL THEN 'CUSIP' END AS missing_cusip,
    CASE WHEN issuer_id IS NULL THEN 'ISSUER' END AS missing_issuer,
    CASE WHEN sector IS NULL THEN 'SECTOR' END AS missing_sector,
    CASE WHEN country IS NULL THEN 'COUNTRY' END AS missing_country
FROM compliance.v_positions
WHERE isin IS NULL OR cusip IS NULL OR issuer_id IS NULL OR sector IS NULL OR country IS NULL;

-- =============================================================================
-- HISTORICAL COMPARISON
-- =============================================================================

-- Day-over-day position changes
CREATE OR REPLACE VIEW compliance.v_position_changes AS
WITH today AS (
    SELECT * FROM compliance.v_positions WHERE snapshot_date = CURRENT_DATE
),
yesterday AS (
    SELECT * FROM compliance.v_positions WHERE snapshot_date = CURRENT_DATE - 1
)
SELECT 
    COALESCE(t.snapshot_date, y.snapshot_date + 1) AS snapshot_date,
    COALESCE(t.fund_id, y.fund_id) AS fund_id,
    COALESCE(t.security_id, y.security_id) AS security_id,
    COALESCE(t.ticker, y.ticker) AS ticker,
    y.quantity AS prev_quantity,
    t.quantity AS curr_quantity,
    COALESCE(t.quantity, 0) - COALESCE(y.quantity, 0) AS quantity_change,
    y.market_value_usd AS prev_market_value,
    t.market_value_usd AS curr_market_value,
    COALESCE(t.market_value_usd, 0) - COALESCE(y.market_value_usd, 0) AS market_value_change,
    CASE 
        WHEN y.security_id IS NULL THEN 'NEW'
        WHEN t.security_id IS NULL THEN 'CLOSED'
        WHEN t.quantity != y.quantity THEN 'CHANGED'
        ELSE 'UNCHANGED'
    END AS change_type
FROM today t
FULL OUTER JOIN yesterday y ON t.fund_id = y.fund_id AND t.security_id = y.security_id
WHERE COALESCE(t.quantity, 0) != COALESCE(y.quantity, 0)
   OR t.security_id IS NULL 
   OR y.security_id IS NULL;

-- =============================================================================
-- GRANT STATEMENTS (Execute with elevated privileges)
-- =============================================================================

/*
-- Create read-only role for compliance service
CREATE ROLE compliance_reader;

-- Grant usage on schema
GRANT USAGE ON SCHEMA compliance TO ROLE compliance_reader;

-- Grant select on all views
GRANT SELECT ON ALL VIEWS IN SCHEMA compliance TO ROLE compliance_reader;

-- Grant to service account
GRANT ROLE compliance_reader TO USER compliance_service_account;

-- Revoke any write permissions (defense in depth)
REVOKE INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA compliance FROM ROLE compliance_reader;
REVOKE CREATE TABLE ON SCHEMA compliance FROM ROLE compliance_reader;
*/
