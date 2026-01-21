-- =============================================================================
-- SNOWFLAKE COMPLIANCE VIEWS FOR RAG SYSTEM
-- =============================================================================
-- Execute this script in your Snowflake environment to create the required
-- views for the Compliance RAG System integration.
--
-- Prerequisites:
--   1. Source tables with positions, prices, and control calculations
--   2. Service account with appropriate permissions
--   3. Dedicated warehouse for compliance queries (recommended)
--
-- Customize the source table references to match your schema.
-- =============================================================================

-- Use your target database
USE DATABASE HEDGE_FUND_DATA;  -- Change to your database

-- Create compliance schema if not exists
CREATE SCHEMA IF NOT EXISTS COMPLIANCE;
USE SCHEMA COMPLIANCE;

-- =============================================================================
-- POSITIONS VIEW
-- =============================================================================
-- Aggregates current position data from your source systems
-- Joins with securities master and pricing data
--
-- Required output columns (adapter expects these):
--   SECURITY_ID, TICKER, SECURITY_NAME, QUANTITY, MARKET_VALUE_USD,
--   CURRENCY, GICS_SECTOR, ISSUER_NAME, ASSET_CLASS, ISIN, CUSIP,
--   PRICE_LOCAL, AS_OF_DATE
-- =============================================================================

CREATE OR REPLACE SECURE VIEW V_POSITIONS_CURRENT AS
SELECT
    -- Position identifiers
    p.SECURITY_ID,
    s.TICKER,
    s.SECURITY_NAME,
    
    -- Quantities and values
    p.QUANTITY,
    ROUND(p.QUANTITY * pr.PRICE_USD, 2) AS MARKET_VALUE_USD,
    
    -- Security attributes
    s.CURRENCY,
    s.GICS_SECTOR,
    s.ISSUER_NAME,
    s.ASSET_CLASS,
    s.ISIN,
    s.CUSIP,
    
    -- Pricing
    pr.PRICE_USD AS PRICE_LOCAL,
    
    -- Date
    p.AS_OF_DATE

FROM RAW.POSITIONS p  -- Change to your positions table
JOIN RAW.SECURITIES s ON p.SECURITY_ID = s.SECURITY_ID  -- Your securities master
JOIN RAW.PRICES pr 
    ON p.SECURITY_ID = pr.SECURITY_ID 
    AND p.AS_OF_DATE = pr.PRICE_DATE  -- Your pricing table

WHERE 
    p.AS_OF_DATE >= DATEADD(day, -30, CURRENT_DATE())
    AND p.FUND_ID = 'MAIN_FUND'  -- Filter by fund if multi-fund
;

COMMENT ON VIEW V_POSITIONS_CURRENT IS 
    'Current positions for compliance RAG system. Read-only view of reconciled position data.';


-- =============================================================================
-- CONTROL RESULTS VIEW
-- =============================================================================
-- Pre-calculated compliance control results from your existing system
-- The RAG system trusts these calculations and generates narratives
--
-- Required output columns:
--   CONTROL_ID, CONTROL_NAME, CONTROL_TYPE, CALCULATED_VALUE,
--   THRESHOLD_VALUE, THRESHOLD_OPERATOR, STATUS, BREACH_AMOUNT,
--   DETAILS_JSON, AS_OF_DATE
-- =============================================================================

CREATE OR REPLACE SECURE VIEW V_CONTROL_RESULTS AS
SELECT
    -- Control identification
    c.CONTROL_ID,
    c.CONTROL_NAME,
    c.CONTROL_TYPE,  -- 'concentration', 'exposure', 'liquidity', etc.
    
    -- Calculation results
    ROUND(c.CALCULATED_VALUE, 6) AS CALCULATED_VALUE,
    ROUND(c.THRESHOLD_VALUE, 6) AS THRESHOLD_VALUE,
    c.THRESHOLD_OPERATOR,  -- 'lte', 'gte', 'lt', 'gt', 'eq'
    
    -- Status (normalized)
    CASE 
        WHEN c.IS_BREACH = TRUE THEN 'fail'
        WHEN c.IS_WARNING = TRUE THEN 'warning'
        ELSE 'pass'
    END AS STATUS,
    
    -- Breach details
    CASE 
        WHEN c.IS_BREACH THEN ROUND(c.CALCULATED_VALUE - c.THRESHOLD_VALUE, 6)
        ELSE NULL 
    END AS BREACH_AMOUNT,
    
    -- Additional context as JSON
    c.DETAILS_JSON,
    
    -- Date
    c.AS_OF_DATE

FROM CURATED.CONTROL_CALCULATIONS c  -- Change to your control results table

WHERE 
    c.AS_OF_DATE >= DATEADD(day, -30, CURRENT_DATE())
    AND c.FUND_ID = 'MAIN_FUND'  -- Filter by fund if multi-fund

ORDER BY
    CASE STATUS 
        WHEN 'fail' THEN 0 
        WHEN 'warning' THEN 1 
        ELSE 2 
    END,
    c.CONTROL_ID
;

COMMENT ON VIEW V_CONTROL_RESULTS IS 
    'Daily compliance control results. Pre-calculated by upstream compliance system.';


-- =============================================================================
-- NAV VIEW
-- =============================================================================
-- Daily Net Asset Value for the fund
--
-- Required output columns:
--   NAV_USD, AS_OF_DATE
-- =============================================================================

CREATE OR REPLACE SECURE VIEW V_NAV_DAILY AS
SELECT
    ROUND(n.NAV_USD, 2) AS NAV_USD,
    n.AS_OF_DATE

FROM CURATED.FUND_NAV n  -- Change to your NAV table

WHERE 
    n.AS_OF_DATE >= DATEADD(day, -30, CURRENT_DATE())
    AND n.FUND_ID = 'MAIN_FUND'  -- Filter by fund if multi-fund
;

COMMENT ON VIEW V_NAV_DAILY IS 
    'Daily NAV values for compliance calculations.';


-- =============================================================================
-- OPTIONAL: HISTORICAL POSITIONS VIEW
-- =============================================================================
-- For trend analysis and historical compliance review

CREATE OR REPLACE SECURE VIEW V_POSITIONS_HISTORICAL AS
SELECT
    p.SECURITY_ID,
    s.TICKER,
    s.SECURITY_NAME,
    p.QUANTITY,
    ROUND(p.QUANTITY * pr.PRICE_USD, 2) AS MARKET_VALUE_USD,
    s.GICS_SECTOR,
    s.ASSET_CLASS,
    p.AS_OF_DATE

FROM RAW.POSITIONS p
JOIN RAW.SECURITIES s ON p.SECURITY_ID = s.SECURITY_ID
JOIN RAW.PRICES pr 
    ON p.SECURITY_ID = pr.SECURITY_ID 
    AND p.AS_OF_DATE = pr.PRICE_DATE

WHERE 
    p.AS_OF_DATE >= DATEADD(day, -365, CURRENT_DATE())
;


-- =============================================================================
-- OPTIONAL: CONTROL TRENDS VIEW
-- =============================================================================
-- For tracking control metrics over time

CREATE OR REPLACE SECURE VIEW V_CONTROL_TRENDS AS
SELECT
    c.CONTROL_ID,
    c.CONTROL_NAME,
    c.CONTROL_TYPE,
    c.CALCULATED_VALUE,
    c.THRESHOLD_VALUE,
    c.AS_OF_DATE,
    
    -- 30-day average
    AVG(c.CALCULATED_VALUE) OVER (
        PARTITION BY c.CONTROL_ID 
        ORDER BY c.AS_OF_DATE 
        ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
    ) AS ROLLING_30D_AVG,
    
    -- Distance from threshold as percentage
    ROUND(
        (c.THRESHOLD_VALUE - c.CALCULATED_VALUE) / NULLIF(c.THRESHOLD_VALUE, 0) * 100, 
        2
    ) AS HEADROOM_PCT

FROM CURATED.CONTROL_CALCULATIONS c
WHERE c.AS_OF_DATE >= DATEADD(day, -90, CURRENT_DATE())
;


-- =============================================================================
-- SECURITY CONFIGURATION
-- =============================================================================
-- Create role for compliance service account

CREATE ROLE IF NOT EXISTS COMPLIANCE_READER;

-- Grant usage on objects
GRANT USAGE ON DATABASE HEDGE_FUND_DATA TO ROLE COMPLIANCE_READER;
GRANT USAGE ON SCHEMA COMPLIANCE TO ROLE COMPLIANCE_READER;
GRANT USAGE ON WAREHOUSE COMPLIANCE_WH TO ROLE COMPLIANCE_READER;  -- Your warehouse

-- Grant SELECT on views
GRANT SELECT ON ALL VIEWS IN SCHEMA COMPLIANCE TO ROLE COMPLIANCE_READER;
GRANT SELECT ON FUTURE VIEWS IN SCHEMA COMPLIANCE TO ROLE COMPLIANCE_READER;

-- Create service account (if not exists)
-- CREATE USER IF NOT EXISTS COMPLIANCE_SVC
--     PASSWORD = 'your_secure_password'  -- Use key-pair in production
--     DEFAULT_ROLE = COMPLIANCE_READER
--     DEFAULT_WAREHOUSE = COMPLIANCE_WH
--     MUST_CHANGE_PASSWORD = FALSE;

-- GRANT ROLE COMPLIANCE_READER TO USER COMPLIANCE_SVC;


-- =============================================================================
-- VALIDATION QUERIES
-- =============================================================================
-- Run these to verify your views are working correctly

-- Check positions view
SELECT 'V_POSITIONS_CURRENT' AS VIEW_NAME, COUNT(*) AS ROW_COUNT 
FROM V_POSITIONS_CURRENT WHERE AS_OF_DATE = CURRENT_DATE();

-- Check control results view
SELECT 'V_CONTROL_RESULTS' AS VIEW_NAME, COUNT(*) AS ROW_COUNT 
FROM V_CONTROL_RESULTS WHERE AS_OF_DATE = CURRENT_DATE();

-- Check NAV view
SELECT 'V_NAV_DAILY' AS VIEW_NAME, NAV_USD 
FROM V_NAV_DAILY WHERE AS_OF_DATE = CURRENT_DATE();

-- Check for any failed controls
SELECT CONTROL_ID, CONTROL_NAME, CALCULATED_VALUE, THRESHOLD_VALUE
FROM V_CONTROL_RESULTS 
WHERE STATUS = 'fail' AND AS_OF_DATE = CURRENT_DATE();
