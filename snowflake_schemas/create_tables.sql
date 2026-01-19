-- =============================================================================
-- SNOWFLAKE TABLE SCHEMAS
-- =============================================================================
-- 
-- Create these tables in your Snowflake database.
-- Your existing systems (Bloomberg/Eze/Geneva/Administrator) should populate them.
--
-- Usage:
--   1. Create these tables in Snowflake
--   2. Set up data feeds from your systems
--   3. The RAG system will read from these tables daily
--
-- =============================================================================

-- =============================================================================
-- DATABASE & SCHEMA
-- =============================================================================

CREATE DATABASE IF NOT EXISTS COMPLIANCE;
USE DATABASE COMPLIANCE;
CREATE SCHEMA IF NOT EXISTS PUBLIC;
USE SCHEMA PUBLIC;

-- =============================================================================
-- TABLE 1: POSITIONS
-- =============================================================================
-- Source: Your Portfolio Management System (Bloomberg AIM, Eze, Geneva, etc.)
-- Frequency: Daily (end of day)

CREATE TABLE IF NOT EXISTS POSITIONS (
    -- Primary identifiers
    SECURITY_ID         VARCHAR(50) NOT NULL,       -- Your internal ID
    TICKER              VARCHAR(20),                 -- Bloomberg ticker
    SEDOL               VARCHAR(7),                  -- SEDOL
    ISIN                VARCHAR(12),                 -- ISIN
    CUSIP               VARCHAR(9),                  -- CUSIP
    
    -- Security info
    SECURITY_NAME       VARCHAR(500) NOT NULL,
    ASSET_CLASS         VARCHAR(50),                 -- equity, fixed_income, derivative, etc.
    SECTOR              VARCHAR(100),                -- GICS sector
    INDUSTRY            VARCHAR(100),                -- GICS industry
    COUNTRY             VARCHAR(50),                 -- Country of risk
    ISSUER              VARCHAR(200),                -- Issuer/parent company
    
    -- Position data
    QUANTITY            NUMBER(20,4) NOT NULL,       -- Shares/units
    MARKET_VALUE        NUMBER(20,2) NOT NULL,       -- In base currency
    COST_BASIS          NUMBER(20,2),                -- Total cost
    CURRENCY            VARCHAR(3) DEFAULT 'USD',
    
    -- Classification
    LONG_SHORT          VARCHAR(10),                 -- LONG or SHORT
    STRATEGY            VARCHAR(100),                -- If multi-strategy fund
    
    -- Metadata
    AS_OF_DATE          DATE NOT NULL,
    SOURCE_SYSTEM       VARCHAR(100),                -- Bloomberg, Eze, etc.
    LOAD_TIMESTAMP      TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    
    PRIMARY KEY (SECURITY_ID, AS_OF_DATE)
);

-- Example data:
-- INSERT INTO POSITIONS VALUES 
-- ('SEC001', 'AAPL', 'B0YQ5W0', 'US0378331005', '037833100', 
--  'Apple Inc', 'equity', 'Technology', 'Technology Hardware', 'US', 'Apple Inc',
--  100000, 15000000.00, 12000000.00, 'USD', 'LONG', 'Core Equity',
--  '2026-01-15', 'Bloomberg AIM', CURRENT_TIMESTAMP());

-- =============================================================================
-- TABLE 2: CONTROL_RESULTS
-- =============================================================================
-- Source: Your Compliance System (pre-calculated by your existing tools)
-- Frequency: Daily (after positions are loaded)
--
-- IMPORTANT: The RAG system does NOT calculate these values.
-- Your existing compliance system calculates them; we just read and narrate.

CREATE TABLE IF NOT EXISTS CONTROL_RESULTS (
    -- Control identification
    CONTROL_ID          VARCHAR(100) NOT NULL,       -- e.g., CONC_ISSUER_001
    CONTROL_NAME        VARCHAR(500) NOT NULL,       -- Human-readable name
    CONTROL_TYPE        VARCHAR(50) NOT NULL,        -- concentration, liquidity, exposure, etc.
    CONTROL_CATEGORY    VARCHAR(50),                 -- regulatory, internal, client
    
    -- The calculation (already done by your system)
    CALCULATED_VALUE    NUMBER(10,4) NOT NULL,       -- e.g., 8.5 (meaning 8.5%)
    THRESHOLD           NUMBER(10,4) NOT NULL,       -- e.g., 10.0 (meaning 10%)
    THRESHOLD_OPERATOR  VARCHAR(10) NOT NULL,        -- gt, gte, lt, lte, eq
    WARNING_THRESHOLD   NUMBER(10,4),                -- Optional early warning
    
    -- Result
    STATUS              VARCHAR(20) NOT NULL,        -- pass, warning, fail
    BREACH_AMOUNT       NUMBER(10,4),                -- How much over/under (if breached)
    
    -- Context (optional, for narrative generation)
    DETAILS             VARIANT,                     -- JSON with extra context
    RELATED_SECURITIES  ARRAY,                       -- Securities involved
    POLICY_REFERENCE    VARCHAR(200),                -- Which policy this relates to
    
    -- Metadata
    AS_OF_DATE          DATE NOT NULL,
    EVALUATED_AT        TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    SOURCE_SYSTEM       VARCHAR(100),
    
    PRIMARY KEY (CONTROL_ID, AS_OF_DATE)
);

-- Example data:
-- INSERT INTO CONTROL_RESULTS VALUES
-- ('CONC_ISSUER_001', 'Single Issuer Concentration', 'concentration', 'regulatory',
--  8.5, 10.0, 'lte', 8.0,
--  'pass', NULL,
--  PARSE_JSON('{"top_issuer": "Apple Inc", "position_count": 3}'),
--  ARRAY_CONSTRUCT('AAPL', 'AAPL.OQ'),
--  'investment_guidelines:section_2.3',
--  '2026-01-15', CURRENT_TIMESTAMP(), 'Internal Compliance System');

-- INSERT INTO CONTROL_RESULTS VALUES
-- ('CONC_SECTOR_001', 'Sector Concentration - Technology', 'concentration', 'internal',
--  32.5, 30.0, 'lte', 25.0,
--  'fail', 2.5,
--  PARSE_JSON('{"sector": "Technology", "holdings_count": 15}'),
--  ARRAY_CONSTRUCT('AAPL', 'MSFT', 'NVDA', 'GOOGL'),
--  'investment_guidelines:section_2.1',
--  '2026-01-15', CURRENT_TIMESTAMP(), 'Internal Compliance System');

-- =============================================================================
-- TABLE 3: FUND_NAV
-- =============================================================================
-- Source: Fund Administrator
-- Frequency: Daily

CREATE TABLE IF NOT EXISTS FUND_NAV (
    FUND_ID             VARCHAR(50) DEFAULT 'MAIN',
    NAV                 NUMBER(20,2) NOT NULL,       -- Total NAV
    NAV_PER_SHARE       NUMBER(20,6),                -- NAV per share/unit
    SHARES_OUTSTANDING  NUMBER(20,4),
    
    -- Breakdown (optional)
    CASH                NUMBER(20,2),
    LONG_MARKET_VALUE   NUMBER(20,2),
    SHORT_MARKET_VALUE  NUMBER(20,2),
    
    -- Metadata
    AS_OF_DATE          DATE NOT NULL,
    SOURCE              VARCHAR(100),                -- Administrator name
    LOAD_TIMESTAMP      TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),
    
    PRIMARY KEY (FUND_ID, AS_OF_DATE)
);

-- Example data:
-- INSERT INTO FUND_NAV VALUES
-- ('MAIN', 2000000000.00, 125.50, 15936254.98,
--  64000000.00, 2100000000.00, -164000000.00,
--  '2026-01-15', 'Northern Trust', CURRENT_TIMESTAMP());

-- =============================================================================
-- VIEWS (Optional but helpful)
-- =============================================================================

CREATE OR REPLACE VIEW V_LATEST_POSITIONS AS
SELECT * FROM POSITIONS
WHERE AS_OF_DATE = (SELECT MAX(AS_OF_DATE) FROM POSITIONS);

CREATE OR REPLACE VIEW V_LATEST_CONTROLS AS
SELECT * FROM CONTROL_RESULTS
WHERE AS_OF_DATE = (SELECT MAX(AS_OF_DATE) FROM CONTROL_RESULTS);

CREATE OR REPLACE VIEW V_BREACHES AS
SELECT * FROM CONTROL_RESULTS
WHERE STATUS IN ('warning', 'fail')
ORDER BY AS_OF_DATE DESC, STATUS;

-- =============================================================================
-- GRANTS
-- =============================================================================
-- Grant read access to the compliance service account

-- CREATE USER IF NOT EXISTS COMPLIANCE_SERVICE PASSWORD = 'your_secure_password';
-- GRANT USAGE ON DATABASE COMPLIANCE TO ROLE COMPLIANCE_READER;
-- GRANT USAGE ON SCHEMA PUBLIC TO ROLE COMPLIANCE_READER;
-- GRANT SELECT ON ALL TABLES IN SCHEMA PUBLIC TO ROLE COMPLIANCE_READER;
-- GRANT ROLE COMPLIANCE_READER TO USER COMPLIANCE_SERVICE;
