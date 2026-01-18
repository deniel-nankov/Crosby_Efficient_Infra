# Commodity Trading Policy

## Document ID: POL-COMM-001
## Effective Date: January 1, 2024
## Last Reviewed: January 1, 2026
## Owner: Chief Investment Officer

---

## 1. Purpose

This policy establishes the framework for commodity trading activities at Master Fund LP. As a commodity-focused hedge fund, we maintain significant exposure to energy, metals, and agricultural commodities through futures contracts, options, and related equities.

## 2. Scope

This policy applies to:
- Exchange-traded commodity futures and options
- Commodity-linked equities (producers, refiners, miners)
- Commodity ETFs and ETNs
- OTC commodity derivatives (swaps, forwards)
- Physical commodity positions (if permitted)

## 3. Permitted Commodity Markets

### 3.1 Energy Commodities

| Commodity | Exchange | Contract Symbol | Contract Size |
|-----------|----------|-----------------|---------------|
| WTI Crude Oil | NYMEX/CME | CL | 1,000 barrels |
| Brent Crude Oil | ICE | BZ | 1,000 barrels |
| Natural Gas (Henry Hub) | NYMEX/CME | NG | 10,000 MMBtu |
| RBOB Gasoline | NYMEX/CME | RB | 42,000 gallons |
| Heating Oil | NYMEX/CME | HO | 42,000 gallons |

**Data Source**: Prices are validated against EIA (Energy Information Administration) daily spot prices.

### 3.2 Precious Metals

| Commodity | Exchange | Contract Symbol | Contract Size |
|-----------|----------|-----------------|---------------|
| Gold | COMEX/CME | GC | 100 troy oz |
| Silver | COMEX/CME | SI | 5,000 troy oz |
| Platinum | NYMEX/CME | PL | 50 troy oz |
| Palladium | NYMEX/CME | PA | 100 troy oz |

**Data Source**: Prices are validated against LBMA (London Bullion Market Association) and World Bank Commodity Markets.

### 3.3 Industrial Metals

| Commodity | Exchange | Contract Symbol | Contract Size |
|-----------|----------|-----------------|---------------|
| Copper | COMEX/CME | HG | 25,000 lbs |
| Aluminum | LME | AL | 25 metric tons |
| Zinc | LME | ZN | 25 metric tons |
| Nickel | LME | NI | 6 metric tons |

### 3.4 Agriculture

| Commodity | Exchange | Contract Symbol | Contract Size |
|-----------|----------|-----------------|---------------|
| Corn | CBOT/CME | ZC | 5,000 bushels |
| Wheat | CBOT/CME | ZW | 5,000 bushels |
| Soybeans | CBOT/CME | ZS | 5,000 bushels |
| Sugar #11 | ICE | SB | 112,000 lbs |
| Coffee | ICE | KC | 37,500 lbs |
| Cotton | ICE | CT | 50,000 lbs |

**Data Source**: Position data validated against CFTC Commitments of Traders (COT) reports.

## 4. Position Limits

### 4.1 Single Commodity Limits

| Category | Maximum Long | Maximum Short |
|----------|-------------|---------------|
| Single energy commodity | 15% of NAV | 10% of NAV |
| Single metal | 10% of NAV | 7% of NAV |
| Single agricultural | 8% of NAV | 5% of NAV |

### 4.2 Sector Limits

| Sector | Maximum Gross | Maximum Net |
|--------|---------------|-------------|
| Energy (crude, products, nat gas) | 40% of NAV | 30% of NAV |
| Precious Metals | 25% of NAV | 20% of NAV |
| Industrial Metals | 20% of NAV | 15% of NAV |
| Agriculture | 20% of NAV | 15% of NAV |

### 4.3 CFTC Position Limits

The Fund must comply with CFTC speculative position limits:

| Commodity | Spot Month Limit | All Months Limit |
|-----------|-----------------|------------------|
| WTI Crude Oil | 6,000 contracts | 30,000 contracts |
| Brent Crude Oil | 6,000 contracts | 10,000 contracts |
| Natural Gas | 2,000 contracts | 20,000 contracts |
| Gold | 6,000 contracts | 20,000 contracts |
| Corn | 1,200 contracts | 57,800 contracts |

**Reference**: CFTC Part 150 - Speculative Position Limits

### 4.4 Aggregation Rules

Per CFTC rules, positions must be aggregated with:
- Accounts under common ownership or control
- Positions held by affiliates
- Positions in economically equivalent contracts

## 5. Rolling Procedures

### 5.1 Roll Schedule

Futures positions must be rolled before the first notice date to avoid physical delivery risk.

| Action | Timing |
|--------|--------|
| Initiate roll | 10 business days before first notice |
| Complete roll | 5 business days before first notice |
| Emergency roll | 2 business days before first notice (CIO approval) |

### 5.2 Roll Cost Budget

- Maximum roll cost: 0.50% of position notional per roll
- Preferred roll timing: During liquid roll window (typically 5-7 days before expiry)
- Roll execution: TWAP or VWAP algorithms preferred

### 5.3 Roll Documentation

Each roll must be documented with:
- Front month contract
- Back month contract
- Roll spread achieved
- Market roll spread (for comparison)
- Deviation explanation (if >5% from market)

## 6. Physical Delivery Prohibition

### 6.1 No Physical Delivery

**The Fund is prohibited from taking or making physical delivery of any commodity.**

This includes:
- Energy products (no crude oil, gasoline, heating oil delivery)
- Metals (no gold, silver, copper delivery)
- Agricultural products (no grain, livestock delivery)

### 6.2 Delivery Avoidance Procedures

1. All positions must be rolled or closed before first notice date
2. Automated alerts at T-15, T-10, T-5, T-2 business days
3. CIO approval required for any position held within T-5 of first notice
4. Emergency liquidation authority granted to Operations at T-2

### 6.3 Exception: Cash-Settled Contracts

Cash-settled contracts (e.g., certain index futures) may be held through settlement.

## 7. Risk Management

### 7.1 Value-at-Risk Limits

| Metric | Limit |
|--------|-------|
| Portfolio VaR (99%, 1-day) | 3% of NAV |
| Commodity-only VaR | 2% of NAV |
| Single commodity VaR | 0.5% of NAV |

### 7.2 Stress Testing

Weekly stress tests must include:

| Scenario | Assumption |
|----------|------------|
| Oil spike | +$30/bbl in 5 days |
| Oil collapse | -$20/bbl in 5 days |
| Gold crash | -15% in 10 days |
| Agricultural shock | +50% corn/wheat in 30 days |
| USD strengthening | +10% DXY in 30 days |

### 7.3 Correlation Monitoring

Monitor correlations between:
- Energy complex (crude, products, natural gas)
- Metals complex (gold, silver, copper)
- Risk assets vs safe haven (crude vs gold)

## 8. Margin Management

### 8.1 Initial Margin Requirements

Maintain excess margin of at least 25% above exchange requirements.

### 8.2 Margin Call Procedures

| Level | Action |
|-------|--------|
| Excess margin < 25% | Alert to PM and CIO |
| Excess margin < 15% | Reduce positions within 24 hours |
| Excess margin < 5% | Immediate position reduction |
| Margin call received | Meet call within exchange deadline |

### 8.3 Clearing Broker Diversification

- Minimum 2 clearing brokers for commodity futures
- No single broker > 60% of total margin
- Quarterly review of broker credit quality

## 9. Data Sources and Validation

### 9.1 Approved Price Sources

| Data Type | Primary Source | Secondary Source |
|-----------|----------------|------------------|
| Energy prices | EIA | Bloomberg |
| Metal prices | LBMA / LME | Bloomberg |
| Agricultural prices | CBOT/CME | Bloomberg |
| Position data | CFTC COT Reports | Prime broker |

### 9.2 Price Validation

Daily price validation includes:
- Comparison to exchange settlement prices
- Variance analysis (>2% triggers review)
- Staleness check (no prices > 24 hours old)

### 9.3 CFTC Reporting

Per CFTC Large Trader Reporting requirements:
- Report positions exceeding reportable levels daily
- File Form 40 (Trader Identification) as required
- Respond to special calls within deadline

## 10. Prohibited Activities

### 10.1 Absolute Prohibitions

- Physical delivery of any commodity
- Corner or squeeze any market
- Trade on material non-public information
- Exceed CFTC speculative position limits
- Trade in banned or sanctioned commodities

### 10.2 Restricted Activities

The following require CIO pre-approval:
- Positions in illiquid back-month contracts
- OTC commodity derivatives
- Commodity-linked structured products
- Positions in newly listed commodity contracts

## 11. Reporting Requirements

### 11.1 Daily Reports

- Position summary by commodity
- Roll calendar and upcoming expirations
- Margin utilization
- P&L by commodity

### 11.2 Weekly Reports

- CFTC position limit utilization
- Correlation analysis
- Stress test results
- Contango/backwardation analysis

### 11.3 Monthly Reports

- Full commodity attribution
- Roll cost analysis
- Comparison to commodity benchmarks (BCOM, GSCI)

## 12. Related Policies

- [Investment Guidelines](investment_guidelines.md)
- [Exposure Limits Policy](exposure_limits.md)
- [Risk Management Framework](risk_management_policy.md)
- [Counterparty Policy](counterparty_policy.md)

## 13. Regulatory References

- CFTC Part 150 - Speculative Position Limits
- CFTC Part 17 - Large Trader Reporting
- CFTC Part 4 - Commodity Pool Operators
- Dodd-Frank Act Title VII - Derivatives Regulation

---

*This document is confidential and for internal compliance use only.*
