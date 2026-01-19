# Exposure Limits Policy

## Document ID: POL-EXP-001
## Effective Date: January 1, 2024
## Last Reviewed: January 1, 2026
## Owner: Chief Investment Officer

---

## 1. Purpose

This policy establishes exposure limits to control the Fund's overall market risk. Exposure limits work in conjunction with concentration limits to ensure the portfolio remains within the risk parameters agreed with investors and documented in our offering materials.

## 2. Definitions

### 2.1 Key Terms

| Term | Definition | Formula |
|------|------------|---------|
| **Long Market Value (LMV)** | Sum of all long positions at market value | Σ (Long Position Prices × Quantities) |
| **Short Market Value (SMV)** | Absolute value of all short positions | \|Σ (Short Position Prices × Quantities)\| |
| **Gross Exposure** | Total market exposure, both sides | (LMV + SMV) / NAV |
| **Net Exposure** | Directional market exposure | (LMV - SMV) / NAV |
| **NAV** | Net Asset Value | Total Assets - Total Liabilities |

### 2.2 Position Treatment

| Position Type | Long Exposure | Short Exposure |
|---------------|---------------|----------------|
| Equity long | Full market value | — |
| Equity short | — | Absolute market value |
| Call option | Delta × Notional | — |
| Put option | — | Delta × Notional |
| Futures long | Notional value | — |
| Futures short | — | Notional value |
| ETF (long) | Market value | — |
| ETF (short) | — | Market value |

## 3. Gross Exposure Limits

### 3.1 Maximum Gross Exposure
**Maximum: 200% of NAV**

This limit ensures:
- Leverage remains within investor expectations
- Margin requirements remain manageable
- Portfolio can be liquidated in stress scenarios

### 3.2 Warning Threshold
**Warning: 180% of NAV**

When gross exposure exceeds 180%:
- Enhanced daily monitoring required
- PM must provide justification
- CIO review within 48 hours

### 3.3 Calculation Example
```
Long Market Value:  $2,500,000,000
Short Market Value:   $500,000,000
NAV:                $2,000,000,000

Gross Exposure = ($2,500,000,000 + $500,000,000) / $2,000,000,000
               = $3,000,000,000 / $2,000,000,000
               = 150%
```

## 4. Net Exposure Limits

### 4.1 Net Long Limit
**Maximum Net Long: 100% of NAV**

A net exposure above 100% indicates:
- Long bias exceeds the value of the fund
- Higher directional market risk
- Potential leverage concerns

### 4.2 Net Short Limit
**Maximum Net Short: -50% of NAV**

A net short position beyond -50% indicates:
- Significant bearish positioning
- Requires CIO approval above -30%

### 4.3 Warning Thresholds
| Direction | Warning | Maximum |
|-----------|---------|---------|
| Net Long | 90% | 100% |
| Net Short | -40% | -50% |

### 4.4 Calculation Example
```
Long Market Value:  $2,500,000,000
Short Market Value:   $500,000,000
NAV:                $2,000,000,000

Net Exposure = ($2,500,000,000 - $500,000,000) / $2,000,000,000
             = $2,000,000,000 / $2,000,000,000
             = 100%
```

## 5. Single Position Limits

### 5.1 Long Position Limit
**Maximum Single Long: 10% of NAV**

This prevents:
- Excessive idiosyncratic risk
- Over-reliance on single security performance
- Concentration in high-conviction ideas

### 5.2 Short Position Limit
**Maximum Single Short: 5% of NAV**

Short positions have asymmetric risk (unlimited loss potential), hence the tighter limit.

### 5.3 Exclusions
The following are excluded from single position limits:
- Index ETFs (SPY, QQQ, IWM) used for hedging
- Sector ETFs used for tactical hedging
- Government securities

### 5.4 Position Limit Table
| Position Type | Maximum % NAV | Rationale |
|---------------|---------------|-----------|
| Single long equity | 10% | Idiosyncratic risk |
| Single short equity | 5% | Asymmetric loss potential |
| Single corporate bond | 5% | Credit concentration |
| Single index ETF hedge | 15% | Lower idiosyncratic risk |

## 6. Monitoring Frequency

### 6.1 Intraday Monitoring
During volatile markets (VIX > 25):
- Exposure calculations every 2 hours
- Real-time alerts for limit breaches
- PM and risk team notifications

### 6.2 End-of-Day Monitoring
Standard monitoring includes:
- Gross exposure calculation
- Net exposure calculation
- Single position checks
- Automated compliance report

### 6.3 Monthly Review
- Exposure trend analysis
- Comparison to peer funds
- Assessment of limit appropriateness

## 7. Breach Procedures

### 7.1 Passive vs Active Breach

**Passive Breach**: Caused by market movement, not trading
- Document with timestamp and cause
- 48-hour cure period for minor breaches
- No trading restriction during cure period

**Active Breach**: Caused by trading activity
- Immediate notification to CIO
- Same-day cure required
- May require trade reversal

### 7.2 Escalation Matrix

| Breach Level | Gross Over | Net Over | Action |
|--------------|------------|----------|--------|
| Warning | 180-190% | 90-95% | Monitor, document |
| Minor | 190-200% | 95-100% | PM cure in 48h |
| Major | 200-210% | 100-110% | CIO cure in 24h |
| Critical | >210% | >110% | Immediate action, CCO |

### 7.3 Cure Actions
Acceptable methods to cure exposure breaches:
1. Reduce long positions (for net long breach)
2. Cover short positions (for gross breach)
3. Add hedges (index shorts to reduce net)
4. Wait for market movement (passive breach only)

## 8. Reporting Requirements

### 8.1 Daily Reports
- Exposure summary (gross and net)
- Top 10 long positions with % NAV
- Top 10 short positions with % NAV
- Positions within 2% of single position limit

### 8.2 Weekly Reports
- Exposure trend (5-day rolling)
- Sector exposure breakdown
- Factor exposure analysis

### 8.3 Investor Reports
- Monthly exposure ranges
- Average gross and net exposure
- No position-level disclosure (confidential)

## 9. Related Policies

- [Investment Guidelines](investment_guidelines.md)
- [Concentration Limits](concentration_limits.md)
- [Leverage Policy](leverage_policy.md)
- [Risk Management Framework](risk_management_policy.md)

## 10. Approval History

| Version | Date | Changes | Approved By |
|---------|------|---------|-------------|
| 2.1 | 2024-01-01 | Clarified ETF exclusions | CIO |
| 2.0 | 2023-07-01 | Added intraday monitoring | Board |
| 1.0 | 2022-01-01 | Initial policy | Board |

---

*This document is confidential and for internal compliance use only.*
