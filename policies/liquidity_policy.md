# Liquidity Risk Management Policy

## Document ID: POL-LIQ-001
## Effective Date: January 1, 2024
## Last Reviewed: January 1, 2026
## Owner: Chief Risk Officer

---

## 1. Purpose

This policy establishes the framework for managing liquidity risk at Master Fund LP. Proper liquidity management ensures the Fund can:
- Meet investor redemption requests
- Satisfy margin calls and collateral requirements
- Take advantage of investment opportunities
- Avoid forced selling at unfavorable prices

## 2. Regulatory Context

This policy is designed to comply with:
- SEC Rule 22e-4 (Liquidity Risk Management Programs)
- Form N-PORT liquidity reporting requirements
- AIFMD liquidity stress testing requirements (where applicable)
- Best practices for hedge fund liquidity management

## 3. Liquidity Classification Framework

### 3.1 Liquidity Buckets

All positions are classified into liquidity buckets based on the time required to liquidate:

| Bucket | Timeframe | Definition | Minimum Required |
|--------|-----------|------------|------------------|
| **T+1** | 0-1 business day | Highly liquid | **10% of NAV** |
| **T+3** | 2-3 business days | Liquid | **25% of NAV** |
| **T+7** | 4-7 business days | Moderately liquid | **40% of NAV** |
| **T+30** | 8-30 calendar days | Less liquid | **60% of NAV** |
| **T+90** | 31-90 calendar days | Illiquid | **80% of NAV** |
| **>T+90** | > 90 calendar days | Highly illiquid | *No minimum* |

### 3.2 Classification Methodology

Liquidity classification is based on:

1. **Average Daily Volume (ADV)**: 20-day trailing average
2. **Participation Rate**: 25% of ADV assumed for liquidation
3. **Price Impact**: Estimated slippage based on position size

```
Days to Liquidate = Position Shares / (ADV × 0.25)
```

### 3.3 Asset-Specific Guidelines

| Asset Type | Typical Bucket | Notes |
|------------|----------------|-------|
| Large-cap US equities (>$10B market cap) | T+1 | High ADV, tight spreads |
| Mid-cap US equities ($2B-$10B) | T+3 | Moderate ADV |
| Small-cap US equities (<$2B) | T+7 to T+30 | Lower ADV, wider spreads |
| International developed | T+3 | Market hours consideration |
| Emerging market equities | T+7 | Lower ADV, potential restrictions |
| US Treasury securities | T+1 | Highly liquid |
| Corporate bonds (IG) | T+3 to T+7 | Dealer market |
| Corporate bonds (HY) | T+7 to T+30 | Less liquid |
| Private investments | >T+90 | Lock-up periods |

## 4. Cash Management

### 4.1 Minimum Cash Buffer
**Minimum Cash: 2% of NAV**

This buffer ensures:
- Coverage for operational expenses (management fees, admin costs)
- Margin for daily settlement obligations
- Cushion for unexpected redemptions

### 4.2 Maximum Cash Guideline
**Maximum Cash: 15% of NAV**

Excess cash above 15% represents:
- Opportunity cost (uninvested capital)
- Potential drag on returns
- Requires PM justification and documentation

### 4.3 Cash Calculation
Cash includes:
- Bank deposits
- Money market funds
- Treasury bills < 90 days to maturity

## 5. Liquidity Stress Testing

### 5.1 Frequency
- **Weekly**: Standard stress tests
- **Ad hoc**: During market dislocations
- **Monthly**: Comprehensive scenario analysis

### 5.2 Stress Scenarios

| Scenario | Assumption | Measurement |
|----------|------------|-------------|
| **Redemption Shock** | 20% NAV redemption in 5 days | Can Fund meet redemptions? |
| **Market Dislocation** | ADV drops 50% | Days to liquidate portfolio |
| **Correlation Stress** | All positions become T+30 or worse | Portfolio liquidity floor |
| **Combined Stress** | 15% redemption + 30% ADV reduction | Worst-case scenario |

### 5.3 Liquidity Coverage Ratio

```
Liquidity Coverage Ratio = Liquid Assets (T+7) / Potential Outflows (30 days)

Minimum Required: 100%
Target: 150%
```

## 6. Redemption Terms Alignment

### 6.1 Current Redemption Terms
- **Notice Period**: 45 days
- **Redemption Frequency**: Monthly
- **Gate Provision**: 25% of NAV per quarter

### 6.2 Portfolio-Redemption Alignment
The portfolio's T+30 liquidity must exceed potential redemptions:

```
T+30 Liquidity % > Maximum Redemption % + 10% buffer
T+30 Liquidity % > 25% + 10% = 35% minimum
```

Current policy requires 60% T+30 liquidity, providing substantial buffer.

## 7. Breach Procedures

### 7.1 Warning Threshold
- **Trigger**: Any liquidity bucket within 5% of minimum
- **Action**: Enhanced monitoring, PM notification
- **Documentation**: Note in daily compliance log

### 7.2 Breach Response

| Breach Severity | Definition | Response Time | Escalation |
|-----------------|------------|---------------|------------|
| Minor | <5% below minimum | 5 business days | PM |
| Major | 5-10% below minimum | 2 business days | CIO |
| Critical | >10% below minimum | Immediate | CCO + Board |

### 7.3 Cure Actions
1. Reduce illiquid positions
2. Suspend new investments in illiquid assets
3. Build cash through liquidation
4. Consider redemption suspension if critical (Board approval required)

## 8. Monitoring and Reporting

### 8.1 Daily Monitoring
- Automated liquidity bucket calculations
- Cash position tracking
- Liquidity coverage ratio

### 8.2 Weekly Reports
- Liquidity summary by bucket
- Days-to-liquidate analysis
- Stress test results

### 8.3 Monthly Reports
- Full liquidity risk assessment
- Peer comparisons
- Redemption pattern analysis

## 9. Roles and Responsibilities

| Role | Responsibility |
|------|----------------|
| Portfolio Manager | Position-level liquidity assessment |
| Risk Manager | Daily liquidity monitoring, stress testing |
| CCO | Policy compliance, exception approval |
| Board | Policy approval, critical breach review |

## 10. Related Policies

- [Investment Guidelines - Section 3](investment_guidelines.md#3-liquidity-requirements)
- [Redemption Policy](redemption_policy.md)
- [Cash Management Procedures](cash_management.md)

---

*This document is confidential and for internal compliance use only.*
