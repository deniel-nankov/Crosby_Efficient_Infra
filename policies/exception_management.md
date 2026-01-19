# Exception Management Policy

## Document ID: POL-EXC-001
## Effective Date: January 1, 2024
## Last Reviewed: January 1, 2026
## Owner: Chief Compliance Officer

---

## 1. Purpose

This policy establishes the procedures for identifying, documenting, escalating, and resolving compliance exceptions and limit breaches. Proper exception management ensures:
- Timely identification and resolution of issues
- Complete audit trail for regulatory examinations
- Consistent treatment of similar situations
- Protection of investor interests

## 2. Scope

This policy covers exceptions related to:
- Investment guideline breaches
- Concentration limit violations
- Exposure limit breaches
- Liquidity requirement failures
- Counterparty limit exceedances
- Risk metric threshold breaches
- Operational control failures

## 3. Breach Classification

### 3.1 Severity Levels

| Severity | Definition | % Over Limit | Response Time |
|----------|------------|--------------|---------------|
| **Critical** | Material breach requiring immediate action | >10% | Immediate |
| **Major** | Significant breach requiring same-day action | 5-10% | Same day |
| **Minor** | Small breach with standard cure period | <5% | 48 hours |
| **Warning** | Approaching limit, no breach | Within 5% of limit | Monitor |

### 3.2 Classification Examples

**Critical Breach:**
- Single issuer at 12% NAV (limit 10%, breach 20% over)
- Gross exposure at 230% (limit 200%, breach 15% over)
- Zero cash (minimum 2%)

**Major Breach:**
- Single issuer at 10.8% NAV (8% over limit)
- Net exposure at 107% (7% over limit)
- T+1 liquidity at 8% (20% under minimum)

**Minor Breach:**
- Single issuer at 10.3% NAV (3% over limit)
- Sector concentration at 31% (3.3% over limit)
- Net exposure at 102% (2% over limit)

**Warning (No Breach):**
- Single issuer at 9.5% NAV (approaching 10% limit)
- Technology sector at 28% (approaching 30% limit)
- T+7 liquidity at 42% (approaching 40% minimum)

## 4. Escalation Matrix

### 4.1 Notification Requirements

| Severity | Immediate Notification | Within 4 Hours | Within 24 Hours |
|----------|----------------------|-----------------|-----------------|
| Critical | PM, CIO, CCO | CEO, Board Chair | Full Board |
| Major | PM, CIO | CCO | CEO |
| Minor | PM | CIO | CCO |
| Warning | PM | — | Weekly summary |

### 4.2 Escalation Triggers

Automatic escalation occurs when:
1. Breach is not cured within required timeframe
2. Multiple related breaches occur simultaneously
3. Same breach occurs twice in 30 days
4. Breach involves prohibited activity

### 4.3 Escalation Levels

```
Level 1: Portfolio Manager
    ↓ (if not cured in required time)
Level 2: Chief Investment Officer
    ↓ (if not cured in 24 additional hours)
Level 3: Chief Compliance Officer + CEO
    ↓ (if not cured in 48 additional hours)
Level 4: Board of Directors
    ↓ (if material and ongoing)
Level 5: Investor Notification
```

## 5. Documentation Requirements

### 5.1 Breach Log Entry

Every breach must be documented with:

| Field | Description | Required |
|-------|-------------|----------|
| Breach ID | Unique identifier | Yes |
| Date/Time Detected | When breach was identified | Yes |
| Control Name | Name of the control breached | Yes |
| Limit | Applicable limit | Yes |
| Actual Value | Value at time of breach | Yes |
| Breach Amount | How much over/under limit | Yes |
| Severity | Critical/Major/Minor | Yes |
| Passive/Active | Was breach caused by trading or market | Yes |
| Root Cause | Why the breach occurred | Yes |
| PM Notified | Date/time PM was notified | Yes |
| Cure Action | What action was taken | Yes |
| Cure Date | When breach was cured | Yes |
| CCO Sign-off | CCO approval of closure | Yes |

### 5.2 Root Cause Analysis

For Major and Critical breaches, a root cause analysis must include:

1. **What happened?** - Factual description of the breach
2. **When did it happen?** - Timeline of events
3. **Why did it happen?** - Underlying cause
4. **Could it have been prevented?** - Control assessment
5. **How was it resolved?** - Cure actions taken
6. **How do we prevent recurrence?** - Process improvements

Root cause analysis must be completed within **5 business days** of breach detection.

### 5.3 Retention Requirements

- Breach logs: 7 years
- Root cause analyses: 7 years
- Email notifications: 7 years
- Supporting documentation: 7 years

## 6. Cure Procedures

### 6.1 Acceptable Cure Actions

**For Concentration Breaches:**
- Reduce position through sales
- Increase NAV through subscriptions
- Passive cure (market movement reduces position)
- Hedge with derivatives (with CIO approval)

**For Exposure Breaches:**
- Reduce long exposure (sell longs)
- Reduce short exposure (cover shorts)
- Add hedges (index shorts)
- Wait for market normalization (passive only)

**For Liquidity Breaches:**
- Liquidate illiquid positions
- Build cash reserves
- Suspend new illiquid investments
- Consider redemption gate (Board approval)

### 6.2 Cure Timeframes

| Severity | Standard Cure Period | Extension Available |
|----------|---------------------|---------------------|
| Critical | Immediate (same day) | No |
| Major | 24 hours | CIO approval, +24h max |
| Minor | 48 hours | CCO approval, +48h max |

### 6.3 Market Conditions Exception

During extreme market conditions (VIX > 40 or market circuit breaker), cure timeframes may be extended with:
- CIO and CCO joint approval
- Board notification
- Enhanced monitoring during extension

## 7. Recurring Breach Protocol

### 7.1 Definition
A recurring breach is:
- Same control breached 3+ times in 90 days
- Same root cause identified in 2+ breaches

### 7.2 Enhanced Procedures

For recurring breaches:
1. CCO-led investigation required
2. Process improvement plan mandatory
3. Potential limit adjustment review
4. Board notification
5. 6-month enhanced monitoring

## 8. Investor Notification

### 8.1 Notification Triggers

Investors must be notified when:
- Breach is material (>10% of limit)
- Breach persists >5 business days
- Breach affects redemption ability
- Breach involves regulatory matter
- Board determines notification warranted

### 8.2 Notification Content

Investor notification includes:
- Nature of the breach
- When it occurred
- Actions taken to cure
- Current status
- Steps to prevent recurrence

### 8.3 Notification Timing

- Material breaches: Within 10 business days
- Redemption-affecting: Immediate
- Regulatory matters: Per regulatory requirements

## 9. Reporting

### 9.1 Daily Exception Report
- All new breaches
- Open breach status
- Breaches cured today

### 9.2 Weekly Exception Summary
- Total breaches by severity
- Breaches by category
- Average time to cure
- Recurring breach flags

### 9.3 Monthly Board Report
- Full exception log
- Trend analysis
- Root cause themes
- Process improvements implemented

### 9.4 Annual Compliance Review
- Exception management effectiveness
- Policy adequacy assessment
- Limit appropriateness review
- Peer comparison

## 10. Roles and Responsibilities

| Role | Responsibilities |
|------|------------------|
| Portfolio Manager | Immediate notification, cure execution |
| Operations | Breach detection, documentation |
| Risk Manager | Severity assessment, monitoring |
| CCO | Policy oversight, sign-off, Board reporting |
| CIO | Major/Critical breach approval |
| Board | Critical breach review, policy approval |

## 11. Related Policies

- [Investment Guidelines](investment_guidelines.md)
- [Concentration Limits](concentration_limits.md)
- [Exposure Limits](exposure_limits.md)
- [Liquidity Policy](liquidity_policy.md)

---

*This document is confidential and for internal compliance use only.*
