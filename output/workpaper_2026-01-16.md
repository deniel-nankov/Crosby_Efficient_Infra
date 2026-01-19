# Daily Compliance Workpaper

**Date:** 2026-01-16  
**Prepared By:** Compliance RAG System  
**Data Source:** MockAdapter  

---

## Executive Summary

| Metric | Value |
|--------|-------|
| Total Controls Tested | 7 |
| Controls Passed | 5 |
| Exceptions Identified | 2 |

---

## Position Summary

| Security | Quantity | Market Value |
|----------|----------|--------------|
| SEC-0001 | 290,152 | $198,675,631.00 |
| SEC-0002 | 553,637 | $100,404,860.00 |
| SEC-0003 | 1,021,149 | $216,606,130.00 |
| SEC-0004 | 331,208 | $138,327,350.00 |
| SEC-0005 | 132,343 | $73,178,764.00 |
| SEC-0006 | 605,836 | $230,978,017.00 |
| SEC-0007 | 322,234 | $174,318,599.00 |
| SEC-0008 | 226,570 | $172,076,695.00 |
| SEC-0009 | 126,897 | $56,526,082.00 |
| SEC-0010 | 745,998 | $154,730,818.00 |
| SEC-0011 | 419,851 | $279,823,502.00 |
| SEC-0012 | 222,485 | $152,291,912.00 |
| SEC-0013 | 194,589 | $141,818,106.00 |
| SEC-0014 | 263,967 | $202,970,176.00 |
| SEC-0015 | 388,121 | $265,638,396.00 |

---

## Control Test Results


**Control: CONC_ISSUER_001**
**Status: PASSED ✓**

Single Issuer Concentration

**Regulatory Basis:**
This control is governed by SEC Rule 206(4)-2. Per policy document ADV-2B-CUSTODY: 
"SEC Rule 206(4)-2 requires registered investment advisers with custody 
            of client funds to maintain those assets with a quali..."

**Evidence:**
- Test Date: 2026-01-16
- Threshold: 10.0 (lte)
- Actual Value: 8.5
- Result: Compliant

No exceptions noted.

---


**Control: CONC_SECTOR_001**
**Status: EXCEPTION ✗**

Sector Concentration - Technology

**Regulatory Basis:**
This control is governed by SEC Rule 206(4)-2. Per policy document ADV-2B-CUSTODY: 
"SEC Rule 206(4)-2 requires registered investment advisers with custody 
            of client funds to maintain those assets with a quali..."

**Exception Details:**
- Test Date: 2026-01-16
- Threshold: 30.0 (lte)
- Actual Value: 28.0
- Breach Amount: N/A

**Required Action:**
Review and document the exception per compliance procedures. 
If the exception persists beyond the remediation period specified in ADV-2B-CUSTODY, 
escalation to the CCO is required.

---


**Control: EXP_GROSS_001**
**Status: PASSED ✓**

Gross Exposure

**Regulatory Basis:**
This control is governed by SEC Rule 206(4)-2. Per policy document ADV-2B-CUSTODY: 
"SEC Rule 206(4)-2 requires registered investment advisers with custody 
            of client funds to maintain those assets with a quali..."

**Evidence:**
- Test Date: 2026-01-16
- Threshold: 200.0 (lte)
- Actual Value: 145.0
- Result: Compliant

No exceptions noted.

---


**Control: EXP_NET_001**
**Status: PASSED ✓**

Net Exposure

**Regulatory Basis:**
This control is governed by SEC Rule 206(4)-2. Per policy document ADV-2B-CUSTODY: 
"SEC Rule 206(4)-2 requires registered investment advisers with custody 
            of client funds to maintain those assets with a quali..."

**Evidence:**
- Test Date: 2026-01-16
- Threshold: 100.0 (lte)
- Actual Value: 72.0
- Result: Compliant

No exceptions noted.

---


**Control: LIQ_T1_001**
**Status: PASSED ✓**

T+1 Liquidity

**Regulatory Basis:**
This control is governed by SEC Rule 206(4)-2. Per policy document ADV-2B-CUSTODY: 
"SEC Rule 206(4)-2 requires registered investment advisers with custody 
            of client funds to maintain those assets with a quali..."

**Evidence:**
- Test Date: 2026-01-16
- Threshold: 10.0 (gte)
- Actual Value: 18.0
- Result: Compliant

No exceptions noted.

---


**Control: LIQ_T7_001**
**Status: EXCEPTION ✗**

T+7 Liquidity

**Regulatory Basis:**
This control is governed by SEC Rule 206(4)-2. Per policy document ADV-2B-CUSTODY: 
"SEC Rule 206(4)-2 requires registered investment advisers with custody 
            of client funds to maintain those assets with a quali..."

**Exception Details:**
- Test Date: 2026-01-16
- Threshold: 40.0 (gte)
- Actual Value: 35.0
- Breach Amount: 5.0

**Required Action:**
Review and document the exception per compliance procedures. 
If the exception persists beyond the remediation period specified in ADV-2B-CUSTODY, 
escalation to the CCO is required.

---


**Control: CASH_MIN_001**
**Status: PASSED ✓**

Minimum Cash Buffer

**Regulatory Basis:**
This control is governed by SEC Rule 206(4)-2. Per policy document ADV-2B-CUSTODY: 
"SEC Rule 206(4)-2 requires registered investment advisers with custody 
            of client funds to maintain those assets with a quali..."

**Evidence:**
- Test Date: 2026-01-16
- Threshold: 2.0 (gte)
- Actual Value: 3.2
- Result: Compliant

No exceptions noted.

---


## Attestation

This workpaper was generated automatically by the Compliance RAG System.
All numerical values are sourced directly from audited client systems.
Narratives are generated using regulatory policy retrieval and LLM assistance.

**Important:** LLM is used ONLY for prose generation. All calculations 
and numerical assertions come directly from source systems.

---

*Generated: 2026-01-16T14:11:40.480057*
