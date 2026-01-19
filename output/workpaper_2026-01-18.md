# Daily Compliance Workpaper

**Date:** 2026-01-18  
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
| SEC-0001 | 1,231,482 | $282,518,408.00 |
| SEC-0002 | 653,541 | $290,066,690.00 |
| SEC-0003 | 680,124 | $293,908,261.00 |
| SEC-0004 | 1,781,225 | $275,868,498.00 |
| SEC-0005 | 333,463 | $211,326,207.00 |
| SEC-0006 | 180,346 | $136,062,147.00 |
| SEC-0007 | 515,067 | $210,110,476.00 |
| SEC-0008 | 642,814 | $297,195,626.00 |
| SEC-0009 | 182,851 | $96,443,303.00 |
| SEC-0010 | 1,077,355 | $281,103,845.00 |
| SEC-0011 | 193,470 | $54,961,226.00 |
| SEC-0012 | 461,405 | $259,544,743.00 |
| SEC-0013 | 416,770 | $249,377,578.00 |
| SEC-0014 | 557,769 | $73,332,884.00 |
| SEC-0015 | 352,290 | $154,747,950.00 |

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
- Test Date: 2026-01-18
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
- Test Date: 2026-01-18
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
- Test Date: 2026-01-18
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
- Test Date: 2026-01-18
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
- Test Date: 2026-01-18
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
- Test Date: 2026-01-18
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
- Test Date: 2026-01-18
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

*Generated: 2026-01-18T16:13:57.154803*
