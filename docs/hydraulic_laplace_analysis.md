# Hydraulic System Laplace Field Analysis Report

**Generated:** January 15, 2026
**Source:** `laplace_field_analysis.csv`
**Domain:** Hydraulic System Monitoring

---

## Executive Summary

This analysis examines the divergence field topology of 17 hydraulic system indicators across two operational states: **DEGRADED** and **HEALTHY**. The Laplace field analysis reveals fundamentally different energy/information flow patterns between system states, providing a geometric signature for system health assessment.

**Key Finding:** In degraded operation, the system exhibits a coherent SOURCE-dominated topology with systematic positive divergence across most sensors. In healthy operation, the field becomes more balanced with several indicators transitioning to SINK behavior—suggesting healthy systems dissipate energy more uniformly.

---

## Methodology

The analysis computes divergence metrics across sliding windows of the hydraulic time series:
- **Total Divergence:** Cumulative divergence over all observations
- **Mean Divergence:** Average divergence per observation
- **Accelerating/Decelerating Windows:** Count of windows with positive/negative divergence rates
- **Field Topology Role:** Classification as SOURCE (net positive), SINK (net negative), or NEUTRAL

---

## System State Comparison

### DEGRADED State (n ≈ 25,000 observations per indicator)

| Indicator | Total Divergence | Mean Divergence | Topology Role |
|-----------|------------------|-----------------|---------------|
| HYD_EPS1  | 10,443.35        | 0.4188          | SOURCE        |
| HYD_TS4   | 350.05           | 0.0140          | SOURCE        |
| HYD_TS2   | 314.42           | 0.0123          | SOURCE        |
| HYD_PS5   | 305.69           | 0.0120          | SOURCE        |
| HYD_TS1   | 304.97           | 0.0120          | SOURCE        |
| HYD_VS1   | 261.17           | 0.0107          | SOURCE        |
| HYD_TS3   | 260.65           | 0.0102          | SOURCE        |
| HYD_SE    | 223.70           | 0.0091          | SOURCE        |
| HYD_PS6   | 202.53           | 0.0080          | SOURCE        |
| HYD_CE    | 132.40           | 0.0055          | SOURCE        |
| HYD_PS3   | 119.17           | 0.0048          | SOURCE        |
| HYD_FS1   | 104.17           | 0.0042          | SOURCE        |
| HYD_PS4   | 87.36            | 0.0035          | SOURCE        |
| HYD_CP    | 69.80            | 0.0028          | SOURCE        |
| HYD_FS2   | 61.11            | 0.0024          | SOURCE        |
| HYD_PS1   | -8.67            | -0.0003         | NEUTRAL       |
| HYD_PS2   | -19.67           | -0.0008         | NEUTRAL       |
| HYD_STABLE_FLAG | -191.05     | -0.0077         | SINK          |

**Degraded State Characteristics:**
- 15 of 18 indicators act as SOURCEs (83%)
- Only 1 SINK (HYD_STABLE_FLAG) and 2 NEUTRAL
- HYD_EPS1 dominates with divergence 30x higher than next indicator
- High variance in HYD_EPS1 (std: 68,108) indicates instability
- Accelerating windows slightly exceed decelerating (avg ratio: 1.05)

### HEALTHY State (n ≈ 192-204 observations per indicator)

| Indicator | Total Divergence | Mean Divergence | Topology Role |
|-----------|------------------|-----------------|---------------|
| HYD_EPS1  | 717.54           | 3.5174          | SOURCE        |
| HYD_STABLE_FLAG | 293.50      | 1.6676          | SOURCE        |
| HYD_SE    | 142.58           | 0.7426          | SOURCE        |
| HYD_FS1   | 106.80           | 0.5563          | SOURCE        |
| HYD_PS3   | 95.34            | 0.4966          | SOURCE        |
| HYD_PS2   | 63.31            | 0.3297          | SOURCE        |
| HYD_PS4   | 49.24            | 0.2414          | SOURCE        |
| HYD_TS2   | 39.07            | 0.1915          | SOURCE        |
| HYD_PS1   | 33.62            | 0.1751          | SOURCE        |
| HYD_TS4   | 3.90             | 0.0191          | SOURCE        |
| HYD_CE    | 2.11             | 0.0110          | SOURCE        |
| HYD_PS5   | 0.70             | 0.0034          | SOURCE        |
| HYD_VS1   | 0.36             | 0.0018          | SOURCE        |
| HYD_CP    | -2.03            | -0.0106         | SINK          |
| HYD_FS2   | -4.18            | -0.0205         | SINK          |
| HYD_PS6   | -7.73            | -0.0379         | SINK          |
| HYD_TS3   | -21.86           | -0.1072         | SINK          |
| HYD_TS1   | -50.37           | -0.2469         | SINK          |

**Healthy State Characteristics:**
- 13 SOURCEs and 5 SINKs (28% SINK vs 6% in degraded)
- HYD_STABLE_FLAG flips from SINK → SOURCE (critical diagnostic)
- Temperature sensors (TS1, TS3) become SINKs in healthy state
- Much lower variance across all indicators
- More balanced accelerating/decelerating window ratios

---

## Critical Diagnostic Indicators

### 1. HYD_STABLE_FLAG (Most Diagnostic)
- **Degraded:** SINK (-191.05 total, -0.0077 mean)
- **Healthy:** SOURCE (+293.50 total, +1.6676 mean)
- **Interpretation:** The stability flag's topology role is the clearest binary classifier for system health

### 2. HYD_EPS1 (Efficiency/Power Sensor)
- **Degraded:** Extreme SOURCE (10,443, std: 68,108)
- **Healthy:** Moderate SOURCE (717, std: 41)
- **Interpretation:** Wild divergence in degraded state indicates pump efficiency instability

### 3. Temperature Sensors (TS1, TS3)
- **Degraded:** Both SOURCE
- **Healthy:** Both SINK
- **Interpretation:** Healthy systems dissipate heat; degraded systems accumulate thermal energy

---

## Field Topology Summary

```
DEGRADED STATE                    HEALTHY STATE
===============                   ==============
SOURCE-dominated (83%)            Balanced (72% SOURCE, 28% SINK)

     ┌──────────────┐                  ┌──────────────┐
     │   EPS1       │ ←── EXTREME      │   EPS1       │ ←── MODERATE
     │   (10,443)   │     SOURCE       │   (717)      │     SOURCE
     └──────────────┘                  └──────────────┘
           ↓                                 ↓
    ┌──────────────┐                  ┌──────────────┐
    │ 14 SENSORS   │                  │ STABLE_FLAG  │ ←── FLIPPED
    │ ALL SOURCE   │                  │   SOURCE     │     TO SOURCE
    └──────────────┘                  └──────────────┘
           ↓                                 ↓
    ┌──────────────┐                  ┌──────────────┐
    │ STABLE_FLAG  │ ←── ONLY SINK    │ TS1,TS3,CP   │ ←── HEAT
    │    SINK      │                  │ FS2,PS6 SINK │     DISSIPATION
    └──────────────┘                  └──────────────┘
```

---

## Conclusions

1. **Degraded systems act as energy accumulators** with nearly all sensors showing positive divergence (SOURCE behavior). Energy/stress builds rather than dissipates.

2. **Healthy systems show balanced topology** with heat dissipation pathways (temperature sensors as SINKs) actively removing accumulated energy.

3. **HYD_STABLE_FLAG topology role is binary diagnostic:** SINK = degraded, SOURCE = healthy.

4. **HYD_EPS1 variance is the leading instability indicator** with 1,650x higher standard deviation in degraded state.

5. **Field topology provides geometric signature** for predictive maintenance: transition from balanced to SOURCE-dominated precedes degradation.

---

## Recommendations

- Monitor HYD_STABLE_FLAG topology role as primary health indicator
- Track HYD_EPS1 divergence variance for early warning
- Alert when temperature sensors (TS1, TS3) transition from SINK → SOURCE
- Consider field topology balance ratio as continuous health metric

---

*Analysis conducted using PRISM Laplace field divergence engine*
