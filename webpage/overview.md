---
title: System Overview
description: How Ørthon's three-layer architecture detects degradation through behavioral geometry
---

Ørthon analyzes complex industrial systems through a paradigm shift: instead of monitoring individual sensor thresholds, it tracks how **relationships between components** evolve over time.

## The Core Thesis

> "Systems lose coherence before they fail."

Traditional predictive maintenance asks: "Has sensor X exceeded threshold Y?"

Ørthon asks: "Are the relationships between components breaking down?"

This shift matters because:

1. **Early detection** — Coherence loss appears before individual failures
2. **Domain agnostic** — The same math works for turbofans, bearings, chemical plants
3. **Interpretable** — You can see *which* relationships are degrading

## Architecture Overview

```
Raw Sensor Data
      ↓
┌─────────────────┐
│  Vector Layer   │  51 behavioral metrics → 740K features
└────────┬────────┘
         ↓
┌─────────────────┐
│ Geometry Layer  │  1,035 pairwise relationships
└────────┬────────┘
         ↓
┌─────────────────┐
│  State Layer    │  Regime detection + coherence tracking
└────────┬────────┘
         ↓
   Diagnostic Output
```

## Data Model

All data is grouped by `(entity_id, timestamp)`:

- **entity_id** — The unit being monitored (engine, bearing, pump)
- **timestamp** — Native sampling frequency preserved (no interpolation)

This grouping enables regime-aware analysis across multiple operating conditions.

## Key Insight: hd_slope

The most predictive feature to emerge from validation is `hd_slope` — the **velocity of coherence loss**.

- Measures how fast a system is moving away from baseline behavior
- Captures degradation dynamics, not just current state
- Works across domains without tuning
