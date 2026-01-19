# PR: Canonical PRISM Schema + Domain Mapping

## Problem

Every time we load a new domain, the codebase gets adjusted:
- `obs_date` vs `timestamp` vs `cycle`
- `unit_id` vs `engine_id` vs `entity_id`
- Hardcoded assumptions about time scale (days, cycles, milliseconds)

This breaks the domain-agnostic promise.

## Solution

**ONE canonical schema. Fetchers conform. YAML maps for reports.**

```
Source Data → Fetcher → PRISM Schema → Engines → Results → YAML → Human Reports
                ↑                                            ↑
           (transforms)                                 (translates)
```

---

## The Canonical PRISM Schema

### `raw/observations.parquet`

| Column | Type | Description |
|--------|------|-------------|
| `entity_id` | string | What fails (engine, bearing, reactor, patient) |
| `signal_id` | string | What we measure (internal sensor name) |
| `timestamp` | float | Time index — cycles, seconds, ordinal, doesn't matter |
| `value` | float | The measurement |

**Optional columns:**
| Column | Type | Description |
|--------|------|-------------|
| `target` | float | RUL, yield, failure time (if known) |
| `op_setting_1..N` | float | Operating conditions (for C-MAPSS style) |

**Rules:**
- `timestamp` is ALWAYS float, NEVER datetime
- `timestamp` is monotonically increasing within an entity
- PRISM doesn't care if it's cycles, seconds, or days
- All engines use `timestamp` — no `obs_date`, `cycle`, `time_idx`

### `raw/signals.parquet`

| Column | Type | Description |
|--------|------|-------------|
| `signal_id` | string | Internal PRISM signal name |
| `source_name` | string | Original name from source data |
| `description` | string | Human-readable description |
| `unit` | string | Measurement unit (optional) |

---

## Domain YAML Structure

Each domain gets a config YAML that maps source → PRISM and back:

```yaml
# config/domains/cmapss.yaml
domain: cmapss
description: "NASA C-MAPSS Turbofan Degradation"

# Time configuration (for reporting only - PRISM treats as float)
time:
  source_column: cycle
  unit: cycles
  description: "Engine operating cycles"

# Entity configuration
entity:
  source_column: unit
  description: "Turbofan engine unit"

# Target variable (optional)
target:
  source_column: RUL
  description: "Remaining Useful Life (cycles)"

# Signal mapping: source_name → prism internal
signals:
  # Temperatures
  T2:
    prism_id: temp_inlet
    description: "Total temperature at fan inlet"
    unit: "°R"
    category: temperature
  T24:
    prism_id: temp_lpc_outlet
    description: "Total temperature at LPC outlet"
    unit: "°R"
    category: temperature
  T30:
    prism_id: temp_hpc_outlet
    description: "Total temperature at HPC outlet"
    unit: "°R"
    category: temperature
  T50:
    prism_id: temp_lpt_outlet
    description: "Total temperature at LPT outlet"
    unit: "°R"
    category: temperature
    
  # Pressures
  P2:
    prism_id: press_fan_inlet
    description: "Pressure at fan inlet"
    unit: "psia"
    category: pressure
  P15:
    prism_id: press_bypass
    description: "Total pressure in bypass-duct"
    unit: "psia"
    category: pressure
  P30:
    prism_id: press_hpc_outlet
    description: "Total pressure at HPC outlet"
    unit: "psia"
    category: pressure
  Ps30:
    prism_id: press_static_hpc
    description: "Static pressure at HPC outlet"
    unit: "psia"
    category: pressure
    
  # Speeds
  Nf:
    prism_id: speed_fan
    description: "Physical fan speed"
    unit: "rpm"
    category: speed
  Nc:
    prism_id: speed_core
    description: "Physical core speed"
    unit: "rpm"
    category: speed
  NRf:
    prism_id: speed_fan_corrected
    description: "Corrected fan speed"
    unit: "rpm"
    category: speed
  NRc:
    prism_id: speed_core_corrected
    description: "Corrected core speed"
    unit: "rpm"
    category: speed

  # Flows
  W31:
    prism_id: flow_hpt_coolant
    description: "HPT coolant bleed"
    unit: "lbm/s"
    category: flow
  W32:
    prism_id: flow_lpt_coolant
    description: "LPT coolant bleed"
    unit: "lbm/s"
    category: flow

  # Ratios
  BPR:
    prism_id: ratio_bypass
    description: "Bypass ratio"
    unit: "-"
    category: ratio
  epr:
    prism_id: ratio_pressure_engine
    description: "Engine pressure ratio"
    unit: "-"
    category: ratio
  farB:
    prism_id: ratio_fuel_air
    description: "Fuel/air ratio"
    unit: "-"
    category: ratio

  # Other
  phi:
    prism_id: flow_coefficient
    description: "Flow coefficient"
    unit: "-"
    category: flow
  htBleed:
    prism_id: bleed_enthalpy
    description: "Bleed enthalpy"
    unit: "-"
    category: bleed

# Operating settings (become metadata, not signals)
op_settings:
  op_setting_1:
    description: "Altitude"
    unit: "ft"
  op_setting_2:
    description: "Mach number"
    unit: "-"
  op_setting_3:
    description: "Throttle resolver angle"
    unit: "deg"

# Cohort hints (optional - for validation)
expected_cohorts:
  temperatures: [T2, T24, T30, T50]
  pressures: [P2, P15, P30, Ps30]
  fan_spool: [Nf, NRf]
  core_spool: [Nc, NRc]
  flows: [W31, W32, phi]
  ratios: [BPR, epr, farB]
```

---

## Fetcher Pattern

All fetchers transform source data to canonical schema:

```python
# fetchers/cmapss_fetcher.py

def fetch(config: dict) -> None:
    """
    Transform C-MAPSS data to PRISM canonical schema.
    
    Source columns: unit, cycle, op1, op2, op3, s1..s21
    Target columns: entity_id, signal_id, timestamp, value
    """
    # Load domain config
    domain_yaml = load_yaml('config/domains/cmapss.yaml')
    
    # Read source data
    df = load_cmapss_txt(...)
    
    # Transform to PRISM schema
    observations = []
    
    for row in df.iter_rows(named=True):
        entity_id = str(row['unit'])
        timestamp = float(row['cycle'])
        
        # Each sensor becomes a row
        for source_name, signal_config in domain_yaml['signals'].items():
            if source_name in row:
                observations.append({
                    'entity_id': entity_id,
                    'signal_id': signal_config['prism_id'],
                    'timestamp': timestamp,
                    'value': float(row[source_name]),
                })
        
        # Add target if present
        if 'RUL' in row:
            observations.append({
                'entity_id': entity_id,
                'signal_id': 'target_rul',
                'timestamp': timestamp,
                'value': float(row['RUL']),
            })
    
    # Write canonical schema
    obs_df = pl.DataFrame(observations)
    write_parquet(obs_df, 'raw', 'observations')
    
    # Write signal metadata
    signals_df = pl.DataFrame([
        {
            'signal_id': cfg['prism_id'],
            'source_name': src_name,
            'description': cfg['description'],
            'unit': cfg.get('unit', ''),
        }
        for src_name, cfg in domain_yaml['signals'].items()
    ])
    write_parquet(signals_df, 'raw', 'signals')
```

---

## Report Translation

When generating reports, translate back to human-readable names:

```python
# prism/utils/report_translator.py

def translate_results(results: pl.DataFrame, domain: str) -> pl.DataFrame:
    """
    Translate PRISM internal names back to source names for reports.
    """
    domain_yaml = load_yaml(f'config/domains/{domain}.yaml')
    
    # Build reverse mapping: prism_id → source_name
    reverse_map = {
        cfg['prism_id']: source_name
        for source_name, cfg in domain_yaml['signals'].items()
    }
    
    # Translate signal_id column
    return results.with_columns(
        pl.col('signal_id').replace(reverse_map).alias('sensor_name')
    )
```

---

## Migration Checklist

### Code Changes (One-Time)

- [ ] Search/replace `obs_date` → `timestamp` in all engines
- [ ] Search/replace `unit_id` → `entity_id` where needed
- [ ] Update `parquet_store.py` schema definitions
- [ ] Add `load_domain_config()` utility
- [ ] Add `translate_results()` for reports

### Per-Domain (Each Fetcher)

- [ ] Create `config/domains/{domain}.yaml`
- [ ] Update fetcher to output canonical schema
- [ ] Validate with: `python -m prism.entry_points.validate_schema`

### Validation Script

```bash
# Check that observations.parquet matches canonical schema
python -m prism.entry_points.validate_schema --domain cmapss

Expected output:
  ✓ entity_id: string (100 unique)
  ✓ signal_id: string (21 unique)
  ✓ timestamp: float (monotonic within entity)
  ✓ value: float (no nulls)
  ✓ Schema valid!
```

---

## Benefits

1. **Domain-agnostic engines** — code never changes for new domains
2. **Time-scale independent** — cycles, seconds, days all work
3. **Human-readable reports** — YAML translates back to source names
4. **Validation** — schema checker catches mismatches early
5. **Onboarding** — new domains just need a YAML + fetcher

---

## Example: Adding a New Domain

```bash
# 1. Create domain config
cp config/domains/template.yaml config/domains/cheme.yaml
# Edit with your signal mappings

# 2. Create fetcher
cp fetchers/template_fetcher.py fetchers/cheme_fetcher.py
# Edit to transform your source data

# 3. Run fetch
python -m prism.entry_points.fetch --domain cheme

# 4. Validate
python -m prism.entry_points.validate_schema --domain cheme

# 5. Run pipeline (unchanged!)
python -m prism.entry_points.signal_vector --domain cheme
python -m prism.entry_points.geometry --domain cheme
# ...
```

**The pipeline code never changes. Only the YAML and fetcher.**
