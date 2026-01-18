# PRISM Fetchers

Standalone data fetchers for acquiring raw observations from industrial, scientific, and clinical sources.

**These are workspace scripts, NOT part of the prism package.**

## Design Principles

- Fetchers do NOT import from `prism`
- Each fetcher exposes one function: `fetch(config) -> list[dict]`
- Fetchers return raw data only - no database writes
- YAML files define what to fetch

## Available Fetchers (16)

### Industrial / Prognostics

| Fetcher | Source | Domain |
|---------|--------|--------|
| `cmapss_fetcher.py` | NASA C-MAPSS | Turbofan run-to-failure |
| `femto_fetcher.py` | FEMTO Bearing | Bearing degradation |
| `cwru_bearing_fetcher.py` | CWRU | Bearing fault diagnosis |
| `nasa_bearing_fetcher.py` | NASA IMS | Bearing run-to-failure |
| `hydraulic_fetcher.py` | UCI Hydraulic | Hydraulic system condition |
| `tep_fetcher.py` | Tennessee Eastman | Chemical process faults |

### Scientific / Simulation

| Fetcher | Source | Domain |
|---------|--------|--------|
| `chemked_fetcher.py` | ChemKED Database | Combustion kinetics |
| `sabiork_fetcher.py` | SABIO-RK API | Enzyme kinetics |
| `the_well_fetcher.py` | The Well | PDE simulations |
| `the_well_pde_fetcher.py` | The Well | Gray-Scott patterns |

### Clinical / Physiological

| Fetcher | Source | Domain |
|---------|--------|--------|
| `physionet_fetcher.py` | PhysioNet | ECG, clinical signals |
| `mimic_fetcher.py` | MIMIC-IV | ICU vital signs |

### Environmental

| Fetcher | Source | Domain |
|---------|--------|--------|
| `usgs_fetcher.py` | USGS | Earthquake catalog |
| `climate_fetcher.py` | NOAA/NASA | Climate data |
| `ecology_fetcher.py` | Various | Ecological time series |
| `epidemiology_fetcher.py` | CDC/WHO | Disease surveillance |

## Usage

### Direct Python

```python
from fetchers.cmapss_fetcher import fetch

config = {
    "data_dir": "data/C_MAPSS",
    "subset": "FD001",
}
observations = fetch(config)
```

### Via PRISM Fetch Entry Point

```bash
python -m prism.db.fetch --cmapss
python -m prism.db.fetch --femto
python -m prism.db.fetch --hydraulic
```

## Return Format

All fetchers return `list[dict]` with these keys:
- `signal_id`: str
- `observed_at`: datetime
- `value`: float
- `source`: str

## YAML Configuration

Put fetch configs in `fetchers/yaml/`:

```yaml
# fetchers/yaml/cmapss.yaml
source: cmapss
data_dir: data/C_MAPSS
subset: FD001
```

## Adding a New Fetcher

1. Create `{source}_fetcher.py` in this directory
2. Implement `fetch(config) -> list[dict]`
3. Create `yaml/{source}.yaml` with configuration
4. Do NOT import anything from `prism`
