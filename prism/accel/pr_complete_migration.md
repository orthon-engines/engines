# PR: Complete Migration — modules/ → engines/ + core/

**Branch:** `main` (continuing from v2 architecture)
**Type:** Final Cleanup
**Priority:** Complete before RMSE validation run

---

## Summary

Architecture v2 created `engines/pointwise/` and `engines/windowed/`. This PR completes the migration by moving remaining computation from `modules/` to `engines/` and creating `core/` for types/utilities.

**The Rule:**
- `engines/` = takes data, returns computed values
- `core/` = types, utilities, no computation

---

## Current State (After v2)

```
prism/
├── engines/
│   ├── pointwise/      ✓ NEW
│   ├── windowed/       ✓ NEW
│   ├── geometry/       ✓ EXISTS
│   ├── state/          ✓ EXISTS
│   └── characterize.py ✓ NEW
│
├── modules/            ← STILL EXISTS (needs cleanup)
│   ├── laplace_transform.py
│   ├── laplace_pairwise.py
│   ├── modes.py
│   ├── wavelet_microscope.py
│   ├── prefilter.py
│   ├── characterize.py     ← DUPLICATE!
│   ├── domain_clock.py
│   ├── signals/
│   ├── geometry/           ← DUPLICATE DIRECTORY
│   └── state/              ← DUPLICATE DIRECTORY
```

---

## Target State

```
prism/
├── core/                        ← NEW (types only)
│   ├── __init__.py
│   ├── signals/
│   │   ├── __init__.py
│   │   └── types.py
│   └── domain_clock.py
│
├── engines/                     ← ALL computation
│   ├── __init__.py              (update exports)
│   ├── characterize.py          ✓ (keep, delete modules/ version)
│   ├── prefilter.py             ← MOVE
│   │
│   ├── pointwise/               ✓ EXISTS
│   ├── windowed/                ✓ EXISTS
│   │
│   ├── laplace/                 ← NEW DIRECTORY
│   │   ├── __init__.py
│   │   ├── transform.py         ← MOVE (was laplace_transform.py)
│   │   └── pairwise.py          ← MOVE (was laplace_pairwise.py)
│   │
│   ├── spectral/                ← NEW DIRECTORY
│   │   ├── __init__.py
│   │   └── wavelet.py           ← MOVE (was wavelet_microscope.py)
│   │
│   ├── geometry/                ✓ EXISTS (merge modules/geometry/)
│   │   └── modes.py             ← MOVE
│   │
│   └── state/                   ✓ EXISTS (merge modules/state/)
│
├── modules/                     ← DELETE AFTER MIGRATION
```

---

## Phase 1: Create New Directories

```bash
# Create core/ structure
mkdir -p prism/core/signals

# Create new engine directories
mkdir -p prism/engines/laplace
mkdir -p prism/engines/spectral
```

---

## Phase 2: Move Files

### 2.1 Laplace Engines (NEW)

```bash
# Move laplace computation to engines/laplace/
mv prism/modules/laplace_transform.py prism/engines/laplace/transform.py
mv prism/modules/laplace_pairwise.py prism/engines/laplace/pairwise.py
```

### 2.2 Spectral Engines (NEW)

```bash
# Move wavelet to engines/spectral/
mv prism/modules/wavelet_microscope.py prism/engines/spectral/wavelet.py
```

### 2.3 Geometry (MERGE)

```bash
# Move modes.py to existing engines/geometry/
mv prism/modules/modes.py prism/engines/geometry/modes.py

# Check for other files in modules/geometry/ that need merging
ls prism/modules/geometry/
# If files exist, merge them carefully (avoid overwrites)
# cp -n prism/modules/geometry/*.py prism/engines/geometry/
```

### 2.4 State (MERGE)

```bash
# Check for files in modules/state/ that need merging
ls prism/modules/state/
# If files exist, merge them carefully
# cp -n prism/modules/state/*.py prism/engines/state/
```

### 2.5 Prefilter

```bash
mv prism/modules/prefilter.py prism/engines/prefilter.py
```

### 2.6 Core (Types/Utilities)

```bash
# Move signals types to core/
mv prism/modules/signals prism/core/signals

# Move domain_clock utility to core/
mv prism/modules/domain_clock.py prism/core/domain_clock.py
```

### 2.7 Handle Duplicate characterize.py

```bash
# Compare the two versions
diff prism/modules/characterize.py prism/engines/characterize.py

# If engines/ version is complete, delete modules/ version
# If modules/ has additional functionality, merge it into engines/
rm prism/modules/characterize.py  # After verification
```

---

## Phase 3: Create __init__.py Files

### prism/core/__init__.py

```python
"""
PRISM Core — Types and utilities.

NOT computation. For computation, see prism.engines.
"""

from prism.core.signals import DenseSignal, SparseSignal, LaplaceField
from prism.core.domain_clock import DomainClock, detect_domain_frequency

__all__ = [
    'DenseSignal',
    'SparseSignal', 
    'LaplaceField',
    'DomainClock',
    'detect_domain_frequency',
]
```

### prism/core/signals/__init__.py

```python
"""
PRISM Signal Types — Data structures for signal representation.
"""

from prism.core.signals.types import (
    DenseSignal,
    SparseSignal,
    LaplaceField,
)

__all__ = [
    'DenseSignal',
    'SparseSignal',
    'LaplaceField',
]
```

### prism/engines/laplace/__init__.py

```python
"""
PRISM Laplace Engines — Laplace domain computation.

The Laplace transform resolves time-varying metrics to field vectors,
enabling direct cross-signal comparison regardless of scale or sampling frequency.
"""

from prism.engines.laplace.transform import (
    compute_laplace_for_series,
    compute_gradient,
    compute_laplacian,
    compute_divergence_for_signal,
)
from prism.engines.laplace.pairwise import (
    compute_pairwise_laplace,
    compute_laplace_coupling,
)

__all__ = [
    # Transform
    'compute_laplace_for_series',
    'compute_gradient',
    'compute_laplacian',
    'compute_divergence_for_signal',
    # Pairwise
    'compute_pairwise_laplace',
    'compute_laplace_coupling',
]
```

### prism/engines/spectral/__init__.py

```python
"""
PRISM Spectral Engines — Frequency domain analysis.

Wavelet microscope for multi-scale degradation detection.
"""

from prism.engines.spectral.wavelet import (
    compute_wavelet_decomposition,
    compute_band_snr_evolution,
    identify_degradation_band,
    run_wavelet_microscope,
    extract_wavelet_features,
)

__all__ = [
    'compute_wavelet_decomposition',
    'compute_band_snr_evolution',
    'identify_degradation_band',
    'run_wavelet_microscope',
    'extract_wavelet_features',
]
```

---

## Phase 4: Update Imports

### Find all affected imports

```bash
# Search for all imports from modules/
grep -r "from prism.modules" prism/ --include="*.py"
grep -r "from prism\.modules" prism/ --include="*.py"
```

### Import Replacement Table

| Old Import | New Import |
|------------|------------|
| `from prism.modules.laplace_transform import X` | `from prism.engines.laplace.transform import X` |
| `from prism.modules.laplace_pairwise import X` | `from prism.engines.laplace.pairwise import X` |
| `from prism.modules.laplace import X` | `from prism.engines.laplace import X` |
| `from prism.modules.modes import X` | `from prism.engines.geometry.modes import X` |
| `from prism.modules.wavelet_microscope import X` | `from prism.engines.spectral.wavelet import X` |
| `from prism.modules.prefilter import X` | `from prism.engines.prefilter import X` |
| `from prism.modules.characterize import X` | `from prism.engines.characterize import X` |
| `from prism.modules.signals import X` | `from prism.core.signals import X` |
| `from prism.modules.domain_clock import X` | `from prism.core.domain_clock import X` |
| `from prism.modules.geometry import X` | `from prism.engines.geometry import X` |
| `from prism.modules.state import X` | `from prism.engines.state import X` |
| `from prism.modules import X` | `from prism.engines import X` or `from prism.core import X` |

### Key Files to Update

These entry points likely have the most imports to fix:

```bash
# Check these files specifically
grep -l "from prism.modules" prism/entry_points/*.py
```

Expected files:
- `prism/entry_points/signal_vector.py`
- `prism/entry_points/geometry.py`
- `prism/entry_points/laplace.py`
- `prism/entry_points/state.py`

---

## Phase 5: Update engines/__init__.py

Add new subpackages to the main engines export:

```python
# Add to prism/engines/__init__.py

# Laplace engines
from prism.engines.laplace import (
    compute_laplace_for_series,
    compute_gradient,
    compute_laplacian,
    compute_pairwise_laplace,
)

# Spectral engines  
from prism.engines.spectral import (
    compute_wavelet_decomposition,
    run_wavelet_microscope,
)

# Geometry additions
from prism.engines.geometry.modes import (
    discover_modes,
    compute_mode_scores,
    compute_affinity_weighted_features,
)

# Update __all__ to include new exports
```

---

## Phase 6: Delete modules/

After all moves and import updates are verified:

```bash
# Verify modules/ is empty or only has __init__.py
ls -la prism/modules/

# Remove the directory
rm -rf prism/modules/
```

---

## Phase 7: Verification

### Test imports work

```bash
# Core types
python -c "from prism.core.signals import DenseSignal, SparseSignal"
python -c "from prism.core.domain_clock import DomainClock"

# Laplace engines
python -c "from prism.engines.laplace import compute_laplace_for_series"
python -c "from prism.engines.laplace.transform import compute_gradient"
python -c "from prism.engines.laplace.pairwise import compute_pairwise_laplace"

# Spectral engines
python -c "from prism.engines.spectral import compute_wavelet_decomposition"
python -c "from prism.engines.spectral.wavelet import run_wavelet_microscope"

# Geometry (modes)
python -c "from prism.engines.geometry.modes import discover_modes"

# Prefilter
python -c "from prism.engines.prefilter import laplacian_prefilter"

# Characterize (single location now)
python -c "from prism.engines.characterize import characterize_signal"
```

### Verify modules/ is gone

```bash
test ! -d prism/modules && echo "modules/ deleted ✓"
```

### Run entry points

```bash
# Quick smoke test
python -m prism.entry_points.signal_vector --help
python -m prism.entry_points.geometry --help
python -m prism.entry_points.laplace --help
python -m prism.entry_points.state --help
```

---

## Checklist

- [ ] Create `prism/core/` directory
- [ ] Create `prism/core/signals/` directory
- [ ] Create `prism/engines/laplace/` directory
- [ ] Create `prism/engines/spectral/` directory
- [ ] Move `laplace_transform.py` → `engines/laplace/transform.py`
- [ ] Move `laplace_pairwise.py` → `engines/laplace/pairwise.py`
- [ ] Move `wavelet_microscope.py` → `engines/spectral/wavelet.py`
- [ ] Move `modes.py` → `engines/geometry/modes.py`
- [ ] Move `prefilter.py` → `engines/prefilter.py`
- [ ] Move `signals/` → `core/signals/`
- [ ] Move `domain_clock.py` → `core/domain_clock.py`
- [ ] Merge `modules/geometry/` into `engines/geometry/` (if files exist)
- [ ] Merge `modules/state/` into `engines/state/` (if files exist)
- [ ] Reconcile duplicate `characterize.py` (delete modules/ version)
- [ ] Create all `__init__.py` files
- [ ] Update ALL imports (use grep to find them all)
- [ ] Update `engines/__init__.py` exports
- [ ] Delete `prism/modules/`
- [ ] Verify all imports work
- [ ] Run entry point smoke tests

---

## Final Structure

```
prism/
├── config/                      # Config loading
├── core/                        # Types and utilities ONLY
│   ├── __init__.py
│   ├── signals/
│   │   ├── __init__.py
│   │   └── types.py
│   └── domain_clock.py
├── db/                          # Parquet I/O
├── engines/                     # ALL computation
│   ├── __init__.py
│   ├── characterize.py
│   ├── prefilter.py
│   ├── pointwise/               # Native resolution
│   ├── windowed/                # Sparse/window-based
│   ├── laplace/                 # Laplace field computation
│   │   ├── __init__.py
│   │   ├── transform.py
│   │   └── pairwise.py
│   ├── spectral/                # Frequency domain
│   │   ├── __init__.py
│   │   └── wavelet.py
│   ├── geometry/                # Geometry computation
│   │   ├── __init__.py
│   │   ├── modes.py
│   │   └── ...
│   └── state/                   # State computation
│       ├── __init__.py
│       └── ...
├── entry_points/                # CLI wrappers
└── utils/                       # Helpers
```

---

## The Rules (Final)

```
If it COMPUTES        → engines/
If it DEFINES TYPES   → core/
If it's a CLI WRAPPER → entry_points/
If it's a HELPER      → utils/
```

**modules/ no longer exists.**

---

## Notes for Claude Code

1. **Check before overwriting** - Some files may already exist in target locations
2. **Merge carefully** - `modules/geometry/` and `modules/state/` may have files that need merging with existing `engines/` directories
3. **grep is your friend** - Find all imports before making changes
4. **Test incrementally** - After each phase, verify imports still work
5. **The duplicate characterize.py** - Compare both versions, keep the more complete one in `engines/`

*Clean separation. No ambiguity. More cowbell.* 🔔
