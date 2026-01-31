"""
PRISM Python Runner

Executes engines from python/ and python_windowed/ folders.
PARALLEL PROCESSING enabled for windowed engines.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
import importlib
import warnings
import multiprocessing as mp
from functools import partial
import os

warnings.filterwarnings('ignore')

# Use all available cores, leave 1 for system
N_WORKERS = max(1, mp.cpu_count() - 1)

# Engine registries
# Note: lyapunov removed - it's a dynamical systems metric computed in dynamics_runner
SIGNAL_ENGINES = [
    'rms', 'peak', 'crest_factor', 'kurtosis', 'skewness',
    'envelope', 'harmonics', 'frequency_bands', 'spectral',
    'hurst', 'entropy', 'garch', 'attractor', 'dmd',
    'pulsation_index', 'rate_of_change', 'time_constant',
    'cycle_counting', 'basin', 'lof'
]

PAIR_ENGINES = [
    'granger', 'transfer_entropy'
]

SYMMETRIC_PAIR_ENGINES = [
    'cointegration', 'mutual_info', 'correlation'
]

WINDOWED_ENGINES = [
    'derivatives', 'manifold', 'stability',
    'rolling_rms', 'rolling_kurtosis', 'rolling_entropy',
    'rolling_hurst', 'rolling_volatility', 'rolling_mean', 'rolling_std',
    'rolling_crest_factor', 'rolling_envelope', 'rolling_range',
    'rolling_pulsation', 'rolling_skewness', 'rolling_lyapunov'
]


def load_engine(engine_name: str, engine_type: str):
    """Dynamically load an engine module."""
    if engine_type == 'windowed':
        module_path = f'prism.engines.rolling.{engine_name}'
    else:
        module_path = f'prism.engines.signal.{engine_name}'

    try:
        module = importlib.import_module(module_path)
        return module.compute
    except (ImportError, AttributeError) as e:
        return None


def _process_single_signal(args):
    """
    Process a single signal with all windowed engines.
    Standalone function for multiprocessing compatibility.
    """
    entity, signal, value, I, df_dict, engines_to_run, params = args

    # Reconstruct DataFrame from dict
    df = pd.DataFrame(df_dict)

    if len(value) < 10:
        return None

    # Load engines in worker process
    engine_funcs = {}
    for name in engines_to_run:
        func = load_engine(name, 'windowed')
        if func:
            engine_funcs[name] = func

    for name, func in engine_funcs.items():
        try:
            engine_params = params.get(name, {})

            # Derivatives needs I
            if name == 'derivatives':
                result = func(value, I, engine_params)
            # Stability needs derivatives first
            elif name == 'stability':
                from prism.engines.rolling import derivatives
                deriv_params = params.get('derivatives', {})
                deriv = derivatives.compute(value, I, deriv_params)
                result = func(value, deriv['dy'], deriv['d2y'], engine_params)
            else:
                # All other windowed engines: func(y, params)
                result = func(value, engine_params)

            for col, vals in result.items():
                df[col] = vals
        except Exception:
            pass

    return df


class PythonRunner:
    """
    Executes Python-based engines.

    Handles:
    - Signal-level engines (one value per signal)
    - Pair engines (directional A→B)
    - Symmetric pair engines (A↔B)
    - Windowed engines (observation-level) — PARALLEL
    """

    def __init__(
        self,
        obs: pd.DataFrame,
        output_dir: Path,
        engines: Dict[str, List[str]],
        params: Dict[str, Any] = None
    ):
        self.obs = obs
        self.output_dir = Path(output_dir)
        self.engines = engines
        self.params = params or {}

        # Index signals
        self.signal_data: Dict[Tuple[str, str], Dict] = {}
        self._index_signals()

        # Results storage
        self.observations_enriched: List[pd.DataFrame] = []
        self.primitives: List[Dict] = []
        self.primitives_pairs: List[Dict] = []
        self.geometry_pairs: List[Dict] = []
        self.manifold_df = pd.DataFrame()

    def _index_signals(self):
        """Index all signals by (unit_id, signal_id)."""
        for (entity, signal), group in self.obs.groupby(['unit_id', 'signal_id']):
            sorted_group = group.sort_values('I')
            self.signal_data[(entity, signal)] = {
                'I': sorted_group['I'].values,
                'value': sorted_group['value'].values,
                'unit': sorted_group['unit'].iloc[0] if 'unit' in sorted_group.columns else None,
                'df': sorted_group,
            }

        self.entities = list(set(k[0] for k in self.signal_data.keys()))
        self.signals_by_entity = {
            e: [k[1] for k in self.signal_data.keys() if k[0] == e]
            for e in self.entities
        }

        print(f"  Indexed {len(self.signal_data)} signals across {len(self.entities)} entities")

    def run(self) -> dict:
        """Execute ALL Python engines. Primitives first, then windowed."""
        self.run_signal_engines()      # 1. Primitives (signal-level)
        self.run_pair_engines()        # 2. Pair primitives
        self.run_symmetric_pair_engines()  # 3. Symmetric pairs
        self.run_windowed_engines()    # 4. Windowed (may need primitives)
        self.export()

        return {
            'signals_processed': len(self.primitives),
            'pairs_processed': len(self.primitives_pairs),
            'symmetric_pairs_processed': len(self.geometry_pairs),
            'observations_enriched': len(self.observations_enriched)
        }

    def run_signal_engines(self):
        """Run ALL signal-level engines. No exceptions."""
        engines_to_run = SIGNAL_ENGINES  # ALL engines, always

        print(f"\n[SIGNAL ENGINES] Running ALL: {engines_to_run}")

        # Load engine functions
        engine_funcs = {}
        for name in engines_to_run:
            func = load_engine(name, 'signal')
            if func:
                engine_funcs[name] = func

        for (entity, signal), data in self.signal_data.items():
            y = data['value']
            I = data['I']
            unit = data['unit']

            if len(y) < 10:
                continue

            # Identity only - no calculations in runner
            row = {
                'unit_id': entity,
                'signal_id': signal,
                'unit': unit,
                'n_points': len(y),
            }

            # Run each engine - engines compute everything
            for name, func in engine_funcs.items():
                try:
                    params = self.params.get(name, {})

                    # Some engines need I
                    if name in ['rate_of_change', 'time_constant']:
                        result = func(y, I, **params) if params else func(y, I)
                    elif params:
                        result = func(y, **params)
                    else:
                        result = func(y)

                    row.update(result)
                except Exception:
                    pass  # Engine failed → NaN implicitly

            self.primitives.append(row)

        print(f"  Processed {len(self.primitives)} signals")

    def run_pair_engines(self):
        """Run ALL pair-level engines. No exceptions."""
        engines_to_run = PAIR_ENGINES  # ALL engines, always

        print(f"\n[PAIR ENGINES] Running ALL: {engines_to_run}")

        # Load engine functions
        engine_funcs = {}
        for name in engines_to_run:
            func = load_engine(name, 'pair')
            if func:
                engine_funcs[name] = func

        for entity in self.entities:
            signals = self.signals_by_entity[entity]

            for i, sig_a in enumerate(signals):
                for sig_b in signals[i+1:]:
                    data_a = self.signal_data[(entity, sig_a)]
                    data_b = self.signal_data[(entity, sig_b)]

                    y_a, y_b = data_a['value'], data_b['value']
                    n = min(len(y_a), len(y_b))
                    if n < 20:
                        continue

                    y_a, y_b = y_a[:n], y_b[:n]

                    # A → B
                    row_ab = {'unit_id': entity, 'source_signal': sig_a, 'target_signal': sig_b}
                    for name, func in engine_funcs.items():
                        try:
                            params = self.params.get(name, {})
                            result = func(y_a, y_b, **params) if params else func(y_a, y_b)
                            row_ab.update(result)
                        except Exception:
                            pass
                    self.primitives_pairs.append(row_ab)

                    # B → A
                    row_ba = {'unit_id': entity, 'source_signal': sig_b, 'target_signal': sig_a}
                    for name, func in engine_funcs.items():
                        try:
                            params = self.params.get(name, {})
                            result = func(y_b, y_a, **params) if params else func(y_b, y_a)
                            row_ba.update(result)
                        except Exception:
                            pass
                    self.primitives_pairs.append(row_ba)

        print(f"  Processed {len(self.primitives_pairs)} directional pairs")

    def run_symmetric_pair_engines(self):
        """Run ALL symmetric pair engines. No exceptions."""
        engines_to_run = SYMMETRIC_PAIR_ENGINES  # ALL engines, always

        print(f"\n[SYMMETRIC PAIR ENGINES] Running ALL: {engines_to_run}")

        # Load engine functions
        engine_funcs = {}
        for name in engines_to_run:
            func = load_engine(name, 'symmetric_pair')
            if func:
                engine_funcs[name] = func

        for entity in self.entities:
            signals = self.signals_by_entity[entity]

            for i, sig_a in enumerate(signals):
                for sig_b in signals[i+1:]:
                    data_a = self.signal_data[(entity, sig_a)]
                    data_b = self.signal_data[(entity, sig_b)]

                    y_a, y_b = data_a['value'], data_b['value']
                    n = min(len(y_a), len(y_b))
                    if n < 50:
                        continue

                    y_a, y_b = y_a[:n], y_b[:n]

                    row = {
                        'unit_id': entity,
                        'signal_a': sig_a,
                        'signal_b': sig_b,
                    }

                    for name, func in engine_funcs.items():
                        try:
                            params = self.params.get(name, {})
                            result = func(y_a, y_b, **params) if params else func(y_a, y_b)
                            row.update(result)
                        except Exception:
                            pass

                    self.geometry_pairs.append(row)

        print(f"  Processed {len(self.geometry_pairs)} symmetric pairs")

    def run_windowed_engines(self):
        """Run ALL windowed engines in PARALLEL. No exceptions."""
        engines_to_run = list(WINDOWED_ENGINES)  # ALL engines, always

        print(f"\n[WINDOWED ENGINES] Running ALL (parallel, {N_WORKERS} workers): {engines_to_run}")

        # Special case for manifold (cross-signal, not parallelizable per-signal)
        if 'manifold' in engines_to_run:
            try:
                from prism.engines.rolling import manifold
                manifold_params = self.params.get('manifold', {})
                self.manifold_df = manifold.compute(self.obs, manifold_params)
                print(f"  Manifold: {len(self.manifold_df):,} points")
            except Exception as e:
                print(f"  Manifold failed: {e}")
            engines_to_run = [e for e in engines_to_run if e != 'manifold']

        # Prepare args for parallel processing
        work_items = []
        for (entity, signal), data in self.signal_data.items():
            # Convert DataFrame to dict for pickling
            df_dict = data['df'].to_dict('list')
            work_items.append((
                entity,
                signal,
                data['value'],
                data['I'],
                df_dict,
                engines_to_run,
                self.params
            ))

        total = len(work_items)
        print(f"  Processing {total} signals across {N_WORKERS} workers...")

        # Process in parallel
        completed = 0
        with mp.Pool(N_WORKERS) as pool:
            for result in pool.imap_unordered(_process_single_signal, work_items):
                if result is not None:
                    self.observations_enriched.append(result)
                completed += 1
                if completed % max(1, total // 10) == 0:
                    print(f"    Progress: {completed}/{total} ({100*completed/total:.0f}%)", flush=True)

        print(f"  Processed {len(self.observations_enriched)} signals")

    def export(self):
        """Export all results to parquet files."""
        print("\n[PYTHON EXPORT]")

        # observations_enriched.parquet
        if self.observations_enriched:
            df = pd.concat(self.observations_enriched, ignore_index=True)
            path = self.output_dir / 'observations_enriched.parquet'
            df.to_parquet(path, index=False)
            print(f"  observations_enriched.parquet: {len(df):,} rows × {len(df.columns)} cols")

        # manifold.parquet
        if len(self.manifold_df) > 0:
            path = self.output_dir / 'manifold.parquet'
            self.manifold_df.to_parquet(path, index=False)
            print(f"  manifold.parquet: {len(self.manifold_df):,} rows")

        # primitives.parquet
        if self.primitives:
            df = pd.DataFrame(self.primitives)
            path = self.output_dir / 'primitives.parquet'
            df.to_parquet(path, index=False)
            print(f"  primitives.parquet: {len(df)} rows × {len(df.columns)} cols")

        # primitives_pairs.parquet
        if self.primitives_pairs:
            df = pd.DataFrame(self.primitives_pairs)
            path = self.output_dir / 'primitives_pairs.parquet'
            df.to_parquet(path, index=False)
            print(f"  primitives_pairs.parquet: {len(df)} rows")

        # geometry.parquet
        if self.geometry_pairs:
            df = pd.DataFrame(self.geometry_pairs)
            path = self.output_dir / 'geometry.parquet'
            df.to_parquet(path, index=False)
            print(f"  geometry.parquet: {len(df)} rows")
