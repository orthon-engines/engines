"""
PRISM LSTM Training - Test if PRISM features help LSTM performance.

Compares:
1. Raw sensors → LSTM (baseline)
2. PRISM features → LSTM
3. Raw + PRISM → LSTM

Usage:
    python lstm_train.py --data FD001
    python lstm_train.py --data FD002 --epochs 100
"""

import argparse
from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error


# =============================================================================
# DATASET
# =============================================================================

class RULDataset(Dataset):
    """Dataset for RUL prediction with sequence data."""

    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


def create_sequences(
    df: pl.DataFrame,
    feature_cols: List[str],
    entity_col: str = 'unit_id',
    time_col: str = 'cycle',
    target_col: str = 'RUL',
    seq_length: int = 30,
) -> Tuple[np.ndarray, np.ndarray, list]:
    """Create sliding window sequences for LSTM training."""

    sequences = []
    targets = []
    entity_ids = []

    for entity in df[entity_col].unique().sort().to_list():
        entity_df = df.filter(pl.col(entity_col) == entity).sort(time_col)

        features = entity_df.select(feature_cols).to_numpy()
        rul = entity_df[target_col].to_numpy()

        for i in range(len(features) - seq_length + 1):
            sequences.append(features[i:i + seq_length])
            targets.append(rul[i + seq_length - 1])
            entity_ids.append(entity)

    return np.array(sequences), np.array(targets), entity_ids


def create_test_sequences(
    df: pl.DataFrame,
    feature_cols: List[str],
    entity_col: str = 'unit_id',
    time_col: str = 'cycle',
    seq_length: int = 30,
) -> Tuple[np.ndarray, list]:
    """Create test sequences (last seq_length cycles per entity)."""

    sequences = []
    entity_ids = []

    for entity in df[entity_col].unique().sort().to_list():
        entity_df = df.filter(pl.col(entity_col) == entity).sort(time_col)

        features = entity_df.select(feature_cols).to_numpy()

        if len(features) >= seq_length:
            seq = features[-seq_length:]
        else:
            pad_length = seq_length - len(features)
            padding = np.tile(features[0], (pad_length, 1))
            seq = np.vstack([padding, features])

        sequences.append(seq)
        entity_ids.append(entity)

    return np.array(sequences), entity_ids


# =============================================================================
# MODEL
# =============================================================================

class Attention(nn.Module):
    """Attention mechanism for LSTM output."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, lstm_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # lstm_output: (batch, seq_len, hidden_size)
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        context = torch.sum(attention_weights * lstm_output, dim=1)
        return context, attention_weights


class LSTMAttentionModel(nn.Module):
    """Bidirectional LSTM with Attention for RUL prediction."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )

        self.attention = Attention(hidden_size * 2)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        context, _ = self.attention(lstm_out)
        return self.fc(context).squeeze(-1)


# =============================================================================
# TRAINING
# =============================================================================

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    lr: float = 0.001,
    device: str = 'cpu',
    patience: int = 15,
) -> nn.Module:
    """Train LSTM with early stopping."""

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for sequences, targets in train_loader:
            sequences, targets = sequences.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences, targets = sequences.to(device), targets.to(device)
                outputs = model(sequences)
                val_loss += criterion(outputs, targets).item()

        val_loss /= len(val_loader)
        val_rmse = np.sqrt(val_loss)

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val RMSE: {val_rmse:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    return model


def evaluate_model(
    model: nn.Module,
    sequences: np.ndarray,
    targets: np.ndarray,
    device: str = 'cpu',
) -> Tuple[float, float, np.ndarray]:
    """Evaluate model on test set."""

    model.eval()
    with torch.no_grad():
        sequences_tensor = torch.FloatTensor(sequences).to(device)
        predictions = model(sequences_tensor).cpu().numpy()

    rmse = np.sqrt(mean_squared_error(targets, predictions))
    mae = mean_absolute_error(targets, predictions)

    return rmse, mae, predictions


# =============================================================================
# DATA LOADING
# =============================================================================

def load_cmapss_raw(data_dir: str, dataset: str) -> Tuple[pl.DataFrame, pl.DataFrame, np.ndarray]:
    """Load raw C-MAPSS data."""

    cols = ['unit_id', 'cycle'] + [f'op_{i}' for i in range(1, 4)] + [f's_{i}' for i in range(1, 22)]

    train_path = Path(data_dir) / f'train_{dataset}.txt'
    test_path = Path(data_dir) / f'test_{dataset}.txt'
    rul_path = Path(data_dir) / f'RUL_{dataset}.txt'

    train_df = pl.read_csv(train_path, separator=' ', has_header=False,
                           new_columns=cols, truncate_ragged_lines=True)
    test_df = pl.read_csv(test_path, separator=' ', has_header=False,
                          new_columns=cols, truncate_ragged_lines=True)

    # Add RUL to train
    max_cycles = train_df.group_by('unit_id').agg(pl.col('cycle').max().alias('max_cycle'))
    train_df = train_df.join(max_cycles, on='unit_id')
    train_df = train_df.with_columns((pl.col('max_cycle') - pl.col('cycle')).alias('RUL'))
    train_df = train_df.drop('max_cycle')

    # Cap RUL at 125 (standard practice)
    train_df = train_df.with_columns(pl.col('RUL').clip(upper_bound=125))

    # Load test RUL
    with open(rul_path, 'r') as f:
        test_rul = np.array([float(line.strip()) for line in f if line.strip()])
    test_rul = np.clip(test_rul, 0, 125)

    return train_df, test_df, test_rul


def load_prism_features(data_dir: str) -> Tuple[Optional[pl.DataFrame], Optional[pl.DataFrame], Optional[pl.DataFrame]]:
    """Load PRISM parquet files if they exist."""

    data_path = Path(data_dir)

    vector_df = None
    geometry_df = None
    state_df = None

    # Try various naming conventions
    for vec_name in ['vector.parquet', 'train_vector.parquet']:
        if (data_path / vec_name).exists():
            vector_df = pl.read_parquet(data_path / vec_name)
            break

    for geo_name in ['geometry.parquet', 'train_geometry.parquet']:
        if (data_path / geo_name).exists():
            geometry_df = pl.read_parquet(data_path / geo_name)
            break

    for state_name in ['state.parquet', 'train_state.parquet']:
        if (data_path / state_name).exists():
            state_df = pl.read_parquet(data_path / state_name)
            break

    return vector_df, geometry_df, state_df


def merge_prism_features(
    raw_df: pl.DataFrame,
    vector_df: Optional[pl.DataFrame],
    geometry_df: Optional[pl.DataFrame],
    state_df: Optional[pl.DataFrame],
) -> Tuple[pl.DataFrame, List[str]]:
    """Merge PRISM features with raw data."""

    result = raw_df.clone()
    prism_cols = []

    # Pivot and merge vector features
    if vector_df is not None:
        print("  Merging vector features...")

        # Create feature name
        vec = vector_df.with_columns(
            (pl.col('engine') + '_' + pl.col('source_signal')).alias('feature_name')
        )

        # Pivot
        vec_pivot = vec.pivot(
            index=['entity_id', 'timestamp'],
            on='feature_name',
            values='value',
            aggregate_function='first'
        )

        # Extract unit_id from entity_id (e.g., "FD002_U029" -> 29)
        vec_pivot = vec_pivot.with_columns([
            pl.col('entity_id').str.extract(r'U(\d+)', 1).cast(pl.Int64).alias('unit_id'),
            pl.col('timestamp').cast(pl.Int64).alias('cycle'),
        ]).drop(['entity_id', 'timestamp'])

        vec_cols = [c for c in vec_pivot.columns if c not in ['unit_id', 'cycle']]
        prism_cols.extend(vec_cols)

        result = result.join(vec_pivot, on=['unit_id', 'cycle'], how='left')
        print(f"    Added {len(vec_cols)} vector features")

    # Merge geometry features
    if geometry_df is not None:
        print("  Merging geometry features...")

        geo_cols = [c for c in geometry_df.columns
                   if c not in ['entity_id', 'timestamp', 'computed_at', 'signal_ids', 'mode_id', 'n_features', 'n_engines']]

        geo = geometry_df.select(['entity_id', 'timestamp'] + geo_cols)
        geo = geo.with_columns([
            pl.col('entity_id').str.extract(r'U(\d+)', 1).cast(pl.Int64).alias('unit_id'),
            pl.col('timestamp').cast(pl.Int64).alias('cycle'),
        ]).drop(['entity_id', 'timestamp'])

        prism_cols.extend(geo_cols)
        result = result.join(geo, on=['unit_id', 'cycle'], how='left')
        print(f"    Added {len(geo_cols)} geometry features")

    # Merge state features
    if state_df is not None:
        print("  Merging state features...")

        state_cols = [c for c in state_df.columns
                     if c not in ['entity_id', 'timestamp', 'computed_at',
                                  'is_failure_signature', 'mode_transition', 'position_dim',
                                  'mode_id', 'state_label', 'failure_signature']]

        state = state_df.select(['entity_id', 'timestamp'] + state_cols)
        state = state.with_columns([
            pl.col('entity_id').str.extract(r'U(\d+)', 1).cast(pl.Int64).alias('unit_id'),
            pl.col('timestamp').cast(pl.Int64).alias('cycle'),
        ]).drop(['entity_id', 'timestamp'])

        prism_cols.extend(state_cols)
        result = result.join(state, on=['unit_id', 'cycle'], how='left')
        print(f"    Added {len(state_cols)} state features")

    # Forward fill PRISM features (for early cycles before geometry window)
    if prism_cols:
        result = result.with_columns([
            pl.col(c).forward_fill().over('unit_id') for c in prism_cols if c in result.columns
        ])
        result = result.with_columns([
            pl.col(c).backward_fill().over('unit_id') for c in prism_cols if c in result.columns
        ])

    return result, prism_cols


# =============================================================================
# EXPERIMENTS
# =============================================================================

def run_experiment(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    test_rul: np.ndarray,
    feature_cols: List[str],
    name: str,
    seq_length: int = 30,
    epochs: int = 50,
    batch_size: int = 256,
    device: str = 'cpu',
) -> dict:
    """Run single experiment."""

    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {name}")
    print(f"{'='*60}")

    # Filter to valid columns
    valid_cols = [c for c in feature_cols if c in train_df.columns and c in test_df.columns]

    # Remove columns with >50% null
    keep_cols = []
    for col in valid_cols:
        null_pct = train_df[col].null_count() / len(train_df)
        if null_pct < 0.5:
            keep_cols.append(col)

    feature_cols = keep_cols
    print(f"Features: {len(feature_cols)}")

    if len(feature_cols) == 0:
        print("No valid features!")
        return {'name': name, 'rmse': float('inf'), 'mae': float('inf'), 'n_features': 0}

    # Fill nulls
    train_clean = train_df.with_columns([pl.col(c).fill_null(0) for c in feature_cols])
    test_clean = test_df.with_columns([pl.col(c).fill_null(0) for c in feature_cols])

    # Create sequences
    print("Creating sequences...")
    train_seqs, train_targets, _ = create_sequences(train_clean, feature_cols, seq_length=seq_length)
    test_seqs, _ = create_test_sequences(test_clean, feature_cols, seq_length=seq_length)

    print(f"Train sequences: {train_seqs.shape}")
    print(f"Test sequences: {test_seqs.shape}")

    # Normalize
    n_samples, seq_len, n_features = train_seqs.shape

    scaler = StandardScaler()
    train_flat = scaler.fit_transform(train_seqs.reshape(-1, n_features))
    train_seqs = train_flat.reshape(n_samples, seq_len, n_features)

    test_flat = scaler.transform(test_seqs.reshape(-1, n_features))
    test_seqs = test_flat.reshape(test_seqs.shape[0], seq_len, n_features)

    # Train/val split
    n_train = int(0.8 * len(train_seqs))
    indices = np.random.permutation(len(train_seqs))
    train_idx, val_idx = indices[:n_train], indices[n_train:]

    train_dataset = RULDataset(train_seqs[train_idx], train_targets[train_idx])
    val_dataset = RULDataset(train_seqs[val_idx], train_targets[val_idx])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Model
    model = LSTMAttentionModel(
        input_size=n_features,
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    print("Training...")
    model = train_model(model, train_loader, val_loader, epochs=epochs, device=device)

    # Evaluate
    rmse, mae, predictions = evaluate_model(model, test_seqs, test_rul, device=device)

    print(f"\n{'-'*40}")
    print(f"TEST RESULTS: {name}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")

    return {
        'name': name,
        'rmse': rmse,
        'mae': mae,
        'n_features': len(feature_cols),
        'predictions': predictions,
    }


def main():
    parser = argparse.ArgumentParser(description='LSTM + Attention with PRISM features')
    parser.add_argument('--data', type=str, default='FD001', help='Dataset (FD001, FD002, etc.)')
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--seq-length', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--device', type=str, default='auto', help='cpu/cuda/mps/auto')
    args = parser.parse_args()

    print("="*60)
    print(f"PRISM LSTM COMPARISON - {args.data}")
    print("="*60)

    # Device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device
    print(f"Device: {device}")

    # Seed for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Load data
    print("\nLoading raw data...")
    train_df, test_df, test_rul = load_cmapss_raw(args.data_dir, args.data)
    print(f"Train: {len(train_df):,} rows, {train_df['unit_id'].n_unique()} units")
    print(f"Test: {len(test_df):,} rows, {test_df['unit_id'].n_unique()} units")

    raw_cols = [f'op_{i}' for i in range(1, 4)] + [f's_{i}' for i in range(1, 22)]

    # Load PRISM
    print("\nLoading PRISM features...")
    vector_df, geometry_df, state_df = load_prism_features(args.data_dir)

    has_prism = any([vector_df is not None, geometry_df is not None, state_df is not None])

    if has_prism:
        train_merged, prism_cols = merge_prism_features(train_df, vector_df, geometry_df, state_df)
        test_merged, _ = merge_prism_features(test_df, vector_df, geometry_df, state_df)
        print(f"Total PRISM features: {len(prism_cols)}")
    else:
        print("No PRISM features found")
        train_merged = train_df
        test_merged = test_df
        prism_cols = []

    results = []

    # Experiment 1: Raw only
    r1 = run_experiment(
        train_df, test_df, test_rul,
        feature_cols=raw_cols,
        name="Raw Sensors",
        seq_length=args.seq_length,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
    )
    results.append(r1)

    if has_prism and len(prism_cols) > 0:
        # Experiment 2: PRISM only
        r2 = run_experiment(
            train_merged, test_merged, test_rul,
            feature_cols=prism_cols,
            name="PRISM Only",
            seq_length=args.seq_length,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=device,
        )
        results.append(r2)

        # Experiment 3: Raw + PRISM
        r3 = run_experiment(
            train_merged, test_merged, test_rul,
            feature_cols=raw_cols + prism_cols,
            name="Raw + PRISM",
            seq_length=args.seq_length,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=device,
        )
        results.append(r3)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Model':<25} {'Features':<10} {'RMSE':<10} {'MAE':<10}")
    print("-"*55)
    for r in results:
        print(f"{r['name']:<25} {r['n_features']:<10} {r['rmse']:<10.4f} {r['mae']:<10.4f}")

    if len(results) >= 3:
        baseline = results[0]['rmse']
        combined = results[2]['rmse']
        pct = (baseline - combined) / baseline * 100
        print(f"\nPRISM improvement: {pct:.1f}%")
        if pct > 0:
            print("✓ PRISM features HELP LSTM")
        else:
            print("✗ PRISM features don't improve LSTM")


if __name__ == "__main__":
    main()
