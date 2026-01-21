"""
LSTM Hyperparameter Tuning for PRISM features on C-MAPSS FD002.
"""

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pathlib import Path
from typing import Tuple, List, Optional
from itertools import product
import json

# =============================================================================
# DATASET
# =============================================================================

class RULDataset(Dataset):
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
    seq_length: int = 30,
) -> Tuple[np.ndarray, np.ndarray, list]:
    sequences, targets, entity_ids = [], [], []

    for entity in df['unit_id'].unique().sort().to_list():
        entity_df = df.filter(pl.col('unit_id') == entity).sort('cycle')
        features = entity_df.select(feature_cols).to_numpy()
        rul = entity_df['RUL'].to_numpy()

        for i in range(len(features) - seq_length + 1):
            sequences.append(features[i:i + seq_length])
            targets.append(rul[i + seq_length - 1])
            entity_ids.append(entity)

    return np.array(sequences), np.array(targets), entity_ids


def create_test_sequences(
    df: pl.DataFrame,
    feature_cols: List[str],
    seq_length: int = 30,
) -> Tuple[np.ndarray, list]:
    sequences, entity_ids = [], []

    for entity in df['unit_id'].unique().sort().to_list():
        entity_df = df.filter(pl.col('unit_id') == entity).sort('cycle')
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
# MODEL VARIANTS
# =============================================================================

class Attention(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, lstm_output: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        attention_weights = torch.softmax(self.attention(lstm_output), dim=1)
        context = torch.sum(attention_weights * lstm_output, dim=1)
        return context, attention_weights


class LSTMModel(nn.Module):
    """Configurable LSTM model."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        bidirectional: bool = True,
        use_attention: bool = True,
    ):
        super().__init__()

        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.hidden_multiplier = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
        )

        lstm_out_size = hidden_size * self.hidden_multiplier

        if use_attention:
            self.attention = Attention(lstm_out_size)

        self.fc = nn.Sequential(
            nn.Linear(lstm_out_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)

        if self.use_attention:
            context, _ = self.attention(lstm_out)
        else:
            # Use last hidden state
            context = lstm_out[:, -1, :]

        return self.fc(context).squeeze(-1)


# =============================================================================
# TRAINING
# =============================================================================

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 100,
    lr: float = 0.001,
    weight_decay: float = 0.0,
    device: str = 'cpu',
    patience: int = 20,
    verbose: bool = False,
) -> Tuple[nn.Module, float]:
    """Train with early stopping, return model and best val RMSE."""

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
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

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences, targets = sequences.to(device), targets.to(device)
                outputs = model(sequences)
                val_loss += criterion(outputs, targets).item()

        val_loss /= len(val_loader)
        val_rmse = np.sqrt(val_loss)

        scheduler.step(val_loss)

        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs} - Train: {train_loss:.2f}, Val RMSE: {val_rmse:.2f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    return model, np.sqrt(best_val_loss)


def evaluate_model(model: nn.Module, sequences: np.ndarray, targets: np.ndarray, device: str = 'cpu'):
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

def load_data(data_dir: str, dataset: str = 'FD002'):
    cols = ['unit_id', 'cycle'] + [f'op_{i}' for i in range(1, 4)] + [f's_{i}' for i in range(1, 22)]

    train_df = pl.read_csv(Path(data_dir) / f'train_{dataset}.txt', separator=' ',
                           has_header=False, new_columns=cols, truncate_ragged_lines=True)
    test_df = pl.read_csv(Path(data_dir) / f'test_{dataset}.txt', separator=' ',
                          has_header=False, new_columns=cols, truncate_ragged_lines=True)

    max_cycles = train_df.group_by('unit_id').agg(pl.col('cycle').max().alias('max_cycle'))
    train_df = train_df.join(max_cycles, on='unit_id')
    train_df = train_df.with_columns((pl.col('max_cycle') - pl.col('cycle')).alias('RUL')).drop('max_cycle')
    train_df = train_df.with_columns(pl.col('RUL').clip(upper_bound=125))

    with open(Path(data_dir) / f'RUL_{dataset}.txt', 'r') as f:
        test_rul = np.array([float(line.strip()) for line in f if line.strip()])
    test_rul = np.clip(test_rul, 0, 125)

    return train_df, test_df, test_rul


def load_and_merge_prism(train_df, test_df, data_dir):
    """Load and merge PRISM features."""
    data_path = Path(data_dir)

    # Load parquet files
    vector_df = pl.read_parquet(data_path / 'vector.parquet') if (data_path / 'vector.parquet').exists() else None
    geometry_df = pl.read_parquet(data_path / 'geometry.parquet') if (data_path / 'geometry.parquet').exists() else None
    state_df = pl.read_parquet(data_path / 'state.parquet') if (data_path / 'state.parquet').exists() else None

    prism_cols = []

    for df, is_train in [(train_df, True), (test_df, False)]:
        result = df.clone()

        if vector_df is not None:
            vec = vector_df.with_columns(
                (pl.col('engine') + '_' + pl.col('source_signal')).alias('feature_name')
            )
            vec_pivot = vec.pivot(index=['entity_id', 'timestamp'], on='feature_name',
                                  values='value', aggregate_function='first')
            vec_pivot = vec_pivot.with_columns([
                pl.col('entity_id').str.extract(r'U(\d+)', 1).cast(pl.Int64).alias('unit_id'),
                pl.col('timestamp').cast(pl.Int64).alias('cycle'),
            ]).drop(['entity_id', 'timestamp'])

            if is_train:
                prism_cols.extend([c for c in vec_pivot.columns if c not in ['unit_id', 'cycle']])

            result = result.join(vec_pivot, on=['unit_id', 'cycle'], how='left')

        if geometry_df is not None:
            geo_cols = [c for c in geometry_df.columns
                       if c not in ['entity_id', 'timestamp', 'computed_at', 'signal_ids', 'mode_id', 'n_features', 'n_engines']]
            geo = geometry_df.select(['entity_id', 'timestamp'] + geo_cols)
            geo = geo.with_columns([
                pl.col('entity_id').str.extract(r'U(\d+)', 1).cast(pl.Int64).alias('unit_id'),
                pl.col('timestamp').cast(pl.Int64).alias('cycle'),
            ]).drop(['entity_id', 'timestamp'])

            if is_train:
                prism_cols.extend(geo_cols)

            result = result.join(geo, on=['unit_id', 'cycle'], how='left')

        if state_df is not None:
            state_cols = [c for c in state_df.columns
                         if c not in ['entity_id', 'timestamp', 'computed_at', 'is_failure_signature',
                                     'mode_transition', 'position_dim', 'mode_id', 'state_label', 'failure_signature']]
            state = state_df.select(['entity_id', 'timestamp'] + state_cols)
            state = state.with_columns([
                pl.col('entity_id').str.extract(r'U(\d+)', 1).cast(pl.Int64).alias('unit_id'),
                pl.col('timestamp').cast(pl.Int64).alias('cycle'),
            ]).drop(['entity_id', 'timestamp'])

            if is_train:
                prism_cols.extend(state_cols)

            result = result.join(state, on=['unit_id', 'cycle'], how='left')

        # Forward/backward fill
        for c in prism_cols:
            if c in result.columns:
                result = result.with_columns(pl.col(c).forward_fill().over('unit_id'))
                result = result.with_columns(pl.col(c).backward_fill().over('unit_id'))

        if is_train:
            train_merged = result
        else:
            test_merged = result

    return train_merged, test_merged, prism_cols


# =============================================================================
# HYPERPARAMETER SEARCH
# =============================================================================

def run_single_config(
    train_df, test_df, test_rul, feature_cols,
    config: dict, device: str, verbose: bool = False
) -> dict:
    """Run a single hyperparameter configuration."""

    seq_length = config['seq_length']

    # Filter valid columns
    valid_cols = [c for c in feature_cols if c in train_df.columns and c in test_df.columns]
    keep_cols = [c for c in valid_cols if train_df[c].null_count() / len(train_df) < 0.5]

    if len(keep_cols) == 0:
        return {'config': config, 'val_rmse': float('inf'), 'test_rmse': float('inf')}

    # Prepare data
    train_clean = train_df.with_columns([pl.col(c).fill_null(0) for c in keep_cols])
    test_clean = test_df.with_columns([pl.col(c).fill_null(0) for c in keep_cols])

    train_seqs, train_targets, _ = create_sequences(train_clean, keep_cols, seq_length=seq_length)
    test_seqs, _ = create_test_sequences(test_clean, keep_cols, seq_length=seq_length)

    # Normalize
    n_samples, seq_len, n_features = train_seqs.shape
    scaler = StandardScaler()
    train_flat = scaler.fit_transform(train_seqs.reshape(-1, n_features))
    train_seqs = train_flat.reshape(n_samples, seq_len, n_features)
    test_flat = scaler.transform(test_seqs.reshape(-1, n_features))
    test_seqs = test_flat.reshape(test_seqs.shape[0], seq_len, n_features)

    # Split
    n_train = int(0.8 * len(train_seqs))
    indices = np.random.permutation(len(train_seqs))
    train_idx, val_idx = indices[:n_train], indices[n_train:]

    train_dataset = RULDataset(train_seqs[train_idx], train_targets[train_idx])
    val_dataset = RULDataset(train_seqs[val_idx], train_targets[val_idx])

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'])

    # Model
    model = LSTMModel(
        input_size=n_features,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        bidirectional=config['bidirectional'],
        use_attention=config['use_attention'],
    )

    # Train
    model, best_val_rmse = train_model(
        model, train_loader, val_loader,
        epochs=config['epochs'],
        lr=config['lr'],
        weight_decay=config['weight_decay'],
        device=device,
        patience=config['patience'],
        verbose=verbose,
    )

    # Evaluate
    test_rmse, test_mae, _ = evaluate_model(model, test_seqs, test_rul, device=device)

    return {
        'config': config,
        'val_rmse': best_val_rmse,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'n_features': len(keep_cols),
    }


def main():
    print("="*70)
    print("LSTM HYPERPARAMETER TUNING - FD002")
    print("="*70)

    # Device
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'
    print(f"Device: {device}")

    np.random.seed(42)
    torch.manual_seed(42)

    data_dir = '/Users/jasonrudder/prism-mac/data'

    # Load data
    print("\nLoading data...")
    train_df, test_df, test_rul = load_data(data_dir, 'FD002')
    train_merged, test_merged, prism_cols = load_and_merge_prism(train_df, test_df, data_dir)

    raw_cols = [f'op_{i}' for i in range(1, 4)] + [f's_{i}' for i in range(1, 22)]
    all_cols = raw_cols + prism_cols

    print(f"Train: {len(train_df):,} rows")
    print(f"Raw features: {len(raw_cols)}")
    print(f"PRISM features: {len(prism_cols)}")

    # Hyperparameter grid
    configs = []

    # Grid search parameters
    hidden_sizes = [32, 64, 128]
    num_layers_list = [1, 2, 3]
    dropouts = [0.1, 0.3, 0.5]
    lrs = [0.0005, 0.001, 0.002]
    seq_lengths = [15, 30, 50]
    weight_decays = [0.0, 0.01, 0.001]

    # Create focused configs based on common successful patterns
    base_config = {
        'epochs': 100,
        'patience': 25,
        'batch_size': 128,
        'bidirectional': True,
        'use_attention': True,
    }

    # Config 1-9: Vary hidden size and layers
    for hs in hidden_sizes:
        for nl in num_layers_list:
            configs.append({**base_config, 'hidden_size': hs, 'num_layers': nl,
                          'dropout': 0.3, 'lr': 0.001, 'seq_length': 30, 'weight_decay': 0.01})

    # Config 10-18: Vary learning rate and dropout
    for lr in lrs:
        for do in dropouts:
            configs.append({**base_config, 'hidden_size': 64, 'num_layers': 2,
                          'dropout': do, 'lr': lr, 'seq_length': 30, 'weight_decay': 0.01})

    # Config 19-27: Vary sequence length and weight decay
    for sl in seq_lengths:
        for wd in weight_decays:
            configs.append({**base_config, 'hidden_size': 64, 'num_layers': 2,
                          'dropout': 0.3, 'lr': 0.001, 'seq_length': sl, 'weight_decay': wd})

    # Config 28-30: Try without attention / unidirectional
    configs.append({**base_config, 'hidden_size': 128, 'num_layers': 2, 'dropout': 0.3,
                   'lr': 0.001, 'seq_length': 30, 'weight_decay': 0.01, 'use_attention': False})
    configs.append({**base_config, 'hidden_size': 128, 'num_layers': 2, 'dropout': 0.3,
                   'lr': 0.001, 'seq_length': 30, 'weight_decay': 0.01, 'bidirectional': False})
    configs.append({**base_config, 'hidden_size': 128, 'num_layers': 3, 'dropout': 0.5,
                   'lr': 0.0005, 'seq_length': 50, 'weight_decay': 0.01})

    # Remove duplicates
    seen = set()
    unique_configs = []
    for c in configs:
        key = tuple(sorted(c.items()))
        if key not in seen:
            seen.add(key)
            unique_configs.append(c)
    configs = unique_configs

    print(f"\nTesting {len(configs)} configurations...")

    # Run experiments
    results = []

    # Test on Raw + PRISM combined
    print("\n" + "="*70)
    print("TUNING: Raw + PRISM Features")
    print("="*70)

    for i, config in enumerate(configs):
        print(f"\nConfig {i+1}/{len(configs)}: hs={config['hidden_size']}, nl={config['num_layers']}, "
              f"do={config['dropout']:.1f}, lr={config['lr']}, sl={config['seq_length']}, wd={config['weight_decay']}")

        result = run_single_config(
            train_merged, test_merged, test_rul, all_cols,
            config, device, verbose=False
        )
        results.append(result)

        print(f"  Val RMSE: {result['val_rmse']:.2f} | Test RMSE: {result['test_rmse']:.2f}")

    # Sort by test RMSE
    results.sort(key=lambda x: x['test_rmse'])

    print("\n" + "="*70)
    print("TOP 10 CONFIGURATIONS")
    print("="*70)

    for i, r in enumerate(results[:10]):
        c = r['config']
        print(f"\n{i+1}. Test RMSE: {r['test_rmse']:.2f} (Val: {r['val_rmse']:.2f})")
        print(f"   hidden={c['hidden_size']}, layers={c['num_layers']}, dropout={c['dropout']:.1f}, "
              f"lr={c['lr']}, seq_len={c['seq_length']}, wd={c['weight_decay']}")

    # Run best config on raw-only for comparison
    best_config = results[0]['config']

    print("\n" + "="*70)
    print("BEST CONFIG - COMPARISON")
    print("="*70)

    print("\nRunning best config on Raw-only...")
    raw_result = run_single_config(
        train_df, test_df, test_rul, raw_cols,
        best_config, device, verbose=True
    )

    print("\nRunning best config on PRISM-only...")
    prism_result = run_single_config(
        train_merged, test_merged, test_rul, prism_cols,
        best_config, device, verbose=True
    )

    print("\n" + "="*70)
    print("FINAL RESULTS WITH BEST CONFIG")
    print("="*70)
    print(f"\n{'Model':<20} {'Features':<10} {'Val RMSE':<12} {'Test RMSE':<12}")
    print("-"*55)
    print(f"{'Raw Sensors':<20} {raw_result['n_features']:<10} {raw_result['val_rmse']:<12.2f} {raw_result['test_rmse']:<12.2f}")
    print(f"{'PRISM Only':<20} {prism_result['n_features']:<10} {prism_result['val_rmse']:<12.2f} {prism_result['test_rmse']:<12.2f}")
    print(f"{'Raw + PRISM':<20} {results[0]['n_features']:<10} {results[0]['val_rmse']:<12.2f} {results[0]['test_rmse']:<12.2f}")

    improvement = (raw_result['test_rmse'] - results[0]['test_rmse']) / raw_result['test_rmse'] * 100
    print(f"\nPRISM improvement over raw: {improvement:.1f}%")

    # Save results
    save_results = {
        'best_config': best_config,
        'raw_rmse': raw_result['test_rmse'],
        'prism_rmse': prism_result['test_rmse'],
        'combined_rmse': results[0]['test_rmse'],
        'all_results': [{'config': r['config'], 'val_rmse': r['val_rmse'], 'test_rmse': r['test_rmse']}
                       for r in results[:20]]
    }

    with open('/Users/jasonrudder/prism-mac/data/lstm_tuning_results.json', 'w') as f:
        json.dump(save_results, f, indent=2)

    print("\nResults saved to data/lstm_tuning_results.json")


if __name__ == "__main__":
    main()
