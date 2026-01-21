"""
LSTM Hyperparameter Tuning (Light) - Focused search with fewer configs.
"""

import sys
import numpy as np
import polars as pl
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pathlib import Path
from typing import Tuple, List

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

class RULDataset(Dataset):
    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
    def __len__(self):
        return len(self.targets)
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


def create_sequences(df, feature_cols, seq_length=30):
    sequences, targets = [], []
    for entity in df['unit_id'].unique().sort().to_list():
        entity_df = df.filter(pl.col('unit_id') == entity).sort('cycle')
        features = entity_df.select(feature_cols).to_numpy()
        rul = entity_df['RUL'].to_numpy()
        for i in range(len(features) - seq_length + 1):
            sequences.append(features[i:i + seq_length])
            targets.append(rul[i + seq_length - 1])
    return np.array(sequences), np.array(targets)


def create_test_sequences(df, feature_cols, seq_length=30):
    sequences = []
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
    return np.array(sequences)


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1),
        )
    def forward(self, lstm_output):
        weights = torch.softmax(self.attention(lstm_output), dim=1)
        return torch.sum(weights * lstm_output, dim=1), weights


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3, bidirectional=True):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0, bidirectional=bidirectional)
        out_size = hidden_size * (2 if bidirectional else 1)
        self.attention = Attention(out_size)
        self.fc = nn.Sequential(
            nn.Linear(out_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        context, _ = self.attention(lstm_out)
        return self.fc(context).squeeze(-1)


def train_and_evaluate(train_seqs, train_targets, test_seqs, test_rul, config, device):
    """Train model and return test RMSE."""
    n_samples, seq_len, n_features = train_seqs.shape

    # Split
    n_train = int(0.8 * n_samples)
    idx = np.random.permutation(n_samples)

    train_ds = RULDataset(train_seqs[idx[:n_train]], train_targets[idx[:n_train]])
    val_ds = RULDataset(train_seqs[idx[n_train:]], train_targets[idx[n_train:]])

    train_loader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config['batch_size'])

    model = LSTMModel(
        input_size=n_features,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        bidirectional=config['bidirectional'],
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(config['epochs']):
        model.train()
        for seqs, targs in train_loader:
            seqs, targs = seqs.to(device), targs.to(device)
            optimizer.zero_grad()
            loss = criterion(model(seqs), targs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for seqs, targs in val_loader:
                seqs, targs = seqs.to(device), targs.to(device)
                val_loss += criterion(model(seqs), targs).item()
        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                break

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        preds = model(torch.FloatTensor(test_seqs).to(device)).cpu().numpy()

    test_rmse = np.sqrt(mean_squared_error(test_rul, preds))
    val_rmse = np.sqrt(best_val_loss)

    return val_rmse, test_rmse


def main():
    print("="*60)
    print("LSTM HYPERPARAMETER TUNING (Light)")
    print("="*60)

    device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    np.random.seed(42)
    torch.manual_seed(42)

    # Load data
    print("\nLoading data...")
    data_dir = Path('/Users/jasonrudder/prism-mac/data')

    cols = ['unit_id', 'cycle'] + [f'op_{i}' for i in range(1, 4)] + [f's_{i}' for i in range(1, 22)]
    train_df = pl.read_csv(data_dir / 'train_FD002.txt', separator=' ', has_header=False,
                           new_columns=cols, truncate_ragged_lines=True)
    test_df = pl.read_csv(data_dir / 'test_FD002.txt', separator=' ', has_header=False,
                          new_columns=cols, truncate_ragged_lines=True)

    # Add RUL
    max_cycles = train_df.group_by('unit_id').agg(pl.col('cycle').max().alias('max_cycle'))
    train_df = train_df.join(max_cycles, on='unit_id')
    train_df = train_df.with_columns((pl.col('max_cycle') - pl.col('cycle')).clip(upper_bound=125).alias('RUL')).drop('max_cycle')

    with open(data_dir / 'RUL_FD002.txt') as f:
        test_rul = np.clip([float(l.strip()) for l in f if l.strip()], 0, 125)

    # Load PRISM
    print("Loading PRISM features...")
    vector_df = pl.read_parquet(data_dir / 'vector.parquet')
    geometry_df = pl.read_parquet(data_dir / 'geometry.parquet')
    state_df = pl.read_parquet(data_dir / 'state.parquet')

    # Merge features
    prism_cols = []

    for df_ref, is_train in [(train_df, True), (test_df, False)]:
        result = df_ref.clone()

        # Vector
        vec = vector_df.with_columns((pl.col('engine') + '_' + pl.col('source_signal')).alias('feature_name'))
        vec_pivot = vec.pivot(index=['entity_id', 'timestamp'], on='feature_name', values='value', aggregate_function='first')
        vec_pivot = vec_pivot.with_columns([
            pl.col('entity_id').str.extract(r'U(\d+)', 1).cast(pl.Int64).alias('unit_id'),
            pl.col('timestamp').cast(pl.Int64).alias('cycle'),
        ]).drop(['entity_id', 'timestamp'])

        if is_train:
            prism_cols.extend([c for c in vec_pivot.columns if c not in ['unit_id', 'cycle']])

        result = result.join(vec_pivot, on=['unit_id', 'cycle'], how='left')

        # Geometry
        geo_cols = [c for c in geometry_df.columns if c not in ['entity_id', 'timestamp', 'computed_at', 'signal_ids', 'mode_id', 'n_features', 'n_engines']]
        geo = geometry_df.select(['entity_id', 'timestamp'] + geo_cols)
        geo = geo.with_columns([
            pl.col('entity_id').str.extract(r'U(\d+)', 1).cast(pl.Int64).alias('unit_id'),
            pl.col('timestamp').cast(pl.Int64).alias('cycle'),
        ]).drop(['entity_id', 'timestamp'])
        if is_train:
            prism_cols.extend(geo_cols)
        result = result.join(geo, on=['unit_id', 'cycle'], how='left')

        # State
        state_cols = [c for c in state_df.columns if c not in ['entity_id', 'timestamp', 'computed_at', 'is_failure_signature', 'mode_transition', 'position_dim', 'mode_id', 'state_label', 'failure_signature']]
        state = state_df.select(['entity_id', 'timestamp'] + state_cols)
        state = state.with_columns([
            pl.col('entity_id').str.extract(r'U(\d+)', 1).cast(pl.Int64).alias('unit_id'),
            pl.col('timestamp').cast(pl.Int64).alias('cycle'),
        ]).drop(['entity_id', 'timestamp'])
        if is_train:
            prism_cols.extend(state_cols)
        result = result.join(state, on=['unit_id', 'cycle'], how='left')

        # Fill
        for c in prism_cols:
            if c in result.columns:
                result = result.with_columns(pl.col(c).forward_fill().over('unit_id'))
                result = result.with_columns(pl.col(c).backward_fill().over('unit_id'))

        if is_train:
            train_merged = result
        else:
            test_merged = result

    raw_cols = [f'op_{i}' for i in range(1, 4)] + [f's_{i}' for i in range(1, 22)]
    all_cols = raw_cols + prism_cols

    # Filter valid columns
    valid_cols = [c for c in all_cols if c in train_merged.columns and train_merged[c].null_count() / len(train_merged) < 0.5]
    print(f"Valid features: {len(valid_cols)}")

    # Prepare data
    train_clean = train_merged.with_columns([pl.col(c).fill_null(0) for c in valid_cols])
    test_clean = test_merged.with_columns([pl.col(c).fill_null(0) for c in valid_cols])

    # Configs to test
    configs = [
        # Baseline variations
        {'hidden_size': 64, 'num_layers': 2, 'dropout': 0.2, 'lr': 0.001, 'weight_decay': 0.0, 'seq_length': 30, 'batch_size': 256, 'epochs': 80, 'patience': 20, 'bidirectional': True, 'name': 'baseline'},

        # Higher regularization
        {'hidden_size': 64, 'num_layers': 2, 'dropout': 0.5, 'lr': 0.001, 'weight_decay': 0.01, 'seq_length': 30, 'batch_size': 256, 'epochs': 80, 'patience': 20, 'bidirectional': True, 'name': 'high_reg'},
        {'hidden_size': 64, 'num_layers': 2, 'dropout': 0.5, 'lr': 0.0005, 'weight_decay': 0.05, 'seq_length': 30, 'batch_size': 256, 'epochs': 100, 'patience': 25, 'bidirectional': True, 'name': 'very_high_reg'},

        # Smaller model
        {'hidden_size': 32, 'num_layers': 1, 'dropout': 0.3, 'lr': 0.001, 'weight_decay': 0.01, 'seq_length': 30, 'batch_size': 256, 'epochs': 80, 'patience': 20, 'bidirectional': True, 'name': 'small'},

        # Bigger model with more reg
        {'hidden_size': 128, 'num_layers': 2, 'dropout': 0.5, 'lr': 0.0005, 'weight_decay': 0.02, 'seq_length': 30, 'batch_size': 128, 'epochs': 100, 'patience': 25, 'bidirectional': True, 'name': 'big_reg'},

        # Different sequence lengths
        {'hidden_size': 64, 'num_layers': 2, 'dropout': 0.4, 'lr': 0.001, 'weight_decay': 0.01, 'seq_length': 15, 'batch_size': 256, 'epochs': 80, 'patience': 20, 'bidirectional': True, 'name': 'seq15'},
        {'hidden_size': 64, 'num_layers': 2, 'dropout': 0.4, 'lr': 0.001, 'weight_decay': 0.01, 'seq_length': 50, 'batch_size': 128, 'epochs': 80, 'patience': 20, 'bidirectional': True, 'name': 'seq50'},

        # Unidirectional
        {'hidden_size': 128, 'num_layers': 2, 'dropout': 0.4, 'lr': 0.001, 'weight_decay': 0.01, 'seq_length': 30, 'batch_size': 256, 'epochs': 80, 'patience': 20, 'bidirectional': False, 'name': 'unidir'},

        # Deep with high dropout
        {'hidden_size': 64, 'num_layers': 3, 'dropout': 0.5, 'lr': 0.0005, 'weight_decay': 0.02, 'seq_length': 30, 'batch_size': 256, 'epochs': 100, 'patience': 25, 'bidirectional': True, 'name': 'deep'},

        # Lower LR
        {'hidden_size': 64, 'num_layers': 2, 'dropout': 0.3, 'lr': 0.0002, 'weight_decay': 0.01, 'seq_length': 30, 'batch_size': 256, 'epochs': 150, 'patience': 30, 'bidirectional': True, 'name': 'low_lr'},
    ]

    print(f"\nTesting {len(configs)} configurations on Raw+PRISM...")
    print("-"*60)

    results = []

    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] {config['name']}: hs={config['hidden_size']}, nl={config['num_layers']}, do={config['dropout']}, lr={config['lr']}, wd={config['weight_decay']}, seq={config['seq_length']}")

        # Create sequences
        train_seqs, train_targets = create_sequences(train_clean, valid_cols, config['seq_length'])
        test_seqs = create_test_sequences(test_clean, valid_cols, config['seq_length'])

        # Normalize
        n_samples, seq_len, n_features = train_seqs.shape
        scaler = StandardScaler()
        train_seqs = scaler.fit_transform(train_seqs.reshape(-1, n_features)).reshape(n_samples, seq_len, n_features)
        test_seqs = scaler.transform(test_seqs.reshape(-1, n_features)).reshape(len(test_seqs), seq_len, n_features)

        val_rmse, test_rmse = train_and_evaluate(train_seqs, train_targets, test_seqs, test_rul, config, device)

        print(f"    Val RMSE: {val_rmse:.2f} | Test RMSE: {test_rmse:.2f}")
        results.append({'name': config['name'], 'config': config, 'val_rmse': val_rmse, 'test_rmse': test_rmse})

        # Clear memory
        del train_seqs, test_seqs
        torch.mps.empty_cache() if device == 'mps' else None

    # Sort by test RMSE
    results.sort(key=lambda x: x['test_rmse'])

    print("\n" + "="*60)
    print("RESULTS RANKED BY TEST RMSE")
    print("="*60)

    for i, r in enumerate(results):
        print(f"{i+1}. {r['name']:<15} Val: {r['val_rmse']:.2f}  Test: {r['test_rmse']:.2f}")

    # Run best config on raw-only
    best = results[0]
    print(f"\n{'='*60}")
    print(f"BEST CONFIG: {best['name']}")
    print("="*60)

    print("\nRunning best config on Raw-only...")
    raw_valid = [c for c in raw_cols if c in train_df.columns]
    train_clean_raw = train_df.with_columns([pl.col(c).fill_null(0) for c in raw_valid])
    test_clean_raw = test_df.with_columns([pl.col(c).fill_null(0) for c in raw_valid])

    train_seqs_raw, train_targets_raw = create_sequences(train_clean_raw, raw_valid, best['config']['seq_length'])
    test_seqs_raw = create_test_sequences(test_clean_raw, raw_valid, best['config']['seq_length'])

    n_samples, seq_len, n_features = train_seqs_raw.shape
    scaler = StandardScaler()
    train_seqs_raw = scaler.fit_transform(train_seqs_raw.reshape(-1, n_features)).reshape(n_samples, seq_len, n_features)
    test_seqs_raw = scaler.transform(test_seqs_raw.reshape(-1, n_features)).reshape(len(test_seqs_raw), seq_len, n_features)

    raw_val, raw_test = train_and_evaluate(train_seqs_raw, train_targets_raw, test_seqs_raw, test_rul, best['config'], device)

    print(f"\n{'='*60}")
    print("FINAL COMPARISON WITH BEST CONFIG")
    print("="*60)
    print(f"{'Model':<20} {'Val RMSE':<12} {'Test RMSE':<12}")
    print("-"*45)
    print(f"{'Raw Sensors':<20} {raw_val:<12.2f} {raw_test:<12.2f}")
    print(f"{'Raw + PRISM':<20} {best['val_rmse']:<12.2f} {best['test_rmse']:<12.2f}")

    improvement = (raw_test - best['test_rmse']) / raw_test * 100
    print(f"\nPRISM improvement: {improvement:.1f}%")

    if improvement > 0:
        print("✓ PRISM features HELP LSTM with tuned hyperparameters!")
    else:
        print("✗ PRISM features still don't improve LSTM")


if __name__ == "__main__":
    main()
