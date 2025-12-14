#!/usr/bin/env python3
"""
Multi-Task GNN Training v3 - TEMPORAL Training with Full Historical Data

Key improvements over v2:
1. Uses ALL 106 US ports (not just 11)
2. Uses ALL historical dates (6 years, 2127 samples per port)
3. Creates temporal training snapshots - each date becomes a training sample
4. Proper time-series features: lags, rolling windows, seasonality
5. Train/val/test split by TIME (prevents data leakage)

This gives us ~2000+ training samples instead of 1!
"""

import os
import sys
import warnings
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, BatchNorm
from torch_geometric.data import Data
import networkx as nx
from tqdm import tqdm
from collections import deque

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent / "src"))

# GPU Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Using device: {device}")

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   TF32 enabled for faster training")

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIGS = {
    'fast': {
        'hidden_channels': 256,
        'num_layers': 3,
        'dropout': 0.2,
        'lr': 0.001,
        'epochs': 30,
        'batch_size': 64,  # Temporal batches
        'truck_sample': 0.3,
        'temporal_window': 365,  # Use last 1 year
        'min_history_days': 30,  # Need 30 days of history for features
    },
    'medium': {
        'hidden_channels': 512,
        'num_layers': 4,
        'dropout': 0.2,
        'lr': 0.0005,
        'epochs': 100,
        'batch_size': 32,
        'truck_sample': 0.5,
        'temporal_window': 730,  # Use last 2 years
        'min_history_days': 60,
    },
    'full': {
        'hidden_channels': 768,
        'num_layers': 6,
        'dropout': 0.15,
        'lr': 0.0003,
        'epochs': 200,
        'batch_size': 16,
        'truck_sample': 1.0,
        'temporal_window': None,  # Use all data
        'min_history_days': 90,
    }
}

# ============================================================================
# MODEL ARCHITECTURE (Same as v2)
# ============================================================================

class MultiTaskSurgeGNN(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 512, 
                 num_layers: int = 4, dropout: float = 0.2):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        # GraphSAGE layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.norms.append(BatchNorm(hidden_channels))
        
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.norms.append(BatchNorm(hidden_channels))
        
        # Task-specific heads
        self.heads = nn.ModuleDict({
            'port_surge': self._make_head(hidden_channels, dropout),
            'rail_congestion': self._make_head(hidden_channels, dropout),
            'terminal_utilization': self._make_head(hidden_channels, dropout),
            'drayage_delay': self._make_head(hidden_channels, dropout),
            'chokepoint_risk': self._make_head(hidden_channels, dropout),
        })
        
        self.n_params = sum(p.numel() for p in self.parameters())
    
    def _make_head(self, hidden_channels: int, dropout: float):
        return nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, edge_index):
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        return {name: head(x) for name, head in self.heads.items()}


# ============================================================================
# DATA PREPARATION - TEMPORAL APPROACH
# ============================================================================

def load_port_activity_full(data_path: Path, country: str = "United States") -> pd.DataFrame:
    """Load ALL port activity data for the specified country."""
    csv_path = data_path / "global_daily_port_activity" / "Daily_Port_Activity_Data_and_Trade_Estimates.csv"
    
    print(f"Loading port activity from {csv_path}...")
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter by country
    df = df[df['country'] == country].copy()
    
    print(f"  Total {country} records: {len(df):,}")
    print(f"  Unique ports: {df['portname'].nunique()}")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    
    return df


def compute_surge_metric(df: pd.DataFrame, 
                         value_col: str = 'portcalls_container',
                         window: int = 30) -> pd.DataFrame:
    """
    Compute surge as deviation from rolling average.
    
    surge = (current - rolling_mean) / (rolling_std + epsilon)
    Normalized to [0, 1] using sigmoid-like transformation.
    """
    df = df.sort_values(['portname', 'date']).copy()
    
    # Group by port and compute rolling stats
    df['rolling_mean'] = df.groupby('portname')[value_col].transform(
        lambda x: x.rolling(window, min_periods=7).mean()
    )
    df['rolling_std'] = df.groupby('portname')[value_col].transform(
        lambda x: x.rolling(window, min_periods=7).std()
    )
    
    # Compute z-score
    epsilon = 0.01
    df['zscore'] = (df[value_col] - df['rolling_mean']) / (df['rolling_std'] + epsilon)
    
    # Convert to [0, 1] using sigmoid
    df['surge'] = 1 / (1 + np.exp(-df['zscore']))
    
    # Cap extreme values
    df['surge'] = df['surge'].clip(0, 1)
    
    return df


def create_temporal_features(df: pd.DataFrame, 
                             value_col: str = 'portcalls_container',
                             lags: List[int] = [1, 7, 14, 30]) -> pd.DataFrame:
    """Create time-series features for each port."""
    df = df.sort_values(['portname', 'date']).copy()
    
    # Lag features
    for lag in lags:
        df[f'lag_{lag}d'] = df.groupby('portname')[value_col].shift(lag)
    
    # Rolling means
    for window in [7, 14, 30]:
        df[f'rolling_mean_{window}d'] = df.groupby('portname')[value_col].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        df[f'rolling_std_{window}d'] = df.groupby('portname')[value_col].transform(
            lambda x: x.rolling(window, min_periods=1).std()
        )
    
    # Seasonality features
    df['day_of_week'] = df['date'].dt.dayofweek / 6  # Normalized 0-1
    df['day_of_month'] = df['date'].dt.day / 31
    df['month'] = df['date'].dt.month / 12
    df['quarter'] = df['date'].dt.quarter / 4
    
    # Year-over-year comparison (if enough data)
    df['yoy_change'] = df.groupby('portname')[value_col].pct_change(periods=365)
    df['yoy_change'] = df['yoy_change'].fillna(0).clip(-1, 1)
    
    # Rate of change
    df['roc_7d'] = df.groupby('portname')[value_col].pct_change(periods=7)
    df['roc_30d'] = df.groupby('portname')[value_col].pct_change(periods=30)
    df[['roc_7d', 'roc_30d']] = df[['roc_7d', 'roc_30d']].fillna(0).clip(-1, 1)
    
    return df


def create_port_graph_features(G: nx.Graph, port_nodes: Dict[str, str]) -> Dict[str, np.ndarray]:
    """Create graph-based features for port nodes."""
    
    print("  Computing graph features for ports...")
    
    # Pre-compute centrality metrics
    try:
        pagerank = nx.pagerank(G, max_iter=50, tol=1e-4)
    except:
        pagerank = {n: 1.0 / len(G) for n in G.nodes()}
    
    try:
        # Sample betweenness for speed
        if len(G) > 10000:
            betweenness = nx.betweenness_centrality(G, k=min(1000, len(G)))
        else:
            betweenness = nx.betweenness_centrality(G)
    except:
        betweenness = {n: 0.0 for n in G.nodes()}
    
    features = {}
    for port_name, node_id in port_nodes.items():
        if node_id in G:
            features[port_name] = np.array([
                pagerank.get(node_id, 0) * 1000,  # PageRank (scaled)
                G.degree(node_id),  # Degree centrality
                betweenness.get(node_id, 0) * 1000,  # Betweenness (scaled)
                1.0,  # is_port flag
                0.0,  # is_terminal flag
            ], dtype=np.float32)
        else:
            features[port_name] = np.zeros(5, dtype=np.float32)
            features[port_name][3] = 1.0  # is_port
    
    return features


def prepare_temporal_dataset(
    port_df: pd.DataFrame,
    G: nx.Graph,
    port_nodes: Dict[str, str],
    horizon: int = 24,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    min_history_days: int = 30
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Create temporal dataset where each date is a training sample.
    
    For each date T:
    - Features: port activity up to T (lags, rolling stats)
    - Labels: surge at T + horizon/24 days (horizon in hours)
    
    Returns:
        train_data, val_data, test_data: Lists of dicts with features and labels
    """
    
    print(f"\n📊 Creating temporal dataset (horizon={horizon}h)...")
    
    # Compute surge metric
    port_df = compute_surge_metric(port_df)
    
    # Create temporal features
    port_df = create_temporal_features(port_df)
    
    # Get unique dates (sorted)
    all_dates = sorted(port_df['date'].unique())
    print(f"  Total dates available: {len(all_dates)}")
    
    # Skip first min_history_days (need history for features)
    horizon_days = max(1, horizon // 24)  # Convert hours to days
    valid_dates = all_dates[min_history_days:-horizon_days] if horizon_days > 0 else all_dates[min_history_days:]
    print(f"  Valid training dates: {len(valid_dates)} (after removing {min_history_days} for warmup, {horizon_days} for horizon)")
    
    # Get graph features for ports
    graph_features = create_port_graph_features(G, port_nodes)
    
    # Feature columns (time-series features)
    ts_feature_cols = [
        'lag_1d', 'lag_7d', 'lag_14d', 'lag_30d',
        'rolling_mean_7d', 'rolling_mean_14d', 'rolling_mean_30d',
        'rolling_std_7d', 'rolling_std_14d', 'rolling_std_30d',
        'day_of_week', 'day_of_month', 'month', 'quarter',
        'yoy_change', 'roc_7d', 'roc_30d'
    ]
    
    # Create samples
    samples = []
    ports = sorted(port_df['portname'].unique())
    print(f"  Ports in dataset: {len(ports)}")
    
    for date in tqdm(valid_dates, desc="  Building temporal samples"):
        # Get data for this date
        day_data = port_df[port_df['date'] == date]
        
        # Get future date for labels
        future_date = date + pd.Timedelta(days=horizon_days)
        future_data = port_df[port_df['date'] == future_date]
        
        if len(future_data) < len(ports) * 0.5:  # Need at least 50% of ports
            continue
        
        # Build feature matrix for this date
        features_list = []
        labels_list = []
        port_mask = []
        
        for port in ports:
            port_today = day_data[day_data['portname'] == port]
            port_future = future_data[future_data['portname'] == port]
            
            if len(port_today) == 0 or len(port_future) == 0:
                continue
            
            # Time-series features (17 features)
            ts_feats = port_today[ts_feature_cols].values[0]
            ts_feats = np.nan_to_num(ts_feats, 0)
            
            # Graph features (5 features)
            graph_feats = graph_features.get(port, np.zeros(5))
            
            # Combine features (22 total)
            port_features = np.concatenate([ts_feats, graph_feats])
            features_list.append(port_features)
            
            # Label: future surge
            labels_list.append(port_future['surge'].values[0])
            port_mask.append(port)
        
        if len(features_list) > 10:  # Need at least 10 ports
            samples.append({
                'date': date,
                'features': np.array(features_list, dtype=np.float32),
                'labels': np.array(labels_list, dtype=np.float32),
                'ports': port_mask
            })
    
    print(f"  Total temporal samples: {len(samples)}")
    
    # Split by time (chronological - no data leakage!)
    n_train = int(len(samples) * train_ratio)
    n_val = int(len(samples) * val_ratio)
    
    train_data = samples[:n_train]
    val_data = samples[n_train:n_train + n_val]
    test_data = samples[n_train + n_val:]
    
    print(f"  Train: {len(train_data)} samples ({train_data[0]['date'].date()} to {train_data[-1]['date'].date()})")
    print(f"  Val:   {len(val_data)} samples ({val_data[0]['date'].date()} to {val_data[-1]['date'].date()})")
    print(f"  Test:  {len(test_data)} samples ({test_data[0]['date'].date()} to {test_data[-1]['date'].date()})")
    
    return train_data, val_data, test_data


# ============================================================================
# TRAINING - TEMPORAL APPROACH
# ============================================================================

def train_temporal_model(
    model: nn.Module,
    train_data: List[Dict],
    val_data: List[Dict],
    edge_index: torch.Tensor,
    full_node_features: torch.Tensor,
    port_indices: Dict[str, int],
    cfg: Dict,
    horizon: int
) -> Dict:
    """
    Train model on temporal samples.
    
    Note: The graph structure (edge_index) is static.
    Only node features and labels change per time step.
    """
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'])
    
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    best_model_state = None
    history = {'train_loss': [], 'val_loss': [], 'val_mae': []}
    
    edge_index = edge_index.to(device)
    
    print(f"\n🏋️ Training for {cfg['epochs']} epochs...")
    print(f"   Train samples: {len(train_data)}, Val samples: {len(val_data)}")
    
    for epoch in range(cfg['epochs']):
        # =====================
        # TRAINING
        # =====================
        model.train()
        train_losses = []
        
        # Shuffle training data
        np.random.shuffle(train_data)
        
        for sample in train_data:
            # Update node features for ports with this sample's features
            x = full_node_features.clone().to(device)
            
            for i, port_name in enumerate(sample['ports']):
                if port_name in port_indices:
                    idx = port_indices[port_name]
                    x[idx, :sample['features'].shape[1]] = torch.tensor(sample['features'][i])
            
            # Labels for port nodes only
            y = torch.zeros(len(full_node_features), device=device)
            mask = torch.zeros(len(full_node_features), dtype=torch.bool, device=device)
            
            for i, port_name in enumerate(sample['ports']):
                if port_name in port_indices:
                    idx = port_indices[port_name]
                    y[idx] = sample['labels'][i]
                    mask[idx] = True
            
            optimizer.zero_grad()
            
            outputs = model(x, edge_index)
            
            # Loss only on port nodes with labels
            loss = criterion(outputs['port_surge'][mask].squeeze(), y[mask])
            
            if not torch.isnan(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_losses.append(loss.item())
        
        # =====================
        # VALIDATION
        # =====================
        model.eval()
        val_losses = []
        val_maes = []
        
        with torch.no_grad():
            for sample in val_data:
                x = full_node_features.clone().to(device)
                
                for i, port_name in enumerate(sample['ports']):
                    if port_name in port_indices:
                        idx = port_indices[port_name]
                        x[idx, :sample['features'].shape[1]] = torch.tensor(sample['features'][i])
                
                y = torch.zeros(len(full_node_features), device=device)
                mask = torch.zeros(len(full_node_features), dtype=torch.bool, device=device)
                
                for i, port_name in enumerate(sample['ports']):
                    if port_name in port_indices:
                        idx = port_indices[port_name]
                        y[idx] = sample['labels'][i]
                        mask[idx] = True
                
                outputs = model(x, edge_index)
                
                val_loss = criterion(outputs['port_surge'][mask].squeeze(), y[mask])
                val_mae = torch.abs(outputs['port_surge'][mask].squeeze() - y[mask]).mean()
                
                if not torch.isnan(val_loss):
                    val_losses.append(val_loss.item())
                    val_maes.append(val_mae.item())
        
        # Update scheduler
        scheduler.step()
        
        # Track metrics
        avg_train_loss = np.mean(train_losses) if train_losses else 0
        avg_val_loss = np.mean(val_losses) if val_losses else 0
        avg_val_mae = np.mean(val_maes) if val_maes else 0
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_mae'].append(avg_val_mae)
        
        # Save best model
        if avg_val_loss < best_val_loss and avg_val_loss > 0:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:3d}/{cfg['epochs']}: "
                  f"train_loss={avg_train_loss:.4f}, "
                  f"val_loss={avg_val_loss:.4f}, "
                  f"val_mae={avg_val_mae:.4f}")
    
    # Restore best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return history


def evaluate_on_test(
    model: nn.Module,
    test_data: List[Dict],
    edge_index: torch.Tensor,
    full_node_features: torch.Tensor,
    port_indices: Dict[str, int]
) -> Dict:
    """Evaluate model on test set."""
    
    model.eval()
    edge_index = edge_index.to(device)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for sample in test_data:
            x = full_node_features.clone().to(device)
            
            for i, port_name in enumerate(sample['ports']):
                if port_name in port_indices:
                    idx = port_indices[port_name]
                    x[idx, :sample['features'].shape[1]] = torch.tensor(sample['features'][i])
            
            outputs = model(x, edge_index)
            
            for i, port_name in enumerate(sample['ports']):
                if port_name in port_indices:
                    idx = port_indices[port_name]
                    all_preds.append(outputs['port_surge'][idx].item())
                    all_labels.append(sample['labels'][i])
    
    preds = np.array(all_preds)
    labels = np.array(all_labels)
    
    mae = np.mean(np.abs(preds - labels))
    rmse = np.sqrt(np.mean((preds - labels) ** 2))
    
    # R² score
    ss_res = np.sum((labels - preds) ** 2)
    ss_tot = np.sum((labels - np.mean(labels)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'n_samples': len(preds)
    }


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Multi-Task GNN v3 - Temporal Training")
    parser.add_argument('--config', type=str, default='fast', choices=['fast', 'medium', 'full'])
    parser.add_argument('--horizons', type=str, default='24,48,72')
    parser.add_argument('--epochs', type=int, default=None)
    args = parser.parse_args()
    
    cfg = CONFIGS[args.config].copy()
    if args.epochs:
        cfg['epochs'] = args.epochs
    
    horizons = [int(h) for h in args.horizons.split(',')]
    
    print("=" * 70)
    print("🚀 MULTI-TASK GNN TRAINING V3 - TEMPORAL APPROACH")
    print("=" * 70)
    print(f"Config: {args.config}")
    print(f"Horizons: {horizons}")
    print(f"Epochs: {cfg['epochs']}")
    print()
    
    # =========================================================================
    # LOAD DATA
    # =========================================================================
    
    from data.loaders import load_rail_nodes, load_rail_lines
    from graph.builder import build_rail_graph, add_location_nodes, connect_locations_to_graph
    from config import US_PORTS, RAIL_TERMINALS
    
    print("📂 Loading data...")
    
    # Rail network
    rail_nodes = load_rail_nodes(filter_us_only=True)
    rail_lines = load_rail_lines(filter_us_only=True)
    print(f"  Rail nodes: {len(rail_nodes):,}")
    
    # Full port activity (ALL US ports, ALL dates)
    data_path = Path(__file__).parent / "data"
    port_df = load_port_activity_full(data_path)
    
    # =========================================================================
    # BUILD GRAPH
    # =========================================================================
    
    print("\n📊 Building graph...")
    G = build_rail_graph(rail_nodes, rail_lines)
    
    # Add ALL ports from US_PORTS config
    # Convert list format to dict format expected by add_location_nodes
    ports_dict = {p['name']: p for p in US_PORTS}
    terminals_dict = {t['name']: t for t in RAIL_TERMINALS}
    
    G = add_location_nodes(G, ports_dict, node_type='port')
    G = add_location_nodes(G, terminals_dict, node_type='terminal')
    G = connect_locations_to_graph(G, ports_dict, node_type='port')
    G = connect_locations_to_graph(G, terminals_dict, node_type='terminal')
    
    print(f"  Total graph nodes: {G.number_of_nodes():,}")
    print(f"  Total graph edges: {G.number_of_edges():,}")
    
    # Node indexing
    node_list = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    
    # Find port nodes
    port_nodes = {}  # port_name (from data) -> node_id
    for node in node_list:
        if G.nodes[node].get('node_type') == 'port':
            name = G.nodes[node].get('name', '')
            # Try to match with data port names
            for data_port in port_df['portname'].unique():
                if data_port.lower() in name.lower() or name.lower() in data_port.lower():
                    port_nodes[data_port] = node
    
    # Also add direct mappings for ports in the data
    for port_name in port_df['portname'].unique():
        if port_name not in port_nodes:
            # Check if there's a matching node
            for node in node_list:
                node_name = G.nodes[node].get('name', '')
                if port_name.lower() in node_name.lower():
                    port_nodes[port_name] = node
                    break
    
    print(f"  Ports matched to graph: {len(port_nodes)}")
    
    # Port indices for training
    port_indices = {port: node_to_idx[node] for port, node in port_nodes.items()}
    
    # =========================================================================
    # PREPARE FEATURES
    # =========================================================================
    
    print("\n🔧 Preparing base node features...")
    
    # Base features for all nodes (will be updated per time step)
    n_nodes = len(node_list)
    feature_dim = 22  # 17 time-series + 5 graph features
    
    full_node_features = torch.zeros(n_nodes, feature_dim)
    
    # Edge index
    edges = list(G.edges())
    edge_index = torch.tensor(
        [[node_to_idx[e[0]], node_to_idx[e[1]]] for e in edges] +
        [[node_to_idx[e[1]], node_to_idx[e[0]]] for e in edges],
        dtype=torch.long
    ).T
    
    print(f"  Node feature dim: {feature_dim}")
    print(f"  Edge index shape: {edge_index.shape}")
    
    # =========================================================================
    # TRAIN FOR EACH HORIZON
    # =========================================================================
    
    output_dir = Path(__file__).parent / "output" / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for horizon in horizons:
        print("\n" + "=" * 70)
        print(f"🎯 TRAINING HORIZON: {horizon}h")
        print("=" * 70)
        
        # Prepare temporal dataset
        train_data, val_data, test_data = prepare_temporal_dataset(
            port_df, G, port_nodes, 
            horizon=horizon,
            min_history_days=cfg['min_history_days']
        )
        
        if len(train_data) < 10:
            print(f"  ⚠ Not enough training samples, skipping horizon {horizon}h")
            continue
        
        # Initialize model
        model = MultiTaskSurgeGNN(
            in_channels=feature_dim,
            hidden_channels=cfg['hidden_channels'],
            num_layers=cfg['num_layers'],
            dropout=cfg['dropout']
        ).to(device)
        
        print(f"\n  Model parameters: {model.n_params:,}")
        
        # Train
        history = train_temporal_model(
            model, train_data, val_data, edge_index, 
            full_node_features, port_indices, cfg, horizon
        )
        
        # Evaluate on test set
        test_metrics = evaluate_on_test(
            model, test_data, edge_index, full_node_features, port_indices
        )
        
        print(f"\n📊 Test Results ({horizon}h):")
        print(f"   MAE:  {test_metrics['mae']:.4f}")
        print(f"   RMSE: {test_metrics['rmse']:.4f}")
        print(f"   R²:   {test_metrics['r2']:.4f}")
        print(f"   Samples: {test_metrics['n_samples']}")
        
        # Save model
        save_path = output_dir / f"gnn_multitask_v3_{horizon}h.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': cfg,
            'horizon': horizon,
            'in_channels': feature_dim,
            'test_metrics': test_metrics,
            'training_history': history,
            'timestamp': datetime.now().isoformat()
        }, save_path)
        print(f"   ✓ Saved to {save_path}")
    
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)

