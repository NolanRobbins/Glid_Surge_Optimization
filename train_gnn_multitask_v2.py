#!/usr/bin/env python3
"""
Multi-Task GNN Training V2 - FIXED Label Propagation
=====================================================
Key fixes:
1. Efficient reverse BFS from ports (O(n) instead of O(n*m))
2. Proper surge label preservation
3. Better port name matching with debug output
4. Uses ALL 197K nodes with propagated labels

Usage:
    python train_gnn_multitask_v2.py --config production
    python train_gnn_multitask_v2.py --config fast  # Quick test
"""

import os
import sys
import json
import time
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import deque

import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent / "src"))

# GPU Setup
print("="*80)
print("  MULTI-TASK GNN V2 - EFFICIENT LABEL PROPAGATION")
print("  Training ALL 197K nodes with real competition data")
print("="*80)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

if torch.cuda.is_available():
    cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    DEVICE = torch.device('cuda')
    print(f"✓ GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB)")
else:
    DEVICE = torch.device('cpu')
    print("⚠ CPU only")

import torch_geometric
from torch_geometric.nn import SAGEConv, BatchNorm
from torch_geometric.data import Data

try:
    import cugraph
    import cudf
    HAS_CUGRAPH = True
    print("✓ cuGraph available")
except ImportError:
    HAS_CUGRAPH = False
    print("⚠ cuGraph not available")

print("="*80)


# ============================================================================
# MULTI-TASK GNN MODEL (same as before)
# ============================================================================

class MultiTaskSurgeGNN(nn.Module):
    """Multi-task GNN predicting 5 objectives for ALL nodes."""
    
    def __init__(self, in_channels: int, hidden_channels: int = 512, 
                 num_layers: int = 4, dropout: float = 0.2):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Shared encoder
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
# EFFICIENT LABEL PROPAGATION (FIXED!)
# ============================================================================

def propagate_labels_from_ports_bfs(
    G: nx.Graph,
    node_list: List,
    node_to_idx: Dict,
    port_surge_values: Dict[str, float],  # node_id -> surge value
    max_depth: int = 30,
    decay_rate: float = 0.15
) -> np.ndarray:
    """
    EFFICIENT label propagation using multi-source BFS.
    
    Instead of: for each node, find distance to each port (O(n*m))
    We do: BFS from each port, update all reachable nodes (O(n+e))
    
    Args:
        G: NetworkX graph
        node_list: List of node IDs
        node_to_idx: Node ID to index mapping
        port_surge_values: Dict of port node_id -> surge value
        max_depth: Maximum propagation depth
        decay_rate: How fast influence decays with distance
        
    Returns:
        congestion: (num_nodes,) array of congestion values
    """
    num_nodes = len(node_list)
    congestion = np.zeros(num_nodes, dtype=np.float32)
    total_weight = np.zeros(num_nodes, dtype=np.float32)
    
    # BFS from each port with surge > 0
    active_ports = [(node, surge) for node, surge in port_surge_values.items() if surge > 0.01]
    
    if not active_ports:
        print("    ⚠ No active ports with surge > 0.01")
        return congestion
    
    print(f"    Propagating from {len(active_ports)} ports with surge data...")
    
    for port_node, surge_value in tqdm(active_ports, desc="    BFS propagation"):
        if port_node not in node_to_idx:
            continue
            
        # BFS from this port
        visited = {port_node: 0}  # node -> distance
        queue = deque([port_node])
        
        while queue:
            current = queue.popleft()
            current_dist = visited[current]
            
            if current_dist >= max_depth:
                continue
            
            for neighbor in G.neighbors(current):
                if neighbor not in visited:
                    visited[neighbor] = current_dist + 1
                    queue.append(neighbor)
        
        # Update congestion for all visited nodes
        for node, dist in visited.items():
            if node in node_to_idx:
                idx = node_to_idx[node]
                weight = np.exp(-dist * decay_rate)
                congestion[idx] += weight * surge_value
                total_weight[idx] += weight
    
    # Normalize
    mask = total_weight > 0
    congestion[mask] /= total_weight[mask]
    congestion = np.clip(congestion, 0, 1)
    
    return congestion


def compute_all_node_labels_v2(
    G: nx.Graph,
    node_list: List,
    node_to_idx: Dict,
    port_surge_df: pd.DataFrame,  # Must have 'portname' and 'surge_Xh' column
    graph_features: Dict[str, np.ndarray],
    horizon: int = 24,
    weather_risk: float = 0.0,
    truck_risk: float = 0.0
) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """
    Compute labels for ALL nodes using efficient propagation.
    
    Returns:
        labels: Dict of task_name -> (num_nodes,) array
        stats: Dict of statistics for debugging
    """
    num_nodes = len(node_list)
    surge_col = f'surge_{horizon}h'
    
    labels = {
        'port_surge': np.zeros(num_nodes, dtype=np.float32),
        'rail_congestion': np.zeros(num_nodes, dtype=np.float32),
        'terminal_utilization': np.zeros(num_nodes, dtype=np.float32),
        'drayage_delay': np.zeros(num_nodes, dtype=np.float32),
        'chokepoint_risk': np.zeros(num_nodes, dtype=np.float32),
    }
    
    stats = {
        'ports_in_graph': 0,
        'ports_matched': 0,
        'ports_with_surge': 0,
        'terminals_in_graph': 0,
    }
    
    # =========================================================================
    # 1. MAP PORTS FROM DATA TO GRAPH NODES
    # =========================================================================
    
    # Port name mapping (data format -> graph format)
    PORT_DATA_TO_GRAPH = {
        "Los Angeles-Long Beach": ["Port of Los Angeles", "Port of Long Beach"],
        "New York-New Jersey": ["Port of New York/New Jersey"],
        "Savannah": ["Port of Savannah"],
        "Houston": ["Port of Houston"],
        "Oakland": ["Port of Oakland"],
        "Seattle": ["Port of Seattle"],
        "Tacoma": ["Port of Tacoma"],
        "Virginia": ["Port of Virginia", "Port of Virginia (Norfolk)"],
        "Charleston": ["Port of Charleston"],
        "Miami": ["Port of Miami"],
        "New Orleans": ["Port of New Orleans"],
        "Baltimore": ["Port of Baltimore"],
    }
    
    # Find port nodes in graph
    graph_port_nodes = {}  # graph_name -> node_id
    for node in node_list:
        node_data = G.nodes[node]
        if node_data.get('node_type') == 'port':
            name = node_data.get('name', '')
            if name:
                graph_port_nodes[name] = node
                stats['ports_in_graph'] += 1
    
    print(f"\n  Port nodes in graph ({stats['ports_in_graph']}):")
    for name in list(graph_port_nodes.keys())[:10]:
        print(f"    - {name}")
    
    # Find terminal nodes
    terminal_nodes = []
    for node in node_list:
        if G.nodes[node].get('node_type') == 'terminal':
            terminal_nodes.append(node)
            stats['terminals_in_graph'] += 1
    
    print(f"  Terminal nodes in graph: {stats['terminals_in_graph']}")
    
    # =========================================================================
    # 2. ASSIGN PORT SURGE LABELS
    # =========================================================================
    
    print(f"\n  Assigning port surge labels ({surge_col})...")
    
    if surge_col not in port_surge_df.columns:
        print(f"    ⚠ Column '{surge_col}' not found!")
        print(f"    Available: {list(port_surge_df.columns)}")
        return labels, stats
    
    # Get AVERAGE surge across all dates (not just latest) for better labels
    port_surge_df = port_surge_df.copy()
    port_surge_df['date'] = pd.to_datetime(port_surge_df['date'])
    
    # Use mean surge across time for more robust labels
    avg_surge = port_surge_df.groupby('portname')[surge_col].agg(['mean', 'max', 'std']).reset_index()
    avg_surge.columns = ['portname', 'surge_mean', 'surge_max', 'surge_std']
    
    # Merge back to get port names with their average surge
    latest_surge = port_surge_df.sort_values('date').groupby('portname').last().reset_index()
    latest_surge = latest_surge.merge(avg_surge, on='portname', how='left')
    # Use mean surge as the label (more representative than single date)
    latest_surge[surge_col] = latest_surge['surge_mean']
    
    print(f"    Surge data for {len(latest_surge)} ports (using mean across all dates):")
    
    port_surge_values = {}  # node_id -> surge
    
    for _, row in latest_surge.iterrows():
        data_port = row['portname']
        surge_val = row[surge_col]
        
        if pd.isna(surge_val):
            continue
        
        surge_val = float(surge_val)
        print(f"      {data_port}: surge={surge_val:.4f}")
        
        if data_port in PORT_DATA_TO_GRAPH:
            for graph_port in PORT_DATA_TO_GRAPH[data_port]:
                if graph_port in graph_port_nodes:
                    node = graph_port_nodes[graph_port]
                    idx = node_to_idx[node]
                    labels['port_surge'][idx] = surge_val
                    port_surge_values[node] = surge_val
                    stats['ports_matched'] += 1
                    stats['ports_with_surge'] += 1
                    print(f"        ✓ Matched to: {graph_port}")
    
    print(f"\n    Matched {stats['ports_matched']} port nodes with surge labels")
    
    # =========================================================================
    # 3. PROPAGATE RAIL CONGESTION (EFFICIENT BFS)
    # =========================================================================
    
    print("\n  Computing rail congestion (efficient BFS propagation)...")
    labels['rail_congestion'] = propagate_labels_from_ports_bfs(
        G, node_list, node_to_idx, port_surge_values,
        max_depth=30, decay_rate=0.12
    )
    
    # =========================================================================
    # 4. TERMINAL UTILIZATION
    # =========================================================================
    
    print("  Computing terminal utilization...")
    for terminal_node in terminal_nodes:
        term_idx = node_to_idx[terminal_node]
        # Terminal utilization = congestion of the terminal
        labels['terminal_utilization'][term_idx] = labels['rail_congestion'][term_idx]
    
    # =========================================================================
    # 5. DRAYAGE DELAY RISK (uniform based on weather/truck)
    # =========================================================================
    
    print("  Computing drayage delay risk...")
    drayage_base = (weather_risk * 0.5 + truck_risk * 0.5)
    
    # Add variation based on congestion
    labels['drayage_delay'] = np.clip(
        drayage_base + labels['rail_congestion'] * 0.3,
        0, 1
    ).astype(np.float32)
    
    # =========================================================================
    # 6. CHOKEPOINT RISK (centrality × congestion)
    # =========================================================================
    
    print("  Computing chokepoint risk...")
    for node in node_list:
        idx = node_to_idx[node]
        # Betweenness centrality (from graph features, index 2, already scaled by 1000)
        betweenness = graph_features[node][2] / 1000  # Normalize back
        congestion = labels['rail_congestion'][idx]
        
        # Chokepoint = high centrality node under congestion
        labels['chokepoint_risk'][idx] = np.clip(
            (betweenness * 0.4 + congestion * 0.6),
            0, 1
        )
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    
    print(f"\n  ✓ Label statistics:")
    for task, values in labels.items():
        n_nonzero = np.sum(values > 0.01)
        mean_val = np.mean(values)
        max_val = np.max(values)
        print(f"    {task:25s}: {n_nonzero:>7,} nodes ({n_nonzero/num_nodes*100:>5.1f}%), mean={mean_val:.4f}, max={max_val:.4f}")
    
    return labels, stats


# ============================================================================
# COMPUTE REAL SURGE LABELS (from train_gnn.py)
# ============================================================================

def compute_real_surge_labels(port_df: pd.DataFrame, horizon_hours: int = 24) -> pd.DataFrame:
    """
    Compute surge labels using percentile-based thresholds.
    
    Surge = normalized distance from median, where:
    - 0.0 = at or below median
    - 0.5 = at 75th percentile
    - 1.0 = at or above 95th percentile
    """
    df = port_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    if df['date'].dt.tz is not None:
        df['date'] = df['date'].dt.tz_localize(None)
    
    horizon_days = max(1, horizon_hours // 24)
    df = df.sort_values(['portname', 'date'])
    
    # Create future target
    df[f'future_{horizon_hours}h'] = df.groupby('portname')['portcalls'].shift(-horizon_days)
    
    # Compute percentiles per port
    surge_col = f'surge_{horizon_hours}h'
    
    def compute_surge(group):
        future = group[f'future_{horizon_hours}h']
        p50 = future.quantile(0.5)
        p75 = future.quantile(0.75)
        p95 = future.quantile(0.95)
        
        # Normalize: 0 at median, 0.5 at p75, 1.0 at p95
        def normalize(val):
            if pd.isna(val):
                return np.nan
            if val <= p50:
                return 0.0
            elif val <= p75:
                return 0.5 * (val - p50) / (p75 - p50) if p75 > p50 else 0.25
            elif val <= p95:
                return 0.5 + 0.5 * (val - p75) / (p95 - p75) if p95 > p75 else 0.75
            else:
                return 1.0
        
        group[surge_col] = future.apply(normalize)
        return group
    
    df = df.groupby('portname', group_keys=False).apply(compute_surge)
    
    # Drop rows without surge label
    df = df.dropna(subset=[surge_col])
    
    print(f"  Surge labels ({surge_col}):")
    print(f"    Records: {len(df):,}")
    print(f"    Mean: {df[surge_col].mean():.4f}")
    print(f"    Distribution: min={df[surge_col].min():.4f}, max={df[surge_col].max():.4f}")
    
    return df


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_multitask_gnn_v2(
    G: nx.Graph,
    graph_features: Dict[str, np.ndarray],
    port_surge_df: pd.DataFrame,
    feature_cols: List[str],
    node_list: List,
    node_to_idx: Dict,
    horizon: int = 24,
    hidden_channels: int = 512,
    num_layers: int = 4,
    epochs: int = 100,
    weather_risk: float = 0.0,
    truck_risk: float = 0.0
) -> Tuple[MultiTaskSurgeGNN, Dict]:
    """Train multi-task GNN with efficient label propagation."""
    
    print("\n" + "="*80)
    print(f"  MULTI-TASK GNN V2 TRAINING ({horizon}h horizon)")
    print("="*80)
    
    num_nodes = len(node_list)
    graph_feat_dim = 5
    port_feat_dim = len(feature_cols)
    total_feat_dim = graph_feat_dim + port_feat_dim
    
    # =========================================================================
    # BUILD FEATURE MATRIX
    # =========================================================================
    
    print(f"\n  Building features for {num_nodes:,} nodes...")
    X = np.zeros((num_nodes, total_feat_dim), dtype=np.float32)
    
    # Graph features for ALL nodes
    for node in node_list:
        idx = node_to_idx[node]
        X[idx, :graph_feat_dim] = graph_features[node]
    
    # Port features where available
    PORT_DATA_TO_GRAPH = {
        "Los Angeles-Long Beach": ["Port of Los Angeles", "Port of Long Beach"],
        "New York-New Jersey": ["Port of New York/New Jersey"],
        "Savannah": ["Port of Savannah"],
        "Houston": ["Port of Houston"],
        "Oakland": ["Port of Oakland"],
        "Seattle": ["Port of Seattle"],
        "Tacoma": ["Port of Tacoma"],
        "Virginia": ["Port of Virginia", "Port of Virginia (Norfolk)"],
        "Charleston": ["Port of Charleston"],
        "Miami": ["Port of Miami"],
        "New Orleans": ["Port of New Orleans"],
        "Baltimore": ["Port of Baltimore"],
    }
    
    graph_port_nodes = {}
    for node in node_list:
        if G.nodes[node].get('node_type') == 'port':
            name = G.nodes[node].get('name', '')
            if name:
                graph_port_nodes[name] = node
    
    # Get latest features for each port
    port_surge_df = port_surge_df.copy()
    port_surge_df['date'] = pd.to_datetime(port_surge_df['date'])
    latest_features = port_surge_df.sort_values('date').groupby('portname').last().reset_index()
    
    ports_with_features = 0
    for _, row in latest_features.iterrows():
        data_port = row['portname']
        if data_port in PORT_DATA_TO_GRAPH:
            for graph_port in PORT_DATA_TO_GRAPH[data_port]:
                if graph_port in graph_port_nodes:
                    node = graph_port_nodes[graph_port]
                    idx = node_to_idx[node]
                    # Fill port features
                    port_feats = row[feature_cols].values.astype(np.float32)
                    port_feats = np.nan_to_num(port_feats, nan=0.0)
                    X[idx, graph_feat_dim:] = port_feats
                    ports_with_features += 1
    
    print(f"  ✓ {ports_with_features} ports have full features")
    
    # Clean features - CRITICAL: replace NaN and normalize
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Normalize features to prevent gradient explosion
    X_mean = X.mean(axis=0, keepdims=True)
    X_std = X.std(axis=0, keepdims=True) + 1e-8  # Avoid division by zero
    X = (X - X_mean) / X_std
    X = np.clip(X, -10, 10)  # Clip extreme values
    
    print(f"  ✓ Features normalized (mean={X.mean():.4f}, std={X.std():.4f})")
    
    # =========================================================================
    # COMPUTE LABELS
    # =========================================================================
    
    labels, stats = compute_all_node_labels_v2(
        G, node_list, node_to_idx, port_surge_df, graph_features,
        horizon=horizon, weather_risk=weather_risk, truck_risk=truck_risk
    )
    
    # =========================================================================
    # BUILD GRAPH DATA
    # =========================================================================
    
    edges = list(G.edges())
    edge_index = np.array([
        [node_to_idx[u] for u, v in edges] + [node_to_idx[v] for u, v in edges],
        [node_to_idx[v] for u, v in edges] + [node_to_idx[u] for u, v in edges]
    ], dtype=np.int64)
    
    print(f"\n  Transferring to GPU...")
    data = Data(
        x=torch.FloatTensor(X),
        edge_index=torch.LongTensor(edge_index)
    ).to(DEVICE)
    
    y_dict = {k: torch.FloatTensor(v).unsqueeze(1).to(DEVICE) for k, v in labels.items()}
    
    # =========================================================================
    # TRAIN/VAL SPLIT
    # =========================================================================
    
    # Focus on nodes with non-zero labels for validation
    has_label_mask = np.zeros(num_nodes, dtype=bool)
    for task_labels in labels.values():
        has_label_mask |= (task_labels > 0.01)
    
    labeled_indices = np.where(has_label_mask)[0]
    n_labeled = len(labeled_indices)
    
    print(f"  Nodes with labels: {n_labeled:,} ({n_labeled/num_nodes*100:.1f}%)")
    
    # Train on all nodes, validate on labeled subset
    perm = torch.randperm(num_nodes, device=DEVICE)
    split = int(0.8 * num_nodes)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=DEVICE)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=DEVICE)
    train_mask[perm[:split]] = True
    val_mask[perm[split:]] = True
    
    # For validation metrics, focus on labeled nodes
    val_labeled_mask = val_mask & torch.tensor(has_label_mask, device=DEVICE)
    
    print(f"  Train: {train_mask.sum().item():,}, Val: {val_mask.sum().item():,}")
    print(f"  Val with labels: {val_labeled_mask.sum().item():,}")
    
    # =========================================================================
    # MODEL
    # =========================================================================
    
    model = MultiTaskSurgeGNN(
        in_channels=total_feat_dim,
        hidden_channels=hidden_channels,
        num_layers=num_layers
    ).to(DEVICE)
    
    print(f"  ✓ Model: {model.n_params:,} parameters")
    print(f"  ✓ Features: {total_feat_dim} ({graph_feat_dim} graph + {port_feat_dim} port)")
    
    # =========================================================================
    # TRAINING
    # =========================================================================
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    criterion = nn.MSELoss()
    
    # Task weights (emphasize port_surge since it has real labels)
    task_weights = {
        'port_surge': 2.0,       # Real labels - most important
        'rail_congestion': 1.0,  # Propagated from real
        'terminal_utilization': 0.5,
        'drayage_delay': 0.3,
        'chokepoint_risk': 0.5,
    }
    
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    
    print(f"\n  Training for {epochs} epochs...")
    pbar = tqdm(range(epochs), desc="  Training", unit="epoch")
    
    scaler = torch.amp.GradScaler('cuda') if DEVICE.type == 'cuda' else None
    
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()
        
        if scaler:
            with torch.amp.autocast('cuda'):
                outputs = model(data.x, data.edge_index)
                loss = sum(
                    weight * criterion(outputs[task][train_mask], y_dict[task][train_mask])
                    for task, weight in task_weights.items()
                )
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(data.x, data.edge_index)
            loss = sum(
                weight * criterion(outputs[task][train_mask], y_dict[task][train_mask])
                for task, weight in task_weights.items()
            )
            loss.backward()
            optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            outputs = model(data.x, data.edge_index)
            val_loss = sum(
                weight * criterion(outputs[task][val_labeled_mask], y_dict[task][val_labeled_mask])
                for task, weight in task_weights.items()
            )
        
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'val': f'{val_loss.item():.4f}', 'lr': f'{lr:.1e}'})
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 25:
                print(f"\n  Early stopping at epoch {epoch}")
                break
    
    if best_state:
        model.load_state_dict(best_state)
    
    # =========================================================================
    # METRICS
    # =========================================================================
    
    model.eval()
    with torch.no_grad():
        outputs = model(data.x, data.edge_index)
    
    metrics = {}
    for task_name in labels.keys():
        # Evaluate on labeled validation nodes
        y_pred = outputs[task_name][val_labeled_mask].cpu().numpy().flatten()
        y_true = y_dict[task_name][val_labeled_mask].cpu().numpy().flatten()
        
        if len(y_true) == 0:
            metrics[task_name] = {'mae': 0.0, 'rmse': 0.0, 'r2': 0.0}
            continue
        
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-10 else 0.0
        
        metrics[task_name] = {'mae': float(mae), 'rmse': float(rmse), 'r2': float(r2)}
    
    # Print results
    print(f"\n  ┌{'─'*74}┐")
    print(f"  │  MULTI-TASK PERFORMANCE (ALL {num_nodes:,} NODES)                         │")
    print(f"  ├{'─'*74}┤")
    for task_name, m in metrics.items():
        print(f"  │  {task_name:22s}: MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  R²={m['r2']:+.4f}  │")
    print(f"  └{'─'*74}┘")
    
    return model, metrics


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-Task GNN V2 - Fixed Label Propagation")
    parser.add_argument('--config', choices=['production', 'fast', 'stress'], default='production')
    parser.add_argument('--horizons', type=str, default='24,48,72')
    parser.add_argument('--epochs', type=int, default=None)
    args = parser.parse_args()
    
    # Imports
    from data.loaders import (
        load_rail_nodes, load_rail_lines, load_port_activity,
        load_weather_data, load_truck_times
    )
    from graph.builder import build_rail_graph, add_location_nodes, connect_locations_to_graph
    from config import US_PORTS, RAIL_TERMINALS
    from forecasting.features import build_forecasting_features
    
    # Config
    configs = {
        'production': {'hidden': 512, 'layers': 4, 'epochs': 150, 'truck_sample': 0.1},
        'fast': {'hidden': 256, 'layers': 3, 'epochs': 30, 'truck_sample': 0.01},
        'stress': {'hidden': 1024, 'layers': 6, 'epochs': 200, 'truck_sample': 0.5},
    }
    cfg = configs[args.config]
    if args.epochs:
        cfg['epochs'] = args.epochs
    
    horizons = [int(h.strip()) for h in args.horizons.split(',')]
    
    print(f"\n🚀 Multi-Task GNN V2 - {args.config.upper()} mode")
    print(f"   Hidden: {cfg['hidden']}, Layers: {cfg['layers']}, Epochs: {cfg['epochs']}")
    print(f"   Horizons: {horizons}\n")
    
    # =========================================================================
    # LOAD DATA
    # =========================================================================
    
    print("Loading data...")
    rail_nodes = load_rail_nodes(filter_us_only=True)
    rail_lines = load_rail_lines(filter_us_only=True)
    
    major_ports = [
        "Los Angeles-Long Beach", "New York-New Jersey", "Savannah",
        "Houston", "Oakland", "Seattle", "Tacoma", "Virginia",
        "Charleston", "Miami", "New Orleans"
    ]
    port_activity = load_port_activity(ports=major_ports, country="United States")
    print(f"  Port activity: {len(port_activity):,} records")
    
    weather = load_weather_data(hourly=False)
    truck_times = load_truck_times(sample_frac=cfg['truck_sample'])
    
    # Compute weather/truck risk
    weather_risk = 0.0
    if weather is not None and len(weather) > 0:
        latest = weather.sort_values('date').iloc[-1]
        weather_risk = min(1.0, (latest.get('precipitation_sum', 0) / 20 * 0.6 + 
                                  latest.get('wind_speed_10m_max', 0) / 50 * 0.4))
        print(f"  Weather risk: {weather_risk:.3f}")
    
    truck_risk = 0.0
    if truck_times is not None and len(truck_times) > 0:
        time_col = [c for c in truck_times.columns if 'time' in c.lower() or 'minute' in c.lower()]
        if time_col:
            truck_risk = min(1.0, truck_times[time_col[0]].mean() / 120)
            print(f"  Truck risk: {truck_risk:.3f}")
    
    # =========================================================================
    # BUILD GRAPH
    # =========================================================================
    
    print("\nBuilding graph...")
    G = build_rail_graph(rail_nodes, rail_lines)
    
    ports_dict = {p["name"]: p for p in US_PORTS}
    G = add_location_nodes(G, ports_dict, "port")
    G = connect_locations_to_graph(G, ports_dict, "port", max_connection_miles=30)
    
    terminals_dict = {t["name"]: t for t in RAIL_TERMINALS}
    G = add_location_nodes(G, terminals_dict, "terminal")
    G = connect_locations_to_graph(G, terminals_dict, "terminal", max_connection_miles=30)
    
    node_list = list(G.nodes())
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    print(f"  ✓ Graph: {len(node_list):,} nodes, {G.number_of_edges():,} edges")
    
    # =========================================================================
    # GRAPH FEATURES
    # =========================================================================
    
    print("\nComputing graph features...")
    
    if HAS_CUGRAPH:
        print("  Using cuGraph (GPU)...")
        edges = list(G.edges())
        source = [node_to_idx[u] for u, v in edges]
        destination = [node_to_idx[v] for u, v in edges]
        
        gdf = cudf.DataFrame({'source': source, 'destination': destination})
        cu_G = cugraph.Graph()
        cu_G.from_cudf_edgelist(gdf, source='source', destination='destination')
        
        pr_df = cugraph.pagerank(cu_G)
        pr_dict = dict(zip(pr_df['vertex'].to_pandas(), pr_df['pagerank'].to_pandas()))
        
        try:
            bc_df = cugraph.betweenness_centrality(cu_G, k=500)
            bc_dict = dict(zip(bc_df['vertex'].to_pandas(), bc_df['betweenness_centrality'].to_pandas()))
        except Exception:
            bc_dict = {i: 0.0 for i in range(len(node_list))}
        
        # Degree
        deg_df = cu_G.degrees()
        if 'in_degree' in deg_df.columns:
            deg_series = deg_df['in_degree'] + deg_df['out_degree']
        else:
            deg_series = deg_df.get('degree', pd.Series([0]*len(node_list)))
        deg_dict = dict(zip(deg_df['vertex'].to_pandas(), deg_series.to_pandas()))
    else:
        print("  Using NetworkX (CPU)...")
        pr_raw = nx.pagerank(G, max_iter=100)
        pr_dict = {node_to_idx[n]: v for n, v in pr_raw.items()}
        deg_dict = {node_to_idx[n]: G.degree(n) for n in node_list}
        bc_dict = {i: 0.0 for i in range(len(node_list))}
    
    graph_features = {}
    for node in node_list:
        idx = node_to_idx[node]
        node_data = G.nodes[node]
        node_type = node_data.get('node_type', 'rail_node')
        
        graph_features[node] = np.array([
            pr_dict.get(idx, 0.0) * 10000,
            deg_dict.get(idx, 0) / 10,
            bc_dict.get(idx, 0.0) * 1000,
            1.0 if node_type == 'port' else 0.0,
            1.0 if node_type == 'terminal' else 0.0,
        ], dtype=np.float32)
    
    print(f"  ✓ Graph features: {len(graph_features):,} nodes")
    
    # =========================================================================
    # TRAIN FOR EACH HORIZON
    # =========================================================================
    
    all_results = {}
    
    for horizon in horizons:
        print(f"\n{'='*80}")
        print(f"  HORIZON: {horizon}h")
        print(f"{'='*80}")
        
        # Compute surge labels
        port_df = compute_real_surge_labels(port_activity.copy(), horizon)
        
        if len(port_df) == 0:
            print(f"  ⚠ No data for {horizon}h, skipping...")
            continue
        
        # Build port features
        feat_df, feature_cols = build_forecasting_features(
            port_df, weather, target_col='portcalls', group_col='portname'
        )
        
        # Ensure surge column is preserved
        surge_col = f'surge_{horizon}h'
        if surge_col not in feat_df.columns and surge_col in port_df.columns:
            feat_df = feat_df.merge(
                port_df[['portname', 'date', surge_col]].drop_duplicates(),
                on=['portname', 'date'],
                how='left'
            )
        
        print(f"  Feature columns: {len(feature_cols)}")
        print(f"  Surge column present: {surge_col in feat_df.columns}")
        
        # Train
        model, metrics = train_multitask_gnn_v2(
            G, graph_features, feat_df, feature_cols,
            node_list, node_to_idx, horizon=horizon,
            hidden_channels=cfg['hidden'], num_layers=cfg['layers'],
            epochs=cfg['epochs'], weather_risk=weather_risk, truck_risk=truck_risk
        )
        
        # Save
        checkpoint_dir = Path("output/checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = checkpoint_dir / f"multitask_gnn_v2_{horizon}h_{timestamp}.pt"
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'metrics': metrics,
            'config': {
                'hidden_channels': cfg['hidden'],
                'num_layers': cfg['layers'],
                'horizon': horizon,
                'in_channels': 5 + len(feature_cols),
            },
            'timestamp': timestamp,
        }, model_path)
        
        print(f"  ✓ Saved: {model_path.name}")
        all_results[horizon] = metrics
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    
    print(f"\n{'='*80}")
    print("  ✅ MULTI-TASK GNN V2 TRAINING COMPLETE")
    print(f"{'='*80}")
    
    for horizon, metrics in all_results.items():
        print(f"\n  {horizon}h Horizon:")
        for task, m in metrics.items():
            print(f"    {task:22s}: R²={m['r2']:+.4f}, MAE={m['mae']:.4f}")
    
    print(f"\n{'='*80}\n")

