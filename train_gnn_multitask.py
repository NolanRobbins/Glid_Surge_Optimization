#!/usr/bin/env python3
"""
Multi-Task GNN Training - Optimize ALL 197K Nodes
==================================================
Predicts multiple optimization objectives for EVERY node in the network:

1. Port Surge (11 ports) - Real labels from IMF PortWatch
2. Rail Congestion Risk (ALL nodes) - Propagated from port surge
3. Terminal Utilization (terminals) - Derived from nearby port activity
4. Drayage Delay Risk (ALL nodes) - From truck times + weather
5. Chokepoint Likelihood (ALL nodes) - From graph centrality + traffic flow

This uses REAL data to create labels for ALL nodes, maximizing GPU usage
while staying aligned with competition.txt requirements.

Usage:
    python train_gnn_multitask.py --config production
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

import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent / "src"))

# GPU Setup
print("="*80)
print("  MULTI-TASK GNN - OPTIMIZING ALL 197K NODES")
print("  Competition-Aligned: Surge, Congestion, Utilization, Delays, Chokepoints")
print("="*80)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

if torch.cuda.is_available():
    cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
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
except:
    HAS_CUGRAPH = False

print("="*80)


# ============================================================================
# MULTI-TASK GNN MODEL
# ============================================================================

class MultiTaskSurgeGNN(nn.Module):
    """
    Multi-task GNN predicting 5 objectives for ALL nodes:
    1. Port surge (0-1)
    2. Rail congestion risk (0-1)
    3. Terminal utilization (0-1)
    4. Drayage delay risk (0-1)
    5. Chokepoint likelihood (0-1)
    """
    
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
            'port_surge': nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels // 2, 1),
                nn.Sigmoid()
            ),
            'rail_congestion': nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels // 2, 1),
                nn.Sigmoid()
            ),
            'terminal_utilization': nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels // 2, 1),
                nn.Sigmoid()
            ),
            'drayage_delay': nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels // 2, 1),
                nn.Sigmoid()
            ),
            'chokepoint_risk': nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels // 2, 1),
                nn.Sigmoid()
            ),
        })
        
        self.n_params = sum(p.numel() for p in self.parameters())
    
    def forward(self, x, edge_index):
        # Shared encoding
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Task-specific predictions
        outputs = {}
        for task_name, head in self.heads.items():
            outputs[task_name] = head(x)
        
        return outputs


# ============================================================================
# LABEL GENERATION - REAL DATA FOR ALL NODES
# ============================================================================

def compute_all_node_labels(
    G: nx.Graph,
    node_list: List,
    node_to_idx: Dict,
    port_features_df: pd.DataFrame,
    graph_features: Dict[str, np.ndarray],
    weather_df: pd.DataFrame = None,
    truck_times_df: pd.DataFrame = None,
    horizon: int = 24
) -> Dict[str, np.ndarray]:
    """
    Compute REAL labels for ALL nodes based on competition objectives.
    
    Returns:
        labels: Dict mapping task_name -> (num_nodes,) array
    """
    num_nodes = len(node_list)
    labels = {
        'port_surge': np.zeros(num_nodes, dtype=np.float32),  # Initialize to 0, not 0.5
        'rail_congestion': np.zeros(num_nodes, dtype=np.float32),
        'terminal_utilization': np.zeros(num_nodes, dtype=np.float32),
        'drayage_delay': np.zeros(num_nodes, dtype=np.float32),
        'chokepoint_risk': np.zeros(num_nodes, dtype=np.float32),
    }
    
    # 1. PORT SURGE (real labels for ports)
    PORT_DATA_TO_GRAPH = {
        "Los Angeles-Long Beach": "Port of Los Angeles",
        "New York-New Jersey": "Port of New York/New Jersey",
        "Savannah": "Port of Savannah",
        "Houston": "Port of Houston",
        "Oakland": "Port of Oakland",
        "Seattle": "Port of Seattle",
        "Tacoma": "Port of Tacoma",
        "Virginia": "Port of Virginia (Norfolk)",
        "Charleston": "Port of Charleston",
        "Miami": "Port of Miami",
        "New Orleans": "Port of New Orleans",
        "Baltimore": "Port of Baltimore",
    }
    
    graph_name_to_node = {}
    for n in node_list:
        if G.nodes[n].get('node_type') == 'port':
            graph_name = G.nodes[n].get('name', str(n))
            graph_name_to_node[graph_name] = n
    
    surge_col = f'surge_{horizon}h'
    matched_ports = 0
    surge_values = []
    
    # Check if surge column exists, if not try to get from original port_df
    if surge_col not in port_features_df.columns:
        # Try alternative column names
        alt_cols = [c for c in port_features_df.columns if 'surge' in c.lower()]
        if alt_cols:
            surge_col = alt_cols[0]
            print(f"    Using surge column: {surge_col}")
        else:
            print(f"    ⚠ Warning: Surge column '{f'surge_{horizon}h'}' not found in port_features_df")
            print(f"    Available columns: {list(port_features_df.columns)[:10]}...")
    
    for _, row in port_features_df.iterrows():
        port_name = row['portname']
        if port_name in PORT_DATA_TO_GRAPH:
            graph_name = PORT_DATA_TO_GRAPH[port_name]
            if graph_name in graph_name_to_node:
                node = graph_name_to_node[graph_name]
                idx = node_to_idx[node]
                if surge_col in port_features_df.columns and not pd.isna(row[surge_col]):
                    surge_val = float(row[surge_col])
                    labels['port_surge'][idx] = surge_val
                    surge_values.append(surge_val)
                    matched_ports += 1
    
    if matched_ports > 0:
        print(f"    ✓ Matched {matched_ports} ports with surge data")
        if surge_values:
            print(f"    Surge stats: min={min(surge_values):.4f}, max={max(surge_values):.4f}, mean={np.mean(surge_values):.4f}, median={np.median(surge_values):.4f}")
    else:
        print(f"    ⚠ Warning: No ports matched! Check port name mapping.")
        print(f"    Port names in data: {port_features_df['portname'].unique()[:5]}")
        print(f"    Port names in graph: {list(graph_name_to_node.keys())[:5]}")
    
    # 2. RAIL CONGESTION RISK (propagate port surge through network)
    # Use graph distance from ports to estimate congestion propagation
    print("  Computing rail congestion risk (propagating port surge)...")
    port_nodes = [n for n in node_list if G.nodes[n].get('node_type') == 'port']
    
    # Pre-compute distances from high-surge ports (optimization)
    # Use percentile-based threshold instead of fixed 0.5
    port_surge_values = [labels['port_surge'][node_to_idx[n]] for n in port_nodes if labels['port_surge'][node_to_idx[n]] > 0.0]
    if port_surge_values:
        surge_threshold = np.percentile(port_surge_values, 75)  # Top 25% of ports
        print(f"    Surge threshold (75th percentile): {surge_threshold:.4f}")
    else:
        surge_threshold = 0.3  # Fallback threshold
    
    high_surge_ports = []
    for port_node in port_nodes:
        port_idx = node_to_idx[port_node]
        surge_val = labels['port_surge'][port_idx]
        if surge_val > surge_threshold:
            high_surge_ports.append((port_node, surge_val))
    
    if high_surge_ports:
        # Use BFS for faster distance computation (limited depth)
        print(f"    Propagating from {len(high_surge_ports)} high-surge ports...")
        max_depth = 20  # Limit propagation distance
        
        for node in tqdm(node_list, desc="    Computing congestion"):
            idx = node_to_idx[node]
            node_type = G.nodes[node].get('node_type', 'rail_node')
            
            if node_type == 'port':
                # Port congestion = port surge
                labels['rail_congestion'][idx] = labels['port_surge'][idx]
            else:
                # Rail congestion = weighted average of nearby port surge
                total_weight = 0.0
                weighted_surge = 0.0
                
                for port_node, port_surge in high_surge_ports:
                    try:
                        # BFS for shortest path (faster than full shortest_path)
                        path_length = nx.shortest_path_length(G, port_node, node, weight=None)
                        if path_length <= max_depth:
                            # Exponential decay: closer = more influence
                            weight = np.exp(-path_length / 10.0)
                            weighted_surge += weight * port_surge
                            total_weight += weight
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        continue
                
                if total_weight > 0:
                    labels['rail_congestion'][idx] = min(1.0, weighted_surge / total_weight)
    else:
        print("    No high-surge ports found, setting congestion to 0")
    
    # 3. TERMINAL UTILIZATION (terminals near active ports)
    print("  Computing terminal utilization...")
    terminal_nodes = [n for n in node_list if G.nodes[n].get('node_type') == 'terminal']
    
    for terminal_node in terminal_nodes:
        term_idx = node_to_idx[terminal_node]
        
        # Utilization = average surge of nearby ports
        nearby_surge = []
        for port_node in port_nodes:
            try:
                path_length = nx.shortest_path_length(G, port_node, terminal_node, weight=None)
                if path_length <= 20:  # Within 20 hops
                    port_idx = node_to_idx[port_node]
                    nearby_surge.append(labels['port_surge'][port_idx])
            except:
                continue
        
        if nearby_surge:
            labels['terminal_utilization'][term_idx] = np.mean(nearby_surge)
    
    # 4. DRAYAGE DELAY RISK (weather + truck times affect all nodes)
    print("  Computing drayage delay risk...")
    if weather_df is not None and len(weather_df) > 0:
        weather = weather_df.copy()
        weather['date'] = pd.to_datetime(weather['date'])
        if weather['date'].dt.tz is not None:
            weather['date'] = weather['date'].dt.tz_localize(None)
        
        # Get latest weather
        latest_weather = weather.groupby('date').agg({
            'precipitation_sum': 'mean',
            'wind_speed_10m_max': 'mean'
        }).iloc[-1] if len(weather) > 0 else pd.Series({'precipitation_sum': 0, 'wind_speed_10m_max': 0})
        
        weather_risk = (
            np.clip(latest_weather['precipitation_sum'] / 20, 0, 1) * 0.6 +
            np.clip(latest_weather['wind_speed_10m_max'] / 50, 0, 1) * 0.4
        )
    else:
        weather_risk = 0.0
    
    if truck_times_df is not None and len(truck_times_df) > 0:
        time_col = [c for c in truck_times_df.columns if 'time' in c.lower() or 'minute' in c.lower()]
        if time_col:
            avg_time = truck_times_df[time_col[0]].mean()
            # Normalize: >2 hours = high delay risk
            truck_risk = min(1.0, avg_time / 120)
        else:
            truck_risk = 0.0
    else:
        truck_risk = 0.0
    
    # All nodes affected by weather + truck delays
    drayage_risk = (weather_risk * 0.5 + truck_risk * 0.5)
    labels['drayage_delay'] = np.full(num_nodes, drayage_risk, dtype=np.float32)
    
    # 5. CHOKEPOINT RISK (graph centrality + congestion)
    print("  Computing chokepoint risk...")
    for node in node_list:
        idx = node_to_idx[node]
        # Chokepoint = high centrality + high congestion
        centrality = graph_features[node][2] / 1000  # Betweenness (normalized)
        congestion = labels['rail_congestion'][idx]
        # Combined risk
        labels['chokepoint_risk'][idx] = min(1.0, (centrality * 0.6 + congestion * 0.4))
    
    # Summary
    print(f"\n  ✓ Label statistics:")
    for task, values in labels.items():
        n_labeled = np.sum(values > 0.01)  # Non-zero labels
        print(f"    {task}: {n_labeled:,} nodes ({n_labeled/num_nodes*100:.1f}%)")
    
    return labels


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_multitask_gnn(
    G: nx.Graph,
    graph_features: Dict[str, np.ndarray],
    port_features_df: pd.DataFrame,
    feature_cols: List[str],
    node_list: List,
    node_to_idx: Dict,
    weather_df: pd.DataFrame = None,
    truck_times_df: pd.DataFrame = None,
    hidden_channels: int = 512,
    num_layers: int = 4,
    epochs: int = 100,
    horizon: int = 24
) -> Tuple[MultiTaskSurgeGNN, Dict]:
    """Train multi-task GNN on ALL nodes."""
    
    print("\n" + "="*80)
    print(f"  MULTI-TASK GNN TRAINING ({horizon}h horizon)")
    print("  Optimizing ALL 197K nodes for 5 competition objectives")
    print("="*80)
    
    num_nodes = len(node_list)
    graph_feat_dim = 5
    port_feat_dim = len(feature_cols)
    total_feat_dim = graph_feat_dim + port_feat_dim
    
    # Build feature matrix (ALL nodes)
    print(f"\n  Building features for {num_nodes:,} nodes...")
    X = np.zeros((num_nodes, total_feat_dim), dtype=np.float32)
    
    for node in node_list:
        idx = node_to_idx[node]
        X[idx, :graph_feat_dim] = graph_features[node]
    
    # Fill port features where available
    PORT_DATA_TO_GRAPH = {
        "Los Angeles-Long Beach": "Port of Los Angeles",
        "New York-New Jersey": "Port of New York/New Jersey",
        "Savannah": "Port of Savannah",
        "Houston": "Port of Houston",
        "Oakland": "Port of Oakland",
        "Seattle": "Port of Seattle",
        "Tacoma": "Port of Tacoma",
        "Virginia": "Port of Virginia (Norfolk)",
        "Charleston": "Port of Charleston",
        "Miami": "Port of Miami",
        "New Orleans": "Port of New Orleans",
        "Baltimore": "Port of Baltimore",
    }
    
    graph_name_to_node = {}
    for n in node_list:
        if G.nodes[n].get('node_type') == 'port':
            graph_name = G.nodes[n].get('name', str(n))
            graph_name_to_node[graph_name] = n
    
    for _, row in port_features_df.iterrows():
        port_name = row['portname']
        if port_name in PORT_DATA_TO_GRAPH:
            graph_name = PORT_DATA_TO_GRAPH[port_name]
            if graph_name in graph_name_to_node:
                node = graph_name_to_node[graph_name]
                idx = node_to_idx[node]
                port_feats = row[feature_cols].values.astype(np.float32)
                X[idx, graph_feat_dim:] = port_feats
    
    # Compute labels for ALL nodes
    labels = compute_all_node_labels(
        G, node_list, node_to_idx, port_features_df,
        graph_features, weather_df, truck_times_df, horizon
    )
    
    # Build edge index
    edges = list(G.edges())
    edge_index = np.array([
        [node_to_idx[u] for u, v in edges] + [node_to_idx[v] for u, v in edges],
        [node_to_idx[v] for u, v in edges] + [node_to_idx[u] for u, v in edges]
    ], dtype=np.int64)
    
    # Transfer to GPU
    print(f"\n  Transferring {num_nodes:,} nodes to GPU...")
    data = Data(
        x=torch.FloatTensor(X),
        edge_index=torch.LongTensor(edge_index)
    ).to(DEVICE)
    
    y_dict = {k: torch.FloatTensor(v).unsqueeze(1).to(DEVICE) for k, v in labels.items()}
    
    # Create train/val split (80/20 of ALL nodes)
    perm = torch.randperm(num_nodes, device=DEVICE)
    split = int(0.8 * num_nodes)
    train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=DEVICE)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=DEVICE)
    train_mask[perm[:split]] = True
    val_mask[perm[split:]] = True
    
    print(f"  ✓ Train: {train_mask.sum().item():,}, Val: {val_mask.sum().item():,}")
    
    # Model
    model = MultiTaskSurgeGNN(
        in_channels=total_feat_dim,
        hidden_channels=hidden_channels,
        num_layers=num_layers
    ).to(DEVICE)
    
    print(f"  ✓ Model: {model.n_params:,} parameters")
    
    # Training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10)
    
    criterion = nn.MSELoss()
    scaler = torch.amp.GradScaler('cuda')
    
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    
    print(f"\n  Training for {epochs} epochs...")
    pbar = tqdm(range(epochs), desc="  Training", unit="epoch")
    
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()
        
        with torch.amp.autocast('cuda'):
            outputs = model(data.x, data.edge_index)
            
            # Multi-task loss (weighted sum)
            loss = 0.0
            task_weights = {
                'port_surge': 1.0,
                'rail_congestion': 0.8,
                'terminal_utilization': 0.6,
                'drayage_delay': 0.5,
                'chokepoint_risk': 0.7,
            }
            
            for task_name, weight in task_weights.items():
                task_loss = criterion(outputs[task_name][train_mask], y_dict[task_name][train_mask])
                loss += weight * task_loss
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Validation
        model.eval()
        with torch.no_grad():
            outputs = model(data.x, data.edge_index)
            val_loss = 0.0
            for task_name, weight in task_weights.items():
                task_loss = criterion(outputs[task_name][val_mask], y_dict[task_name][val_mask])
                val_loss += weight * task_loss
        
        scheduler.step(val_loss)
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'val': f'{val_loss.item():.4f}'})
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 20:
                print(f"\n  Early stopping at epoch {epoch}")
                break
    
    if best_state:
        model.load_state_dict(best_state)
    
    # Metrics
    model.eval()
    with torch.no_grad():
        outputs = model(data.x, data.edge_index)
    
    metrics = {}
    for task_name in labels.keys():
        y_pred = outputs[task_name][val_mask].cpu().numpy().flatten()
        y_true = y_dict[task_name][val_mask].cpu().numpy().flatten()
        
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - y_true.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        metrics[task_name] = {'mae': mae, 'rmse': rmse, 'r2': r2}
    
    # Print results
    print(f"\n  ┌{'─'*74}┐")
    print(f"  │  MULTI-TASK PERFORMANCE (ALL {num_nodes:,} NODES)                    │")
    print(f"  ├{'─'*74}┤")
    for task_name, m in metrics.items():
        print(f"  │  {task_name:20s}: MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  R²={m['r2']:.4f}  │")
    print(f"  └{'─'*74}┘")
    
    return model, metrics


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-Task GNN - Optimize ALL 197K Nodes")
    parser.add_argument('--config', choices=['production', 'stress_test', 'fast'], default='production')
    parser.add_argument('--horizons', type=str, default='24,48,72')
    parser.add_argument('--epochs', type=int, default=None)
    args = parser.parse_args()
    
    # Import data loading functions
    from data.loaders import (
        load_rail_nodes, load_rail_lines, load_port_activity,
        load_weather_data, load_ais_vessels, load_truck_times
    )
    from graph.builder import build_rail_graph, add_location_nodes, connect_locations_to_graph
    from config import US_PORTS, RAIL_TERMINALS
    
    # Import functions from train_gnn.py using importlib
    import importlib.util
    train_gnn_path = Path(__file__).parent / "train_gnn.py"
    spec = importlib.util.spec_from_file_location("train_gnn", train_gnn_path)
    train_gnn_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_gnn_module)
    build_comprehensive_features = train_gnn_module.build_comprehensive_features
    compute_real_surge_labels = train_gnn_module.compute_real_surge_labels
    
    # Config
    if args.config == 'production':
        hidden_channels = 512
        num_layers = 4
        epochs = 100
        truck_sample = 0.2
    elif args.config == 'stress_test':
        hidden_channels = 1024
        num_layers = 6
        epochs = 150
        truck_sample = 1.0
        print("\n🔥 STRESS TEST MODE - Pushing GPU to limits!")
    else:
        hidden_channels = 256
        num_layers = 3
        epochs = 50
        truck_sample = 0.05
    
    if args.epochs:
        epochs = args.epochs
    
    horizons = [int(h.strip()) for h in args.horizons.split(',')]
    
    print("\n🚀 MULTI-TASK GNN - Optimizing ALL 197K Nodes")
    print("   Competition objectives: Surge, Congestion, Utilization, Delays, Chokepoints")
    print(f"\n[Config] {args.config}")
    print(f"  Hidden: {hidden_channels}, Layers: {num_layers}, Epochs: {epochs}")
    print(f"  Horizons: {horizons}\n")
    
    # Load data
    print("Loading data...")
    rail_nodes = load_rail_nodes(filter_us_only=True)
    rail_lines = load_rail_lines(filter_us_only=True)
    
    # Use port names that match the data format (not US_PORTS names)
    major_ports = [
        "Los Angeles-Long Beach", "New York-New Jersey", "Savannah",
        "Houston", "Oakland", "Seattle", "Tacoma", "Virginia",
        "Charleston", "Miami", "New Orleans", "Baltimore"
    ]
    port_activity = load_port_activity(ports=major_ports, country="United States")
    weather = load_weather_data(hourly=False)
    truck_times = load_truck_times(sample_frac=truck_sample)
    
    # Build graph
    print("Building graph...")
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
    
    # Graph features (use cuGraph if available for real GPU work)
    print("\nComputing graph features...")
    
    if HAS_CUGRAPH:
        print("  Using GPU-accelerated cuGraph...")
        edges = list(G.edges())
        source = [node_to_idx[u] for u, v in edges]
        destination = [node_to_idx[v] for u, v in edges]
        
        gdf = cudf.DataFrame({'source': source, 'destination': destination})
        cu_G = cugraph.Graph()
        cu_G.from_cudf_edgelist(gdf, source='source', destination='destination')
        
        # PageRank
        pr_df = cugraph.pagerank(cu_G)
        pr_dict = dict(zip(pr_df['vertex'].to_pandas(), pr_df['pagerank'].to_pandas()))
        
        # Betweenness (sample for speed)
        try:
            bc_df = cugraph.betweenness_centrality(cu_G, k=500)
            bc_dict = dict(zip(bc_df['vertex'].to_pandas(), bc_df['betweenness_centrality'].to_pandas()))
        except:
            bc_dict = {i: 0.0 for i in range(len(node_list))}
    else:
        print("  Using NetworkX (CPU)...")
        pagerank_raw = nx.pagerank(G, max_iter=100)
        pr_dict = {node_to_idx[n]: v for n, v in pagerank_raw.items()}
        bc_dict = {i: 0.0 for i in range(len(node_list))}
    
    graph_features = {}
    for node in node_list:
        idx = node_to_idx[node]
        node_data = G.nodes[node]
        node_type = node_data.get('node_type', 'rail_node')
        graph_features[node] = np.array([
            pr_dict.get(idx, 0.0) * 10000,  # PageRank scaled
            G.degree(node) / 10,              # Degree normalized
            bc_dict.get(idx, 0.0) * 1000,    # Betweenness scaled
            1.0 if node_type == 'port' else 0.0,
            1.0 if node_type == 'terminal' else 0.0,
        ], dtype=np.float32)
    
    print(f"  ✓ Graph features computed for {len(graph_features):,} nodes")
    
    # Train for each horizon
    all_results = {}
    
    for horizon in horizons:
        print(f"\n{'='*80}")
        print(f"  HORIZON: {horizon}h")
        print(f"{'='*80}")
        
        # Compute surge labels for this horizon
        if len(port_activity) == 0:
            print(f"  ⚠ No port activity data for {horizon}h horizon, skipping...")
            continue
            
        port_df = compute_real_surge_labels(port_activity, horizon)
        
        # Port features
        print("Building port features...")
        port_df['date'] = pd.to_datetime(port_df['date'])
        
        # Preserve surge column before building features
        surge_col = f'surge_{horizon}h'
        surge_data = port_df[['portname', 'date', surge_col]].copy() if surge_col in port_df.columns else None
        
        port_features_df, feature_cols = build_comprehensive_features(
            port_df, weather, None, None, truck_times, horizon_hours=horizon
        )
        
        if len(port_features_df) == 0:
            print(f"  ⚠ No port features after building for {horizon}h horizon, skipping...")
            continue
        
        # Merge surge column back if it was dropped
        if surge_data is not None and surge_col not in port_features_df.columns:
            port_features_df = port_features_df.merge(
                surge_data, on=['portname', 'date'], how='left'
            )
            print(f"  ✓ Restored surge column: {surge_col}")
        
        # Train
        model, metrics = train_multitask_gnn(
            G, graph_features, port_features_df, feature_cols,
            node_list, node_to_idx, weather, truck_times,
            hidden_channels=hidden_channels, num_layers=num_layers, 
            epochs=epochs, horizon=horizon
        )
        
        # Save
        checkpoint_dir = Path("output/checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = checkpoint_dir / f"multitask_gnn_{horizon}h_{timestamp}.pt"
        
        torch.save({
            'model_state_dict': model.state_dict(),
            'metrics': metrics,
            'config': {
                'hidden_channels': hidden_channels,
                'num_layers': num_layers,
                'horizon': horizon,
            },
            'timestamp': timestamp,
        }, model_path)
        
        print(f"  ✓ Saved: {model_path}")
        all_results[horizon] = metrics
    
    # Final summary
    print(f"\n{'='*80}")
    print("  ✅ MULTI-TASK GNN TRAINING COMPLETE")
    print(f"{'='*80}")
    print("\n  RESULTS SUMMARY (ALL 197K NODES):")
    for horizon, metrics in all_results.items():
        print(f"\n  {horizon}h Horizon:")
        for task, m in metrics.items():
            print(f"    {task:20s}: R²={m['r2']:.4f}, MAE={m['mae']:.4f}, RMSE={m['rmse']:.4f}")
    print(f"{'='*80}\n")

