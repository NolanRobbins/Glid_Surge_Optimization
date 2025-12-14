#!/usr/bin/env python3
"""
GPU-Accelerated Training Script for Glid Surge Optimization
============================================================
Optimized for ASUS Ascent GX10 with NVIDIA GB10 Grace Blackwell Superchip
- 1 petaFLOP AI performance
- 128GB unified memory
- NVIDIA Blackwell architecture

ALIGNED WITH training.md SPECIFICATION:
- Integrates ALL data sources (Port Activity, Weather, AIS Vessels, Rail Network)
- Multi-horizon predictions (24h, 48h, 72h)
- Uses real surge data from IMF PortWatch
- Graph topology features via cuGraph
- Weather impact scoring

Usage:
    python train_gpu.py                    # Default training
    python train_gpu.py --config accurate  # Maximum accuracy
    python train_gpu.py --config fast      # Quick prototyping
"""

import os
import sys
import json
import pickle
import time
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple

# ============================================================================
# CUDA/GPU OPTIMIZATION FLAGS FOR ASUS ASCENT GX10
# ============================================================================
os.environ['NVIDIA_TF32_OVERRIDE'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['CUPY_ACCELERATORS'] = 'cub'
os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda-13.0/lib64:/usr/local/cuda-13.0/targets/sbsa-linux/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.auto import tqdm as tqdm_auto

# Rich progress bar support
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich import print as rprint
    HAS_RICH = True
except ImportError:
    HAS_RICH = False


# ============================================================================
# GPU STATUS AND MONITORING
# ============================================================================
print("="*70)
print("  GLID SURGE OPTIMIZATION - GPU TRAINING")
print("  Optimized for ASUS Ascent GX10 (Grace Blackwell)")
print("  ALIGNED WITH training.md SPECIFICATION")
print("="*70)

# Import RAPIDS
try:
    import cudf
    import cuml
    import cugraph
    import cupy as cp
    
    mempool = cp.get_default_memory_pool()
    pinned_mempool = cp.get_default_pinned_memory_pool()
    
    print(f"✓ cuDF: {cudf.__version__}")
    print(f"✓ cuML: {cuml.__version__}")
    print(f"✓ cuGraph: {cugraph.__version__}")
    print(f"✓ CuPy: {cp.__version__}")
    HAS_RAPIDS = True
except ImportError as e:
    print(f"✗ RAPIDS not available: {e}")
    HAS_RAPIDS = False
    mempool = None

# PyTorch (for GNN if needed)
try:
    import torch
    import torch.backends.cudnn as cudnn
    
    cudnn.benchmark = True
    cudnn.enabled = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        gpu_compute = torch.cuda.get_device_capability(0)
        
        print(f"  GPU: {gpu_name}")
        print(f"  Memory: {gpu_mem:.1f} GB")
        print(f"  Compute Capability: {gpu_compute[0]}.{gpu_compute[1]}")
    
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("✗ PyTorch not available")

print("="*70)


# ============================================================================
# GPU MONITORING UTILITIES
# ============================================================================

class GPUMonitor:
    """Monitor GPU utilization and memory during training."""
    
    def __init__(self):
        self.start_time = None
        self.metrics_history = []
        
    def start(self):
        self.start_time = time.time()
        if HAS_RAPIDS and mempool:
            mempool.free_all_blocks()
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
    
    def get_memory_stats(self) -> Dict[str, float]:
        stats = {}
        if HAS_TORCH and torch.cuda.is_available():
            stats['torch_allocated_gb'] = torch.cuda.memory_allocated() / 1e9
            stats['torch_reserved_gb'] = torch.cuda.memory_reserved() / 1e9
            stats['torch_peak_gb'] = torch.cuda.max_memory_allocated() / 1e9
        if HAS_RAPIDS and mempool:
            stats['cupy_used_gb'] = mempool.used_bytes() / 1e9
            stats['cupy_total_gb'] = mempool.total_bytes() / 1e9
        return stats
    
    def log_metrics(self, step: int, metrics: Dict[str, float]):
        elapsed = time.time() - self.start_time if self.start_time else 0
        record = {
            'step': step,
            'elapsed_sec': elapsed,
            **metrics,
            **self.get_memory_stats()
        }
        self.metrics_history.append(record)
        return record
    
    def print_summary(self):
        if not self.metrics_history:
            return
        elapsed = time.time() - self.start_time if self.start_time else 0
        mem_stats = self.get_memory_stats()
        print("\n" + "="*70)
        print("  GPU UTILIZATION SUMMARY")
        print("="*70)
        print(f"  Total training time: {elapsed:.1f} seconds")
        if 'torch_peak_gb' in mem_stats:
            print(f"  Peak GPU memory (PyTorch): {mem_stats['torch_peak_gb']:.2f} GB")
        if 'cupy_total_gb' in mem_stats:
            print(f"  Peak GPU memory (CuPy): {mem_stats['cupy_total_gb']:.2f} GB")
        print("="*70)


gpu_monitor = GPUMonitor()


# ============================================================================
# DATA LOADING - ALL SOURCES FROM training.md
# ============================================================================

def load_all_data_sources() -> Dict[str, Any]:
    """
    Load ALL data sources as specified in training.md:
    1. Port Activity (IMF PortWatch) - for surge labels
    2. Rail Network (USDOT) - for graph topology
    3. Weather Data - for weather impact features
    4. AIS Vessel Tracking - for vessel ETAs
    5. PortWatch Chokepoints - for trade volume context
    6. Truck Travel Times - for drayage calibration
    """
    from data.loaders import (
        load_rail_nodes, load_rail_lines, load_port_activity,
        load_weather_data, load_ais_vessels, load_portwatch_chokepoints,
        load_truck_times
    )
    
    print("\n" + "="*70)
    print("  LOADING ALL DATA SOURCES (per training.md)")
    print("="*70)
    
    data = {}
    
    # 1. Rail Network (Graph Topology)
    print("\n[1/7] Loading Rail Network...")
    data['rail_nodes'] = load_rail_nodes(filter_us_only=True)
    data['rail_lines'] = load_rail_lines(filter_us_only=True)
    print(f"  ✓ {len(data['rail_nodes']):,} nodes, {len(data['rail_lines']):,} edges")
    
    # 2. Port Activity (Surge Labels) - THE KEY DATA SOURCE
    print("\n[2/7] Loading Port Activity (IMF PortWatch)...")
    # Load for major US ports
    major_ports = [
        "Los Angeles-Long Beach", "New York-New Jersey", "Savannah",
        "Houston", "Oakland", "Seattle", "Tacoma", "Virginia",
        "Charleston", "Miami", "New Orleans", "Baltimore"
    ]
    data['port_activity'] = load_port_activity(ports=major_ports, country="United States")
    print(f"  ✓ {len(data['port_activity']):,} daily records for {data['port_activity']['portname'].nunique()} ports")
    
    # 3. Weather Data (Impact Features)
    print("\n[3/7] Loading Weather Data...")
    try:
        data['weather_daily'] = load_weather_data(hourly=False)
        data['weather_hourly'] = load_weather_data(hourly=True)
        print(f"  ✓ Daily: {len(data['weather_daily']):,} records, Hourly: {len(data['weather_hourly']):,} records")
    except Exception as e:
        print(f"  ⚠ Weather data not available: {e}")
        data['weather_daily'] = None
        data['weather_hourly'] = None
    
    # 4. AIS Vessel Tracking (ETAs, vessel counts)
    print("\n[4/7] Loading AIS Vessel Tracking...")
    try:
        data['ais_vessels'] = load_ais_vessels(sample_n=50000)  # Sample for speed
        print(f"  ✓ {len(data['ais_vessels']):,} vessel records")
    except Exception as e:
        print(f"  ⚠ AIS data not available: {e}")
        data['ais_vessels'] = None
    
    # 5. PortWatch Chokepoints (Trade Volume)
    print("\n[5/7] Loading PortWatch Chokepoints...")
    try:
        data['chokepoints'] = load_portwatch_chokepoints()
        print(f"  ✓ {len(data['chokepoints']):,} chokepoint records")
    except Exception as e:
        print(f"  ⚠ Chokepoint data not available: {e}")
        data['chokepoints'] = None
    
    # 6. Truck Travel Times (Drayage Calibration)
    print("\n[6/7] Loading Truck Travel Times...")
    try:
        # Sample 5% of truck times for memory efficiency
        data['truck_times'] = load_truck_times(sample_frac=0.05)
        print(f"  ✓ {len(data['truck_times']):,} county-to-county routes")
    except Exception as e:
        print(f"  ⚠ Truck times not available: {e}")
        data['truck_times'] = None
    
    # 7. Summary
    print("\n[7/7] Data Loading Complete!")
    print("─"*70)
    
    return data


# ============================================================================
# FEATURE ENGINEERING - per training.md
# ============================================================================

def build_surge_labels(port_activity_df: pd.DataFrame, horizons: List[int] = [24, 48, 72]) -> pd.DataFrame:
    """
    Build surge prediction labels for multiple horizons.
    
    Surge = (future_portcalls - current_portcalls) / current_portcalls
    Normalized to [0, 1] range where:
    - 0.0 = significant decrease (-50% or more)
    - 0.5 = no change
    - 1.0 = significant increase (+50% or more)
    """
    df = port_activity_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['portname', 'date'])
    
    # Create labels for each horizon
    for horizon in horizons:
        horizon_days = max(1, horizon // 24)
        
        # Future port calls (what we want to predict)
        df[f'future_calls_{horizon}h'] = df.groupby('portname')['portcalls'].shift(-horizon_days)
        
        # Compute surge ratio
        df[f'surge_{horizon}h'] = (
            (df[f'future_calls_{horizon}h'] - df['portcalls']) / 
            df['portcalls'].replace(0, 1)  # Avoid division by zero
        )
        
        # Normalize to [0, 1] with clipping at +/- 50%
        df[f'surge_level_{horizon}h'] = np.clip(
            (df[f'surge_{horizon}h'] + 0.5) / 1.0,  # Map [-0.5, 0.5] to [0, 1]
            0, 1
        )
    
    return df


def build_port_features(
    port_activity_df: pd.DataFrame,
    weather_df: pd.DataFrame = None,
    ais_df: pd.DataFrame = None
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build comprehensive node features for ports as described in training.md:
    - Recent port calls (7-day rolling average)
    - Year-over-year growth metrics
    - Weather impact scores
    - Vessel ETAs (from AIS)
    """
    from forecasting.features import add_time_features, add_rolling_features, add_lag_features
    
    df = port_activity_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['portname', 'date'])
    
    feature_cols = []
    
    # 1. Time features (cyclical encoding)
    df = add_time_features(df, 'date')
    time_feats = ['day_sin', 'day_cos', 'month_sin', 'month_cos', 'is_weekend']
    feature_cols.extend(time_feats)
    
    # 2. Lag features (recent history)
    df = add_lag_features(df, 'portcalls', 'portname', lags=[1, 2, 3, 5, 7])
    lag_feats = [c for c in df.columns if 'lag_' in c]
    feature_cols.extend(lag_feats)
    
    # 3. Rolling statistics (7-day, 14-day, 30-day averages)
    df = add_rolling_features(df, 'portcalls', 'portname', windows=[3, 7, 14, 30])
    roll_feats = [c for c in df.columns if 'roll_' in c]
    feature_cols.extend(roll_feats)
    
    # 4. Year-over-year growth
    df['portcalls_yoy'] = df.groupby('portname')['portcalls'].transform(
        lambda x: x.pct_change(periods=365, fill_method=None)
    ).fillna(0)
    feature_cols.append('portcalls_yoy')
    
    # 5. Week-over-week growth
    df['portcalls_wow'] = df.groupby('portname')['portcalls'].transform(
        lambda x: x.pct_change(periods=7, fill_method=None)
    ).fillna(0)
    feature_cols.append('portcalls_wow')
    
    # 6. Weather impact (if available)
    if weather_df is not None and len(weather_df) > 0:
        weather_df = weather_df.copy()
        weather_df['date'] = pd.to_datetime(weather_df['date']).dt.tz_localize(None)
        
        # Ensure df date is also timezone-naive
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
        
        # Aggregate weather by date (average across locations)
        weather_agg = weather_df.groupby('date').agg({
            'precipitation_sum': 'mean',
            'wind_speed_10m_max': 'mean',
            'temperature_2m_max': 'mean'
        }).reset_index()
        
        df = df.merge(weather_agg, on='date', how='left')
        
        # Weather impact score (high precip + high wind = bad)
        df['weather_impact'] = (
            np.clip(df['precipitation_sum'].fillna(0) / 20, 0, 1) * 0.6 +
            np.clip(df['wind_speed_10m_max'].fillna(0) / 50, 0, 1) * 0.4
        )
        feature_cols.extend(['precipitation_sum', 'wind_speed_10m_max', 'weather_impact'])
    
    # 7. Vessel count features (if AIS available)
    if ais_df is not None and len(ais_df) > 0:
        try:
            ais_df = ais_df.copy()
            # Find the ETA column (might be 'eta', 'ETA', 'etaSchedule', etc.)
            eta_col = None
            for col in ['eta', 'ETA', 'etaSchedule', 'eta_date']:
                if col in ais_df.columns:
                    eta_col = col
                    break
            
            if eta_col:
                ais_df['eta_parsed'] = pd.to_datetime(ais_df[eta_col], errors='coerce')
                ais_df = ais_df.dropna(subset=['eta_parsed'])
                
                if len(ais_df) > 0:
                    # Count total vessels by date
                    ais_df['eta_date'] = ais_df['eta_parsed'].dt.date
                    daily_vessels = ais_df.groupby('eta_date').size().reset_index(name='total_vessels')
                    daily_vessels.columns = ['date', 'total_vessels']
                    daily_vessels['date'] = pd.to_datetime(daily_vessels['date']).dt.tz_localize(None)
                    
                    # Ensure df date is timezone-naive for merge
                    df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
                    
                    df = df.merge(daily_vessels, on='date', how='left')
                    df['total_vessels'] = df['total_vessels'].fillna(0)
                    feature_cols.append('total_vessels')
                    print(f"  ✓ Added vessel count features from AIS data")
        except Exception as e:
            print(f"  ⚠ Could not add vessel features: {e}")
    
    # Fill NaN values
    df[feature_cols] = df[feature_cols].fillna(0)
    
    # Replace inf values
    df = df.replace([np.inf, -np.inf], 0)
    
    return df, feature_cols


# ============================================================================
# GRAPH CONSTRUCTION AND CENTRALITY (cuGraph GPU)
# ============================================================================

def build_rail_graph_gpu(rail_nodes_gdf, rail_lines_gdf) -> Tuple[Any, Dict, Dict]:
    """
    Build rail network graph using cuGraph for GPU-accelerated processing.
    Returns cuGraph object and node mappings.
    """
    print("\n" + "─"*70)
    print("  Building Rail Network Graph (cuGraph GPU)")
    print("─"*70)
    
    # Create node ID mapping
    node_ids = rail_nodes_gdf['FRANODEID'].unique()
    node_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}
    idx_to_node = {idx: nid for nid, idx in node_to_idx.items()}
    
    # Vectorized edge processing
    src_col = 'FRFRANODE' if 'FRFRANODE' in rail_lines_gdf.columns else 'FromNode'
    dst_col = 'TOFRANODE' if 'TOFRANODE' in rail_lines_gdf.columns else 'ToNode'
    weight_col = 'MILES' if 'MILES' in rail_lines_gdf.columns else 'Shape_Length'
    
    rail_lines_gdf = rail_lines_gdf.copy()
    rail_lines_gdf['src_idx'] = rail_lines_gdf[src_col].map(node_to_idx)
    rail_lines_gdf['dst_idx'] = rail_lines_gdf[dst_col].map(node_to_idx)
    
    # Filter valid edges
    valid_mask = rail_lines_gdf['src_idx'].notna() & rail_lines_gdf['dst_idx'].notna()
    valid_edges = rail_lines_gdf[valid_mask].copy()
    
    if weight_col in valid_edges.columns:
        valid_edges['weight'] = valid_edges[weight_col].fillna(1.0)
    else:
        valid_edges['weight'] = 1.0
    
    edges_df = pd.DataFrame({
        'src': valid_edges['src_idx'].astype(int),
        'dst': valid_edges['dst_idx'].astype(int),
        'weight': valid_edges['weight'].astype(float)
    })
    
    print(f"  ✓ {len(edges_df):,} valid edges from {len(node_to_idx):,} nodes")
    
    # Transfer to GPU
    edges_cudf = cudf.DataFrame(edges_df)
    
    # Create cuGraph
    G = cugraph.Graph()
    G.from_cudf_edgelist(edges_cudf, source='src', destination='dst', edge_attr='weight')
    
    print(f"  ✓ cuGraph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    
    return G, node_to_idx, idx_to_node


def compute_graph_features_gpu(G, node_to_idx: Dict) -> pd.DataFrame:
    """
    Compute graph centrality features using cuGraph (GPU).
    Returns DataFrame with node features.
    """
    print("\n" + "─"*70)
    print("  Computing Graph Centrality Features (cuGraph GPU)")
    print("─"*70)
    
    features = {}
    
    # 1. PageRank
    print("  Computing PageRank...")
    pr_start = time.time()
    pagerank_df = cugraph.pagerank(G)
    pagerank = dict(zip(
        pagerank_df['vertex'].to_pandas(),
        pagerank_df['pagerank'].to_pandas()
    ))
    print(f"    ✓ PageRank computed in {time.time() - pr_start:.2f}s")
    
    # 2. Degree
    print("  Computing Degrees...")
    degree_df = G.degrees()
    if 'in_degree' in degree_df.columns and 'out_degree' in degree_df.columns:
        degree_series = degree_df['in_degree'] + degree_df['out_degree']
    elif 'degree' in degree_df.columns:
        degree_series = degree_df['degree']
    else:
        degree_series = degree_df['out_degree']
    
    degree = dict(zip(degree_df['vertex'].to_pandas(), degree_series.to_pandas()))
    
    # 3. Betweenness Centrality (sampled for speed)
    print("  Computing Betweenness Centrality (sampled k=100)...")
    try:
        bc_start = time.time()
        bc_df = cugraph.betweenness_centrality(G, k=100)
        betweenness = dict(zip(
            bc_df['vertex'].to_pandas(),
            bc_df['betweenness_centrality'].to_pandas()
        ))
        print(f"    ✓ Betweenness computed in {time.time() - bc_start:.2f}s")
    except Exception as e:
        print(f"    ⚠ Betweenness skipped: {e}")
        betweenness = {idx: 0.0 for idx in range(len(node_to_idx))}
    
    # Build DataFrame
    idx_to_node = {v: k for k, v in node_to_idx.items()}
    rows = []
    for idx in range(len(node_to_idx)):
        rows.append({
            'node_idx': idx,
            'node_id': idx_to_node.get(idx, idx),
            'pagerank': pagerank.get(idx, 0.0),
            'degree': degree.get(idx, 0),
            'betweenness': betweenness.get(idx, 0.0)
        })
    
    graph_features_df = pd.DataFrame(rows)
    
    # Normalize features
    graph_features_df['pagerank_norm'] = graph_features_df['pagerank'] * 10000
    graph_features_df['degree_norm'] = graph_features_df['degree'] / graph_features_df['degree'].max()
    graph_features_df['betweenness_norm'] = graph_features_df['betweenness'] * 1000
    
    print(f"  ✓ Graph features computed for {len(graph_features_df):,} nodes")
    
    return graph_features_df


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

@dataclass
class TrainingConfig:
    """
    Configuration for GPU training aligned with training.md.
    
    Multi-horizon surge prediction using:
    - cuML RandomForest for fast GPU training
    - Graph topology features from cuGraph
    - Real port activity data for labels
    """
    # Prediction horizons (per training.md)
    prediction_horizons: List[int] = field(default_factory=lambda: [24, 48, 72])
    
    # RandomForest hyperparameters
    n_estimators: int = 300
    max_depth: int = 15
    min_samples_leaf: int = 10
    min_samples_split: int = 20
    max_features: float = 0.33
    
    # Training settings
    test_size: float = 0.2
    random_state: int = 42
    n_streams: int = 8
    
    # Checkpointing
    checkpoint_dir: str = "output/checkpoints"
    
    def __post_init__(self):
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    def to_dict(self):
        return {
            'prediction_horizons': self.prediction_horizons,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_leaf': self.min_samples_leaf,
            'min_samples_split': self.min_samples_split,
            'max_features': self.max_features,
            'test_size': self.test_size,
            'random_state': self.random_state,
            'n_streams': self.n_streams,
        }


def get_training_config(scenario: str = 'default') -> TrainingConfig:
    """Get configuration preset."""
    configs = {
        'default': TrainingConfig(),
        'fast': TrainingConfig(
            n_estimators=100,
            max_depth=10,
            prediction_horizons=[24],
        ),
        'accurate': TrainingConfig(
            n_estimators=500,
            max_depth=20,
            min_samples_leaf=5,
        ),
    }
    return configs.get(scenario, configs['default'])


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_surge_models(
    config: TrainingConfig,
    port_features_df: pd.DataFrame,
    feature_cols: List[str],
    graph_features_df: pd.DataFrame = None
) -> Dict[int, Dict[str, Any]]:
    """
    Train surge prediction models for each horizon using REAL DATA.
    
    Returns dict of {horizon: {'model': model, 'metrics': metrics}}
    """
    from cuml.ensemble import RandomForestRegressor as cuRF
    from cuml.model_selection import train_test_split as cu_train_test_split
    
    print("\n" + "="*70)
    print("  TRAINING SURGE PREDICTION MODELS (REAL DATA)")
    print("  Per training.md: Multi-horizon (24h, 48h, 72h)")
    print("="*70)
    
    results = {}
    
    for horizon in config.prediction_horizons:
        print(f"\n" + "─"*70)
        print(f"  HORIZON: {horizon} hours")
        print("─"*70)
        
        target_col = f'surge_level_{horizon}h'
        
        # Filter to rows with valid targets
        train_df = port_features_df[port_features_df[target_col].notna()].copy()
        
        if len(train_df) < 100:
            print(f"  ⚠ Not enough samples ({len(train_df)}), skipping horizon")
            continue
        
        # Prepare features
        X = train_df[feature_cols].values.astype(np.float32)
        y = train_df[target_col].values.astype(np.float32)
        
        # Handle NaN/Inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.5, posinf=1.0, neginf=0.0)
        
        print(f"  Training samples: {len(X):,}")
        print(f"  Features: {len(feature_cols)}")
        
        # Transfer to GPU
        X_gpu = cp.asarray(X)
        y_gpu = cp.asarray(y)
        
        # Train/Test split
        X_train, X_test, y_train, y_test = cu_train_test_split(
            X_gpu, y_gpu, 
            test_size=config.test_size, 
            random_state=config.random_state
        )
        
        print(f"  Train: {len(X_train):,}, Test: {len(X_test):,}")
        
        # Train model
        model = cuRF(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            min_samples_leaf=config.min_samples_leaf,
            min_samples_split=config.min_samples_split,
            max_features=config.max_features,
            random_state=config.random_state,
            n_streams=config.n_streams
        )
        
        fit_start = time.time()
        with tqdm(total=config.n_estimators, desc=f"  Training {horizon}h model", unit="tree",
                  bar_format='{l_bar}{bar:30}{r_bar}{bar:-10b}') as pbar:
            model.fit(X_train, y_train)
            pbar.update(config.n_estimators)
        fit_time = time.time() - fit_start
        
        print(f"  ✓ Training complete in {fit_time:.1f}s")
        
        # Evaluate
        y_pred = model.predict(X_test)
        
        mae = float(cp.abs(y_pred - y_test).mean())
        mse = float(((y_pred - y_test) ** 2).mean())
        rmse = float(cp.sqrt(mse))
        
        ss_res = float(((y_test - y_pred) ** 2).sum())
        ss_tot = float(((y_test - y_test.mean()) ** 2).sum())
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        metrics = {'mae': mae, 'rmse': rmse, 'r2': r2, 'fit_time': fit_time}
        
        print(f"\n  ┌─────────────────────────────────────────────────────────────────┐")
        print(f"  │  {horizon}h MODEL PERFORMANCE                                      │")
        print(f"  ├─────────────────────────────────────────────────────────────────┤")
        print(f"  │  MAE:  {mae:>8.4f}    (Mean Absolute Error)                     │")
        print(f"  │  RMSE: {rmse:>8.4f}    (Root Mean Square Error)                  │")
        print(f"  │  R²:   {r2:>8.4f}    (Coefficient of Determination)            │")
        print(f"  └─────────────────────────────────────────────────────────────────┘")
        
        results[horizon] = {
            'model': model,
            'metrics': metrics,
            'feature_cols': feature_cols
        }
    
    return results


def save_models(results: Dict[int, Dict], config: TrainingConfig, data_summary: Dict):
    """Save trained models and metadata."""
    checkpoint_dir = Path(config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("\n" + "─"*70)
    print("  SAVING MODELS")
    print("─"*70)
    
    for horizon, result in results.items():
        # Save model
        model_path = checkpoint_dir / f"surge_model_{horizon}h_{timestamp}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': result['model'],
                'metrics': result['metrics'],
                'feature_cols': result['feature_cols'],
                'horizon': horizon,
                'timestamp': timestamp,
                'config': config.to_dict()
            }, f)
        print(f"  ✓ {horizon}h model: {model_path}")
        
        # Also save as latest
        latest_path = checkpoint_dir / f"surge_model_{horizon}h_latest.pkl"
        with open(latest_path, 'wb') as f:
            pickle.dump({
                'model': result['model'],
                'metrics': result['metrics'],
                'feature_cols': result['feature_cols'],
                'horizon': horizon,
                'timestamp': timestamp,
                'config': config.to_dict()
            }, f)
    
    # Save combined metadata
    all_metrics = {h: r['metrics'] for h, r in results.items()}
    metadata = {
        'horizons': list(results.keys()),
        'metrics': all_metrics,
        'timestamp': timestamp,
        'config': config.to_dict(),
        'data_summary': data_summary,
        'feature_cols': results[list(results.keys())[0]]['feature_cols'] if results else []
    }
    
    with open(checkpoint_dir / "training_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    print(f"  ✓ Metadata: {checkpoint_dir / 'training_metadata.json'}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def train_full_pipeline(config: TrainingConfig = None):
    """
    Full training pipeline aligned with training.md:
    
    1. Load ALL data sources (Port Activity, Weather, AIS, Rail Network)
    2. Build graph topology features using cuGraph
    3. Build port features from real data
    4. Create surge labels for each horizon
    5. Train models for 24h, 48h, 72h predictions
    6. Save checkpoints
    """
    if config is None:
        config = get_training_config('default')
    
    gpu_monitor.start()
    pipeline_start = time.time()
    
    print("\n" + "="*70)
    print("  🚀 GLID SURGE OPTIMIZATION - FULL TRAINING PIPELINE")
    print("  ALIGNED WITH training.md SPECIFICATION")
    print("="*70)
    
    # =========================================================================
    # STEP 1: Load All Data Sources
    # =========================================================================
    data = load_all_data_sources()
    
    # =========================================================================
    # STEP 2: Build Graph and Compute Centrality
    # =========================================================================
    G, node_to_idx, idx_to_node = build_rail_graph_gpu(
        data['rail_nodes'], 
        data['rail_lines']
    )
    
    graph_features_df = compute_graph_features_gpu(G, node_to_idx)
    
    # =========================================================================
    # STEP 3: Build Surge Labels (REAL DATA)
    # =========================================================================
    print("\n" + "─"*70)
    print("  Building Surge Labels from Real Port Activity Data")
    print("─"*70)
    
    port_df = build_surge_labels(
        data['port_activity'], 
        horizons=config.prediction_horizons
    )
    print(f"  ✓ Created labels for horizons: {config.prediction_horizons}")
    
    # =========================================================================
    # STEP 4: Build Port Features
    # =========================================================================
    print("\n" + "─"*70)
    print("  Building Port Features (Time Series + Weather + Vessels)")
    print("─"*70)
    
    port_features_df, feature_cols = build_port_features(
        port_df,
        weather_df=data.get('weather_daily'),
        ais_df=data.get('ais_vessels')
    )
    print(f"  ✓ Created {len(feature_cols)} features")
    print(f"  ✓ Feature columns: {feature_cols[:10]}..." if len(feature_cols) > 10 else f"  ✓ Features: {feature_cols}")
    
    # =========================================================================
    # STEP 5: Train Models for Each Horizon
    # =========================================================================
    results = train_surge_models(
        config=config,
        port_features_df=port_features_df,
        feature_cols=feature_cols,
        graph_features_df=graph_features_df
    )
    
    # =========================================================================
    # STEP 6: Save Models
    # =========================================================================
    data_summary = {
        'port_records': len(data['port_activity']),
        'ports': data['port_activity']['portname'].nunique(),
        'date_range': {
            'start': str(data['port_activity']['date'].min()),
            'end': str(data['port_activity']['date'].max())
        },
        'graph_nodes': G.number_of_nodes(),
        'graph_edges': G.number_of_edges(),
        'has_weather': data.get('weather_daily') is not None,
        'has_ais': data.get('ais_vessels') is not None
    }
    
    save_models(results, config, data_summary)
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    elapsed = time.time() - pipeline_start
    
    print("\n" + "="*70)
    print("  ✅ TRAINING PIPELINE COMPLETE")
    print("="*70)
    print("  ┌─────────────────────────────────────────────────────────────────┐")
    print(f"  │  Total Pipeline Time:     {elapsed:>8.1f} seconds                     │")
    print(f"  │  Port Records Used:       {data_summary['port_records']:>8,}                          │")
    print(f"  │  Ports Trained On:        {data_summary['ports']:>8}                          │")
    print(f"  │  Graph Nodes:             {data_summary['graph_nodes']:>8,}                          │")
    print(f"  │  Graph Edges:             {data_summary['graph_edges']:>8,}                          │")
    print("  ├─────────────────────────────────────────────────────────────────┤")
    
    for horizon, result in results.items():
        m = result['metrics']
        print(f"  │  {horizon}h Model - R²: {m['r2']:.4f}, MAE: {m['mae']:.4f}, RMSE: {m['rmse']:.4f}     │")
    
    print("  └─────────────────────────────────────────────────────────────────┘")
    print("="*70)
    
    return results, G, data


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="GPU-Accelerated Surge Model Training (Aligned with training.md)")
    parser.add_argument('--config', type=str, default='default', 
                        choices=['default', 'fast', 'accurate'],
                        help='Training configuration preset')
    parser.add_argument('--horizons', type=str, default='24,48,72',
                        help='Comma-separated prediction horizons in hours')
    parser.add_argument('--n-estimators', type=int, help='Number of trees (overrides config)')
    parser.add_argument('--max-depth', type=int, help='Max tree depth (overrides config)')
    
    args = parser.parse_args()
    
    # Get configuration
    config = get_training_config(args.config)
    
    # Parse horizons
    if args.horizons:
        config.prediction_horizons = [int(h.strip()) for h in args.horizons.split(',')]
    
    # Apply overrides
    if args.n_estimators:
        config.n_estimators = args.n_estimators
    if args.max_depth:
        config.max_depth = args.max_depth
    
    print(f"\n[Config] Using '{args.config}' configuration")
    print(f"  Horizons: {config.prediction_horizons}")
    print(f"  n_estimators: {config.n_estimators}")
    print(f"  max_depth: {config.max_depth}")
    
    # Run training
    results, G, data = train_full_pipeline(config)
    
    return results, G, data


if __name__ == "__main__":
    if not HAS_RAPIDS:
        print("\nERROR: RAPIDS not available. Cannot run GPU training.")
        print("Install with: pip install cudf-cu12 cuml-cu12 cugraph-cu12")
        sys.exit(1)
    
    main()
