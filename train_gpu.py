#!/usr/bin/env python3
"""
GPU-Accelerated Training Script for Glid Surge Optimization
============================================================
Uses NVIDIA RAPIDS (cuGraph, cuML, cuDF) on DGX Spark with 128GB memory.

Usage:
    export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH
    python train_gpu.py
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Set CUDA library path
os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda-13.0/lib64:/usr/local/cuda-13.0/targets/sbsa-linux/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
import pandas as pd
from tqdm import tqdm

# Check GPU availability
print("="*60)
print("GLID SURGE OPTIMIZATION - GPU TRAINING")
print("="*60)

# Import RAPIDS
try:
    import cudf
    import cuml
    import cugraph
    import cupy as cp
    print(f"✓ cuDF: {cudf.__version__}")
    print(f"✓ cuML: {cuml.__version__}")
    print(f"✓ cuGraph: {cugraph.__version__}")
    print(f"✓ CuPy: {cp.__version__}")
    HAS_RAPIDS = True
except ImportError as e:
    print(f"✗ RAPIDS not available: {e}")
    HAS_RAPIDS = False

# Import PyTorch
try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    HAS_TORCH = True
except ImportError:
    print("✗ PyTorch not available")
    HAS_TORCH = False

print("="*60)


def load_rail_network_gpu():
    """Load rail network into cuGraph for GPU-accelerated processing."""
    from data.loaders import load_rail_nodes, load_rail_lines
    
    print("\n[1/4] Loading rail network...")
    nodes_gdf = load_rail_nodes(filter_us_only=True)
    lines_gdf = load_rail_lines(filter_us_only=True)
    
    print(f"  Nodes: {len(nodes_gdf):,}")
    print(f"  Lines: {len(lines_gdf):,}")
    
    # Convert to cuDF for GPU processing
    print("\n[2/4] Converting to GPU DataFrames...")
    
    # Create node ID mapping
    node_ids = nodes_gdf['FRANODEID'].unique()
    node_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}
    
    # Create edge list for cuGraph
    edges = []
    for _, row in tqdm(lines_gdf.iterrows(), total=len(lines_gdf), desc="  Processing edges"):
        src = node_to_idx.get(row.get('FRFRANODE', row.get('FromNode', None)))
        dst = node_to_idx.get(row.get('TOFRANODE', row.get('ToNode', None)))
        if src is not None and dst is not None:
            miles = row.get('MILES', row.get('Shape_Length', 1.0))
            if pd.isna(miles):
                miles = 1.0
            edges.append({
                'src': src,
                'dst': dst,
                'weight': float(miles)
            })
    
    edges_df = pd.DataFrame(edges)
    print(f"  Valid edges: {len(edges_df):,}")
    
    # Convert to cuDF
    edges_cudf = cudf.DataFrame(edges_df)
    
    # Create cuGraph
    print("\n[3/4] Building cuGraph on GPU...")
    G = cugraph.Graph()
    G.from_cudf_edgelist(edges_cudf, source='src', destination='dst', edge_attr='weight')
    
    print(f"  cuGraph nodes: {G.number_of_nodes():,}")
    print(f"  cuGraph edges: {G.number_of_edges():,}")
    
    return G, nodes_gdf, node_to_idx


def compute_centrality_gpu(G):
    """Compute graph centrality metrics on GPU - ALL STAYS ON GPU."""
    print("\n[4/4] Computing centrality on GPU...")
    
    # PageRank (GPU-accelerated)
    print("  Computing PageRank...")
    pagerank = cugraph.pagerank(G)
    
    # Betweenness Centrality (sampled for speed)
    print("  Computing betweenness centrality (sampled)...")
    try:
        betweenness = cugraph.betweenness_centrality(G, k=100)
    except Exception as e:
        print(f"    Skipping betweenness: {e}")
        betweenness = None
    
    # Degree
    print("  Computing degree...")
    degree = G.degrees()
    
    return {
        'pagerank': pagerank,
        'betweenness': betweenness,
        'degree': degree
    }


def train_surge_model_gpu(centrality_data):
    """Train surge prediction model using cuML RandomForest - ALL ON GPU."""
    print("\n" + "="*60)
    print("TRAINING SURGE PREDICTION MODEL")
    print("  Graph features: cuGraph (GPU)")
    print("  ML model: cuML RandomForest (GPU)")
    print("  Data: cuDF (GPU) - NO CPU TRANSFER")
    print("="*60)
    
    import cupy as cp
    from cuml.ensemble import RandomForestRegressor as cuRF
    from cuml.model_selection import train_test_split as cu_train_test_split
    
    # Keep all data on GPU - NO .to_numpy() calls
    print("\n[1/3] Preparing GPU training data...")
    
    # Get cuGraph-computed centrality features (stay on GPU via cuDF)
    pagerank_series = centrality_data['pagerank']['pagerank']
    degree_df = centrality_data['degree']
    degree_series = degree_df['in_degree'] + degree_df['out_degree']
    
    n_nodes = len(pagerank_series)
    print(f"  Graph nodes with features: {n_nodes:,}")
    
    # Get betweenness if available
    if centrality_data['betweenness'] is not None:
        betweenness_series = centrality_data['betweenness']['betweenness_centrality']
    else:
        betweenness_series = cudf.Series(cp.zeros(n_nodes))
    
    # Generate training data directly on GPU using cupy
    print("\n[2/3] Generating training samples on GPU...")
    n_samples = 50000
    
    # Random indices on GPU
    cp.random.seed(42)
    indices = cp.random.randint(0, n_nodes, size=n_samples)
    
    # Convert cuDF series to cupy arrays (stays on GPU)
    pr_gpu = cp.asarray(pagerank_series.values)
    deg_gpu = cp.asarray(degree_series.values)
    btw_gpu = cp.asarray(betweenness_series.values)
    
    # Sample features from graph centrality
    pr_samples = pr_gpu[indices] * 10000
    deg_samples = deg_gpu[indices].astype(cp.float32)
    btw_samples = btw_gpu[indices] * 1000
    
    # Generate time features on GPU
    hours = cp.random.randint(0, 24, size=n_samples).astype(cp.float32)
    days = cp.random.randint(0, 7, size=n_samples).astype(cp.float32)
    months = cp.random.randint(1, 13, size=n_samples).astype(cp.float32)
    
    hour_sin = cp.sin(2 * cp.pi * hours / 24)
    hour_cos = cp.cos(2 * cp.pi * hours / 24)
    day_sin = cp.sin(2 * cp.pi * days / 7)
    day_cos = cp.cos(2 * cp.pi * days / 7)
    is_weekend = (days >= 5).astype(cp.float32)
    is_business = ((hours >= 6) & (hours <= 18)).astype(cp.float32)
    
    # Stack features (all on GPU)
    X_gpu = cp.column_stack([
        pr_samples, deg_samples, btw_samples,
        hours, days, months,
        hour_sin, hour_cos, day_sin, day_cos,
        is_weekend, is_business
    ]).astype(cp.float32)
    
    # Generate target on GPU
    y_gpu = (30 + pr_samples * 5 + deg_samples * 0.3 - btw_samples * 0.1 +
             cp.sin(hours / 24 * cp.pi) * 15 +
             is_weekend * -10 +
             cp.random.randn(n_samples).astype(cp.float32) * 5)
    y_gpu = cp.maximum(0, y_gpu)
    
    print(f"  Training data shape: {X_gpu.shape}")
    print(f"  Data location: GPU (cupy)")
    
    # Train/test split on GPU
    X_train, X_test, y_train, y_test = cu_train_test_split(
        X_gpu, y_gpu, test_size=0.2, random_state=42
    )
    
    # Train cuML RandomForest on GPU
    print("\n[3/3] Training cuML RandomForest on GPU...")
    model = cuRF(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_streams=4
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate on GPU
    y_pred = model.predict(X_test)
    
    # Compute metrics on GPU
    mae = float(cp.abs(y_pred - y_test).mean())
    mse = float(((y_pred - y_test) ** 2).mean())
    rmse = float(cp.sqrt(mse))
    
    ss_res = float(((y_test - y_pred) ** 2).sum())
    ss_tot = float(((y_test - y_test.mean()) ** 2).sum())
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    print(f"\n  Model Performance (GPU):")
    print(f"    MAE:  {mae:.2f}")
    print(f"    RMSE: {rmse:.2f}")
    print(f"    R²:   {r2:.3f}")
    
    # Feature importance
    feature_names = [
        'pagerank', 'degree', 'betweenness',
        'hour', 'day', 'month',
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
        'is_weekend', 'is_business_hours'
    ]
    
    print(f"\n  Feature Importance:")
    try:
        importances = model.feature_importances_
        importance_dict = dict(zip(feature_names, importances))
        for feat, imp in sorted(importance_dict.items(), key=lambda x: -x[1])[:5]:
            print(f"    {feat}: {imp:.3f}")
    except Exception as e:
        print(f"    (Feature importance not available: {e})")
    
    return model, {'mae': mae, 'rmse': rmse, 'r2': r2}


def train_with_cugraph_features():
    """Main training pipeline using cuGraph for feature extraction."""
    start_time = datetime.now()
    
    # Load rail network into cuGraph
    G, nodes_gdf, node_to_idx = load_rail_network_gpu()
    
    # Compute centrality features on GPU
    centrality = compute_centrality_gpu(G)
    
    # Train surge model
    model, metrics = train_surge_model_gpu(centrality)
    
    # Summary
    elapsed = (datetime.now() - start_time).total_seconds()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"  Total time: {elapsed:.1f} seconds")
    print(f"  Graph nodes: {G.number_of_nodes():,}")
    print(f"  Graph edges: {G.number_of_edges():,}")
    print(f"  Model R²: {metrics['r2']:.3f}")
    print("="*60)
    
    return model, G, centrality


if __name__ == "__main__":
    if not HAS_RAPIDS:
        print("\nERROR: RAPIDS not available. Cannot run GPU training.")
        print("Install with: pip install cudf-cu12 cuml-cu12 cugraph-cu12")
        sys.exit(1)
    
    model, G, centrality = train_with_cugraph_features()

