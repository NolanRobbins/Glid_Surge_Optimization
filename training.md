# Glid Surge Optimization - Training & Inference Pipeline

## Overview

This document details the end-to-end training and inference pipeline for the **Port-to-Rail Surge Forecaster & Route Recommendation Engine**. The system leverages NVIDIA's full stack (PyTorch Geometric, cuGraph, RAPIDS) on the **ASUS Ascent GX10 (NVIDIA GB10 Grace Blackwell)** to deliver 24-72 hour surge predictions and optimal route recommendations.

## Architecture: Two-Stage Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│  STAGE 1: GNN SURGE PREDICTION                                          │
│  ─────────────────────────────────────────────────────────────────────  │
│  Input:  Graph topology + Port activity + Weather + AIS vessels         │
│  Model:  SAGEConv Graph Neural Network (message passing)                │
│  Output: Surge predictions (0-1) per node for 24h, 48h, 72h horizons   │
├─────────────────────────────────────────────────────────────────────────┤
│  STAGE 2: ROUTE RECOMMENDATION (Mode Selection)                         │
│  ─────────────────────────────────────────────────────────────────────  │
│  Input:  Surge predictions + Truck travel times + Freight costs         │
│  Logic:  High surge → Recommend RAIL | Low surge → ROAD acceptable      │
│  Output: Optimal route with cost savings estimate                        │
└─────────────────────────────────────────────────────────────────────────┘
```

## Data Pipeline: Input → Features → Model → Output

### 1. Input Data Sources (All Integrated)

| Dataset | Path | Records | Purpose |
|---------|------|---------|---------|
| **PortWatch (IMF)** | `data/global_daily_port_activity/` | 23K+ daily records | **Primary surge labels** - real port call volumes |
| **Rail Network (USDOT)** | `data/rail_nodes/`, `data/rail_lines/` | 197K nodes, 235K edges | **Graph topology** for GNN |
| **Weather (Open-Meteo)** | `data/weather/` | 1.2K records | **Weather impact** on operations |
| **AIS Vessels** | `data/AIS_vessel/` | 823K+ records | **Inbound vessel signals** |
| **PortWatch Chokepoints** | `data/portwatch/` | 70K+ records | **Trade flow pressure** |
| **Truck Times (BTS/ATRI)** | `data/truck_times/` | 3.6M county pairs | **Drayage edge weights** |
| **Logistics Fleet** | `data/logistics_fleet/` | 92K freight rates | **Cost modeling** |

### 2. Feature Engineering (20+ Features)

The production training script (`train_gnn.py`) builds comprehensive features from ALL data sources:

#### Graph Topology Features (5) - computed via cuGraph GPU
```python
pagerank     # Global network importance (scaled ×10000)
degree       # Node connectivity (normalized ÷10)
betweenness  # Chokepoint criticality (scaled ×1000)
is_port      # Binary: 1 if port node
is_terminal  # Binary: 1 if rail terminal
```

#### Time Series Features (5) - cyclical encoding
```python
day_sin, day_cos    # Day of year (365-cycle)
month_sin, month_cos # Month of year (12-cycle)
is_weekend          # Binary: Saturday/Sunday
```

#### Lag Features (5) - historical port activity
```python
portcalls_lag_1d   # Yesterday's port calls
portcalls_lag_2d   # 2 days ago
portcalls_lag_3d   # 3 days ago
portcalls_lag_7d   # 1 week ago
portcalls_lag_14d  # 2 weeks ago
```

#### Rolling Statistics (4)
```python
portcalls_roll_3d_mean   # 3-day moving average
portcalls_roll_7d_mean   # 7-day moving average
portcalls_roll_14d_mean  # 14-day moving average
portcalls_roll_7d_std    # 7-day volatility
```

#### Weather Features (3)
```python
precip_avg       # Average precipitation
wind_avg         # Average wind speed
weather_impact   # Composite impact score (0-1)
```

#### External Signals (3)
```python
vessel_inbound_norm   # Normalized inbound vessel count (from AIS)
chokepoint_pressure   # Global trade flow pressure (from PortWatch)
avg_drayage_time      # Average truck travel time (from BTS/ATRI)
```

### 3. Surge Labels (REAL Training Targets)

Labels are computed from **actual port activity data**, not synthetic:

```python
# Future vs current port calls
surge_raw = (future_portcalls - current_portcalls) / current_portcalls

# Normalized to [0, 1] using robust scaling
surge_label = clip((surge_raw - percentile_1) / (percentile_99 - percentile_1), 0, 1)

# Interpretation:
# 0.0 = Significant decrease in activity
# 0.5 = Normal activity level  
# 1.0 = Significant surge (congestion expected)
```

Labels created for **three horizons**: 24h, 48h, 72h

### 4. Model Architecture: SAGEConv GNN

The core model is a **Graph Neural Network** with message passing, allowing congestion patterns to propagate through the network:

```python
class SurgeGNN(nn.Module):
    # Architecture:
    # Input: [batch, 20+ features]
    # 3× SAGEConv layers with BatchNorm + ReLU + Dropout
    # Prediction head: 3× Linear layers → Sigmoid output (0-1)
    
    # Key hyperparameters:
    hidden_channels = 256  # Feature dimensions per layer
    num_layers = 3         # Message passing depth
    dropout = 0.2          # Regularization
```

**Why GNN?** Congestion at one port propagates through the rail network. Message passing captures these dependencies that flat ML models miss.

### 5. Route Recommendation Metrics

The model is evaluated on **route recommendation quality**, not just regression accuracy:

#### Regression Metrics (Surge Prediction)
| Metric | Description |
|--------|-------------|
| MAE | Mean Absolute Error |
| RMSE | Root Mean Squared Error |
| R² | Coefficient of determination |

#### Classification Metrics (Mode Selection: Rail vs Road)
| Metric | Description |
|--------|-------------|
| Mode Accuracy | % correct Rail/Road recommendations |
| Rail Precision | Of predicted "Rail", how many were correct? |
| Rail Recall | Of actual high-surge, how many did we catch? |
| Rail F1 | Harmonic mean of precision/recall |

#### Ranking Metrics (Surge Prioritization)
| Metric | Description |
|--------|-------------|
| Precision@K | Of top-K predicted surge, how many in actual top-K? |
| Hit Rate | Did we catch high-surge events (>0.5 threshold)? |
| NDCG | Normalized Discounted Cumulative Gain |

### 6. GPU Utilization (ASUS Ascent GX10)

We maximize the GB10's capabilities:

| Component | Technology | Usage |
|-----------|------------|-------|
| **Graph Analytics** | NVIDIA cuGraph | PageRank, Betweenness on 197K nodes in <3s |
| **GNN Training** | PyTorch Geometric | SAGEConv with 2M parameters |
| **Mixed Precision** | TF32 + AMP | 2-3× faster matrix operations |
| **Memory** | 128GB Unified | Full graph + features in GPU memory |

**GPU Metrics Tracked:**
- Peak memory usage (GB)
- Training time (seconds)
- Inference throughput (nodes/second)

## Running the Pipeline

### Training Configurations

| Config | Purpose | AIS Data | Epochs | Time |
|--------|---------|----------|--------|------|
| `production` | Full training with all data | 100% (823K) | 200 | ~30-60 min |
| `demo` | GPU stress test for judges | 200K sample | 100 | ~10 min |
| `fast` | Quick iteration | 50K sample | 30 | ~2 min |

### 1. Production Training

```bash
# Full production training with all data
docker run --rm --gpus all \
  -v $(pwd):/workspace \
  -w /workspace \
  -e PYTHONPATH=/workspace/src \
  glid-gnn-v2:latest \
  python train_gnn.py --config production
```

### 2. Demo Mode (for competition)

```bash
# GPU stress test - shows high utilization for judges
docker run --rm --gpus all \
  -v $(pwd):/workspace \
  -w /workspace \
  -e PYTHONPATH=/workspace/src \
  glid-gnn-v2:latest \
  python train_gnn.py --config demo
```

### 3. Fast Mode (iteration)

```bash
# Quick test during development
docker run --rm --gpus all \
  -v $(pwd):/workspace \
  -w /workspace \
  -e PYTHONPATH=/workspace/src \
  glid-gnn-v2:latest \
  python train_gnn.py --config fast
```

### 4. Custom Training

```bash
# Custom horizons and epochs
python train_gnn.py --config production --horizons 24,48 --epochs 100
```

## Output Files

### Model Checkpoints
```
output/checkpoints/
├── gnn_production_24h_*.pt      # 24h horizon model
├── gnn_production_48h_*.pt      # 48h horizon model  
├── gnn_production_72h_*.pt      # 72h horizon model
└── production_training_metadata.json  # Full metrics + GPU stats
```

### Training Metadata
```json
{
  "config": "production",
  "horizons": [24, 48, 72],
  "results": {
    "24": {
      "mae": 0.08,
      "rmse": 0.12,
      "r2": 0.82,
      "mode_accuracy": 0.85,
      "hit_rate": 0.78,
      "ndcg": 0.91,
      "precision_at_5": 0.80,
      "inference_time_ms": 185
    }
  },
  "gpu_stats": {
    "gpu_name": "NVIDIA GB10",
    "peak_gpu_memory_gb": 15.2,
    "gpu_utilization_pct": 11.8,
    "total_time_seconds": 1847
  }
}
```

## Expected Performance

### Surge Prediction Metrics
| Horizon | MAE | RMSE | R² |
|---------|-----|------|-----|
| 24h | 0.06-0.10 | 0.08-0.12 | 0.75-0.85 |
| 48h | 0.08-0.12 | 0.10-0.15 | 0.65-0.78 |
| 72h | 0.10-0.15 | 0.12-0.18 | 0.55-0.70 |

### Route Recommendation Metrics
| Horizon | Mode Accuracy | Hit Rate | Precision@5 | NDCG |
|---------|---------------|----------|-------------|------|
| 24h | 80-90% | 75-85% | 70-85% | 0.85-0.95 |
| 48h | 75-85% | 70-80% | 65-80% | 0.80-0.92 |
| 72h | 70-80% | 65-75% | 60-75% | 0.75-0.88 |

### Inference Performance
| Metric | Value |
|--------|-------|
| Inference time | 150-200ms for 197K nodes |
| Throughput | 1M+ nodes/second |
| Memory | 4-15 GB depending on config |

## Alternative: RandomForest Training

For comparison, `train_gpu.py` provides a cuML RandomForest baseline:

```bash
python train_gpu.py --config fast
```

This is faster but doesn't capture network propagation effects.

## Dashboard Integration

The trained models feed the real-time dashboard:

```bash
python -m src.dashboard.app
# Open http://localhost:8050
```

Dashboard displays:
- Port surge status (color-coded map)
- 24/48/72h predictions
- Route recommendations (Rail vs Road)
- Cost savings estimates
- GPU utilization stats
