"""
Production GNN Inference Script
===============================
Loads trained GNN model and generates dashboard payload with:
- Surge predictions (24, 48, 72h horizons)
- Optimal dispatch windows
- Route optimization recommendations

Usage:
    python run_production_inference.py
    python run_production_inference.py --model-path models/gnn_model.pt
    python run_production_inference.py --export output/dashboard_payload.json
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import MODELS_DIR, OUTPUT_DIR, US_PORTS

# Check for PyTorch first (required for GNN models)
# This MUST happen before importing gnn_model.py which defines PyTorch-dependent classes
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
    print(f"[Inference] PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"[Inference] CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("[Inference] Using CPU")
except ImportError as e:
    HAS_TORCH = False
    print("\n" + "="*80)
    print("  ❌ ERROR: PyTorch is not available")
    print("="*80)
    print("\n  GNN inference requires PyTorch and torch-geometric.")
    print("\n  Install with:")
    print("    pip install torch torch-geometric")
    print("\n  Or for CPU-only:")
    print("    pip install torch torch-geometric --index-url https://download.pytorch.org/whl/cpu")
    print("="*80 + "\n")
    sys.exit(1)

# Now safe to import GNN model components
try:
    from torch_geometric.data import Data
    from torch_geometric.nn import SAGEConv, BatchNorm
    import torch.nn.functional as F
    from forecasting.gnn_model import (
        SurgeGNNModel, GNNConfig, 
        _normalize_name, _activity_port_to_component_port_names, 
        _build_name_to_graph_node_id
    )
    
    # Define SurgeGNN matching the PRODUCTION training architecture (from train_gnn.py)
    # This MUST match the model used during training
    class SurgeGNN(nn.Module):
        """SAGEConv GNN for surge prediction - matches train_gnn.py architecture."""
        
        def __init__(self, in_channels: int, hidden_channels: int = 256, 
                     out_channels: int = 1, num_layers: int = 3, dropout: float = 0.2):
            super().__init__()
            self.num_layers = num_layers
            self.dropout = dropout
            
            self.convs = nn.ModuleList()
            self.norms = nn.ModuleList()
            
            self.convs.append(SAGEConv(in_channels, hidden_channels))
            self.norms.append(BatchNorm(hidden_channels))
            
            for _ in range(num_layers - 1):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))
                self.norms.append(BatchNorm(hidden_channels))
            
            # Predictor architecture MUST match training exactly
            self.predictor = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, hidden_channels // 2),
                nn.ReLU(),
                nn.Linear(hidden_channels // 2, out_channels),
                nn.Sigmoid()
            )
        
        def forward(self, x, edge_index):
            for conv, norm in zip(self.convs, self.norms):
                x = conv(x, edge_index)
                x = norm(x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            return self.predictor(x)
    
    HAS_GNN_MODEL = True
except ImportError as e:
    print(f"⚠ ERROR: Failed to import GNN components: {e}")
    print("   Install missing dependencies:")
    print("   pip install torch torch-geometric")
    sys.exit(1)
except NameError as e:
    print(f"⚠ ERROR: Name error in GNN model: {e}")
    print("   This may indicate a missing PyTorch dependency.")
    sys.exit(1)
from forecasting.features import build_forecasting_features
from data.loaders import (
    load_rail_nodes, load_rail_lines, load_port_activity,
    load_weather_data, load_truck_times
)
from graph.builder import build_rail_graph, add_location_nodes, connect_locations_to_graph
from optimization.dispatcher import DispatchScheduler
from optimization.cost_calculator import CostCalculator
from graph.route_adapter import RouteAdapter


def _json_sanitize(x: Any) -> Any:
    """Convert numpy types and NaN/Inf into JSON-safe primitives."""
    import math
    if x is None:
        return None
    if isinstance(x, (str, int, bool)):
        return x
    if isinstance(x, float):
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    if isinstance(x, (np.floating, np.integer)):
        return _json_sanitize(x.item())
    if isinstance(x, (list, tuple)):
        return [_json_sanitize(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _json_sanitize(v) for k, v in x.items()}
    return str(x)


def load_production_ports() -> List[str]:
    """Get list of top 11 production ports."""
    # Top 11 ports from production training
    return [
        "Los Angeles-Long Beach",
        "New York-New Jersey", 
        "Savannah",
        "Houston",
        "Oakland",
        "Seattle",
        "Tacoma",
        "Virginia",
        "Charleston",
        "Miami",
        "New Orleans"
    ]


def build_inference_graph(ports: List[str]) -> tuple:
    """
    Build transportation graph with ports connected.
    
    Returns:
        (graph, name_to_node_id_map)
    """
    print("\n[1/3] Loading rail network...")
    rail_nodes = load_rail_nodes(filter_us_only=True)
    rail_lines = load_rail_lines(filter_us_only=True)
    
    print("[2/3] Building rail graph...")
    G = build_rail_graph(rail_nodes, rail_lines)
    
    # Add port nodes
    print("[3/3] Adding port nodes to graph...")
    component_port_names = []
    for ap in ports:
        component_port_names.extend(_activity_port_to_component_port_names(ap))
    component_port_names = list(dict.fromkeys(component_port_names))  # unique
    
    ports_dict = {p["name"]: p for p in US_PORTS if p["name"] in component_port_names}
    G = add_location_nodes(G, ports_dict, "port")
    G = connect_locations_to_graph(G, ports_dict, "port", max_connection_miles=20)
    
    name_to_node = _build_name_to_graph_node_id(G)
    
    print(f"  ✓ Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    print(f"  ✓ Ports mapped: {len(ports_dict)}")
    
    return G, name_to_node


def prepare_inference_features(
    port_df: pd.DataFrame,
    graph: Any,
    name_to_node: Dict[str, str],
    ports: List[str],
    target_in_channels: int = 25
) -> tuple:
    """
    Prepare node features for GNN inference.
    MUST match training feature computation exactly (train_gnn.py).
    
    Training uses:
    - 5 graph features: PageRank, Degree, Betweenness, IsPort, IsTerminal
    - 20 port features: from build_comprehensive_features
    
    Args:
        port_df: Port activity data
        graph: NetworkX graph
        name_to_node: Port name to node ID mapping
        ports: List of port names
        target_in_channels: Expected feature dimension (from checkpoint)
    
    Returns:
        (node_features_dict, feature_dim, port_feature_cols)
    """
    print("\n[1/3] Computing graph features (matching training)...")
    
    # Compute graph centrality features - EXACTLY as train_gnn.py does
    node_list = list(graph.nodes())
    node_to_idx = {n: i for i, n in enumerate(node_list)}
    
    # Try cuGraph if available, else CPU
    try:
        import cugraph
        import cudf
        HAS_CUGRAPH = True
    except ImportError:
        HAS_CUGRAPH = False
    
    if HAS_CUGRAPH:
        print("  Using NVIDIA cuGraph (GPU)...")
        edges = list(graph.edges())
        source = [node_to_idx[u] for u, v in edges]
        destination = [node_to_idx[v] for u, v in edges]
        
        gdf = cudf.DataFrame({'source': source, 'destination': destination})
        cu_G = cugraph.Graph()
        cu_G.from_cudf_edgelist(gdf, source='source', destination='destination')
        
        pr_df = cugraph.pagerank(cu_G)
        pr_dict = dict(zip(pr_df['vertex'].to_pandas(), pr_df['pagerank'].to_pandas()))
        
        deg_df = cu_G.degrees()
        if 'in_degree' in deg_df.columns:
            deg_series = deg_df['in_degree'] + deg_df['out_degree']
        else:
            deg_series = deg_df['degree']
        deg_dict = dict(zip(deg_df['vertex'].to_pandas(), deg_series.to_pandas()))
        
        try:
            bc_df = cugraph.betweenness_centrality(cu_G, k=100)
            bc_dict = dict(zip(bc_df['vertex'].to_pandas(), bc_df['betweenness_centrality'].to_pandas()))
        except Exception:
            bc_dict = {i: 0.0 for i in range(len(node_list))}
    else:
        print("  Using NetworkX (CPU)...")
        import networkx as nx
        pr_raw = nx.pagerank(graph, max_iter=50)
        pr_dict = {node_to_idx[n]: v for n, v in pr_raw.items()}
        deg_dict = {node_to_idx[n]: graph.degree(n) for n in node_list}
        bc_dict = {i: 0.0 for i in range(len(node_list))}
    
    # Build 5-dim graph features matching train_gnn.py
    graph_feat_dim = 5
    graph_features = {}
    for node in node_list:
        idx = node_to_idx[node]
        node_data = graph.nodes[node]
        node_type = node_data.get('node_type', 'rail_node')
        
        graph_features[node] = np.array([
            pr_dict.get(idx, 0.0) * 10000,   # PageRank scaled
            deg_dict.get(idx, 0) / 10,        # Degree normalized  
            bc_dict.get(idx, 0.0) * 1000,     # Betweenness scaled
            1.0 if node_type == 'port' else 0.0,
            1.0 if node_type == 'terminal' else 0.0,
        ], dtype=np.float32)
    
    print(f"  ✓ Graph features: {len(graph_features):,} nodes × {graph_feat_dim} features")
    
    # Compute port features
    print("\n[2/3] Building port features...")
    weather_df = None
    try:
        weather_df = load_weather_data(hourly=False)
    except Exception as e:
        print(f"  Warning: Weather unavailable ({e})")
    
    feat_df, feat_cols = build_forecasting_features(
        port_df, weather_df, target_col="portcalls", group_col="portname"
    )
    feat_df = feat_df.sort_values(["portname", "date"])
    
    # Get latest features for each port
    latest_date_by_port = feat_df.groupby("portname")["date"].max()
    latest_feat_rows = []
    for port in ports:
        if port not in latest_date_by_port:
            continue
        max_d = latest_date_by_port[port]
        row = feat_df[(feat_df["portname"] == port) & (feat_df["date"] == max_d)]
        if not row.empty:
            latest_feat_rows.append(row.iloc[0])
    
    if not latest_feat_rows:
        raise ValueError("No recent port data found for inference")
    
    latest_df = pd.DataFrame(latest_feat_rows).set_index("portname")
    print(f"  ✓ Inference snapshot: {latest_df['date'].min().date()} to {latest_df['date'].max().date()}")
    
    # Port feature dimension should match training
    port_feat_dim = target_in_channels - graph_feat_dim
    if port_feat_dim < 0:
        port_feat_dim = 0
    
    # Select matching feature columns (first port_feat_dim features)
    available_feat_cols = [c for c in feat_cols if c in latest_df.columns]
    if len(available_feat_cols) > port_feat_dim:
        selected_feat_cols = available_feat_cols[:port_feat_dim]
    else:
        selected_feat_cols = available_feat_cols
    
    print(f"  ✓ Port features: {len(selected_feat_cols)} of {len(feat_cols)} available")
    
    # Build combined feature matrix
    print("\n[3/3] Building combined feature matrix...")
    
    total_feat_dim = graph_feat_dim + port_feat_dim
    print(f"  Target dimension: {target_in_channels} (graph={graph_feat_dim} + port={port_feat_dim})")
    
    # Mapping from port data names to graph names
    PORT_DATA_TO_GRAPH = {
        "Los Angeles-Long Beach": "Port of Los Angeles",
        "New York-New Jersey": "Port of New York/New Jersey",
        "Savannah": "Port of Savannah",
        "Houston": "Port of Houston",
        "Oakland": "Port of Oakland",
        "Seattle": "Port of Seattle",
        "Tacoma": "Port of Tacoma",
        "Virginia": "Port of Virginia",
        "Charleston": "Port of Charleston",
        "Miami": "Port of Miami",
        "New Orleans": "Port of New Orleans",
    }
    
    graph_name_to_node = {_normalize_name(name_to_node.get(g, g)): n 
                         for g, n in name_to_node.items() if n}
    
    port_name_to_node = {}
    for data_name, graph_name in PORT_DATA_TO_GRAPH.items():
        graph_key = _normalize_name(graph_name)
        if graph_key in name_to_node:
            port_name_to_node[data_name] = name_to_node[graph_key]
    
    # Initialize all nodes with graph features + zeros for port features
    node_features = {}
    for node in node_list:
        port_feats = np.zeros(port_feat_dim, dtype=np.float32)
        node_features[node] = np.concatenate([graph_features[node], port_feats])
    
    # Update port nodes with actual port features
    matched_ports = set()
    for portname in latest_df.index:
        if portname in port_name_to_node:
            node = port_name_to_node[portname]
            if node in node_features:
                # Get port features (pad/truncate to match expected dim)
                if len(selected_feat_cols) > 0:
                    port_feats = latest_df.loc[portname, selected_feat_cols].values.astype(np.float32)
                else:
                    port_feats = np.zeros(port_feat_dim, dtype=np.float32)
                
                # Pad if needed
                if len(port_feats) < port_feat_dim:
                    port_feats = np.pad(port_feats, (0, port_feat_dim - len(port_feats)))
                elif len(port_feats) > port_feat_dim:
                    port_feats = port_feats[:port_feat_dim]
                
                node_features[node] = np.concatenate([graph_features[node], port_feats])
                matched_ports.add(portname)
    
    print(f"  ✓ Matched {len(matched_ports)} ports to graph nodes")
    
    # Ensure finite and correct dimension
    for nid in list(node_features.keys()):
        feat = node_features[nid]
        feat = np.nan_to_num(feat, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        
        # Ensure exact dimension match
        if len(feat) < target_in_channels:
            feat = np.pad(feat, (0, target_in_channels - len(feat)))
        elif len(feat) > target_in_channels:
            feat = feat[:target_in_channels]
        
        node_features[nid] = feat
    
    print(f"  ✓ Final features: {len(node_features):,} nodes × {target_in_channels} features")
    
    return node_features, port_feat_dim, selected_feat_cols


def generate_route_options(
    port_predictions: Dict[str, float],
    route_adapter: Optional[RouteAdapter] = None
) -> Optional[Dict[str, Any]]:
    """
    Generate route optimization options based on surge predictions.
    
    For now, returns a simplified route options structure.
    In production, this would use RouteAdapter to compute actual routes.
    """
    # Simplified route options - in production, integrate with RouteAdapter
    # This matches the structure expected by the dashboard
    
    # Example: Long Beach to Fleet Yards route
    origin_surge = port_predictions.get("Port of Long Beach", 0.5)
    if origin_surge is None:
        origin_surge = 0.5
    
    congestion_factor = 1.0 + (float(origin_surge) * 0.8)
    
    # Road-only route
    road_route = {
        "id": "route-road-001",
        "origin": {
            "type": "port",
            "id": "port-long-beach",
            "coordinates": [-118.216458, 33.754185],
            "name": "Port of Long Beach"
        },
        "destination": {
            "type": "intermodal_terminal",
            "id": "facility-fleet-yards",
            "coordinates": [-118.2200, 33.8190],
            "name": "Fleet Yards Inc."
        },
        "segments": [{
            "id": "seg-road-direct",
            "segmentType": "road",
            "startPoint": {"type": "port", "id": "port-long-beach", "coordinates": [-118.216458, 33.754185]},
            "endPoint": {"type": "intermodal_terminal", "id": "facility-fleet-yards", "coordinates": [-118.2200, 33.8190]},
            "coordinates": [[-118.216458, 33.754185], [-118.2200, 33.8190]],
            "distance": 7.2,
            "estimatedTime": 0.15 * congestion_factor,
            "cost": 27.50 * congestion_factor,
            "mode": "road",
            "metadata": {
                "roadType": "highway",
                "trafficConditions": "high" if origin_surge > 0.7 else "medium"
            }
        }],
        "totalDistance": 7.2,
        "totalTime": 0.15 * congestion_factor,
        "totalCost": 27.50 * congestion_factor,
        "transitions": [],
        "optimizationScore": max(0, 100 - (0.15 * congestion_factor * 100) - (27.50 * congestion_factor * 0.5)),
        "metadata": {
            "routeType": "road_only",
            "transitionCount": 0,
            "surgeLevel": float(origin_surge)
        }
    }
    
    # Intermodal route (optimized)
    intermodal_route = {
        "id": "route-intermodal-opt",
        "origin": road_route["origin"],
        "destination": road_route["destination"],
        "segments": [
            {
                "id": "seg-dray-001",
                "segmentType": "road",
                "distance": 1.0,
                "estimatedTime": 0.05,
                "mode": "road",
                "startPoint": road_route["origin"],
                "endPoint": {"type": "rail_node", "id": "node-ictf", "coordinates": [-118.216458, 33.754185]},
                "coordinates": []
            },
            {
                "id": "seg-rail-001",
                "segmentType": "rail_line",
                "distance": 6.2,
                "estimatedTime": 0.10,
                "mode": "rail",
                "cost": 15.00,
                "startPoint": {"type": "rail_node", "id": "node-ictf", "coordinates": [-118.216458, 33.754185]},
                "endPoint": road_route["destination"],
                "coordinates": []
            }
        ],
        "totalDistance": 7.2,
        "totalTime": 0.15,
        "totalCost": 15.00,
        "transitions": [{
            "point": road_route["origin"],
            "transitionType": "road_to_rail",
            "facility": {
                "name": "ICTF Automated Gate",
                "type": "intermodal_terminal",
                "waitTime": 5
            }
        }],
        "optimizationScore": 85.0,  # Higher score for intermodal
        "metadata": {
            "routeType": "mixed",
            "transitionCount": 1,
            "notes": "Recommended: Bypasses gate congestion via rail shuttle"
        }
    }
    
    routes = [intermodal_route, road_route]
    routes.sort(key=lambda x: x["optimizationScore"], reverse=True)
    
    return {
        "requestId": "req-production-inference",
        "routes": routes,
        "summary": {
            "totalOptions": len(routes),
            "fastestRoute": min(routes, key=lambda x: x["totalTime"])["id"],
            "cheapestRoute": min(routes, key=lambda x: x["totalCost"])["id"],
            "mostEfficientRoute": routes[0]["id"]
        },
        "metadata": {
            "generatedAt": datetime.utcnow().isoformat() + "Z",
            "dataSources": ["GNN-Surge-Model", "NTAD-Rail"],
            "optimizationEngine": "Glid-Surge-Optimizer-v1"
        }
    }


def generate_dispatch_windows(
    port_predictions: Dict[str, float],
    horizon: int = 24
) -> List[Dict[str, Any]]:
    """
    Generate optimal dispatch windows based on surge predictions.
    
    Uses surge level to determine dispatch priority:
    - Low surge (< 0.3): HIGH priority - good time to dispatch
    - Medium surge (0.3-0.7): MEDIUM priority - acceptable time
    - High surge (> 0.7): DELAY - wait for congestion to clear
    """
    from datetime import timedelta
    
    windows = []
    now = datetime.now()
    
    for port_name, surge_level in port_predictions.items():
        if surge_level is None:
            surge_level = 0.5  # Default
        
        surge_level = float(surge_level)
        
        # Determine priority and recommendation based on surge
        if surge_level < 0.3:
            priority = "HIGH"
            route_rec = "intermodal"
            dwell_reduction = 4.0
            notes = "Low congestion - optimal dispatch window"
        elif surge_level < 0.7:
            priority = "MEDIUM"
            route_rec = "intermodal"
            dwell_reduction = 2.0
            notes = "Moderate congestion - acceptable dispatch window"
        else:
            priority = "DELAY"
            route_rec = "hold"
            dwell_reduction = 0.0
            notes = f"High congestion ({surge_level:.2f}) - delay dispatch if possible"
        
        # Create window starting now, ending in horizon hours
        window_start = now
        window_end = now + timedelta(hours=horizon)
        
        windows.append({
            "port": port_name,
            "startTime": window_start.isoformat(),
            "endTime": window_end.isoformat(),
            "priority": priority,
            "surgeLevel": surge_level,
            "expectedDwellReduction": dwell_reduction,
            "routeRecommendation": route_rec,
            "notes": notes
        })
    
    # Sort by priority (HIGH first)
    priority_order = {"HIGH": 0, "MEDIUM": 1, "DELAY": 2}
    windows.sort(key=lambda w: priority_order.get(w["priority"], 1))
    
    return windows


def run_production_inference(
    model_path: Optional[Path] = None,
    export_path: Optional[Path] = None,
    horizons: List[int] = None
) -> Dict[str, Any]:
    """
    Run production inference pipeline.
    
    Args:
        model_path: Path to trained GNN model (default: auto-detect)
        export_path: Path to export dashboard payload (default: output/dashboard_payload.json)
        horizons: Prediction horizons in hours (default: [24, 48, 72])
    
    Returns:
        Dashboard payload dictionary
    """
    if horizons is None:
        horizons = [24, 48, 72]
    
    # Dependencies should already be checked at import time, but double-check
    if not HAS_TORCH or not HAS_GNN_MODEL:
        raise RuntimeError(
            "PyTorch and GNN model components are required.\n"
            "Install with: pip install torch torch-geometric"
        )
    
    print("\n" + "="*80)
    print("  PRODUCTION GNN INFERENCE")
    print("="*80)
    
    # 1. Load models (production models are saved per-horizon)
    print("\n[STEP 1] Loading GNN models...")
    from config import OUTPUT_DIR
    # SurgeGNN is defined at top of file to match train_gnn.py architecture
    
    checkpoint_dir = OUTPUT_DIR / "checkpoints"
    
    # Production models from train_gnn.py are saved per horizon as raw SurgeGNN models
    # We need to load a model for each horizon
    horizon_models = {}
    horizon_configs = {}
    
    for horizon in horizons:
        # Look for production model for this horizon
        horizon_pattern = f"gnn_production_{horizon}h_*.pt"
        horizon_files = list(checkpoint_dir.glob(horizon_pattern))
        
        if horizon_files:
            # Get most recent for this horizon
            horizon_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            model_path = horizon_files[0]
            print(f"  Found {horizon}h model: {model_path.name}")
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Get config from checkpoint
            saved_config = checkpoint.get('config', {})
            config = GNNConfig(
                hidden_channels=saved_config.get('hidden_channels', 128),
                num_layers=saved_config.get('num_layers', 2),
                dropout=saved_config.get('dropout', 0.15)
            )
            horizon_configs[horizon] = config
            
            # We'll need to know feature dimensions - will be determined when we build features
            # For now, store the checkpoint to load later
            horizon_models[horizon] = {
                'checkpoint': checkpoint,
                'path': model_path
            }
            print(f"    ✓ Loaded {horizon}h checkpoint")
        else:
            # Fallback: try single model approach
            if not horizon_models:  # Only try fallback once
                fallback_path = MODELS_DIR / "checkpoints" / "gnn_best_checkpoint.pt"
                if fallback_path.exists():
                    model = SurgeGNNModel()
                    if model.load(fallback_path, load_checkpoint=True):
                        # Use same model for all horizons
                        for h in horizons:
                            horizon_models[h] = model
                        print(f"  Using fallback model from {fallback_path}")
                        break
    
    if not horizon_models:
        raise FileNotFoundError(
            f"No production models found in {checkpoint_dir}.\n"
            f"Expected files: gnn_production_24h_*.pt, gnn_production_48h_*.pt, gnn_production_72h_*.pt\n"
            f"Train models first with: python train_gnn.py --config production"
        )
    
    print(f"  ✓ Loaded {len(horizon_models)} checkpoint(s) for {len(horizons)} horizon(s)")
    
    # 2. Load data
    print("\n[STEP 2] Loading data...")
    ports = load_production_ports()
    port_df = load_port_activity(ports=ports, country="United States")
    print(f"  ✓ Port activity: {len(port_df):,} records")
    
    # 3. Build graph
    print("\n[STEP 3] Building inference graph...")
    graph, name_to_node = build_inference_graph(ports)
    
    # 4. Detect model input dimension from first checkpoint
    print("\n[STEP 4] Detecting model architecture...")
    first_horizon = horizons[0]
    if first_horizon in horizon_models:
        first_ckpt = horizon_models[first_horizon]['checkpoint']
        first_conv_key = 'convs.0.lin_l.weight'
        if first_conv_key in first_ckpt['model_state_dict']:
            target_in_channels = first_ckpt['model_state_dict'][first_conv_key].shape[1]
            print(f"  ✓ Detected model in_channels: {target_in_channels}")
        else:
            target_in_channels = 25  # Default from production training
            print(f"  ⚠ Could not detect in_channels, using default: {target_in_channels}")
    else:
        target_in_channels = 25
        print(f"  ⚠ No checkpoint found, using default in_channels: {target_in_channels}")
    
    # 5. Prepare features matching the model's expected dimension
    print("\n[STEP 5] Preparing inference features...")
    node_features, feat_dim, feat_cols = prepare_inference_features(
        port_df, graph, name_to_node, ports, target_in_channels=target_in_channels
    )
    
    # 6. Run predictions for each horizon
    print("\n[STEP 6] Running predictions...")
    results = {
        "snapshot_date": datetime.now().date().isoformat(),
        "horizons": {},
        "route_options": None,
        "dispatch_windows": []
    }
    
    for horizon in horizons:
        print(f"\n  Predicting {horizon}h horizon...")
        
        # Get model checkpoint for this horizon
        if horizon not in horizon_models:
            print(f"    ⚠ No model for {horizon}h, skipping...")
            continue
        
        model_data = horizon_models[horizon]
        
        # If it's already a loaded model (fallback case), use it directly
        if isinstance(model_data, SurgeGNNModel):
            model = model_data
            model.set_graph(graph)
            predictions = model.predict(node_features)
        else:
            # Reconstruct SurgeGNN model from checkpoint
            checkpoint = model_data['checkpoint']
            config = horizon_configs[horizon]
            
            # Detect in_channels from checkpoint (first conv layer weight shape)
            state_dict = checkpoint['model_state_dict']
            first_conv_key = 'convs.0.lin_l.weight'
            if first_conv_key in state_dict:
                saved_in_channels = state_dict[first_conv_key].shape[1]
                print(f"    Detected model in_channels: {saved_in_channels}")
            else:
                # Fallback - shouldn't happen
                sample_feat = next(iter(node_features.values()))
                saved_in_channels = len(sample_feat)
            
            # Reconstruct model with EXACT same architecture as training
            model = SurgeGNN(
                in_channels=saved_in_channels,  # Use saved dimension, not current features
                hidden_channels=config.hidden_channels,
                out_channels=1,
                num_layers=config.num_layers,
                dropout=config.dropout
            )
            
            # Load state dict
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Move to device
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            model.eval()
            
            # Prepare data for prediction (Data already imported at top)
            
            node_list = list(graph.nodes())
            node_to_idx = {n: i for i, n in enumerate(node_list)}
            
            # Build feature matrix
            X = np.zeros((len(node_list), saved_in_channels), dtype=np.float32)
            for node, features in node_features.items():
                if node in node_to_idx:
                    X[node_to_idx[node]] = features
            
            # Build edge index
            edges = list(graph.edges())
            edge_index = np.array([
                [node_to_idx[u] for u, v in edges] + [node_to_idx[v] for u, v in edges],
                [node_to_idx[v] for u, v in edges] + [node_to_idx[u] for u, v in edges]
            ], dtype=np.int64)
            
            # Convert to tensors
            data = Data(
                x=torch.FloatTensor(X).to(device),
                edge_index=torch.LongTensor(edge_index).to(device)
            )
            
            # Run prediction
            with torch.no_grad():
                out = model(data.x, data.edge_index)
                pred_dict = {}
                for i, node in enumerate(node_list):
                    pred_dict[node] = out[i].item()
                predictions = pred_dict
        
        # Aggregate port predictions
        port_preds = {}
        for ap in ports:
            # Map activity port to component port names
            component_names = _activity_port_to_component_port_names(ap)
            
            # Get node IDs for this port
            node_ids = []
            for comp in component_names:
                nid = name_to_node.get(_normalize_name(comp))
                if nid:
                    node_ids.append(nid)
            
            # Aggregate predictions
            pred_values = [predictions.get(nid, 0.5) for nid in node_ids if nid in predictions]
            
            if pred_values:
                avg_pred = float(np.mean(pred_values))
            else:
                avg_pred = 0.5  # Default
            
            # Assign to component names
            for comp in component_names:
                port_preds[comp] = avg_pred
        
        results["horizons"][str(horizon)] = {
            "metrics": {
                "mse": None,  # Would need validation data
                "rmse": None,
                "mae": None
            },
            "port_predictions": port_preds,
            "inference_date": datetime.now().date().isoformat()
        }
        
        print(f"    ✓ Predictions for {len(port_preds)} ports")
    
    # 7. Generate route options (using 24h predictions)
    print("\n[STEP 7] Generating route options...")
    port_preds_24h = results["horizons"]["24"]["port_predictions"]
    route_options = generate_route_options(port_preds_24h)
    results["route_options"] = route_options
    print("  ✓ Route options generated")
    
    # 8. Generate dispatch windows
    print("\n[STEP 8] Generating dispatch windows...")
    dispatch_windows = generate_dispatch_windows(port_preds_24h, horizon=24)
    results["dispatch_windows"] = dispatch_windows
    print(f"  ✓ {len(dispatch_windows)} dispatch windows generated")
    
    # 8. Export dashboard payload
    if export_path is None:
        export_path = OUTPUT_DIR / "dashboard_payload.json"
    
    export_path.parent.mkdir(parents=True, exist_ok=True)
    export_path.write_text(json.dumps(_json_sanitize(results), indent=2))
    print(f"\n  ✓ Dashboard payload exported to {export_path}")
    
    print("\n" + "="*80)
    print("  ✅ INFERENCE COMPLETE")
    print("="*80)
    print(f"\n  Predictions: {len(horizons)} horizons")
    print(f"  Ports: {len(port_preds_24h)}")
    print(f"  Routes: {len(route_options['routes']) if route_options else 0}")
    print(f"  Dispatch windows: {len(dispatch_windows)}")
    print("="*80 + "\n")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run production GNN inference for dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to trained GNN model (default: auto-detect)"
    )
    
    parser.add_argument(
        "--export",
        type=Path,
        default=None,
        help="Export path for dashboard payload (default: output/dashboard_payload.json)"
    )
    
    parser.add_argument(
        "--horizons",
        type=str,
        default="24,48,72",
        help="Comma-separated prediction horizons in hours (default: 24,48,72)"
    )
    
    args = parser.parse_args()
    
    horizons = [int(h.strip()) for h in args.horizons.split(",") if h.strip()]
    
    try:
        results = run_production_inference(
            model_path=args.model_path,
            export_path=args.export,
            horizons=horizons
        )
        print("\n✅ Success!")
        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
