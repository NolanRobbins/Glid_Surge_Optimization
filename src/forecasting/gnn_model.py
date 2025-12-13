"""
Graph Neural Network for Surge Prediction
==========================================
Uses GNN to model congestion propagation through the transportation network.

NVIDIA Ecosystem Integration:
- PyTorch with CUDA for GPU-accelerated training
- cuGraph for fast graph analytics (when available)
- cuML for GPU-accelerated preprocessing (when available)

Optimized for ASUS Ascent GX10:
- NVIDIA GB10 Grace Blackwell Superchip
- 1 petaFLOP AI performance
- 128GB unified memory
- TF32 and mixed precision training

Why GNN > XGBoost for this problem:
1. Congestion PROPAGATES through the network (message passing)
2. Spatial dependencies between nodes matter
3. Network topology directly informs predictions
"""

import numpy as np
import pandas as pd
import networkx as nx
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import warnings
from tqdm import tqdm

# Check for NVIDIA/GPU libraries
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.optim import Adam, AdamW
    from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
    import torch.backends.cudnn as cudnn
    
    HAS_TORCH = True
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Enable GPU optimizations for ASUS Ascent GX10
    if torch.cuda.is_available():
        # Enable cuDNN autotuner (finds fastest algorithms)
        cudnn.benchmark = True
        cudnn.enabled = True
        
        # Enable TF32 for Blackwell/Ampere+ GPUs (2-3x faster)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Check for mixed precision support
        HAS_AMP = hasattr(torch.cuda, 'amp') and torch.cuda.is_available()
        
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        compute_cap = torch.cuda.get_device_capability(0)
        
        print(f"[GNN] PyTorch device: {DEVICE}")
        print(f"[GNN] GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        print(f"[GNN] Compute Capability: {compute_cap[0]}.{compute_cap[1]}")
        print(f"[GNN] TF32 enabled: ✓")
        print(f"[GNN] Mixed Precision (AMP): {'✓' if HAS_AMP else '✗'}")
    else:
        HAS_AMP = False
        print(f"[GNN] PyTorch device: {DEVICE}")
        
except ImportError:
    HAS_TORCH = False
    HAS_AMP = False
    DEVICE = None
    print("[GNN] PyTorch not available")

# Try PyTorch Geometric (preferred for GNNs)
try:
    import torch_geometric
    from torch_geometric.nn import GCNConv, GATConv, SAGEConv, BatchNorm
    from torch_geometric.data import Data, DataLoader
    from torch_geometric.utils import from_networkx
    HAS_PYG = True
    print(f"[GNN] PyTorch Geometric version: {torch_geometric.__version__}")
except ImportError:
    HAS_PYG = False
    print("[GNN] PyTorch Geometric not available - will use fallback")

# Try NVIDIA RAPIDS cuGraph for GPU-accelerated graph analytics
try:
    import cugraph
    import cudf
    HAS_CUGRAPH = True
    print("[GNN] NVIDIA cuGraph available - GPU graph analytics enabled")
except ImportError:
    HAS_CUGRAPH = False

# Try NVIDIA RAPIDS cuML for GPU-accelerated ML
try:
    import cuml
    from cuml.preprocessing import StandardScaler as cuStandardScaler
    HAS_CUML = True
    print("[GNN] NVIDIA cuML available - GPU ML preprocessing enabled")
except ImportError:
    HAS_CUML = False

import sys
import json
from datetime import datetime
sys.path.append(str(Path(__file__).parent.parent))
from config import MODELS_DIR


def _normalize_name(s: str) -> str:
    return " ".join(str(s).strip().lower().split())


def _build_name_to_graph_node_id(G: nx.Graph) -> Dict[str, str]:
    """
    Map human-readable location names (node attribute 'name') to graph node IDs.
    Only includes nodes that have a 'name' attribute.
    """
    out: Dict[str, str] = {}
    for node_id, data in G.nodes(data=True):
        name = data.get("name")
        if not name:
            continue
        out.setdefault(_normalize_name(name), node_id)
    return out


def audit_graph_data_alignment(
    G: nx.Graph,
    port_activity_df: pd.DataFrame,
    ports: Optional[List[str]] = None,
    horizon_hours: int = 24,
) -> Dict[str, Any]:
    """
    Validate that competition data ports can be mapped to graph nodes and produce labels.
    Returns a dict with high-signal counts and coverage stats.
    """
    df = port_activity_df.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])

    if ports:
        ports_norm = {_normalize_name(p) for p in ports}
        df = df[df["portname"].map(_normalize_name).isin(ports_norm)].copy()

    name_to_node = _build_name_to_graph_node_id(G)

    def _map_portname_to_node_ids(portname: str) -> List[str]:
        out = []
        for comp in _activity_port_to_component_port_names(portname):
            nid = name_to_node.get(_normalize_name(comp))
            if nid:
                out.append(nid)
        return out

    df["graph_node_ids"] = df["portname"].map(_map_portname_to_node_ids)
    mapped = df["graph_node_ids"].map(bool)

    horizon_days = max(1, horizon_hours // 24)
    df = df.sort_values(["portname", "date"])
    if "portcalls" in df.columns:
        df[f"target_{horizon_hours}h"] = df.groupby("portname")["portcalls"].shift(-horizon_days)
    else:
        df[f"target_{horizon_hours}h"] = np.nan

    has_label = df[f"target_{horizon_hours}h"].notna()

    return {
        "graph_nodes": int(G.number_of_nodes()),
        "graph_edges": int(G.number_of_edges()),
        "ports_in_df": int(df["portname"].nunique()),
        "rows": int(len(df)),
        "mapped_rows": int(mapped.sum()),
        "mapped_ports": int(df.loc[mapped, "portname"].nunique()),
        "labeled_rows": int((mapped & has_label).sum()),
        "label_horizon_hours": int(horizon_hours),
        "label_horizon_days": int(horizon_days),
        "example_mappings": (
            df.loc[mapped, ["portname", "graph_node_ids"]]
            .assign(graph_node_ids=lambda x: x["graph_node_ids"].map(tuple))
            .drop_duplicates()
            .head(10)
            .to_dict(orient="records")
        ),
        "unmapped_ports": sorted(df.loc[~mapped, "portname"].dropna().unique().tolist())[:25],
    }


def _default_activity_ports() -> List[str]:
    # Names as they appear in the Global Daily Port Activity dataset
    return [
        "Los Angeles-Long Beach",
        "New York-New Jersey",
        "Savannah",
        "Houston",
        "Oakland",
    ]


def _activity_port_to_component_port_names(activity_port: str) -> List[str]:
    """
    Map a port activity 'portname' (dataset naming) to one or more configured port names.
    Some datasets aggregate port complexes (e.g. Los Angeles-Long Beach).
    """
    ap = _normalize_name(activity_port)
    if ap in {"los angeles-long beach", "los angeles long beach"}:
        return ["Port of Los Angeles", "Port of Long Beach"]
    if ap in {"new york-new jersey", "new york new jersey"}:
        return ["Port of New York/New Jersey"]
    # For most ports, the config uses "Port of X"
    return [f"Port of {activity_port}"]


@dataclass
class GNNConfig:
    """
    Configuration for GNN model - Research-backed hyperparameters.
    
    Optimized for:
    - SPARSE transportation graphs (~250K nodes, ~300K edges, avg degree ~2.4)
    - Surge/congestion prediction (regression task)
    - Message passing for congestion propagation modeling
    
    Research references:
    - "Inductive Representation Learning on Large Graphs" (Hamilton et al., 2017) - GraphSAGE
    - "Traffic Flow Prediction via Spatial Temporal Graph Neural Network" (2020)
    - "Graph Neural Networks for Transportation Network Analysis" (2023)
    
    Key insights for transportation graphs:
    1. SPARSE graphs benefit from GraphSAGE over GCN (better neighbor sampling)
    2. 2-3 layers optimal - prevents over-smoothing in sparse graphs
    3. Higher hidden_channels (128-256) for large graphs with many features
    4. Lower dropout (0.1-0.3) for regression tasks vs classification
    5. Lower learning rate (0.0005-0.001) for stability with large graphs
    """
    # Architecture
    hidden_channels: int = 128      # Increased for 250K node graph (was 64)
    num_layers: int = 2             # Reduced to prevent over-smoothing (was 3)
    dropout: float = 0.15           # Lower for regression (was 0.2)
    conv_type: str = 'sage'         # GraphSAGE - best for inductive learning on large sparse graphs
    
    # Training
    learning_rate: float = 0.0005   # Lower for stability with large graph (was 0.001)
    weight_decay: float = 1e-4      # L2 regularization (NEW - prevents overfitting)
    epochs: int = 200               # More epochs with early stopping (was 100)
    patience: int = 15              # More patience for convergence (was 10)
    
    # Batch settings (for mini-batch training on large graphs)
    batch_size: int = 1024          # For neighbor sampling
    num_neighbors: list = None      # Neighbor sampling per layer [25, 10]
    
    # Learning rate scheduling
    lr_scheduler: str = 'plateau'   # 'plateau', 'cosine', 'step'
    lr_patience: int = 5            # Patience for LR reduction
    lr_factor: float = 0.5          # LR reduction factor
    
    # GPU Optimization (for ASUS Ascent GX10 / Grace Blackwell)
    use_amp: bool = True            # Mixed precision training (FP16/BF16)
    gradient_clip: float = 1.0      # Gradient clipping for stability
    compile_model: bool = False     # torch.compile (PyTorch 2.0+)
    
    def __post_init__(self):
        if self.num_neighbors is None:
            # Default: sample 25 neighbors at layer 1, 10 at layer 2
            # This is optimal for sparse graphs per GraphSAGE paper
            self.num_neighbors = [25, 10] if self.num_layers >= 2 else [25]


# Alternative configs for different scenarios
def get_config_for_scenario(scenario: str = 'default') -> GNNConfig:
    """
    Get research-backed configuration for different scenarios.
    
    Scenarios:
    - 'default': Balanced config for general surge prediction
    - 'fast': Quick training for prototyping
    - 'accurate': Maximum accuracy, longer training
    - 'large_graph': Optimized for graphs > 200K nodes
    - 'small_graph': Optimized for graphs < 10K nodes
    """
    configs = {
        'default': GNNConfig(),
        
        'fast': GNNConfig(
            hidden_channels=64,
            num_layers=2,
            epochs=50,
            patience=5,
            batch_size=2048,
        ),
        
        'accurate': GNNConfig(
            hidden_channels=256,
            num_layers=3,
            dropout=0.2,
            learning_rate=0.0003,
            weight_decay=5e-5,
            epochs=500,
            patience=30,
            batch_size=512,
        ),
        
        'large_graph': GNNConfig(
            # For graphs > 200K nodes like our rail network
            hidden_channels=128,
            num_layers=2,  # Fewer layers = less over-smoothing
            dropout=0.1,
            learning_rate=0.0005,
            weight_decay=1e-4,
            epochs=200,
            patience=20,
            batch_size=1024,
            num_neighbors=[15, 10],  # Smaller samples for memory
        ),
        
        'small_graph': GNNConfig(
            hidden_channels=64,
            num_layers=3,
            dropout=0.3,
            learning_rate=0.001,
            epochs=100,
            patience=10,
        ),
    }
    
    return configs.get(scenario, configs['default'])


class SurgeGNN(nn.Module):
    """
    Graph Neural Network for predicting port/terminal surge levels.
    
    Architecture:
    - Input: Node features (historical activity, weather, time features)
    - Message Passing: Learn how congestion propagates through network
    - Output: Predicted surge level for each node
    
    The key insight: A node's future congestion depends on:
    1. Its own historical patterns
    2. Congestion at neighboring nodes (propagation)
    3. Network topology (chokepoints vs. well-connected nodes)
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 64,
        out_channels: int = 1,
        num_layers: int = 3,
        dropout: float = 0.2,
        conv_type: str = 'sage'
    ):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Choose convolution type
        if conv_type == 'gcn':
            Conv = GCNConv
        elif conv_type == 'gat':
            Conv = lambda i, o: GATConv(i, o, heads=4, concat=False)
        else:  # sage (default) - best for inductive learning
            Conv = SAGEConv
        
        # Build layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        # Input layer
        self.convs.append(Conv(in_channels, hidden_channels))
        self.norms.append(BatchNorm(hidden_channels))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.convs.append(Conv(hidden_channels, hidden_channels))
            self.norms.append(BatchNorm(hidden_channels))
        
        # Output layer
        self.convs.append(Conv(hidden_channels, hidden_channels))
        self.norms.append(BatchNorm(hidden_channels))
        
        # Final prediction head
        self.predictor = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels)
        )
    
    def forward(self, x, edge_index):
        """
        Forward pass with message passing.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Graph connectivity [2, num_edges]
            
        Returns:
            Node predictions [num_nodes, out_channels]
        """
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        return self.predictor(x)


class TransportGraphDataset:
    """
    Prepares transportation graph data for GNN training.
    
    Uses NVIDIA cuGraph when available for fast graph processing.
    """
    
    def __init__(self, G: nx.Graph):
        self.G = G
        self.node_list = list(G.nodes())
        self.node_to_idx = {n: i for i, n in enumerate(self.node_list)}
        
    def prepare_pyg_data(
        self,
        node_features: Dict[str, np.ndarray],
        targets: Dict[str, float] = None
    ) -> Data:
        """
        Convert NetworkX graph to PyTorch Geometric Data object.
        
        Args:
            node_features: Dict mapping node_id to feature array
            targets: Optional dict mapping node_id to target value
            
        Returns:
            PyG Data object ready for training
        """
        # Build feature matrix
        num_nodes = len(self.node_list)
        feature_dim = len(next(iter(node_features.values()))) if node_features else 10
        
        x = np.zeros((num_nodes, feature_dim))
        for node, features in node_features.items():
            if node in self.node_to_idx:
                x[self.node_to_idx[node]] = features
        
        # Build edge index
        edges = list(self.G.edges())
        edge_index = np.array([
            [self.node_to_idx[u] for u, v in edges] + [self.node_to_idx[v] for u, v in edges],
            [self.node_to_idx[v] for u, v in edges] + [self.node_to_idx[u] for u, v in edges]
        ])
        
        # Build targets
        if targets is not None:
            y = np.array([targets.get(n, 0.0) for n in self.node_list])
        else:
            y = np.zeros(num_nodes)
        
        # Create PyG Data object
        data = Data(
            x=torch.FloatTensor(x),
            edge_index=torch.LongTensor(edge_index),
            y=torch.FloatTensor(y).unsqueeze(1)
        )
        
        return data
    
    def compute_node_features_cugraph(self) -> Dict[str, np.ndarray]:
        """
        Use NVIDIA cuGraph for GPU-accelerated graph feature computation.
        """
        if not HAS_CUGRAPH:
            return self._compute_node_features_cpu()
        
        print("  Using NVIDIA cuGraph for graph analytics...")
        
        # Convert to cuGraph format
        edges = list(self.G.edges())
        source = [self.node_to_idx[u] for u, v in edges]
        destination = [self.node_to_idx[v] for u, v in edges]
        
        gdf = cudf.DataFrame({
            'source': source,
            'destination': destination
        })
        
        cu_G = cugraph.Graph()
        cu_G.from_cudf_edgelist(gdf, source='source', destination='destination')
        
        # Compute centrality metrics on GPU
        pagerank = cugraph.pagerank(cu_G)
        degree = cu_G.degrees()
        
        # Combine features
        features = {}
        pr_dict = dict(zip(pagerank['vertex'].to_pandas(), pagerank['pagerank'].to_pandas()))
        # cuGraph API changed across versions:
        # - Older: columns ['vertex', 'degree']
        # - Newer (RAPIDS 25.x): columns ['in_degree', 'out_degree', 'vertex']
        if 'degree' in degree.columns:
            deg_series = degree['degree']
        elif 'in_degree' in degree.columns and 'out_degree' in degree.columns:
            deg_series = degree['in_degree'] + degree['out_degree']
        elif 'out_degree' in degree.columns:
            deg_series = degree['out_degree']
        elif 'in_degree' in degree.columns:
            deg_series = degree['in_degree']
        else:
            raise KeyError(f"Unexpected cuGraph degrees() columns: {list(degree.columns)}")
        deg_dict = dict(zip(degree['vertex'].to_pandas(), deg_series.to_pandas()))
        
        for node in self.node_list:
            idx = self.node_to_idx[node]
            features[node] = np.array([
                deg_dict.get(idx, 0),
                pr_dict.get(idx, 0),
                0, 0, 0, 0, 0, 0, 0, 0  # Placeholder for other features
            ])
        
        return features
    
    def _compute_node_features_cpu(self) -> Dict[str, np.ndarray]:
        """Fallback CPU computation of node features."""
        print("  Computing node features on CPU...")
        
        degree = dict(self.G.degree())
        
        features = {}
        for node in tqdm(self.node_list, desc="  Node features", unit="node"):
            # Get edge data
            edges = self.G.edges(node, data=True)
            rail_classes = []
            for _, _, data in edges:
                if 'rail_class' in data:
                    rail_classes.append(data['rail_class'])
            
            features[node] = np.array([
                degree.get(node, 0),                          # Degree
                0.0,                                          # Pagerank (placeholder)
                np.mean(rail_classes) if rail_classes else 0, # Avg rail class
                len([r for r in rail_classes if r == 1]),     # Class 1 connections
                0, 0, 0, 0, 0, 0                              # Placeholder
            ])
        
        return features


class SurgeGNNModel:
    """
    High-level interface for GNN-based surge prediction.
    
    Integrates with NVIDIA ecosystem:
    - PyTorch CUDA for GPU training
    - cuGraph for graph analytics
    - cuML for preprocessing
    """
    
    def __init__(
        self,
        config: GNNConfig = None,
        graph: nx.Graph = None
    ):
        self.config = config or GNNConfig()
        self.graph = graph
        self.model = None
        self.dataset = None
        self.is_fitted = False
        
        # Report GPU status
        if HAS_TORCH and torch.cuda.is_available():
            print(f"[GNN] GPU: {torch.cuda.get_device_name(0)}")
            print(f"[GNN] CUDA version: {torch.version.cuda}")
        
    def set_graph(self, G: nx.Graph):
        """Set the transportation graph."""
        self.graph = G
        self.dataset = TransportGraphDataset(G)
        
    def fit(
        self,
        train_data: pd.DataFrame,
        target_col: str = 'surge_level',
        node_col: str = 'node_id',
        node_features: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, float]:
        """
        Train the GNN model.
        
        Args:
            train_data: Training data with node features and targets
            target_col: Column containing target values
            node_col: Column containing node identifiers
            
        Returns:
            Training metrics
        """
        if not HAS_TORCH or not HAS_PYG:
            print("[GNN] PyTorch Geometric not available, using fallback XGBoost")
            return self._fit_fallback(train_data, target_col, node_col)
        
        if self.graph is None:
            raise ValueError("Graph not set. Call set_graph() first.")
        
        print(f"\n[GNN] Training on {DEVICE}...")
        print(f"  Config: {self.config}")
        
        # Prepare node features:
        # - If caller provides node_features, we use them (recommended for real competition data)
        # - Otherwise, we fall back to graph-only features.
        if node_features is None:
            if HAS_CUGRAPH:
                node_features = self.dataset.compute_node_features_cugraph()
            else:
                node_features = self.dataset._compute_node_features_cpu()
        
        # Prepare targets
        targets = dict(zip(train_data[node_col], train_data[target_col]))
        
        # Create PyG data
        data = self.dataset.prepare_pyg_data(node_features, targets)
        data = data.to(DEVICE)
        
        # Initialize model
        in_channels = data.x.shape[1]
        self.model = SurgeGNN(
            in_channels=in_channels,
            hidden_channels=self.config.hidden_channels,
            out_channels=1,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
            conv_type=self.config.conv_type
        ).to(DEVICE)
        
        # Training setup with weight decay (L2 regularization)
        optimizer = Adam(
            self.model.parameters(), 
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay  # L2 regularization
        )
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            patience=self.config.lr_patience,
            factor=self.config.lr_factor,
            min_lr=1e-6
        )
        criterion = nn.MSELoss()
        
        # Log training configuration
        print(f"\n  ┌─────────────────────────────────────────────────────────────────┐")
        print(f"  │ Model parameters: {sum(p.numel() for p in self.model.parameters()):>10,}                             │")
        print(f"  │ Learning rate:    {self.config.learning_rate:>10}                             │")
        print(f"  │ Weight decay:     {self.config.weight_decay:>10}                             │")
        print(f"  │ Hidden channels:  {self.config.hidden_channels:>10}                             │")
        print(f"  │ Num layers:       {self.config.num_layers:>10}                             │")
        print(f"  │ Conv type:        {self.config.conv_type:>10}                             │")
        print(f"  │ Dropout:          {self.config.dropout:>10}                             │")
        print(f"  └─────────────────────────────────────────────────────────────────┘")
        
        # Setup mixed precision training (AMP) for GPU optimization
        use_amp = self.config.use_amp and HAS_AMP
        if use_amp:
            scaler = torch.cuda.amp.GradScaler()
            print(f"  ✓ Mixed Precision Training (AMP) enabled")
        else:
            scaler = None
            print(f"  ✗ Mixed Precision Training disabled")
        
        # Train/val split (only on labeled nodes)
        num_nodes = data.x.shape[0]
        labeled_mask = torch.tensor(
            [n in targets for n in self.dataset.node_list],
            dtype=torch.bool,
            device=DEVICE
        )
        labeled_idx = labeled_mask.nonzero(as_tuple=False).view(-1)
        if labeled_idx.numel() < 3:
            raise ValueError(
                f"Not enough labeled nodes to train ({int(labeled_idx.numel())}). "
                f"Check node_id mapping between graph and labels."
            )
        perm = labeled_idx[torch.randperm(labeled_idx.numel(), device=DEVICE)]
        split = int(0.8 * perm.numel())
        train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=DEVICE)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=DEVICE)
        train_mask[perm[:split]] = True
        val_mask[perm[split:]] = True
        
        # Training loop with checkpointing
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        best_epoch = 0
        train_start = time.time()
        
        # Create checkpoint directory
        checkpoint_dir = MODELS_DIR / 'checkpoints'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / 'gnn_best_checkpoint.pt'
        
        print(f"  Checkpoint: {checkpoint_path}")
        print()
        
        # Enhanced progress bar with live metrics
        pbar = tqdm(
            range(self.config.epochs), 
            desc="  Training", 
            unit="epoch",
            bar_format='{l_bar}{bar:30}{r_bar}{bar:-10b}'
        )
        
        for epoch in pbar:
            epoch_start = time.time()
            
            # ===================== TRAINING =====================
            self.model.train()
            optimizer.zero_grad()
            
            if use_amp:
                # Mixed precision forward pass
                with torch.cuda.amp.autocast():
                    out = self.model(data.x, data.edge_index)
                    train_loss = criterion(out[train_mask], data.y[train_mask])
                
                # Scaled backward pass
                scaler.scale(train_loss).backward()
                
                # Gradient clipping for stability
                if self.config.gradient_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard precision
                out = self.model(data.x, data.edge_index)
                train_loss = criterion(out[train_mask], data.y[train_mask])
                train_loss.backward()
                
                # Gradient clipping for stability
                if self.config.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
                
                optimizer.step()
            
            # ===================== VALIDATION =====================
            self.model.eval()
            with torch.no_grad():
                if use_amp:
                    with torch.cuda.amp.autocast():
                        out = self.model(data.x, data.edge_index)
                        val_loss = criterion(out[val_mask], data.y[val_mask])
                else:
                    out = self.model(data.x, data.edge_index)
                    val_loss = criterion(out[val_mask], data.y[val_mask])
            
            scheduler.step(val_loss)
            
            # Update progress bar with live metrics
            epoch_time = time.time() - epoch_start
            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f'{train_loss.item():.4f}',
                'val': f'{val_loss.item():.4f}',
                'lr': f'{current_lr:.1e}',
                'best': f'{best_val_loss:.4f}'
            })
            
            # Checkpointing - save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_epoch = epoch
                # Save best model state (in memory for speed)
                best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                # Also save to disk periodically (every improvement)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': val_loss.item(),
                    'train_loss': train_loss.item(),
                    'config': self.config,
                    'node_list': self.dataset.node_list if self.dataset else None,
                    'in_channels': in_channels,
                    'timestamp': datetime.now().isoformat()
                }, checkpoint_path)
                tqdm.write(f"  ✓ Epoch {epoch}: New best val_loss={val_loss:.4f} - checkpoint saved")
            else:
                patience_counter += 1
                # Early stopping check
                if patience_counter >= self.config.patience:
                    tqdm.write(f"  Early stopping at epoch {epoch} (best was epoch {best_epoch})")
                    break
            
            if epoch % 20 == 0 and epoch > 0:
                tqdm.write(f"  Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, patience={patience_counter}/{self.config.patience}")
        
        # Restore best model state
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"  Restored best model from epoch {best_epoch}")
        
        self.is_fitted = True
        
        # Final metrics
        self.model.eval()
        with torch.no_grad():
            out = self.model(data.x, data.edge_index)
            mse = criterion(out[val_mask], data.y[val_mask]).item()
            mae = F.l1_loss(out[val_mask], data.y[val_mask]).item()
        
        metrics = {
            'mse': mse,
            'rmse': np.sqrt(mse),
            'mae': mae
        }
        
        print(f"\n[GNN] Training complete: MAE={mae:.4f}, RMSE={np.sqrt(mse):.4f}")
        return metrics
    
    def _fit_fallback(
        self,
        train_data: pd.DataFrame,
        target_col: str,
        node_col: str
    ) -> Dict[str, float]:
        """Fallback to XGBoost if PyG not available."""
        from .surge_model import SurgePredictionModel
        
        print("[GNN] Falling back to XGBoost...")
        fallback = SurgePredictionModel(model_type='xgboost', graph=self.graph)
        return fallback.fit(train_data, target_col=target_col)
    
    def predict(self, node_features: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Make predictions for nodes.
        
        Args:
            node_features: Dict mapping node_id to feature array
            
        Returns:
            Dict mapping node_id to predicted surge level
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if not HAS_TORCH or not HAS_PYG:
            return {}
        
        data = self.dataset.prepare_pyg_data(node_features)
        data = data.to(DEVICE)
        
        self.model.eval()
        with torch.no_grad():
            out = self.model(data.x, data.edge_index)
        
        predictions = {}
        for i, node in enumerate(self.dataset.node_list):
            predictions[node] = out[i].item()
        
        return predictions
    
    def save(self, path: Path = None):
        """Save model to disk."""
        if path is None:
            path = MODELS_DIR / 'gnn_model.pt'
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.model is not None:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'config': self.config,
                'node_list': self.dataset.node_list if self.dataset else None,
                'is_fitted': self.is_fitted,
                'timestamp': datetime.now().isoformat()
            }, path)
            print(f"[GNN] Model saved to {path}")
    
    def load(self, path: Path = None, load_checkpoint: bool = False):
        """
        Load model from disk.
        
        Args:
            path: Path to saved model. If None, uses default location.
            load_checkpoint: If True, tries to load from checkpoint first for resume.
        """
        if path is None:
            if load_checkpoint:
                checkpoint_path = MODELS_DIR / 'checkpoints' / 'gnn_best_checkpoint.pt'
                if checkpoint_path.exists():
                    path = checkpoint_path
                    print(f"[GNN] Loading from checkpoint: {path}")
                else:
                    path = MODELS_DIR / 'gnn_model.pt'
            else:
                path = MODELS_DIR / 'gnn_model.pt'
        
        if not Path(path).exists():
            print(f"[GNN] No saved model found at {path}")
            return False
        
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
        self.config = checkpoint.get('config', self.config)
        
        # Reconstruct model if we have the architecture info
        in_channels = checkpoint.get('in_channels', 10)
        if 'model_state_dict' in checkpoint:
            self.model = SurgeGNN(
                in_channels=in_channels,
                hidden_channels=self.config.hidden_channels,
                out_channels=1,
                num_layers=self.config.num_layers,
                dropout=self.config.dropout,
                conv_type=self.config.conv_type
            ).to(DEVICE)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.is_fitted = checkpoint.get('is_fitted', True)
            
        if 'node_list' in checkpoint and checkpoint['node_list'] is not None:
            if self.graph is not None:
                self.dataset = TransportGraphDataset(self.graph)
        
        timestamp = checkpoint.get('timestamp', 'unknown')
        epoch = checkpoint.get('epoch', 'N/A')
        val_loss = checkpoint.get('val_loss', 'N/A')
        
        print(f"[GNN] Model loaded from {path}")
        print(f"  Timestamp: {timestamp}")
        print(f"  Epoch: {epoch}, Val Loss: {val_loss}")
        
        return True
    
    def resume_training(
        self,
        train_data: pd.DataFrame,
        target_col: str = 'surge_level',
        node_col: str = 'node_id',
        additional_epochs: int = None
    ) -> Dict[str, float]:
        """
        Resume training from a checkpoint.
        
        Args:
            train_data: Training data
            target_col: Target column name
            node_col: Node ID column name
            additional_epochs: Extra epochs to train (uses config.epochs if None)
            
        Returns:
            Training metrics
        """
        checkpoint_path = MODELS_DIR / 'checkpoints' / 'gnn_best_checkpoint.pt'
        
        if not checkpoint_path.exists():
            print("[GNN] No checkpoint found, starting fresh training")
            return self.fit(train_data, target_col, node_col)
        
        print(f"[GNN] Resuming from checkpoint: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        start_epoch = checkpoint.get('epoch', 0) + 1
        self.config = checkpoint.get('config', self.config)
        
        if additional_epochs:
            self.config.epochs = start_epoch + additional_epochs
        
        print(f"  Resuming from epoch {start_epoch}")
        print(f"  Previous best val_loss: {checkpoint.get('val_loss', 'N/A')}")
        
        # Continue with normal training (which will handle checkpointing)
        return self.fit(train_data, target_col, node_col)


def check_nvidia_ecosystem() -> Dict[str, bool]:
    """Check which NVIDIA libraries are available."""
    status = {
        'pytorch': HAS_TORCH,
        'cuda': HAS_TORCH and torch.cuda.is_available(),
        'pytorch_geometric': HAS_PYG,
        'cugraph': HAS_CUGRAPH,
        'cuml': HAS_CUML
    }
    
    print("\n" + "="*50)
    print("NVIDIA ECOSYSTEM STATUS")
    print("="*50)
    for lib, available in status.items():
        symbol = "✓" if available else "✗"
        print(f"  {symbol} {lib}")
    
    if status['cuda']:
        print(f"\n  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA: {torch.version.cuda}")
    print("="*50 + "\n")
    
    return status


if __name__ == "__main__":
    import argparse
    from data.loaders import load_port_activity, load_rail_nodes, load_rail_lines
    from graph.builder import build_rail_graph, add_location_nodes, connect_locations_to_graph
    from graph.route_specs import load_route_spec
    from config import US_PORTS, RAIL_TERMINALS
    from forecasting.features import build_forecasting_features

    parser = argparse.ArgumentParser(description="Train SurgeGNN (PyTorch Geometric) using competition datasets.")
    parser.add_argument("--mode", choices=["smoke", "audit", "real"], default="smoke")
    parser.add_argument("--horizon-hours", type=int, default=24, help="Single horizon for audit/real (hours).")
    parser.add_argument("--horizons", type=str, default="24,48,72", help="Comma-separated horizons (hours) for real training/export.")
    parser.add_argument("--ports", type=str, default="", help="Comma-separated ports; accepts dataset names (e.g. 'Oakland') or 'Port of X'.")
    parser.add_argument("--max-ports", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--export-json", type=str, default="", help="If set, writes predictions JSON to this path.")
    parser.add_argument("--route-spec", type=str, default="", help="Path to a GeoJSON Feature route spec (e.g. long-beach-to-fleet-yards-route.json)")
    parser.add_argument("--radius-min-miles", type=float, default=40.0)
    parser.add_argument("--radius-max-miles", type=float, default=50.0)
    args = parser.parse_args()

    _ = check_nvidia_ecosystem()

    if args.mode == "smoke":
        print("Creating test graph...")
        G = nx.barabasi_albert_graph(100, 3)
        for u, v in G.edges():
            G[u][v]["rail_class"] = np.random.choice([1, 2, 3])
            G[u][v]["miles"] = np.random.uniform(5, 50)

        train_df = pd.DataFrame({
            "node_id": list(G.nodes()),
            "surge_level": np.random.uniform(0, 1, G.number_of_nodes())
        })

        config = GNNConfig(hidden_channels=32, num_layers=2, epochs=args.epochs)
        model = SurgeGNNModel(config=config, graph=G)
        model.set_graph(G)
        if HAS_TORCH and HAS_PYG:
            metrics = model.fit(train_df)
            print(f"\nFinal metrics: {metrics}")
        else:
            print("\nSkipping GNN training - PyTorch Geometric not installed")
        sys.exit(0)

    # REAL/AUDIT
    requested = [p.strip() for p in args.ports.split(",") if p.strip()]
    if not requested:
        activity_ports = _default_activity_ports()[: max(1, args.max_ports)]
    else:
        # Allow either dataset names ("Oakland") or human names ("Port of Oakland")
        activity_ports = []
        for p in requested:
            pn = _normalize_name(p)
            if pn.startswith("port of "):
                activity_ports.append(p[8:])  # drop "Port of "
            else:
                activity_ports.append(p)
        # special-case: if user asks for LA or Long Beach, use the combined dataset label
        req_norm = {_normalize_name(p) for p in requested}
        if "port of los angeles" in req_norm or "port of long beach" in req_norm:
            activity_ports = [ap for ap in activity_ports if _normalize_name(ap) not in {"los angeles", "long beach"}]
            activity_ports.append("Los Angeles-Long Beach")
        if "port of new york/new jersey" in req_norm:
            activity_ports = [ap for ap in activity_ports if _normalize_name(ap) not in {"new york/new jersey"}]
            activity_ports.append("New York-New Jersey")
        activity_ports = activity_ports[: max(1, args.max_ports)]

    print(f"Selected activity ports (dataset portname): {activity_ports}")
    port_df = load_port_activity(ports=activity_ports, country="United States")

    # Build rail graph + add selected ports as nodes connected to the rail network
    rail_nodes = load_rail_nodes(filter_us_only=True)
    rail_lines = load_rail_lines(filter_us_only=True)
    G = build_rail_graph(rail_nodes, rail_lines)
    # Build graph port nodes using configured port names corresponding to the selected activity ports
    component_port_names: List[str] = []
    for ap in activity_ports:
        component_port_names.extend(_activity_port_to_component_port_names(ap))
    component_port_names = list(dict.fromkeys(component_port_names))  # stable unique
    ports_dict = {p["name"]: p for p in US_PORTS if p["name"] in component_port_names}
    G = add_location_nodes(G, ports_dict, "port")
    G = connect_locations_to_graph(G, ports_dict, "port", max_connection_miles=20)

    # Add intermodal rail terminals (candidates for 40-50 mile optimization)
    terminals_dict = {t["name"]: t for t in RAIL_TERMINALS}
    G = add_location_nodes(G, terminals_dict, "terminal")
    G = connect_locations_to_graph(G, terminals_dict, "terminal", max_connection_miles=20)

    # Optional demo route spec: add destination facility as a node so it can be routed to/from
    route_info: Optional[Dict[str, Any]] = None
    if args.route_spec:
        rs = load_route_spec(args.route_spec)
        # Add destination as a node type "facility" using the route endpoint
        facility_dict = {
            rs.destination_name: {
                "name": rs.destination_name,
                "lat": rs.destination_latlon[0],
                "lon": rs.destination_latlon[1],
            }
        }
        G = add_location_nodes(G, facility_dict, "facility")
        G = connect_locations_to_graph(G, facility_dict, "facility", max_connection_miles=20)
        route_info = {
            "name": rs.name,
            "origin": rs.origin_name,
            "destination": rs.destination_name,
            "distance": rs.distance,
            "distance_unit": rs.distance_unit,
            "estimated_time": rs.estimated_time,
            "estimated_time_unit": rs.estimated_time_unit,
            "origin_latlon": rs.origin_latlon,
            "destination_latlon": rs.destination_latlon,
        }

    stats = audit_graph_data_alignment(G, port_df, ports=activity_ports, horizon_hours=args.horizon_hours)
    print("\n=== DATA ↔ GRAPH ALIGNMENT REPORT ===")
    for k, v in stats.items():
        if k in ("unmapped_ports", "example_mappings"):
            continue
        print(f"  {k}: {v}")
    print(f"  example_mappings: {stats['example_mappings']}")
    if stats["unmapped_ports"]:
        print(f"  unmapped_ports (first 25): {stats['unmapped_ports']}")

    if args.mode == "audit":
        sys.exit(0)

    # REAL: multi-horizon training on a single snapshot date (latest date with labels)
    horizons = [int(h.strip()) for h in args.horizons.split(",") if h.strip()]
    horizons = [h for h in horizons if h > 0]
    if not horizons:
        horizons = [args.horizon_hours]

    feat_df, feat_cols = build_forecasting_features(port_df, weather_df=None, target_col="portcalls", group_col="portname")
    feat_df = feat_df.sort_values(["portname", "date"])

    name_to_node = _build_name_to_graph_node_id(G)

    def _map_activity_port_to_node_ids(portname: str) -> List[str]:
        out = []
        for comp in _activity_port_to_component_port_names(portname):
            nid = name_to_node.get(_normalize_name(comp))
            if nid:
                out.append(nid)
        return out

    # Choose snapshot date based on the smallest horizon (max data availability)
    min_h = min(horizons)
    min_days = max(1, min_h // 24)
    tmp = feat_df.copy()
    tmp[f"target_{min_h}h"] = tmp.groupby("portname")["portcalls"].shift(-min_days)
    labeled_tmp = tmp.dropna(subset=[f"target_{min_h}h"]).copy()
    if labeled_tmp.empty:
        raise ValueError("No labeled rows after shifting. Check data availability.")
    snapshot_date = labeled_tmp["date"].max()
    print(f"\nTraining snapshot date: {snapshot_date.date()} (based on {min_h}h horizon)")

    # Build per-port feature vectors at snapshot date
    snap_feat = feat_df[feat_df["date"] == snapshot_date].copy()
    snap_feat = snap_feat.set_index("portname")

    # Graph-only base features for all nodes (degree, pagerank)
    base_graph_feat: Dict[str, np.ndarray] = {}
    try:
        # Use cuGraph if available to compute pagerank/degree quickly
        if HAS_CUGRAPH:
            edges = list(G.edges())
            node_list = list(G.nodes())
            node_to_idx = {n: i for i, n in enumerate(node_list)}
            gdf = cudf.DataFrame({
                "source": [node_to_idx[u] for u, v in edges],
                "destination": [node_to_idx[v] for u, v in edges],
            })
            cu_G = cugraph.Graph()
            cu_G.from_cudf_edgelist(gdf, source="source", destination="destination")
            pr = cugraph.pagerank(cu_G)
            deg = cu_G.degrees()
            pr_dict = dict(zip(pr["vertex"].to_pandas(), pr["pagerank"].to_pandas()))
            if "degree" in deg.columns:
                deg_series = deg["degree"]
            else:
                deg_series = deg.get("in_degree", 0) + deg.get("out_degree", 0)
            deg_dict = dict(zip(deg["vertex"].to_pandas(), deg_series.to_pandas()))
            for n in node_list:
                idx = node_to_idx[n]
                base_graph_feat[n] = np.array([deg_dict.get(idx, 0.0), pr_dict.get(idx, 0.0)], dtype=np.float32)
        else:
            # CPU fallback
            deg = dict(G.degree())
            pr = nx.pagerank(G, max_iter=50)
            for n in G.nodes():
                base_graph_feat[n] = np.array([float(deg.get(n, 0)), float(pr.get(n, 0.0))], dtype=np.float32)
    except Exception:
        # Very safe fallback: degree only
        deg = dict(G.degree())
        for n in G.nodes():
            base_graph_feat[n] = np.array([float(deg.get(n, 0)), 0.0], dtype=np.float32)

    # Combine graph features + port activity features for node_features
    port_feat_dim = len(feat_cols)
    def _port_feat_for(portname: str) -> np.ndarray:
        if portname not in snap_feat.index:
            return np.zeros(port_feat_dim, dtype=np.float32)
        row = snap_feat.loc[portname]
        return row[feat_cols].to_numpy(dtype=np.float32, copy=True)

    node_features: Dict[str, np.ndarray] = {}
    for n in G.nodes():
        node_features[n] = np.concatenate([base_graph_feat[n], np.zeros(port_feat_dim, dtype=np.float32)], axis=0)

    # Fill port nodes with real activity features
    for portname in snap_feat.index.tolist():
        node_ids = _map_activity_port_to_node_ids(portname)
        pf = _port_feat_for(portname)
        for nid in node_ids:
            if nid in node_features:
                node_features[nid] = np.concatenate([base_graph_feat[nid], pf], axis=0)

    results = {
        "snapshot_date": str(snapshot_date.date()),
        "horizons": {},
        "ports": activity_ports,
        "route_spec": route_info,
        "candidate_policy": {
            "radius_min_miles": args.radius_min_miles,
            "radius_max_miles": args.radius_max_miles,
        },
    }

    for h in horizons:
        h_days = max(1, h // 24)
        tmp = feat_df.copy()
        tmp[f"target_{h}h"] = tmp.groupby("portname")["portcalls"].shift(-h_days)
        snap = tmp[tmp["date"] == snapshot_date].copy()
        snap = snap.dropna(subset=[f"target_{h}h"])
        if snap.empty:
            print(f"[WARN] No labels for horizon {h}h at snapshot date; skipping")
            continue
        snap["node_ids"] = snap["portname"].map(_map_activity_port_to_node_ids)
        snap = snap.explode("node_ids").rename(columns={"node_ids": "node_id"})
        snap = snap.dropna(subset=["node_id"])
        train_df = snap[["node_id", f"target_{h}h"]].rename(columns={f"target_{h}h": "surge_level"})
        train_df["surge_level"] = train_df["surge_level"].astype(float)

        config = GNNConfig(hidden_channels=128, num_layers=2, epochs=args.epochs, conv_type="sage")
        model = SurgeGNNModel(config=config, graph=G)
        model.set_graph(G)
        metrics = model.fit(train_df, target_col="surge_level", node_col="node_id", node_features=node_features)
        print(f"\n[H={h}h] Final metrics: {metrics}")

        # Inference: predict surge for all port nodes at snapshot
        pred = model.predict({nid: node_features[nid] for nid in G.nodes() if nid in node_features})
        port_preds = {}
        for ap in activity_ports:
            ids = _map_activity_port_to_node_ids(ap)
            for comp in _activity_port_to_component_port_names(ap):
                nid = name_to_node.get(_normalize_name(comp))
                if nid and nid in pred:
                    port_preds[comp] = pred[nid]
        results["horizons"][str(h)] = {"metrics": metrics, "port_predictions": port_preds}

    if args.export_json:
        out_path = Path(args.export_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2))
        print(f"\nWrote predictions JSON to: {out_path}")


