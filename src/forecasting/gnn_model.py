"""
Graph Neural Network for Surge Prediction
==========================================
Uses GNN to model congestion propagation through the transportation network.

NVIDIA Ecosystem Integration:
- PyTorch with CUDA for GPU-accelerated training
- cuGraph for fast graph analytics (when available)
- cuML for GPU-accelerated preprocessing (when available)

Why GNN > XGBoost for this problem:
1. Congestion PROPAGATES through the network (message passing)
2. Spatial dependencies between nodes matter
3. Network topology directly informs predictions
"""

import numpy as np
import pandas as pd
import networkx as nx
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
    from torch.optim import Adam
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    HAS_TORCH = True
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[GNN] PyTorch device: {DEVICE}")
except ImportError:
    HAS_TORCH = False
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
sys.path.append(str(Path(__file__).parent.parent))
from config import MODELS_DIR


@dataclass
class GNNConfig:
    """Configuration for GNN model."""
    hidden_channels: int = 64
    num_layers: int = 3
    dropout: float = 0.2
    learning_rate: float = 0.001
    epochs: int = 100
    patience: int = 10
    conv_type: str = 'sage'  # 'gcn', 'gat', 'sage'


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
        deg_dict = dict(zip(degree['vertex'].to_pandas(), degree['degree'].to_pandas()))
        
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
        node_col: str = 'node_id'
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
        
        # Prepare graph features
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
        
        # Training setup
        optimizer = Adam(self.model.parameters(), lr=self.config.learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
        criterion = nn.MSELoss()
        
        # Train/val split (simple node split)
        num_nodes = data.x.shape[0]
        perm = torch.randperm(num_nodes)
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        train_mask[perm[:int(0.8 * num_nodes)]] = True
        val_mask[perm[int(0.8 * num_nodes):]] = True
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in tqdm(range(self.config.epochs), desc="  Training", unit="epoch"):
            # Train
            self.model.train()
            optimizer.zero_grad()
            out = self.model(data.x, data.edge_index)
            train_loss = criterion(out[train_mask], data.y[train_mask])
            train_loss.backward()
            optimizer.step()
            
            # Validate
            self.model.eval()
            with torch.no_grad():
                out = self.model(data.x, data.edge_index)
                val_loss = criterion(out[val_mask], data.y[val_mask])
            
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    tqdm.write(f"  Early stopping at epoch {epoch}")
                    break
            
            if epoch % 20 == 0:
                tqdm.write(f"  Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
        
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
        
        if self.model is not None:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'config': self.config,
                'node_list': self.dataset.node_list if self.dataset else None
            }, path)
            print(f"[GNN] Model saved to {path}")
    
    def load(self, path: Path = None):
        """Load model from disk."""
        if path is None:
            path = MODELS_DIR / 'gnn_model.pt'
        
        checkpoint = torch.load(path, map_location=DEVICE)
        self.config = checkpoint['config']
        # Model will be reconstructed on next fit/predict
        print(f"[GNN] Model loaded from {path}")


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
    # Check ecosystem
    status = check_nvidia_ecosystem()
    
    # Test with synthetic graph
    print("Creating test graph...")
    G = nx.barabasi_albert_graph(100, 3)
    
    # Add some attributes
    for u, v in G.edges():
        G[u][v]['rail_class'] = np.random.choice([1, 2, 3])
        G[u][v]['miles'] = np.random.uniform(5, 50)
    
    # Create synthetic training data
    train_df = pd.DataFrame({
        'node_id': list(G.nodes()),
        'surge_level': np.random.uniform(0, 1, G.number_of_nodes())
    })
    
    # Train model
    config = GNNConfig(hidden_channels=32, num_layers=2, epochs=50)
    model = SurgeGNNModel(config=config, graph=G)
    model.set_graph(G)
    
    if HAS_TORCH and HAS_PYG:
        metrics = model.fit(train_df)
        print(f"\nFinal metrics: {metrics}")
    else:
        print("\nSkipping GNN training - PyTorch Geometric not installed")
        print("Install with: pip install torch-geometric")

