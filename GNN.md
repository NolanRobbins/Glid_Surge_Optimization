### Glid Surge Optimization – GNN Model (Pitch Specs)

This document is the “sales sheet” for our **Graph Neural Network (GNN)** approach powering the Port → Truck → Rail surge forecasting and utilization optimization demo.

---

### Why a GNN is the best model for this competition
The challenge (`competition.txt`) is fundamentally **networked**: congestion is not an isolated time series per port—it **propagates through rail topology**, intermodal terminals, and chokepoints. A GNN is purpose-built to:

- **Model propagation**: a surge at a port can raise risk at downstream rail nodes/terminals via connectivity.
- **Learn from topology + time**: combine **temporal signals** (lags/rolling seasonality) with **graph structure** (centrality/connectivity).
- **Produce node-level intelligence**: predictions for **ports + terminals + rail nodes**, not just aggregated KPIs.

This maps directly to the required insights:
- **24–72 hour surge predictions**
- **rail congestion / underutilization warnings**
- **chokepoint risk identification**
- **dispatch window recommendations**
- **unified dashboard visualization**

---

### What our GNN outputs (actionable, not just a score)
Our model is designed as a **multi-task risk forecaster** across the multimodal network:

- **Port surge risk** (forecast horizon: 24–72h)
- **Rail congestion risk** (propagated across the rail graph)
- **Terminal utilization risk** (terminal nodes driven by nearby congestion)
- **Drayage delay risk** (first-mile pressure proxy)
- **Chokepoint risk** (centrality × congestion: “high-importance node under stress”)

This directly supports real operator decisions (Judging: “Value & Impact” + “Usability”):
- Where will congestion hit next?
- Which terminals should we shift volume to?
- What is the best dispatch window (avoid peak surge/peak risk)?

---

### Model architecture (what we actually trained)
**Training script**: `train_gnn_multitask_v3.py`  
**Backbone**: **GraphSAGE** (`SAGEConv`) with **BatchNorm**, ReLU, Dropout  
**Heads**: 5 task-specific MLP heads (`Linear → ReLU → Dropout → Linear → Sigmoid`)

**Fast configuration used for the demo run (competition-ready)**:
- **Graph**: ~**197k** rail nodes, ~**225k** rail edges (+ ports/terminals attached)
- **Ports in data**: **106** US ports (2019–2024 daily history)
- **Temporal samples**: ~**2,096** time slices for 24h horizon (train/val/test split by time)
- **Feature width (`in_channels`)**: **22**
  - **17 temporal features**: lags, rolling means/std, seasonality, YoY change, rate-of-change
  - **5 graph features**: PageRank, degree, betweenness, is_port, is_terminal
- **Hidden width**: **256**
- **Depth**: **3** GraphSAGE layers
- **Dropout**: **0.2**
- **Optimizer**: AdamW

Why this is a good “fast” competition model:
- large enough to learn non-trivial patterns
- small enough to train reliably within hackathon time constraints
- scales to the full rail graph (not a toy subgraph)

---

### NVIDIA / DGX Spark (“GX10”) story: why this runs better here
This is a core judging category (“NVIDIA Ecosystem & Spark Utility”).

We leveraged the NVIDIA stack in **two high-impact places**:

#### 1) GPU-accelerated spatial graph attachment (RAPIDS cuML)
When building the multimodal graph, we must connect ports/terminals to the nearest rail nodes.  
Doing this naively is expensive (ports × rail nodes distance checks).

- We use **RAPIDS cuML** `NearestNeighbors` (GPU) in `src/graph/builder.py` to connect locations to the rail network quickly.
- This is a real acceleration story: it turns a “Python-loop bottleneck” into a GPU-optimized neighbor search.

#### 2) GPU-accelerated GNN training (PyTorch + CUDA, TF32 enabled)
Training runs on GPU (`cuda`) and enables **TF32** for faster matmuls:
- `torch.backends.cuda.matmul.allow_tf32 = True`
- `torch.backends.cudnn.allow_tf32 = True`

In practice we see sustained high GPU utilization during training when the pipeline is healthy—exactly what we want for the “Performance” frontier factor.

**Note on VRAM**: VRAM usage doesn’t need to “fill up” to be optimal. For full-graph GNNs, the goal is **high GPU utilization**, not necessarily maximum memory consumption. We’ve observed ~5–6GB VRAM with strong utilization during training, which indicates we’re compute-bound (good).

---

### Why judges should care (mapped to the rubric)
From `judging_criteria.txt`:

#### Technical Execution & Completeness (30)
- **End-to-end working system**: real data ingestion → feature engineering → graph build → training/inference → dashboard outputs.
- **Complex pipeline under the hood**: spatio-temporal learning on a 197k-node rail network is not a “static dashboard”.

#### NVIDIA Ecosystem & Spark Utility (30)
- **Uses major NVIDIA tools**: RAPIDS **cuML** + CUDA PyTorch training.
- **Spark story**: this pipeline benefits from running locally on NVIDIA hardware for speed/scale and rapid iteration—exactly what a “systems engineering” build should demonstrate.

#### Value & Impact (20)
- **Non-obvious insights**: chokepoint risk isn’t “traffic at 5pm”—it’s topology-aware risk at critical nodes.
- **Usability**: outputs translate directly to operational actions (dispatch windows, reroute/reposition recommendations, early warnings).

#### Frontier Factor (20)
- **Creativity**: multi-task GNN forecasting across a national rail graph (ports + terminals + rail nodes).
- **Performance**: GPU-accelerated neighbor search + GPU training; scale to hundreds of thousands of nodes.

---

### Demo talking points (30-second pitch)
- “This is not a point forecast; it’s a **network forecaster**. Surges propagate across the rail graph.”
- “We train on **years of daily port activity** and project risk across **197k rail nodes**.”
- “We use **RAPIDS cuML on GPU** to attach ports/terminals to the rail network fast, then run **GraphSAGE** training on GPU with TF32.”
- “Outputs are operational: **dispatch windows**, **terminal utilization risk**, and **chokepoint alerts**—ready for a real planner tomorrow.”


