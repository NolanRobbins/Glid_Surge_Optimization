# Glid First-Mile Surge Optimization System
## Implementation Strategy & Architecture

---

## 1. Executive Summary

This document outlines the strategy for building a **Port-to-Rail First-Mile Optimization System** for Glid's autonomous rail vehicles (Raden/Glider-M). The system will:

- **Minimize dwell time** at ports/warehouses
- **Maximize round-trip load efficiency** ("there & back")
- **Optimize routes within 40-50 mile radius** 
///// **Account for Class 1-3 rail regulations and train spacing**
- **Provide 24-72 hour surge predictions, Beyond hardware, Glīd wins by using an on-device LLM trained on massive port/rail datasets to predict the *exact* optimal pickup window and routing strategy (Street vs. Rail) that beats standard drayage every time.
**
- **Demonstrate customer cost savings vs existing drayage industry**

---

## 2. Glid Client Location Analysis

### Current Clients to Model:
| Client | Type | Likely Region | Strategy Focus |
|--------|------|---------------|----------------|
| Port of Woodland | Port | CA (Sacramento Area) | Port-to-rail corridor optimization |
| Sierra | Industrial | CA/NV | Mountain routing, weather impacts |
| Newlab | Tech Hub | NYC Area | Urban first-mile logistics |
| County of Riverside | Government | Southern CA | Regional freight distribution |
| Taylor Transport Inc | Carrier | Multi-region | Fleet coordination |
| Portland Vancouver Junction Railroad | Short Line | OR/WA | Class 2-3 rail connections |
| Great Plains Industrial Park | Industrial | Midwest (KS/NE) | Agricultural freight surge |
| Kansas Proving Grounds | Testing | KS | Low-volume, high-value cargo |
| Mendocino Railway | Short Line | Northern CA | Scenic/heritage + freight hybrid |

###Potential next clients 
    
###Port of Los Angeles: The busiest container port in the Western Hemisphere; faces massive pressure to reduce truck emissions and congestion. 2. Port of Long Beach: The "sister" port to Los Angeles; heavily invested in green technology and rail efficiency. 3. Pacific Harbor Line (PHL): Critical Entity. PHL is the short-line railroad that manages all rail traffic inside the LA/Long Beach ports. They are the "gatekeeper" Glid would need to partner with to operate on port tracks. 4. BNSF Railway / Union Pacific (Intermodal Divisions): While these are national giants, Glid would likely target their specific intermodal facilities in Hobart or Commerce (near LA) to prove they can move containers off the main lines without cranes.


### Key Geographic Clusters:
1. **West Coast Corridor** (LA/Long Beach → Inland Empire → Portland)
2. **Central Plains Hub** (Kansas City region)
3. **Northern California** (Oakland → Sacramento → Woodland)

---

## 3. System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    GLID SURGE OPTIMIZATION SYSTEM                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────────────┐ │
│  │ DATA INGESTION│   │  FORECASTING │   │   ROUTE OPTIMIZATION     │ │
│  │    LAYER      │ → │    ENGINE    │ → │        ENGINE            │ │
│  └──────────────┘   └──────────────┘   └──────────────────────────┘ │
│         │                   │                       │                │
│         ▼                   ▼                       ▼                │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────────────┐ │
│  │ Port Activity │   │ Surge Alerts │   │  Dispatch Scheduler      │ │
│  │ Rail Nodes    │   │ 24-72hr      │   │  Load Balancer           │ │
│  │ Truck Times   │   │ predictions  │   │  Dwell Minimizer         │ │
│  │ Weather       │   │              │   │                          │ │
│  │ AIS Vessels   │   │              │   │                          │ │
│  └──────────────┘   └──────────────┘   └──────────────────────────┘ │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                    REAL-TIME DASHBOARD                          │ │
│  │  • Port surge status    • Optimal dispatch windows              │ │
│  │  • Rail utilization     • Cost savings metrics                  │ │
│  │  • First-mile delays    • Route recommendations                 │ │
│  └─────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. Graph-Based Routing Model (Your Core Intuition)

You're absolutely right about using a **graph model**. Here's how we implement it:

### 4.1 Multi-Modal Transportation Graph

```python
# Graph Node Types:
NODES = {
    'port': 'Origin nodes (cargo arrival)',
    'warehouse': 'Customer staging locations', 
    'rail_terminal': 'Class 1-3 rail connection points',
    'customer_site': 'Glid vehicle home bases'
}

# Edge Types with Weights:
EDGES = {
    'road_segment': {
        'weight': 'travel_time_minutes',  # From truck times data
        'capacity': 'vehicles_per_hour',
        'distance': 'miles'
    },
    'rail_segment': {
        'weight': 'transit_time_hours',
        'class': '1|2|3',  # FRA classification
        'train_spacing': 'minutes_between_trains',
        'capacity': 'cars_per_day'
    }
}
```

### 4.2 Graph Construction Strategy

1. **Extract 40-50 Mile Radius Subgraphs** around each Glid client
2. **Build Bipartite Matching** between ports/warehouses and rail terminals
3. **Weight edges** with:
   - Travel time (from BTS/ATRI county-to-county data)
   - Weather-adjusted delays (from Open-Meteo data)
   - Rail class restrictions (from NTAD rail network)
   - Historical congestion patterns

### 4.3 Key Algorithms to Implement

| Algorithm | Purpose | NVIDIA Acceleration |
|-----------|---------|---------------------|
| **Dijkstra's/A*** | Shortest path finding | cuGraph GPU-accelerated |
| **Vehicle Routing Problem (VRP)** | Multi-vehicle dispatch | cuOpt solver |
| **Max Flow / Min Cost** | Load balancing | RAPIDS cuGraph |
| **Traveling Salesman (TSP)** | Round-trip optimization | cuOpt |
| **Time-Window Scheduling** | Dwell minimization | Custom CUDA kernels |

---

## 5. Data Integration Plan

### 5.1 Available Datasets → Use Cases

| Dataset | Records | Key Use |
|---------|---------|---------|
| **Rail Nodes (NTAD)** | 250,129 nodes (197K US) | Build rail network graph |
| **Rail Lines (NTAD)** | 302,689 segments | Edge connectivity, Class 1 identification |
| **Truck Travel Times (BTS/ATRI)** | ~3.64M county pairs | Road travel time estimation |
| **AIS Vessel Tracking** | 823K+ records | Port arrival predictions |
| **Global Port Activity** | Daily by port | Surge detection signals |
| **PortWatch Chokepoints** | Daily transit data | Supply chain disruption alerts |
| **PortWatch Disruptions** | Event database | Risk modeling |
| **Weather (hourly/daily)** | 52 US hubs | Delay prediction features |
| **Logistics Fleet** | Trucks, costs, freight | Drayage modeling |

### 5.2 Data Pipeline

```
Raw Data → Preprocessing → Feature Engineering → Model Training → Inference
    │           │                   │                  │            │
    ▼           ▼                   ▼                  ▼            ▼
  GeoJSON    Spatial Join       Lag Features      XGBoost/      Real-time
  CSV         County FIPS       Rolling Avgs      LightGBM      Predictions
  Excel       Coordinate        Weather Encode    RAPIDS cuML   Dashboard
              Matching          Time Features
```

---

## 6. Optimization Objectives (Formalized)

### 6.1 Primary Objective Function

```
MINIMIZE: Total_Cost = α(Dwell_Time) + β(Empty_Miles) + γ(Late_Arrivals) + δ(Energy_Cost)

SUBJECT TO:
  - Route distance ≤ 50 miles (one-way)
  - Train spacing ≥ minimum_headway[rail_class]
  - Vehicle capacity ≤ max_load
  - Time windows respected
  - Return trips maximize backhaul load
```

### 6.2 Constraint Implementation

```src/forecasting
# Class 1-3 Rail Constraints
RAIL_CONSTRAINTS = {
    'class_1': {'max_speed_mph': 60, 'min_headway_min': 20, 'tonnage_limit': 'unlimited'},
    'class_2': {'max_speed_mph': 25, 'min_headway_min': 30, 'tonnage_limit': 'moderate'},
    'class_3': {'max_speed_mph': 10, 'min_headway_min': 45, 'tonnage_limit': 'limited'}
}

# 40-50 Mile Radius Constraint
def is_valid_route(origin, destination, graph):
    distance = nx.shortest_path_length(graph, origin, destination, weight='miles')
    return 40 <= distance <= 50
```

---

## 7. Forecasting Model Architecture

### 7.1 Surge Prediction (24-72 hours)

**Features:**
- Port call counts (lagged 1-7 days)
- AIS inbound vessel count & ETA
- Day of week, month seasonality
- Weather conditions (precipitation, wind, visibility)
- Historical surge patterns
- Chokepoint congestion (Suez, Panama data as leading indicators)

**Target Variable:**
- Port cargo volume (next 24/48/72 hours)
- Rail terminal utilization rate

**Model:**
- Gradient Boosting (XGBoost/LightGBM)
- LSTM for sequence patterns
- Ensemble for robustness

### 7.2 Dwell Time Prediction

**Features:**
- Current port congestion
- Available rail capacity
- Truck availability
- Time of day
- Weather conditions

**Target:**
- Expected dwell time in hours

---

## 8. Dashboard Design (Deliverable)

### 8.1 Key Metrics to Display

| Metric | Description | Visualization |
|--------|-------------|---------------|
| **Surge Alert Level** | Red/Yellow/Green by port | Color-coded map |
| **Predicted Dwell Time** | Hours at each location | Gauge charts |
| **Optimal Dispatch Windows** | Best 4-hour windows | Timeline view |
| **Route Efficiency Score** | Load factor (both ways) | % indicator |
| **Cost Savings** | $ saved vs traditional | Running counter |
| **Rail Utilization** | Capacity usage by terminal | Bar charts |

### 8.2 Technology Stack

```
Frontend: Dash + Plotly (interactive)
Backend: FastAPI (REST endpoints)
Maps: Folium/Plotly Mapbox (geospatial viz)
Database: Parquet files (fast columnar)
Compute: RAPIDS cuDF/cuGraph (GPU acceleration)
```

---

## 9. Implementation Roadmap

### Phase 1: Data Foundation (Days 1-2)
- [ ] Parse and clean all datasets
- [ ] Build unified coordinate system
- [ ] Create rail network graph from NTAD data
- [ ] Map Glid client locations to nearest nodes

### Phase 2: Graph Model (Days 2-3)
- [ ] Construct multi-modal transportation graph
- [ ] Implement 40-50 mile radius filtering
- [ ] Add travel time weights from BTS/ATRI data
- [ ] Integrate weather delay factors

### Phase 3: Forecasting (Days 3-4)
- [ ] Feature engineering pipeline
- [ ] Train surge prediction models
- [ ] Validate on holdout data
- [ ] Deploy inference endpoints

### Phase 4: Optimization (Days 4-5)
- [ ] Implement VRP solver for dispatch
- [ ] Add round-trip load balancing
- [ ] Integrate rail spacing constraints
- [ ] Minimize dwell time objective

### Phase 5: Dashboard & Demo (Days 5-6)
- [ ] Build interactive Dash dashboard
- [ ] Create customer savings calculator
- [ ] Prepare presentation materials
- [ ] Document "Spark Story" for judges

---

## 10. NVIDIA DGX Spark Leverage Points

### 10.1 Why This Runs Better on DGX Spark

1. **128GB Unified Memory**: Hold entire rail network graph (250K+ nodes, 300K+ edges) in GPU memory alongside ML models
2. **RAPIDS cuGraph**: GPU-accelerated graph algorithms for route optimization
3. **RAPIDS cuML**: Fast gradient boosting training on large datasets
4. **cuOpt**: NVIDIA's optimization solver for VRP problems
5. **Local Inference**: Low-latency predictions without cloud roundtrips

### 10.2 Recommended NVIDIA Stack

```python
# GPU-accelerated imports
import cudf  # GPU DataFrames
import cugraph  # GPU Graph Analytics  
import cuml  # GPU Machine Learning
from nvidia_cuopt import cuopt  # Optimization solver
```

---

## 11. Success Metrics (Proving Customer Savings)

### 11.1 Key Performance Indicators

| KPI | Baseline (Traditional) | Glid Optimized | Improvement |
|-----|------------------------|----------------|-------------|
| Avg Dwell Time | 48 hours | Target: 12 hours | 75% reduction |
| Empty Mile % | 35% | Target: 10% | 71% reduction |
| On-Time Delivery | 82% | Target: 95% | +13 points |
| Cost per Container-Mile | $X.XX | $Y.YY | Z% savings |

### 11.2 Cost Savings Formula

```
Annual Savings = 
    (Reduced_Dwell_Hours × Hourly_Storage_Cost) +
    (Eliminated_Empty_Miles × Per_Mile_Cost) +
    (Avoided_Late_Penalties × Penalty_Rate) +
    (Optimized_Energy × Energy_Savings)
```

---

## 12. Next Steps

1. **Start with Graph Construction** - Parse rail nodes/lines into NetworkX
2. **Geocode Glid Clients** - Map client names to lat/long coordinates  
3. **Build 40-50 Mile Subgraphs** - Extract relevant network around each client
4. **Prototype Surge Model** - Quick XGBoost on port activity data
5. **Wire Up Dashboard** - Basic Dash app with map and metrics

Would you like me to start implementing any of these components?

