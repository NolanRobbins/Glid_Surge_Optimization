# Production Inference Guide

## Overview

Now that you have a trained GNN model on the top 11 ports, this guide explains how to run production inference and connect it to the dashboard.

## What's Been Set Up

### ✅ Completed Components

1. **Production Inference Script** (`run_production_inference.py`)
   - Loads trained GNN model
   - Generates predictions for 24, 48, 72h horizons
   - Creates dashboard payload JSON
   - Integrates dispatcher for optimal windows
   - Generates route optimization recommendations

2. **Dashboard Integration**
   - Dashboard already reads from `output/dashboard_payload.json`
   - All visualization components are ready
   - Auto-refreshes every 30 seconds (configurable)

3. **Optimization Models**
   - **Dispatcher** (`src/optimization/dispatcher.py`): Finds optimal dispatch windows
   - **VRP Solver** (`src/optimization/vrp_solver.py`): Vehicle routing optimization
   - **Cost Calculator** (`src/optimization/cost_calculator.py`): Cost savings analysis

## Running Production Inference

### Basic Usage

```bash
# Run inference with default settings
python run_production_inference.py

# This will:
# 1. Auto-detect your trained GNN model
# 2. Load latest port activity data
# 3. Generate predictions for 24, 48, 72h
# 4. Export to output/dashboard_payload.json
```

### Advanced Usage

```bash
# Specify model path
python run_production_inference.py --model-path models/checkpoints/gnn_best_checkpoint.pt

# Custom export path
python run_production_inference.py --export custom/path/payload.json

# Custom horizons
python run_production_inference.py --horizons 12,24,36,48
```

## Dashboard Workflow

### 1. Generate Predictions

```bash
python run_production_inference.py
```

This creates/updates `output/dashboard_payload.json` with:
- Surge predictions for all ports (24, 48, 72h)
- Route optimization options
- Optimal dispatch windows
- Cost savings estimates

### 2. Start Dashboard

```bash
# From project root
python -m src.dashboard.app

# Or if using Dash directly
cd src/dashboard
python app.py
```

The dashboard will:
- Load `output/dashboard_payload.json` automatically
- Display surge alerts based on predictions
- Show route recommendations
- Update every 30 seconds (checks for new payload)

## Integration Status

### ✅ Fully Integrated

- **GNN Surge Predictions** → Dashboard payload
- **Dispatcher Optimization** → Dispatch windows in payload
- **Route Options** → Route recommendations in payload

### 🔄 Partially Integrated

- **Route Adapter**: Currently uses simplified route generation. To use full route adapter:
  ```python
  # In run_production_inference.py, replace generate_route_options() with:
  route_adapter = RouteAdapter()
  route_options = route_adapter.compute_route(
      origin=[lon, lat],
      destination=[lon, lat],
      mode="auto"
  )
  ```

- **VRP Solver**: Available but not automatically called. Can be integrated for multi-vehicle scenarios.

### 📋 Next Steps (Optional Enhancements)

1. **Real-time Route Computation**
   - Integrate `RouteAdapter` for actual route calculation
   - Use Mapbox API for road segments
   - Query rail network for rail segments

2. **Scheduled Inference**
   - Set up cron job or scheduled task to run inference periodically
   - Example: Run every 6 hours to update predictions
   ```bash
   # Add to crontab
   0 */6 * * * cd /path/to/project && python run_production_inference.py
   ```

3. **API Integration**
   - The FastAPI server (`src/api/server.py`) can serve predictions
   - Dashboard could call API instead of reading JSON file
   - Enables real-time updates without file I/O

4. **Multi-Task GNN**
   - If you've trained the multi-task model (`train_gnn_multitask.py`), you can:
   - Predict rail congestion for ALL 197K nodes
   - Predict terminal utilization
   - Predict drayage delays
   - Predict chokepoint likelihood

## Current Workflow

```
┌─────────────────────────────────────────────────────────┐
│ 1. Train GNN Model (train_gnn.py)                       │
│    → Saves to models/checkpoints/gnn_best_checkpoint.pt  │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ 2. Run Production Inference (run_production_inference.py)│
│    → Loads model                                        │
│    → Generates predictions                              │
│    → Creates output/dashboard_payload.json              │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│ 3. Start Dashboard (src/dashboard/app.py)              │
│    → Reads dashboard_payload.json                       │
│    → Displays visualizations                           │
│    → Auto-refreshes every 30s                          │
└─────────────────────────────────────────────────────────┘
```

## Testing the Integration

### Quick Test

```bash
# 1. Run inference
python run_production_inference.py

# 2. Check payload was created
cat output/dashboard_payload.json | head -50

# 3. Start dashboard
python -m src.dashboard.app

# 4. Open browser to http://localhost:8050
#    (or port specified in DASHBOARD_CONFIG)
```

### Verify Predictions

The payload should contain:
- `horizons.24.port_predictions`: Surge levels for each port (24h)
- `horizons.48.port_predictions`: Surge levels (48h)
- `horizons.72.port_predictions`: Surge levels (72h)
- `route_options.routes`: Array of route recommendations
- `dispatch_windows`: Optimal dispatch time windows

## Troubleshooting

### Model Not Found

```
FileNotFoundError: Model not found at models/gnn_model.pt
```

**Solution**: Train model first or specify path:
```bash
python run_production_inference.py --model-path models/checkpoints/gnn_best_checkpoint.pt
```

### No Recent Data

```
ValueError: No recent data found for inference
```

**Solution**: Ensure port activity data is up to date:
```bash
python data/download_datasets.py  # Update datasets
```

### Dashboard Shows "Waiting for Data"

**Solution**: Ensure `output/dashboard_payload.json` exists and has valid data:
```bash
python run_production_inference.py  # Regenerate payload
```

## Performance Notes

- **Inference Time**: ~30-60 seconds for 11 ports, 3 horizons
- **Memory Usage**: ~2-4 GB (depends on graph size)
- **GPU**: Uses GPU if available, falls back to CPU

## Production Deployment

For production deployment:

1. **Schedule Regular Inference**
   ```bash
   # Run every 6 hours
   0 */6 * * * /path/to/python /path/to/run_production_inference.py
   ```

2. **Monitor Dashboard**
   - Dashboard auto-refreshes and picks up new payloads
   - No restart needed when payload updates

3. **API Alternative** (Optional)
   - Use FastAPI server for real-time predictions
   - Dashboard calls API endpoints instead of reading file
   - Better for multi-user scenarios

## Summary

✅ **You can now link the model to the dashboard!**

The production inference script (`run_production_inference.py`) is ready to:
1. Load your trained GNN model
2. Generate surge predictions
3. Create dashboard payload
4. Integrate optimization models

**Next immediate step**: Run `python run_production_inference.py` and then start the dashboard to see your predictions visualized!
