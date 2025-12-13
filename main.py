"""
Glid Surge Optimization System - Main Entry Point
==================================================
Orchestrates data loading, graph construction, model training,
and optimization for first-mile rail logistics.

Usage:
    python main.py --stage all          # Run full pipeline
    python main.py --stage data         # Load and process data only
    python main.py --stage graph        # Build transportation graph
    python main.py --stage train        # Train forecasting models
    python main.py --stage optimize     # Run optimization
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config import (
    PROJECT_ROOT, OUTPUT_DIR, MODELS_DIR,
    GLID_CLIENTS, US_PORTS, RAIL_TERMINALS
)


def stage_data():
    """Stage 1: Load and process all datasets."""
    print("\n" + "="*60)
    print("STAGE 1: DATA LOADING & PROCESSING")
    print("="*60)
    
    from data.loaders import (
        load_rail_nodes, load_rail_lines, 
        load_weather_data, load_ais_vessels
    )
    
    # Load rail network
    print("\n[1/4] Loading rail network...")
    rail_nodes = load_rail_nodes(filter_us_only=True)
    rail_lines = load_rail_lines(filter_us_only=True)
    
    # Load weather
    print("\n[2/4] Loading weather data...")
    weather_daily = load_weather_data(hourly=False)
    weather_hourly = load_weather_data(hourly=True)
    
    # Load AIS (sample for speed)
    print("\n[3/4] Loading AIS vessel data...")
    ais_vessels = load_ais_vessels(sample_n=50000)
    
    # Summary
    print("\n[4/4] Data Loading Summary:")
    print(f"  • Rail nodes: {len(rail_nodes):,}")
    print(f"  • Rail lines: {len(rail_lines):,}")
    print(f"  • Weather daily: {len(weather_daily):,} records")
    print(f"  • Weather hourly: {len(weather_hourly):,} records")
    print(f"  • AIS vessels: {len(ais_vessels):,} records")
    
    return {
        'rail_nodes': rail_nodes,
        'rail_lines': rail_lines,
        'weather_daily': weather_daily,
        'weather_hourly': weather_hourly,
        'ais_vessels': ais_vessels
    }


def stage_graph(data: dict = None):
    """Stage 2: Build multi-modal transportation graph."""
    print("\n" + "="*60)
    print("STAGE 2: GRAPH CONSTRUCTION")
    print("="*60)
    
    from data.loaders import load_rail_nodes, load_rail_lines
    from graph.builder import (
        build_rail_graph, add_location_nodes,
        connect_locations_to_graph, extract_client_subgraphs
    )
    
    # Load data if not provided
    if data is None:
        rail_nodes = load_rail_nodes(filter_us_only=True)
        rail_lines = load_rail_lines(filter_us_only=True)
    else:
        rail_nodes = data['rail_nodes']
        rail_lines = data['rail_lines']
    
    # Build base rail graph
    print("\n[1/3] Building rail network graph...")
    G = build_rail_graph(rail_nodes, rail_lines)
    
    # Add Glid client locations
    print("\n[2/3] Adding Glid client nodes...")
    G = add_location_nodes(G, GLID_CLIENTS, 'client')
    G = connect_locations_to_graph(G, GLID_CLIENTS, 'client', max_connection_miles=20)
    
    # Extract subgraphs for each client
    print("\n[3/3] Extracting 50-mile subgraphs per client...")
    subgraphs = extract_client_subgraphs(G, GLID_CLIENTS, radius_miles=50)
    
    # Summary
    print("\nGraph Construction Summary:")
    print(f"  • Total nodes: {G.number_of_nodes():,}")
    print(f"  • Total edges: {G.number_of_edges():,}")
    print(f"  • Client subgraphs: {len(subgraphs)}")
    for client_id, sg in subgraphs.items():
        print(f"    - {client_id}: {sg.number_of_nodes()} nodes, {sg.number_of_edges()} edges")
    
    return {'graph': G, 'subgraphs': subgraphs}


def stage_train(data: dict = None, graph_data: dict = None):
    """Stage 3: Train forecasting models."""
    print("\n" + "="*60)
    print("STAGE 3: MODEL TRAINING")
    print("="*60)
    
    import pandas as pd
    import numpy as np
    from forecasting.surge_model import SurgePredictionModel
    from forecasting.dwell_model import DwellTimeModel
    from data.loaders import load_weather_data
    
    # Get graph if available (for graph-enhanced features)
    graph = None
    if graph_data is not None and 'graph' in graph_data:
        graph = graph_data['graph']
        print(f"\n[INFO] Using transportation graph for topology features")
        print(f"       Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    # Create synthetic training data since we don't have real port call history
    # In production, this would come from actual port activity data
    print("\n[1/3] Preparing training data...")
    
    # Generate synthetic port activity data
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=365, freq='D')
    ports = ['Port of LA', 'Port of Long Beach', 'Port of Oakland']
    
    training_data = []
    for port in ports:
        base = 80 + np.random.randint(0, 40)
        seasonal = np.sin(np.arange(365) / 365 * 2 * np.pi) * 20
        weekly = np.sin(np.arange(365) / 7 * 2 * np.pi) * 10
        noise = np.random.randn(365) * 8
        
        for i, date in enumerate(dates):
            training_data.append({
                'date': date,
                'portname': port,
                'portcalls': max(10, int(base + seasonal[i] + weekly[i] + noise[i]))
            })
    
    train_df = pd.DataFrame(training_data)
    print(f"  Generated {len(train_df):,} training samples")
    
    # Load weather data for features
    weather_df = None
    try:
        weather_df = load_weather_data(hourly=False)
        print(f"  Loaded {len(weather_df):,} weather records")
    except Exception as e:
        print(f"  Warning: Could not load weather data: {e}")
    
    # Train surge prediction model (with graph topology features if available)
    print("\n[2/3] Training surge prediction model...")
    surge_model = SurgePredictionModel(model_type='xgboost', graph=graph)
    surge_metrics = surge_model.fit(train_df, weather_df=weather_df)
    
    print("\n  Surge Model Performance:")
    for horizon, metrics in surge_metrics.items():
        print(f"    {horizon}h: MAE={metrics['mae']:.2f}, R²={metrics['r2']:.3f}")
    
    # Save surge model
    surge_model.save()
    print(f"  Model saved to {MODELS_DIR / 'surge_model.joblib'}")
    
    # Train dwell time model (using synthetic data)
    print("\n[3/3] Training dwell time model...")
    
    # Generate synthetic dwell data
    dwell_data = []
    for _ in range(1000):
        congestion = np.random.random()
        rail_cap = np.random.randint(20, 100)
        hour = np.random.randint(0, 24)
        day = np.random.randint(0, 7)
        is_rainy = np.random.random() < 0.2
        is_windy = np.random.random() < 0.1
        
        # Synthetic dwell time based on factors
        dwell = 12 + congestion * 24 - (rail_cap / 100) * 8
        dwell += 4 if is_rainy else 0
        dwell += 8 if is_windy else 0
        dwell += 2 if day >= 5 else 0  # Weekend
        dwell += np.random.randn() * 3
        
        dwell_data.append({
            'congestion_level': congestion,
            'rail_capacity_pct': rail_cap,
            'hour_of_day': hour,
            'day_of_week': day,
            'is_weekend': 1 if day >= 5 else 0,
            'is_night': 1 if hour < 6 or hour > 22 else 0,
            'is_rainy': int(is_rainy),
            'is_high_wind': int(is_windy),
            'hour_sin': np.sin(2 * np.pi * hour / 24),
            'hour_cos': np.cos(2 * np.pi * hour / 24),
            'dwell_hours': max(4, dwell)
        })
    
    dwell_df = pd.DataFrame(dwell_data)
    
    dwell_model = DwellTimeModel()
    dwell_metrics = dwell_model.fit(dwell_df)
    
    print(f"\n  Dwell Model Performance:")
    print(f"    MAE: {dwell_metrics['mae']:.2f} hours")
    print(f"    RMSE: {dwell_metrics['rmse']:.2f} hours")
    
    dwell_model.save()
    print(f"  Model saved to {MODELS_DIR / 'dwell_model.joblib'}")
    
    return {
        'surge_model': surge_model,
        'dwell_model': dwell_model,
        'surge_metrics': surge_metrics,
        'dwell_metrics': dwell_metrics
    }


def stage_optimize(data: dict = None, graph_data: dict = None, models: dict = None):
    """Stage 4: Run optimization examples."""
    print("\n" + "="*60)
    print("STAGE 4: OPTIMIZATION")
    print("="*60)
    
    from optimization.dispatcher import DispatchScheduler
    from optimization.vrp_solver import VRPSolver
    from optimization.cost_calculator import CostCalculator
    import pandas as pd
    
    # Run dispatch scheduling
    print("\n[1/3] Dispatch Scheduling...")
    scheduler = DispatchScheduler()
    
    plan = scheduler.create_dispatch_plan(
        location="Port of Los Angeles",
        container_count=100
    )
    
    print(f"\n  Dispatch Plan for {plan.total_containers} containers:")
    print(f"  Expected daily savings: ${plan.expected_cost_savings:,.0f}")
    print(f"  Utilization score: {plan.utilization_score:.1%}")
    print(f"  Optimal windows: {len(plan.windows)}")
    
    for window in plan.windows:
        print(f"\n    [{window.priority.upper()}] {window.start_time.strftime('%a %H:%M')} - {window.end_time.strftime('%H:%M')}")
        print(f"      {window.route_recommendation}")
    
    # Run VRP example
    print("\n[2/3] Vehicle Routing Optimization...")
    import numpy as np
    
    solver = VRPSolver(num_vehicles=3, vehicle_capacity=40, max_route_distance=100)
    
    # Create test scenario: depot + 8 pickup points
    np.random.seed(42)
    distance_matrix = np.random.rand(9, 9) * 30 + 5
    np.fill_diagonal(distance_matrix, 0)
    distance_matrix = (distance_matrix + distance_matrix.T) / 2  # Make symmetric
    
    demands = [0, 12, 8, 15, 10, 18, 7, 14, 11]  # Depot = 0
    
    solution = solver.solve(distance_matrix, demands)
    
    print(f"\n  VRP Solution:")
    print(f"  Vehicles used: {solution.vehicles_used}")
    print(f"  Total distance: {solution.total_distance:.1f} miles")
    print(f"  Total time: {solution.total_time:.1f} hours")
    print(f"  Optimal: {solution.is_optimal}")
    
    for i, route in enumerate(solution.routes):
        route_demand = sum(demands[n] for n in route if n != 0)
        print(f"    Vehicle {i+1}: {' → '.join(map(str, route))} (load: {route_demand})")
    
    # Cost analysis
    print("\n[3/3] Cost Savings Analysis...")
    calc = CostCalculator()
    
    report = calc.calculate_savings(
        containers=100,
        distance_miles=45,
        traditional_dwell=48,
        glid_dwell=12
    )
    
    print(calc.format_savings_report(report))
    
    return {
        'dispatch_plan': plan,
        'vrp_solution': solution,
        'savings_report': report
    }


def run_pipeline(stages: list = None):
    """Run the complete optimization pipeline."""
    if stages is None:
        stages = ['data', 'graph', 'train', 'optimize']
    
    print("\n" + "="*60)
    print("GLID SURGE OPTIMIZATION SYSTEM")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    results = {}
    
    if 'data' in stages:
        results['data'] = stage_data()
    
    if 'graph' in stages:
        data = results.get('data')
        results['graph'] = stage_graph(data)
    
    if 'train' in stages:
        data = results.get('data')
        graph_data = results.get('graph')
        results['models'] = stage_train(data, graph_data)
    
    if 'optimize' in stages:
        results['optimization'] = stage_optimize(
            results.get('data'),
            results.get('graph'),
            results.get('models')
        )
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    return results


def main():
    """Main entry point with CLI support."""
    parser = argparse.ArgumentParser(
        description="Glid Surge Optimization System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --stage all          Run full pipeline
  python main.py --stage data         Load data only
  python main.py --stage graph        Build graph only
  python main.py --stage train        Train models only
  python main.py --stage optimize     Run optimization only
  python main.py --stage data,graph   Run multiple stages
        """
    )
    
    parser.add_argument(
        '--stage', '-s',
        type=str,
        default='all',
        help='Pipeline stage(s) to run: all, data, graph, train, optimize (comma-separated)'
    )
    
    args = parser.parse_args()
    
    # Parse stages
    if args.stage == 'all':
        stages = ['data', 'graph', 'train', 'optimize']
    else:
        stages = [s.strip() for s in args.stage.split(',')]
    
    # Run pipeline
    results = run_pipeline(stages)
    
    return results


if __name__ == "__main__":
    main()

