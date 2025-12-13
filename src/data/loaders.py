"""
Data Loaders for Glid Surge Optimization
=========================================
Functions to load and preprocess all datasets.
"""

import json
import pandas as pd
import geopandas as gpd
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    RAIL_NODES_PATH, RAIL_LINES_PATH, TRUCK_TIMES_PATH,
    PORT_ACTIVITY_PATH, WEATHER_DAILY_PATH, WEATHER_HOURLY_PATH,
    AIS_VESSEL_PATH, PORTWATCH_CHOKEPOINTS_PATH, CLASS_1_RAILROADS
)


def load_rail_nodes(filter_us_only: bool = True) -> gpd.GeoDataFrame:
    """
    Load NTAD Rail Network Nodes.
    
    Args:
        filter_us_only: If True, only return US nodes (default: True)
        
    Returns:
        GeoDataFrame with rail node locations and attributes
    """
    print("Loading rail nodes...")
    gdf = gpd.read_file(RAIL_NODES_PATH)
    
    if filter_us_only:
        gdf = gdf[gdf['COUNTRY'] == 'US'].copy()
        print(f"  Filtered to {len(gdf):,} US nodes")
    else:
        print(f"  Loaded {len(gdf):,} total nodes")
    
    return gdf


def load_rail_lines(filter_us_only: bool = True, filter_class1: bool = False) -> gpd.GeoDataFrame:
    """
    Load NTAD Rail Network Lines.
    
    Args:
        filter_us_only: If True, only return US segments
        filter_class1: If True, only return Class I railroad segments
        
    Returns:
        GeoDataFrame with rail line segments and attributes
    """
    print("Loading rail lines...")
    gdf = gpd.read_file(RAIL_LINES_PATH)
    
    if filter_us_only:
        gdf = gdf[gdf['COUNTRY'] == 'US'].copy()
        print(f"  Filtered to {len(gdf):,} US segments")
    
    if filter_class1:
        gdf = gdf[gdf['RROWNER1'].isin(CLASS_1_RAILROADS)].copy()
        print(f"  Filtered to {len(gdf):,} Class I segments")
    
    return gdf


def load_truck_times(
    sample_frac: Optional[float] = None,
    origin_states: Optional[list] = None,
    dest_states: Optional[list] = None
) -> pd.DataFrame:
    """
    Load BTS/ATRI County-to-County Truck Travel Times.
    
    Args:
        sample_frac: Optional fraction to sample (for testing)
        origin_states: Optional list of origin state abbreviations to filter
        dest_states: Optional list of destination state abbreviations to filter
        
    Returns:
        DataFrame with travel time data between county pairs
    """
    print("Loading truck travel times...")
    
    # Use chunked reading for large file
    chunks = []
    for chunk in tqdm(pd.read_csv(TRUCK_TIMES_PATH, chunksize=100000), desc="  Reading chunks"):
        if origin_states:
            chunk = chunk[chunk['Origin State'].isin(origin_states)]
        if dest_states:
            chunk = chunk[chunk['Destination State'].isin(dest_states)]
        chunks.append(chunk)
    
    df = pd.concat(chunks, ignore_index=True)
    
    if sample_frac:
        df = df.sample(frac=sample_frac, random_state=42)
    
    print(f"  Loaded {len(df):,} county pairs")
    return df


def load_port_activity(
    ports: Optional[list] = None,
    country: str = "United States"
) -> pd.DataFrame:
    """
    Load Global Daily Port Activity data.
    
    Args:
        ports: Optional list of port names to filter
        country: Country to filter (default: United States)
        
    Returns:
        DataFrame with daily port activity metrics
    """
    print("Loading port activity data...")
    
    # Chunked reading for large file
    chunks = []
    for chunk in tqdm(pd.read_csv(PORT_ACTIVITY_PATH, chunksize=50000), desc="  Reading chunks"):
        if country:
            chunk = chunk[chunk['country'] == country]
        if ports:
            chunk = chunk[chunk['portname'].isin(ports)]
        chunks.append(chunk)
    
    df = pd.concat(chunks, ignore_index=True)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"  Loaded {len(df):,} records")
    return df


def load_weather_data(
    hourly: bool = False,
    locations: Optional[list] = None
) -> pd.DataFrame:
    """
    Load weather data from Open-Meteo.
    
    Args:
        hourly: If True, load hourly data; otherwise daily
        locations: Optional list of location names to filter
        
    Returns:
        DataFrame with weather conditions
    """
    path = WEATHER_HOURLY_PATH if hourly else WEATHER_DAILY_PATH
    print(f"Loading {'hourly' if hourly else 'daily'} weather data...")
    
    df = pd.read_csv(path)
    
    if hourly:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    else:
        df['date'] = pd.to_datetime(df['date'])
    
    if locations:
        df = df[df['location_name'].isin(locations)]
    
    print(f"  Loaded {len(df):,} records for {df['location_name'].nunique()} locations")
    return df


def load_ais_vessels(sample_n: Optional[int] = None) -> pd.DataFrame:
    """
    Load AIS Vessel Tracking data.
    
    Args:
        sample_n: Optional number of rows to sample
        
    Returns:
        DataFrame with vessel positions and ETAs
    """
    print("Loading AIS vessel tracking data...")
    
    if sample_n:
        df = pd.read_csv(AIS_VESSEL_PATH, nrows=sample_n)
    else:
        df = pd.read_csv(AIS_VESSEL_PATH)
    
    # Parse date columns
    date_cols = ['updated', 'etdSchedule', 'etd', 'atd', 'etaSchedule', 'eta', 'ata']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    
    print(f"  Loaded {len(df):,} vessel records")
    return df


def load_portwatch_chokepoints() -> gpd.GeoDataFrame:
    """
    Load PortWatch Chokepoint Transit data.
    
    Returns:
        GeoDataFrame with daily chokepoint transit volumes
    """
    print("Loading PortWatch chokepoints...")
    gdf = gpd.read_file(PORTWATCH_CHOKEPOINTS_PATH)
    gdf['date'] = pd.to_datetime(gdf['date'])
    print(f"  Loaded {len(gdf):,} records")
    return gdf


def load_all_data(
    filter_us: bool = True,
    sample_truck_times: float = 0.1
) -> Dict[str, Any]:
    """
    Load all datasets into a single dictionary.
    
    Args:
        filter_us: Filter to US-only data where applicable
        sample_truck_times: Fraction of truck times to sample
        
    Returns:
        Dictionary with all loaded datasets
    """
    return {
        'rail_nodes': load_rail_nodes(filter_us_only=filter_us),
        'rail_lines': load_rail_lines(filter_us_only=filter_us),
        'truck_times': load_truck_times(sample_frac=sample_truck_times),
        'weather_daily': load_weather_data(hourly=False),
        'weather_hourly': load_weather_data(hourly=True),
        'ais_vessels': load_ais_vessels(sample_n=10000),  # Sample for speed
    }


if __name__ == "__main__":
    # Test loading
    nodes = load_rail_nodes()
    print(f"\nSample node properties: {nodes.columns.tolist()}")
    
    lines = load_rail_lines()
    print(f"\nSample line properties: {lines.columns.tolist()}")
    
    weather = load_weather_data()
    print(f"\nWeather locations: {weather['location_name'].unique()[:5]}")

