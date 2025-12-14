#!/usr/bin/env python3
"""
Open-Meteo Weather Data Download Script for Port-to-Rail Surge Forecaster
Downloads comprehensive weather data for major US logistics hubs (ports + rail terminals)

Weather variables selected for freight delay prediction:
- Precipitation (rain, snow, freezing rain)
- Wind (speed, gusts - affects trucks and port cranes)
- Visibility (fog, blowing snow)
- Temperature extremes (rail sun kinks, diesel gelling)
- Weather codes (severe weather identification)

API: https://open-meteo.com/en/docs/gfs-api
"""

import requests
import json
import csv
import time
from datetime import datetime, timedelta
from pathlib import Path

# Major US Logistics Hubs - Ports and Rail Intermodal Terminals
# Format: (name, latitude, longitude, type)
US_LOGISTICS_HUBS = [
    # === MAJOR CONTAINER PORTS ===
    ("Port of Los Angeles", 33.7397, -118.2610, "port"),
    ("Port of Long Beach", 33.7545, -118.2169, "port"),
    ("Port of New York/New Jersey", 40.6699, -74.1468, "port"),
    ("Port of Savannah", 32.0835, -81.0998, "port"),
    ("Port of Houston", 29.7266, -95.2657, "port"),
    ("Port of Seattle", 47.5798, -122.3474, "port"),
    ("Port of Tacoma", 47.2675, -122.4135, "port"),
    ("Port of Oakland", 37.7956, -122.2789, "port"),
    ("Port of Virginia (Norfolk)", 36.8899, -76.3204, "port"),
    ("Port of Charleston", 32.7876, -79.9403, "port"),
    ("Port of Miami", 25.7701, -80.1702, "port"),
    ("Port of New Orleans", 29.9352, -90.0269, "port"),
    ("Port of Baltimore", 39.2598, -76.5789, "port"),
    ("Port of Philadelphia", 39.8937, -75.1380, "port"),
    ("Port of Boston", 42.3522, -71.0410, "port"),
    
    # === MAJOR RAIL INTERMODAL TERMINALS ===
    # West Coast
    ("BNSF Los Angeles (Hobart)", 33.9719, -118.2070, "rail"),
    ("UP Los Angeles (ICTF)", 33.8108, -118.2234, "rail"),
    ("BNSF San Bernardino", 34.0827, -117.3044, "rail"),
    ("UP Oakland", 37.8136, -122.3008, "rail"),
    
    # Southwest
    ("UP Phoenix", 33.4221, -112.0539, "rail"),
    ("BNSF Phoenix", 33.4297, -112.1093, "rail"),
    ("UP El Paso", 31.7892, -106.4235, "rail"),
    
    # Texas/Gulf
    ("BNSF Houston (Pearland)", 29.5584, -95.3201, "rail"),
    ("UP Houston (Englewood)", 29.7453, -95.3150, "rail"),
    ("BNSF Dallas (Alliance)", 32.9877, -97.3091, "rail"),
    ("UP Dallas (Mesquite)", 32.7736, -96.5876, "rail"),
    ("UP San Antonio", 29.3649, -98.5277, "rail"),
    
    # Midwest
    ("BNSF Chicago (Cicero)", 41.8456, -87.7539, "rail"),
    ("UP Chicago (Global 4)", 41.6547, -87.6084, "rail"),
    ("NS Chicago (Landers)", 41.6489, -87.5561, "rail"),
    ("CSX Chicago (59th St)", 41.7881, -87.6298, "rail"),
    ("BNSF Kansas City", 39.1202, -94.6270, "rail"),
    ("UP Kansas City", 39.0847, -94.5869, "rail"),
    ("NS Detroit", 42.3080, -83.0656, "rail"),
    ("BNSF Memphis", 35.0654, -90.0249, "rail"),
    ("NS Columbus", 39.9826, -82.9655, "rail"),
    ("CSX Cincinnati", 39.1155, -84.5064, "rail"),
    ("UP St. Louis", 38.6105, -90.2079, "rail"),
    
    # East Coast
    ("NS Atlanta (Inman)", 33.7823, -84.4130, "rail"),
    ("CSX Atlanta (Fairburn)", 33.5518, -84.5949, "rail"),
    ("NS Savannah", 32.0617, -81.1228, "rail"),
    ("CSX Jacksonville", 30.3539, -81.7587, "rail"),
    ("NS Charlotte", 35.2087, -80.8544, "rail"),
    ("CSX Baltimore", 39.2768, -76.5533, "rail"),
    ("NS Harrisburg", 40.2910, -76.8867, "rail"),
    ("CSX North Bergen NJ", 40.8040, -74.0218, "rail"),
    
    # Pacific Northwest
    ("BNSF Seattle", 47.5572, -122.3352, "rail"),
    ("UP Portland", 45.5593, -122.7274, "rail"),
    
    # Additional Strategic Locations
    ("Denver (Front Range)", 39.7817, -104.8773, "rail"),
    ("Salt Lake City", 40.7765, -111.9303, "rail"),
    ("Minneapolis", 44.9537, -93.0900, "rail"),
    ("Indianapolis", 39.7797, -86.1350, "rail"),
]

# Open-Meteo API Configuration
BASE_URL = "https://api.open-meteo.com/v1/gfs"

# Hourly variables for detailed forecasting (24-72 hour window)
HOURLY_VARS = [
    "temperature_2m",
    "apparent_temperature", 
    "precipitation",
    "rain",
    "snowfall",
    "snow_depth",
    "weather_code",
    "wind_speed_10m",
    "wind_gusts_10m",
    "visibility",
    "precipitation_probability",
    "freezing_level_height",
    "cloud_cover",
    "surface_pressure",
    "is_day",
]

# Daily variables for planning
DAILY_VARS = [
    "weather_code",
    "temperature_2m_max",
    "temperature_2m_min",
    "apparent_temperature_max",
    "apparent_temperature_min",
    "precipitation_sum",
    "rain_sum",
    "snowfall_sum",
    "precipitation_hours",
    "precipitation_probability_max",
    "wind_speed_10m_max",
    "wind_gusts_10m_max",
    "wind_direction_10m_dominant",
    "sunrise",
    "sunset",
]


def fetch_weather_data(lat, lon, location_name, location_type, forecast_days=16, past_days=7):
    """
    Fetch weather data from Open-Meteo API for a single location.
    
    Args:
        lat: Latitude
        lon: Longitude  
        location_name: Name of the location
        location_type: 'port' or 'rail'
        forecast_days: Days of forecast (max 16)
        past_days: Days of historical data (max 92)
    
    Returns:
        dict with hourly and daily data
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(HOURLY_VARS),
        "daily": ",".join(DAILY_VARS),
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "inch",
        "timezone": "America/New_York",
        "forecast_days": forecast_days,
        "past_days": past_days,
    }
    
    try:
        response = requests.get(BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        data['location_name'] = location_name
        data['location_type'] = location_type
        return data
    except requests.exceptions.RequestException as e:
        print(f"  ✗ Error fetching data for {location_name}: {e}")
        return None


def flatten_hourly_data(api_response):
    """Convert API response to flat list of hourly records."""
    if not api_response or 'hourly' not in api_response:
        return []
    
    hourly = api_response['hourly']
    times = hourly.get('time', [])
    
    records = []
    for i, timestamp in enumerate(times):
        record = {
            'timestamp': timestamp,
            'location_name': api_response['location_name'],
            'location_type': api_response['location_type'],
            'latitude': api_response['latitude'],
            'longitude': api_response['longitude'],
            'elevation_m': api_response.get('elevation', None),
        }
        
        for var in HOURLY_VARS:
            if var in hourly and i < len(hourly[var]):
                record[var] = hourly[var][i]
            else:
                record[var] = None
        
        records.append(record)
    
    return records


def flatten_daily_data(api_response):
    """Convert API response to flat list of daily records."""
    if not api_response or 'daily' not in api_response:
        return []
    
    daily = api_response['daily']
    times = daily.get('time', [])
    
    records = []
    for i, date in enumerate(times):
        record = {
            'date': date,
            'location_name': api_response['location_name'],
            'location_type': api_response['location_type'],
            'latitude': api_response['latitude'],
            'longitude': api_response['longitude'],
            'elevation_m': api_response.get('elevation', None),
        }
        
        for var in DAILY_VARS:
            if var in daily and i < len(daily[var]):
                record[var] = daily[var][i]
            else:
                record[var] = None
        
        records.append(record)
    
    return records


def save_to_csv(records, filename, fieldnames):
    """Save records to CSV file."""
    if not records:
        print(f"  ⚠ No records to save for {filename}")
        return
    
    filepath = Path(__file__).parent / filename
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    
    print(f"  ✓ Saved {len(records):,} records to {filename}")


def main():
    print("=" * 70)
    print("Open-Meteo Weather Data Download for Port-to-Rail Surge Forecaster")
    print("=" * 70)
    print(f"Downloading weather for {len(US_LOGISTICS_HUBS)} US logistics hubs...")
    print(f"Forecast: 16 days | Historical: 7 days")
    print(f"Hourly variables: {len(HOURLY_VARS)} | Daily variables: {len(DAILY_VARS)}")
    print()
    
    all_hourly_records = []
    all_daily_records = []
    
    for i, (name, lat, lon, loc_type) in enumerate(US_LOGISTICS_HUBS, 1):
        print(f"[{i:2}/{len(US_LOGISTICS_HUBS)}] Fetching: {name} ({loc_type})...")
        
        data = fetch_weather_data(lat, lon, name, loc_type)
        
        if data:
            hourly_records = flatten_hourly_data(data)
            daily_records = flatten_daily_data(data)
            
            all_hourly_records.extend(hourly_records)
            all_daily_records.extend(daily_records)
            
            print(f"       ✓ {len(hourly_records)} hourly, {len(daily_records)} daily records")
        
        # Be nice to the free API - small delay between requests
        if i < len(US_LOGISTICS_HUBS):
            time.sleep(0.5)
    
    print()
    print("=" * 70)
    print("Saving data to CSV files...")
    print("=" * 70)
    
    # Define field names for CSV
    hourly_fields = [
        'timestamp', 'location_name', 'location_type', 'latitude', 'longitude', 'elevation_m'
    ] + HOURLY_VARS
    
    daily_fields = [
        'date', 'location_name', 'location_type', 'latitude', 'longitude', 'elevation_m'
    ] + DAILY_VARS
    
    # Generate timestamp for file names
    timestamp = datetime.now().strftime("%Y%m%d")
    
    # Save hourly data
    save_to_csv(
        all_hourly_records, 
        f"weather_hourly_{timestamp}.csv",
        hourly_fields
    )
    
    # Save daily data
    save_to_csv(
        all_daily_records,
        f"weather_daily_{timestamp}.csv", 
        daily_fields
    )
    
    # Also save a combined "current + forecast" summary
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Total hourly records: {len(all_hourly_records):,}")
    print(f"Total daily records: {len(all_daily_records):,}")
    print(f"Locations covered: {len(US_LOGISTICS_HUBS)}")
    print(f"  - Ports: {sum(1 for _, _, _, t in US_LOGISTICS_HUBS if t == 'port')}")
    print(f"  - Rail terminals: {sum(1 for _, _, _, t in US_LOGISTICS_HUBS if t == 'rail')}")
    print()
    print("Files saved:")
    print(f"  - weather_hourly_{timestamp}.csv")
    print(f"  - weather_daily_{timestamp}.csv")
    print()
    print("Weather variables included:")
    print("  Hourly:", ", ".join(HOURLY_VARS[:5]), "...")
    print("  Daily:", ", ".join(DAILY_VARS[:5]), "...")
    print()
    print("✓ Weather data download complete!")


if __name__ == "__main__":
    main()







