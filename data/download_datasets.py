#!/usr/bin/env python3
"""
Script to download datasets for the Port-to-Rail Surge Forecaster competition.
Some datasets require manual download or special authentication.
"""

import os
import requests
import json
from pathlib import Path

# Create data directory
data_dir = Path(__file__).parent
data_dir.mkdir(exist_ok=True)

def download_file(url, filename, headers=None):
    """Download a file from a URL."""
    try:
        print(f"Downloading {filename}...")
        response = requests.get(url, headers=headers, stream=True, timeout=30)
        response.raise_for_status()
        
        filepath = data_dir / filename
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"✓ Successfully downloaded {filename} ({filepath.stat().st_size / 1024:.1f} KB)")
        return True
    except Exception as e:
        print(f"✗ Failed to download {filename}: {e}")
        return False

def download_arcgis_feature_layer(base_url, layer_id, output_filename):
    """Download data from ArcGIS Feature Layer using export endpoint."""
    try:
        # Try the export endpoint
        export_url = f"{base_url}/export?f=geojson"
        print(f"Attempting to download from {export_url}...")
        
        response = requests.get(export_url, timeout=60)
        if response.status_code == 200:
            content_type = response.headers.get('Content-Type', '')
            if 'json' in content_type or 'geojson' in content_type.lower():
                filepath = data_dir / output_filename
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                print(f"✓ Successfully downloaded {output_filename}")
                return True
            else:
                print(f"✗ Unexpected content type: {content_type}")
        else:
            print(f"✗ Export endpoint returned status {response.status_code}")
    except Exception as e:
        print(f"✗ Failed: {e}")
    
    return False

def main():
    print("=" * 60)
    print("Dataset Download Script for Port-to-Rail Competition")
    print("=" * 60)
    print()
    
    # 1. PortWatch - IMF
    print("1. PortWatch – Daily Port Activity (IMF)")
    print("   URL: https://portwatch.imf.org/pages/data-and-methodology")
    print("   ⚠ Manual download required - visit the URL to access data")
    print("   Note: PortWatch may require API access or manual data export")
    print()
    
    # 2. North American Rail Network - Lines
    print("2. North American Rail Network - Lines")
    rail_lines_url = "https://geodata.bts.gov/datasets/usdot::north-american-rail-network-lines/about"
    print(f"   URL: {rail_lines_url}")
    
    # Try alternative download methods
    rail_lines_export = "https://geodata.bts.gov/arcgis/rest/services/USDOT/North_American_Rail_Network/MapServer/0/export"
    if download_arcgis_feature_layer(
        "https://geodata.bts.gov/arcgis/rest/services/USDOT/North_American_Rail_Network/MapServer/0",
        0,
        "north_american_rail_network_lines.geojson"
    ):
        pass
    else:
        print("   ⚠ Direct download failed - may require manual download from:")
        print("   https://geodata.bts.gov/datasets/usdot::north-american-rail-network-lines/about")
    print()
    
    # 3. North American Rail Network - Nodes
    print("3. North American Rail Network - Nodes")
    rail_nodes_url = "https://data-usdot.opendata.arcgis.com/datasets/usdot::north-american-rail-network-nodes/about"
    print(f"   URL: {rail_nodes_url}")
    
    if download_arcgis_feature_layer(
        "https://data-usdot.opendata.arcgis.com/arcgis/rest/services/USDOT/North_American_Rail_Network/MapServer/1",
        1,
        "north_american_rail_network_nodes.geojson"
    ):
        pass
    else:
        print("   ⚠ Direct download failed - may require manual download from:")
        print("   https://data-usdot.opendata.arcgis.com/datasets/usdot::north-american-rail-network-nodes/about")
    print()
    
    # 4. County-to-County Truck Travel Times
    print("4. County-to-County Truck Travel Times (BTS/ATRI)")
    truck_url = "https://rosap.ntl.bts.gov/view/dot/85073"
    print(f"   URL: {truck_url}")
    print("   ⚠ Manual download required - visit ROSAP to download the dataset")
    print("   Note: ROSAP datasets typically require account registration")
    print()
    
    # 5. Logistics Fleet Data (Kaggle)
    print("5. Logistics Fleet Data (Kaggle)")
    kaggle_url = "https://www.kaggle.com/datasets/syednaveed05/logistics-fleet-data/data"
    print(f"   URL: {kaggle_url}")
    print("   ⚠ Requires Kaggle CLI or manual download")
    print("   To download using Kaggle CLI:")
    print("   1. Install: pip install kaggle")
    print("   2. Set up credentials: ~/.kaggle/kaggle.json")
    print("   3. Run: kaggle datasets download -d syednaveed05/logistics-fleet-data -p data/")
    print()
    
    print("=" * 60)
    print("Download Summary")
    print("=" * 60)
    print(f"Data directory: {data_dir}")
    print("\nFiles downloaded:")
    for file in sorted(data_dir.glob("*")):
        if file.is_file() and not file.name.endswith('.py'):
            size_kb = file.stat().st_size / 1024
            print(f"  - {file.name} ({size_kb:.1f} KB)")
    
    print("\n⚠ Note: Some datasets require manual download or special authentication.")
    print("   Please refer to the individual URLs above for download instructions.")

if __name__ == "__main__":
    main()









