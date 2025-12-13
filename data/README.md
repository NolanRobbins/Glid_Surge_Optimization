# Dataset Download Instructions

This directory contains datasets for the Port-to-Rail Surge Forecaster & Utilization Optimizer competition.

## Core Required Datasets

### 1. PortWatch – Daily Port Activity (IMF) (DONE)
**URL:** https://portwatch.imf.org/pages/data-and-methodology

**Download Instructions:**
- Visit the URL above
- Navigate to the data download section
- PortWatch may require:
  - API access (check for API documentation)
  - Manual data export through their web interface
  - Registration/login for data access

**Expected Files:**
- Port activity data (CSV, JSON, or API endpoints)
- Vessel arrivals/departures
- Cargo volumes
- Time-in-port metrics

---

### 2. North American Rail Network - Lines (DONE)
**URL:** https://geodata.bts.gov/datasets/usdot::north-american-rail-network-lines/about

**Download Instructions:**
1. Visit the URL above
2. Click on the dataset page
3. Look for a "Download" or "Export" button
4. Select your preferred format (GeoJSON, Shapefile, CSV, etc.)
5. Save the file to this directory

**Alternative Method (ArcGIS REST API):**
- The dataset is available via ArcGIS REST API, but requires proper authentication
- You can try accessing: `https://geodata.bts.gov/arcgis/rest/services/USDOT/North_American_Rail_Network/MapServer/0`

**Expected Files:**
- `north_american_rail_network_lines.geojson` or `.shp` files
- Rail line segments with attributes

---

### 3. North American Rail Network - Nodes (DONE)
**URL:** https://data-usdot.opendata.arcgis.com/datasets/usdot::north-american-rail-network-nodes/about

**Download Instructions:**
1. Visit the URL above
2. Click on the dataset page
3. Look for a "Download" or "Export" button
4. Select your preferred format (GeoJSON, Shapefile, CSV, etc.)
5. Save the file to this directory

**Expected Files:**
- `north_american_rail_network_nodes.geojson` or `.shp` files
- Rail nodes and intermodal terminals with attributes

---

### 4. County-to-County Truck Travel Times (BTS/ATRI) (DONE)
**URL:** https://data.bts.gov/api/views/ez58-m3b4/rows.csv?accessType=DOWNLOAD

**Download Instructions:**
- **Direct CSV Download:** The dataset can be downloaded directly via CSV export:
  ```bash
  curl -L -o "BTS_ATRI_County_to_County_Truck_Travel_Times.csv" \
    "https://data.bts.gov/api/views/ez58-m3b4/rows.csv?accessType=DOWNLOAD"
  ```
- **Alternative:** Visit https://data.bts.gov/Transportation/BTS-ATRI-Freight-Mobility-Initiative-County-to/ez58-m3b4

**Downloaded File:**
- ✅ `truck_times/BTS_ATRI_County_to_County_Truck_Travel_Times.csv` (232MB, ~3.64M rows)
- Contains: Origin/Destination county pairs, Year (2023), Movements, 25th/50th/75th percentile travel times (minutes)

---

### 5. Logistics Fleet Data (Kaggle) (DONE)
**URL:** https://www.kaggle.com/datasets/syednaveed05/logistics-fleet-data/data

**Download Instructions - Method 1 (Kaggle CLI):**
```bash
# Install Kaggle CLI
pip install kaggle

# Set up credentials (get API token from Kaggle account settings)
# Place kaggle.json in ~/.kaggle/kaggle.json
# Format: {"username":"your_username","key":"your_api_key"}

# Download the dataset
kaggle datasets download -d syednaveed05/logistics-fleet-data -p data/
unzip data/logistics-fleet-data.zip -d data/
```

**Download Instructions - Method 2 (Manual):**
1. Visit the URL above
2. Click "Download" button (requires Kaggle account)
3. Extract the ZIP file to this directory

**Expected Files:**
- CSV files with vehicle-level freight data
- Cost and operational characteristics
- Fleet-level behavior and delay information

---

## Optional Enrichment Datasets

### Global Daily Port Activity (Kaggle)
- Search Kaggle for "Global Daily Port Activity"
- Download using Kaggle CLI or manual download

### AIS Vessel Tracking Data
- **NOAA:** Search NOAA data portals for AIS data
- **Kaggle:** Search for "AIS vessel tracking" datasets

### Weather Data APIs (DONE)
**Source:** Open-Meteo GFS API (https://open-meteo.com/en/docs/gfs-api)

**Downloaded Files:**
- ✅ `weather/weather_hourly_YYYYMMDD.csv` (~3.8MB, 28,704 records)
- ✅ `weather/weather_daily_YYYYMMDD.csv` (~174KB, 1,196 records)

**Coverage:**
- 52 US logistics hubs (15 major ports + 37 rail intermodal terminals)
- 7 days historical + 16 days forecast
- Hourly resolution for 24-72hr delay prediction

**Variables (selected for freight delay prediction):**
- **Hourly:** temperature, apparent_temperature, precipitation, rain, snowfall, snow_depth, weather_code, wind_speed, wind_gusts, visibility, precipitation_probability, freezing_level, cloud_cover, surface_pressure
- **Daily:** weather_code, temp max/min, precipitation_sum, snowfall_sum, wind_speed_max, wind_gusts_max, sunrise/sunset

**To refresh data:**
```bash
cd data/weather/
python3 download_weather_data.py
```

---

## Automated Download Script

A Python script `download_datasets.py` is provided to attempt automated downloads where possible. Run it with:

```bash
cd data/
python3 download_datasets.py
```

**Note:** Many datasets require manual download due to:
- Authentication requirements
- Web interface access
- Terms of service restrictions
- API key requirements

---

## Directory Structure

After downloading, your `/data` directory should contain:

```
data/
├── README.md
├── download_datasets.py
├── portwatch_data/          # PortWatch data files (manual download required)
├── north_american_rail_network_lines.geojson  # (manual download required)
├── north_american_rail_network_nodes.geojson  # (manual download required)
├── truck_travel_times/      # County-to-county travel times (manual download required)
└── logistics_fleet_data/    # Kaggle logistics fleet data (Kaggle CLI or manual download)
```

**Note:** The datasets listed above require manual download as they are not directly accessible via simple HTTP requests. Please follow the download instructions for each dataset.

---

## Notes

- Some datasets are large and may take time to download
- Ensure you have sufficient disk space
- Check dataset licenses and terms of use
- Some datasets may require registration or API keys
- Keep your API keys secure and never commit them to version control

---

## Troubleshooting

**Issue: Cannot access ArcGIS datasets**
- Solution: Try accessing through the web interface and use the download button
- Some datasets may require creating a free ArcGIS account

**Issue: Kaggle download fails**
- Solution: Verify your `kaggle.json` credentials are correct
- Ensure you've accepted the dataset's terms of use on Kaggle

**Issue: ROSAP requires login**
- Solution: Create a free account at ROSAP
- Some datasets may have access restrictions

---

## Contact

For dataset-specific issues:
- **PortWatch:** Contact through https://portwatch.imf.org
- **BTS/ArcGIS:** Email NTAD@dot.gov
- **ROSAP:** Check ROSAP help documentation
- **Kaggle:** Use Kaggle forums for dataset questions

