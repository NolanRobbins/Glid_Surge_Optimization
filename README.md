# Port Traffic & Container Pickup Planning

A Next.js application for managing container pickup logistics, route planning, and customer route tracking with real-time forecasting and competitive pricing analysis.

## Features

### Customer Routes Page (`/`)
- **Route Management**: View and manage customer routes within Great Plains Industrial Park
- **Route Status Tracking**: Monitor routes by status (scheduled, in_transit, completed)
- **Route Visualization**: Interactive 3D map showing optimized routes and competitor routes
- **Cost Comparison**: Compare our pricing vs competitor pricing with detailed cost breakdowns
- **Route Details**: View distance, weight, pickup/dropoff times, and cost estimates for each route

### Container Pickup Planning (`/newpage`)
- **Multi-Step Form**: Step-by-step container pickup configuration
  1. Pickup Data: Select pickup location (port), destination, container details, and estimated ship arrival
  2. Forecast Timeline: View logistical forecast including port delays, congestion, and available vehicles
  3. Route Options: Generate and select from multiple route options with detailed comparisons

- **Port Traffic Forecasting**:
  - Average ships at port by day/week
  - Port delay calculations based on crane availability and ship volume
  - Vehicle traffic congestion forecasts
  - Estimated time delays and dwelling ship predictions

- **Route Generation**: 
  - Multiple route options with distance, time, and cost calculations
  - Route visualization on interactive 3D map with elevation data
  - Competitive pricing analysis (our service vs competitors)

## Pages

### Home (`/`)
The customer routes page displays all routes within Great Plains Industrial Park, Kansas. Users can:
- View route list with status indicators
- Select routes to view on map
- See both optimized and competitor routes
- Compare costs and savings

### Container Pickup (`/newpage`)
The container pickup planning page allows users to:
- Create new container pickups with detailed configuration
- View forecast data for port conditions and delays
- Generate route options from Port of Long Beach to destinations
- Analyze route costs and time estimates

## Data Sources

- **Synthetic Customer Routes**: Stored in `data/syntheticCustomerRoutes.json`
- **Port Database**: ArcGIS port database for port location data
- **Mapbox Directions API**: Real-time route calculation and optimization
- **Port Infrastructure**: Crane count and port capacity data for delay calculations

## Getting Started

1. Install dependencies:
```bash
npm install
```

2. Set up your Mapbox token (required for map visualization and route calculation):
   - Get a Mapbox access token from [mapbox.com](https://www.mapbox.com/)
   - Create a `.env.local` file in the root directory
   - Add your token: `NEXT_PUBLIC_MAPBOX_TOKEN=your_token_here`

3. Run the development server:
```bash
npm run dev
```

4. Open [http://localhost:3000](http://localhost:3000) in your browser.

## Tech Stack

- **Next.js 14**: React framework with App Router
- **React 18**: UI library
- **TypeScript**: Type-safe development
- **Mapbox GL JS**: Interactive 3D map visualization and routing
- **Three.js**: 3D rendering for map features
- **Tailwind CSS**: Styling and responsive design
- **ArcGIS REST API**: Port database integration

## Pricing Model

- **Our Service**: Base price ($150) + ton-mile calculation ($0.08 - $0.20 per ton-mile)
- **Competitor**: Base price ($200) + ton-mile calculation ($2.30 per ton-mile)

## Key Locations

- **Port of Long Beach**: `[-118.216458, 33.754185]` - Primary pickup location
- **Fleet Yards/DAMCO Distribution**: `[-118.2200, 33.8190]` - Default destination
- **Great Plains Industrial Park**: `[-95.194034, 37.332823]` - Customer routes area

## Project Structure

```
├── app/
│   ├── page.tsx              # Customer routes page
│   ├── newpage/
│   │   └── page.tsx          # Container pickup planning page
│   └── layout.tsx            # Root layout
├── components/
│   ├── ContainerPickupForm.tsx    # Main pickup form with stepper
│   ├── ContainerPickupMap.tsx     # Map for container pickup page
│   ├── CustomerRoutesList.tsx     # Route list component
│   ├── CustomerRoutesMap.tsx      # Map for customer routes page
│   └── PortTrafficMap3D.tsx       # Base 3D map component
├── lib/
│   ├── mapboxDirections.ts        # Route calculation utilities
│   ├── pricing.ts                 # Pricing calculations
│   ├── routeCalculation.ts        # Route distance and utilities
│   ├── syntheticDataService.ts    # Synthetic data loader
│   └── ...
├── data/
│   └── syntheticCustomerRoutes.json  # Customer route data
└── types/
    └── portTraffic.ts             # TypeScript type definitions
```

## Environment Variables

- `NEXT_PUBLIC_MAPBOX_TOKEN`: (Required) Mapbox access token for map visualization and routing

## Building for Production

```bash
npm run build
npm start
```

## Python API (Surge + Routing)

Run the lightweight FastAPI service for surge inference and route computation:

```bash
pip install -r requirements.txt
uvicorn src.api.server:app --reload --port 8000
```

Endpoints:
- `GET /health` / `GET /ready`
- `POST /predict/surge` – tabular feature rows + optional `horizon`
- `POST /routes/compute` – `{ origin:[lon,lat], destination:[lon,lat], mode:"rail"|"road"|"auto" }`
- `GET /routes/{id}` – serve saved GeoJSON (e.g., `long-beach-to-fleet-yards-route`)
- `GET /network/rail/lines` / `GET /network/rail/nodes` – cached local GeoJSON slices

Sample requests:

```bash
# Surge prediction (example feature row)
curl -X POST http://localhost:8000/predict/surge \
  -H "Content-Type: application/json" \
  -d '{
        "horizon": 24,
        "rows": [
          {"date": "2024-01-01", "portname": "Port of Long Beach", "portcalls": 42}
        ]
      }'

# Route compute: Long Beach port -> Fleet Yards/DAMCO
curl -X POST http://localhost:8000/routes/compute \
  -H "Content-Type: application/json" \
  -d '{
        "origin": [-118.216458, 33.754185],
        "destination": [-118.2200, 33.8190],
        "mode": "auto"
      }'
```

## License

This project is private and proprietary.
