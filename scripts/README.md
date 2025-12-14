# Update Competitor Routes Script

This script updates competitor routes in the synthetic data to use real street-following routes from Mapbox Directions API. This ensures that competitor routes follow actual roads on the map instead of straight-line paths.

## Prerequisites

- Node.js 18+ (for built-in `fetch` support)
- Mapbox access token (get one from [mapbox.com](https://www.mapbox.com/))

## Usage

### Option 1: Using environment variable (Recommended)

1. Make sure you have a `.env.local` file in the project root with:
   ```
   NEXT_PUBLIC_MAPBOX_TOKEN=your_mapbox_token_here
   ```

2. Run the script:
   ```bash
   npm run update-routes
   ```
   or
   ```bash
   node scripts/updateCompetitorRoutes.js
   ```

### Option 2: Pass token as argument

```bash
node scripts/updateCompetitorRoutes.js your_mapbox_token_here
```

## What it does

1. Loads all routes from `data/syntheticCustomerRoutes.json`
2. For each route with a competitor route, fetches the real road-based route from Mapbox
3. Updates the competitor route coordinates to follow actual streets
4. Saves the updated data back to the JSON file

## Notes

- The script includes a small delay (100ms) between API calls to avoid rate limiting
- If a route fails to fetch, the existing waypoints are kept
- The script will update all competitor routes that exist in the data file

