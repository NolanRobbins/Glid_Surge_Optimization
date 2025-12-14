---
name: GlidVsMapboxOptimizations
overview: Implement Class I near-infinite avoidance with fallback in routing, expose optimization metrics via the API, and integrate the Next.js UI so Mapbox routes act as competitor baselines while Glid routes represent optimized rail-first paths with visible cost savings, with hazard-safe geometry (no straight-line crossings over water/buildings; last-mile uses existing roadways).
todos:
  - id: backend-class1-avoid
    content: Implement two-pass Class I avoidance routing with fallback and Class I mileage metrics in src/graph/routing.py
    status: pending
  - id: backend-response-fields
    content: Expose used_class1/class1_distance_miles/class1_avoidance_mode in route_adapter + api response
    status: pending
    dependencies:
      - backend-class1-avoid
  - id: backend-hazard-safe-geometry
    content: Ensure hazard-safe geometry: road connectors use existing roadways (Mapbox/OSRM) and rail geometry uses track polylines; never draw straight-line connectors that cut across water/buildings
    status: pending
    dependencies:
      - backend-response-fields
  - id: frontend-glid-client
    content: Add lib/glidApi.ts and call /routes/compute from ContainerPickupForm for optimized route
    status: pending
    dependencies:
      - backend-hazard-safe-geometry
  - id: frontend-savings-ui
    content: Display optimization badges and savings (Glid vs Mapbox competitor) in RouteOptionCard/Cost panels
    status: pending
    dependencies:
      - frontend-glid-client
---

# GlidOptimizedVsCompetitorUI

This plan keeps the current code **unchanged for now**, but lays out the exact next steps to (a) add a Class I “near-infinite” avoidance with fallback in routing, and (b) integrate the UI so **Mapbox = competitor road baseline** and **Glid API = optimized rail-first route**, with savings shown to users.

## Backend: Class I hard-penalty-with-fallback

- Add a two-pass routing helper in [`src/graph/routing.py`](/home/asus/Desktop/Glid_Surge_Optimization/src/graph/routing.py):
- **Pass 1**: compute a path on a graph view that **excludes edges where `is_class1 == True`**.
- **Pass 2 (fallback)**: if no path exists, rerun on the full graph.
- Compute additional metrics:
- `used_class1` (bool)
- `class1_distance_miles` (float)
- `class1_avoidance_mode` (`"avoided"|"fallback_used"`)
- Wire the helper into [`src/graph/route_adapter.py`](/home/asus/Desktop/Glid_Surge_Optimization/src/graph/route_adapter.py) so `/routes/compute` automatically uses Class I avoidance.
- Extend `/routes/compute` response in [`src/api/server.py`](/home/asus/Desktop/Glid_Surge_Optimization/src/api/server.py):
- Add the new fields to `route.properties` and optionally top-level fields for easier frontend consumption.

## Backend: Optimization-friendly response schema

- Ensure `route.properties` contains at minimum:
- `total_distance_miles`, `total_time_hours`
- `rail_distance_miles`, `road_distance_miles`
- `used_class1`, `class1_distance_miles`, `class1_avoidance_mode`
- existing `violations`/`is_valid`
- Confirm the GeoJSON `LineString` shape matches the UI’s expected `[lng, lat]` coordinate arrays.

## Geometry safety (hazard avoidance + last-mile road connectors)

- **No straight-line driving to the rail path**: for any “drive” segment (origin → first rail node and last rail node → destination), generate geometry using an actual road routing engine (e.g. **Mapbox Directions driving**, OSRM, Valhalla), so the vehicle follows existing roadways instead of cutting across buildings/water.
- **Rail geometry must follow rail right-of-way**: avoid rendering node-to-node chords that visually cross hazards; prefer storing/returning rail edge polylines sourced from the rail line dataset used by `builder.py` (or an equivalent track centerline source) and stitch them into the final LineString.
- **Optional guardrail check** (recommended for demos): if you have water/building polygons available (e.g. OSM features), run a quick “segment intersects hazard polygon” sanity check on the produced geometry and fail/fallback rather than displaying an unsafe route.

## Frontend: Mapbox as competitor, Glid as optimized

- Add a small client helper (new file) e.g. [`lib/glidApi.ts`](/home/asus/Desktop/Glid_Surge_Optimization/lib/glidApi.ts):
- Function to POST to `http://localhost:8000/routes/compute` and return `{ coordinates, metrics }`.
- Update [`components/ContainerPickupForm.tsx`](/home/asus/Desktop/Glid_Surge_Optimization/components/ContainerPickupForm.tsx):
- Keep existing Mapbox calls to build **competitor** road routes.
- For the **optimized** route:
- call Glid API once origin/destination are known (or when entering Step 3 / selecting a route option)
- store returned Glid coordinates into the `Route` used by the map
- store Glid metrics for display.
- Display optimization badges/fields (likely via `RouteOptionCard`):
- “Class I avoided” vs “Fallback used Class I”
- rail/road split
- time and distance (Glid) + price deltas vs competitor.
- Update the map usage:
- `ContainerPickupMap` and `CustomerRoutesMap` already render `Route.coordinates`; ensure they receive **Glid coordinates** for the optimized route and **Mapbox coordinates** for competitor.

## Pricing and savings display

- Continue using existing pricing utilities in [`lib/pricing.ts`](/home/asus/Desktop/Glid_Surge_Optimization/lib/pricing.ts).
- Compute two comparable costs:
- **Optimized cost**: based on Glid route distance (prefer `total_distance_miles` converted to ton-mi).
- **Competitor cost**: based on Mapbox road distance.
- Surface savings in:
- [`components/CustomerRoutesList.tsx`](/home/asus/Desktop/Glid_Surge_Optimization/components/CustomerRoutesList.tsx) (list/cost cards)
- [`components/CustomerRoutesMap.tsx`](/home/asus/Desktop/Glid_Surge_Optimization/components/CustomerRoutesMap.tsx) (cost panel)

## Validation / demo workflow

- Verify API returns stable JSON (avoid terminal-wrapped output by using `curl | python -m json.tool` or `curl | jq`).
- Demo story:
- **Optimized** (Glid): rail-first graph path with Class I avoidance.
- **Competitor** (Mapbox): road-following route with realistic street geometry.
- Show: cost deltas + “Class I avoided” badge.
