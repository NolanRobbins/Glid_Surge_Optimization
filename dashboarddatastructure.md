# Multi-Modal Route Data Structure

## Overview

This document outlines the data structures required to generate optimized multi-modal routes for a vehicle capable of traveling on both roads and rails. Routes can transition between road segments, rail nodes, and rail lines to optimize container delivery from origin to destination.

## Core Data Structures

### RoutePoint
A point in space that can represent an origin, destination, waypoint, or transition point between transportation modes.

```typescript
interface RoutePoint {
  type: 'port' | 'rail_node' | 'road' | 'intermodal_terminal'
  id: string                    // Unique identifier
  coordinates: [number, number] // [longitude, latitude]
  name?: string                 // Human-readable name/location
  metadata?: {
    address?: string
    facilityType?: string
    capacity?: number
    operatingHours?: string
  }
}
```

### Route Segment
Represents a continuous path segment between two points using a single transportation mode.

```typescript
interface RouteSegment {
  id: string
  segmentType: 'road' | 'rail_line' | 'rail_node_transition'
  startPoint: RoutePoint
  endPoint: RoutePoint
  coordinates: [number, number][] // Detailed path coordinates
  distance: number                // Distance in kilometers
  estimatedTime: number           // Estimated travel time in hours
  cost: number                    // Cost for this segment
  mode: 'road' | 'rail'          // Transportation mode
  metadata?: {
    // Road-specific metadata
    roadType?: 'highway' | 'street' | 'local'
    trafficConditions?: 'low' | 'medium' | 'high'
    tollRoad?: boolean
    speedLimit?: number
    
    // Rail-specific metadata
    railLineId?: string           // Reference to rail line
    fromNodeId?: number           // FRA node ID (if rail)
    toNodeId?: number             // FRA node ID (if rail)
    trackCount?: number
    owner?: string                // Railroad owner
    lineType?: 'main_line' | 'branch_line' | 'freight_line'
    intermodalCapable?: boolean
  }
}
```

### Multi-Modal Route
A complete route from origin to destination that may include multiple segments with different transportation modes.

```typescript
interface MultiModalRoute {
  id: string
  origin: RoutePoint
  destination: RoutePoint
  segments: RouteSegment[]        // Ordered sequence of route segments
  totalDistance: number           // Total distance in kilometers
  totalTime: number               // Total estimated time in hours
  totalCost: number               // Total cost
  transitions: TransitionPoint[]  // Points where mode changes occur
  optimizationScore?: number      // Score for route comparison
  metadata: {
    routeType: 'road_only' | 'rail_only' | 'mixed'
    transitionCount: number       // Number of road-to-rail or rail-to-road transitions
    earliestDeparture?: Date
    latestArrival?: Date
    estimatedDeparture?: Date
    estimatedArrival?: Date
  }
}
```

### Transition Point
A specific location where the vehicle transitions from one transportation mode to another (road ↔ rail).

```typescript
interface TransitionPoint {
  point: RoutePoint
  transitionType: 'road_to_rail' | 'rail_to_road'
  facility?: {
    name: string
    type: 'intermodal_terminal' | 'rail_yard' | 'transfer_station'
    capacity?: number
    waitTime?: number             // Average wait time in minutes
    cost?: number                 // Transition cost
  }
  coordinates: [number, number]
  connectingSegments: {
    incoming: string              // Segment ID before transition
    outgoing: string              // Segment ID after transition
  }
}
```

## Rail Network Data Structures

### Rail Node
A point on the rail network that can serve as a connection point or transition location.

```typescript
interface RailNode {
  id: number                      // OBJECTID or FRA_NODE
  fraNode?: number                // Federal Railroad Administration node ID
  coordinates: [number, number]   // [longitude, latitude]
  type: 'intermodal_terminal' | 'junction' | 'yard' | 'station' | 'waypoint'
  properties: {
    state?: string
    country?: string
    owner?: string                // Railroad owner
    tracks?: number
    yardName?: string
    division?: string
    subdivision?: string
    branch?: string
    intermodalCapable: boolean    // Can vehicle transition here?
    capacity?: number             // Container handling capacity
  }
}
```

### Rail Line Segment
A continuous rail line between two rail nodes.

```typescript
interface RailLineSegment {
  id: number                      // OBJECTID or FRAARCID
  fraArcId?: number
  fromNodeId: number              // FRFRANODE - starting rail node
  toNodeId: number                // TOFRANODE - ending rail node
  coordinates: [number, number][] // Detailed path along rail line
  distance: number                // Distance in kilometers (from KM field)
  properties: {
    owner?: string
    state?: string
    country?: string
    tracks?: number
    lineType: 'main_line' | 'branch_line' | 'passenger_line' | 'freight_line' | 'intermodal_line'
    net?: string
    division?: string
    subdivision?: string
    branch?: string
    strategicNetwork?: boolean    // STRACNET indicator
    passengerService?: boolean    // PASSNGR indicator
  }
}
```

## Route Generation Data Requirements

### Origin & Destination Input

```typescript
interface RouteRequest {
  origin: {
    coordinates: [number, number] // [longitude, latitude]
    address?: string
    type?: 'port' | 'warehouse' | 'address' | 'rail_node'
    id?: string
  }
  destination: {
    coordinates: [number, number] // [longitude, latitude]
    address?: string
    type?: 'port' | 'warehouse' | 'address' | 'rail_node'
    id?: string
  }
  constraints?: {
    maxDistance?: number           // Maximum route distance in km
    maxTime?: number               // Maximum travel time in hours
    maxCost?: number               // Maximum cost
    maxTransitions?: number        // Maximum number of mode transitions
    preferredMode?: 'road' | 'rail' | 'mixed'
    avoidTolls?: boolean
    avoidRailOnly?: boolean        // Prefer roads where possible
  }
  optimization?: {
    priority: 'time' | 'cost' | 'distance' | 'transitions'
    weightTime?: number
    weightCost?: number
    weightDistance?: number
  }
}
```

### Road Network Data
- **Source**: Mapbox Directions API or OpenStreetMap
- **Required Data**:
  - Road segments with coordinates
  - Road type classification (highway, street, local)
  - Distance between points
  - Estimated travel time
  - Traffic conditions (when available)

### Rail Network Data
- **Source**: ArcGIS NTAD North American Rail Network
  - Rail Nodes: `NTAD_North_American_Rail_Network_Nodes`
  - Rail Lines: `NTAD_North_American_Rail_Network_Lines`
- **Required Data**:
  - Rail node coordinates (for transition points)
  - Rail line segments with paths
  - Node connectivity (from/to relationships)
  - Line classification (main, branch, freight)
  - Intermodal terminal identification

### Generated Route Options
Multiple route variations are generated to provide customers with choices:

```typescript
interface RouteOptions {
  requestId: string
  routes: MultiModalRoute[]       // Array of alternative routes, sorted by optimization score
  summary: {
    totalOptions: number
    fastestRoute?: string          // Route ID
    cheapestRoute?: string         // Route ID
    shortestRoute?: string         // Route ID
    mostEfficientRoute?: string    // Route ID (best balance)
  }
  metadata: {
    generatedAt: Date
    dataSources: string[]
    cacheKey?: string
  }
}
```

## Route Generation Process

### 1. Road Segment Generation
- Use Mapbox Directions API to get road coordinates from origin to nearest rail node or destination
- Extract coordinate arrays: `[longitude, latitude][]`
- Calculate distance and estimated time
- Classify road segments by type

### 2. Rail Node Selection
- Query rail network for nodes within proximity of route
- Filter for intermodal-capable nodes (YARDNAME, terminals)
- Identify nodes with road access (for transitions)
- Select optimal transition points based on:
  - Proximity to route
  - Intermodal capability
  - Rail network connectivity

### 3. Rail Line Path Generation
- Query rail network lines connected to selected nodes
- Extract rail line paths (coordinate arrays from geometry.paths)
- Calculate rail segment distances (from KM field or coordinate calculation)
- Identify optimal rail segments to reach destination or next transition

### 4. Mode Transitions
- Identify transition points where road meets rail
- Validate transition feasibility (intermodal terminal exists)
- Calculate transition costs and wait times
- Ensure smooth coordinate connection between segments

### 5. Route Assembly
- Combine road segments, rail node transitions, and rail line segments
- Generate complete coordinate path: `[road_coords] → [rail_node] → [rail_line_coords] → [rail_node] → [road_coords]`
- Calculate totals: distance, time, cost
- Apply optimization scoring

## Optimization Criteria

### Scoring Factors
1. **Total Distance** (kilometers)
2. **Total Time** (hours) - includes:
   - Road travel time
   - Rail travel time
   - Transition wait times
3. **Total Cost** (dollars) - includes:
   - Road segment costs (fuel, tolls, labor)
   - Rail segment costs (usage fees)
   - Transition costs (handling, docking)
4. **Number of Transitions** (fewer is generally better)
5. **Route Reliability** (road conditions, rail availability)

### Route Comparison
```typescript
interface RouteComparison {
  routeId: string
  score: number                  // Weighted optimization score
  metrics: {
    distance: number             // km
    time: number                 // hours
    cost: number                 // dollars
    transitions: number
    reliability: number          // 0-1 score
  }
  advantages: string[]           // e.g., ["Lowest cost", "Fewest transitions"]
  disadvantages: string[]        // e.g., ["Longer distance", "Higher time"]
}
```

## Data Sources & APIs

### Road Data
- **Mapbox Directions API**: Road routing and coordinate generation
  - Endpoint: `https://api.mapbox.com/directions/v5/{profile}/{coordinates}`
  - Returns: Driving route with detailed coordinates

### Rail Network Data
- **Rail Nodes API**: 
  - Endpoint: `https://services.arcgis.com/.../NTAD_North_American_Rail_Network_Nodes/FeatureServer/0/query`
  - Returns: Rail node locations with metadata

- **Rail Lines API**:
  - Endpoint: `https://services.arcgis.com/.../NTAD_North_American_Rail_Network_Lines/FeatureServer/0/query`
  - Returns: Rail line segments with paths and connectivity

### Coordinate System
- All coordinates use **WGS84 (EPSG:4326)**
- Format: `[longitude, latitude]` - **Important: longitude first**
- GeoJSON standard for map rendering

## Example Route Structure

```typescript
{
  id: "route-001",
  origin: {
    type: "port",
    id: "long-beach-port",
    coordinates: [-118.216458, 33.754185],
    name: "Port of Long Beach"
  },
  destination: {
    type: "warehouse",
    id: "fleet-yards-damco",
    coordinates: [-118.2200, 33.8190],
    name: "Fleet Yards Inc./DAMCO DISTRIBUTION INC"
  },
  segments: [
    {
      id: "seg-road-001",
      segmentType: "road",
      startPoint: { /* origin */ },
      endPoint: { /* rail node */ },
      coordinates: [[-118.216458, 33.754185], [-118.2165, 33.7545], ...],
      distance: 2.5,
      estimatedTime: 0.05,
      cost: 15.50,
      mode: "road",
      metadata: { roadType: "highway", trafficConditions: "medium" }
    },
    {
      id: "seg-rail-001",
      segmentType: "rail_line",
      startPoint: { /* rail node */ },
      endPoint: { /* rail node */ },
      coordinates: [[-118.2150, 33.7600], [-118.2145, 33.7650], ...],
      distance: 4.2,
      estimatedTime: 0.15,
      cost: 8.75,
      mode: "rail",
      metadata: {
        railLineId: "FRA-12345",
        fromNodeId: 123,
        toNodeId: 456,
        lineType: "main_line"
      }
    },
    {
      id: "seg-road-002",
      segmentType: "road",
      startPoint: { /* rail node */ },
      endPoint: { /* destination */ },
      coordinates: [[-118.2200, 33.8150], [-118.2200, 33.8190]],
      distance: 0.5,
      estimatedTime: 0.01,
      cost: 3.25,
      mode: "road"
    }
  ],
  totalDistance: 7.2,
  totalTime: 0.21,
  totalCost: 27.50,
  transitions: [
    {
      point: { /* rail node coordinates */ },
      transitionType: "road_to_rail",
      facility: {
        name: "Long Beach Intermodal Terminal",
        type: "intermodal_terminal",
        waitTime: 15,
        cost: 5.00
      }
    }
  ],
  metadata: {
    routeType: "mixed",
    transitionCount: 1
  }
}
```

## Implementation Notes

1. **Coordinate Continuity**: Ensure coordinates from one segment connect smoothly to the next segment at transition points
2. **Distance Calculation**: Use Haversine formula for great circle distance, or sum segment distances for route totals
3. **Time Estimation**: 
   - Roads: Based on distance and average speed (consider traffic)
   - Rails: Based on distance and rail speed limits
   - Add transition wait times
4. **Cost Calculation**: Combine base costs per mode with distance-based pricing
5. **Caching**: Cache rail network data and frequently used route segments for performance
6. **Validation**: Ensure all segments have valid coordinates and connect properly
