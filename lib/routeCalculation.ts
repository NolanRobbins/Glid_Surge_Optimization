/**
 * Route calculation utilities
 */

export interface RoutePoint {
  type: 'port' | 'rail_node'
  id: string
  coordinates: [number, number] // [lng, lat]
  name?: string
}

export interface Route {
  origin: RoutePoint
  destination: RoutePoint
  coordinates: [number, number][] // Array of [lng, lat] points
  distance?: number // in kilometers
  waypoints?: RoutePoint[] // Optional waypoints
}

/**
 * Calculate great circle distance between two points using Haversine formula
 * @param point1 [longitude, latitude]
 * @param point2 [longitude, latitude]
 * @returns distance in kilometers
 */
export function calculateDistance(
  point1: [number, number],
  point2: [number, number]
): number {
  const R = 6371 // Earth's radius in kilometers
  const [lng1, lat1] = point1
  const [lng2, lat2] = point2

  const dLat = ((lat2 - lat1) * Math.PI) / 180
  const dLon = ((lng2 - lng1) * Math.PI) / 180

  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos((lat1 * Math.PI) / 180) *
      Math.cos((lat2 * Math.PI) / 180) *
      Math.sin(dLon / 2) *
      Math.sin(dLon / 2)

  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a))
  return R * c
}

/**
 * Generate great circle route between two points
 * Creates intermediate points along the great circle arc
 * @param origin [longitude, latitude]
 * @param destination [longitude, latitude]
 * @param numPoints Number of intermediate points (default: 50)
 * @returns Array of [lng, lat] coordinates
 */
export function calculateGreatCircleRoute(
  origin: [number, number],
  destination: [number, number],
  numPoints: number = 50
): [number, number][] {
  const [lng1, lat1] = origin
  const [lng2, lat2] = destination

  // Convert to radians
  const lat1Rad = (lat1 * Math.PI) / 180
  const lng1Rad = (lng1 * Math.PI) / 180
  const lat2Rad = (lat2 * Math.PI) / 180
  const lng2Rad = (lng2 * Math.PI) / 180

  // Calculate angular distance
  const d = Math.acos(
    Math.sin(lat1Rad) * Math.sin(lat2Rad) +
      Math.cos(lat1Rad) * Math.cos(lat2Rad) * Math.cos(lng2Rad - lng1Rad)
  )

  const coordinates: [number, number][] = []

  // Generate points along the great circle
  for (let i = 0; i <= numPoints; i++) {
    const f = i / numPoints

    const A = Math.sin((1 - f) * d) / Math.sin(d)
    const B = Math.sin(f * d) / Math.sin(d)

    const x =
      A * Math.cos(lat1Rad) * Math.cos(lng1Rad) +
      B * Math.cos(lat2Rad) * Math.cos(lng2Rad)
    const y =
      A * Math.cos(lat1Rad) * Math.sin(lng1Rad) +
      B * Math.cos(lat2Rad) * Math.sin(lng2Rad)
    const z = A * Math.sin(lat1Rad) + B * Math.sin(lat2Rad)

    const lat = Math.atan2(z, Math.sqrt(x * x + y * y))
    const lng = Math.atan2(y, x)

    coordinates.push([(lng * 180) / Math.PI, (lat * 180) / Math.PI])
  }

  return coordinates
}
