/**
 * Mapbox Directions API service
 * Gets driving routes that follow roads
 */

const MAPBOX_TOKEN = process.env.NEXT_PUBLIC_MAPBOX_TOKEN || ''

export interface DirectionsRoute {
  geometry: {
    coordinates: [number, number][] // Array of [lng, lat] coordinates
    type: string
  }
  distance: number // in meters
  duration: number // in seconds
}

export interface DirectionsResponse {
  routes: DirectionsRoute[]
  code: string
}

/**
 * Get a driving route between two points using Mapbox Directions API
 * Returns a route that follows roads
 * @param origin - [lng, lat] coordinates
 * @param destination - [lng, lat] coordinates
 * @param profile - Routing profile: 'driving' (default), 'driving-traffic' (for real-time traffic), 'walking', 'cycling'
 * @param alternatives - Whether to return alternative routes (default: false)
 */
export async function getDrivingRoute(
  origin: [number, number], // [lng, lat]
  destination: [number, number], // [lng, lat]
  profile: 'driving' | 'driving-traffic' | 'walking' | 'cycling' = 'driving',
  alternatives: boolean = false
): Promise<DirectionsRoute | null> {
  if (!MAPBOX_TOKEN) {
    console.warn('Mapbox token not available, cannot fetch driving route')
    return null
  }

  try {
    // Mapbox Directions API endpoint
    // Format: origin coordinates;destination coordinates
    const coordinates = `${origin[0]},${origin[1]};${destination[0]},${destination[1]}`
    const alternativesParam = alternatives ? '&alternatives=true' : ''
    const url = `https://api.mapbox.com/directions/v5/mapbox/${profile}/${coordinates}?geometries=geojson&access_token=${MAPBOX_TOKEN}${alternativesParam}`
    
    const response = await fetch(url)
    
    if (!response.ok) {
      console.warn('Mapbox Directions API request failed:', response.status, response.statusText)
      return null
    }
    
    const data: DirectionsResponse = await response.json()
    
    if (data.routes && data.routes.length > 0) {
      return data.routes[0] // Return the first (usually best) route
    }
    
    return null
  } catch (error) {
    console.warn('Error fetching driving route from Mapbox:', error)
    return null
  }
}

/**
 * Get alternative routes between two points
 * Returns multiple route options
 */
export async function getAlternativeRoutes(
  origin: [number, number],
  destination: [number, number],
  profile: 'driving' | 'driving-traffic' | 'walking' | 'cycling' = 'driving'
): Promise<DirectionsRoute[]> {
  if (!MAPBOX_TOKEN) {
    console.warn('Mapbox token not available, cannot fetch alternative routes')
    return []
  }

  try {
    const coordinates = `${origin[0]},${origin[1]};${destination[0]},${destination[1]}`
    const url = `https://api.mapbox.com/directions/v5/mapbox/${profile}/${coordinates}?geometries=geojson&alternatives=true&access_token=${MAPBOX_TOKEN}`
    
    const response = await fetch(url)
    
    if (!response.ok) {
      console.warn('Mapbox Directions API request failed:', response.status, response.statusText)
      return []
    }
    
    const data: DirectionsResponse = await response.json()
    
    if (data.routes && data.routes.length > 0) {
      return data.routes // Return all routes
    }
    
    return []
  } catch (error) {
    console.warn('Error fetching alternative routes from Mapbox:', error)
    return []
  }
}
