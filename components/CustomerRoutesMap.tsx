'use client'

import { useEffect, useRef, useState } from 'react'
import mapboxgl from 'mapbox-gl'
import 'mapbox-gl/dist/mapbox-gl.css'
import { Route, calculateDistance } from '@/lib/routeCalculation'
import { calculateTonMileCost, calculateCompetitorCost, PRICING } from '@/lib/pricing'
import { CustomerRoute } from './CustomerRoutesList'

const MAPBOX_TOKEN = process.env.NEXT_PUBLIC_MAPBOX_TOKEN || ''

// Great Plains Industrial Park, Kansas coordinates
const GREAT_PLAINS_INDUSTRIAL_PARK_COORDS: [number, number] = [-95.194034, 37.332823]

interface CustomerRoutesMapProps {
  route?: Route | null
  competitorRoute?: Route | null // Competitor route for comparison
  containerResized?: boolean
  initialCenter?: [number, number] // Optional initial map center [lng, lat]
  initialZoom?: number // Optional initial zoom level
  selectedRoute?: CustomerRoute | null // Selected route for cost calculation
}

export default function CustomerRoutesMap({
  route,
  competitorRoute,
  containerResized,
  initialCenter = GREAT_PLAINS_INDUSTRIAL_PARK_COORDS,
  initialZoom = 14,
  selectedRoute,
}: CustomerRoutesMapProps) {
  const mapContainer = useRef<HTMLDivElement>(null)
  const map = useRef<mapboxgl.Map | null>(null)
  const [mapLoaded, setMapLoaded] = useState(false)
  const [costPanelExpanded, setCostPanelExpanded] = useState(true)

  // Initialize Mapbox map
  useEffect(() => {
    if (!mapContainer.current || map.current) return

    if (!MAPBOX_TOKEN) {
      console.warn('Mapbox token not found. Map visualization requires NEXT_PUBLIC_MAPBOX_TOKEN')
      return
    }

    mapboxgl.accessToken = MAPBOX_TOKEN

    try {
      map.current = new mapboxgl.Map({
        container: mapContainer.current,
        style: 'mapbox://styles/mapbox/dark-v11',
        center: initialCenter,
        zoom: initialZoom,
        pitch: 60,
        bearing: 0,
        antialias: true,
      })

      map.current.on('load', () => {
        setMapLoaded(true)

        if (!map.current) return

        // Add terrain source
        if (!map.current.getSource('mapbox-dem')) {
          map.current.addSource('mapbox-dem', {
            type: 'raster-dem',
            url: 'mapbox://mapbox.mapbox-terrain-dem-v1',
            tileSize: 512,
            maxzoom: 14,
          })
        }

        // Set terrain
        map.current.setTerrain({
          source: 'mapbox-dem',
          exaggeration: 1.5,
        })

        // Add 3D buildings
        const layers = map.current.getStyle().layers
        let firstSymbolId = ''
        for (const layer of layers) {
          if (layer.type === 'symbol') {
            firstSymbolId = layer.id
            break
          }
        }

        if (!map.current.getLayer('3d-buildings')) {
          map.current.addLayer({
            id: '3d-buildings',
            source: 'composite',
            'source-layer': 'building',
            filter: ['==', 'extrude', 'true'],
            type: 'fill-extrusion',
            minzoom: 14,
            paint: {
              'fill-extrusion-color': '#aaa',
              'fill-extrusion-height': [
                'interpolate',
                ['linear'],
                ['zoom'],
                15, 0,
                15.05, ['get', 'height'],
              ],
              'fill-extrusion-base': [
                'interpolate',
                ['linear'],
                ['zoom'],
                15, 0,
                15.05, ['get', 'min_height'],
              ],
              'fill-extrusion-opacity': 0.6,
            },
          }, firstSymbolId)
        }

        // Center on initial location
        setTimeout(() => {
          if (!map.current) return
          map.current.flyTo({
            center: initialCenter,
            zoom: initialZoom,
            pitch: 60,
            duration: 2000,
          })
        }, 500)
      })

      map.current.on('error', (e) => {
        console.error('Mapbox error:', e)
      })
    } catch (error) {
      console.error('Error initializing Mapbox map:', error)
    }

    return () => {
      if (map.current) {
        map.current.remove()
        map.current = null
      }
    }
  }, [initialCenter, initialZoom])

  // Handle container resize
  useEffect(() => {
    if (map.current && containerResized !== undefined) {
      setTimeout(() => {
        map.current?.resize()
      }, 100)
    }
  }, [containerResized])

  // Display routes on map (both our optimized route and competitor route)
  useEffect(() => {
    if (!mapLoaded || !map.current) {
      if (!route && !competitorRoute && map.current && mapLoaded && initialCenter) {
        map.current.flyTo({
          center: initialCenter,
          zoom: initialZoom || 12,
          duration: 1000,
        })
      }
      return
    }

    // Helper function to get route coordinates
    const getRouteCoordinates = (r: Route): [number, number][] => {
      if (r.coordinates && r.coordinates.length > 0) {
        return r.coordinates.map(coord => coord as [number, number])
      } else if (r.waypoints && r.waypoints.length > 0) {
        return r.waypoints.map(wp => wp.coordinates as [number, number])
      } else {
        return [
          r.origin.coordinates as [number, number],
          r.destination.coordinates as [number, number]
        ]
      }
    }

    // Wait for style to be loaded before adding sources/layers
    const addRouteLayers = () => {
      if (!map.current) return

      // Remove existing route layers if any
      if (map.current.getLayer('route')) {
        map.current.removeLayer('route')
      }
      if (map.current.getSource('route')) {
        map.current.removeSource('route')
      }
      if (map.current.getLayer('competitor-route')) {
        map.current.removeLayer('competitor-route')
      }
      if (map.current.getSource('competitor-route')) {
        map.current.removeSource('competitor-route')
      }

      let allCoordinates: [number, number][] = []
      let bounds: mapboxgl.LngLatBounds | null = null

      // Add our optimized route (green)
      if (route) {
        const routeCoordinates = getRouteCoordinates(route)
        allCoordinates = [...routeCoordinates]
        
        const routeGeoJSON: GeoJSON.Feature<GeoJSON.LineString> = {
          type: 'Feature',
          properties: {},
          geometry: {
            type: 'LineString',
            coordinates: routeCoordinates,
          },
        }

        try {
          map.current.addSource('route', {
            type: 'geojson',
            data: routeGeoJSON,
          })

          map.current.addLayer({
            id: 'route',
            type: 'line',
            source: 'route',
            layout: {
              'line-join': 'round',
              'line-cap': 'round',
            },
            paint: {
              'line-color': '#10b981', // Green for our optimized route
              'line-width': 4,
              'line-opacity': 0.8,
            },
          })

          bounds = routeCoordinates.reduce((b, coord) => {
            return b.extend(coord)
          }, new mapboxgl.LngLatBounds(routeCoordinates[0], routeCoordinates[0]))
        } catch (error) {
          console.error('Error adding route source:', error)
        }
      }

      // Add competitor route (red)
      if (competitorRoute) {
        const competitorCoordinates = getRouteCoordinates(competitorRoute)
        allCoordinates = [...allCoordinates, ...competitorCoordinates]
        
        const competitorGeoJSON: GeoJSON.Feature<GeoJSON.LineString> = {
          type: 'Feature',
          properties: {},
          geometry: {
            type: 'LineString',
            coordinates: competitorCoordinates,
          },
        }

        try {
          map.current.addSource('competitor-route', {
            type: 'geojson',
            data: competitorGeoJSON,
          })

          map.current.addLayer({
            id: 'competitor-route',
            type: 'line',
            source: 'competitor-route',
            layout: {
              'line-join': 'round',
              'line-cap': 'round',
            },
            paint: {
              'line-color': '#ef4444', // Red for competitor route
              'line-width': 4,
              'line-opacity': 0.6,
              'line-dasharray': [2, 2], // Dashed line to distinguish from our route
            },
          })

          // Extend bounds to include competitor route
          if (bounds) {
            competitorCoordinates.forEach(coord => bounds!.extend(coord))
          } else {
            bounds = competitorCoordinates.reduce((b, coord) => {
              return b.extend(coord)
            }, new mapboxgl.LngLatBounds(competitorCoordinates[0], competitorCoordinates[0]))
          }
        } catch (error) {
          console.error('Error adding competitor route source:', error)
        }
      }

      // Fit map to route bounds
      if (bounds && allCoordinates.length > 0) {
        // If we have initialCenter, check if route is far from it
        if (initialCenter) {
          const routeCenterLng = allCoordinates.reduce((sum, coord) => sum + coord[0], 0) / allCoordinates.length
          const routeCenterLat = allCoordinates.reduce((sum, coord) => sum + coord[1], 0) / allCoordinates.length
          
          const distanceLng = routeCenterLng - initialCenter[0]
          const distanceLat = routeCenterLat - initialCenter[1]
          const distanceDegrees = Math.sqrt(distanceLng * distanceLng + distanceLat * distanceLat)
          
          // If route center is more than ~5 degrees away, use initialCenter instead
          if (distanceDegrees > 5) {
            console.log('Route is far from initialCenter, using initialCenter instead')
            map.current.flyTo({
              center: initialCenter,
              zoom: initialZoom || 12,
              duration: 1000,
            })
            return
          }
        }
        
        map.current.fitBounds(bounds, {
          padding: 50,
          duration: 1000,
        })
      }
    }

    // Check if style is loaded, if not wait for it
    if (map.current.isStyleLoaded()) {
      addRouteLayers()
    } else {
      map.current.once('style.load', () => {
        addRouteLayers()
      })
    }

    return () => {
      if (map.current) {
        if (map.current.getLayer('route')) {
          map.current.removeLayer('route')
        }
        if (map.current.getSource('route')) {
          map.current.removeSource('route')
        }
        if (map.current.getLayer('competitor-route')) {
          map.current.removeLayer('competitor-route')
        }
        if (map.current.getSource('competitor-route')) {
          map.current.removeSource('competitor-route')
        }
      }
    }
  }, [mapLoaded, route, competitorRoute, initialCenter, initialZoom])

  // Calculate costs for the selected route
  const calculateRouteCosts = (route: CustomerRoute | null | undefined) => {
    if (!route) return null
    
    let ourDistanceKm = 0
    let competitorDistanceKm = 0
    
    // Get our route distance
    if (route.route?.distance) {
      ourDistanceKm = route.route.distance
    } else if (route.route?.origin && route.route?.destination) {
      ourDistanceKm = calculateDistance(route.route.origin.coordinates, route.route.destination.coordinates)
    }
    
    // Get competitor route distance
    if (route.competitorRoute?.distance) {
      competitorDistanceKm = route.competitorRoute.distance
    } else if (route.competitorRoute?.coordinates && route.competitorRoute.coordinates.length > 0) {
      let totalDistance = 0
      for (let i = 0; i < route.competitorRoute.coordinates.length - 1; i++) {
        totalDistance += calculateDistance(
          route.competitorRoute.coordinates[i],
          route.competitorRoute.coordinates[i + 1]
        )
      }
      competitorDistanceKm = totalDistance
    } else {
      competitorDistanceKm = ourDistanceKm
    }
    
    // Convert to miles
    const ourDistanceMiles = ourDistanceKm * 0.621371
    const competitorDistanceMiles = competitorDistanceKm * 0.621371
    
    // Use container weight or default to 15 tons
    const weightTons = route.containerWeight || 15
    
    if (ourDistanceMiles <= 0) {
      return null
    }
    
    // Calculate costs
    const ourCost = calculateTonMileCost(ourDistanceMiles, weightTons, PRICING.STANDARD)
    const competitorCost = calculateCompetitorCost(competitorDistanceMiles, weightTons)
    const savings = competitorCost - ourCost
    const savingsPercent = ((savings / competitorCost) * 100).toFixed(0)
    
    return {
      ourCost,
      competitorCost,
      savings,
      savingsPercent,
      distanceMiles: ourDistanceMiles,
      competitorDistanceMiles,
      weightTons,
    }
  }

  const costs = calculateRouteCosts(selectedRoute)

  return (
    <div className="w-full h-full relative">
      <div ref={mapContainer} className="w-full h-full" />
      
      {/* Cost Estimate Panel - Bottom Right */}
      {costs && (
        <div className="absolute bottom-4 right-4 z-[1000] w-64">
          <div className="bg-white/75 backdrop-blur-[50px] rounded-lg border border-gray-200/50 shadow-lg">
            {/* Panel Header */}
            <button
              onClick={() => setCostPanelExpanded(!costPanelExpanded)}
              className="w-full px-4 py-3 flex items-center justify-between text-left hover:bg-white/20 transition-colors rounded-t-lg"
            >
              <h3 className="text-sm font-semibold text-gray-900">Cost Estimate</h3>
              <svg
                className={`w-4 h-4 text-gray-500 transition-transform ${costPanelExpanded ? 'rotate-180' : ''}`}
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </button>
            
            {/* Panel Content */}
            {costPanelExpanded && (
              <div className="px-4 pb-4 space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex-1">
                    <span className="text-xs font-medium text-gray-900">Our Cost</span>
                    <div className="text-xs text-gray-700 mt-1">
                      {costs.distanceMiles.toFixed(1)} mi × {costs.weightTons.toFixed(1)} tons × ${PRICING.STANDARD.toFixed(2)}/ton-mi
                    </div>
                  </div>
                  <div className="text-lg font-bold text-green-600 ml-4">
                    ${costs.ourCost.toFixed(2)}
                  </div>
                </div>
                <div className="flex items-center justify-between">
                  <div className="flex-1">
                    <span className="text-xs font-medium text-gray-900">Competitor Cost</span>
                    <div className="text-xs text-gray-700 mt-1">
                      {costs.competitorDistanceMiles.toFixed(1)} mi × {costs.weightTons.toFixed(1)} tons × ${PRICING.COMPETITOR.toFixed(2)}/ton-mi
                    </div>
                  </div>
                  <div className="text-lg font-bold text-red-800 ml-4 line-through">
                    ${costs.competitorCost.toFixed(2)}
                  </div>
                </div>
                <div className="flex items-center justify-between">
                  <div className="flex-1">
                    <span className="text-xs font-medium text-gray-900">You Save</span>
                    <div className="text-xs text-gray-700 mt-1">
                      {costs.savingsPercent}% less than competitors
                    </div>
                  </div>
                  <div className="text-lg font-bold text-gray-900 ml-4">
                    ${costs.savings.toFixed(2)}
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
