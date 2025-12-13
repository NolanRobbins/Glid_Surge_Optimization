'use client'

import { useEffect, useRef, useState } from 'react'
import mapboxgl from 'mapbox-gl'
import 'mapbox-gl/dist/mapbox-gl.css'
import { Route } from '@/lib/routeCalculation'

const MAPBOX_TOKEN = process.env.NEXT_PUBLIC_MAPBOX_TOKEN || ''

// Great Plains Industrial Park, Kansas coordinates
const GREAT_PLAINS_INDUSTRIAL_PARK_COORDS: [number, number] = [-95.194034, 37.332823]

interface CustomerRoutesMapProps {
  route?: Route | null
  competitorRoute?: Route | null // Competitor route for comparison
  containerResized?: boolean
  initialCenter?: [number, number] // Optional initial map center [lng, lat]
  initialZoom?: number // Optional initial zoom level
}

export default function CustomerRoutesMap({
  route,
  competitorRoute,
  containerResized,
  initialCenter = GREAT_PLAINS_INDUSTRIAL_PARK_COORDS,
  initialZoom = 14,
}: CustomerRoutesMapProps) {
  const mapContainer = useRef<HTMLDivElement>(null)
  const map = useRef<mapboxgl.Map | null>(null)
  const [mapLoaded, setMapLoaded] = useState(false)

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

  return (
    <div ref={mapContainer} className="w-full h-full" />
  )
}
