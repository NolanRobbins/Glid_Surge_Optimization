'use client'

import { useEffect, useRef, useState, useCallback } from 'react'
import mapboxgl from 'mapbox-gl'
import 'mapbox-gl/dist/mapbox-gl.css'
import { Route } from '@/lib/routeCalculation'

const MAPBOX_TOKEN = process.env.NEXT_PUBLIC_MAPBOX_TOKEN || ''

// Port of Los Angeles/Long Beach coordinates
// Long Beach Port: 33.754185° N, -118.216458° W
const LONG_BEACH_PORT_COORDS: [number, number] = [-118.216458, 33.754185]

interface ContainerPickupMapProps {
  route?: Route | null
  containerResized?: boolean
  initialCenter?: [number, number] // Optional initial map center [lng, lat]
  initialZoom?: number // Optional initial zoom level
}

export default function ContainerPickupMap({
  route,
  containerResized,
  initialCenter = LONG_BEACH_PORT_COORDS,
  initialZoom = 14,
}: ContainerPickupMapProps) {
  const mapContainer = useRef<HTMLDivElement>(null)
  const map = useRef<mapboxgl.Map | null>(null)
  const [mapLoaded, setMapLoaded] = useState(false)

  // Helper function to add elevation heatmap layer
  const addElevationHeatmapLayer = useCallback(() => {
    if (!map.current || !map.current.isStyleLoaded()) return

    // Remove existing layers and sources if present
    if (map.current.getLayer('elevation-heatmap')) {
      map.current.removeLayer('elevation-heatmap')
    }
    if (map.current.getLayer('elevation-colored-hillshade')) {
      map.current.removeLayer('elevation-colored-hillshade')
    }
    if (map.current.getSource('elevation-rgb')) {
      try {
        map.current.removeSource('elevation-rgb')
      } catch {
        // Source might not exist, that's okay
      }
    }
    if (map.current.getSource('elevation-dem-colored')) {
      try {
        map.current.removeSource('elevation-dem-colored')
      } catch {
        // Source might not exist, that's okay
      }
    }

    // Find insertion point - after base layers but before labels
    const layers = map.current.getStyle().layers
    let beforeId: string | undefined
    
    // Try to find first symbol/label layer
    for (const layer of layers) {
      if (layer.type === 'symbol') {
        beforeId = layer.id
        break
      }
    }

    // Hide all base map layers first to show only elevation
    const currentMap = map.current
    if (!currentMap) return
    
    const allLayers = currentMap.getStyle().layers
    allLayers.forEach((layer: mapboxgl.AnyLayer) => {
      if (layer.id !== 'elevation-heatmap' && layer.id !== 'elevation-colored-hillshade') {
        try {
          currentMap.setLayoutProperty(layer.id, 'visibility', 'none')
        } catch {
          // Some layers might not be modifiable
        }
      }
    })

    // Add DEM source for hillshade visualization
    if (!map.current.getSource('elevation-dem-colored')) {
      map.current.addSource('elevation-dem-colored', {
        type: 'raster-dem',
        url: 'mapbox://mapbox.mapbox-terrain-dem-v1',
        tileSize: 512,
      })
    }

    // Create a hillshade layer with green-to-red color scheme
    map.current.addLayer({
      id: 'elevation-colored-hillshade',
      type: 'hillshade',
      source: 'elevation-dem-colored',
      paint: {
        'hillshade-shadow-color': '#00ff00', // Green for shadows (lower elevation)
        'hillshade-highlight-color': '#ff0000', // Red for highlights (higher elevation)  
        'hillshade-accent-color': '#ffff00', // Yellow for mid-range elevation
        'hillshade-exaggeration': 2.0,
        'hillshade-illumination-direction': 315,
        'hillshade-illumination-anchor': 'viewport',
      },
    }, beforeId)
    
    // Add a semi-transparent terrain-rgb layer underneath for additional detail
    map.current.addSource('elevation-rgb', {
      type: 'raster',
      url: 'mapbox://mapbox.terrain-rgb',
      tileSize: 256,
    })

    map.current.addLayer({
      id: 'elevation-heatmap',
      type: 'raster',
      source: 'elevation-rgb',
      paint: {
        'raster-opacity': 0.3,
        'raster-resampling': 'linear',
      },
    }, beforeId)
  }, [])

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

        // Set terrain with exaggeration for elevation view
        map.current.setTerrain({
          source: 'mapbox-dem',
          exaggeration: 2.5,
        })

        // Add elevation heatmap on initial load
        map.current.once('style.load', () => {
          setTimeout(() => {
            if (map.current && map.current.isStyleLoaded() && !map.current.getLayer('elevation-heatmap')) {
              addElevationHeatmapLayer()
            }
          }, 800)
        })

        // Center on initial location
        setTimeout(() => {
          if (!map.current) return
          map.current.flyTo({
            center: initialCenter,
            zoom: initialZoom || 10,
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
  }, [initialCenter, initialZoom, addElevationHeatmapLayer])

  // Handle container resize
  useEffect(() => {
    if (map.current && containerResized !== undefined) {
      setTimeout(() => {
        map.current?.resize()
      }, 100)
    }
  }, [containerResized])

  // Display route on map
  useEffect(() => {
    if (!mapLoaded || !map.current || !route) {
      if (!route && map.current && mapLoaded && initialCenter) {
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

    const routeCoordinates = getRouteCoordinates(route)
    
    const routeGeoJSON: GeoJSON.Feature<GeoJSON.LineString> = {
      type: 'Feature',
      properties: {},
      geometry: {
        type: 'LineString',
        coordinates: routeCoordinates,
      },
    }

    // Wait for style to be loaded before adding sources/layers
    const addRouteLayer = () => {
      if (!map.current) return

      // Remove existing route layer if any
      if (map.current.getLayer('route')) {
        map.current.removeLayer('route')
      }
      if (map.current.getSource('route')) {
        map.current.removeSource('route')
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
            'line-color': '#10b981', // Green for our route
            'line-width': 4,
            'line-opacity': 0.8,
          },
        })

        // Fit map to route bounds
        if (initialCenter) {
          const routeCenterLng = routeCoordinates.reduce((sum, coord) => sum + coord[0], 0) / routeCoordinates.length
          const routeCenterLat = routeCoordinates.reduce((sum, coord) => sum + coord[1], 0) / routeCoordinates.length
          
          const distanceLng = routeCenterLng - initialCenter[0]
          const distanceLat = routeCenterLat - initialCenter[1]
          const distanceDegrees = Math.sqrt(distanceLng * distanceLng + distanceLat * distanceLat)
          
          if (distanceDegrees > 5) {
            map.current.flyTo({
              center: initialCenter,
              zoom: initialZoom || 12,
              duration: 1000,
            })
            return
          }
        }
        
        const bounds = routeCoordinates.reduce((bounds, coord) => {
          return bounds.extend(coord)
        }, new mapboxgl.LngLatBounds(routeCoordinates[0], routeCoordinates[0]))

        map.current.fitBounds(bounds, {
          padding: 50,
          duration: 1000,
        })
      } catch (error) {
        console.error('Error adding route source:', error)
      }
    }

    // Check if style is loaded, if not wait for it
    if (map.current.isStyleLoaded()) {
      addRouteLayer()
    } else {
      map.current.once('style.load', () => {
        addRouteLayer()
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
      }
    }
  }, [mapLoaded, route, initialCenter, initialZoom])

  return (
    <div ref={mapContainer} className="w-full h-full" />
  )
}
