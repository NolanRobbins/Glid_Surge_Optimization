'use client'

import { useEffect, useRef, useState, useCallback } from 'react'
import mapboxgl from 'mapbox-gl'
import 'mapbox-gl/dist/mapbox-gl.css'
import * as THREE from 'three'
import { Route } from '@/lib/routeCalculation'
import { fetchRailNetwork, railLinesToGeoJSON } from '@/api/railNetworkService'
import { fetchRailNodes, railNodesToGeoJSON } from '@/api/railNodesService'

const MAPBOX_TOKEN = process.env.NEXT_PUBLIC_MAPBOX_TOKEN || ''

// Port of Los Angeles/Long Beach coordinates
// Long Beach Port: 33.754185° N, -118.216458° W
const LONG_BEACH_PORT_COORDS: [number, number] = [-118.216458, 33.754185]


interface PortTrafficMap3DProps {
  route?: Route | null
  competitorRoute?: Route | null // Competitor route for comparison
  onRouteWaypointAdd?: (coordinates: [number, number], index: number) => void
  onRouteWaypointMove?: (coordinates: [number, number], index: number) => void
  editableWaypoints?: Array<{ id: string; coordinates: [number, number] }>
  containerResized?: boolean
  onRailNodeClick?: (nodeId: string) => void
  showElevationHeatmap?: boolean // Enable elevation heat map visualization
  initialCenter?: [number, number] // Optional initial map center [lng, lat]
  initialZoom?: number // Optional initial zoom level
}

export default function PortTrafficMap3D({
  route,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  competitorRoute: _competitorRoute,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  onRouteWaypointAdd: _onRouteWaypointAdd,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  onRouteWaypointMove: _onRouteWaypointMove,
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  editableWaypoints: _editableWaypoints = [],
  containerResized,
  onRailNodeClick,
  showElevationHeatmap = false,
  initialCenter,
  initialZoom,
}: PortTrafficMap3DProps) {
  const mapContainer = useRef<HTMLDivElement>(null)
  const map = useRef<mapboxgl.Map | null>(null)
  const [mapLoaded, setMapLoaded] = useState(false)
  const [railLinesLoaded, setRailLinesLoaded] = useState(false)
  const [railNodesLoaded, setRailNodesLoaded] = useState(false)
  const isAnimatingRef = useRef(false)
  const trainObjectRef = useRef<THREE.Group | null>(null)
  const currentPathRef = useRef<[number, number][] | null>(null)
  const animationProgressRef = useRef<number>(0)
  const customLayerRef = useRef<mapboxgl.CustomLayerInterface | null>(null)
  const sceneRef = useRef<THREE.Scene | null>(null)
  const cameraRef = useRef<THREE.Camera | null>(null)
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null)
  const modelTransformRef = useRef<{
    translateX: number
    translateY: number
    translateZ: number
    rotateX: number
    rotateY: number
    rotateZ: number
    scale: number
  } | null>(null)

  // Initialize Mapbox map with 3D features
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
        style: 'mapbox://styles/mapbox/dark-v11', // Dark themed map
        center: initialCenter || LONG_BEACH_PORT_COORDS, // Use provided center or default to Long Beach Port
        zoom: initialZoom || 15, // Use provided zoom or default to 15
        pitch: 60, // Tilt for 3D view
        bearing: 0,
        antialias: true,
      })

      map.current.on('load', () => {
        console.log('Mapbox 3D map load event fired')
        setMapLoaded(true)
        console.log('Mapbox 3D map loaded, mapLoaded state set to true')

        if (!map.current) return

        // Add terrain source first
        if (!map.current.getSource('mapbox-dem')) {
          map.current.addSource('mapbox-dem', {
            type: 'raster-dem',
            url: 'mapbox://mapbox.mapbox-terrain-dem-v1',
            tileSize: 512,
            maxzoom: 14,
          })
        }

        // Set terrain with exaggeration (increased for elevation view)
        map.current.setTerrain({
          source: 'mapbox-dem',
          exaggeration: showElevationHeatmap ? 2.5 : 1.5,
        })


        // Add elevation heatmap on initial load if enabled
        if (showElevationHeatmap) {
          map.current.once('style.load', () => {
            setTimeout(() => {
              if (map.current && map.current.isStyleLoaded() && !map.current.getLayer('elevation-heatmap')) {
                addElevationHeatmapLayer()
              }
            }, 800)
          })
        }

        // Add 3D buildings layer (skip in elevation-only mode)
        if (showElevationHeatmap) {
          // Skip 3D buildings in elevation mode
        } else {
          const layers = map.current.getStyle().layers

        // Find the first symbol layer to add 3D buildings before it
        let firstSymbolId = ''
        for (const layer of layers) {
          if (layer.type === 'symbol') {
            firstSymbolId = layer.id
            break
          }
        }

          // Check if 3D buildings layer already exists
          if (!map.current.getLayer('3d-buildings')) {
            // Add 3D buildings extrusion
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
                  15,
                  0,
                  15.05,
                  ['get', 'height'],
                ],
                'fill-extrusion-base': [
                  'interpolate',
                  ['linear'],
                  ['zoom'],
                  15,
                  0,
                  15.05,
                  ['get', 'min_height'],
                ],
                'fill-extrusion-opacity': 0.6,
              },
            }, firstSymbolId)
          }
        }

        // Make roads stand out with bright colors (skip in elevation-only mode)
        if (!showElevationHeatmap) {
          const layers = map.current.getStyle().layers
          layers.forEach((layer: mapboxgl.AnyLayer) => {
          const layerId = layer.id
          if (layerId.includes('road') || layerId.includes('street') || layerId.includes('highway')) {
            try {
              map.current!.setLayoutProperty(layerId, 'visibility', 'visible')
              
              // Color all roads blue
              if (layer.type === 'line') {
                map.current!.setPaintProperty(layerId, 'line-color', '#3b82f6') // Blue for all roads
                // Keep existing width or set a reasonable default
                try {
                  const currentWidth = map.current!.getPaintProperty(layerId, 'line-width')
                  if (!currentWidth) {
                    map.current!.setPaintProperty(layerId, 'line-width', [
                      'interpolate',
                      ['linear'],
                      ['zoom'],
                      10, 1,
                      15, 2,
                      20, 3
                    ])
                  }
                } catch {
                  // Width might not be settable, that's okay
                }
              }
            } catch (error) {
              // Some layers might not be modifiable, ignore errors
              console.log(`Could not style layer ${layerId}:`, error)
            }
          }
          })
        }

        // Ensure we're centered on initial center or Long Beach port (unless elevation mode, then use wider view)
        // Use a small delay to ensure map is fully initialized
        setTimeout(() => {
          if (!map.current) return
          
          if (!showElevationHeatmap) {
            map.current.flyTo({
              center: initialCenter || LONG_BEACH_PORT_COORDS,
              zoom: initialZoom || 15,
              pitch: 60,
              duration: 2000,
            })
          } else {
            // For elevation view, start with a wider view to see terrain features
            map.current.flyTo({
              center: initialCenter || LONG_BEACH_PORT_COORDS,
              zoom: initialZoom || 10,
              pitch: 60,
              duration: 2000,
            })
          }
        }, 500)

        // Load rail network and nodes after map is ready (skip if elevation-only mode)
        if (!showElevationHeatmap) {
          // Use requestAnimationFrame to ensure map is fully rendered
          requestAnimationFrame(() => {
            setTimeout(() => {
              console.log('Attempting to load rail network and nodes')
              if (map.current) {
                // Call load functions directly - they check their own state
                // Load rail network first, which will call loadRailNodes when done
                loadRailNetwork()
                // Also try loading nodes directly (in case network load doesn't trigger it)
                // But only if nodes aren't already loaded
                if (!railNodesLoaded) {
                  loadRailNodes()
                }
              } else {
                console.error('Map instance not available when trying to load rails')
              }
            }, 500)
          })
        }
      })

      map.current.on('error', (e) => {
        console.error('Mapbox error:', e)
      })

      // Add click handler to place train (disabled in elevation-only mode)
      if (!showElevationHeatmap) {
        map.current.on('click', (e) => {
          // Only place train if not clicking on rail nodes
          // Check if layers exist before querying
          const layersToCheck: string[] = []
          if (map.current!.getLayer('rail-nodes')) {
            layersToCheck.push('rail-nodes')
          }
          
          let shouldPlaceTrain = true
          if (layersToCheck.length > 0) {
            try {
              const features = map.current!.queryRenderedFeatures(e.point, {
                layers: layersToCheck
              })
              if (features.length > 0) {
                shouldPlaceTrain = false
              }
            } catch (error) {
              // If query fails, just place the train anyway
              console.warn('Error querying features:', error)
            }
          }
          
          if (shouldPlaceTrain) {
            // Place train at clicked location
            placeTrainAtCoordinates(e.lngLat.lng, e.lngLat.lat)
          }
        })
      }

    } catch (error) {
      console.error('Error initializing Mapbox map:', error)
    }

    return () => {
      if (map.current) {
        map.current.remove()
        map.current = null
      }
    }
  }, [initialCenter, initialZoom]) // Include initialCenter and initialZoom in dependencies

  // Ensure map centers on initialCenter when it loads (if provided)
  useEffect(() => {
    if (map.current && mapLoaded && initialCenter && !route) {
      // Only center if no route is set, to avoid conflicts
      map.current.flyTo({
        center: initialCenter,
        zoom: initialZoom || 12,
        duration: 1000,
      })
    }
  }, [mapLoaded, initialCenter, initialZoom, route])

  // Handle container resize
  useEffect(() => {
    if (map.current && containerResized !== undefined) {
      setTimeout(() => {
        map.current?.resize()
      }, 100)
    }
  }, [containerResized])


  // Load rail network - same as main map
  const loadRailNetwork = async () => {
    if (!map.current || railLinesLoaded) return

    try {
      console.log('Loading rail network...')
      setRailLinesLoaded(true)

      // Load all data (no bounds) - use cache if available
      console.log('Fetching rail network (all data, using cache if available)...')
      const railFeatures = await fetchRailNetwork(undefined, 50000, true)

      if (railFeatures.length === 0) {
        console.log('No rail features found')
        setRailLinesLoaded(false)
        return
      }

      // Convert to GeoJSON
      console.log(`Converting ${railFeatures.length} rail features to GeoJSON...`)
      const geoJSON = await railLinesToGeoJSON(railFeatures, (processed, total) => {
        if (processed % 5000 === 0) {
          console.log(`GeoJSON conversion progress: ${processed}/${total}`)
        }
      })
      console.log('GeoJSON conversion complete, features:', geoJSON.features.length)

      // Remove existing rail layers if they exist
      if (map.current.getLayer('rail-lines-glow')) {
        map.current.removeLayer('rail-lines-glow')
      }
      if (map.current.getLayer('rail-lines')) {
        map.current.removeLayer('rail-lines')
      }
      if (map.current.getLayer('rail-lines-outline')) {
        map.current.removeLayer('rail-lines-outline')
      }
      if (map.current.getSource('rail-lines')) {
        map.current.removeSource('rail-lines')
      }

      // Add rail lines source - use setData to avoid stack overflow
      try {
        map.current.addSource('rail-lines', {
          type: 'geojson',
          data: {
            type: 'FeatureCollection',
            features: [], // Start with empty features
          },
        })

        // Set data after source is created
        const source = map.current.getSource('rail-lines') as mapboxgl.GeoJSONSource
        source.setData(geoJSON)
      } catch (error) {
        console.error('Error adding rail lines source:', error)
        // Fallback: try with smaller dataset
        if (railFeatures.length > 10000) {
          console.log('Trying with first 10,000 segments...')
          const limitedFeatures = railFeatures.slice(0, 10000)
          const limitedGeoJSON = await railLinesToGeoJSON(limitedFeatures)
          map.current.addSource('rail-lines', {
            type: 'geojson',
            data: limitedGeoJSON,
          })
        } else {
          throw error
        }
      }

      // Find a good layer to insert rail lines before
      const layers = map.current.getStyle().layers
      const insertBeforeLayer = layers.find((layer: mapboxgl.AnyLayer) =>
        layer.type === 'symbol' || layer.id.includes('label')
      )?.id

      // Add rail lines outline first (so main line appears on top) - thicker for visibility
      map.current.addLayer({
        id: 'rail-lines-outline',
        type: 'line',
        source: 'rail-lines',
        layout: {
          'line-join': 'round',
          'line-cap': 'round',
        },
        paint: {
          'line-color': '#000000',
          'line-width': [
            'interpolate',
            ['linear'],
            ['zoom'],
            3, 3,
            5, 4,
            10, 6,
            15, 8,
          ],
          'line-opacity': 0.6,
        },
      }, insertBeforeLayer)

      // Add rail lines layer on top of outline - color by segment type (same as main map)
      map.current.addLayer({
        id: 'rail-lines',
        type: 'line',
        source: 'rail-lines',
        layout: {
          'line-join': 'round',
          'line-cap': 'round',
        },
        paint: {
          'line-color': [
            'case',
            ['==', ['get', 'segmentType'], 'main_line'],
            '#ff0000', // Bright red for main lines
            ['==', ['get', 'segmentType'], 'intermodal_line'],
            '#00ff00', // Bright green for intermodal lines
            ['==', ['get', 'segmentType'], 'passenger_line'],
            '#0000ff', // Blue for passenger lines
            ['==', ['get', 'segmentType'], 'branch_line'],
            '#ff8800', // Orange for branch lines
            ['==', ['get', 'net'], 'S'],
            '#ff0000', // Fallback: red for main network
            ['==', ['get', 'net'], 'O'],
            '#ff8800', // Fallback: orange for other
            '#ec4899', // Pink for unknown/freight
          ],
          'line-width': [
            'interpolate',
            ['linear'],
            ['zoom'],
            3, 2,
            5, 3,
            10, 5,
            15, 7,
          ],
          'line-opacity': 1.0,
        },
      }, insertBeforeLayer)

      // Add a glow effect for better visibility
      map.current.addLayer({
        id: 'rail-lines-glow',
        type: 'line',
        source: 'rail-lines',
        layout: {
          'line-join': 'round',
          'line-cap': 'round',
        },
        paint: {
          'line-color': [
            'case',
            ['==', ['get', 'segmentType'], 'main_line'],
            '#ff0000',
            ['==', ['get', 'segmentType'], 'intermodal_line'],
            '#00ff00',
            ['==', ['get', 'segmentType'], 'passenger_line'],
            '#0000ff',
            ['==', ['get', 'segmentType'], 'branch_line'],
            '#ff8800',
            '#ec4899',
          ],
          'line-width': [
            'interpolate',
            ['linear'],
            ['zoom'],
            3, 4,
            5, 6,
            10, 8,
            15, 10,
          ],
          'line-opacity': 0.3,
          'line-blur': 2,
        },
      }, insertBeforeLayer)

      console.log(`Successfully added rail network layers with ${geoJSON.features.length} features`)
      setRailLinesLoaded(true)
    } catch (error) {
      console.error('Error loading rail network:', error)
      setRailLinesLoaded(false)
    }
  }

  // Load rail nodes - same as main map
  const loadRailNodes = async () => {
    if (!map.current || railNodesLoaded) return

    try {
      console.log('Loading rail nodes...')
      setRailNodesLoaded(true)

      // Wait for style to be loaded before removing/adding sources
      if (!map.current.isStyleLoaded()) {
        map.current.once('style.load', () => {
          setTimeout(() => {
            loadRailNodes()
          }, 100)
        })
        setRailNodesLoaded(false)
        return
      }

      // Remove existing rail nodes layer if it exists
      if (map.current.getLayer('rail-nodes')) {
        map.current.removeLayer('rail-nodes')
      }
      if (map.current.getSource('rail-nodes')) {
        try {
          map.current.removeSource('rail-nodes')
        } catch {
          // Source might not exist
        }
      }

      console.log('Fetching rail nodes for entire North America (no bounds filter)...')
      const railNodes = await fetchRailNodes(200000) // Increased limit

      if (railNodes.length === 0) {
        console.log('No rail nodes found')
        setRailNodesLoaded(false)
        return
      }

      // Convert to GeoJSON
      const geoJSON = railNodesToGeoJSON(railNodes)

      // Double-check source doesn't exist before adding
      if (map.current.getSource('rail-nodes')) {
        console.warn('Rail-nodes source still exists, removing it...')
        try {
          map.current.removeSource('rail-nodes')
        } catch (e) {
          console.error('Error removing rail-nodes source before adding:', e)
        }
      }

      // Helper function to add the layer (defined before use)
      const addRailNodesLayer = () => {
        if (!map.current) return

        // Remove existing layer if it exists
        if (map.current.getLayer('rail-nodes')) {
          map.current.removeLayer('rail-nodes')
        }

        // Find a good layer to insert rail nodes before
        const layers = map.current.getStyle().layers
        const insertBeforeLayer = layers.find((layer: mapboxgl.AnyLayer) =>
          layer.type === 'symbol' || layer.id.includes('label')
        )?.id

        // Add rail nodes as circles - color by node type (same as main map)
        map.current.addLayer({
          id: 'rail-nodes',
          type: 'circle',
          source: 'rail-nodes',
          paint: {
            'circle-color': [
              'case',
              ['==', ['get', 'nodeType'], 'intermodal_terminal'],
              '#00ff00', // Bright green for intermodal terminals
              ['==', ['get', 'nodeType'], 'yard'],
              '#ffff00', // Yellow for yards
              ['==', ['get', 'nodeType'], 'station'],
              '#0000ff', // Blue for stations
              ['==', ['get', 'nodeType'], 'junction'],
              '#ff00ff', // Magenta for junctions
              ['==', ['get', 'net'], 'S'],
              '#ff0000', // Fallback: red for main network nodes
              ['==', ['get', 'net'], 'O'],
              '#ff8800', // Fallback: orange for other
              '#ffaa00', // Yellow for unknown
            ],
            'circle-radius': [
              'interpolate',
              ['linear'],
              ['zoom'],
              3,
              ['case',
                ['==', ['get', 'nodeType'], 'intermodal_terminal'], 6,
                ['==', ['get', 'nodeType'], 'yard'], 5,
                ['==', ['get', 'nodeType'], 'station'], 4,
                ['==', ['get', 'nodeType'], 'junction'], 4,
                2
              ],
              5,
              ['case',
                ['==', ['get', 'nodeType'], 'intermodal_terminal'], 8,
                ['==', ['get', 'nodeType'], 'yard'], 6,
                ['==', ['get', 'nodeType'], 'station'], 5,
                ['==', ['get', 'nodeType'], 'junction'], 5,
                3
              ],
              10,
              ['case',
                ['==', ['get', 'nodeType'], 'intermodal_terminal'], 10,
                ['==', ['get', 'nodeType'], 'yard'], 7,
                ['==', ['get', 'nodeType'], 'station'], 6,
                ['==', ['get', 'nodeType'], 'junction'], 6,
                4
              ],
              15,
              ['case',
                ['==', ['get', 'nodeType'], 'intermodal_terminal'], 12,
                ['==', ['get', 'nodeType'], 'yard'], 9,
                ['==', ['get', 'nodeType'], 'station'], 8,
                ['==', ['get', 'nodeType'], 'junction'], 8,
                6
              ],
            ],
            'circle-stroke-width': [
              'case',
              ['==', ['get', 'nodeType'], 'intermodal_terminal'], 2,
              1
            ],
            'circle-stroke-color': '#000000',
            'circle-opacity': 0.8,
          },
        }, insertBeforeLayer)
      }

      // Add rail nodes source with error handling
      try {
        map.current.addSource('rail-nodes', {
          type: 'geojson',
          data: geoJSON,
        })
        // Add the layer after source is added successfully
        addRailNodesLayer()
      } catch (error) {
        // If source already exists, remove it and retry
        if (error instanceof Error && error.message.includes('already a source')) {
          console.warn('Source already exists, removing and retrying...')
          if (map.current.getSource('rail-nodes')) {
            map.current.removeSource('rail-nodes')
          }
          // Wait a moment before retrying
          setTimeout(() => {
            if (map.current && !map.current.getSource('rail-nodes')) {
              try {
                map.current.addSource('rail-nodes', {
                  type: 'geojson',
                  data: geoJSON,
                })
                // Continue with layer creation
                addRailNodesLayer()
              } catch (retryError) {
                console.error('Error adding rail-nodes source after retry:', retryError)
                setRailNodesLoaded(false)
              }
            }
          }, 50)
          return // Exit early, will continue in setTimeout
        } else {
          throw error // Re-throw if it's a different error
        }
      }

      console.log(`Loaded ${railNodes.length} rail nodes`)

      // Add click handler for rail nodes
      if (onRailNodeClick) {
        const clickHandler = (e: mapboxgl.MapLayerMouseEvent) => {
          if (!e.features || e.features.length === 0) return
          const feature = e.features[0]
          const properties = feature.properties

          // Use OBJECTID as primary identifier (same as main map)
          const nodeId = properties?.OBJECTID?.toString() || properties?.FRA_NODE?.toString() || 'unknown'

          if (nodeId !== 'unknown') {
            onRailNodeClick(nodeId)
          }
        }

        map.current.on('click', 'rail-nodes', clickHandler)

        // Change cursor on hover
        map.current.on('mouseenter', 'rail-nodes', () => {
          if (map.current) {
            map.current.getCanvas().style.cursor = 'pointer'
          }
        })

        map.current.on('mouseleave', 'rail-nodes', () => {
          if (map.current) {
            map.current.getCanvas().style.cursor = ''
          }
        })
      }
    } catch (error) {
      console.error('Error loading rail nodes:', error)
      setRailNodesLoaded(false)
    }
  }


  // Helper function to add elevation heatmap layer
  const addElevationHeatmapLayer = useCallback(() => {
    if (!map.current || !map.current.isStyleLoaded()) return

    // Remove existing layers and sources if present
    if (map.current.getLayer('elevation-heatmap')) {
      map.current.removeLayer('elevation-heatmap')
    }
    if (map.current.getLayer('elevation-hillshade')) {
      map.current.removeLayer('elevation-hillshade')
    }
    if (map.current.getSource('elevation-rgb')) {
      try {
        map.current.removeSource('elevation-rgb')
        } catch {
          // Source might not exist, that's okay
        }
      }
      if (map.current.getSource('elevation-dem-heatmap')) {
        try {
          map.current.removeSource('elevation-dem-heatmap')
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
    // Shadows = lower elevation = green, Highlights = higher elevation = red
    map.current.addLayer({
      id: 'elevation-colored-hillshade',
      type: 'hillshade',
      source: 'elevation-dem-colored',
      paint: {
        'hillshade-shadow-color': '#00ff00', // Green for shadows (lower elevation)
        'hillshade-highlight-color': '#ff0000', // Red for highlights (higher elevation)  
        'hillshade-accent-color': '#ffff00', // Yellow for mid-range elevation
        'hillshade-exaggeration': 2.0, // Increase to make elevation differences more visible
        'hillshade-illumination-direction': 315, // Light direction
        'hillshade-illumination-anchor': 'viewport', // Fixed illumination relative to viewport
      },
    }, beforeId)
    
    // Add a semi-transparent terrain-rgb layer underneath for additional detail
    // Even though we can't decode it properly, it might add some texture
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
        'raster-opacity': 0.3, // Low opacity as a base layer
        'raster-resampling': 'linear',
      },
    }, beforeId)
  }, [showElevationHeatmap])

  // Toggle elevation heatmap when prop changes
  useEffect(() => {
    if (!map.current || !mapLoaded) return

    const updateElevationLayer = () => {
      if (!map.current) return
      
      if (showElevationHeatmap) {
        // Add elevation layer if it doesn't exist
        // Wait a bit for map to be fully ready
        const addElevationWhenReady = () => {
          if (map.current && map.current.isStyleLoaded()) {
            if (!map.current.getLayer('elevation-heatmap')) {
              addElevationHeatmapLayer()
            }
          } else {
            map.current?.once('style.load', () => {
              setTimeout(() => {
                if (map.current && !map.current.getLayer('elevation-heatmap')) {
                  addElevationHeatmapLayer()
                }
              }, 200)
            })
          }
        }
        
        // Try immediately if loaded, otherwise wait
        if (map.current.isStyleLoaded()) {
          setTimeout(addElevationWhenReady, 300)
        } else {
          map.current.once('style.load', () => {
            setTimeout(addElevationWhenReady, 300)
          })
        }
      } else {
        // Remove elevation layers if they exist
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
        
        // Restore base map layers visibility
        const currentMap = map.current
        if (!currentMap) return
        
        const allLayers = currentMap.getStyle().layers
        allLayers.forEach((layer: mapboxgl.AnyLayer) => {
          try {
            // Restore visibility for all layers
            currentMap.setLayoutProperty(layer.id, 'visibility', 'visible')
          } catch {
            // Some layers might not be modifiable, that's okay
          }
        })
      }
    }

    updateElevationLayer()
  }, [showElevationHeatmap, mapLoaded, addElevationHeatmapLayer])


  // Create 3D train object (scaled appropriately for Mapbox coordinates)
  const createTrainObject = (): THREE.Group => {
    const trainGroup = new THREE.Group()

    // Train body (main cargo container) - dimensions in meters
    // Scale will be applied by modelTransform, so use real-world dimensions
    const bodyLength = 12 // meters (shorter)
    const bodyWidth = 3 // meters
    const bodyHeight = 4 // meters
    
    const bodyGeometry = new THREE.BoxGeometry(bodyLength, bodyHeight, bodyWidth)
    const bodyMaterial = new THREE.MeshStandardMaterial({ 
      color: '#3b82f6', // Blue color
      metalness: 0.3,
      roughness: 0.7,
    })
    const body = new THREE.Mesh(bodyGeometry, bodyMaterial)
    body.position.set(0, bodyHeight / 2, 0) // Position body so bottom is at origin
    trainGroup.add(body)

    // Train wheels - 4 wheels
    const wheelRadius = 0.5 // meters
    const wheelWidth = 0.3 // meters
    const wheelGeometry = new THREE.CylinderGeometry(wheelRadius, wheelRadius, wheelWidth, 16)
    const wheelMaterial = new THREE.MeshStandardMaterial({ color: '#1f2937' })
    
    const wheelPositions = [
      [-bodyLength / 3, wheelRadius, -bodyWidth / 2 - wheelWidth / 2],
      [-bodyLength / 3, wheelRadius, bodyWidth / 2 + wheelWidth / 2],
      [bodyLength / 3, wheelRadius, -bodyWidth / 2 - wheelWidth / 2],
      [bodyLength / 3, wheelRadius, bodyWidth / 2 + wheelWidth / 2],
    ]

    wheelPositions.forEach(([x, y, z]) => {
      const wheel = new THREE.Mesh(wheelGeometry, wheelMaterial)
      wheel.rotation.x = Math.PI / 2 // Rotate wheel to be horizontal
      wheel.position.set(x, y, z)
      trainGroup.add(wheel)
    })

    return trainGroup
  }

  // Calculate position along path
  const getPositionAlongPath = (path: [number, number][], progress: number): { lng: number; lat: number; bearing: number } | null => {
    if (path.length < 2) return null

    // Calculate total path length
    let totalLength = 0
    const segmentLengths: number[] = []
    for (let i = 0; i < path.length - 1; i++) {
      const [lng1, lat1] = path[i]
      const [lng2, lat2] = path[i + 1]
      const dx = lng2 - lng1
      const dy = lat2 - lat1
      const segmentLength = Math.sqrt(dx * dx + dy * dy)
      segmentLengths.push(segmentLength)
      totalLength += segmentLength
    }

    // Find target distance
    const targetDistance = totalLength * progress

    // Find which segment we're on
    let accumulatedLength = 0
    for (let i = 0; i < segmentLengths.length; i++) {
      if (accumulatedLength + segmentLengths[i] >= targetDistance) {
        // We're on this segment
        const segmentProgress = (targetDistance - accumulatedLength) / segmentLengths[i]
        const [lng1, lat1] = path[i]
        const [lng2, lat2] = path[i + 1]
        
        const lng = lng1 + (lng2 - lng1) * segmentProgress
        const lat = lat1 + (lat2 - lat1) * segmentProgress
        
        // Calculate bearing
        const dx = lng2 - lng1
        const dy = lat2 - lat1
        const bearing = Math.atan2(dy, dx) * (180 / Math.PI)
        
        return { lng, lat, bearing }
      }
      accumulatedLength += segmentLengths[i]
    }

    // If we've reached the end, return last point
    const lastPoint = path[path.length - 1]
    const secondLastPoint = path[path.length - 2]
    const dx = lastPoint[0] - secondLastPoint[0]
    const dy = lastPoint[1] - secondLastPoint[1]
    const bearing = Math.atan2(dy, dx) * (180 / Math.PI)
    
    return { lng: lastPoint[0], lat: lastPoint[1], bearing }
  }

  // Find closest road to coordinates
  const findClosestRoad = async (lng: number, lat: number): Promise<[number, number][] | null> => {
    if (!map.current) return null

    try {
      // Query for road features near the clicked point
      const point = map.current.project([lng, lat])
      const radius = 200 // pixels - increased radius
      
      // Query rendered features in a box around the point
      // Don't specify layers - query all and filter
      const features = map.current.queryRenderedFeatures(
        [
          [point.x - radius, point.y - radius],
          [point.x + radius, point.y + radius]
        ]
      )

      console.log(`Found ${features.length} total features near click point`)

      // Filter for road-related layers - check layer ID for road keywords
      const roadFeatures = features.filter(f => {
        const layerId = (f.layer?.id || '').toLowerCase()
        const isRoadLayer = layerId.includes('road') || 
                           layerId.includes('street') || 
                           layerId.includes('highway') ||
                           layerId.includes('motorway') ||
                           layerId.includes('trunk') ||
                           layerId.includes('bridge-road') ||
                           layerId.includes('tunnel-road')
        
        const isLineString = f.geometry.type === 'LineString' || f.geometry.type === 'MultiLineString'
        
        if (isRoadLayer && isLineString) {
          console.log(`Found road feature in layer: ${f.layer?.id}`)
        }
        
        return isRoadLayer && isLineString
      })

      console.log(`Found ${roadFeatures.length} road features`)

      if (roadFeatures.length === 0) {
        console.log('No road features found nearby')
        return null
      }

      // Find the closest road segment to the click point
      let closestFeature: GeoJSON.Feature<GeoJSON.LineString> | null = null
      let minDistance = Infinity
      // eslint-disable-next-line @typescript-eslint/no-unused-vars
      const _clickPoint: [number, number] = [lng, lat]

      for (const feature of roadFeatures) {
        let coords: [number, number][] = []
        
        // Handle both LineString and MultiLineString
        if (feature.geometry.type === 'LineString') {
          coords = feature.geometry.coordinates as [number, number][]
        } else if (feature.geometry.type === 'MultiLineString') {
          // Flatten MultiLineString - use the first line or combine all
          const multiCoords = feature.geometry.coordinates as [number, number][][]
          if (multiCoords.length > 0) {
            coords = multiCoords[0] // Use first line for now
          }
        }
        
        if (coords.length < 2) continue
        
        // Find closest point on this road segment
        for (let i = 0; i < coords.length - 1; i++) {
          const [lng1, lat1] = coords[i]
          const [lng2, lat2] = coords[i + 1]
          
          // Project click point onto segment
          const dx = lng2 - lng1
          const dy = lat2 - lat1
          const segmentLength = Math.sqrt(dx * dx + dy * dy)
          
          if (segmentLength > 0) {
            const t = Math.max(0, Math.min(1,
              ((lng - lng1) * dx + (lat - lat1) * dy) / (segmentLength * segmentLength)
            ))
            
            const projLng = lng1 + t * dx
            const projLat = lat1 + t * dy
            
            const dist = Math.sqrt(
              (lng - projLng) ** 2 + (lat - projLat) ** 2
            )
            
            if (dist < minDistance) {
              minDistance = dist
              // Create a LineString feature from the coords
              closestFeature = {
                type: 'Feature',
                properties: feature.properties || {},
                geometry: {
                  type: 'LineString',
                  coordinates: coords
                }
              } as GeoJSON.Feature<GeoJSON.LineString>
            }
          }
        }
      }

      if (closestFeature && closestFeature.geometry.type === 'LineString') {
        const coords = closestFeature.geometry.coordinates as [number, number][]
        console.log(`Found closest road with ${coords.length} points, distance: ${minDistance} degrees`)
        return coords
      }

      return null
    } catch (error) {
      console.error('Error finding closest road:', error)
      return null
    }
  }

  // Find and continue on a new road segment from current position
  const findAndContinueOnRoad = async (lng: number, lat: number) => {
    if (!map.current) return

    console.log(`Finding continuation road from: ${lat}, ${lng}`)
    
    // Find closest road
    const roadPath = await findClosestRoad(lng, lat)
    
    if (roadPath && roadPath.length >= 2) {
      // Use the road path for navigation
      currentPathRef.current = roadPath
      animationProgressRef.current = 0.1 // Start a bit into the path to avoid immediate end
      isAnimatingRef.current = true
      // Animation started
      console.log(`Continuing on new road segment with ${roadPath.length} points`)
    } else {
      // No road found, stop animation
      isAnimatingRef.current = false
      // Animation stopped
      console.log('No road found to continue on, stopping')
    }
  }

  // Place train at clicked coordinates and navigate to closest road
  const placeTrainAtCoordinates = async (lng: number, lat: number) => {
    if (!map.current) {
      console.warn('Cannot place train: map not ready')
      return
    }

    console.log(`Placing train at coordinates: ${lat}, ${lng}`)

    // Find closest road
    const roadPath = await findClosestRoad(lng, lat)
    
    if (!roadPath || roadPath.length < 2) {
      console.warn('Could not find a road nearby, placing train at click location')
      currentPathRef.current = [[lng, lat]]
      // Animation stopped
      isAnimatingRef.current = false
    } else {
      // Find the closest point on the road to the click location
      let closestPointOnRoad: [number, number] | null = null
      let minDist = Infinity
      let closestIndex = 0
      
      for (let i = 0; i < roadPath.length - 1; i++) {
        const [lng1, lat1] = roadPath[i]
        const [lng2, lat2] = roadPath[i + 1]
        
        const dx = lng2 - lng1
        const dy = lat2 - lat1
        const segmentLength = Math.sqrt(dx * dx + dy * dy)
        
        if (segmentLength > 0) {
          const t = Math.max(0, Math.min(1,
            ((lng - lng1) * dx + (lat - lat1) * dy) / (segmentLength * segmentLength)
          ))
          
          const projLng = lng1 + t * dx
          const projLat = lat1 + t * dy
          
          const dist = Math.sqrt(
            (lng - projLng) ** 2 + (lat - projLat) ** 2
          )
          
          if (dist < minDist) {
            minDist = dist
            closestPointOnRoad = [projLng, projLat]
            closestIndex = i
          }
        }
      }
      
      if (closestPointOnRoad) {
        // Create path: from click location to road, then along road
        // First phase: navigate from click location to the closest point on road
        const pathToRoad: [number, number][] = [[lng, lat], closestPointOnRoad]
        
        // Second phase: navigate along the road starting from closest point
        // Extract the road segment starting from closest point
        const roadSegment = roadPath.slice(closestIndex)
        
        // Combine both phases (remove duplicate point at junction)
        currentPathRef.current = [...pathToRoad, ...roadSegment.slice(1)]
        animationProgressRef.current = 0
        // Animation started
        isAnimatingRef.current = true
        console.log(`Train will navigate to road, then along road with ${currentPathRef.current.length} points`)
      } else {
        // Fallback: just use the road path
        currentPathRef.current = roadPath
        animationProgressRef.current = 0
        // Animation started
        isAnimatingRef.current = true
        console.log(`Train will navigate along road with ${roadPath.length} points`)
      }
    }

    // Remove existing layer if it exists
    if (customLayerRef.current && map.current.getLayer('train-3d')) {
      map.current.removeLayer('train-3d')
    }

    // Calculate model transform based on coordinates
    const modelOrigin: [number, number] = [lng, lat]
    // Set altitude to be above ground (in meters)
    // We'll use a small offset to ensure it's above terrain
    const modelAltitude = 5 // 5 meters above ground
    // Rotate to lay flat: rotate 90 degrees around X axis to make it horizontal
    const modelRotate = [Math.PI / 2, 0, 0] // Rotate 90 degrees around X axis
    const modelAsMercatorCoordinate = mapboxgl.MercatorCoordinate.fromLngLat(
      modelOrigin,
      modelAltitude
    )

    const modelTransform = {
      translateX: modelAsMercatorCoordinate.x,
      translateY: modelAsMercatorCoordinate.y,
      translateZ: modelAsMercatorCoordinate.z,
      rotateX: modelRotate[0],
      rotateY: modelRotate[1],
      rotateZ: modelRotate[2],
      scale: modelAsMercatorCoordinate.meterInMercatorCoordinateUnits(),
    }
    
    // Store transform in ref so render function can access it
    modelTransformRef.current = modelTransform

    const customLayer: mapboxgl.CustomLayerInterface = {
      id: 'train-3d',
      type: 'custom',
      renderingMode: '3d',
      onAdd: (mapInstance, gl) => {
        // Create Three.js scene
        const scene = new THREE.Scene()
        sceneRef.current = scene

        // Create camera (will be synced with map camera)
        const camera = new THREE.Camera()
        cameraRef.current = camera

        // Create renderer
        const renderer = new THREE.WebGLRenderer({
          canvas: mapInstance.getCanvas(),
          context: gl as WebGLRenderingContext,
          antialias: true,
        })
        renderer.autoClear = false
        renderer.setPixelRatio(window.devicePixelRatio)
        rendererRef.current = renderer

        // Create train object
        const train = createTrainObject()
        trainObjectRef.current = train
        scene.add(train)

        // Add lighting
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8)
        directionalLight.position.set(0, -70, 100).normalize()
        scene.add(directionalLight)
        
        const directionalLight2 = new THREE.DirectionalLight(0xffffff, 0.6)
        directionalLight2.position.set(0, 70, 100).normalize()
        scene.add(directionalLight2)
        
        scene.add(new THREE.AmbientLight(0xffffff, 0.4))
      },
      render: (gl, matrix) => {
        if (!trainObjectRef.current || !cameraRef.current || !rendererRef.current || !sceneRef.current || !map.current) return
        if (!modelTransformRef.current) return

        const camera = cameraRef.current
        const renderer = rendererRef.current
        const scene = sceneRef.current
        const modelTransform = modelTransformRef.current
        const modelAltitude = 5

        // Get the map's transformation matrix
        const m = new THREE.Matrix4().fromArray(matrix)

        // Update train position if animating along a path
        if (isAnimatingRef.current && currentPathRef.current && currentPathRef.current.length > 1) {
          // Get position along path
          const position = getPositionAlongPath(currentPathRef.current, animationProgressRef.current)
          if (position) {
            // Debug log occasionally
            if (Math.floor(animationProgressRef.current * 100) % 10 === 0 && animationProgressRef.current % 0.1 < 0.001) {
              console.log(`Animating train: progress=${animationProgressRef.current.toFixed(3)}, position=[${position.lat.toFixed(6)}, ${position.lng.toFixed(6)}]`)
            }
            // Update model transform with new position
            const modelAsMercatorCoordinate = mapboxgl.MercatorCoordinate.fromLngLat(
              [position.lng, position.lat],
              modelAltitude
            )
            modelTransform.translateX = modelAsMercatorCoordinate.x
            modelTransform.translateY = modelAsMercatorCoordinate.y
            modelTransform.translateZ = modelAsMercatorCoordinate.z
            
            // Update rotation to face direction of travel
            modelTransform.rotateZ = (position.bearing * Math.PI) / 180

            // Update animation progress (smooth continuous movement)
            animationProgressRef.current += 0.0008 // Adjust speed here
            if (animationProgressRef.current >= 1) {
              // Reached end of current path - find next road segment to continue
              const currentPosition = position
              if (currentPosition) {
                // Find a new road segment starting near the end position
                findAndContinueOnRoad(currentPosition.lng, currentPosition.lat).catch(err => {
                  console.warn('Could not continue on road:', err)
                  // Stop animation if we can't find a new road
                  isAnimatingRef.current = false
                  // Animation stopped
                })
              } else {
                // Stop animation if no position
                isAnimatingRef.current = false
                // Animation stopped
              }
            }
          }
        }

        // Create rotation matrices
        const rotationX = new THREE.Matrix4().makeRotationAxis(
          new THREE.Vector3(1, 0, 0),
          modelTransform.rotateX
        )
        const rotationY = new THREE.Matrix4().makeRotationAxis(
          new THREE.Vector3(0, 1, 0),
          modelTransform.rotateY
        )
        const rotationZ = new THREE.Matrix4().makeRotationAxis(
          new THREE.Vector3(0, 0, 1),
          modelTransform.rotateZ
        )

        // Create the model transformation matrix
        const l = new THREE.Matrix4()
          .makeTranslation(
            modelTransform.translateX,
            modelTransform.translateY,
            modelTransform.translateZ
          )
          .scale(
            new THREE.Vector3(
              modelTransform.scale,
              -modelTransform.scale, // Flip Y axis for Mapbox
              modelTransform.scale
            )
          )
          .multiply(rotationX)
          .multiply(rotationY)
          .multiply(rotationZ)

        // Apply transformations to camera
        camera.projectionMatrix = m.multiply(l)

        // Render scene
        try {
          // Reset WebGL state (this preserves Mapbox's viewport)
          renderer.resetState()
          
          // Render the scene
          renderer.render(scene, camera)
          
          // Always trigger repaint to ensure continuous rendering
          map.current.triggerRepaint()
        } catch (error) {
          console.error('Error rendering train:', error)
        }
      },
    }

    customLayerRef.current = customLayer
    try {
      map.current.addLayer(customLayer)
      console.log('Train layer added successfully')
    } catch (error) {
      console.error('Error adding train layer:', error)
    }
  }

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (map.current && customLayerRef.current) {
        try {
          map.current.removeLayer('train-3d')
        } catch {
          // Layer might not exist
        }
      }
    }
  }, [])

  return (
    <div ref={mapContainer} className="w-full h-full relative">
      {!mapLoaded && (
        <div className="absolute inset-0 flex items-center justify-center bg-gray-100 z-0">
          <div className="text-center">
            <div className="w-12 h-12 border-4 border-gray-300 border-t-blue-500 rounded-full animate-spin mx-auto mb-4" />
            <div className="text-gray-600">Loading 3D map...</div>
          </div>
        </div>
      )}

    </div>
  )
}
