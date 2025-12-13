/**
 * Service for fetching North American rail network nodes
 */

// Using the NTAD North American Rail Network Nodes endpoint
const RAIL_NODES_API_URL = 'https://services.arcgis.com/xOi1kZaI0eWDREZv/arcgis/rest/services/NTAD_North_American_Rail_Network_Nodes/FeatureServer/0/query'

export interface RailNodeFeature {
  attributes: {
    OBJECTID: number
    FRA_NODE?: number
    STFIPS?: string
    STATEAB?: string
    COUNTRY?: string
    RROWNER1?: string
    RROWNER2?: string
    RROWNER3?: string
    NET?: string
    TRACKS?: number
    // Additional fields that might indicate intermodal terminals
    YARDNAME?: string
    DIVISION?: string
    SUBDIV?: string
    BRANCH?: string
    PASSNGR?: string // Passenger service indicator
    STRACNET?: string // Strategic rail corridor network
    [key: string]: any
  }
  geometry: {
    x: number // longitude
    y: number // latitude
  }
}

export interface RailNodesResponse {
  objectIdFieldName: string
  geometryType: string
  spatialReference: {
    wkid: number
    latestWkid: number
  }
  features: RailNodeFeature[]
  exceededTransferLimit?: boolean
}

/**
 * Fetches rail network nodes from ArcGIS API
 * Handles pagination to get all available nodes
 */
export async function fetchRailNodes(
  maxRecords: number = 200000
): Promise<RailNodeFeature[]> {
  const allFeatures: RailNodeFeature[] = []
  let offset = 0
  const pageSize = 2000 // Increased page size for faster fetching
  let hasMore = true
  let consecutiveEmptyPages = 0

  console.log('Starting to fetch rail nodes data...')

  while (hasMore && allFeatures.length < maxRecords) {
    const queryParams = new URLSearchParams({
      where: '1=1', // Get all records - no spatial filter
      outFields: '*',
      outSR: '4326',
      f: 'json',
      resultRecordCount: pageSize.toString(),
      resultOffset: offset.toString(),
    })

    try {
      const response = await fetch(`${RAIL_NODES_API_URL}?${queryParams.toString()}`)
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data: RailNodesResponse = await response.json()
      
      if ((data as any).error) {
        throw new Error((data as any).error.message || 'API returned an error')
      }
      
      if (!data.features || !Array.isArray(data.features)) {
        console.warn('API response missing features array:', data)
        hasMore = false
        break
      }

      if (data.features.length === 0) {
        consecutiveEmptyPages++
        if (consecutiveEmptyPages >= 3) {
          console.log('Received 3 consecutive empty pages. Stopping pagination.')
          hasMore = false
          break
        }
        offset += pageSize
        continue
      }

      consecutiveEmptyPages = 0 // Reset counter on successful fetch
      allFeatures.push(...data.features)
      offset += pageSize

      console.log(`Fetched page: ${data.features.length} nodes (total: ${allFeatures.length})`)

      // Check if there's more data
      if (data.exceededTransferLimit) {
        hasMore = true
        console.log('API indicates more data available, continuing...')
      } else if (data.features.length < pageSize) {
        // Got less than a full page, probably no more data
        hasMore = false
        console.log('Received partial page, assuming all data fetched')
      } else {
        // Got a full page, might be more
        hasMore = true
      }

      // Safety check to avoid infinite loops
      if (offset > maxRecords) {
        console.warn(`Reached max records limit: ${maxRecords}`)
        hasMore = false
      }
      
      // Small delay to avoid overwhelming the API
      await new Promise(resolve => setTimeout(resolve, 100))
    } catch (error) {
      console.error('Error fetching rail nodes data:', error)
      // If we have some data, return what we have
      if (allFeatures.length > 0) {
        console.warn(`Returning ${allFeatures.length} rail nodes despite error`)
        return allFeatures
      }
      throw error
    }
  }

  console.log(`Total rail nodes fetched: ${allFeatures.length}`)
  return allFeatures
}

/**
 * Converts rail node features to GeoJSON format for Mapbox
 */
export type NodeType = 'rail_node' | 'intermodal_terminal' | 'yard' | 'junction' | 'station'

export function identifyNodeType(node: RailNodeFeature): NodeType {
  const attrs = node.attributes
  
  // Check for intermodal terminal indicators
  if (attrs.YARDNAME && attrs.YARDNAME.trim() !== '') {
    // Yards often serve as intermodal terminals
    const yardName = attrs.YARDNAME.toLowerCase()
    if (yardName.includes('intermodal') || yardName.includes('terminal') || yardName.includes('container')) {
      return 'intermodal_terminal'
    }
    return 'yard'
  }
  
  // Check for passenger stations (often have intermodal connections)
  if (attrs.PASSNGR && attrs.PASSNGR !== null && attrs.PASSNGR !== '') {
    return 'station'
  }
  
  // Check for multiple tracks (junctions typically have more tracks)
  if (attrs.TRACKS && attrs.TRACKS > 2) {
    return 'junction'
  }
  
  // Default to rail node
  return 'rail_node'
}

export function railNodesToGeoJSON(features: RailNodeFeature[]): GeoJSON.FeatureCollection {
  return {
    type: 'FeatureCollection',
    features: features.map(feature => {
      const nodeType = identifyNodeType(feature)
      
      // Build location string
      const locationParts: string[] = []
      if (feature.attributes.YARDNAME) locationParts.push(feature.attributes.YARDNAME)
      if (feature.attributes.STATEAB) locationParts.push(feature.attributes.STATEAB)
      if (feature.attributes.COUNTRY) locationParts.push(feature.attributes.COUNTRY)
      const locationString = locationParts.length > 0 
        ? locationParts.join(', ')
        : `${feature.geometry.y.toFixed(4)}, ${feature.geometry.x.toFixed(4)}`
      
      return {
        type: 'Feature' as const,
        properties: {
          id: feature.attributes.OBJECTID,
          fraNode: feature.attributes.FRA_NODE || 0,
          owner: feature.attributes.RROWNER1 || 'Unknown',
          state: feature.attributes.STATEAB || '',
          country: feature.attributes.COUNTRY || '',
          tracks: feature.attributes.TRACKS || 0,
          net: feature.attributes.NET || '',
          nodeType: nodeType,
          yardName: feature.attributes.YARDNAME || '',
          division: feature.attributes.DIVISION || '',
          subdivision: feature.attributes.SUBDIV || '',
          branch: feature.attributes.BRANCH || '',
          passenger: feature.attributes.PASSNGR || '',
          stracnet: feature.attributes.STRACNET || '',
          // Enhanced location data
          longitude: feature.geometry.x,
          latitude: feature.geometry.y,
          locationString: locationString,
        },
        geometry: {
          type: 'Point' as const,
          coordinates: [feature.geometry.x, feature.geometry.y], // [lng, lat]
        },
      }
    }),
  }
}

