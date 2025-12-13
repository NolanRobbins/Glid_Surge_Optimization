/**
 * Service for fetching North American rail network data
 * Includes caching and incremental loading to prevent stack overflow
 */

const RAIL_API_URL = 'https://services.arcgis.com/xOi1kZaI0eWDREZv/arcgis/rest/services/NTAD_North_American_Rail_Network_Lines/FeatureServer/0/query'
const CACHE_KEY = 'rail_network_cache'
const CACHE_VERSION = '1.0'
const CACHE_EXPIRY_DAYS = 7

export interface RailLineFeature {
  attributes: {
    OBJECTID: number
    FRAARCID?: number
    FRFRANODE?: number // From node ID
    TOFRANODE?: number // To node ID
    STFIPS?: string
    STATEAB?: string
    COUNTRY?: string
    RROWNER1?: string
    RROWNER2?: string
    RROWNER3?: string
    MILES?: number
    KM?: number
    NET?: string
    TRACKS?: number
    // Additional fields for line classification
    DIVISION?: string
    SUBDIV?: string
    BRANCH?: string
    PASSNGR?: string // Passenger service indicator
    STRACNET?: string // Strategic rail corridor network
    [key: string]: any
  }
  geometry: {
    paths: number[][][] // Array of coordinate arrays
  }
}

export type LineSegmentType = 'main_line' | 'branch_line' | 'passenger_line' | 'freight_line' | 'intermodal_line' | 'yard_track'

interface CachedRailData {
  version: string
  timestamp: number
  features: RailLineFeature[]
}

/**
 * Safely calculates min/max of large arrays by processing in chunks
 */
function safeMinMax(values: number[]): { min: number; max: number } {
  if (values.length === 0) return { min: 0, max: 0 }
  if (values.length === 1) return { min: values[0], max: values[0] }
  
  // Process in chunks to avoid stack overflow
  const chunkSize = 10000
  let min = Infinity
  let max = -Infinity
  
  for (let i = 0; i < values.length; i += chunkSize) {
    const chunk = values.slice(i, i + chunkSize)
    const chunkMin = Math.min(...chunk)
    const chunkMax = Math.max(...chunk)
    min = Math.min(min, chunkMin)
    max = Math.max(max, chunkMax)
  }
  
  return { min, max }
}

/**
 * Safely extracts coordinates from features in chunks
 */
function extractCoordinates(features: RailLineFeature[]): { lngs: number[]; lats: number[] } {
  const lngs: number[] = []
  const lats: number[] = []
  const chunkSize = 1000
  
  for (let i = 0; i < features.length; i += chunkSize) {
    const chunk = features.slice(i, i + chunkSize)
    chunk.forEach(feature => {
      feature.geometry.paths.forEach(path => {
        path.forEach(coord => {
          lngs.push(coord[0])
          lats.push(coord[1])
        })
      })
    })
  }
  
  return { lngs, lats }
}

export function identifyLineSegmentType(line: RailLineFeature): LineSegmentType {
  const attrs = line.attributes
  
  // Check for passenger lines
  if (attrs.PASSNGR && attrs.PASSNGR !== null && attrs.PASSNGR !== '') {
    return 'passenger_line'
  }
  
  // Check for branch lines
  if (attrs.BRANCH && attrs.BRANCH.trim() !== '') {
    return 'branch_line'
  }
  
  // Check for strategic/intermodal corridors
  if (attrs.STRACNET && attrs.STRACNET !== null && attrs.STRACNET !== '') {
    return 'intermodal_line'
  }
  
  // Check for main network (NET='S' typically means main network)
  if (attrs.NET === 'S') {
    return 'main_line'
  }
  
  // Default to freight line
  return 'freight_line'
}

export interface RailNetworkResponse {
  objectIdFieldName: string
  geometryType: string
  spatialReference: {
    wkid: number
    latestWkid: number
  }
  features: RailLineFeature[]
  exceededTransferLimit?: boolean
}

/**
 * Checks if a feature intersects with given bounds
 */
function featureIntersectsBounds(
  feature: RailLineFeature,
  bounds?: { minLng: number; minLat: number; maxLng: number; maxLat: number }
): boolean {
  if (!bounds) return true
  
  // Check if any coordinate in the feature is within bounds
  for (const path of feature.geometry.paths) {
    for (const coord of path) {
      const [lng, lat] = coord
      if (
        lng >= bounds.minLng &&
        lng <= bounds.maxLng &&
        lat >= bounds.minLat &&
        lat <= bounds.maxLat
      ) {
        return true
      }
    }
  }
  return false
}

/**
 * Gets cached rail network data
 */
async function getCachedRailData(): Promise<RailLineFeature[] | null> {
  if (typeof window === 'undefined') return null
  
  try {
    const cached = localStorage.getItem(CACHE_KEY)
    if (!cached) return null
    
    const data: CachedRailData = JSON.parse(cached)
    
    // Check version
    if (data.version !== CACHE_VERSION) {
      localStorage.removeItem(CACHE_KEY)
      return null
    }
    
    // Check expiry
    const expiryTime = data.timestamp + (CACHE_EXPIRY_DAYS * 24 * 60 * 60 * 1000)
    if (Date.now() > expiryTime) {
      localStorage.removeItem(CACHE_KEY)
      return null
    }
    
    console.log(`Using cached rail network data (${data.features.length} features)`)
    return data.features
  } catch (error) {
    console.warn('Error reading cache:', error)
    return null
  }
}

/**
 * Saves rail network data to cache
 */
async function saveCachedRailData(features: RailLineFeature[]): Promise<void> {
  if (typeof window === 'undefined') return
  
  try {
    const data: CachedRailData = {
      version: CACHE_VERSION,
      timestamp: Date.now(),
      features,
    }
    
    // Check storage size (localStorage has ~5-10MB limit)
    const json = JSON.stringify(data)
    const sizeMB = new Blob([json]).size / (1024 * 1024)
    
    if (sizeMB > 8) {
      console.warn(`Cache size (${sizeMB.toFixed(2)}MB) too large, not caching`)
      return
    }
    
    localStorage.setItem(CACHE_KEY, json)
    console.log(`Cached rail network data (${features.length} features, ${sizeMB.toFixed(2)}MB)`)
  } catch (error) {
    if (error instanceof DOMException && error.name === 'QuotaExceededError') {
      console.warn('Cache storage quota exceeded, clearing old cache')
      localStorage.removeItem(CACHE_KEY)
    } else {
      console.warn('Error saving cache:', error)
    }
  }
}

/**
 * Fetches rail network data from ArcGIS API with caching and incremental loading
 * @param bounds Optional bounds to filter features (for incremental loading)
 * @param maxRecords Maximum records to fetch
 * @param useCache Whether to use cached data if available
 */
export async function fetchRailNetwork(
  bounds?: { minLng: number; minLat: number; maxLng: number; maxLat: number },
  maxRecords: number = 50000, // Reduced default to prevent stack overflow
  useCache: boolean = true
): Promise<RailLineFeature[]> {
  // Try cache first if enabled
  if (useCache && !bounds) {
    const cached = await getCachedRailData()
    if (cached) {
      // Filter by bounds if provided
      if (bounds) {
        return cached.filter(f => featureIntersectsBounds(f, bounds))
      }
      return cached
    }
  }

  const allFeatures: RailLineFeature[] = []
  let offset = 0
  const pageSize = 1000 // Reduced page size for better memory management
  let hasMore = true
  let consecutiveEmptyPages = 0

  console.log('Starting to fetch rail network data...', bounds ? 'with bounds filter' : 'all data')

  while (hasMore && allFeatures.length < maxRecords) {
    const queryParams = new URLSearchParams({
      where: '1=1',
      outFields: '*',
      outSR: '4326',
      f: 'json',
      resultRecordCount: pageSize.toString(),
      resultOffset: offset.toString(),
    })

    try {
      const response = await fetch(`${RAIL_API_URL}?${queryParams.toString()}`)
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data: RailNetworkResponse = await response.json()
      
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

      consecutiveEmptyPages = 0
      
      // Filter by bounds if provided (client-side filtering for incremental loading)
      const filteredFeatures = bounds
        ? data.features.filter(f => featureIntersectsBounds(f, bounds))
        : data.features
      
      allFeatures.push(...filteredFeatures)
      offset += pageSize

      console.log(`Fetched page: ${data.features.length} segments (${filteredFeatures.length} in bounds, total: ${allFeatures.length})`)

      // Log geographic distribution safely (only for first page to avoid stack overflow)
      if (allFeatures.length === filteredFeatures.length && filteredFeatures.length > 0 && filteredFeatures.length < 1000) {
        const { lngs, lats } = extractCoordinates(filteredFeatures)
        if (lngs.length > 0) {
          const { min: pageMinLng, max: pageMaxLng } = safeMinMax(lngs)
          const { min: pageMinLat, max: pageMaxLat } = safeMinMax(lats)
          console.log(`Page bounds: ${pageMinLng.toFixed(2)}, ${pageMinLat.toFixed(2)} to ${pageMaxLng.toFixed(2)}, ${pageMaxLat.toFixed(2)}`)
        }
      }

      // Check if we've reached the max records limit (based on collected features, not offset)
      if (allFeatures.length >= maxRecords) {
        console.log(`Reached max records limit: ${maxRecords} features collected. Stopping fetch but will render collected data.`)
        hasMore = false
        // Still return what we have - don't break, just stop fetching more
        break
      }

      // Check if there's more data from API
      if (data.exceededTransferLimit) {
        hasMore = true
        console.log('API indicates more data available, continuing...')
      } else if (data.features.length < pageSize) {
        hasMore = false
        console.log('Received partial page, assuming all data fetched')
      } else {
        hasMore = true
      }
      
      // Delay to avoid overwhelming the API and allow UI to update
      await new Promise(resolve => setTimeout(resolve, 100))
    } catch (error) {
      console.error('Error fetching rail network data:', error)
      if (allFeatures.length > 0) {
        console.warn(`Returning ${allFeatures.length} rail segments despite error`)
        return allFeatures
      }
      throw error
    }
  }

  console.log(`Total rail segments fetched: ${allFeatures.length}${allFeatures.length >= maxRecords ? ` (limit reached: ${maxRecords})` : ''}`)
  
  // Cache the data if we fetched all (no bounds filter) and didn't hit limit
  if (!bounds && allFeatures.length > 0 && allFeatures.length < maxRecords) {
    await saveCachedRailData(allFeatures)
  }
  
  // Always return what we have, even if we hit the limit
  if (allFeatures.length > 0) {
    return allFeatures
  }
  
  // If we have no features, return empty array
  return []
}

/**
 * Converts rail line features to GeoJSON format for Mapbox
 * Processes in chunks with delays to prevent stack overflow
 */
export async function railLinesToGeoJSON(
  features: RailLineFeature[],
  onProgress?: (processed: number, total: number) => void
): Promise<GeoJSON.FeatureCollection> {
  console.log(`Converting ${features.length} rail features to GeoJSON...`)
  
  const batchSize = 5000 // Smaller batches for better memory management
  const allGeoFeatures: GeoJSON.Feature[] = []
  
  for (let i = 0; i < features.length; i += batchSize) {
    const batch = features.slice(i, i + batchSize)
    const batchFeatures = batch.map(feature => {
      const segmentType = identifyLineSegmentType(feature)
      
      return {
        type: 'Feature' as const,
        properties: {
          id: feature.attributes.OBJECTID,
          fraArcId: feature.attributes.FRAARCID || 0,
          fromNode: feature.attributes.FRFRANODE || 0,
          toNode: feature.attributes.TOFRANODE || 0,
          owner: feature.attributes.RROWNER1 || 'Unknown',
          state: feature.attributes.STATEAB || '',
          country: feature.attributes.COUNTRY || '',
          miles: feature.attributes.MILES || 0,
          km: feature.attributes.KM || 0,
          tracks: feature.attributes.TRACKS || 0,
          net: feature.attributes.NET || '',
          segmentType: segmentType,
          division: feature.attributes.DIVISION || '',
          subdivision: feature.attributes.SUBDIV || '',
          branch: feature.attributes.BRANCH || '',
          passenger: feature.attributes.PASSNGR || '',
          stracnet: feature.attributes.STRACNET || '',
        },
        geometry: {
          type: 'MultiLineString' as const,
          coordinates: feature.geometry.paths.map(path => 
            path.map(coord => [coord[0], coord[1]]) // Convert to [lng, lat]
          ),
        },
      }
    })
    
    allGeoFeatures.push(...batchFeatures)
    
    const processed = Math.min(i + batchSize, features.length)
    if (onProgress) {
      onProgress(processed, features.length)
    }
    
    if (i + batchSize < features.length) {
      console.log(`Processed ${processed}/${features.length} features...`)
      // Yield to event loop to prevent blocking
      await new Promise(resolve => setTimeout(resolve, 0))
    }
  }
  
  console.log(`GeoJSON conversion complete: ${allGeoFeatures.length} features`)
  
  return {
    type: 'FeatureCollection',
    features: allGeoFeatures,
  }
}

/**
 * Fetches rail network data incrementally based on map bounds
 * This allows loading only visible data to prevent stack overflow
 */
export async function fetchRailNetworkIncremental(
  bounds: { minLng: number; minLat: number; maxLng: number; maxLat: number },
  maxRecords: number = 10000
): Promise<RailLineFeature[]> {
  // Expand bounds slightly to load nearby data
  const padding = 2.0 // degrees
  const expandedBounds = {
    minLng: bounds.minLng - padding,
    minLat: bounds.minLat - padding,
    maxLng: bounds.maxLng + padding,
    maxLat: bounds.maxLat + padding,
  }
  
  return fetchRailNetwork(expandedBounds, maxRecords, true)
}
