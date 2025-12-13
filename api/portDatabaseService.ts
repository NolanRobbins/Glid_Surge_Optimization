/**
 * Service for fetching port database with coordinates from PortWatch
 */

const PORT_DATABASE_API_URL = 'https://services9.arcgis.com/weJ1QsnbMYJlCHdG/arcgis/rest/services/PortWatch_ports_database/FeatureServer/0/query'

export interface PortDatabaseEntry {
  portid: string
  portname: string
  country: string
  ISO3: string
  continent: string
  fullname: string
  lat: number
  lon: number
  vessel_count_total: number
  vessel_count_container: number
  vessel_count_dry_bulk: number
  vessel_count_general_cargo: number
  vessel_count_RoRo: number
  vessel_count_tanker: number
  industry_top1?: string
  industry_top2?: string
  industry_top3?: string
  share_country_maritime_import?: number
  share_country_maritime_export?: number
  LOCODE?: string
  pageid?: string
  ObjectId: number
}

export interface PortDatabaseResponse {
  objectIdFieldName: string
  geometryType: string
  spatialReference: {
    wkid: number
    latestWkid: number
  }
  fields: Array<{
    name: string
    type: string
    alias: string
  }>
  exceededTransferLimit: boolean
  features: Array<{
    attributes: PortDatabaseEntry
    geometry: {
      x: number // longitude
      y: number // latitude
    }
  }>
}

/**
 * Fetches port database with coordinates
 * Handles pagination to get all ports
 */
export async function fetchPortDatabase(
  maxRecords: number = 100000
): Promise<Map<string, PortDatabaseEntry>> {
  const allPorts: PortDatabaseEntry[] = []
  let offset = 0
  const pageSize = 2000
  let hasMore = true
  let consecutiveEmptyPages = 0

  console.log('Starting to fetch all ports from database...')

  while (hasMore && allPorts.length < maxRecords) {
    const queryParams = new URLSearchParams({
      where: '1=1', // Get all ports
      outFields: '*',
      outSR: '4326',
      f: 'json',
      resultRecordCount: pageSize.toString(),
      resultOffset: offset.toString(),
    })

    try {
      const response = await fetch(`${PORT_DATABASE_API_URL}?${queryParams.toString()}`)
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data: PortDatabaseResponse = await response.json()
      
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
      
      // Process features
      data.features.forEach(feature => {
        const port = feature.attributes
        // Use coordinates from geometry if lat/lon are missing
        if (!port.lat || !port.lon) {
          port.lat = feature.geometry.y
          port.lon = feature.geometry.x
        }
        allPorts.push(port)
      })

      offset += pageSize
      console.log(`Fetched page: ${data.features.length} ports (total: ${allPorts.length})`)

      // Check if there's more data
      if (data.exceededTransferLimit) {
        hasMore = true
        console.log('API indicates more data available, continuing...')
      } else if (data.features.length < pageSize) {
        hasMore = false
        console.log('Received partial page, assuming all data fetched')
      } else {
        hasMore = true
      }

      // Safety check
      if (offset > maxRecords) {
        console.warn(`Reached max records limit: ${maxRecords}`)
        hasMore = false
      }
      
      // Small delay to avoid overwhelming the API
      await new Promise(resolve => setTimeout(resolve, 100))
    } catch (error) {
      console.error('Error fetching port database:', error)
      if (allPorts.length > 0) {
        console.warn(`Returning ${allPorts.length} ports despite error`)
        break
      }
      throw error
    }
  }

  console.log(`Total ports fetched: ${allPorts.length}`)

  // Create a map of portid -> port entry
  const portMap = new Map<string, PortDatabaseEntry>()
  allPorts.forEach(port => {
    portMap.set(port.portid, port)
  })

  return portMap
}

/**
 * Gets coordinates for a port from the database
 */
export async function getPortCoordinates(
  portId: string,
  portDatabase: Map<string, PortDatabaseEntry>
): Promise<[number, number] | null> {
  const port = portDatabase.get(portId)
  if (port && port.lat && port.lon) {
    return [port.lon, port.lat] // [lng, lat] format for Mapbox
  }
  return null
}

