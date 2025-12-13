import { PortTrafficData, ArcGISResponse, PortSummary } from '@/types/portTraffic'

const ARCGIS_API_URL = 'https://services9.arcgis.com/weJ1QsnbMYJlCHdG/arcgis/rest/services/Daily_Trade_Data/FeatureServer/0/query'

export interface QueryParams {
  where?: string
  outFields?: string
  orderByFields?: string
  resultOffset?: number
  resultRecordCount?: number
  returnGeometry?: boolean
  returnDistinctValues?: boolean
}

/**
 * Fetches port traffic data from ArcGIS API
 * Returns data with pagination info
 */
async function fetchPortTrafficDataPage(
  params: QueryParams = {}
): Promise<{ data: PortTrafficData[]; exceededTransferLimit: boolean; hasMore: boolean }> {
  const {
    where = '1=1',
    outFields = '*',
    orderByFields,
    resultOffset = 0,
    resultRecordCount = 1000,
    returnGeometry = false,
    returnDistinctValues = false,
  } = params

  // Build query parameters - match the working example format
  const queryParams = new URLSearchParams({
    where,
    outFields,
    outSR: '4326',
    f: 'json',
  })
  
  // Add pagination parameters
  if (resultOffset > 0) {
    queryParams.append('resultOffset', resultOffset.toString())
  }
  queryParams.append('resultRecordCount', resultRecordCount.toString())
  
  // Add optional parameters
  if (returnGeometry) {
    queryParams.append('returnGeometry', 'true')
  }
  
  if (returnDistinctValues) {
    queryParams.append('returnDistinctValues', 'true')
  }
  
  // Add orderByFields if provided (some services may not support this)
  if (orderByFields && orderByFields.trim() !== '') {
    try {
      queryParams.append('orderByFields', orderByFields)
    } catch (e) {
      // If orderByFields causes issues, skip it
      console.warn('Could not add orderByFields parameter:', e)
    }
  }

  try {
    const response = await fetch(`${ARCGIS_API_URL}?${queryParams.toString()}`)
    
    if (!response.ok) {
      const errorText = await response.text()
      throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`)
    }

    const data: ArcGISResponse = await response.json()
    
    // Check if the response has an error
    if ((data as any).error) {
      const errorMsg = (data as any).error.message || JSON.stringify((data as any).error)
      throw new Error(`API returned an error: ${errorMsg}`)
    }
    
    // Check if features array exists
    if (!data.features || !Array.isArray(data.features)) {
      console.warn('API response missing features array:', data)
      // Return empty result instead of throwing to allow app to continue
      return {
        data: [],
        exceededTransferLimit: false,
        hasMore: false,
      }
    }
    
    return {
      data: data.features.map(feature => feature.attributes),
      exceededTransferLimit: data.exceededTransferLimit || false,
      hasMore: data.exceededTransferLimit && data.features.length > 0,
    }
  } catch (error) {
    console.error('Error fetching port traffic data:', error)
    // Re-throw with more context
    if (error instanceof Error) {
      throw new Error(`Failed to fetch port traffic data: ${error.message}`)
    }
    throw new Error('Failed to fetch port traffic data: Unknown error')
  }
}

/**
 * Fetches port traffic data from ArcGIS API (single page)
 * This is a convenience wrapper that returns just the data array
 */
export async function fetchPortTrafficData(
  params: QueryParams = {}
): Promise<PortTrafficData[]> {
  const result = await fetchPortTrafficDataPage(params)
  return result.data
}

/**
 * Fetches all port traffic data with automatic pagination
 * Note: Date filtering is done client-side to avoid ArcGIS API date format issues
 */
export async function fetchAllPortTrafficData(
  dateRange?: { start?: Date; end?: Date },
  portId?: string,
  country?: string,
  maxRecords?: number
): Promise<PortTrafficData[]> {
  // Build WHERE clause - avoid date filtering in WHERE clause due to ArcGIS date format requirements
  // We'll filter dates client-side instead
  let whereClause = '1=1'
  
  // Only add non-date filters to WHERE clause
  if (portId) {
    whereClause += ` AND portid = '${portId.replace(/'/g, "''")}'`
  }
  
  if (country) {
    whereClause += ` AND country = '${country.replace(/'/g, "''")}'`
  }

  const allData: PortTrafficData[] = []
  let offset = 0
  const pageSize = 1000
  let hasMore = true
  const maxRecordsLimit = maxRecords || Infinity
  let consecutiveEmptyPages = 0

  while (hasMore && allData.length < maxRecordsLimit) {
    const result = await fetchPortTrafficDataPage({
      where: whereClause,
      resultOffset: offset,
      resultRecordCount: pageSize,
    })

    // Add data to our collection
    if (result.data.length > 0) {
      allData.push(...result.data)
      consecutiveEmptyPages = 0
    } else {
      consecutiveEmptyPages++
      // If we get 2 consecutive empty pages, stop
      if (consecutiveEmptyPages >= 2) {
        hasMore = false
        break
      }
    }
    
    // Check if we should continue fetching
    // Continue if: we got a full page AND (transfer limit was exceeded OR we haven't hit max records)
    const gotFullPage = result.data.length === pageSize
    const shouldContinue = gotFullPage && (result.exceededTransferLimit || allData.length < maxRecordsLimit)
    
    if (!shouldContinue) {
      hasMore = false
    } else {
      offset += result.data.length
    }
    
    // Safety check to prevent infinite loops
    if (allData.length >= maxRecordsLimit) {
      hasMore = false
    }
  }

  // Apply date filtering client-side if dateRange was provided
  let filteredData = allData
  if (dateRange?.start || dateRange?.end) {
    filteredData = allData.filter(record => {
      const recordDate = record.date
      if (dateRange.start && recordDate < dateRange.start.getTime()) {
        return false
      }
      if (dateRange.end && recordDate > dateRange.end.getTime()) {
        return false
      }
      return true
    })
  }

  return filteredData.slice(0, maxRecordsLimit)
}

/**
 * Aggregates port traffic data by port
 */
export function aggregatePortData(data: PortTrafficData[]): PortSummary[] {
  if (!data || !Array.isArray(data) || data.length === 0) {
    return []
  }

  const portMap = new Map<string, {
    portid: string
    portname: string
    country: string
    ISO3: string
    totalPortcalls: number
    totalImport: number
    totalExport: number
    dayCount: number
  }>()

  data.forEach(record => {
    const key = record.portid
    const existing = portMap.get(key) || {
      portid: record.portid,
      portname: record.portname,
      country: record.country,
      ISO3: record.ISO3,
      totalPortcalls: 0,
      totalImport: 0,
      totalExport: 0,
      dayCount: 0,
    }

    existing.totalPortcalls += record.portcalls
    existing.totalImport += record.import || 0
    existing.totalExport += record.export || 0
    existing.dayCount += 1

    portMap.set(key, existing)
  })

  return Array.from(portMap.values()).map(port => ({
    portid: port.portid,
    portname: port.portname,
    country: port.country,
    ISO3: port.ISO3,
    totalPortcalls: port.totalPortcalls,
    totalImport: port.totalImport,
    totalExport: port.totalExport,
    avgDailyPortcalls: port.dayCount > 0 ? port.totalPortcalls / port.dayCount : 0,
    avgDailyImport: port.dayCount > 0 ? port.totalImport / port.dayCount : 0,
    avgDailyExport: port.dayCount > 0 ? port.totalExport / port.dayCount : 0,
  }))
}

/**
 * Gets unique countries from data
 */
export function getUniqueCountries(data: PortTrafficData[]): string[] {
  const countries = new Set(data.map(d => d.country).filter(Boolean))
  return Array.from(countries).sort()
}

/**
 * Gets unique ports from data
 */
export function getUniquePorts(data: PortTrafficData[]): Array<{ portid: string; portname: string; country: string }> {
  const portMap = new Map<string, { portid: string; portname: string; country: string }>()
  
  data.forEach(d => {
    if (!portMap.has(d.portid)) {
      portMap.set(d.portid, {
        portid: d.portid,
        portname: d.portname,
        country: d.country,
      })
    }
  })
  
  return Array.from(portMap.values()).sort((a, b) => a.portname.localeCompare(b.portname))
}

/**
 * Filters data by cargo type
 */
export function filterByCargoType(
  data: PortTrafficData[],
  cargoType: 'container' | 'dry_bulk' | 'general_cargo' | 'roro' | 'tanker' | 'cargo' | 'all'
): PortTrafficData[] {
  if (!data || !Array.isArray(data)) {
    return []
  }
  
  if (cargoType === 'all') {
    return data
  }

  return data.map(record => {
    const filtered = { ...record }
    
    // Get cargo-specific values with proper type handling
    const portcallsKey = `portcalls_${cargoType}` as keyof PortTrafficData
    const importKey = `import_${cargoType}` as keyof PortTrafficData
    const exportKey = `export_${cargoType}` as keyof PortTrafficData
    
    // Filter portcalls
    filtered.portcalls = (record[portcallsKey] as number) || 0
    filtered.portcalls_cargo = (record[portcallsKey] as number) || 0
    
    // Filter imports
    filtered.import = (record[importKey] as number) || 0
    filtered.import_cargo = (record[importKey] as number) || 0
    
    // Filter exports
    filtered.export = (record[exportKey] as number) || 0
    filtered.export_cargo = (record[exportKey] as number) || 0
    
    return filtered
  }).filter(record => record.portcalls > 0 || record.import > 0 || record.export > 0)
}

