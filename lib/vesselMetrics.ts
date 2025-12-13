import { PortTrafficData } from '@/types/portTraffic'

export interface VesselMetrics {
  arrivals: number
  departures: number
  totalVesselMovements: number
  avgTimeInPort: number // in hours
  maxTimeInPort: number // in hours
  minTimeInPort: number // in hours
  cargoVolume: {
    total: number
    import: number
    export: number
    byType: {
      container: number
      dry_bulk: number
      general_cargo: number
      roro: number
      tanker: number
    }
  }
  vesselUtilization: number // percentage
  peakDays: number
}

/**
 * Calculates vessel arrivals and departures
 * In the data, portcalls represent both arrivals and departures
 */
export function calculateVesselMovements(data: PortTrafficData[]): {
  arrivals: number
  departures: number
  totalMovements: number
} {
  // Each port call represents both an arrival and departure
  const totalPortcalls = data.reduce((sum, d) => sum + d.portcalls, 0)
  
  // For daily data, we can assume arrivals = departures = portcalls
  // (vessels that arrive also depart on the same day or next day)
  return {
    arrivals: totalPortcalls,
    departures: totalPortcalls,
    totalMovements: totalPortcalls * 2, // Arrivals + Departures
  }
}

/**
 * Estimates time-in-port based on consecutive days with port calls
 * This is an approximation since we don't have exact arrival/departure times
 */
export function estimateTimeInPort(data: PortTrafficData[]): {
  avg: number
  max: number
  min: number
  distribution: Array<{ hours: number; count: number }>
} {
  if (data.length === 0) {
    return { avg: 0, max: 0, min: 0, distribution: [] }
  }

  // Sort by date
  const sorted = [...data].sort((a, b) => a.date - b.date)
  
  // Group by port to analyze consecutive days
  const portGroups = new Map<string, PortTrafficData[]>()
  sorted.forEach(d => {
    if (!portGroups.has(d.portid)) {
      portGroups.set(d.portid, [])
    }
    portGroups.get(d.portid)!.push(d)
  })

  const timeInPortValues: number[] = []
  const distribution = new Map<number, number>()

  portGroups.forEach((portData) => {
    // For each day with port calls, estimate time in port
    portData.forEach((day, index) => {
      if (day.portcalls > 0) {
        // Estimate: if there are port calls on consecutive days, vessel stayed longer
        const nextDay = portData[index + 1]
        const prevDay = portData[index - 1]
        
        let estimatedHours = 12 // Default: half a day (12 hours)
        
        if (nextDay && nextDay.portcalls > 0) {
          // Vessel likely stayed overnight
          estimatedHours = 24
        } else if (prevDay && prevDay.portcalls > 0) {
          // Vessel arrived previous day
          estimatedHours = 18
        }
        
        // Adjust based on cargo volume (more cargo = longer stay)
        const cargoVolume = (day.import || 0) + (day.export || 0)
        if (cargoVolume > 10000) {
          estimatedHours += 6 // Large cargo = longer stay
        } else if (cargoVolume > 1000) {
          estimatedHours += 3
        }
        
        // Cap at reasonable maximum (72 hours = 3 days)
        estimatedHours = Math.min(estimatedHours, 72)
        
        timeInPortValues.push(estimatedHours)
        
        // Round to nearest 6 hours for distribution
        const rounded = Math.round(estimatedHours / 6) * 6
        distribution.set(rounded, (distribution.get(rounded) || 0) + day.portcalls)
      }
    })
  })

  if (timeInPortValues.length === 0) {
    return { avg: 0, max: 0, min: 0, distribution: [] }
  }

  const avg = timeInPortValues.reduce((a, b) => a + b, 0) / timeInPortValues.length
  const max = Math.max(...timeInPortValues)
  const min = Math.min(...timeInPortValues)

  return {
    avg,
    max,
    min,
    distribution: Array.from(distribution.entries())
      .map(([hours, count]) => ({ hours, count }))
      .sort((a, b) => a.hours - b.hours),
  }
}

/**
 * Calculates comprehensive vessel metrics
 */
export function calculateVesselMetrics(data: PortTrafficData[]): VesselMetrics {
  const movements = calculateVesselMovements(data)
  const timeInPort = estimateTimeInPort(data)
  
  // Calculate cargo volumes
  const totalImport = data.reduce((sum, d) => sum + (d.import || 0), 0)
  const totalExport = data.reduce((sum, d) => sum + (d.export || 0), 0)
  const totalCargo = totalImport + totalExport

  // Calculate cargo by type
  const cargoByType = {
    container: data.reduce((sum, d) => sum + (d.import_container || 0) + (d.export_container || 0), 0),
    dry_bulk: data.reduce((sum, d) => sum + (d.import_dry_bulk || 0) + (d.export_dry_bulk || 0), 0),
    general_cargo: data.reduce((sum, d) => sum + (d.import_general_cargo || 0) + (d.export_general_cargo || 0), 0),
    roro: data.reduce((sum, d) => sum + (d.import_roro || 0) + (d.export_roro || 0), 0),
    tanker: data.reduce((sum, d) => sum + (d.import_tanker || 0) + (d.export_tanker || 0), 0),
  }

  // Calculate vessel utilization (days with port calls / total days)
  const uniqueDays = new Set(data.map(d => d.date)).size
  const daysWithPortcalls = data.filter(d => d.portcalls > 0).length
  const vesselUtilization = uniqueDays > 0 ? (daysWithPortcalls / uniqueDays) * 100 : 0

  // Calculate peak days (days with above-average port calls)
  const avgPortcallsPerDay = movements.arrivals / uniqueDays
  const peakDays = data.filter(d => d.portcalls > avgPortcallsPerDay * 1.5).length

  return {
    arrivals: movements.arrivals,
    departures: movements.departures,
    totalVesselMovements: movements.totalMovements,
    avgTimeInPort: timeInPort.avg,
    maxTimeInPort: timeInPort.max,
    minTimeInPort: timeInPort.min,
    cargoVolume: {
      total: totalCargo,
      import: totalImport,
      export: totalExport,
      byType: cargoByType,
    },
    vesselUtilization,
    peakDays,
  }
}

/**
 * Calculates metrics for a specific port
 */
export function calculatePortVesselMetrics(
  portId: string,
  data: PortTrafficData[]
): VesselMetrics {
  const portData = data.filter(d => d.portid === portId)
  return calculateVesselMetrics(portData)
}

