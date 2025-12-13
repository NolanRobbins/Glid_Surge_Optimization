/**
 * Port infrastructure configuration
 * Contains port-specific infrastructure data like cranes, berths, etc.
 */

export interface PortInfrastructure {
  portId?: string
  portName: string
  gantryCranes: number
  berths?: number
  containerTerminals?: number
  notes?: string
}

/**
 * Port infrastructure database
 * Maps port names/IDs to their infrastructure configuration
 */
export const PORT_INFRASTRUCTURE: Map<string, PortInfrastructure> = new Map([
  // Long Beach Port
  ['Long Beach', {
    portName: 'Long Beach',
    gantryCranes: 73,
    berths: 80,
    containerTerminals: 6,
    notes: 'One of the largest container ports in the US'
  }],
  ['Port of Long Beach', {
    portName: 'Port of Long Beach',
    gantryCranes: 73,
    berths: 80,
    containerTerminals: 6,
    notes: 'One of the largest container ports in the US'
  }],
  ['Los Angeles-Long Beach', {
    portName: 'Los Angeles-Long Beach',
    gantryCranes: 73,
    berths: 80,
    containerTerminals: 6,
    notes: 'Long Beach port within LA-LB complex'
  }],
])

/**
 * Get port infrastructure by port name or ID
 */
export function getPortInfrastructure(portNameOrId: string): PortInfrastructure | null {
  // Try exact match first
  if (PORT_INFRASTRUCTURE.has(portNameOrId)) {
    return PORT_INFRASTRUCTURE.get(portNameOrId) || null
  }
  
  // Try case-insensitive partial match
  const normalized = portNameOrId.toLowerCase()
  for (const [key, infra] of PORT_INFRASTRUCTURE.entries()) {
    if (key.toLowerCase().includes(normalized) || normalized.includes(key.toLowerCase())) {
      return infra
    }
  }
  
  return null
}

/**
 * Calculate container unloading rate based on port infrastructure
 * @param portNameOrId Port name or ID
 * @param defaultRate Default containers per hour per crane (default: 40)
 * @returns Total containers per hour that can be unloaded
 */
export function calculateUnloadingRate(portNameOrId: string, defaultRate: number = 40): number {
  const infra = getPortInfrastructure(portNameOrId)
  
  if (infra && infra.gantryCranes > 0) {
    // Each crane can handle approximately 40 containers per hour
    // With 73 cranes, we can unload much faster
    // However, not all cranes are available simultaneously (maintenance, scheduling, etc.)
    // Assume 80% utilization for realistic calculation
    const availableCranes = Math.floor(infra.gantryCranes * 0.8)
    return availableCranes * defaultRate
  }
  
  // Default: assume 1 crane if no infrastructure data
  return defaultRate
}

/**
 * Get number of available cranes for a port
 */
export function getAvailableCranes(portNameOrId: string, utilizationRate: number = 0.8): number {
  const infra = getPortInfrastructure(portNameOrId)
  
  if (infra && infra.gantryCranes > 0) {
    return Math.floor(infra.gantryCranes * utilizationRate)
  }
  
  return 1 // Default: 1 crane
}

/**
 * Calculate expected delay based on number of cranes and ships at port
 * @param portNameOrId Port name or ID
 * @param shipsAtPort Number of ships currently at the port
 * @param baseTimeInPort Base average time in port (hours) from service data
 * @param utilizationRate Crane utilization rate (default: 0.8 = 80%)
 * @returns Expected delay in hours (scaled based on crane capacity vs ship count)
 */
export function calculateExpectedDelay(
  portNameOrId: string,
  shipsAtPort: number,
  baseTimeInPort: number,
  utilizationRate: number = 0.8
): {
  expectedDelay: number
  delayMultiplier: number
  craneCapacity: number
  congestionLevel: 'low' | 'medium' | 'high' | 'critical'
} {
  const infra = getPortInfrastructure(portNameOrId)
  const totalCranes = infra?.gantryCranes || 1
  const availableCranes = Math.floor(totalCranes * utilizationRate)
  
  // Each crane can typically handle 1 ship at a time (simplified model)
  // In reality, a ship might use multiple cranes, but for delay calculation we use 1:1 ratio
  const craneCapacity = availableCranes
  
  // Calculate delay multiplier based on ship-to-crane ratio
  // If ships <= cranes: minimal delay (multiplier ~1.0)
  // If ships > cranes: exponential increase in delay
  let delayMultiplier = 1.0
  
  if (shipsAtPort <= craneCapacity) {
    // Under capacity: minimal delay, slight increase as we approach capacity
    delayMultiplier = 1.0 + ((shipsAtPort / craneCapacity) * 0.2) // 1.0 to 1.2
  } else {
    // Over capacity: significant delay increase
    const overCapacity = shipsAtPort - craneCapacity
    const overCapacityRatio = overCapacity / craneCapacity
    // Exponential scaling: 1.2x base for 1x over capacity, 2x for 2x over, etc.
    delayMultiplier = 1.2 + (overCapacityRatio * 0.8) // 1.2 to ~3.0+ for heavy congestion
  }
  
  // Cap maximum delay multiplier at 3.5x to prevent unrealistic values
  delayMultiplier = Math.min(delayMultiplier, 3.5)
  
  const expectedDelay = baseTimeInPort * delayMultiplier
  
  // Determine congestion level
  let congestionLevel: 'low' | 'medium' | 'high' | 'critical'
  if (shipsAtPort <= craneCapacity * 0.7) {
    congestionLevel = 'low'
  } else if (shipsAtPort <= craneCapacity) {
    congestionLevel = 'medium'
  } else if (shipsAtPort <= craneCapacity * 1.5) {
    congestionLevel = 'high'
  } else {
    congestionLevel = 'critical'
  }
  
  return {
    expectedDelay,
    delayMultiplier,
    craneCapacity,
    congestionLevel
  }
}
