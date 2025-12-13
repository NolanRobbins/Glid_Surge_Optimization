/**
 * Service for loading synthetic customer routes data
 */

import { CustomerRoute } from '@/components/CustomerRoutesList'
import { RoutePoint } from '@/lib/routeCalculation'
import { calculateDistance } from '@/lib/routeCalculation'
import syntheticData from '@/data/syntheticCustomerRoutes.json'

interface SyntheticRouteData {
  id: string
  containerNumber: string
  shipmentDate: string
  originPort: string
  storageFacility?: string
  destinationPort: string
  status: 'scheduled' | 'in_transit' | 'completed'
  estimatedDelivery?: string
  vehicleType?: 'raden' | 'gliderm' | 'ship' | 'truck' | 'train' | 'plane'
  containerWeight?: number // Weight in tons
  pickupTime?: string // ISO date string
  dropoffTime?: string // ISO date string
  route?: {
    origin: RoutePoint
    destination: RoutePoint
    coordinates: [number, number][]
  }
}

interface SyntheticData {
  routes: SyntheticRouteData[]
  routePoints: Record<string, RoutePoint>
}

/**
 * Convert synthetic route data to CustomerRoute with proper Date objects and Route objects
 */
export function loadSyntheticCustomerRoutes(): CustomerRoute[] {
  const data = syntheticData as unknown as SyntheticData
  
  return data.routes.map((routeData): CustomerRoute => {
    const route: CustomerRoute = {
      id: routeData.id,
      containerNumber: routeData.containerNumber,
      shipmentDate: new Date(routeData.shipmentDate),
      originPort: routeData.originPort,
      destinationPort: routeData.destinationPort,
      status: routeData.status,
      vehicleType: routeData.vehicleType,
    }

    // Add optional fields
    if (routeData.storageFacility) {
      route.storageFacility = routeData.storageFacility
    }

    if (routeData.estimatedDelivery) {
      route.estimatedDelivery = new Date(routeData.estimatedDelivery)
    }

    if (routeData.containerWeight) {
      route.containerWeight = routeData.containerWeight
    }

    if (routeData.pickupTime) {
      route.pickupTime = new Date(routeData.pickupTime)
    }

    if (routeData.dropoffTime) {
      route.dropoffTime = new Date(routeData.dropoffTime)
    }

    if ('estimatedPickupTime' in routeData && routeData.estimatedPickupTime) {
      route.estimatedPickupTime = new Date(routeData.estimatedPickupTime as string)
    }

    if ('estimatedDropoffTime' in routeData && routeData.estimatedDropoffTime) {
      route.estimatedDropoffTime = new Date(routeData.estimatedDropoffTime as string)
    }

    // Convert route data if present
    if (routeData.route) {
      const origin = routeData.route.origin
      const destination = routeData.route.destination
      
      route.route = {
        origin,
        destination,
        coordinates: routeData.route.coordinates as [number, number][],
        distance: calculateDistance(origin.coordinates, destination.coordinates),
      }
    }

    // Convert competitor route data if present
    if ('competitorRoute' in routeData && routeData.competitorRoute) {
      const compRoute = routeData.competitorRoute as { origin: RoutePoint; destination: RoutePoint; coordinates: number[][] }
      const origin = compRoute.origin
      const destination = compRoute.destination
      
      route.competitorRoute = {
        origin,
        destination,
        coordinates: compRoute.coordinates as [number, number][],
        distance: calculateDistance(origin.coordinates, destination.coordinates),
      }
    }

    return route
  })
}

/**
 * Get route points from synthetic data
 */
export function getSyntheticRoutePoints(): Record<string, RoutePoint> {
  const data = syntheticData as unknown as SyntheticData
  return data.routePoints
}
