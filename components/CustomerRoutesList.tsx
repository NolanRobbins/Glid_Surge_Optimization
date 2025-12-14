'use client'

import React, { useState, useMemo } from 'react'
import { Route, RoutePoint, calculateDistance } from '@/lib/routeCalculation'
import { calculateTonMileCost, calculateCompetitorCost, PRICING } from '@/lib/pricing'
import { getSyntheticRoutePoints } from '@/lib/syntheticDataService'
import { PortTrafficData } from '@/types/portTraffic'
import { PortDatabaseEntry } from '@/api/portDatabaseService'
import RouteOptionCard from './RouteOptionCard'

export interface CustomerRoute {
  id: string
  containerNumber: string
  shipmentDate: Date
  originPort: string
  destinationPort: string
  storageFacility?: string // Storage facility name/address
  status: 'scheduled' | 'in_transit' | 'completed' // Simplified statuses for industrial park routes
  route?: Route // Optional optimized route data if available
  competitorRoute?: Route // Optional competitor route data for comparison
  estimatedDelivery?: Date
  vehicleType?: 'raden' | 'gliderm' | 'ship' | 'truck' | 'train' | 'plane'
  vehicleImage?: string // Optional custom image URL
  containerWeight?: number // Weight in tons for cost calculation
  pickupTime?: Date // Actual pickup time at origin
  dropoffTime?: Date // Actual dropoff time at destination
  estimatedPickupTime?: Date // Estimated pickup time at origin
  estimatedDropoffTime?: Date // Estimated dropoff time at destination
}

interface CustomerRoutesListProps {
  routes: CustomerRoute[]
  onRouteSelect?: (route: CustomerRoute) => void
  trafficData?: PortTrafficData[] // Optional, not used for industrial park routes
  allPorts?: Map<string, PortDatabaseEntry> // Optional, not used for industrial park routes
}

export default function CustomerRoutesList({ routes, onRouteSelect }: CustomerRoutesListProps) {
  const [selectedRoute, setSelectedRoute] = useState<CustomerRoute | null>(null)
  const [showOptimizedRoutes, setShowOptimizedRoutes] = useState(false)
  
  // Get route points for distance calculation
  const routePoints = useMemo(() => getSyntheticRoutePoints(), [])
  
  // Helper to find route point by name
  const findRoutePoint = (name: string): RoutePoint | null => {
    const normalizedName = name.toLowerCase().trim()
    for (const point of Object.values(routePoints)) {
      const pointName = point.name?.toLowerCase().trim() || ''
      const pointId = point.id?.toLowerCase().trim() || ''
      // Match by name or ID
      if (pointName === normalizedName || 
          pointId === normalizedName.replace(/\s+/g, '-') ||
          normalizedName.includes(pointName) ||
          pointName.includes(normalizedName)) {
        return point
      }
    }
    return null
  }
  
  // Get route distance in miles
  const getRouteDistance = (route: CustomerRoute): number | null => {
    // Try to get distance from route object
    if (route.route?.distance) {
      return route.route.distance * 0.621371 // Convert km to miles
    }
    
    // Calculate distance from route points if available
    const originPoint = findRoutePoint(route.originPort)
    const destPoint = findRoutePoint(route.storageFacility || route.destinationPort)
    
    if (originPoint && destPoint) {
      const distanceKm = calculateDistance(originPoint.coordinates, destPoint.coordinates)
      return distanceKm * 0.621371 // Convert km to miles
    }
    
    return null
  }
  
  // Calculate costs for a route
  const calculateRouteCosts = (route: CustomerRoute) => {
<<<<<<< HEAD
    let distanceKm = 0
=======
>>>>>>> 9e835cc (new components to handle api responses)
    let ourDistanceKm = 0
    let competitorDistanceKm = 0
    
    // Get our route distance
    if (route.route?.distance) {
<<<<<<< HEAD
      distanceKm = route.route.distance
=======
>>>>>>> 9e835cc (new components to handle api responses)
      ourDistanceKm = route.route.distance
    } else {
      // Calculate distance from route points if available
      const originPoint = findRoutePoint(route.originPort)
      const destPoint = findRoutePoint(route.storageFacility || route.destinationPort)
      
      if (originPoint && destPoint) {
<<<<<<< HEAD
        distanceKm = calculateDistance(originPoint.coordinates, destPoint.coordinates)
=======
>>>>>>> 9e835cc (new components to handle api responses)
        ourDistanceKm = calculateDistance(originPoint.coordinates, destPoint.coordinates)
      }
    }

    // Get competitor route distance (use road-based distance if available)
    if (route.competitorRoute?.distance) {
      competitorDistanceKm = route.competitorRoute.distance
    } else if (route.competitorRoute?.coordinates && route.competitorRoute.coordinates.length > 0) {
      // Calculate path distance from competitor route coordinates
      let totalDistance = 0
      for (let i = 0; i < route.competitorRoute.coordinates.length - 1; i++) {
        totalDistance += calculateDistance(
          route.competitorRoute.coordinates[i],
          route.competitorRoute.coordinates[i + 1]
        )
      }
      competitorDistanceKm = totalDistance
    } else {
      // Fallback to our route distance if competitor route distance not available
      competitorDistanceKm = ourDistanceKm
    }
    
    // Get competitor route distance (use road-based distance if available)
    if (route.competitorRoute?.distance) {
      competitorDistanceKm = route.competitorRoute.distance
    } else if (route.competitorRoute?.coordinates && route.competitorRoute.coordinates.length > 0) {
      // Calculate path distance from competitor route coordinates
      let totalDistance = 0
      for (let i = 0; i < route.competitorRoute.coordinates.length - 1; i++) {
        totalDistance += calculateDistance(
          route.competitorRoute.coordinates[i],
          route.competitorRoute.coordinates[i + 1]
        )
      }
      competitorDistanceKm = totalDistance
    } else {
      // Fallback to our route distance if competitor route distance not available
      competitorDistanceKm = ourDistanceKm
    }
    
    // Convert to miles
<<<<<<< HEAD
    const distanceMiles = distanceKm * 0.621371
=======
>>>>>>> 9e835cc (new components to handle api responses)
    const ourDistanceMiles = ourDistanceKm * 0.621371
    const competitorDistanceMiles = competitorDistanceKm * 0.621371
    
    // Use container weight or default to 15 tons (average container weight)
    const weightTons = route.containerWeight || 15
    
    // Only calculate if we have distance
    if (ourDistanceMiles <= 0) {
      return null
    }
    
    // Calculate our cost (using standard pricing)
    const ourCost = calculateTonMileCost(ourDistanceMiles, weightTons, PRICING.STANDARD)
    
<<<<<<< HEAD
    // Calculate competitor cost
=======
    // Calculate competitor cost using competitor route distance
>>>>>>> 9e835cc (new components to handle api responses)
    const competitorCost = calculateCompetitorCost(competitorDistanceMiles, weightTons)
    
    // Calculate savings
    const savings = competitorCost - ourCost
    const savingsPercent = ((savings / competitorCost) * 100).toFixed(0)
    
    return {
      ourCost,
      competitorCost,
      savings,
      savingsPercent,
      distanceMiles: ourDistanceMiles,
      competitorDistanceMiles,
      weightTons,
    }
  }
  
  const formatDate = (date: Date) => {
    return new Intl.DateTimeFormat('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
    }).format(date)
  }

  const formatDateTime = (date: Date) => {
    return new Intl.DateTimeFormat('en-US', {
      month: 'short',
      day: 'numeric',
      hour: 'numeric',
      minute: '2-digit',
      hour12: true,
    }).format(date)
  }

  const formatTime = (date: Date) => {
    return new Intl.DateTimeFormat('en-US', {
      hour: 'numeric',
      minute: '2-digit',
      hour12: true,
    }).format(date)
  }

  const getStatusColor = (status: CustomerRoute['status']) => {
    switch (status) {
      case 'completed':
        return 'bg-green-100 text-green-800 ring-green-600/20'
      case 'in_transit':
        return 'bg-blue-100 text-blue-800 ring-blue-600/20'
      case 'scheduled':
        return 'bg-gray-100 text-gray-800 ring-gray-600/20'
      default:
        return 'bg-gray-100 text-gray-800 ring-gray-600/20'
    }
  }

  const getStatusLabel = (status: CustomerRoute['status']) => {
    switch (status) {
      case 'completed':
        return 'Completed'
      case 'in_transit':
        return 'In Transit'
      case 'scheduled':
        return 'Scheduled'
      default:
        return status
    }
  }

  const getVehicleImageSrc = (vehicleType?: string): string | null => {
    if (!vehicleType) return null
    
    const type = vehicleType.toLowerCase()
    
    // Try common image extensions for Rāden and GlīderM
    if (type === 'raden') {
      // Try different extensions - adjust based on actual file extension
      return '/raden.png' // Change to .jpg, .jpeg, or .webp if needed
    }
    
    if (type === 'gliderm' || type === 'gliderm') {
      return '/gliderm.png' // Change to .jpg, .jpeg, or .webp if needed
    }
    
    return null
  }

  const getVehicleIcon = (vehicleType?: string) => {
    switch (vehicleType) {
      case 'ship':
        return (
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="text-blue-600">
            <path d="M3 21h18" />
            <path d="M5 21V7l8-4v18" />
            <path d="M19 21V11l-6-4" />
            <path d="M9 9v.01" />
            <path d="M9 12v.01" />
            <path d="M9 15v.01" />
            <path d="M9 18v.01" />
          </svg>
        )
      case 'truck':
        return (
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="text-green-600">
            <path d="M16 3h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-2" />
            <path d="M4 19h2" />
            <path d="M2 3h4v12h10V3h4" />
            <circle cx="6" cy="18" r="2" />
            <circle cx="20" cy="18" r="2" />
          </svg>
        )
      case 'train':
        return (
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="text-purple-600">
            <rect x="4" y="3" width="16" height="12" rx="2" />
            <path d="M4 7h16" />
            <path d="M12 15v4" />
            <path d="M8 19h8" />
            <circle cx="8" cy="19" r="2" />
            <circle cx="16" cy="19" r="2" />
          </svg>
        )
      case 'plane':
        return (
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="text-orange-600">
            <path d="M17.8 19.2L16 11l3.5-3.5C21 6 21.5 4 21 3c-1-.5-3 0-4.5 1.5L13 8 4.8 6.2c-.5-.1-.9.1-1.1.5l-.3.5c-.2.5-.1 1 .3 1.3L9 12l-2 3H4l-1 1 3 2 2 3 1-1v-3l3-2 3.5 5.3c.3.4.8.5 1.3.3l.5-.2c.4-.3.6-.7.5-1.2z" />
          </svg>
        )
      default:
        return (
          <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" className="text-gray-400">
            <rect x="3" y="3" width="18" height="18" rx="2" />
            <path d="M3 9h18" />
            <path d="M9 21V9" />
          </svg>
        )
    }
  }

  // Show detail view if a route is selected
  if (selectedRoute) {
    return (
      <div className="space-y-4">
        {/* Back Button */}
        <button
          onClick={() => setSelectedRoute(null)}
          className="flex items-center gap-2 text-sm text-gray-600 hover:text-gray-900 transition-colors"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
          </svg>
          <span>Back to Routes</span>
        </button>

        {/* Route Details */}
        <div className="space-y-6">
          {/* Header */}
          <div className="flex items-start justify-between">
            <div className="flex-1">
              <div className="flex items-center gap-3 mb-2">
                <div className="flex-shrink-0 w-12 h-12 flex items-center justify-center bg-gray-50 rounded-lg border border-gray-200 overflow-hidden">
                  {selectedRoute.vehicleImage ? (
                    <img 
                      src={selectedRoute.vehicleImage} 
                      alt={selectedRoute.vehicleType || 'Vehicle'} 
                      className="w-full h-full object-cover"
                    />
                  ) : getVehicleImageSrc(selectedRoute.vehicleType) ? (
                    <img 
                      src={getVehicleImageSrc(selectedRoute.vehicleType)!} 
                      alt={selectedRoute.vehicleType || 'Vehicle'} 
                      className="w-full h-full object-cover"
                    />
                  ) : (
                    <div className="scale-75">
                      {getVehicleIcon(selectedRoute.vehicleType)}
                    </div>
                  )}
                </div>
                <div>
                  <h2 className="text-lg font-semibold text-gray-900">
                    {selectedRoute.containerNumber}
                  </h2>
                  {selectedRoute.vehicleType && (
                    <p className="text-sm text-gray-600">
                      {selectedRoute.vehicleType === 'raden' ? 'Raden' : selectedRoute.vehicleType === 'gliderm' ? 'GlīderM' : selectedRoute.vehicleType || 'Unknown'}
                    </p>
                  )}
                </div>
                <span
                  className={`inline-flex items-center rounded-full px-3 py-1 text-xs font-medium ring-1 ring-inset ${getStatusColor(
                    selectedRoute.status
                  )}`}
                >
                  {getStatusLabel(selectedRoute.status)}
                </span>
              </div>
            </div>
          </div>

              {/* Route Information */}
              <div className="space-y-4">
                <div className="flex items-center gap-2 text-sm text-gray-700">
                  <svg className="h-4 w-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
                  </svg>
                  <span className="font-medium">{selectedRoute.originPort}</span>
                  <span className="text-gray-400">→</span>
                  <span className="font-medium">{selectedRoute.storageFacility || selectedRoute.destinationPort}</span>
                </div>
                {(() => {
                  const distance = getRouteDistance(selectedRoute)
                  if (distance !== null) {
                    return (
                      <div className="flex items-center gap-2 text-sm text-gray-600">
                        <svg className="h-4 w-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7" />
                        </svg>
                        <span className="font-medium">Distance: {distance.toFixed(1)} miles</span>
                      </div>
                    )
                  }
                  return null
                })()}
                {selectedRoute.containerWeight && (
                  <div className="flex items-center gap-2 text-sm text-gray-600">
                    <svg className="h-4 w-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" />
                    </svg>
                    <span className="font-medium">Weight: {selectedRoute.containerWeight.toFixed(1)} tons</span>
                  </div>
                )}
              </div>

          {/* Route Details */}
          <div className="pt-4 border-t border-gray-200 space-y-4">
            <div className="space-y-2 text-sm">
              <div className="flex items-center gap-2 text-gray-600">
                <svg className="h-4 w-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                </svg>
                <span>Shipment Date: {formatDate(selectedRoute.shipmentDate)}</span>
              </div>
              {selectedRoute.pickupTime && (
                <div className="flex items-center gap-2 text-gray-600">
                  <svg className="h-4 w-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <span>Pickup Time: {formatDateTime(selectedRoute.pickupTime)}</span>
                </div>
              )}
              {selectedRoute.estimatedPickupTime && !selectedRoute.pickupTime && (
                <div className="flex items-center gap-2 text-gray-600">
                  <svg className="h-4 w-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <span>Estimated Pickup Time: {formatDateTime(selectedRoute.estimatedPickupTime)}</span>
                </div>
              )}
              {selectedRoute.dropoffTime && (
                <div className="flex items-center gap-2 text-gray-600">
                  <svg className="h-4 w-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                  <span>Dropoff Time: {formatDateTime(selectedRoute.dropoffTime)}</span>
                </div>
              )}
              {selectedRoute.estimatedDropoffTime && !selectedRoute.dropoffTime && (
                <div className="flex items-center gap-2 text-gray-600">
                  <svg className="h-4 w-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                  <span>Estimated Dropoff Time: {formatDateTime(selectedRoute.estimatedDropoffTime)}</span>
                </div>
              )}
              {selectedRoute.estimatedDelivery && (
                <div className="flex items-center gap-2 text-gray-600">
                  <svg className="h-4 w-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <span>Estimated Delivery: {formatDate(selectedRoute.estimatedDelivery)}</span>
                </div>
              )}
            </div>
            
            {/* Generate Optimized Routes Button */}
            <div className="pt-4 border-t border-gray-200">
              <button
                type="button"
                onClick={() => {
                  setShowOptimizedRoutes(!showOptimizedRoutes)
                }}
                className="w-full px-4 py-2 bg-black text-white text-sm font-medium rounded-lg hover:bg-gray-900 transition-colors"
              >
                Generate optimized routes
              </button>
              
<<<<<<< HEAD
              return (
                <div className="pt-4 border-t border-gray-200">
                  {/* Generate Optimized Routes Button */}
                  <button
                    type="button"
                    onClick={() => {
                      setShowOptimizedRoutes(!showOptimizedRoutes)
                    }}
                    className="w-full px-4 py-2 bg-black text-white text-sm font-medium rounded-lg hover:bg-gray-900 transition-colors mb-4"
                  >
                    Generate optimized routes
                  </button>

                  {/* Route Option Cards */}
                  {showOptimizedRoutes && (
                    <div className="mb-4 space-y-4">
                      <RouteOptionCard
                        title="Standard Route"
                        description="Scheduled pickup time"
                        distance={24.2}
                        estimatedTime={0.5}
                        cost={164.52}
                        optimizationLevel="good"
                        departureTime="Dec 13, 5:31 PM"
                        arrivalTime="Dec 13, 6:00 PM"
                      />
                      <RouteOptionCard
                        title="Express Route"
                        description="Faster delivery with priority handling"
                        distance={22.8}
                        estimatedTime={0.4}
                        cost={198.75}
                        optimizationLevel="optimal"
                        departureTime="Dec 13, 5:15 PM"
                        arrivalTime="Dec 13, 5:39 PM"
                      />
                      <RouteOptionCard
                        title="Economy Route"
                        description="Most cost-effective option"
                        distance={26.5}
                        estimatedTime={0.7}
                        cost={142.2}
                        optimizationLevel="standard"
                        departureTime="Dec 13, 5:45 PM"
                        arrivalTime="Dec 13, 6:27 PM"
                      />
                    </div>
                  )}

                  <h3 className="text-sm font-semibold text-gray-900 mb-3">Cost Estimate</h3>
                  <div className="flex items-center gap-4">
                    <div className="flex-1 p-3 bg-green-50 rounded-lg border border-green-200">
                      <div>
                        <span className="text-xs font-medium text-gray-700">Our Cost</span>
                        <div className="text-lg font-bold text-green-700 mt-0.5">${costs.ourCost.toFixed(2)}</div>
                        <div className="text-xs text-gray-500 mt-0.5">
                          {costs.distanceMiles.toFixed(1)} mi × {costs.weightTons.toFixed(1)} tons × ${PRICING.STANDARD.toFixed(2)}/ton-mi
                        </div>
                      </div>
                    </div>
                    <div className="flex-1 p-3 bg-gray-50 rounded-lg border border-gray-200">
                      <div>
                        <span className="text-xs font-medium text-gray-700">Competitor</span>
                        <div className="text-lg font-bold text-gray-600 mt-0.5 line-through">${costs.competitorCost.toFixed(2)}</div>
                        <div className="text-xs text-gray-500 mt-0.5">
                          {costs.competitorDistanceMiles.toFixed(1)} mi × {costs.weightTons.toFixed(1)} tons × ${PRICING.COMPETITOR.toFixed(2)}/ton-mi
                        </div>
                      </div>
                    </div>
                    <div className="flex-1 p-3 bg-blue-50 rounded-lg border border-blue-200">
                      <div>
                        <span className="text-xs font-medium text-gray-700">You Save</span>
                        <div className="text-lg font-bold text-blue-700 mt-0.5">
                          ${costs.savings.toFixed(2)}
                        </div>
                        <div className="text-sm font-semibold text-blue-600 mt-0.5">
                          {costs.savingsPercent}% less than competitors
                        </div>
                      </div>
                    </div>
                  </div>
=======
              {/* Route Option Cards */}
              {showOptimizedRoutes && (
                <div className="mt-4 space-y-4">
                  <RouteOptionCard
                    title="Standard Route"
                    description="Scheduled pickup time"
                    distance={24.2}
                    estimatedTime={0.5}
                    cost={164.52}
                    optimizationLevel="good"
                    departureTime="Dec 13, 5:31 PM"
                    arrivalTime="Dec 13, 6:00 PM"
                  />
                  <RouteOptionCard
                    title="Express Route"
                    description="Faster delivery with priority handling"
                    distance={22.8}
                    estimatedTime={0.4}
                    cost={198.75}
                    optimizationLevel="optimal"
                    departureTime="Dec 13, 5:15 PM"
                    arrivalTime="Dec 13, 5:39 PM"
                  />
                  <RouteOptionCard
                    title="Economy Route"
                    description="Most cost-effective option"
                    distance={26.5}
                    estimatedTime={0.7}
                    cost={142.20}
                    optimizationLevel="standard"
                    departureTime="Dec 13, 5:45 PM"
                    arrivalTime="Dec 13, 6:27 PM"
                  />
>>>>>>> 9e835cc (new components to handle api responses)
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-3">
      {routes.length === 0 ? (
        <div className="text-center py-12">
          <svg
            className="mx-auto h-12 w-12 text-gray-400"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            aria-hidden="true"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
            />
          </svg>
          <h3 className="mt-2 text-sm font-semibold text-gray-900">No routes</h3>
          <p className="mt-1 text-sm text-gray-500">Get started by creating a new route.</p>
        </div>
      ) : (
        routes.map((route) => (
          <div
            key={route.id}
            onClick={() => {
              setSelectedRoute(route)
              // Always call onRouteSelect to update the map
              onRouteSelect?.(route)
            }}
            className={`relative rounded-lg border border-gray-200 bg-white p-3 shadow-sm hover:shadow-md transition-shadow cursor-pointer ${
              onRouteSelect ? 'hover:border-gray-300' : ''
            }`}
          >
            <div className="flex flex-col gap-2">
              {/* First Row: Image, Shipment Number, Status */}
              <div className="flex items-center gap-3">
                {/* Vehicle Image */}
                <div className="flex-shrink-0 w-12 h-12 flex items-center justify-center bg-gray-50 rounded-lg border border-gray-200 overflow-hidden">
                  {route.vehicleImage ? (
                    <img 
                      src={route.vehicleImage} 
                      alt={route.vehicleType || 'Vehicle'} 
                      className="w-full h-full object-cover"
                    />
                  ) : getVehicleImageSrc(route.vehicleType) ? (
                    <img 
                      src={getVehicleImageSrc(route.vehicleType)!} 
                      alt={route.vehicleType || 'Vehicle'} 
                      className="w-full h-full object-cover"
                    />
                  ) : (
                    <div className="scale-75">
                      {getVehicleIcon(route.vehicleType)}
                    </div>
                  )}
                </div>
                
                {/* Container Number and Vehicle Name */}
                <div className="flex-1 min-w-0">
                  <h3 className="text-sm font-semibold text-gray-900 truncate">
                    {route.containerNumber}
                  </h3>
                  {/* Vehicle Name */}
                  {route.vehicleType && (
                    <div className="text-xs text-gray-600 truncate mt-0.5">
                      {route.vehicleType === 'raden' ? 'Raden' : route.vehicleType === 'gliderm' ? 'GlīderM' : route.vehicleType || 'Unknown'}
                    </div>
                  )}
                </div>
                
                {/* Status */}
                <span
                  className={`inline-flex items-center rounded-full px-2 py-1 text-xs font-medium ring-1 ring-inset ${getStatusColor(
                    route.status
                  )}`}
                >
                  {getStatusLabel(route.status)}
                </span>
              </div>

              {/* Route Information */}
              <div className="space-y-2 pt-2">
                <div className="flex items-center gap-2 text-sm text-gray-700">
                  <svg className="h-4 w-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
                  </svg>
                  <span className="font-medium">{route.originPort}</span>
                  <span className="text-gray-400">→</span>
                  <span className="font-medium">{route.storageFacility || route.destinationPort}</span>
                </div>
                {(() => {
                  const distance = getRouteDistance(route)
                  if (distance !== null) {
                    return (
                      <div className="flex items-center gap-2 text-xs text-gray-600">
                        <svg className="h-3.5 w-3.5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7" />
                        </svg>
                        <span className="font-medium">Distance: {distance.toFixed(1)} mi</span>
                      </div>
                    )
                  }
                  return null
                })()}
                {route.containerWeight && (
                  <div className="flex items-center gap-2 text-xs text-gray-600">
                    <svg className="h-3.5 w-3.5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 7l-8-4-8 4m16 0l-8 4m8-4v10l-8 4m0-10L4 7m8 4v10M4 7v10l8 4" />
                    </svg>
                    <span className="font-medium">Weight: {route.containerWeight.toFixed(1)} tons</span>
                  </div>
                )}
                {(route.pickupTime || route.estimatedPickupTime) && (
                  <div className="flex items-center gap-2 text-xs text-gray-600">
                    <svg className="h-3.5 w-3.5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <span>
                      {route.pickupTime ? `Pickup: ${formatTime(route.pickupTime)}` : `Est. Pickup: ${formatTime(route.estimatedPickupTime!)}`}
                    </span>
                  </div>
                )}
                {(route.dropoffTime || route.estimatedDropoffTime) && (
                  <div className="flex items-center gap-2 text-xs text-gray-600">
                    <svg className="h-3.5 w-3.5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                    </svg>
                    <span>
                      {route.dropoffTime ? `Dropoff: ${formatTime(route.dropoffTime)}` : `Est. Dropoff: ${formatTime(route.estimatedDropoffTime!)}`}
                    </span>
                  </div>
                )}
                {route.estimatedDelivery && (
                  <div className="flex items-center gap-2 text-xs text-gray-500">
                    <svg className="h-3.5 w-3.5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
                    </svg>
                    <span>Est. Delivery: {formatDate(route.estimatedDelivery)}</span>
                  </div>
                )}
                
                {/* Cost Estimates */}
                {(() => {
                  const costs = calculateRouteCosts(route)
                  if (!costs) return null
                  
                  return (
                    <div className="pt-2 border-t border-gray-100">
                      <div className="flex items-center gap-4">
                        <div className="flex flex-col">
                          <span className="text-xs font-medium text-gray-700">Our Cost</span>
                          <span className="text-sm font-semibold text-green-600">${costs.ourCost.toFixed(2)}</span>
                        </div>
                        <div className="flex flex-col">
                          <span className="text-xs font-medium text-gray-700">Competitor</span>
                          <span className="text-sm text-gray-600 line-through">${costs.competitorCost.toFixed(2)}</span>
                        </div>
                        <div className="flex flex-col">
                          <span className="text-xs font-medium text-gray-700">You Save</span>
                          <span className="text-sm font-semibold text-green-700">
                            ${costs.savings.toFixed(2)} ({costs.savingsPercent}%)
                          </span>
                        </div>
                      </div>
                    </div>
                  )
                })()}
              </div>
            </div>
          </div>
        ))
      )}
    </div>
  )
}

