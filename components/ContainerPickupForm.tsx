'use client'

import React, { useState, useEffect, useRef, useMemo } from 'react'
import { RoutePoint, Route, calculateDistance } from '@/lib/routeCalculation'
import { calculateUnloadingRate, calculateExpectedDelay } from '@/lib/portInfrastructure'
import { getDrivingRoute, getAlternativeRoutes, DirectionsRoute } from '@/lib/mapboxDirections'
import { CustomerRoute } from './CustomerRoutesList'
import { PortTrafficData } from '@/types/portTraffic'
import { calculatePortVesselMetrics } from '@/lib/vesselMetrics'
import { PortDatabaseEntry } from '@/api/portDatabaseService'
import { calculateTonMileCost, getPricingForRouteType } from '@/lib/pricing'
import RouteOptionCard from './RouteOptionCard'

// RouteOptionsList interfaces and types - we'll recreate the component inline
interface TrafficLevel {
  level: 'low' | 'medium' | 'high'
  label: string
  description: string
}

export interface RouteOption {
  id: string
  name: string
  description: string
  mileage: number // in miles
  cost: number // in dollars
  time: number // in hours
  departureTime: Date
  arrivalTime: Date
  trafficLevel: TrafficLevel
  routeCoordinates?: [number, number][] // Real route coordinates from Mapbox
  routeDistance?: number // Actual route distance in km
  routeDuration?: number // Actual route duration in seconds
}

interface RouteOptionsDisplayProps {
  originPort: string
  destinationPort: string
  vehiclePickupTime: Date
  estimatedTimeToStorage: Date
  containerArrivalAtPort?: Date
  avgTimeInPort: number
  route?: Route
  trafficData?: PortTrafficData[]
  allPorts?: Map<string, PortDatabaseEntry>
  totalContainersOnShip?: number
  containerPosition?: 'front' | 'middle' | 'end'
  containerWeight?: number // Weight in tons for cost calculation
  onRouteOptionSelect?: (routeOption: RouteOption) => void
}

// Component to display route options with forecasts
function RouteOptionsDisplay({
  originPort,
  vehiclePickupTime,
  estimatedTimeToStorage,
  containerArrivalAtPort,
  avgTimeInPort,
  route,
  trafficData = [],
  allPorts = new Map(),
  totalContainersOnShip = 1000,
  containerPosition = 'middle',
  containerWeight = 1, // Default to 1 ton if not provided
  onRouteOptionSelect,
}: RouteOptionsDisplayProps) {
  const formatDateTime = (date: Date) => {
    return new Intl.DateTimeFormat('en-US', {
      month: 'short',
      day: 'numeric',
      hour: 'numeric',
      minute: '2-digit',
      hour12: true,
    }).format(date)
  }

  // Find port ID from port name
  const findPortIdByName = (portName: string): string | null => {
    for (const [portId, port] of allPorts.entries()) {
      if (port.portname?.toLowerCase().includes(portName.toLowerCase()) ||
          portName.toLowerCase().includes(port.portname?.toLowerCase() || '')) {
        return portId
      }
    }
    return null
  }

  // Calculate container offload time
  const calculateContainerOffloadTime = (containerArrivalTime: Date, totalContainers: number, position: 'front' | 'middle' | 'end'): Date => {
    // Get port-specific unloading rate based on infrastructure (e.g., gantry cranes)
    // Long Beach has 73 gantry cranes, which significantly increases unloading capacity
    const unloadingRate = calculateUnloadingRate(originPort, 40) // 40 containers per hour per crane
    const totalUnloadingHours = totalContainers / unloadingRate
    
    let containerUnloadOffset = 0
    if (position === 'front') {
      containerUnloadOffset = totalUnloadingHours * 0.1
    } else if (position === 'middle') {
      containerUnloadOffset = totalUnloadingHours * 0.5
    } else if (position === 'end') {
      containerUnloadOffset = totalUnloadingHours * 0.9
    }
    
    return new Date(
      containerArrivalTime.getTime() + 
      (avgTimeInPort * 60 * 60 * 1000) +
      (containerUnloadOffset * 60 * 60 * 1000)
    )
  }

  // Get traffic level
  const getTrafficLevel = (pickupTime: Date): TrafficLevel => {
    const portId = findPortIdByName(originPort)
    if (!portId || trafficData.length === 0) {
      return { level: 'medium', label: 'Normal', description: 'Average traffic expected' }
    }

    const portTraffic = trafficData.filter(d => d.portid === portId)
    if (portTraffic.length === 0) {
      return { level: 'medium', label: 'Normal', description: 'Average traffic expected' }
    }

    const totalPortCalls = portTraffic.reduce((sum, d) => sum + (d.portcalls || 0), 0)
    const uniqueDays = new Set(portTraffic.map(d => d.date)).size
    const avgDailyPortCalls = uniqueDays > 0 ? totalPortCalls / uniqueDays : 0

    const hourOfDay = pickupTime.getHours()
    const isPeakHour = (hourOfDay >= 8 && hourOfDay < 12) || (hourOfDay >= 13 && hourOfDay < 17)
    const isLowHour = hourOfDay >= 22 || hourOfDay < 6
    
    let estimatedHourlyCalls: number
    if (isLowHour) {
      estimatedHourlyCalls = (avgDailyPortCalls / 24) * 0.5
    } else if (isPeakHour) {
      estimatedHourlyCalls = (avgDailyPortCalls / 24) * 1.5
    } else {
      estimatedHourlyCalls = avgDailyPortCalls / 24
    }

    const avgHourlyCalls = avgDailyPortCalls / 24
    if (estimatedHourlyCalls < avgHourlyCalls * 0.7) {
      return { level: 'low', label: 'Low Traffic', description: 'Light traffic expected' }
    } else if (estimatedHourlyCalls > avgHourlyCalls * 1.3) {
      return { level: 'high', label: 'High Traffic', description: 'Heavy traffic expected' }
    } else {
      return { level: 'medium', label: 'Normal', description: 'Average traffic expected' }
    }
  }

  // Get traffic indicator styling
  const getTrafficIndicatorStyle = (level: TrafficLevel['level']) => {
    switch (level) {
      case 'low':
        return 'bg-green-100 text-green-800 border-green-200'
      case 'high':
        return 'bg-red-100 text-red-800 border-red-200'
      default:
        return 'bg-gray-100 text-gray-800 border-gray-200'
    }
  }

  // Get traffic icon
  const getTrafficIcon = (level: TrafficLevel['level']) => {
    switch (level) {
      case 'low':
        return '✓'
      case 'high':
        return '⚠'
      default:
        return '○'
    }
  }

  // Calculate base distance - ensure we have a valid distance
  // Use a local copy to prevent mutations
  const routeDistance = route?.distance
  const baseDistanceKm = routeDistance && routeDistance > 0 && !isNaN(routeDistance) ? routeDistance : 50
  const baseDistanceMiles = baseDistanceKm * 0.621371
  
  // Calculate base transit time - ensure we have valid times
  const timeDiff = estimatedTimeToStorage.getTime() - vehiclePickupTime.getTime()
  const baseTransitHours = timeDiff > 0 ? timeDiff / (1000 * 60 * 60) : Math.max(0.1, baseDistanceMiles / 50) // Fallback: assume 50 mph if times are invalid
  
  // Debug logging
  console.log('RouteOptionsDisplay: Base calculations', {
    routeProp: route,
    routeDistance,
    routeDistanceKm: routeDistance,
    routeDistanceMiles: baseDistanceMiles,
    baseDistanceKm,
    baseDistanceMiles,
    timeDiff,
    timeDiffHours: timeDiff / (1000 * 60 * 60),
    baseTransitHours,
    vehiclePickupTime: vehiclePickupTime.toISOString(),
    estimatedTimeToStorage: estimatedTimeToStorage.toISOString(),
    hasRoute: !!route,
    routeOrigin: route?.origin,
    routeDestination: route?.destination,
    routeCoordinates: route?.coordinates?.length || 0,
    routeFull: route
  })

  // Get origin and destination coordinates from route
  const originCoords: [number, number] | null = route?.origin?.coordinates || null
  const destinationCoords: [number, number] | null = route?.destination?.coordinates || null

  // State to store fetched routes
  const [fetchedRoutes, setFetchedRoutes] = useState<Map<string, DirectionsRoute>>(new Map())
  
  // Fetch real routes for all options when route coordinates are available
  useEffect(() => {
    if (!originCoords || !destinationCoords) return

    const fetchRoutes = async () => {
      const routesMap = new Map<string, DirectionsRoute>()

      try {
        // Fetch standard route (fastest with traffic)
        const standardRoute = await getDrivingRoute(originCoords, destinationCoords, 'driving-traffic')
        if (standardRoute) {
          routesMap.set('standard', standardRoute)
        }

        // Fetch alternative routes for express and economy
        const alternatives = await getAlternativeRoutes(originCoords, destinationCoords, 'driving-traffic')
        if (alternatives.length > 0) {
          // Use the fastest alternative for express
          const fastestAlt = alternatives.sort((a, b) => a.duration - b.duration)[0]
          routesMap.set('express', fastestAlt)
          
          // Use the shortest alternative for economy
          const shortestAlt = alternatives.sort((a, b) => a.distance - b.distance)[0]
          routesMap.set('economy', shortestAlt)
        }

        // For early and late routes, use the same base route but with different timing
        if (standardRoute) {
          routesMap.set('early', standardRoute)
          routesMap.set('late', standardRoute)
        }
      } catch (error) {
        console.warn('Error fetching routes:', error)
      }

      setFetchedRoutes(routesMap)
    }

    fetchRoutes()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [originCoords, destinationCoords])
  
  // Generate route options
  const routeOptions: RouteOption[] = useMemo(() => {
    console.log('RouteOptionsDisplay: Generating route options with:', {
      baseDistanceKm,
      baseDistanceMiles,
      baseTransitHours,
      vehiclePickupTime: vehiclePickupTime.toISOString(),
      estimatedTimeToStorage: estimatedTimeToStorage.toISOString(),
      containerArrivalAtPort: containerArrivalAtPort?.toISOString(),
      hasRoute: !!route,
      routeDistance: route?.distance
    })
    
    const options: RouteOption[] = []
    
    if (!containerArrivalAtPort) {
      const standardOption = {
        id: 'standard',
        name: 'Standard Route',
        description: 'Scheduled pickup time',
        mileage: baseDistanceMiles,
        cost: calculateTonMileCost(baseDistanceMiles, containerWeight, getPricingForRouteType('standard')),
        time: baseTransitHours,
        departureTime: vehiclePickupTime,
        arrivalTime: estimatedTimeToStorage,
        trafficLevel: getTrafficLevel(vehiclePickupTime),
      }
      console.log('RouteOptionsDisplay: Standard option (no containerArrivalAtPort):', standardOption)
      options.push(standardOption)
      return options
    }
    
    const actualContainerOffShipTime = calculateContainerOffloadTime(
      containerArrivalAtPort,
      totalContainersOnShip,
      containerPosition
    )
    
    console.log('RouteOptionsDisplay: Container offload calculation:', {
      containerArrivalAtPort: containerArrivalAtPort.toISOString(),
      totalContainersOnShip,
      containerPosition,
      actualContainerOffShipTime: actualContainerOffShipTime.toISOString(),
      avgTimeInPort
    })
    
    // Early pickup option
    const earlyPickupTime = new Date(actualContainerOffShipTime.getTime() + 30 * 60 * 1000)
    const shouldIncludeEarly = earlyPickupTime <= vehiclePickupTime || earlyPickupTime.getTime() < vehiclePickupTime.getTime() + 2 * 60 * 60 * 1000
    console.log('RouteOptionsDisplay: Early option check:', {
      earlyPickupTime: earlyPickupTime.toISOString(),
      vehiclePickupTime: vehiclePickupTime.toISOString(),
      shouldIncludeEarly,
      timeDiff: earlyPickupTime.getTime() - vehiclePickupTime.getTime()
    })
    
    if (shouldIncludeEarly) {
      const earlyRoute = fetchedRoutes.get('early')
      const earlyDistanceKm = (earlyRoute && earlyRoute.distance && earlyRoute.distance > 0)
        ? earlyRoute.distance / 1000 
        : baseDistanceKm * 0.95
      const earlyDistanceMiles = earlyDistanceKm * 0.621371
      const earlyDurationHours = (earlyRoute && earlyRoute.duration && earlyRoute.duration > 0)
        ? earlyRoute.duration / 3600 
        : baseTransitHours * 0.9
      const earlyArrival = new Date(earlyPickupTime.getTime() + earlyDurationHours * 60 * 60 * 1000)
      
      console.log('RouteOptionsDisplay: Early option calculation:', {
        earlyRoute: earlyRoute ? { distance: earlyRoute.distance, duration: earlyRoute.duration } : null,
        earlyDistanceKm,
        earlyDistanceMiles,
        earlyDurationHours,
        containerWeight,
        cost: calculateTonMileCost(earlyDistanceMiles, containerWeight, getPricingForRouteType('early'))
      })
      
      options.push({
        id: 'early',
        name: 'Early Pickup',
        description: `Pickup ${Math.round((vehiclePickupTime.getTime() - earlyPickupTime.getTime()) / (1000 * 60))} min early`,
        mileage: earlyDistanceMiles,
        cost: calculateTonMileCost(earlyDistanceMiles, containerWeight, getPricingForRouteType('early')),
        time: earlyDurationHours,
        departureTime: earlyPickupTime,
        arrivalTime: earlyArrival,
        trafficLevel: getTrafficLevel(earlyPickupTime),
        routeCoordinates: earlyRoute?.geometry.coordinates,
        routeDistance: earlyDistanceKm,
        routeDuration: earlyRoute?.duration,
      })
    }
    
    // Standard option
    const standardPickupTime = actualContainerOffShipTime.getTime() > vehiclePickupTime.getTime() 
      ? actualContainerOffShipTime 
      : vehiclePickupTime
    const standardRoute = fetchedRoutes.get('standard')
    const standardDistanceKm = (standardRoute && standardRoute.distance && standardRoute.distance > 0) 
      ? standardRoute.distance / 1000 
      : baseDistanceKm
    const standardDistanceMiles = standardDistanceKm * 0.621371
    const standardTransitHours = (standardRoute && standardRoute.duration && standardRoute.duration > 0)
      ? standardRoute.duration / 3600 
      : baseTransitHours
    const standardArrival = new Date(standardPickupTime.getTime() + standardTransitHours * 60 * 60 * 1000)
    
    console.log('RouteOptionsDisplay: Standard option calculation:', {
      standardRoute: standardRoute ? { distance: standardRoute.distance, duration: standardRoute.duration } : null,
      standardDistanceKm,
      standardDistanceMiles,
      standardTransitHours,
      baseDistanceKm,
      baseTransitHours,
      containerWeight,
      standardPickupTime: standardPickupTime.toISOString(),
      standardArrival: standardArrival.toISOString(),
      cost: calculateTonMileCost(standardDistanceMiles, containerWeight, getPricingForRouteType('standard'))
    })
    
    options.push({
      id: 'standard',
      name: 'Standard Route',
      description: `Container off ship at ${formatDateTime(actualContainerOffShipTime)}`,
      mileage: standardDistanceMiles,
      cost: calculateTonMileCost(standardDistanceMiles, containerWeight, getPricingForRouteType('standard')),
      time: standardTransitHours,
      departureTime: standardPickupTime,
      arrivalTime: standardArrival,
      trafficLevel: getTrafficLevel(standardPickupTime),
      routeCoordinates: standardRoute?.geometry.coordinates,
      routeDistance: standardDistanceKm,
      routeDuration: standardRoute?.duration,
    })
    
    // Late pickup option
    const latePickupTime = new Date(actualContainerOffShipTime.getTime() + 2 * 60 * 60 * 1000)
    const shouldIncludeLate = latePickupTime > standardPickupTime
    console.log('RouteOptionsDisplay: Late option check:', {
      latePickupTime: latePickupTime.toISOString(),
      standardPickupTime: standardPickupTime.toISOString(),
      shouldIncludeLate
    })
    
    if (shouldIncludeLate) {
      const lateRoute = fetchedRoutes.get('late')
      const lateDistanceKm = (lateRoute && lateRoute.distance && lateRoute.distance > 0)
        ? lateRoute.distance / 1000 
        : baseDistanceKm * 1.05
      const lateDistanceMiles = lateDistanceKm * 0.621371
      const lateTransitHours = (lateRoute && lateRoute.duration && lateRoute.duration > 0)
        ? lateRoute.duration / 3600 
        : baseTransitHours * 1.1
      const lateArrival = new Date(latePickupTime.getTime() + lateTransitHours * 60 * 60 * 1000)
      
      console.log('RouteOptionsDisplay: Late option calculation:', {
        lateRoute: lateRoute ? { distance: lateRoute.distance, duration: lateRoute.duration } : null,
        lateDistanceKm,
        lateDistanceMiles,
        lateTransitHours,
        containerWeight,
        cost: calculateTonMileCost(lateDistanceMiles, containerWeight, getPricingForRouteType('late'))
      })
      
      options.push({
        id: 'late',
        name: 'Delayed Pickup',
        description: `Pickup ${Math.round((latePickupTime.getTime() - standardPickupTime.getTime()) / (1000 * 60))} min late`,
        mileage: lateDistanceMiles,
        cost: calculateTonMileCost(lateDistanceMiles, containerWeight, getPricingForRouteType('late')),
        time: lateTransitHours,
        departureTime: latePickupTime,
        arrivalTime: lateArrival,
        trafficLevel: getTrafficLevel(latePickupTime),
        routeCoordinates: lateRoute?.geometry.coordinates,
        routeDistance: lateDistanceKm,
        routeDuration: lateRoute?.duration,
      })
    }
    
    // Express option - use fastest alternative route
    const expressRoute = fetchedRoutes.get('express') || standardRoute
    const expressDistanceKm = (expressRoute && expressRoute.distance && expressRoute.distance > 0)
      ? expressRoute.distance / 1000 
      : baseDistanceKm * 1.1
    const expressDistanceMiles = expressDistanceKm * 0.621371
    const expressHours = (expressRoute && expressRoute.duration && expressRoute.duration > 0)
      ? expressRoute.duration / 3600 
      : standardTransitHours * 0.75
    const expressArrival = new Date(standardPickupTime.getTime() + expressHours * 60 * 60 * 1000)
    
    console.log('RouteOptionsDisplay: Express option calculation:', {
      expressRoute: expressRoute ? { distance: expressRoute.distance, duration: expressRoute.duration } : null,
      expressDistanceKm,
      expressDistanceMiles,
      expressHours,
      containerWeight,
      cost: calculateTonMileCost(expressDistanceMiles, containerWeight, getPricingForRouteType('express'))
    })
    
    options.push({
      id: 'express',
      name: 'Express Route',
      description: 'Priority route with faster delivery',
      mileage: expressDistanceMiles,
      cost: calculateTonMileCost(expressDistanceMiles, containerWeight, getPricingForRouteType('express')),
      time: expressHours,
      departureTime: standardPickupTime,
      arrivalTime: expressArrival,
      trafficLevel: getTrafficLevel(standardPickupTime),
      routeCoordinates: expressRoute?.geometry.coordinates,
      routeDistance: expressDistanceKm,
      routeDuration: expressRoute?.duration,
    })
    
    // Economy option - use shortest alternative route
    const economyRoute = fetchedRoutes.get('economy') || standardRoute
    const economyDistanceKm = (economyRoute && economyRoute.distance && economyRoute.distance > 0)
      ? economyRoute.distance / 1000 
      : baseDistanceKm * 0.9
    const economyDistanceMiles = economyDistanceKm * 0.621371
    const economyHours = (economyRoute && economyRoute.duration && economyRoute.duration > 0)
      ? economyRoute.duration / 3600 
      : standardTransitHours * 1.25
    const economyArrival = new Date(standardPickupTime.getTime() + economyHours * 60 * 60 * 1000)
    
    console.log('RouteOptionsDisplay: Economy option calculation:', {
      economyRoute: economyRoute ? { distance: economyRoute.distance, duration: economyRoute.duration } : null,
      economyDistanceKm,
      economyDistanceMiles,
      economyHours,
      containerWeight,
      cost: calculateTonMileCost(economyDistanceMiles, containerWeight, getPricingForRouteType('economy'))
    })
    
    options.push({
      id: 'economy',
      name: 'Economy Route',
      description: 'Cost-optimized route',
      mileage: economyDistanceMiles,
      cost: calculateTonMileCost(economyDistanceMiles, containerWeight, getPricingForRouteType('economy')),
      time: economyHours,
      departureTime: standardPickupTime,
      arrivalTime: economyArrival,
      trafficLevel: getTrafficLevel(standardPickupTime),
      routeCoordinates: economyRoute?.geometry.coordinates,
      routeDistance: economyDistanceKm,
      routeDuration: economyRoute?.duration,
    })
    
    const sortedOptions = options.sort((a, b) => a.cost - b.cost)
    console.log('RouteOptionsDisplay: Generated route options:', sortedOptions.map(opt => ({
      id: opt.id,
      name: opt.name,
      mileage: opt.mileage,
      cost: opt.cost,
      time: opt.time
    })))
    return sortedOptions
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [baseDistanceMiles, baseTransitHours, vehiclePickupTime, estimatedTimeToStorage, containerArrivalAtPort, avgTimeInPort, totalContainersOnShip, containerPosition, containerWeight, originPort, trafficData, allPorts, fetchedRoutes, route])
  
  return (
    <div className="space-y-3">
      <h3 className="text-sm font-semibold text-gray-900 mb-3">Route Options</h3>
      {routeOptions.map((option) => {
        console.log('RouteOptionsDisplay: Rendering option card:', {
          id: option.id,
          name: option.name,
          mileage: option.mileage,
          time: option.time,
          cost: option.cost,
          mileageFormatted: option.mileage.toFixed(1),
          timeFormatted: option.time.toFixed(1),
          costFormatted: option.cost.toFixed(2)
        })
        return (
        <div 
          key={option.id} 
          className={`rounded-lg p-4 border border-gray-200 cursor-pointer transition-all hover:border-blue-400 hover:shadow-md ${onRouteOptionSelect ? '' : ''}`}
          onClick={() => onRouteOptionSelect?.(option)}
        >
          <div className="flex items-start justify-between mb-3">
            <div className="flex-1">
              <div className="flex items-center gap-2">
                <h4 className="text-sm font-semibold text-gray-900">{option.name}</h4>
                <span
                  className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium border ${getTrafficIndicatorStyle(option.trafficLevel.level)}`}
                  title={option.trafficLevel.description}
                >
                  {getTrafficIcon(option.trafficLevel.level)}
                  {option.trafficLevel.label}
                </span>
              </div>
              <p className="text-xs text-gray-500 mt-0.5">{option.description}</p>
            </div>
          </div>
          
          <div className="grid grid-cols-3 gap-4">
            <div>
              <div className="text-xs text-gray-500 mb-1">Mileage</div>
              <div className="text-base font-semibold text-gray-900">
                {option.mileage.toFixed(1)} mi
              </div>
            </div>
            <div>
              <div className="text-xs text-gray-500 mb-1">Time</div>
              <div className="text-base font-semibold text-gray-900">
                {option.time.toFixed(1)} hrs
              </div>
            </div>
            <div>
              <div className="text-xs text-gray-500 mb-1">Cost</div>
              <div className="text-base font-semibold text-gray-900">
                ${option.cost.toFixed(2)}
              </div>
            </div>
          </div>
          
          <div className="mt-3 pt-3 border-t border-gray-100 text-xs text-gray-500">
            <div className="flex justify-between">
              <span>Departure:</span>
              <span>{formatDateTime(option.departureTime)}</span>
            </div>
            <div className="flex justify-between mt-1">
              <span>Arrival:</span>
              <span>{formatDateTime(option.arrivalTime)}</span>
            </div>
          </div>
        </div>
        )
      })}
    </div>
  )
}

interface ContainerPickupFormProps {
  onRouteCreate: (route: CustomerRoute) => void
  allPorts?: Map<string, PortDatabaseEntry>
  trafficData?: PortTrafficData[]
  onRouteOptionSelect?: (routeOption: RouteOption, customerRoute: CustomerRoute) => void
  onStepChange?: (step: 'pickup' | 'forecast' | 'routes') => void
  initialStep?: 'pickup' | 'forecast' | 'routes'
}

interface TrafficProjection {
  portTraffic: {
    level: 'low' | 'medium' | 'high'
    label: string
    description: string
    avgDailyPortCalls: number
    estimatedHourlyCalls: number
    avgTimeInPort: number
  }
  roadTraffic: {
    level: 'low' | 'medium' | 'high'
    label: string
    description: string
  }
}

interface GeocodeResult {
  placeName: string
  coordinates: [number, number]
}

export default function ContainerPickupForm({ onRouteCreate, allPorts = new Map(), trafficData = [], onRouteOptionSelect, onStepChange, initialStep = 'pickup' }: ContainerPickupFormProps) {
  const [pickupLocation, setPickupLocation] = useState('Los Angeles, CA')
  const [pickupCoordinates, setPickupCoordinates] = useState<[number, number] | null>([-118.2437, 34.0522]) // Los Angeles coordinates
  const [destinationLocation, setDestinationLocation] = useState('Fleet Yards Inc./DAMCO DISTRIBUTION INC')
  const [destinationCoordinates, setDestinationCoordinates] = useState<[number, number] | null>([-118.1937, 33.7701]) // Long Beach coordinates (will be updated when geocoded)
  const [containerNumber, setContainerNumber] = useState('')
  const [containerWeight, setContainerWeight] = useState<string>('') // in tons
  const [pickupDate, setPickupDate] = useState<string>('') // ISO date string
  const [estimatedShipArrival, setEstimatedShipArrival] = useState<string>('') // Time string (HH:mm)
  const [step, setStep] = useState<'pickup' | 'forecast' | 'routes'>(initialStep) // Stepper state
  const [vehicleType, setVehicleType] = useState<'raden' | 'gliderm'>('raden')
  const [createdRoute, setCreatedRoute] = useState<CustomerRoute | null>(null) // Store created route to show options
  const createdRouteRef = useRef<CustomerRoute | null>(null) // Ref to persist route across re-renders
  const [pickupSuggestions, setPickupSuggestions] = useState<GeocodeResult[]>([])
  const [destinationSuggestions, setDestinationSuggestions] = useState<GeocodeResult[]>([])
  const [isSearching, setIsSearching] = useState(false)
  const [isCreatingRoute, setIsCreatingRoute] = useState(false)
  const [showPickupSuggestions, setShowPickupSuggestions] = useState(false)
  const [showDestinationSuggestions, setShowDestinationSuggestions] = useState(false)
  const pickupContainerRef = useRef<HTMLDivElement>(null)
  const destinationContainerRef = useRef<HTMLDivElement>(null)
  const pickupSuggestionsRef = useRef<HTMLDivElement>(null)
  const destinationSuggestionsRef = useRef<HTMLDivElement>(null)

  // Close suggestions when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (pickupSuggestionsRef.current && !pickupSuggestionsRef.current.contains(event.target as Node)) {
        setShowPickupSuggestions(false)
      }
      if (destinationSuggestionsRef.current && !destinationSuggestionsRef.current.contains(event.target as Node)) {
        setShowDestinationSuggestions(false)
      }
    }

    document.addEventListener('mousedown', handleClickOutside)
    return () => {
      document.removeEventListener('mousedown', handleClickOutside)
    }
  }, [])

  // Geocode address using Mapbox Geocoding API
  const geocodeAddress = async (query: string): Promise<GeocodeResult[]> => {
    const MAPBOX_TOKEN = process.env.NEXT_PUBLIC_MAPBOX_TOKEN
    if (!MAPBOX_TOKEN) {
      console.warn('Mapbox token not found')
      return []
    }

    try {
      const response = await fetch(
        `https://api.mapbox.com/geocoding/v5/mapbox.places/${encodeURIComponent(query)}.json?access_token=${MAPBOX_TOKEN}&limit=5&types=place,locality,neighborhood,address,poi`
      )

      if (response.ok) {
        const data = await response.json()
        return data.features.map((feature: { place_name: string; center: [number, number] }) => ({
          placeName: feature.place_name,
          coordinates: feature.center as [number, number],
        }))
      }
    } catch (error) {
      console.error('Geocoding error:', error)
    }

    return []
  }

  // Search ports in the database
  const searchPorts = (query: string): GeocodeResult[] => {
    const results: GeocodeResult[] = []
    const lowerQuery = query.toLowerCase()

    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    for (const [_portId, port] of allPorts.entries()) {
      const portName = port.portname?.toLowerCase() || ''
      const country = port.country?.toLowerCase() || ''
      
      if (portName.includes(lowerQuery) || country.includes(lowerQuery)) {
        if (port.lat && port.lon) {
          results.push({
            placeName: `${port.portname}, ${port.country}`,
            coordinates: [port.lon, port.lat],
          })
        }
      }

      if (results.length >= 5) break
    }

    return results
  }

  // Handle pickup location input
  const handlePickupChange = async (value: string) => {
    setPickupLocation(value)
    
    if (value.length > 2) {
      setIsSearching(true)
      
      // Search ports first
      const portResults = searchPorts(value)
      
      // Also geocode if not found in ports
      let geocodeResults: GeocodeResult[] = []
      if (portResults.length < 5) {
        geocodeResults = await geocodeAddress(value)
      }
      
      // Combine and deduplicate results
      const allResults = [...portResults, ...geocodeResults]
      const uniqueResults = allResults.filter((result, index, self) =>
        index === self.findIndex((r) => r.placeName === result.placeName)
      )
      
      setPickupSuggestions(uniqueResults.slice(0, 5))
      setShowPickupSuggestions(true)
      setIsSearching(false)
    } else {
      setPickupSuggestions([])
      setShowPickupSuggestions(false)
    }
  }

  // Handle destination location input
  const handleDestinationChange = async (value: string) => {
    setDestinationLocation(value)
    
    if (value.length > 2) {
      setIsSearching(true)
      
      // Search ports first
      const portResults = searchPorts(value)
      
      // Also geocode if not found in ports
      let geocodeResults: GeocodeResult[] = []
      if (portResults.length < 5) {
        geocodeResults = await geocodeAddress(value)
      }
      
      // Combine and deduplicate results
      const allResults = [...portResults, ...geocodeResults]
      const uniqueResults = allResults.filter((result, index, self) =>
        index === self.findIndex((r) => r.placeName === result.placeName)
      )
      
      setDestinationSuggestions(uniqueResults.slice(0, 5))
      setShowDestinationSuggestions(true)
      setIsSearching(false)
    } else {
      setDestinationSuggestions([])
      setShowDestinationSuggestions(false)
    }
  }

  // Select pickup location
  const selectPickupLocation = (suggestion: GeocodeResult) => {
    setPickupLocation(suggestion.placeName)
    setPickupCoordinates(suggestion.coordinates)
    setShowPickupSuggestions(false)
  }

  // Select destination location
  const selectDestinationLocation = (suggestion: GeocodeResult) => {
    setDestinationLocation(suggestion.placeName)
    setDestinationCoordinates(suggestion.coordinates)
    setShowDestinationSuggestions(false)
  }

  // Initialize default locations (Los Angeles and Long Beach) after functions are defined
  useEffect(() => {
    const initializeDefaultLocations = async () => {
      // Only initialize if locations are still at default values
      if (pickupLocation === 'Los Angeles, CA' && destinationLocation === 'Long Beach, CA') {
        // Try to find Los Angeles in ports first
        const laPortResults = searchPorts('Los Angeles')
        if (laPortResults.length > 0) {
          const laResult = laPortResults[0]
          setPickupLocation(laResult.placeName)
          setPickupCoordinates(laResult.coordinates)
        } else {
          // Fallback to geocoding
          const laGeocodeResults = await geocodeAddress('Los Angeles, CA')
          if (laGeocodeResults.length > 0) {
            const laResult = laGeocodeResults[0]
            setPickupLocation(laResult.placeName)
            setPickupCoordinates(laResult.coordinates)
          }
        }

        // Try to find Long Beach in ports first
        const lbPortResults = searchPorts('Long Beach')
        if (lbPortResults.length > 0) {
          const lbResult = lbPortResults[0]
          setDestinationLocation(lbResult.placeName)
          setDestinationCoordinates(lbResult.coordinates)
        } else {
          // Fallback to geocoding
          const lbGeocodeResults = await geocodeAddress('Long Beach, CA')
          if (lbGeocodeResults.length > 0) {
            const lbResult = lbGeocodeResults[0]
            setDestinationLocation(lbResult.placeName)
            setDestinationCoordinates(lbResult.coordinates)
          }
        }
      }
    }

    initializeDefaultLocations()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [allPorts])

  // Find port ID from port name
  const findPortIdByName = (portName: string): string | null => {
    for (const [portId, port] of allPorts.entries()) {
      if (port.portname?.toLowerCase().includes(portName.toLowerCase()) ||
          portName.toLowerCase().includes(port.portname?.toLowerCase() || '')) {
        return portId
      }
    }
    return null
  }

  // Calculate traffic projections based on pickup date and location
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const calculateTrafficProjections = useMemo((): TrafficProjection | null => {
    if (!pickupDate || !pickupLocation || !pickupCoordinates) {
      return null
    }

    // Combine date with default noon time into a Date object
    const pickupDateTime = new Date(`${pickupDate}T12:00:00`)
    if (isNaN(pickupDateTime.getTime())) {
      return null
    }

    // Find port ID from pickup location
    const portId = findPortIdByName(pickupLocation)
    
    // Calculate port traffic
    let portTrafficLevel: 'low' | 'medium' | 'high' = 'medium'
    let portTrafficLabel = 'Normal'
    let portTrafficDescription = 'Average traffic expected'
    let avgDailyPortCalls = 0
    let estimatedHourlyCalls = 0
    let avgTimeInPort = 0

    if (portId && trafficData && trafficData.length > 0) {
      // Get traffic data for this port
      const portTrafficData = trafficData.filter(d => d.portid === portId)
      
      if (portTrafficData.length > 0) {
        // Calculate port metrics
        const portMetrics = calculatePortVesselMetrics(portId, portTrafficData)
        avgTimeInPort = portMetrics.avgTimeInPort

        // Calculate average daily port calls
        const totalPortCalls = portTrafficData.reduce((sum, d) => sum + (d.portcalls || 0), 0)
        const uniqueDays = new Set(portTrafficData.map(d => d.date)).size
        avgDailyPortCalls = uniqueDays > 0 ? totalPortCalls / uniqueDays : 0

        // Estimate hourly port calls based on time of day
        const hourOfDay = pickupDateTime.getHours()
        const isPeakHour = (hourOfDay >= 8 && hourOfDay < 12) || (hourOfDay >= 13 && hourOfDay < 17)
        const isLowHour = hourOfDay >= 22 || hourOfDay < 6

        if (isLowHour) {
          estimatedHourlyCalls = (avgDailyPortCalls / 24) * 0.5
        } else if (isPeakHour) {
          estimatedHourlyCalls = (avgDailyPortCalls / 24) * 1.5
        } else {
          estimatedHourlyCalls = avgDailyPortCalls / 24
        }

        // Determine traffic level
        const avgHourlyCalls = avgDailyPortCalls / 24
        if (estimatedHourlyCalls < avgHourlyCalls * 0.7) {
          portTrafficLevel = 'low'
          portTrafficLabel = 'Low Traffic'
          portTrafficDescription = 'Light traffic expected at port'
        } else if (estimatedHourlyCalls > avgHourlyCalls * 1.3) {
          portTrafficLevel = 'high'
          portTrafficLabel = 'High Traffic'
          portTrafficDescription = 'Heavy traffic expected at port'
        }
      }
    }

    // Calculate road traffic based on time of day and day of week
    const hourOfDay = pickupDateTime.getHours()
    const dayOfWeek = pickupDateTime.getDay() // 0 = Sunday, 6 = Saturday
    const isWeekend = dayOfWeek === 0 || dayOfWeek === 6
    
    // Road traffic heuristics:
    // Peak hours: 7-9am (morning rush), 4-6pm (evening rush)
    // Off-peak: 10pm-6am, weekends generally lower
    let roadTrafficLevel: 'low' | 'medium' | 'high' = 'medium'
    let roadTrafficLabel = 'Normal'
    let roadTrafficDescription = 'Average road traffic expected'

    const isRushHour = (hourOfDay >= 7 && hourOfDay < 9) || (hourOfDay >= 16 && hourOfDay < 18)
    const isOffPeak = hourOfDay >= 22 || hourOfDay < 6

    if (isWeekend) {
      if (isOffPeak) {
        roadTrafficLevel = 'low'
        roadTrafficLabel = 'Low Traffic'
        roadTrafficDescription = 'Light weekend traffic expected'
      } else {
        roadTrafficLevel = 'medium'
        roadTrafficLabel = 'Normal'
        roadTrafficDescription = 'Moderate weekend traffic'
      }
    } else {
      if (isRushHour) {
        roadTrafficLevel = 'high'
        roadTrafficLabel = 'High Traffic'
        roadTrafficDescription = 'Rush hour traffic expected'
      } else if (isOffPeak) {
        roadTrafficLevel = 'low'
        roadTrafficLabel = 'Low Traffic'
        roadTrafficDescription = 'Light traffic expected'
      } else {
        roadTrafficLevel = 'medium'
        roadTrafficLabel = 'Normal'
        roadTrafficDescription = 'Average road traffic expected'
      }
    }

    return {
      portTraffic: {
        level: portTrafficLevel,
        label: portTrafficLabel,
        description: portTrafficDescription,
        avgDailyPortCalls,
        estimatedHourlyCalls,
        avgTimeInPort,
      },
      roadTraffic: {
        level: roadTrafficLevel,
        label: roadTrafficLabel,
        description: roadTrafficDescription,
      },
    }
  }, [pickupDate, pickupLocation, pickupCoordinates, trafficData, allPorts])

  // Recommend vehicle type based on container weight
  // GlīderM: Best for lighter loads (0-15 tons) - more efficient for smaller containers
  // Raden: Best for heavier loads (15+ tons) - better capacity for heavy cargo
  const recommendVehicleType = (weight: number): 'raden' | 'gliderm' => {
    if (weight <= 0) return 'raden' // Default
    return weight <= 15 ? 'gliderm' : 'raden'
  }

  // Update vehicle type when weight changes
  useEffect(() => {
    if (containerWeight) {
      const weightNum = parseFloat(containerWeight)
      if (!isNaN(weightNum) && weightNum > 0) {
        const recommended = recommendVehicleType(weightNum)
        setVehicleType(recommended)
      }
    }
  }, [containerWeight])

  // Generate container number if empty
  const generateContainerNumber = () => {
    if (!containerNumber) {
      const prefix = vehicleType === 'raden' ? 'RDN' : 'GLD'
      const random = Math.floor(Math.random() * 10000000).toString().padStart(7, '0')
      setContainerNumber(`${prefix}${random}`)
    }
  }

  // Create route
  const handleCreateRoute = async () => {
    if (!pickupCoordinates || !destinationCoordinates) {
      alert('Please select both pickup and destination locations')
      return
    }

    if (!containerWeight || parseFloat(containerWeight) <= 0) {
      alert('Please enter a valid container weight')
      return
    }

    if (!containerNumber.trim()) {
      generateContainerNumber()
    }

    setIsCreatingRoute(true)
    try {
      // Create route points
      const origin: RoutePoint = {
        type: 'port',
        id: `pickup-${Date.now()}`,
        coordinates: pickupCoordinates,
        name: pickupLocation,
      }

      const destination: RoutePoint = {
        type: 'port',
        id: `destination-${Date.now()}`,
        coordinates: destinationCoordinates,
        name: destinationLocation,
      }

      // Calculate initial route with straight line
      const initialRoute: Route = {
        origin,
        destination,
        coordinates: [pickupCoordinates, destinationCoordinates],
        distance: calculateDistance(pickupCoordinates, destinationCoordinates),
      }

      // Try to get driving route
      let finalRoute = initialRoute
      try {
        const drivingRoute = await getDrivingRoute(pickupCoordinates, destinationCoordinates)
        if (drivingRoute && drivingRoute.geometry && drivingRoute.geometry.coordinates && drivingRoute.geometry.coordinates.length > 0) {
          // Mapbox returns distance in meters, convert to kilometers
          const distanceInKm = drivingRoute.distance / 1000
          finalRoute = {
            ...initialRoute,
            coordinates: drivingRoute.geometry.coordinates as [number, number][],
            distance: distanceInKm, // Now in kilometers
          }
          console.log('Using driving route with', finalRoute.coordinates.length, 'coordinates')
        } else {
          console.log('Driving route not available, using straight line route')
        }
      } catch (error) {
        console.warn('Could not fetch driving route, using straight line:', error)
      }

      // Ensure route has valid coordinates (fallback to initial route if needed)
      if (!finalRoute.coordinates || finalRoute.coordinates.length === 0) {
        console.warn('Route has no coordinates, using fallback')
        finalRoute.coordinates = [pickupCoordinates, destinationCoordinates]
      }
      
      // Ensure route has valid distance
      if (!finalRoute.distance || finalRoute.distance === 0 || isNaN(finalRoute.distance)) {
        console.warn('Route has no distance, calculating fallback')
        const fallbackDistance = calculateDistance(pickupCoordinates, destinationCoordinates)
        finalRoute.distance = fallbackDistance > 0 ? fallbackDistance : 50 // Minimum 50km if calculation fails
        console.log('Fallback distance calculated:', finalRoute.distance)
      }
      
      // Log route details for debugging
      console.log('Route details before creating CustomerRoute:', {
        distance: finalRoute.distance,
        distanceKm: finalRoute.distance,
        distanceMiles: finalRoute.distance ? finalRoute.distance * 0.621371 : 0,
        coordinateCount: finalRoute.coordinates?.length || 0,
        hasOrigin: !!finalRoute.origin,
        hasDestination: !!finalRoute.destination,
        originCoords: finalRoute.origin?.coordinates,
        destCoords: finalRoute.destination?.coordinates,
        routeFull: finalRoute
      })
      
      // Ensure we have at least 2 coordinates (origin and destination)
      if (finalRoute.coordinates.length < 2) {
        console.warn('Route has insufficient coordinates, adding fallback')
        finalRoute.coordinates = [pickupCoordinates, destinationCoordinates]
      }
      
      console.log('Final route prepared:', {
        hasCoordinates: !!finalRoute.coordinates,
        coordinateCount: finalRoute.coordinates?.length || 0,
        distance: finalRoute.distance,
        hasOrigin: !!finalRoute.origin,
        hasDestination: !!finalRoute.destination
      })

      // Create customer route
      // Use selected pickup date, default to noon if not set
      const pickupDateTime = pickupDate
        ? new Date(`${pickupDate}T12:00:00`)
        : new Date()
      
      // Ensure pickup time is valid and not in the past
      const validPickupTime = pickupDateTime.getTime() > Date.now() 
        ? pickupDateTime 
        : new Date(Date.now() + 60 * 60 * 1000) // 1 hour from now if invalid

      // Estimate delivery time based on route distance (assuming average speed)
      const avgSpeedMph = 50 // Average speed in miles per hour
      const routeDistanceKm = finalRoute.distance && finalRoute.distance > 0 ? finalRoute.distance : 50 // Ensure we have a valid distance
      const routeDistanceMiles = routeDistanceKm * 0.621371
      const transitHours = Math.max(0.1, routeDistanceMiles / avgSpeedMph) // Ensure at least 0.1 hours
      const estimatedDelivery = new Date(validPickupTime.getTime() + transitHours * 60 * 60 * 1000)
      
      console.log('Estimated delivery calculation:', {
        routeDistanceKm,
        routeDistanceMiles,
        transitHours,
        pickupTime: validPickupTime.toISOString(),
        estimatedDelivery: estimatedDelivery.toISOString(),
        timeDiff: estimatedDelivery.getTime() - validPickupTime.getTime()
      })

      const newRoute: CustomerRoute = {
        id: `route-${Date.now()}`,
        containerNumber: containerNumber || `${vehicleType === 'raden' ? 'RDN' : 'GLD'}${Math.floor(Math.random() * 10000000).toString().padStart(7, '0')}`,
        shipmentDate: new Date(),
        originPort: pickupLocation,
        storageFacility: destinationLocation,
        destinationPort: destinationLocation,
        status: 'scheduled',
        route: finalRoute,
        estimatedDelivery,
        vehicleType,
      }

      // Store created route to show options
      console.log('Created route:', {
        id: newRoute.id,
        hasRoute: !!newRoute.route,
        routeCoordinates: newRoute.route?.coordinates?.length || 0,
        routeDistance: newRoute.route?.distance,
        routeDistanceKm: newRoute.route?.distance,
        routeDistanceMiles: newRoute.route?.distance ? newRoute.route.distance * 0.621371 : 0,
        originPort: newRoute.originPort,
        destinationPort: newRoute.destinationPort,
        vehiclePickupTime: newRoute.estimatedPickupTime?.toISOString() || validPickupTime.toISOString(),
        estimatedTimeToStorage: newRoute.estimatedDropoffTime?.toISOString() || estimatedDelivery.toISOString(),
        containerArrivalAtPort: undefined,
        routeFull: newRoute.route,
        newRouteFull: newRoute
      })
      
      // Verify route is valid before setting
      if (!newRoute.route || !newRoute.route.coordinates || newRoute.route.coordinates.length < 2) {
        console.error('Route validation failed:', newRoute)
        throw new Error('Route is missing required coordinates')
      }
      
      // Store in ref immediately to persist across re-renders (do this FIRST)
      createdRouteRef.current = {
        ...newRoute,
        route: {
          ...newRoute.route,
          distance: newRoute.route.distance, // Ensure distance is preserved
          origin: newRoute.route.origin,
          destination: newRoute.route.destination,
          coordinates: newRoute.route.coordinates ? [...newRoute.route.coordinates] : []
        }
      }
      
      console.log('Stored route in ref:', {
        distance: createdRouteRef.current.route?.distance,
        distanceKm: createdRouteRef.current.route?.distance,
        distanceMiles: ((createdRouteRef.current.route?.distance || 0) * 0.621371),
        hasCoordinates: (createdRouteRef.current.route?.coordinates?.length || 0) > 0,
        coordinateCount: createdRouteRef.current.route?.coordinates?.length || 0,
        estimatedTimeToStorage: createdRouteRef.current.estimatedDropoffTime?.toISOString(),
        vehiclePickupTime: createdRouteRef.current.estimatedPickupTime?.toISOString(),
        routeOrigin: createdRouteRef.current.route?.origin,
        routeDestination: createdRouteRef.current.route?.destination,
        routeFull: createdRouteRef.current.route,
        newRouteFull: createdRouteRef.current
      })
      
      // Set the created route immediately
      setCreatedRoute(newRoute)
      
      // Advance to routes step
      setStep('routes')
      
      // Call parent callback to add route
      if (onRouteCreate) {
        try {
          onRouteCreate(newRoute)
        } catch (error) {
          console.error('Error in onRouteCreate callback:', error)
        }
      }
    } catch (error) {
      console.error('Error creating route:', error)
      alert('Failed to create route. Please try again.')
    } finally {
      setIsCreatingRoute(false)
    }
  }

  // Sync step with initialStep prop when it changes
  useEffect(() => {
    if (initialStep && initialStep !== step) {
      setStep(initialStep)
    }
  }, [initialStep])

  // Notify parent of step changes
  useEffect(() => {
    if (onStepChange) {
      onStepChange(step)
    }
  }, [step, onStepChange])

  // Handler for Next button (step 1 -> step 2)
  const handleNext = () => {
    if (pickupLocation && pickupCoordinates && pickupDate && containerWeight && parseFloat(containerWeight) > 0) {
      setStep('forecast')
    } else {
      alert('Please fill in all required fields: Pickup Location, Pickup Date, and Container Weight')
    }
  }


  // Helper function to find port ID by name (duplicated here for use in avgTimeInPort)
  const findPortIdByNameLocal = (portName: string): string | null => {
    for (const [portId, port] of allPorts.entries()) {
      if (port.portname?.toLowerCase().includes(portName.toLowerCase()) ||
          portName.toLowerCase().includes(port.portname?.toLowerCase() || '')) {
        return portId
      }
    }
    return null
  }

  // Calculate average time in port for the created route
  const avgTimeInPort = useMemo(() => {
    if (!createdRoute || !createdRoute.originPort || !trafficData || trafficData.length === 0) {
      return 24 // Default 24 hours
    }

    const portId = findPortIdByNameLocal(createdRoute.originPort)
    if (!portId) return 24

    const portTrafficData = trafficData.filter(d => d.portid === portId)
    if (portTrafficData.length === 0) return 24

    const portMetrics = calculatePortVesselMetrics(portId, portTrafficData)
    return portMetrics.avgTimeInPort
  }, [createdRoute, trafficData, allPorts])

  // Check if initial step is complete (pickup location, date, weight)
  const isInitialStepComplete = pickupLocation && pickupCoordinates && pickupDate && containerWeight && parseFloat(containerWeight) > 0

  // Check if destination is selected (for route options)
  const isDestinationSelected = destinationLocation && destinationCoordinates

  // Helper to find port ID
  const findPortIdForForecast = useMemo(() => {
    if (!pickupLocation || !allPorts || allPorts.size === 0) {
      console.log('findPortIdForForecast: No pickup location or ports available')
      return null
    }
    for (const [pid, port] of allPorts.entries()) {
      if (port.portname?.toLowerCase().includes(pickupLocation.toLowerCase()) ||
          pickupLocation.toLowerCase().includes(port.portname?.toLowerCase() || '')) {
        console.log('findPortIdForForecast: Found port ID', pid, 'for', pickupLocation)
        return pid
      }
    }
    console.log('findPortIdForForecast: No matching port found for', pickupLocation)
    return null
  }, [pickupLocation, allPorts])

  // Generate fake port busyness data for visualization
  const generateFakePortBusynessData = (selectedDate: Date) => {
    const selectedMonth = selectedDate.getMonth() + 1
    const selectedYear = selectedDate.getFullYear()
    const daysInMonth = new Date(selectedYear, selectedMonth, 0).getDate()
    const dailyCalls: { day: number; calls: number }[] = []
    
    // Generate realistic daily data with some variation
    const baseCalls = 45 + Math.random() * 20 // Base 45-65 ships per day
    const selectedDay = selectedDate.getDate()
    
    for (let day = 1; day <= daysInMonth; day++) {
      // Add variation: weekends are quieter, mid-week is busier
      const dayOfWeek = new Date(selectedYear, selectedMonth - 1, day).getDay()
      let multiplier = 1.0
      if (dayOfWeek === 0 || dayOfWeek === 6) multiplier = 0.7 // Weekends
      else if (dayOfWeek >= 1 && dayOfWeek <= 3) multiplier = 1.2 // Mon-Wed
      
      // Selected day gets a bit more traffic
      if (day === selectedDay) multiplier *= 1.1
      
      const calls = Math.round(baseCalls * multiplier + (Math.random() * 10 - 5))
      dailyCalls.push({ day, calls: Math.max(calls, 10) })
    }
    
    const maxCalls = Math.max(...dailyCalls.map(d => d.calls), 1)
    const avgCalls = dailyCalls.reduce((sum, d) => sum + d.calls, 0) / dailyCalls.length
    
    return {
      dailyCalls,
      maxCalls,
      avgCalls,
      totalCalls: dailyCalls.reduce((sum, d) => sum + d.calls, 0),
      selectedDay: selectedDate.getDate(),
    }
  }

  // Calculate port busyness for the selected month
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const _portBusynessData = useMemo(() => {
    if (!pickupLocation || !pickupDate) {
      return null
    }

    const selectedDate = new Date(pickupDate)
    
    // Try to get real data first
    if (trafficData && trafficData.length > 0 && findPortIdForForecast) {
      const selectedMonth = selectedDate.getMonth() + 1
      const selectedYear = selectedDate.getFullYear()

      // Filter traffic data for this port and month
      const monthData = trafficData.filter(d => {
        if (d.portid !== findPortIdForForecast) return false
        const dataDate = new Date(d.date)
        const dataMonth = dataDate.getMonth() + 1
        const dataYear = dataDate.getFullYear()
        return dataMonth === selectedMonth && dataYear === selectedYear
      })

      if (monthData.length > 0) {
        // Calculate daily averages from real data
        const daysInMonth = new Date(selectedYear, selectedMonth, 0).getDate()
        const dailyCalls: { day: number; calls: number }[] = []

        for (let day = 1; day <= daysInMonth; day++) {
          const dayData = monthData.filter(d => {
            const dataDate = new Date(d.date)
            return dataDate.getDate() === day
          })
          const totalCalls = dayData.reduce((sum, d) => sum + (d.portcalls || 0), 0)
          dailyCalls.push({ day, calls: totalCalls })
        }

        const maxCalls = Math.max(...dailyCalls.map(d => d.calls), 1)
        const avgCalls = dailyCalls.reduce((sum, d) => sum + d.calls, 0) / dailyCalls.length

        return {
          dailyCalls,
          maxCalls,
          avgCalls,
          totalCalls: monthData.reduce((sum, d) => sum + (d.portcalls || 0), 0),
          selectedDay: selectedDate.getDate(),
        }
      }
    }
    
    // Fallback to fake data
    return generateFakePortBusynessData(selectedDate)
  }, [pickupLocation, pickupDate, trafficData, findPortIdForForecast])

  // Generate fake hourly traffic data
  const generateFakeHourlyTrafficData = (selectedDate: Date) => {
    const hourlyData: { hour: number; avgCalls: number }[] = []
    const baseHourlyCalls = 2.5 // Base ships per hour
    
    for (let hour = 0; hour < 24; hour++) {
      let multiplier = 0.8 // Base activity
      if (hour >= 6 && hour < 10) multiplier = 1.6 // Morning peak
      else if (hour >= 14 && hour < 18) multiplier = 1.4 // Afternoon peak
      else if (hour >= 22 || hour < 6) multiplier = 0.4 // Night low
      
      // Add some randomness
      const variation = 1 + (Math.random() * 0.3 - 0.15) // ±15% variation
      hourlyData.push({
        hour,
        avgCalls: baseHourlyCalls * multiplier * variation
      })
    }

    const maxHourlyCalls = Math.max(...hourlyData.map(h => h.avgCalls), 1)

    return {
      hourlyData,
      maxHourlyCalls,
      selectedHour: selectedDate.getHours(),
    }
  }

  // Calculate hourly vehicle/truck traffic for selected day (based on export cargo)
  const hourlyVehicleTrafficData = useMemo(() => {
    if (!pickupLocation || !pickupDate) {
      return null
    }

    const selectedDate = new Date(pickupDate)
    
    // Try to get real data first
    if (trafficData && trafficData.length > 0 && findPortIdForForecast) {
      const selectedMonth = selectedDate.getMonth() + 1
      const selectedYear = selectedDate.getFullYear()

      const monthData = trafficData.filter(d => {
        if (d.portid !== findPortIdForForecast) return false
        const dataDate = new Date(d.date)
        return dataDate.getMonth() + 1 === selectedMonth && dataDate.getFullYear() === selectedYear
      })

      if (monthData.length > 0) {
        const dayOfWeek = selectedDate.getDay()
        const sameDayOfWeekData = monthData.filter(d => {
          const dataDate = new Date(d.date)
          return dataDate.getDay() === dayOfWeek
        })

        // Calculate average daily export cargo (represents cargo being transported out)
        const avgDailyExport = sameDayOfWeekData.length > 0
          ? sameDayOfWeekData.reduce((sum, d) => sum + (d.export || 0), 0) / sameDayOfWeekData.length
          : monthData.reduce((sum, d) => sum + (d.export || 0), 0) / monthData.length

        // Estimate vehicles/trucks based on export cargo (assume ~20 tons per truck)
        const avgDailyVehicles = avgDailyExport / 20

        const hourlyData: { hour: number; avgVehicles: number }[] = []
        for (let hour = 0; hour < 24; hour++) {
          // Peak hours for truck traffic: 6-10 AM and 2-6 PM (when cargo is being transported out)
          let multiplier = 0.6
          if (hour >= 6 && hour < 10) multiplier = 1.5 // Morning rush
          else if (hour >= 14 && hour < 18) multiplier = 1.4 // Afternoon rush
          else if (hour >= 10 && hour < 14) multiplier = 1.1 // Mid-day
          else if (hour >= 22 || hour < 6) multiplier = 0.3 // Night/early morning
          
          hourlyData.push({
            hour,
            avgVehicles: (avgDailyVehicles / 24) * multiplier
          })
        }

        const maxHourlyVehicles = Math.max(...hourlyData.map(h => h.avgVehicles), 1)

        return {
          hourlyData,
          maxHourlyVehicles,
          selectedHour: selectedDate.getHours(),
        }
      }
    }
    
    // Fallback to fake data
    const hourlyData: { hour: number; avgVehicles: number }[] = []
    for (let hour = 0; hour < 24; hour++) {
      let multiplier = 0.6
      if (hour >= 6 && hour < 10) multiplier = 1.5
      else if (hour >= 14 && hour < 18) multiplier = 1.4
      else if (hour >= 10 && hour < 14) multiplier = 1.1
      else if (hour >= 22 || hour < 6) multiplier = 0.3
      
      hourlyData.push({
        hour,
        avgVehicles: (50 / 24) * multiplier // Base ~50 vehicles per day
      })
    }

    const maxHourlyVehicles = Math.max(...hourlyData.map(h => h.avgVehicles), 1)

    return {
      hourlyData,
      maxHourlyVehicles,
      selectedHour: selectedDate.getHours(),
    }
  }, [pickupLocation, pickupDate, trafficData, findPortIdForForecast])

  // Calculate hourly traffic for selected day (currently unused but kept for future use)
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const hourlyTrafficData = useMemo(() => {
    if (!pickupLocation || !pickupDate) {
      return null
    }

    const selectedDate = new Date(pickupDate)
    
    // Try to get real data first
    if (trafficData && trafficData.length > 0 && findPortIdForForecast) {
      const selectedMonth = selectedDate.getMonth() + 1
      const selectedYear = selectedDate.getFullYear()

      const monthData = trafficData.filter(d => {
        if (d.portid !== findPortIdForForecast) return false
        const dataDate = new Date(d.date)
        return dataDate.getMonth() + 1 === selectedMonth && dataDate.getFullYear() === selectedYear
      })

      if (monthData.length > 0) {
        const dayOfWeek = selectedDate.getDay()
        const sameDayOfWeekData = monthData.filter(d => {
          const dataDate = new Date(d.date)
          return dataDate.getDay() === dayOfWeek
        })

        const avgDailyCalls = sameDayOfWeekData.length > 0
          ? sameDayOfWeekData.reduce((sum, d) => sum + (d.portcalls || 0), 0) / sameDayOfWeekData.length
          : monthData.reduce((sum, d) => sum + (d.portcalls || 0), 0) / monthData.length

        const hourlyData: { hour: number; avgCalls: number }[] = []
        for (let hour = 0; hour < 24; hour++) {
          let multiplier = 0.8
          if (hour >= 6 && hour < 10) multiplier = 1.4
          else if (hour >= 14 && hour < 18) multiplier = 1.3
          else if (hour >= 22 || hour < 6) multiplier = 0.5
          
          hourlyData.push({
            hour,
            avgCalls: (avgDailyCalls / 24) * multiplier
          })
        }

        const maxHourlyCalls = Math.max(...hourlyData.map(h => h.avgCalls), 1)

        return {
          hourlyData,
          maxHourlyCalls,
          selectedHour: selectedDate.getHours(),
        }
      }
    }
    
    // Fallback to fake data
    return generateFakeHourlyTrafficData(selectedDate)
  }, [pickupLocation, pickupDate, trafficData, findPortIdForForecast])

  // Generate fake weekly forecast data
  const generateFakeWeeklyForecastData = (selectedDate: Date) => {
    const dayNames = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
    const delayByDay: { day: string; avgDelay: number; dwellingShips: number; avgShipsAtPort: number; avgPortCalls: number }[] = []
    
    // Base delay times (hours) - weekdays are busier
    const baseDelays = [18, 24, 26, 28, 27, 25, 20] // Sun-Sat
    const basePortCalls = [15, 25, 28, 30, 29, 27, 20] // Average port calls per day
    
    for (let day = 0; day < 7; day++) {
      // Add some variation
      const delayVariation = 1 + (Math.random() * 0.2 - 0.1) // ±10%
      const callsVariation = 1 + (Math.random() * 0.2 - 0.1)
      
      const avgDelay = Math.round(baseDelays[day] * delayVariation * 10) / 10
      const avgPortCalls = Math.round(basePortCalls[day] * callsVariation * 10) / 10
      // Average ships at port = average port calls * (avg time in port / 24 hours)
      const avgShipsAtPort = avgPortCalls * (avgDelay / 24)
      const dwellingShips = Math.round(avgShipsAtPort)
      
      delayByDay.push({
        day: dayNames[day],
        avgDelay,
        dwellingShips,
        avgShipsAtPort,
        avgPortCalls
      })
    }

    const maxDelay = Math.max(...delayByDay.map(d => d.avgDelay), 1)
    const maxDwelling = Math.max(...delayByDay.map(d => d.dwellingShips), 1)
    const avgDelay = delayByDay.reduce((sum, d) => sum + d.avgDelay, 0) / delayByDay.length
    const avgDwelling = Math.round(delayByDay.reduce((sum, d) => sum + d.dwellingShips, 0) / delayByDay.length)

    return {
      delayByDay,
      dwellingByDay: delayByDay.map(d => ({ day: d.day, ships: d.dwellingShips })),
      maxDelay,
      maxDwelling,
      avgDelay,
      avgDwelling,
      selectedDayIndex: selectedDate.getDay(),
    }
  }

  // Calculate average ships at port for 5 days before and 5 days after selected date
  const shipsAtPort10DayWindow = useMemo(() => {
    if (!pickupLocation || !pickupDate || !trafficData || trafficData.length === 0 || !findPortIdForForecast) {
      return null
    }

    const selectedDate = new Date(pickupDate)
    const portData = trafficData.filter(d => d.portid === findPortIdForForecast)
    
    if (portData.length === 0) {
      return null
    }

    const portMetrics = calculatePortVesselMetrics(findPortIdForForecast, portData)
    const dayData: Array<{ date: Date; dateStr: string; avgShipsAtPort: number; avgPortCalls: number; avgTimeInPort: number }> = []

    // Calculate for 5 days before, selected day, and 5 days after (11 days total)
    for (let offset = -5; offset <= 5; offset++) {
      const targetDate = new Date(selectedDate)
      targetDate.setDate(targetDate.getDate() + offset)
      
      // Find data for this specific date or same day of week
      const dayOfWeek = targetDate.getDay()
      const sameDayData = portData.filter(d => {
        const dataDate = new Date(d.date)
        return dataDate.getDay() === dayOfWeek
      })
      
      if (sameDayData.length > 0) {
        const dayMetrics = calculatePortVesselMetrics(findPortIdForForecast, sameDayData)
        const dayAvgCalls = sameDayData.reduce((sum, d) => sum + (d.portcalls || 0), 0) / sameDayData.length
        const avgShipsAtPort = dayAvgCalls * (dayMetrics.avgTimeInPort / 24)
        
        dayData.push({
          date: targetDate,
          dateStr: targetDate.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
          avgShipsAtPort,
          avgPortCalls: dayAvgCalls,
          avgTimeInPort: dayMetrics.avgTimeInPort
        })
      } else {
        // Fallback to overall port metrics
        const avgDailyCalls = portData.reduce((sum, d) => sum + (d.portcalls || 0), 0) / portData.length
        const avgShipsAtPort = avgDailyCalls * (portMetrics.avgTimeInPort / 24)
        
        dayData.push({
          date: targetDate,
          dateStr: targetDate.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
          avgShipsAtPort,
          avgPortCalls: avgDailyCalls,
          avgTimeInPort: portMetrics.avgTimeInPort
        })
      }
    }

    const maxShips = Math.max(...dayData.map(d => d.avgShipsAtPort), 1)

    return {
      dayData,
      maxShips,
      selectedDayIndex: 5 // The selected date is at index 5 (5 days before it)
    }
  }, [pickupLocation, pickupDate, trafficData, findPortIdForForecast])

  // Calculate port delay times and dwelling ships for selected week
  const weeklyForecastData = useMemo(() => {
    if (!pickupLocation || !pickupDate) {
      return null
    }

    const selectedDate = new Date(pickupDate)
    const selectedDayIndex = selectedDate.getDay() // 0 = Sunday, 6 = Saturday
    
    // Try to get real data first from PortTrafficData service
    if (trafficData && trafficData.length > 0 && findPortIdForForecast) {
      const portData = trafficData.filter(d => d.portid === findPortIdForForecast)
      
      if (portData.length > 0) {
        console.log('Using real port traffic data for weekly forecast:', {
          portId: findPortIdForForecast,
          totalRecords: portData.length,
          selectedDate: selectedDate.toISOString(),
          selectedDayIndex
        })
        
        const portMetrics = calculatePortVesselMetrics(findPortIdForForecast, portData)
        const avgDailyCalls = portData.reduce((sum, d) => sum + (d.portcalls || 0), 0) / portData.length

        const delayByDay: { day: string; avgDelay: number; dwellingShips: number; avgShipsAtPort: number; avgPortCalls: number }[] = []
        const dayNames = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
        
        for (let day = 0; day < 7; day++) {
          // Filter data for this specific day of week
          const dayData = portData.filter(d => {
            const dataDate = new Date(d.date)
            return dataDate.getDay() === day
          })
          
          if (dayData.length > 0) {
            // Calculate metrics for this specific day of week using real service data
            const dayMetrics = calculatePortVesselMetrics(findPortIdForForecast, dayData)
            const dayAvgCalls = dayData.reduce((sum, d) => sum + (d.portcalls || 0), 0) / dayData.length
            
            // Average ships at port = average port calls * (avg time in port / 24 hours)
            // This represents how many ships are typically at the port at any given time on this day
            const avgShipsAtPort = dayAvgCalls * (dayMetrics.avgTimeInPort / 24)
            
            console.log(`Day ${dayNames[day]} (${day}):`, {
              records: dayData.length,
              avgPortCalls: dayAvgCalls.toFixed(2),
              avgTimeInPort: dayMetrics.avgTimeInPort.toFixed(2),
              avgShipsAtPort: avgShipsAtPort.toFixed(2)
            })
            
            delayByDay.push({
              day: dayNames[day],
              avgDelay: dayMetrics.avgTimeInPort,
              dwellingShips: Math.round(avgShipsAtPort),
              avgShipsAtPort: avgShipsAtPort,
              avgPortCalls: dayAvgCalls
            })
          } else {
            // Fallback to overall port metrics if no data for this specific day
            const avgShipsAtPort = avgDailyCalls * (portMetrics.avgTimeInPort / 24)
            delayByDay.push({
              day: dayNames[day],
              avgDelay: portMetrics.avgTimeInPort,
              dwellingShips: Math.round(avgShipsAtPort),
              avgShipsAtPort: avgShipsAtPort,
              avgPortCalls: avgDailyCalls
            })
          }
        }

        const maxDelay = Math.max(...delayByDay.map(d => d.avgDelay), 1)
        const maxDwelling = Math.max(...delayByDay.map(d => d.dwellingShips), 1)
        const avgDwelling = delayByDay.reduce((sum, d) => sum + d.avgShipsAtPort, 0) / delayByDay.length

        const selectedDayData = delayByDay[selectedDayIndex]
        console.log('Selected day data (from service):', {
          selectedDay: dayNames[selectedDayIndex],
          avgShipsAtPort: selectedDayData?.avgShipsAtPort.toFixed(2),
          avgPortCalls: selectedDayData?.avgPortCalls.toFixed(2),
          avgDelay: selectedDayData?.avgDelay.toFixed(2)
        })

        return {
          delayByDay,
          dwellingByDay: delayByDay.map(d => ({ day: d.day, ships: d.dwellingShips })),
          maxDelay,
          maxDwelling,
          avgDelay: portMetrics.avgTimeInPort,
          avgDwelling: Math.round(avgDwelling),
          selectedDayIndex,
        }
      }
    }
    
    // Fallback to fake data only if no real data available
    console.warn('No real port traffic data available, using fallback data for:', pickupLocation)
    return generateFakeWeeklyForecastData(selectedDate)
  }, [pickupLocation, pickupDate, trafficData, findPortIdForForecast])

  // Use ref as fallback if state was cleared - ensure we always have valid route data
  const routeToDisplay = useMemo(() => {
    const route = createdRoute || createdRouteRef.current
    if (!route || !route.route) {
      console.log('routeToDisplay: No route available', { hasCreatedRoute: !!createdRoute, hasRef: !!createdRouteRef.current })
      return null
    }
    
    console.log('routeToDisplay: Processing route', {
      distance: route.route.distance,
      hasCoordinates: route.route.coordinates?.length > 0,
      estimatedTimeToStorage: route.estimatedDropoffTime?.toISOString(),
      vehiclePickupTime: route.estimatedPickupTime?.toISOString()
    })
    
    // Create a new route object to avoid mutations
    let routeDistance = route.route.distance
    if (!routeDistance || routeDistance === 0 || isNaN(routeDistance)) {
      console.warn('Route has invalid distance, recalculating:', routeDistance)
      if (route.route.origin?.coordinates && route.route.destination?.coordinates) {
        routeDistance = calculateDistance(
          route.route.origin.coordinates,
          route.route.destination.coordinates
        )
      } else if (route.route.coordinates && route.route.coordinates.length >= 2) {
        routeDistance = calculateDistance(
          route.route.coordinates[0],
          route.route.coordinates[route.route.coordinates.length - 1]
        )
      } else {
        routeDistance = 50 // Fallback
      }
      console.log('Recalculated distance:', routeDistance)
    }
    
    // Ensure estimatedDropoffTime is valid
    let estimatedDropoffTime = route.estimatedDropoffTime
    if (!estimatedDropoffTime || (route.estimatedPickupTime && estimatedDropoffTime.getTime() === route.estimatedPickupTime.getTime())) {
      const routeDistanceKm = routeDistance || 50
      const routeDistanceMiles = routeDistanceKm * 0.621371
      const avgSpeedMph = 50
      const transitHours = Math.max(0.1, routeDistanceMiles / avgSpeedMph)
      const pickupTime = route.estimatedPickupTime || new Date()
      estimatedDropoffTime = new Date(pickupTime.getTime() + transitHours * 60 * 60 * 1000)
      console.log('Recalculated estimatedDropoffTime:', estimatedDropoffTime.toISOString())
    }
    
    // Return a new object with validated data (don't mutate the original)
    const validatedRoute = {
      ...route,
      route: {
        ...route.route,
        distance: routeDistance,
        origin: route.route.origin,
        destination: route.route.destination,
        coordinates: route.route.coordinates ? [...route.route.coordinates] : []
      },
      estimatedDropoffTime
    }
    
    console.log('routeToDisplay: Returning validated route', {
      distance: validatedRoute.route.distance,
      estimatedDropoffTime: validatedRoute.estimatedDropoffTime?.toISOString()
    })
    
    return validatedRoute
  }, [createdRoute])

  // Debug: Log route data when it changes
  useEffect(() => {
    if (routeToDisplay) {
      console.log('routeToDisplay changed:', {
        hasRoute: !!routeToDisplay.route,
        distance: routeToDisplay.route?.distance,
        hasCoordinates: routeToDisplay.route?.coordinates?.length > 0,
        estimatedTimeToStorage: routeToDisplay.estimatedDropoffTime?.toISOString(),
        vehiclePickupTime: routeToDisplay.estimatedPickupTime?.toISOString()
      })
    } else {
      console.log('routeToDisplay is null')
    }
  }, [routeToDisplay])


  // Stepper view - shows different steps based on step state
  return (
    <div className="h-full flex flex-col">
      {/* Step 1: Pickup Data Form */}
      {step === 'pickup' && (
        <div className="flex-1 overflow-y-auto">
          <div className="max-w-4xl mx-auto p-6">
            <div className="mb-6">
              <h2 className="text-2xl font-semibold text-gray-900 mb-2">Step 1: Pickup Information</h2>
              <p className="text-sm text-gray-600">Enter your pickup details to get started</p>
            </div>
            
            <div className="space-y-6">
              <div className="space-y-4">
                {/* Pickup Location */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Pickup Location *
                  </label>
                  <div className="relative" ref={pickupContainerRef}>
                    <input
                      type="text"
                      value={pickupLocation}
                      onChange={(e) => handlePickupChange(e.target.value)}
                      placeholder="Search for pickup location"
                      disabled={isSearching}
                      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 disabled:bg-gray-100 disabled:cursor-not-allowed"
                    />
                    {isSearching && (
                      <div className="absolute right-3 top-1/2 transform -translate-y-1/2">
                        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
                      </div>
                    )}
                    {showPickupSuggestions && pickupSuggestions.length > 0 && (
                      <div
                        ref={pickupSuggestionsRef}
                        className="absolute z-10 w-full mt-1 bg-white border border-gray-300 rounded-lg shadow-lg max-h-60 overflow-y-auto"
                      >
                        {pickupSuggestions.map((suggestion, idx) => (
                          <div
                            key={idx}
                            onClick={() => selectPickupLocation(suggestion)}
                            className="px-3 py-2 hover:bg-gray-100 cursor-pointer text-sm"
                          >
                            {suggestion.placeName}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </div>

                {/* Pickup Date */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Pickup Date *
                  </label>
                  <input
                    type="date"
                    value={pickupDate}
                    onChange={(e) => setPickupDate(e.target.value)}
                    min={new Date().toISOString().split('T')[0]}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  />
                </div>

                {/* Estimated Ship Arrival */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Estimated Ship Arrival Time
                  </label>
                  <input
                    type="time"
                    value={estimatedShipArrival}
                    onChange={(e) => setEstimatedShipArrival(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  />
                  <p className="mt-1 text-xs text-gray-500">Expected arrival time at port (on pickup date)</p>
                </div>

                {/* Container Weight */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Container Weight (tons) *
                  </label>
                  <input
                    type="number"
                    step="0.1"
                    min="0"
                    value={containerWeight}
                    onChange={(e) => setContainerWeight(e.target.value)}
                    placeholder="Enter weight in tons"
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  />
                </div>

                {/* Destination Location */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Drop-off Location *
                  </label>
                  <div className="relative" ref={destinationContainerRef}>
                    <input
                      type="text"
                      value={destinationLocation}
                      onChange={(e) => handleDestinationChange(e.target.value)}
                      placeholder="Search for drop-off location"
                      disabled={isSearching}
                      className={`w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 disabled:bg-gray-100 disabled:cursor-not-allowed ${
                        destinationCoordinates ? 'border-green-500 bg-green-50' : 'border-gray-300'
                      }`}
                    />
                    {isSearching && (
                      <div className="absolute right-3 top-1/2 transform -translate-y-1/2">
                        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
                      </div>
                    )}
                    {!isSearching && destinationCoordinates && (
                      <div className="absolute right-3 top-1/2 transform -translate-y-1/2">
                        <svg className="w-5 h-5 text-green-600" fill="currentColor" viewBox="0 0 20 20">
                          <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                        </svg>
                      </div>
                    )}
                    {showDestinationSuggestions && destinationSuggestions.length > 0 && (
                      <div
                        ref={destinationSuggestionsRef}
                        className="absolute z-10 w-full mt-1 bg-white border border-gray-300 rounded-lg shadow-lg max-h-60 overflow-y-auto"
                      >
                        {destinationSuggestions.map((suggestion, idx) => (
                          <div
                            key={idx}
                            onClick={() => selectDestinationLocation(suggestion)}
                            className="px-3 py-2 hover:bg-gray-100 cursor-pointer text-sm"
                          >
                            {suggestion.placeName}
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* Container Number */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Container Number (optional)
                </label>
                <input
                  type="text"
                  value={containerNumber}
                  onChange={(e) => setContainerNumber(e.target.value)}
                  placeholder="Auto-generated if empty"
                  className="w-full max-w-md px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                />
              </div>

              {/* Next Button */}
              <div className="flex justify-end pt-4">
                <button
                  type="button"
                  onClick={handleNext}
                  disabled={!pickupLocation || !pickupCoordinates || !pickupDate || !containerWeight || parseFloat(containerWeight) <= 0 || !destinationLocation || !destinationCoordinates}
                  className="px-6 py-2 bg-black text-white rounded-lg hover:bg-gray-900 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors font-medium flex items-center justify-center gap-2"
                >
                  Next
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                  </svg>
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Step 2: Forecast Timeline */}
      {step === 'forecast' && (
        <div className="flex-1 overflow-y-auto flex flex-col">
          <div className="flex-1">
            <div className="max-w-4xl mx-auto p-6">
              <div className="mb-6 flex items-center justify-between">
                <div>
                  <h2 className="text-2xl font-semibold text-gray-900 mb-2">Step 2: Forecast & Logistics</h2>
                  <p className="text-sm text-gray-600">Review forecast data and available vehicles</p>
                </div>
                <button
                  type="button"
                  onClick={() => setStep('pickup')}
                  className="px-4 py-2 text-sm bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors font-medium flex items-center justify-center gap-2"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                  </svg>
                  Back
                </button>
              </div>
              
              {pickupDate && pickupLocation && isInitialStepComplete ? (
                <div className="flow-root p-4">
                  <ul role="list" className="-mb-8">
                {/* Forecast Impact Steps (before pickup location) */}
                {(() => {
                  if (!weeklyForecastData) return null
                  
                  const selectedDayIndex = new Date(pickupDate).getDay()
                  const selectedDayData = weeklyForecastData.delayByDay[selectedDayIndex]
                  
                  // Calculate expected delay based on cranes and ships
                  const shipsAtPort = selectedDayData?.avgShipsAtPort || selectedDayData?.dwellingShips || 0
                  const baseTimeInPort = selectedDayData?.avgDelay || weeklyForecastData.avgDelay
                  const delayCalculation = calculateExpectedDelay(
                    pickupLocation,
                    shipsAtPort,
                    baseTimeInPort
                  )
                  
                  // Always show delay steps if data exists, even if below threshold
                  const showDelays = selectedDayData && delayCalculation.expectedDelay > 0
                  const showDwelling = selectedDayData && selectedDayData.dwellingShips > 0
                  
                  if (!showDelays && !showDwelling) return null
                  
                  return (
                    <>
                      {showDelays && (
                        <li>
                          <div className="relative pb-8">
                            <span className="absolute left-3 top-3 -ml-px h-full w-0.5 bg-slate-300" aria-hidden="true" />
                            <div className="relative flex space-x-3">
                              <div>
                                <span className="flex h-5 w-5 items-center justify-center rounded-full bg-red-500 ring-1 ring-white">
                                  <svg className="h-3 w-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                                  </svg>
                                </span>
                              </div>
                              <div className="flex min-w-0 flex-1 justify-between space-x-4 pt-1.5">
                                <div className="flex-1">
                                  <p className="text-sm font-semibold text-gray-900 leading-tight">
                                    Port Delay Alert
                                  </p>
                                  <div className="mt-2 space-y-1.5">
                                    <p className="text-xs text-gray-600 leading-tight">
                                      Ships at port: <span className="font-semibold text-gray-900">{shipsAtPort.toFixed(1)}</span> | Available cranes: <span className="font-semibold text-gray-900">{delayCalculation.craneCapacity}</span>
                                    </p>
                                    <p className="text-xs text-gray-600 leading-tight">
                                      Base time at port: <span className="font-semibold text-gray-900">{baseTimeInPort.toFixed(1)} hours</span>
                                    </p>
                                    <p className="text-xs text-gray-600 leading-tight">
                                      Expected delay: <span className={`font-semibold ${
                                        delayCalculation.congestionLevel === 'critical' ? 'text-red-700' :
                                        delayCalculation.congestionLevel === 'high' ? 'text-red-600' :
                                        delayCalculation.congestionLevel === 'medium' ? 'text-orange-600' :
                                        'text-yellow-600'
                                      }`}>
                                        {delayCalculation.expectedDelay.toFixed(1)} hours
                                      </span>
                                      <span className="text-gray-500 ml-1">
                                        ({delayCalculation.delayMultiplier.toFixed(2)}x multiplier, {delayCalculation.congestionLevel} congestion)
                                      </span>
                                    </p>
                                  </div>
                                  <p className="mt-2 text-xs text-red-600 leading-tight">
                                    This may impact your pickup time at the port
                                  </p>
                                  
                                  {/* Average Ships at Port - 10 Day Window Visualization */}
                                  {shipsAtPort10DayWindow && (
                                    <div className="mt-3">
                                      <p className="text-xs font-medium text-gray-700 mb-2">Average Ships at Port (5 days before/after)</p>
                                      <div className="flex items-end gap-1">
                                        {shipsAtPort10DayWindow.dayData.map((dayInfo, index) => {
                                          const heightPercent = (dayInfo.avgShipsAtPort / shipsAtPort10DayWindow.maxShips) * 100
                                          // Color scale: green (low) -> yellow -> orange -> red (high)
                                          let color = 'bg-green-400'
                                          if (heightPercent > 75) color = 'bg-red-500'
                                          else if (heightPercent > 50) color = 'bg-orange-500'
                                          else if (heightPercent > 25) color = 'bg-yellow-500'
                                          
                                          const isSelectedDay = index === shipsAtPort10DayWindow.selectedDayIndex
                                          
                                          return (
                                            <div
                                              key={index}
                                              className="flex flex-col items-center flex-1"
                                              title={`${dayInfo.dateStr}: ${dayInfo.avgShipsAtPort.toFixed(1)} ships`}
                                            >
                                              <div
                                                className={`w-full ${color} rounded-t transition-all hover:opacity-80 ${
                                                  isSelectedDay ? 'ring-2 ring-blue-600 ring-offset-1' : ''
                                                }`}
                                                style={{ height: `${Math.max(heightPercent, 5)}%`, minHeight: '4px' }}
                                              />
                                              <span className="text-[8px] text-gray-500 mt-0.5 text-center leading-tight">
                                                {dayInfo.dateStr}
                                              </span>
                                            </div>
                                          )
                                        })}
                                      </div>
                                      <div className="mt-1 flex items-center justify-between text-[9px] text-gray-500">
                                        <span>{shipsAtPort10DayWindow.dayData[0]?.dateStr}</span>
                                        <span className="text-gray-400">Average ships at port</span>
                                        <span>{shipsAtPort10DayWindow.dayData[shipsAtPort10DayWindow.dayData.length - 1]?.dateStr}</span>
                                      </div>
                                    </div>
                                  )}
                                </div>
                                <div className="whitespace-nowrap text-right text-sm text-gray-500">
                                  <span className="inline-flex items-center rounded-full bg-red-50 px-2 py-1 text-xs font-medium text-red-700 ring-1 ring-inset ring-red-600/20">
                                    High Delay
                                  </span>
                                </div>
                              </div>
                            </div>
                          </div>
                        </li>
                      )}
                      
                      {showDwelling && (
                        <li>
                          <div className="relative pb-8">
                            <span className="absolute left-3 top-3 -ml-px h-full w-0.5 bg-slate-300" aria-hidden="true" />
                            <div className="relative flex space-x-3">
                              <div>
                                <span className="flex h-5 w-5 items-center justify-center rounded-full bg-orange-500 ring-1 ring-white">
                                  <svg className="h-3 w-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4" />
                                  </svg>
                                </span>
                              </div>
                              <div className="flex min-w-0 flex-1 justify-between space-x-4 pt-1.5">
                                <div className="flex-1">
                                  <p className="text-sm font-semibold text-gray-900 leading-tight">
                                    High Port Congestion
                                  </p>
                                  <p className="mt-0.5 text-xs text-gray-500 leading-tight">
                                    Estimated vehicles transporting cargo: <span className="font-semibold text-orange-600">{selectedDayData.dwellingShips} ships worth of cargo</span> (avg: {weeklyForecastData.avgDwelling})
                                  </p>
                                  <p className="mt-1 text-xs text-orange-600 leading-tight">
                                    High truck/vehicle traffic may cause delays
                                  </p>
                                  
                                  {/* Vehicle/Truck Traffic by Hour Visualization */}
                                  {hourlyVehicleTrafficData && (
                                    <div className="mt-3">
                                      <p className="text-xs font-medium text-gray-700 mb-2">Vehicle Traffic by Hour</p>
                                      <div className="flex items-end gap-0.5">
                                        {hourlyVehicleTrafficData.hourlyData.map((hourData) => {
                                          const heightPercent = (hourData.avgVehicles / hourlyVehicleTrafficData.maxHourlyVehicles) * 100
                                          // Color scale: green (low) -> yellow -> orange -> red (high)
                                          let color = 'bg-green-400'
                                          if (heightPercent > 75) color = 'bg-red-500'
                                          else if (heightPercent > 50) color = 'bg-orange-500'
                                          else if (heightPercent > 25) color = 'bg-yellow-500'
                                          
                                          const isSelectedHour = hourData.hour === hourlyVehicleTrafficData.selectedHour
                                          
                                          return (
                                            <div
                                              key={hourData.hour}
                                              className="flex flex-col items-center flex-1"
                                              title={`${hourData.hour}:00 - ${hourData.avgVehicles.toFixed(1)} vehicles`}
                                            >
                                              <div
                                                className={`w-full ${color} rounded-t transition-all hover:opacity-80 ${
                                                  isSelectedHour ? 'ring-2 ring-orange-600 ring-offset-1' : ''
                                                }`}
                                                style={{ height: `${Math.max(heightPercent, 5)}%`, minHeight: '4px' }}
                                              />
                                              {hourData.hour % 6 === 0 && (
                                                <span className="text-[8px] text-gray-500 mt-0.5">{hourData.hour}</span>
                                              )}
                                            </div>
                                          )
                                        })}
                                      </div>
                                      <div className="mt-1 flex items-center justify-between text-[9px] text-gray-500">
                                        <span>0:00</span>
                                        <span className="text-gray-400">Vehicles per hour</span>
                                        <span>23:00</span>
                                      </div>
                                    </div>
                                  )}
                                </div>
                                <div className="whitespace-nowrap text-right text-sm text-gray-500">
                                  <span className="inline-flex items-center rounded-full bg-orange-50 px-2 py-1 text-xs font-medium text-orange-700 ring-1 ring-inset ring-orange-600/20">
                                    High Traffic
                                  </span>
                                </div>
                              </div>
                            </div>
                          </div>
                        </li>
                      )}
                    </>
                  )
                })()}
                
                {/* Stop 1: Pickup Location (Port) */}
                <li>
                  <div className="relative pb-8">
                    <span className="absolute left-3 top-3 -ml-px h-full w-0.5 bg-slate-300" aria-hidden="true" />
                    <div className="relative flex space-x-3">
                      <div>
                        <span className="flex h-5 w-5 items-center justify-center rounded-full bg-blue-500 ring-1 ring-white">
                          <svg className="h-3 w-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17.657 16.657L13.414 20.9a1.998 1.998 0 01-2.827 0l-4.244-4.243a8 8 0 1111.314 0z" />
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 11a3 3 0 11-6 0 3 3 0 016 0z" />
                          </svg>
                        </span>
                      </div>
                      <div className="flex min-w-0 flex-1 justify-between space-x-4 pt-1.5">
                        <div>
                          <p className="text-sm font-semibold text-gray-900 leading-tight">
                            {pickupLocation}
                          </p>
                          {pickupCoordinates && (
                            <p className="mt-0.5 text-xs text-gray-500 leading-tight">
                              Coordinates: {pickupCoordinates[1].toFixed(4)}, {pickupCoordinates[0].toFixed(4)}
                            </p>
                          )}
                        </div>
                        <div className="whitespace-nowrap text-right text-sm text-gray-500">
                          <span className="inline-flex items-center rounded-full bg-blue-50 px-2 py-1 text-xs font-medium text-blue-700 ring-1 ring-inset ring-blue-600/20">
                            Port
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                </li>
                
                {/* Intermediate: Available Vehicles */}
                {containerWeight && parseFloat(containerWeight) > 0 && (() => {
                  const weight = parseFloat(containerWeight)
                  const recommended = recommendVehicleType(weight)
                  const isRadenRecommended = recommended === 'raden'
                  const isGlidermRecommended = recommended === 'gliderm'
                  
                  // Simulate vehicle data with backhaul information
                  const pickupDateObj = new Date(pickupDate)
                  const radenHasBackhaul = Math.random() > 0.3
                  const glidermHasBackhaul = Math.random() > 0.4
                  
                  // Calculate estimated arrival times (vehicles arriving before pickup time)
                  const radenArrivalTime = new Date(pickupDateObj)
                  radenArrivalTime.setHours(pickupDateObj.getHours() - 2, pickupDateObj.getMinutes() - 30)
                  
                  const glidermArrivalTime = new Date(pickupDateObj)
                  glidermArrivalTime.setHours(pickupDateObj.getHours() - 1, pickupDateObj.getMinutes() - 15)
                  
                  // Calculate available pickup window (time between arrival and scheduled pickup)
                  const radenPickupWindow = Math.round((pickupDateObj.getTime() - radenArrivalTime.getTime()) / (1000 * 60)) // minutes
                  const glidermPickupWindow = Math.round((pickupDateObj.getTime() - glidermArrivalTime.getTime()) / (1000 * 60)) // minutes
                  
                  // Mock pickup destinations
                  const radenDestination = radenHasBackhaul ? 'Warehouse District A - Downtown' : null
                  const glidermDestination = glidermHasBackhaul ? 'Distribution Center B - Industrial Zone' : null
                  
                  const formatTime = (date: Date) => {
                    return date.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', hour12: true })
                  }
                  
                  return (
                    <li>
                      <div className="relative pb-8">
                        <span className="absolute left-3 top-3 -ml-px h-full w-0.5 bg-slate-300" aria-hidden="true" />
                        <div className="relative flex space-x-3">
                          <div>
                            <span className="flex h-5 w-5 items-center justify-center rounded-full bg-indigo-500 ring-1 ring-white">
                              <svg className="h-3 w-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4" />
                              </svg>
                            </span>
                          </div>
                          <div className="flex min-w-0 flex-1 pt-1.5">
                            <div className="w-full">
                              <p className="text-sm font-semibold text-gray-900 leading-tight mb-3">
                                Available Vehicles
                              </p>
                              <p className="text-xs text-gray-600 leading-tight mb-4">
                                Vehicles with containers to deliver to the port, enabling efficient backhaul and optimal fleet utilization.
                              </p>
                              
                              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                {/* Raden Vehicle Card */}
                                <div className="bg-white rounded-lg border-2 border-gray-200 p-4">
                                  <div className="flex items-start gap-3 mb-3">
                                    <img
                                      src="/raden.png"
                                      alt="Raden"
                                      className="w-12 h-12 object-contain flex-shrink-0"
                                    />
                                    <div className="flex-1">
                                      <div className="flex items-center gap-2 mb-1">
                                        <p className="text-sm font-semibold text-gray-900">Raden</p>
                                        {(isRadenRecommended || radenHasBackhaul) && (
                                          <span className="inline-flex items-center rounded-full bg-gray-100 px-2 py-0.5 text-xs font-medium text-gray-700">
                                            Recommended
                                          </span>
                                        )}
                                      </div>
                                      <p className="text-xs text-gray-500">Optimized for heavy loads (&gt;15 tons)</p>
                                    </div>
                                  </div>
                                  
                                  {radenHasBackhaul ? (
                                    <div className="space-y-2 pt-2 border-t border-gray-200">
                                      <div>
                                        <p className="text-xs font-medium text-gray-700 mb-1">Pickup Destination</p>
                                        <p className="text-xs text-gray-900">{radenDestination}</p>
                                      </div>
                                      <div>
                                        <p className="text-xs font-medium text-gray-700 mb-1">Est. Arrival at Port</p>
                                        <p className="text-xs text-gray-900">{formatTime(radenArrivalTime)}</p>
                                      </div>
                                      <div>
                                        <p className="text-xs font-medium text-gray-700 mb-1">Available Pickup Window</p>
                                        <p className="text-xs text-gray-900">{radenPickupWindow} minutes</p>
                                      </div>
                                    </div>
                                  ) : (
                                    <div className="pt-2 border-t border-gray-200">
                                      <p className="text-xs text-gray-500">No backhaul available</p>
                                    </div>
                                  )}
                                </div>
                                
                                {/* GlīderM Vehicle Card */}
                                <div className="bg-white rounded-lg border-2 border-gray-200 p-4">
                                  <div className="flex items-start gap-3 mb-3">
                                    <img
                                      src="/gliderm.png"
                                      alt="GlīderM"
                                      className="w-12 h-12 object-contain flex-shrink-0"
                                    />
                                    <div className="flex-1">
                                      <div className="flex items-center gap-2 mb-1">
                                        <p className="text-sm font-semibold text-gray-900">GlīderM</p>
                                        {(isGlidermRecommended || glidermHasBackhaul) && (
                                          <span className="inline-flex items-center rounded-full bg-gray-100 px-2 py-0.5 text-xs font-medium text-gray-700">
                                            Recommended
                                          </span>
                                        )}
                                      </div>
                                      <p className="text-xs text-gray-500">Optimized for light loads (≤15 tons)</p>
                                    </div>
                                  </div>
                                  
                                  {glidermHasBackhaul ? (
                                    <div className="space-y-2 pt-2 border-t border-gray-200">
                                      <div>
                                        <p className="text-xs font-medium text-gray-700 mb-1">Pickup Destination</p>
                                        <p className="text-xs text-gray-900">{glidermDestination}</p>
                                      </div>
                                      <div>
                                        <p className="text-xs font-medium text-gray-700 mb-1">Est. Arrival at Port</p>
                                        <p className="text-xs text-gray-900">{formatTime(glidermArrivalTime)}</p>
                                      </div>
                                      <div>
                                        <p className="text-xs font-medium text-gray-700 mb-1">Available Pickup Window</p>
                                        <p className="text-xs text-gray-900">{glidermPickupWindow} minutes</p>
                                      </div>
                                    </div>
                                  ) : (
                                    <div className="pt-2 border-t border-gray-200">
                                      <p className="text-xs text-gray-500">No backhaul available</p>
                                    </div>
                                  )}
                                </div>
                              </div>
                              
                              <p className="mt-4 text-xs text-gray-500 leading-tight">
                                Container weight: {weight} tons • Backhaul optimization enabled
                              </p>
                            </div>
                          </div>
                        </div>
                      </div>
                    </li>
                  )
                })()}
                
                {/* Stop 2: Pickup Time */}
                <li>
                  <div className="relative pb-8">
                    <span className="absolute left-3 top-3 -ml-px h-full w-0.5 bg-slate-300" aria-hidden="true" />
                    <div className="relative flex space-x-3">
                      <div>
                        <span className="flex h-5 w-5 items-center justify-center rounded-full bg-green-500 ring-1 ring-white">
                          <svg className="h-3 w-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                          </svg>
                        </span>
                      </div>
                      <div className="flex min-w-0 flex-1 justify-between space-x-4 pt-1.5">
                        <div className="flex-1">
                          <p className="text-sm font-semibold text-gray-900 leading-tight">
                            Pickup Time
                          </p>
                          <p className="mt-0.5 text-xs text-gray-500 leading-tight">
                            {new Date(pickupDate).toLocaleDateString('en-US', { 
                              weekday: 'long', 
                              month: 'long', 
                              day: 'numeric', 
                              year: 'numeric' 
                            })}
                          </p>
                          
                          {/* Container Pickup Forecasts */}
                          {(() => {
                            const pickupDateObj = new Date(pickupDate)
                            const scheduledPickupTime = new Date(pickupDateObj)
                            scheduledPickupTime.setHours(12, 0, 0, 0) // Default to noon
                            
                            // Use estimated ship arrival time if provided, otherwise calculate (typically 2-4 hours before pickup for offloading)
                            let shipArrivalTime: Date
                            if (estimatedShipArrival && pickupDate) {
                              // Combine pickup date with estimated arrival time
                              const [hours, minutes] = estimatedShipArrival.split(':').map(Number)
                              shipArrivalTime = new Date(pickupDateObj)
                              shipArrivalTime.setHours(hours, minutes || 0, 0, 0)
                            } else {
                              shipArrivalTime = new Date(scheduledPickupTime)
                              shipArrivalTime.setHours(scheduledPickupTime.getHours() - 3) // Ship arrives 3 hours before pickup
                            }
                            
                            // On-time scenario: ship arrives on time, container ready at scheduled pickup
                            const onTimeContainerReady = new Date(scheduledPickupTime)
                            
                            // Delayed ship scenario: ship delayed by forecast delay amount
                            const shipDelayHours = weeklyForecastData ? weeklyForecastData.avgDelay : 4
                            const delayedShipArrival = new Date(shipArrivalTime)
                            delayedShipArrival.setHours(delayedShipArrival.getHours() + shipDelayHours)
                            
                            // Container offloading takes 1-2 hours after ship arrival
                            const offloadingTime = 1.5 // hours
                            const delayedContainerReady = new Date(delayedShipArrival)
                            delayedContainerReady.setHours(delayedContainerReady.getHours() + offloadingTime)
                            
                            const formatDateTime = (date: Date) => {
                              return date.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit', hour12: true })
                            }
                            
                            return (
                              <div className="mt-3 grid grid-cols-1 md:grid-cols-2 gap-3">
                                {/* On-Time Scenario */}
                                <div className="bg-white border border-gray-200 rounded-lg p-3">
                                  <div className="flex items-center gap-2 mb-2">
                                    <svg className="w-4 h-4 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                                    </svg>
                                    <p className="text-xs font-semibold text-gray-900">On-Time Pickup Forecast</p>
                                  </div>
                                  <div className="space-y-1.5 text-xs">
                                    <div className="flex justify-between">
                                      <span className="text-gray-600">Ship Arrival:</span>
                                      <span className="font-medium text-gray-900">{formatDateTime(shipArrivalTime)}</span>
                                    </div>
                                    <div className="flex justify-between">
                                      <span className="text-gray-600">Container Ready:</span>
                                      <span className="font-medium text-gray-900">{formatDateTime(onTimeContainerReady)}</span>
                                    </div>
                                    <div className="flex justify-between">
                                      <span className="text-gray-600">Pickup Window:</span>
                                      <span className="font-medium text-gray-900">Available</span>
                                    </div>
                                  </div>
                                </div>
                                
                                {/* Delayed Ship Scenario */}
                                <div className="bg-white border border-gray-200 rounded-lg p-3">
                                  <div className="flex items-center gap-2 mb-2">
                                    <svg className="w-4 h-4 text-orange-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                                    </svg>
                                    <p className="text-xs font-semibold text-gray-900">Delayed Ship Scenario</p>
                                  </div>
                                  <div className="space-y-1.5 text-xs">
                                    <div className="flex justify-between">
                                      <span className="text-gray-600">Ship Delay:</span>
                                      <span className="font-medium text-gray-900">+{shipDelayHours.toFixed(1)} hours</span>
                                    </div>
                                    <div className="flex justify-between">
                                      <span className="text-gray-600">Delayed Arrival:</span>
                                      <span className="font-medium text-gray-900">{formatDateTime(delayedShipArrival)}</span>
                                    </div>
                                    <div className="flex justify-between">
                                      <span className="text-gray-600">Container Ready:</span>
                                      <span className="font-medium text-gray-900">{formatDateTime(delayedContainerReady)}</span>
                                    </div>
                                    <div className="flex justify-between">
                                      <span className="text-gray-600">New Pickup Time:</span>
                                      <span className="font-medium text-gray-900">{formatDateTime(delayedContainerReady)}</span>
                                    </div>
                                    <div className="mt-2 pt-2 border-t border-gray-200">
                                      <p className="text-[10px] text-gray-600">
                                        ⚠️ Pickup time may shift by {Math.round((delayedContainerReady.getTime() - scheduledPickupTime.getTime()) / (1000 * 60))} minutes
                                      </p>
                                    </div>
                                  </div>
                                </div>
                              </div>
                            )
                          })()}
                        </div>
                        <div className="whitespace-nowrap text-right text-sm text-gray-500">
                          <span className="inline-flex items-center rounded-full bg-green-50 px-2 py-1 text-xs font-medium text-green-700 ring-1 ring-inset ring-green-600/20">
                            Scheduled
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                </li>
                
                {/* Stop 3: Dropoff Location */}
                <li>
                  <div className="relative">
                    <div className="relative flex space-x-3">
                      <div>
                        <span className="flex h-5 w-5 items-center justify-center rounded-full bg-purple-500 ring-1 ring-white">
                          <svg className="h-3 w-3 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                          </svg>
                        </span>
                      </div>
                      <div className="flex min-w-0 flex-1 justify-between space-x-4 pt-1.5">
                        <div>
                          <p className="text-sm font-semibold text-gray-900 leading-tight">
                            {destinationLocation || 'Not specified'}
                          </p>
                          {destinationCoordinates && (
                            <p className="mt-0.5 text-xs text-gray-500 leading-tight">
                              Coordinates: {destinationCoordinates[1].toFixed(4)}, {destinationCoordinates[0].toFixed(4)}
                            </p>
                          )}
                        </div>
                        <div className="whitespace-nowrap text-right text-sm text-gray-500">
                          <span className="inline-flex items-center rounded-full bg-purple-50 px-2 py-1 text-xs font-medium text-purple-700 ring-1 ring-inset ring-purple-600/20">
                            Destination
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                </li>
                </ul>
              </div>
            ) : (
              <div className="text-center py-12 text-gray-500">
                Please complete Step 1 to view forecast data
              </div>
            )}
            </div>
          </div>
          
          {/* Generate Routes Button at bottom */}
          <div className="flex-shrink-0 border-t border-gray-200 bg-white p-6">
            <div className="max-w-4xl mx-auto flex justify-end">
              <button
                type="button"
                onClick={handleCreateRoute}
                disabled={isCreatingRoute || !isDestinationSelected}
                className="px-6 py-2 bg-black text-white rounded-lg hover:bg-gray-900 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors font-medium flex items-center justify-center gap-2"
              >
                {isCreatingRoute ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                    <span>Generating Routes...</span>
                  </>
                ) : (
                  <>
                    Generate Routes
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Step 3: Route Results */}
      {step === 'routes' && (
        <div className="flex-1 overflow-y-auto">
          <div className="max-w-4xl mx-auto p-6">
            {(() => {
              // Show dummy data if no route data available
              if (!routeToDisplay || !routeToDisplay.route || !routeToDisplay.route.coordinates || routeToDisplay.route.coordinates.length < 2 || !routeToDisplay.route.distance || routeToDisplay.route.distance <= 0) {
                return (
                  <div className="space-y-6">
                    <div className="mb-4 flex items-center justify-between">
                      <div>
                        <h2 className="text-2xl font-semibold text-gray-900 mb-2">Step 3: Optimized Routes</h2>
                        <p className="text-sm text-gray-600">Review optimized route options</p>
                      </div>
                    </div>
                    
                    {/* Dummy Route Options using RouteOptionCard component */}
                    <div className="space-y-4">
                      <RouteOptionCard
                        title="Route Option 1: Standard Route"
                        description="Optimized for cost and time efficiency"
                        distance={45.2}
                        estimatedTime={1.2}
                        cost={245.50}
                      />
                      <RouteOptionCard
                        title="Route Option 2: Express Route"
                        description="Faster delivery with priority handling"
                        distance={42.8}
                        estimatedTime={0.9}
                        cost={298.75}
                      />
                      <RouteOptionCard
                        title="Route Option 3: Economy Route"
                        description="Most cost-effective option"
                        distance={47.5}
                        estimatedTime={1.5}
                        cost={198.20}
                      />
                    </div>
                  </div>
                )
              }
              
              // Show actual route data if available
              return (
                <div className="space-y-6">
                  <div className="mb-4 flex items-center justify-between">
                    <div>
                      <h2 className="text-2xl font-semibold text-gray-900 mb-2">Step 3: Route Options</h2>
                      <p className="text-sm text-gray-600">Select a route option to view details</p>
                    </div>
                    <button
                      type="button"
                      onClick={() => setStep('forecast')}
                      className="px-4 py-2 text-sm bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition-colors font-medium flex items-center justify-center gap-2"
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                      </svg>
                      Back
                    </button>
                  </div>

                  {/* Route Options List */}
                  <RouteOptionsDisplay
                    originPort={routeToDisplay.originPort}
                    destinationPort={routeToDisplay.storageFacility || routeToDisplay.destinationPort}
                    vehiclePickupTime={routeToDisplay.estimatedPickupTime || new Date()}
                    containerWeight={parseFloat(containerWeight) || 1}
                    estimatedTimeToStorage={(() => {
                      // Ensure we have a valid estimatedDropoffTime
                      if (routeToDisplay.estimatedDropoffTime) {
                        console.log('RouteOptionsDisplay: Using existing estimatedDropoffTime:', {
                          estimatedDropoffTime: routeToDisplay.estimatedDropoffTime.toISOString(),
                          routeDistance: routeToDisplay.route?.distance,
                          routeDistanceKm: routeToDisplay.route?.distance,
                          routeDistanceMiles: routeToDisplay.route?.distance ? routeToDisplay.route.distance * 0.621371 : 0
                        })
                        return routeToDisplay.estimatedDropoffTime
                      }
                      // Calculate estimated time based on route distance if not set
                      const routeDistanceKm = routeToDisplay.route?.distance || 50
                      const routeDistanceMiles = routeDistanceKm * 0.621371
                      const avgSpeedMph = 50
                      const transitHours = routeDistanceMiles / avgSpeedMph
                      const pickupTime = routeToDisplay.estimatedPickupTime || new Date()
                      const calculated = new Date(pickupTime.getTime() + transitHours * 60 * 60 * 1000)
                      console.log('RouteOptionsDisplay: Calculated estimatedDropoffTime:', {
                        routeDistanceKm,
                        routeDistanceMiles,
                        transitHours,
                        calculated: calculated.toISOString(),
                        pickupTime: pickupTime.toISOString()
                      })
                      return calculated
                    })()}
                    containerArrivalAtPort={undefined}
                    avgTimeInPort={avgTimeInPort}
                    route={(() => {
                      console.log('RouteOptionsDisplay: Route prop being passed:', {
                        hasRoute: !!routeToDisplay.route,
                        routeDistance: routeToDisplay.route?.distance,
                        routeDistanceKm: routeToDisplay.route?.distance,
                        routeDistanceMiles: routeToDisplay.route?.distance ? routeToDisplay.route.distance * 0.621371 : 0,
                        hasCoordinates: routeToDisplay.route?.coordinates?.length > 0,
                        coordinateCount: routeToDisplay.route?.coordinates?.length || 0,
                        routeOrigin: routeToDisplay.route?.origin,
                        routeDestination: routeToDisplay.route?.destination,
                        routeFull: routeToDisplay.route
                      })
                      return routeToDisplay.route
                    })()}
                    trafficData={trafficData}
                    allPorts={allPorts}
                    totalContainersOnShip={1000}
                    containerPosition={'middle'}
                    onRouteOptionSelect={onRouteOptionSelect ? (option) => onRouteOptionSelect(option, routeToDisplay) : undefined}
                  />
                </div>
              )
            })()}
          </div>
        </div>
      )}
    </div>
  )
}
