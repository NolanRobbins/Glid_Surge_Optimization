'use client'

import { useState, useEffect, Suspense } from 'react'
import { usePathname, useSearchParams } from 'next/navigation'
import dynamic from 'next/dynamic'
import { PortTrafficData } from '@/types/portTraffic'
import { PortDatabaseEntry } from '@/api/portDatabaseService'
import { CustomerRoute } from '@/components/CustomerRoutesList'
import ContainerPickupForm, { RouteOption } from '@/components/ContainerPickupForm'
import { Route } from '@/lib/routeCalculation'
import { getDrivingRoute } from '@/lib/mapboxDirections'
import { loadSyntheticCustomerRoutes } from '@/lib/syntheticDataService'

const ContainerPickupMap = dynamic(() => import('@/components/ContainerPickupMap'), {
  ssr: false,
})

function NewPageContent() {
  const pathname = usePathname()
  const searchParams = useSearchParams()
  const [data, setData] = useState<PortTrafficData[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [allPorts, setAllPorts] = useState<Map<string, PortDatabaseEntry>>(new Map())
  const [currentRoute, setCurrentRoute] = useState<Route | null>(null) // Start with no route to show initial center
  const [routePlanningVisible, setRoutePlanningVisible] = useState(true)

  const [hasCreatedPickup, setHasCreatedPickup] = useState(false)
  // Initialize step from URL query parameter if present
  const initialStep = searchParams?.get('step') as 'pickup' | 'forecast' | 'routes' | null
  const [currentFormStep, setCurrentFormStep] = useState<'pickup' | 'forecast' | 'routes'>(
    initialStep && ['pickup', 'forecast', 'routes'].includes(initialStep) ? initialStep : 'pickup'
  )

  // Handle new route creation
  const handleRouteCreate = (newRoute: CustomerRoute) => {
    setCustomerRoutes(prev => [newRoute, ...prev])
    // Set as current route to display on map
    if (newRoute.route) {
      setCurrentRoute(newRoute.route)
    }
    // Show map after first pickup is created
    setHasCreatedPickup(true)
  }

  // Handle route option selection
  const handleRouteOptionSelect = (routeOption: RouteOption, customerRoute: CustomerRoute) => {
    // Create a route from the selected option to display on map using real route coordinates
    if (customerRoute.route && routeOption.routeCoordinates && routeOption.routeCoordinates.length > 0) {
      // Create a modified route with real coordinates from Mapbox
      const modifiedRoute: Route = {
        ...customerRoute.route,
        coordinates: routeOption.routeCoordinates,
        distance: routeOption.routeDistance || customerRoute.route.distance,
      }
      setCurrentRoute(modifiedRoute)
    } else if (customerRoute.route) {
      // Fallback to original route if no coordinates available
      setCurrentRoute(customerRoute.route)
    }
  }

  // Load synthetic customer routes data
  const [customerRoutes, setCustomerRoutes] = useState<CustomerRoute[]>(() => {
    return loadSyntheticCustomerRoutes()
  })

  // Fetch driving route for in-transit shipment
  useEffect(() => {
    const fetchRouteForInTransit = async () => {
      const inTransitRoute = customerRoutes.find(route => route.status === 'in_transit')
      
      if (!inTransitRoute || !inTransitRoute.route) return
      
      // If route coordinates are just origin/destination, fetch driving route
      if (inTransitRoute.route.coordinates.length === 2) {
        const drivingRoute = await getDrivingRoute(
          inTransitRoute.route.origin.coordinates,
          inTransitRoute.route.destination.coordinates
        )
        
        if (drivingRoute && drivingRoute.geometry.coordinates.length > 0) {
          // Update the route with road-following coordinates
          const updatedRoute: Route = {
            ...inTransitRoute.route,
            coordinates: drivingRoute.geometry.coordinates,
            distance: drivingRoute.distance / 1000, // Convert meters to kilometers
          }
          
          // Update customer routes with the new route data
          setCustomerRoutes(prevRoutes => 
            prevRoutes.map(route => 
              route.id === inTransitRoute.id
                ? { ...route, route: updatedRoute }
                : route
            )
          )
          
          // Set current route for map display only if user explicitly selects a route
          // Don't auto-set route on page load to preserve initialCenter
          // if (!currentRoute) {
          //   setCurrentRoute(updatedRoute)
          // }
        }
      }
    }
    
    fetchRouteForInTransit()
  }, [customerRoutes]) // Removed currentRoute from dependencies to prevent auto-setting

  useEffect(() => {
    async function loadData() {
      try {
        setLoading(true)
        setError(null)
        
        // Fetch recent data with automatic pagination
        // Date filtering is done client-side to avoid ArcGIS API date format issues
        let fetchedData: PortTrafficData[] = []
        
        try {
          const { fetchAllPortTrafficData } = await import('@/api/portTrafficService')
          
          // Calculate date range for last 90 days
          const endDate = new Date()
          const startDate = new Date()
          startDate.setDate(startDate.getDate() - 90)
          
          // Fetch data with pagination and client-side date filtering
          // Limited to 10000 records initially, then filtered to last 90 days
          fetchedData = await fetchAllPortTrafficData(
            {
              start: startDate,
              end: endDate,
            },
            undefined, // portId
            undefined, // country
            10000 // maxRecords - fetch more to ensure we have enough after date filtering
          )
          
          // Sort by date descending to show most recent first
          fetchedData.sort((a, b) => b.date - a.date)
          
          // If we have very few records after filtering, fetch more without date limit
          // and filter client-side (this handles edge cases)
          if (fetchedData.length < 100) {
            console.log('Few records in date range, fetching more data...')
            const moreData = await fetchAllPortTrafficData(
              undefined, // no dateRange - fetch all available
              undefined, // portId
              undefined, // country
              10000 // maxRecords
            )
            
            // Filter to last 90 days client-side
            const startTimestamp = startDate.getTime()
            const endTimestamp = endDate.getTime()
            const filtered = moreData
              .filter(d => d.date >= startTimestamp && d.date <= endTimestamp)
              .sort((a, b) => b.date - a.date)
            
            if (filtered.length > 0) {
              fetchedData = filtered
            } else {
              // If still no data in date range, use the most recent 2000 records
              fetchedData = moreData.sort((a, b) => b.date - a.date).slice(0, 2000)
            }
          }
        } catch (fetchError) {
          throw new Error(`Failed to fetch data: ${fetchError instanceof Error ? fetchError.message : 'Unknown error'}`)
        }
        
        if (!fetchedData || fetchedData.length === 0) {
          throw new Error('No data returned from API. The API might be temporarily unavailable or the query returned no results.')
        }
        
        setData(fetchedData)
        
        // Load all ports from database for search functionality
        try {
          const { fetchPortDatabase } = await import('@/api/portDatabaseService')
          const portDatabase = await fetchPortDatabase()
          setAllPorts(portDatabase)
          console.log(`Loaded ${portDatabase.size} ports from database for search`)
        } catch (portDbError) {
          console.warn('Failed to load port database for search:', portDbError)
          // Continue without port database - search will still work with traffic data
        }

      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load data')
        console.error('Error loading data:', err)
      } finally {
        setLoading(false)
      }
    }

    loadData()
  }, [])



  if (loading) {
    return (
      <main className="w-screen h-screen flex items-center justify-center bg-gray-900 text-gray-200">
        <div className="text-center">
          <div className="w-12 h-12 border-4 border-gray-700 border-t-blue-500 rounded-full animate-spin mx-auto mb-4" />
          <div>Loading port traffic data...</div>
        </div>
      </main>
    )
  }

  if (error) {
    return (
      <main className="w-screen h-screen flex items-center justify-center bg-gray-900 text-gray-200">
        <div className="p-6 bg-red-900 rounded-lg border border-red-800 max-w-md">
          <h2 className="mb-3 text-red-500">Error Loading Data</h2>
          <p className="text-red-300 mb-4">{error}</p>
          <button
            onClick={() => window.location.reload()}
            className="inline-flex items-center justify-center rounded-md bg-red-600 px-4 py-2 text-sm font-semibold text-white shadow-sm hover:bg-red-500 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-red-600 transition-colors"
          >
            Retry
          </button>
        </div>
      </main>
    )
  }

  return (
    <main className="w-screen h-screen flex bg-gray-50 text-gray-800 relative">
      {/* Vertical Navigation Rail - Left Side */}
      <div className="w-20 flex flex-col items-center justify-between py-8 z-[1000] h-full">
        <div className="flex flex-col items-center gap-3">
          {/* Logo/Avatar */}
          <div>
            <img 
              src="/gld_logo.jpeg" 
              alt="Logo" 
              className="w-10 h-10 rounded-lg object-contain"
            />
          </div>
          
          {/* Navigation buttons */}
          <div className="flex flex-col gap-1 p-1 rounded-full border border-gray-200">
            <a
              href="/"
              title="Routes View"
              className={`inline-flex h-9 w-9 items-center justify-center rounded-full border border-gray-200 ring-1 ring-inset ring-gray-900/5 transition-colors ${
                pathname === '/' 
                  ? 'bg-black text-white hover:bg-gray-900' 
                  : 'bg-white text-gray-500 hover:bg-gray-100 hover:text-gray-900'
              }`}
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="transition-colors">
                <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"></path>
                <circle cx="12" cy="10" r="3"></circle>
              </svg>
            </a>
            <a
              href="/newpage"
              title="Elevation View"
              className={`inline-flex h-9 w-9 items-center justify-center rounded-full border border-gray-200 ring-1 ring-inset ring-gray-900/5 transition-colors ${
                pathname === '/newpage' 
                  ? 'bg-black text-white hover:bg-gray-900' 
                  : 'bg-white text-gray-500 hover:bg-gray-100 hover:text-gray-900'
              }`}
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="transition-colors">
                <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path>
                <polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline>
                <line x1="12" y1="22.08" x2="12" y2="12"></line>
              </svg>
            </a>
          </div>
          
        </div>
        
        {/* User Avatar */}
        <div className="flex flex-col items-center gap-3">
          <button
            className="inline-flex h-10 w-10 items-center justify-center rounded-full border border-gray-200 bg-white hover:bg-gray-100 shadow-sm ring-1 ring-inset ring-gray-900/5 transition-colors overflow-hidden"
            title="User Profile"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
              <circle cx="12" cy="7" r="4"></circle>
            </svg>
            <img 
              src="/kevinprofile.jpeg" 
              alt="User Profile" 
              className="w-full h-full object-cover"
            />
          </button>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-hidden flex flex-col">
        <div className={`flex-1 p-6 flex overflow-hidden relative transition-all duration-500 ease-in-out ${hasCreatedPickup && routePlanningVisible && currentFormStep === 'routes' ? 'gap-6' : 'gap-0'}`}>
            {/* Route Planning Form Panel - Full Width Until First Pickup Created or when map is hidden */}
            <div 
              className={`flex-shrink-0 overflow-hidden ${
                !hasCreatedPickup || routePlanningVisible || currentFormStep !== 'routes' ? 'opacity-100' : 'opacity-0 pointer-events-none'
              }`}
              style={{ 
                width: !hasCreatedPickup || currentFormStep !== 'routes' ? '100%' : (routePlanningVisible ? '420px' : '0px'),
                transition: 'width 500ms cubic-bezier(0.4, 0, 0.2, 1), opacity 400ms ease-in-out'
              }}
            >
              <div className="bg-white/55 backdrop-blur-[50px] rounded-2xl overflow-hidden flex flex-col h-full border border-gray-200">
                  {/* Header */}
                  <div className="w-full px-4 py-3 flex items-center justify-between text-gray-900 flex-shrink-0 border-b border-white/20">
                    <div className="flex items-center gap-2">
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7" />
                      </svg>
                      <span className="text-sm font-semibold">Create Container Pickup</span>
                    </div>
                      {hasCreatedPickup && (
                        <button
                          onClick={() => setRoutePlanningVisible(false)}
                          className="p-1 hover:bg-white/20 rounded transition-colors"
                          title="Collapse panel"
                        >
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                          </svg>
                        </button>
                      )}
                  </div>

                {/* Content */}
            <div className="flex-1 overflow-y-auto px-4 py-4">
              <ContainerPickupForm
                onRouteCreate={handleRouteCreate}
                allPorts={allPorts}
                trafficData={data}
                onRouteOptionSelect={handleRouteOptionSelect}
                onStepChange={setCurrentFormStep}
                initialStep={currentFormStep}
              />
            </div>
              </div>
            </div>

            {/* Map - always show when on routes step, centered on Great Plains Industrial Park */}
            {hasCreatedPickup && currentFormStep === 'routes' && (
              <div className="flex-1 rounded-2xl overflow-hidden bg-white relative transition-all duration-500 ease-in-out border border-gray-200">
                {/* Button to show route planning panel when hidden */}
                <button
                onClick={() => setRoutePlanningVisible(true)}
                className={`absolute top-4 left-4 z-[1000] bg-white/55 backdrop-blur-[50px] rounded-lg px-4 py-2 flex items-center gap-2 text-gray-900 hover:bg-white/70 shadow-lg transition-all duration-400 ease-in-out ${
                  routePlanningVisible ? 'opacity-0 pointer-events-none scale-95 translate-x-[-10px]' : 'opacity-100 pointer-events-auto scale-100 translate-x-0'
                }`}
                title="Show route planning panel"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 20l-5.447-2.724A1 1 0 013 16.382V5.618a1 1 0 011.447-.894L9 7m0 13l6-3m-6 3V7m6 10l4.553 2.276A1 1 0 0021 18.382V7.618a1 1 0 00-.553-.894L15 4m0 13V4m0 0L9 7" />
                </svg>
                <span className="text-sm font-semibold">Show Routes</span>
              </button>
                <ContainerPickupMap
                  initialCenter={[-118.216458, 33.754185]} // Long Beach Port, Los Angeles
                  initialZoom={14} // Closer zoom to show the port area in detail
                  route={currentRoute} // Show route if selected, otherwise map stays centered on port
                  containerResized={routePlanningVisible}
                />
              </div>
            )}
          </div>
        </div>
      </main>
    )
}

export default function NewPage() {
  return (
    <Suspense fallback={
      <main className="w-screen h-screen flex items-center justify-center bg-gray-50 text-gray-800">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-gray-900 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading...</p>
        </div>
      </main>
    }>
      <NewPageContent />
    </Suspense>
  )
}
