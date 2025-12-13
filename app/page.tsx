'use client'

import { useState, useEffect } from 'react'
import { usePathname } from 'next/navigation'
import dynamic from 'next/dynamic'
import CustomerRoutesList, { CustomerRoute } from '@/components/CustomerRoutesList'
import { Route } from '@/lib/routeCalculation'
import { getDrivingRoute } from '@/lib/mapboxDirections'
import { loadSyntheticCustomerRoutes } from '@/lib/syntheticDataService'

const CustomerRoutesMap = dynamic(() => import('@/components/CustomerRoutesMap'), {
  ssr: false,
})


export default function Home() {
  const pathname = usePathname()
  const [currentRoute, setCurrentRoute] = useState<Route | null>(null)
  const [currentCompetitorRoute, setCurrentCompetitorRoute] = useState<Route | null>(null)
  const [routePlanningVisible, setRoutePlanningVisible] = useState(true)

  // Load synthetic customer routes data
  const [customerRoutes, setCustomerRoutes] = useState<CustomerRoute[]>(() => {
    return loadSyntheticCustomerRoutes()
  })

  // Fetch driving route for in-transit shipment (simplified - no port calculations)
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
          
          // Set current route for map display
          if (!currentRoute) {
            setCurrentRoute(updatedRoute)
          }
        }
      } else if (!currentRoute) {
        // Route already has coordinates, just display it
        setCurrentRoute(inTransitRoute.route)
      }
    }
    
    fetchRouteForInTransit()
  }, [customerRoutes, currentRoute])

  // Handle route selection from CustomerRoutesList
  const handleRouteSelect = (route: CustomerRoute) => {
    setCurrentRoute(route.route || null)
    setCurrentCompetitorRoute(route.competitorRoute || null)
  }


  return (
    <main className="w-screen h-screen flex bg-gray-50 text-gray-800 relative">
      {/* Vertical Navigation Rail - Left Side */}
      <div className="w-20 flex flex-col items-center justify-between py-5 z-[1000] h-full">
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
          <div className="flex flex-col gap-2">
            <a
              href="/"
              title="Routes View"
              className={`inline-flex h-9 w-9 items-center justify-center rounded-full border border-gray-200 shadow-sm ring-1 ring-inset ring-gray-900/5 transition-colors ${
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
              className={`inline-flex h-9 w-9 items-center justify-center rounded-full border border-gray-200 shadow-sm ring-1 ring-inset ring-gray-900/5 transition-colors ${
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
            className="inline-flex h-10 w-10 items-center justify-center rounded-full border border-gray-200 bg-white text-gray-500 hover:bg-gray-100 hover:text-gray-900 shadow-sm ring-1 ring-inset ring-gray-900/5 transition-colors"
            title="User Profile"
          >
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path>
              <circle cx="12" cy="7" r="4"></circle>
            </svg>
          </button>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-hidden flex flex-col">
        <div className={`flex-1 p-6 flex overflow-hidden relative transition-all duration-500 ease-in-out ${routePlanningVisible ? 'gap-6' : 'gap-0'}`}>
          {/* Route Planning Form Panel - Left Side */}
          <div 
            className={`flex-shrink-0 overflow-hidden ${
              routePlanningVisible ? 'opacity-100' : 'opacity-0 pointer-events-none'
            }`}
            style={{ 
              width: routePlanningVisible ? '420px' : '0px',
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
                    <span className="text-sm font-semibold">Customer Routes</span>
                  </div>
                  <button
                    onClick={() => setRoutePlanningVisible(false)}
                    className="p-1 hover:bg-white/20 rounded transition-colors"
                    title="Collapse panel"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                </div>

              {/* Content */}
          <div className="flex-1 overflow-y-auto px-4 py-4">
            <CustomerRoutesList
              routes={customerRoutes}
              onRouteSelect={handleRouteSelect}
            />
          </div>
            </div>
          </div>

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
            <CustomerRoutesMap
              initialCenter={[-95.194034, 37.332823]} // Great Plains Industrial Park, Kansas
              initialZoom={14} // Closer zoom to show the industrial park area in detail
              route={currentRoute}
              competitorRoute={currentCompetitorRoute}
              containerResized={routePlanningVisible}
            />
          </div>
        </div>
      </div>
    </main>
  )
}
