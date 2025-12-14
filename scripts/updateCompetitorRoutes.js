/**
 * Script to update competitor routes with real street-following routes from Mapbox
 * This ensures competitor routes follow actual roads on the map
 */

const fs = require('fs');
const path = require('path');

// Load environment variables from .env.local manually
const envPath = path.join(__dirname, '../.env.local');
if (fs.existsSync(envPath)) {
  const envContent = fs.readFileSync(envPath, 'utf8');
  const envLines = envContent.split('\n');
  for (const line of envLines) {
    const trimmedLine = line.trim();
    // Skip comments and empty lines
    if (trimmedLine && !trimmedLine.startsWith('#')) {
      const [key, ...valueParts] = trimmedLine.split('=');
      if (key && valueParts.length > 0) {
        const value = valueParts.join('=').trim().replace(/^["']|["']$/g, ''); // Remove quotes if present
        process.env[key.trim()] = value;
      }
    }
  }
}

// Get token from environment variable or command line argument
const MAPBOX_TOKEN = process.argv[2] || process.env.NEXT_PUBLIC_MAPBOX_TOKEN || '';

if (!MAPBOX_TOKEN) {
  console.error('Error: Mapbox token not found');
  console.error('Usage: node scripts/updateCompetitorRoutes.js [MAPBOX_TOKEN]');
  console.error('Or set NEXT_PUBLIC_MAPBOX_TOKEN in .env.local file');
  process.exit(1);
}

const DATA_FILE = path.join(__dirname, '../data/syntheticCustomerRoutes.json');

/**
 * Calculate distance between two points using Haversine formula
 * @param point1 [longitude, latitude]
 * @param point2 [longitude, latitude]
 * @returns distance in kilometers
 */
function calculateDistance(point1, point2) {
  const R = 6371; // Earth's radius in kilometers
  const [lng1, lat1] = point1;
  const [lng2, lat2] = point2;

  const dLat = ((lat2 - lat1) * Math.PI) / 180;
  const dLon = ((lng2 - lng1) * Math.PI) / 180;

  const a =
    Math.sin(dLat / 2) * Math.sin(dLat / 2) +
    Math.cos((lat1 * Math.PI) / 180) *
      Math.cos((lat2 * Math.PI) / 180) *
      Math.sin(dLon / 2) *
      Math.sin(dLon / 2);

  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return R * c;
}

/**
 * Calculate total distance along a path of coordinates
 * @param coordinates Array of [lng, lat] coordinates
 * @returns total distance in kilometers
 */
function calculatePathDistance(coordinates) {
  if (coordinates.length < 2) {
    return 0;
  }

  let totalDistance = 0;
  for (let i = 0; i < coordinates.length - 1; i++) {
    totalDistance += calculateDistance(coordinates[i], coordinates[i + 1]);
  }
  return totalDistance;
}

/**
 * Fetch driving route from Mapbox Directions API
 * Returns both coordinates and distance
 */
async function getDrivingRoute(origin, destination) {
  const [lng1, lat1] = origin;
  const [lng2, lat2] = destination;
  const coordinates = `${lng1},${lat1};${lng2},${lat2}`;
  const url = `https://api.mapbox.com/directions/v5/mapbox/driving/${coordinates}?geometries=geojson&access_token=${MAPBOX_TOKEN}`;
  
  try {
    const response = await fetch(url);
    
    if (!response.ok) {
      const errorText = await response.text();
      console.warn(`Mapbox API error (${response.status}): ${errorText}`);
      return null;
    }
    
    const data = await response.json();
    
    if (data.routes && data.routes.length > 0) {
      const route = data.routes[0];
      const routeCoordinates = route.geometry.coordinates;
      // Mapbox returns distance in meters, convert to kilometers
      const distanceKm = route.distance ? route.distance / 1000 : calculatePathDistance(routeCoordinates);
      
      return {
        coordinates: routeCoordinates,
        distance: distanceKm
      };
    }
    
    return null;
  } catch (error) {
    console.warn(`Error fetching route: ${error.message}`);
    return null;
  }
}

/**
 * Update competitor routes with real street routes
 */
async function updateCompetitorRoutes() {
  console.log('Loading routes data...');
  const data = JSON.parse(fs.readFileSync(DATA_FILE, 'utf8'));
  
  const routesToUpdate = data.routes.filter(route => route.route && !route.competitorRoute);
  const routesWithCompetitor = data.routes.filter(route => route.competitorRoute);
  
  console.log(`Found ${routesToUpdate.length} routes without competitor routes`);
  console.log(`Found ${routesWithCompetitor.length} routes with competitor routes to update`);
  
  let updated = 0;
  let failed = 0;
  
  // Update existing competitor routes
  for (const route of routesWithCompetitor) {
    if (!route.route || !route.competitorRoute) continue;
    
    const origin = route.route.origin.coordinates;
    const destination = route.route.destination.coordinates;
    
    console.log(`\nUpdating route ${route.id}: ${route.originPort} → ${route.destinationPort}`);
    console.log(`  Origin: [${origin[0]}, ${origin[1]}]`);
    console.log(`  Destination: [${destination[0]}, ${destination[1]}]`);
    
    const routeData = await getDrivingRoute(origin, destination);
    
    if (routeData && routeData.coordinates && routeData.coordinates.length > 0) {
      route.competitorRoute.coordinates = routeData.coordinates;
      route.competitorRoute.distance = routeData.distance;
      console.log(`  ✓ Updated with ${routeData.coordinates.length} road coordinates`);
      console.log(`  ✓ Distance: ${routeData.distance.toFixed(2)} km`);
      updated++;
      
      // Add small delay to avoid rate limiting
      await new Promise(resolve => setTimeout(resolve, 100));
    } else {
      // Calculate distance from existing coordinates if available
      if (route.competitorRoute.coordinates && route.competitorRoute.coordinates.length > 0) {
        route.competitorRoute.distance = calculatePathDistance(route.competitorRoute.coordinates);
        console.log(`  ✗ Failed to fetch new route, calculated distance: ${route.competitorRoute.distance.toFixed(2)} km`);
      } else {
        console.log(`  ✗ Failed to fetch road route (keeping existing waypoints)`);
      }
      failed++;
    }
  }
  
  // Add competitor routes for routes that don't have them
  for (const route of routesToUpdate) {
    if (!route.route) continue;
    
    const origin = route.route.origin.coordinates;
    const destination = route.route.destination.coordinates;
    
    console.log(`\nAdding competitor route for route ${route.id}: ${route.originPort} → ${route.destinationPort}`);
    
    const routeData = await getDrivingRoute(origin, destination);
    
    if (routeData && routeData.coordinates && routeData.coordinates.length > 0) {
      route.competitorRoute = {
        origin: route.route.origin,
        destination: route.route.destination,
        coordinates: routeData.coordinates,
        distance: routeData.distance
      };
      console.log(`  ✓ Added competitor route with ${routeData.coordinates.length} road coordinates`);
      console.log(`  ✓ Distance: ${routeData.distance.toFixed(2)} km`);
      updated++;
      
      // Add small delay to avoid rate limiting
      await new Promise(resolve => setTimeout(resolve, 100));
    } else {
      console.log(`  ✗ Failed to fetch road route`);
      failed++;
    }
  }
  
  // Save updated data
  console.log(`\n\nSaving updated routes...`);
  fs.writeFileSync(DATA_FILE, JSON.stringify(data, null, 2));
  
  console.log(`\n✓ Update complete!`);
  console.log(`  Updated: ${updated} routes`);
  console.log(`  Failed: ${failed} routes`);
}

// Run the update
updateCompetitorRoutes().catch(error => {
  console.error('Fatal error:', error);
  process.exit(1);
});

