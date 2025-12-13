/**
 * Pricing configuration for route cost calculations
 * Based on ton-mile pricing model
 */

// Our pricing: $0.08 to $0.20 per ton-mile
export const PRICING = {
  // Base price per trip (applies to all routes)
  BASE_PRICE: 150,    // $150 base price per trip
  
  // Our pricing tiers (per ton-mile)
  ECONOMY: 0.08,      // $0.08 per ton-mile - most cost-effective
  STANDARD: 0.12,     // $0.12 per ton-mile - standard rate
  PREMIUM: 0.14,      // $0.14 per ton-mile - early/late pickup
  EXPRESS: 0.20,      // $0.20 per ton-mile - fastest/priority
  
  // Competitor pricing
  COMPETITOR: 2.30,   // $2.30 per ton-mile - competitor rate
  COMPETITOR_BASE: 200, // $200 base price per trip
} as const

/**
 * Calculate cost based on ton-mile pricing with base price
 * Formula: Total Cost = Base Price + (Weight in tons × Distance in miles × Rate per ton-mile)
 * 
 * Step 1: Calculate Total Ton-Miles = Weight (tons) × Distance (miles)
 * Step 2: Calculate Ton-Mile Cost = Total Ton-Miles × Rate Per Ton-Mile
 * Step 3: Calculate Total Cost = Base Price + Ton-Mile Cost
 * 
 * Example: $150 base + (20 tons × 500 miles × $0.08/ton-mile) = $150 + $800 = $950
 * 
 * @param distanceMiles Distance in miles
 * @param weightTons Weight in tons
 * @param pricePerTonMile Price per ton-mile (default: STANDARD)
 * @param basePrice Base price per trip (default: BASE_PRICE)
 * @returns Total cost in dollars
 */
export function calculateTonMileCost(
  distanceMiles: number,
  weightTons: number,
  pricePerTonMile: number = PRICING.STANDARD,
  basePrice: number = PRICING.BASE_PRICE
): number {
  // Total Ton-Miles = Weight (tons) × Distance (miles)
  const totalTonMiles = weightTons * distanceMiles
  
  // Ton-Mile Cost = Total Ton-Miles × Rate Per Ton-Mile
  const tonMileCost = totalTonMiles * pricePerTonMile
  
  // Total Cost = Base Price + Ton-Mile Cost
  return basePrice + tonMileCost
}

/**
 * Get pricing tier for route type
 */
export function getPricingForRouteType(routeType: 'economy' | 'standard' | 'early' | 'late' | 'express'): number {
  switch (routeType) {
    case 'economy':
      return PRICING.ECONOMY
    case 'standard':
      return PRICING.STANDARD
    case 'early':
    case 'late':
      return PRICING.PREMIUM
    case 'express':
      return PRICING.EXPRESS
    default:
      return PRICING.STANDARD
  }
}

/**
 * Calculate competitor cost for comparison
 */
export function calculateCompetitorCost(distanceMiles: number, weightTons: number): number {
  return calculateTonMileCost(distanceMiles, weightTons, PRICING.COMPETITOR, PRICING.COMPETITOR_BASE)
}
