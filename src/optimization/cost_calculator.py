"""
Cost Calculator for Glid Optimization
======================================
Calculates and compares costs between traditional drayage and Glid.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import GLID_VEHICLE, OPTIMIZATION


@dataclass
class CostBreakdown:
    """Detailed cost breakdown."""
    dwell_cost: float
    transportation_cost: float
    empty_mile_cost: float
    penalty_cost: float
    energy_cost: float
    total_cost: float


@dataclass
class SavingsReport:
    """Report comparing Glid vs traditional costs."""
    traditional_cost: CostBreakdown
    glid_cost: CostBreakdown
    savings: float
    savings_percentage: float
    annual_projection: float


class CostCalculator:
    """
    Calculator for comparing Glid vs traditional drayage costs.
    
    Models the financial impact of:
    - Reduced dwell time
    - Eliminated empty miles
    - Fewer late penalties
    - Energy efficiency
    """
    
    # Default cost parameters ($/unit)
    DEFAULT_COSTS = {
        'storage_per_hour': 15.0,      # Container storage
        'truck_per_mile': 3.50,         # Traditional truck
        'glid_per_mile': 1.80,          # Glid vehicle
        'empty_mile_cost': 2.00,        # Deadhead cost
        'late_penalty': 250.0,          # Per late delivery
        'energy_per_mile': 0.45,        # Fuel/electricity
        'glid_energy_per_mile': 0.15,   # Electric efficiency
    }
    
    def __init__(self, cost_params: Dict[str, float] = None):
        """
        Initialize calculator with cost parameters.
        
        Args:
            cost_params: Optional custom cost parameters
        """
        self.costs = {**self.DEFAULT_COSTS, **(cost_params or {})}
    
    def calculate_traditional_costs(
        self,
        containers: int,
        distance_miles: float,
        dwell_hours: float = 48.0,
        empty_mile_pct: float = 0.35,
        late_delivery_pct: float = 0.18
    ) -> CostBreakdown:
        """
        Calculate traditional drayage costs.
        
        Args:
            containers: Number of containers
            distance_miles: One-way distance
            dwell_hours: Average dwell time
            empty_mile_pct: Percentage of empty miles
            late_delivery_pct: Percentage of late deliveries
            
        Returns:
            CostBreakdown for traditional approach
        """
        round_trip_miles = distance_miles * 2
        
        dwell_cost = containers * dwell_hours * self.costs['storage_per_hour']
        
        transportation_cost = containers * round_trip_miles * self.costs['truck_per_mile']
        
        empty_miles = round_trip_miles * empty_mile_pct
        empty_mile_cost = containers * empty_miles * self.costs['empty_mile_cost']
        
        late_deliveries = int(containers * late_delivery_pct)
        penalty_cost = late_deliveries * self.costs['late_penalty']
        
        energy_cost = containers * round_trip_miles * self.costs['energy_per_mile']
        
        total = dwell_cost + transportation_cost + empty_mile_cost + penalty_cost + energy_cost
        
        return CostBreakdown(
            dwell_cost=dwell_cost,
            transportation_cost=transportation_cost,
            empty_mile_cost=empty_mile_cost,
            penalty_cost=penalty_cost,
            energy_cost=energy_cost,
            total_cost=total
        )
    
    def calculate_glid_costs(
        self,
        containers: int,
        distance_miles: float,
        dwell_hours: float = 12.0,  # Optimized
        empty_mile_pct: float = 0.10,  # Backhaul optimization
        late_delivery_pct: float = 0.05  # Better scheduling
    ) -> CostBreakdown:
        """
        Calculate Glid optimized costs.
        
        Args:
            containers: Number of containers
            distance_miles: One-way distance
            dwell_hours: Optimized dwell time
            empty_mile_pct: Reduced empty miles
            late_delivery_pct: Reduced late deliveries
            
        Returns:
            CostBreakdown for Glid approach
        """
        round_trip_miles = distance_miles * 2
        
        dwell_cost = containers * dwell_hours * self.costs['storage_per_hour']
        
        transportation_cost = containers * round_trip_miles * self.costs['glid_per_mile']
        
        empty_miles = round_trip_miles * empty_mile_pct
        empty_mile_cost = containers * empty_miles * self.costs['empty_mile_cost']
        
        late_deliveries = int(containers * late_delivery_pct)
        penalty_cost = late_deliveries * self.costs['late_penalty']
        
        energy_cost = containers * round_trip_miles * self.costs['glid_energy_per_mile']
        
        total = dwell_cost + transportation_cost + empty_mile_cost + penalty_cost + energy_cost
        
        return CostBreakdown(
            dwell_cost=dwell_cost,
            transportation_cost=transportation_cost,
            empty_mile_cost=empty_mile_cost,
            penalty_cost=penalty_cost,
            energy_cost=energy_cost,
            total_cost=total
        )
    
    def calculate_savings(
        self,
        containers: int,
        distance_miles: float,
        traditional_dwell: float = 48.0,
        glid_dwell: float = 12.0,
        days_per_year: int = 250
    ) -> SavingsReport:
        """
        Calculate savings comparing Glid to traditional drayage.
        
        Args:
            containers: Containers per day
            distance_miles: Route distance
            traditional_dwell: Traditional dwell hours
            glid_dwell: Glid dwell hours
            days_per_year: Operating days
            
        Returns:
            SavingsReport with comparison
        """
        traditional = self.calculate_traditional_costs(
            containers, distance_miles, traditional_dwell
        )
        
        glid = self.calculate_glid_costs(
            containers, distance_miles, glid_dwell
        )
        
        daily_savings = traditional.total_cost - glid.total_cost
        savings_pct = (daily_savings / traditional.total_cost) * 100 if traditional.total_cost > 0 else 0
        
        return SavingsReport(
            traditional_cost=traditional,
            glid_cost=glid,
            savings=daily_savings,
            savings_percentage=savings_pct,
            annual_projection=daily_savings * days_per_year
        )
    
    def format_savings_report(self, report: SavingsReport) -> str:
        """Format savings report as readable string."""
        lines = [
            "=" * 60,
            "GLID COST SAVINGS ANALYSIS",
            "=" * 60,
            "",
            "TRADITIONAL DRAYAGE COSTS (Daily):",
            f"  Dwell/Storage:      ${report.traditional_cost.dwell_cost:,.2f}",
            f"  Transportation:     ${report.traditional_cost.transportation_cost:,.2f}",
            f"  Empty Miles:        ${report.traditional_cost.empty_mile_cost:,.2f}",
            f"  Late Penalties:     ${report.traditional_cost.penalty_cost:,.2f}",
            f"  Energy/Fuel:        ${report.traditional_cost.energy_cost:,.2f}",
            f"  TOTAL:              ${report.traditional_cost.total_cost:,.2f}",
            "",
            "GLID OPTIMIZED COSTS (Daily):",
            f"  Dwell/Storage:      ${report.glid_cost.dwell_cost:,.2f}",
            f"  Transportation:     ${report.glid_cost.transportation_cost:,.2f}",
            f"  Empty Miles:        ${report.glid_cost.empty_mile_cost:,.2f}",
            f"  Late Penalties:     ${report.glid_cost.penalty_cost:,.2f}",
            f"  Energy/Fuel:        ${report.glid_cost.energy_cost:,.2f}",
            f"  TOTAL:              ${report.glid_cost.total_cost:,.2f}",
            "",
            "-" * 60,
            f"DAILY SAVINGS:        ${report.savings:,.2f} ({report.savings_percentage:.1f}%)",
            f"ANNUAL SAVINGS:       ${report.annual_projection:,.2f}",
            "=" * 60,
        ]
        return "\n".join(lines)


if __name__ == "__main__":
    # Test cost calculation
    calc = CostCalculator()
    
    report = calc.calculate_savings(
        containers=50,
        distance_miles=45
    )
    
    print(calc.format_savings_report(report))






