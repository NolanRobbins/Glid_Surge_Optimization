"""
Dispatch Scheduler for Glid Vehicles
=====================================
Optimizes dispatch timing to minimize dwell time and maximize efficiency.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import GLID_VEHICLE, OPTIMIZATION


@dataclass
class DispatchWindow:
    """Optimal dispatch window recommendation."""
    start_time: datetime
    end_time: datetime
    priority: str  # 'high', 'medium', 'low'
    expected_dwell_reduction_hours: float
    expected_load_factor: float
    route_recommendation: str
    notes: List[str]


@dataclass
class DispatchPlan:
    """Complete dispatch plan for a time period."""
    windows: List[DispatchWindow]
    total_containers: int
    expected_cost_savings: float
    utilization_score: float


class DispatchScheduler:
    """
    Scheduler for optimizing Glid vehicle dispatch.
    
    Determines optimal time windows for dispatching vehicles
    from customer sites to ports/rail terminals.
    """
    
    def __init__(self):
        self.forecast_cache: Dict[str, pd.DataFrame] = {}
    
    def find_optimal_windows(
        self,
        surge_forecast: pd.DataFrame,
        dwell_predictions: pd.DataFrame,
        location: str,
        horizon_hours: int = 72,
        num_windows: int = 3
    ) -> List[DispatchWindow]:
        """
        Find optimal dispatch windows based on forecasts.
        
        Args:
            surge_forecast: Surge predictions
            dwell_predictions: Dwell time predictions
            location: Location name
            horizon_hours: Planning horizon
            num_windows: Number of windows to recommend
            
        Returns:
            List of DispatchWindow recommendations
        """
        windows = []
        
        # Calculate score for each hour in horizon
        now = datetime.now()
        scores = []
        
        for hour_offset in range(0, horizon_hours, 4):  # Check every 4 hours
            check_time = now + timedelta(hours=hour_offset)
            
            # Get predictions for this time (simplified)
            surge_score = self._get_surge_score(surge_forecast, check_time, location)
            dwell_score = self._get_dwell_score(dwell_predictions, check_time, location)
            
            # Lower surge + lower dwell = better window
            combined_score = (1 - surge_score) * 0.6 + (1 - dwell_score) * 0.4
            
            scores.append({
                'time': check_time,
                'score': combined_score,
                'surge': surge_score,
                'dwell': dwell_score
            })
        
        # Sort by score (higher is better)
        scores.sort(key=lambda x: x['score'], reverse=True)
        
        # Create top windows
        for i, score_data in enumerate(scores[:num_windows]):
            priority = 'high' if i == 0 else ('medium' if i == 1 else 'low')
            
            window = DispatchWindow(
                start_time=score_data['time'],
                end_time=score_data['time'] + timedelta(hours=OPTIMIZATION.dispatch_window_hours),
                priority=priority,
                expected_dwell_reduction_hours=(1 - score_data['dwell']) * OPTIMIZATION.max_dwell_hours,
                expected_load_factor=0.7 + score_data['score'] * 0.25,  # 70-95%
                route_recommendation=self._get_route_recommendation(score_data),
                notes=self._generate_notes(score_data)
            )
            windows.append(window)
        
        return windows
    
    def _get_surge_score(
        self,
        forecast: pd.DataFrame,
        check_time: datetime,
        location: str
    ) -> float:
        """Get surge intensity score (0-1) for a time."""
        # Simplified: return random-ish score for now
        # In production, this would query the actual forecast
        hour = check_time.hour
        
        # Peak hours (8-10 AM, 2-4 PM) have higher surge
        if 8 <= hour <= 10 or 14 <= hour <= 16:
            return 0.7 + np.random.random() * 0.2
        elif 22 <= hour or hour <= 5:
            return 0.2 + np.random.random() * 0.2
        else:
            return 0.4 + np.random.random() * 0.2
    
    def _get_dwell_score(
        self,
        predictions: pd.DataFrame,
        check_time: datetime,
        location: str
    ) -> float:
        """Get expected dwell time score (0-1) for a time."""
        hour = check_time.hour
        day = check_time.weekday()
        
        # Weekend = lower dwell (less congestion)
        weekend_factor = 0.7 if day >= 5 else 1.0
        
        # Night = lower dwell
        if 22 <= hour or hour <= 5:
            return 0.3 * weekend_factor
        elif 6 <= hour <= 9:
            return 0.6 * weekend_factor
        else:
            return 0.5 * weekend_factor
    
    def _get_route_recommendation(self, score_data: Dict) -> str:
        """Generate route recommendation based on conditions."""
        if score_data['surge'] < 0.4:
            return "Rail-priority route: Low port congestion, maximize rail utilization"
        elif score_data['surge'] > 0.7:
            return "Hybrid route: Use street bypass to avoid port congestion"
        else:
            return "Standard route: Balanced rail/street approach"
    
    def _generate_notes(self, score_data: Dict) -> List[str]:
        """Generate explanatory notes for a window."""
        notes = []
        
        if score_data['surge'] < 0.4:
            notes.append("✓ Low port congestion expected")
        elif score_data['surge'] > 0.7:
            notes.append("⚠ High port congestion - consider delay")
        
        if score_data['dwell'] < 0.4:
            notes.append("✓ Fast turnaround expected")
        elif score_data['dwell'] > 0.6:
            notes.append("⚠ Extended dwell time likely")
        
        if score_data['time'].weekday() >= 5:
            notes.append("📅 Weekend operations - reduced staffing")
        
        if score_data['time'].hour < 6 or score_data['time'].hour > 22:
            notes.append("🌙 Night operations - verify terminal hours")
        
        return notes
    
    def create_dispatch_plan(
        self,
        location: str,
        container_count: int,
        surge_forecast: pd.DataFrame = None,
        dwell_predictions: pd.DataFrame = None
    ) -> DispatchPlan:
        """
        Create comprehensive dispatch plan.
        
        Args:
            location: Location name
            container_count: Number of containers to move
            surge_forecast: Surge predictions
            dwell_predictions: Dwell predictions
            
        Returns:
            Complete DispatchPlan
        """
        # Get optimal windows
        windows = self.find_optimal_windows(
            surge_forecast or pd.DataFrame(),
            dwell_predictions or pd.DataFrame(),
            location
        )
        
        # Calculate expected savings
        baseline_dwell_hours = 48  # Traditional
        optimized_dwell = sum(
            OPTIMIZATION.max_dwell_hours - w.expected_dwell_reduction_hours
            for w in windows
        ) / len(windows)
        
        hours_saved = baseline_dwell_hours - optimized_dwell
        storage_cost_per_hour = 15  # $/container/hour
        cost_savings = hours_saved * storage_cost_per_hour * container_count
        
        # Calculate utilization score
        avg_load_factor = sum(w.expected_load_factor for w in windows) / len(windows)
        
        return DispatchPlan(
            windows=windows,
            total_containers=container_count,
            expected_cost_savings=cost_savings,
            utilization_score=avg_load_factor
        )


if __name__ == "__main__":
    # Test dispatch scheduling
    scheduler = DispatchScheduler()
    
    plan = scheduler.create_dispatch_plan(
        location="Port of Los Angeles",
        container_count=100
    )
    
    print(f"Dispatch Plan for {plan.total_containers} containers:")
    print(f"Expected savings: ${plan.expected_cost_savings:,.0f}")
    print(f"Utilization score: {plan.utilization_score:.1%}")
    print("\nOptimal Windows:")
    
    for window in plan.windows:
        print(f"\n  {window.priority.upper()}: {window.start_time.strftime('%a %H:%M')} - {window.end_time.strftime('%H:%M')}")
        print(f"    Route: {window.route_recommendation}")
        for note in window.notes:
            print(f"    {note}")







