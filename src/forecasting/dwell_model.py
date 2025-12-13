"""
Dwell Time Prediction Model
===========================
Predicts expected dwell time at ports/warehouses.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
import joblib
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import MODELS_DIR


@dataclass
class DwellPrediction:
    """Dwell time prediction result."""
    predicted_hours: float
    confidence_low: float
    confidence_high: float
    factors: Dict[str, float]


class DwellTimeModel:
    """
    Model for predicting container dwell time.
    
    Estimates how long cargo will sit idle based on:
    - Current port congestion
    - Rail capacity availability
    - Time of day/week
    - Weather conditions
    """
    
    def __init__(self):
        self.model = None
        self.feature_cols: List[str] = []
        self.is_fitted = False
    
    def _build_features(
        self,
        congestion_level: float,
        rail_capacity_pct: float,
        hour_of_day: int,
        day_of_week: int,
        is_rainy: bool = False,
        is_high_wind: bool = False
    ) -> Dict[str, float]:
        """Build feature dictionary for prediction."""
        return {
            'congestion_level': congestion_level,
            'rail_capacity_pct': rail_capacity_pct,
            'hour_of_day': hour_of_day,
            'day_of_week': day_of_week,
            'is_weekend': 1 if day_of_week >= 5 else 0,
            'is_night': 1 if hour_of_day < 6 or hour_of_day > 22 else 0,
            'is_rainy': int(is_rainy),
            'is_high_wind': int(is_high_wind),
            'hour_sin': np.sin(2 * np.pi * hour_of_day / 24),
            'hour_cos': np.cos(2 * np.pi * hour_of_day / 24),
        }
    
    def fit(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Fit dwell time model on historical data.
        
        Args:
            df: DataFrame with dwell time records
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.feature_cols = [
            'congestion_level', 'rail_capacity_pct', 'hour_of_day',
            'day_of_week', 'is_weekend', 'is_night', 'is_rainy',
            'is_high_wind', 'hour_sin', 'hour_cos'
        ]
        
        X = df[self.feature_cols]
        y = df['dwell_hours']
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train with progress indicator
        print("  Training dwell time model...")
        with tqdm(total=100, desc="  GradientBoosting", unit="trees") as pbar:
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                verbose=0
            )
            self.model.fit(X_train, y_train)
            pbar.update(100)
        
        y_pred = self.model.predict(X_val)
        
        metrics = {
            'mae': mean_absolute_error(y_val, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_val, y_pred))
        }
        
        self.is_fitted = True
        return metrics
    
    def predict(
        self,
        congestion_level: float,
        rail_capacity_pct: float,
        hour_of_day: int,
        day_of_week: int,
        is_rainy: bool = False,
        is_high_wind: bool = False
    ) -> DwellPrediction:
        """
        Predict dwell time for given conditions.
        
        Args:
            congestion_level: 0-1 scale of port congestion
            rail_capacity_pct: Available rail capacity (0-100%)
            hour_of_day: Hour (0-23)
            day_of_week: Day (0=Monday, 6=Sunday)
            is_rainy: Weather flag
            is_high_wind: Weather flag
            
        Returns:
            DwellPrediction object
        """
        if not self.is_fitted:
            # Return heuristic estimate
            base_hours = 24
            base_hours += congestion_level * 24
            base_hours -= (rail_capacity_pct / 100) * 12
            if is_rainy:
                base_hours += 4
            if is_high_wind:
                base_hours += 8
            
            return DwellPrediction(
                predicted_hours=max(4, base_hours),
                confidence_low=max(2, base_hours - 8),
                confidence_high=base_hours + 12,
                factors={'congestion': congestion_level, 'rail_capacity': rail_capacity_pct}
            )
        
        features = self._build_features(
            congestion_level, rail_capacity_pct, hour_of_day,
            day_of_week, is_rainy, is_high_wind
        )
        
        X = pd.DataFrame([features])[self.feature_cols]
        prediction = self.model.predict(X)[0]
        
        # Estimate confidence interval (simple ± 20%)
        return DwellPrediction(
            predicted_hours=max(0, prediction),
            confidence_low=max(0, prediction * 0.8),
            confidence_high=prediction * 1.2,
            factors=features
        )
    
    def save(self, path: Path = None):
        """Save model to disk."""
        if path is None:
            path = MODELS_DIR / 'dwell_model.joblib'
        
        joblib.dump({
            'model': self.model,
            'feature_cols': self.feature_cols
        }, path)
    
    def load(self, path: Path = None):
        """Load model from disk."""
        if path is None:
            path = MODELS_DIR / 'dwell_model.joblib'
        
        data = joblib.load(path)
        self.model = data['model']
        self.feature_cols = data['feature_cols']
        self.is_fitted = True


if __name__ == "__main__":
    # Test heuristic prediction
    model = DwellTimeModel()
    
    result = model.predict(
        congestion_level=0.7,
        rail_capacity_pct=40,
        hour_of_day=14,
        day_of_week=2,
        is_rainy=True
    )
    
    print(f"Predicted dwell time: {result.predicted_hours:.1f} hours")
    print(f"Confidence interval: {result.confidence_low:.1f} - {result.confidence_high:.1f} hours")

