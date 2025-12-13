"""
Surge Prediction Model
======================
Predicts port cargo volume surges 24-72 hours ahead.

Uses BOTH:
1. Time-series features (lags, rolling averages, seasonality)
2. Graph topology features (centrality, connectivity, congestion propagation)
"""

import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from pathlib import Path
from tqdm import tqdm

try:
    import xgboost as xgb
    import lightgbm as lgb
    HAS_BOOSTING = True
except ImportError:
    HAS_BOOSTING = False

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import FORECASTING, MODELS_DIR
from .features import build_forecasting_features
from .graph_features import GraphEnhancedForecaster, build_graph_feature_matrix


@dataclass
class PredictionResult:
    """Results from surge prediction."""
    predictions: np.ndarray
    prediction_horizon_hours: int
    model_name: str
    metrics: Dict[str, float]


class SurgePredictionModel:
    """
    Model for predicting port cargo surges.
    
    Uses gradient boosting (XGBoost/LightGBM) to predict
    port activity levels 24, 48, and 72 hours ahead.
    
    HYBRID APPROACH:
    - Time-series features: lags, rolling stats, seasonality
    - Graph features: node centrality, connectivity, congestion propagation
    
    The graph provides structural context that pure time-series models miss:
    - A port with high betweenness centrality is a critical chokepoint
    - Congestion propagates through connected nodes
    - Well-connected nodes have more routing alternatives
    """
    
    def __init__(
        self,
        model_type: str = 'xgboost',
        prediction_horizons: List[int] = None,
        graph: nx.Graph = None
    ):
        """
        Initialize surge prediction model.
        
        Args:
            model_type: 'xgboost' or 'lightgbm'
            prediction_horizons: Hours ahead to predict (default: [24, 48, 72])
            graph: Optional transportation network graph for topology features
        """
        self.model_type = model_type
        self.prediction_horizons = prediction_horizons or FORECASTING.prediction_horizons
        self.models: Dict[int, Any] = {}
        self.feature_cols: List[str] = []
        self.is_fitted = False
        self.graph = graph
        self.graph_enhancer = GraphEnhancedForecaster(graph) if graph else None
        self.graph_features_df = None
    
    def _create_model(self):
        """Create a new model instance."""
        if self.model_type == 'xgboost' and HAS_BOOSTING:
            return xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'lightgbm' and HAS_BOOSTING:
            return lgb.LGBMRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        else:
            # Fallback to sklearn
            from sklearn.ensemble import GradientBoostingRegressor
            return GradientBoostingRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
    
    def fit(
        self,
        df: pd.DataFrame,
        target_col: str = 'portcalls',
        group_col: str = 'portname',
        weather_df: pd.DataFrame = None
    ) -> Dict[int, Dict[str, float]]:
        """
        Fit models for each prediction horizon.
        
        Combines time-series features with graph topology features.
        
        Args:
            df: Training data
            target_col: Target variable column
            group_col: Grouping column
            weather_df: Optional weather data
            
        Returns:
            Dictionary of metrics for each horizon
        """
        # Build time-series features (without weather if dates don't match)
        try:
            feature_df, self.feature_cols = build_forecasting_features(
                df, weather_df, target_col, group_col
            )
        except Exception as e:
            print(f"  Warning: Weather merge failed ({e}), continuing without weather")
            feature_df, self.feature_cols = build_forecasting_features(
                df, None, target_col, group_col
            )
        
        # ADD GRAPH TOPOLOGY FEATURES if graph is available
        if self.graph_enhancer is not None and self.graph is not None:
            print("  Adding graph topology features...")
            locations = feature_df[group_col].unique().tolist()
            
            try:
                # Compute and cache graph features
                self.graph_features_df = self.graph_enhancer.compute_graph_features(locations)
                
                # Merge graph features into training data
                feature_df = self.graph_enhancer.add_graph_features_to_data(
                    feature_df, location_col=group_col
                )
                
                # Add graph feature columns to feature list
                graph_cols = [c for c in feature_df.columns if c.startswith('graph_')]
                self.feature_cols.extend(graph_cols)
                print(f"  Added {len(graph_cols)} graph features: {graph_cols}")
            except Exception as e:
                print(f"  Warning: Graph feature extraction failed ({e})")
        
        # Ensure feature columns don't contain NaN-only columns
        valid_cols = []
        for col in self.feature_cols:
            if col in feature_df.columns and feature_df[col].notna().any():
                valid_cols.append(col)
        self.feature_cols = valid_cols
        
        # Fill any remaining NaN
        feature_df[self.feature_cols] = feature_df[self.feature_cols].fillna(0)
        
        metrics = {}
        
        for horizon in tqdm(self.prediction_horizons, desc="Training horizons", unit="model"):
            tqdm.write(f"  Training model for {horizon}-hour prediction...")
            
            # Create target shifted by horizon (in days for daily data)
            horizon_days = max(1, horizon // 24)
            feature_df[f'target_{horizon}h'] = feature_df.groupby(group_col)[target_col].shift(-horizon_days)
            
            # Prepare data
            train_df = feature_df.dropna(subset=[f'target_{horizon}h']).copy()
            
            if len(train_df) < 10:
                tqdm.write(f"  Warning: Not enough training samples ({len(train_df)}), skipping horizon")
                continue
            
            X = train_df[self.feature_cols]
            y = train_df[f'target_{horizon}h']
            
            # Split
            test_size = min(0.2, max(0.1, 5 / len(X)))  # Ensure reasonable test size
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
            
            # Train
            model = self._create_model()
            model.fit(X_train, y_train)
            self.models[horizon] = model
            
            # Evaluate
            y_pred = model.predict(X_val)
            metrics[horizon] = {
                'mae': mean_absolute_error(y_val, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_val, y_pred)),
                'r2': r2_score(y_val, y_pred)
            }
            tqdm.write(f"  MAE: {metrics[horizon]['mae']:.2f}, R²: {metrics[horizon]['r2']:.3f}")
        
        self.is_fitted = True
        return metrics
    
    def predict(
        self,
        df: pd.DataFrame,
        horizon: int = 24,
        weather_df: pd.DataFrame = None
    ) -> PredictionResult:
        """
        Make predictions for a specific horizon.
        
        Args:
            df: Input data
            horizon: Prediction horizon in hours
            weather_df: Optional weather data
            
        Returns:
            PredictionResult object
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if horizon not in self.models:
            raise ValueError(f"No model for {horizon}-hour horizon")
        
        # Build features
        feature_df, _ = build_forecasting_features(
            df, weather_df, 'portcalls', 'portname'
        )
        
        # Fill NaN with forward fill for prediction
        feature_df[self.feature_cols] = feature_df[self.feature_cols].ffill()
        
        X = feature_df[self.feature_cols]
        predictions = self.models[horizon].predict(X)
        
        return PredictionResult(
            predictions=predictions,
            prediction_horizon_hours=horizon,
            model_name=self.model_type,
            metrics={}
        )
    
    def get_feature_importance(self, horizon: int = 24) -> pd.DataFrame:
        """Get feature importance for a model."""
        if not self.is_fitted or horizon not in self.models:
            return pd.DataFrame()
        
        model = self.models[horizon]
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        else:
            return pd.DataFrame()
        
        return pd.DataFrame({
            'feature': self.feature_cols,
            'importance': importance
        }).sort_values('importance', ascending=False)
    
    def save(self, path: Path = None):
        """Save model to disk."""
        if path is None:
            path = MODELS_DIR / 'surge_model.joblib'
        
        joblib.dump({
            'models': self.models,
            'feature_cols': self.feature_cols,
            'model_type': self.model_type,
            'prediction_horizons': self.prediction_horizons
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path: Path = None):
        """Load model from disk."""
        if path is None:
            path = MODELS_DIR / 'surge_model.joblib'
        
        data = joblib.load(path)
        self.models = data['models']
        self.feature_cols = data['feature_cols']
        self.model_type = data['model_type']
        self.prediction_horizons = data['prediction_horizons']
        self.is_fitted = True
        print(f"Model loaded from {path}")


if __name__ == "__main__":
    # Test model
    dates = pd.date_range('2023-01-01', periods=365, freq='D')
    sample_df = pd.DataFrame({
        'date': dates,
        'portname': 'Test Port',
        'portcalls': np.random.randint(10, 50, size=365) + np.sin(np.arange(365) / 7) * 10
    })
    
    model = SurgePredictionModel()
    metrics = model.fit(sample_df)
    
    print("\nFeature importance:")
    print(model.get_feature_importance(24).head(10))

