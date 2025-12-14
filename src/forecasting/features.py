"""
Feature Engineering for Forecasting Models
===========================================
Build features for surge prediction and dwell time estimation.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import FORECASTING


def add_time_features(df: pd.DataFrame, date_col: str = 'date') -> pd.DataFrame:
    """
    Add time-based features to DataFrame.
    
    Args:
        df: DataFrame with date column
        date_col: Name of date column
        
    Returns:
        DataFrame with time features added
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    
    df['day_of_week'] = df[date_col].dt.dayofweek
    df['day_of_month'] = df[date_col].dt.day
    df['week_of_year'] = df[date_col].dt.isocalendar().week
    df['month'] = df[date_col].dt.month
    df['quarter'] = df[date_col].dt.quarter
    df['year'] = df[date_col].dt.year
    
    # Cyclical encoding
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Binary flags
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_month_start'] = (df['day_of_month'] <= 3).astype(int)
    df['is_month_end'] = (df['day_of_month'] >= 28).astype(int)
    
    return df


def add_lag_features(
    df: pd.DataFrame,
    target_col: str,
    group_col: str = None,
    lags: List[int] = None
) -> pd.DataFrame:
    """
    Add lagged features for a target variable.
    
    Args:
        df: DataFrame with target column
        target_col: Column to create lags for
        group_col: Optional column to group by (e.g., port_id)
        lags: List of lag periods (default from config)
        
    Returns:
        DataFrame with lag features added
    """
    if lags is None:
        lags = FORECASTING.lag_features
    
    df = df.copy()
    
    for lag in lags:
        col_name = f'{target_col}_lag_{lag}'
        if group_col:
            df[col_name] = df.groupby(group_col)[target_col].shift(lag)
        else:
            df[col_name] = df[target_col].shift(lag)
    
    return df


def add_rolling_features(
    df: pd.DataFrame,
    target_col: str,
    group_col: str = None,
    windows: List[int] = None,
    aggs: List[str] = None
) -> pd.DataFrame:
    """
    Add rolling window statistics.
    
    Args:
        df: DataFrame with target column
        target_col: Column to compute rolling stats for
        group_col: Optional column to group by
        windows: List of window sizes (default from config)
        aggs: List of aggregations (default: ['mean', 'std', 'min', 'max'])
        
    Returns:
        DataFrame with rolling features added
    """
    if windows is None:
        windows = FORECASTING.rolling_windows
    if aggs is None:
        aggs = ['mean', 'std', 'min', 'max']
    
    df = df.copy()
    
    for window in windows:
        for agg in aggs:
            col_name = f'{target_col}_roll_{window}d_{agg}'
            if group_col:
                grouped = df.groupby(group_col)[target_col]
                df[col_name] = grouped.transform(
                    lambda x: x.rolling(window, min_periods=1).agg(agg)
                )
            else:
                df[col_name] = df[target_col].rolling(window, min_periods=1).agg(agg)
    
    return df


def add_weather_features(
    df: pd.DataFrame,
    weather_df: pd.DataFrame,
    merge_on: List[str] = None
) -> pd.DataFrame:
    """
    Merge weather features into main DataFrame.
    
    Args:
        df: Main DataFrame
        weather_df: Weather DataFrame
        merge_on: Columns to merge on (auto-detected if None)
        
    Returns:
        DataFrame with weather features added
    """
    # Select relevant weather columns
    weather_cols = [
        'weather_code', 'temperature_2m_max', 'temperature_2m_min',
        'precipitation_sum', 'wind_speed_10m_max', 'wind_gusts_10m_max'
    ]
    
    # Normalize date columns to naive datetime (strip timezone info)
    # This prevents merge errors between datetime64[ns, UTC] and datetime64[ns]
    if 'date' in df.columns:
        if hasattr(df['date'].dtype, 'tz') and df['date'].dtype.tz is not None:
            df = df.copy()
            df['date'] = df['date'].dt.tz_localize(None)
    if 'date' in weather_df.columns:
        if hasattr(weather_df['date'].dtype, 'tz') and weather_df['date'].dtype.tz is not None:
            weather_df = weather_df.copy()
            weather_df['date'] = weather_df['date'].dt.tz_localize(None)
    
    # Auto-detect merge columns
    if merge_on is None:
        merge_on = []
        if 'date' in df.columns and 'date' in weather_df.columns:
            merge_on.append('date')
        if 'location_name' in df.columns and 'location_name' in weather_df.columns:
            merge_on.append('location_name')
    
    # If no valid merge columns, use average weather by date
    if not merge_on or 'location_name' not in df.columns:
        # Aggregate weather across all locations by date
        if 'date' in df.columns and 'date' in weather_df.columns:
            existing_cols = [c for c in weather_cols if c in weather_df.columns]
            weather_agg = weather_df.groupby('date')[existing_cols].mean().reset_index()
            df = df.merge(weather_agg, on='date', how='left')
        else:
            # Can't merge, just return original
            return df
    else:
        # Standard merge
        existing_cols = [c for c in weather_cols if c in weather_df.columns]
        cols_to_merge = merge_on + existing_cols
        cols_to_merge = [c for c in cols_to_merge if c in weather_df.columns]
        
        weather_subset = weather_df[cols_to_merge].copy()
        df = df.merge(weather_subset, on=merge_on, how='left')
    
    # Add derived features
    if 'precipitation_sum' in df.columns:
        df['is_rainy'] = (df['precipitation_sum'] > 0.1).astype(int)
        df['is_heavy_rain'] = (df['precipitation_sum'] > 10).astype(int)
    
    if 'wind_speed_10m_max' in df.columns:
        df['is_high_wind'] = (df['wind_speed_10m_max'] > 30).astype(int)
    
    return df


def build_forecasting_features(
    port_activity_df: pd.DataFrame,
    weather_df: pd.DataFrame = None,
    target_col: str = 'portcalls',
    group_col: str = 'portname'
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build complete feature set for forecasting.
    
    Args:
        port_activity_df: Port activity data
        weather_df: Optional weather data
        target_col: Target variable column
        group_col: Column to group by
        
    Returns:
        Tuple of (feature DataFrame, list of feature column names)
    """
    df = port_activity_df.copy()
    
    # Ensure date is datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    # Add time features
    df = add_time_features(df, 'date')
    
    # Add lag features
    df = add_lag_features(df, target_col, group_col)
    
    # Add rolling features
    df = add_rolling_features(df, target_col, group_col)
    
    # Add weather if available
    if weather_df is not None:
        weather_copy = weather_df.copy()
        if 'date' in weather_copy.columns:
            weather_copy['date'] = pd.to_datetime(weather_copy['date'])
        df = add_weather_features(df, weather_copy)
    
    # Fill any NaN values from weather merge with 0
    df = df.fillna(0)
    
    # Get feature column names (exclude date, target, and ID columns)
    exclude_cols = ['date', target_col, group_col, 'portid', 'ObjectId', 
                    'country', 'ISO3', 'year', 'month', 'day']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    return df, feature_cols


if __name__ == "__main__":
    # Test feature engineering
    import pandas as pd
    
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    sample_df = pd.DataFrame({
        'date': dates,
        'portname': 'Test Port',
        'portcalls': np.random.randint(10, 50, size=100)
    })
    
    df, features = build_forecasting_features(sample_df)
    print(f"Created {len(features)} features:")
    print(features[:10])

