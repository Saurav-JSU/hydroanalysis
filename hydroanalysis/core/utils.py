"""
Utility functions for the HydroAnalysis package.
"""

import logging
import os
from math import radians, cos, sin, asin, sqrt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from hydroanalysis.config import CONFIG

def setup_logging(log_file=None, level=None):
    """
    Set up logging configuration.
    
    Parameters
    ----------
    log_file : str, optional
        Path to log file. If None, uses the value from CONFIG.
    level : str, optional
        Logging level. If None, uses the value from CONFIG.
        
    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    # Use config values if not specified
    if log_file is None:
        log_file = CONFIG['logging']['file']
    
    if level is None:
        level_str = CONFIG['logging']['level']
        level = getattr(logging, level_str)
    
    # Create log directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=level,
        format=CONFIG['logging']['format'],
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # Create logger instance
    logger = logging.getLogger('hydroanalysis')
    logger.info(f"Logging initialized. Level: {logging.getLevelName(level)}, File: {log_file}")
    
    return logger

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points on the earth.
    
    Parameters
    ----------
    lat1 : float
        Latitude of first point in decimal degrees
    lon1 : float
        Longitude of first point in decimal degrees
    lat2 : float
        Latitude of second point in decimal degrees
    lon2 : float
        Longitude of second point in decimal degrees
        
    Returns
    -------
    float
        Distance between points in kilometers
    """
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers
    
    return c * r

def date_range(start_date, end_date, freq='D'):
    """
    Create a date range between two dates.
    
    Parameters
    ----------
    start_date : datetime.datetime or str
        Start date
    end_date : datetime.datetime or str
        End date
    freq : str, optional
        Frequency string (default: 'D' for daily)
        
    Returns
    -------
    pandas.DatetimeIndex
        Date range
    """
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
        
    return pd.date_range(start=start_date, end=end_date, freq=freq)

def find_closest_stations(station_metadata, target_lat, target_lon, max_distance=None, max_stations=None):
    """
    Find the closest stations to a target location.
    
    Parameters
    ----------
    station_metadata : pandas.DataFrame
        DataFrame with station metadata including Latitude and Longitude columns
    target_lat : float
        Latitude of target location
    target_lon : float
        Longitude of target location
    max_distance : float, optional
        Maximum distance in kilometers (if None, no limit)
    max_stations : int, optional
        Maximum number of stations to return (if None, no limit)
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with closest stations sorted by distance
    """
    if 'Latitude' not in station_metadata.columns or 'Longitude' not in station_metadata.columns:
        raise ValueError("station_metadata must contain 'Latitude' and 'Longitude' columns")
    
    # Calculate distance for each station
    distances = []
    for _, row in station_metadata.iterrows():
        if pd.notna(row['Latitude']) and pd.notna(row['Longitude']):
            distance = haversine_distance(target_lat, target_lon, row['Latitude'], row['Longitude'])
            
            # Create a copy of the row to avoid SettingWithCopyWarning
            row_copy = row.copy()
            row_copy['Distance'] = distance
            
            distances.append(row_copy)
    
    # Create DataFrame and sort by distance
    if distances:
        distance_df = pd.DataFrame(distances)
        distance_df = distance_df.sort_values('Distance')
        
        # Apply filters
        if max_distance is not None:
            distance_df = distance_df[distance_df['Distance'] <= max_distance]
        
        if max_stations is not None:
            distance_df = distance_df.head(max_stations)
        
        return distance_df
    else:
        return pd.DataFrame()

def to_datetime_index(df, date_col='Date'):
    """
    Convert a DataFrame to use a datetime index.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to convert
    date_col : str, optional
        Name of the date column
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with datetime index
    """
    if date_col not in df.columns:
        raise ValueError(f"DataFrame does not contain column '{date_col}'")
    
    # Make a copy to avoid modifying the original
    result = df.copy()
    
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(result[date_col]):
        result[date_col] = pd.to_datetime(result[date_col], errors='coerce')
    
    # Set index and drop date column
    result = result.set_index(date_col)
    
    return result

def resample_timeseries(df, freq='D', agg_func='mean', date_col=None):
    """
    Resample a time series DataFrame to a different frequency.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to resample
    freq : str, optional
        Target frequency (default: 'D' for daily)
    agg_func : str or dict, optional
        Aggregation function or dictionary of functions
    date_col : str, optional
        Name of the date column (if None, assumes df is already indexed by date)
        
    Returns
    -------
    pandas.DataFrame
        Resampled DataFrame
    """
    # If date_col is specified, convert to datetime index
    if date_col is not None:
        df = to_datetime_index(df, date_col)
    
    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be DatetimeIndex or date_col must be specified")
    
    # Resample the DataFrame
    if isinstance(agg_func, dict):
        # Different aggregation for different columns
        resampled = df.resample(freq).agg(agg_func)
    else:
        # Same aggregation for all columns
        resampled = df.resample(freq).agg(agg_func)
    
    return resampled

def moving_average(series, window=3, center=True):
    """
    Calculate moving average for a series.
    
    Parameters
    ----------
    series : pandas.Series
        Series to calculate moving average
    window : int, optional
        Window size (default: 3)
    center : bool, optional
        Whether to center the window (default: True)
        
    Returns
    -------
    pandas.Series
        Series with moving average
    """
    return series.rolling(window=window, center=center).mean()

def detect_outliers(series, method='iqr', threshold=1.5):
    """
    Detect outliers in a series.
    
    Parameters
    ----------
    series : pandas.Series
        Series to detect outliers
    method : str, optional
        Method to use: 'iqr' (Interquartile Range) or 'zscore'
    threshold : float, optional
        Threshold for outlier detection
        
    Returns
    -------
    pandas.Series
        Boolean series with True for outliers
    """
    if method.lower() == 'iqr':
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        return (series < lower_bound) | (series > upper_bound)
    
    elif method.lower() == 'zscore':
        z_scores = (series - series.mean()) / series.std()
        return z_scores.abs() > threshold
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'iqr' or 'zscore'.")

def convert_units(values, from_unit, to_unit):
    """
    Convert values between different units.
    
    Parameters
    ----------
    values : float or array-like
        Values to convert
    from_unit : str
        Source unit
    to_unit : str
        Target unit
        
    Returns
    -------
    float or array-like
        Converted values
    """
    # Define conversion factors for different units
    conversion_factors = {
        # Precipitation
        'mm_to_m': 0.001,
        'm_to_mm': 1000,
        'inch_to_mm': 25.4,
        'mm_to_inch': 1/25.4,
        
        # Discharge
        'm3s_to_ft3s': 35.3147,  # Cubic meters per second to cubic feet per second
        'ft3s_to_m3s': 1/35.3147,
        
        # Temperature
        'c_to_f': lambda x: x * 9/5 + 32,
        'f_to_c': lambda x: (x - 32) * 5/9,
        
        # Time
        'day_to_hour': 24,
        'hour_to_day': 1/24,
    }
    
    # Create the conversion key
    conversion_key = f"{from_unit.lower()}_to_{to_unit.lower()}"
    
    if conversion_key in conversion_factors:
        factor = conversion_factors[conversion_key]
        
        # If the factor is a function, apply it
        if callable(factor):
            return factor(values)
        else:
            return values * factor
    else:
        raise ValueError(f"Unsupported unit conversion: {from_unit} to {to_unit}")

def calculate_nash_sutcliffe(observed, simulated):
    """
    Calculate Nash-Sutcliffe Efficiency (NSE) between observed and simulated values.
    
    Parameters
    ----------
    observed : array-like
        Observed values
    simulated : array-like
        Simulated values
        
    Returns
    -------
    float
        Nash-Sutcliffe Efficiency value (-∞ to 1)
    """
    # Remove NaN values
    mask = ~np.isnan(observed) & ~np.isnan(simulated)
    observed = np.array(observed)[mask]
    simulated = np.array(simulated)[mask]
    
    if len(observed) == 0:
        return np.nan
    
    # Calculate NSE
    mean_observed = np.mean(observed)
    numerator = np.sum((observed - simulated) ** 2)
    denominator = np.sum((observed - mean_observed) ** 2)
    
    if denominator == 0:
        return np.nan
    
    return 1 - (numerator / denominator)

def calculate_percentile_threshold(values, percentile=95):
    """
    Calculate a threshold based on a percentile of values.
    
    Parameters
    ----------
    values : array-like
        Values to calculate threshold
    percentile : float, optional
        Percentile value (0-100)
        
    Returns
    -------
    float
        Threshold value
    """
    return np.nanpercentile(values, percentile)

def get_season(date):
    """
    Get the season for a given date.
    
    Parameters
    ----------
    date : datetime.datetime
        Date to determine season
        
    Returns
    -------
    str
        Season name: 'winter', 'spring', 'summer', or 'fall'
    """
    # Extract month and day for comparison
    month = date.month
    day = date.day
    
    # Define seasons (Northern Hemisphere)
    if (month == 12 and day >= 21) or month <= 2 or (month == 3 and day < 20):
        return 'winter'
    elif (month == 3 and day >= 20) or month <= 5 or (month == 6 and day < 21):
        return 'spring'
    elif (month == 6 and day >= 21) or month <= 8 or (month == 9 and day < 22):
        return 'summer'
    else:
        return 'fall'

def calculate_wet_days(precipitation, threshold=1.0):
    """
    Calculate number of wet days in a precipitation series.
    
    Parameters
    ----------
    precipitation : array-like
        Daily precipitation values
    threshold : float, optional
        Threshold for wet day (mm)
        
    Returns
    -------
    int
        Number of wet days
    """
    return np.sum(np.array(precipitation) >= threshold)

def calculate_consecutive_dry_days(precipitation, threshold=1.0):
    """
    Calculate maximum consecutive dry days in a precipitation series.
    
    Parameters
    ----------
    precipitation : array-like
        Daily precipitation values
    threshold : float, optional
        Threshold for wet day (mm)
        
    Returns
    -------
    int
        Maximum consecutive dry days
    """
    # Convert to dry day binary indicator (1 for dry, 0 for wet)
    dry_days = np.array(precipitation) < threshold
    
    # Find runs of consecutive dry days
    runs = []
    current_run = 0
    
    for is_dry in dry_days:
        if is_dry:
            current_run += 1
        else:
            if current_run > 0:
                runs.append(current_run)
            current_run = 0
    
    # Don't forget the last run
    if current_run > 0:
        runs.append(current_run)
    
    # Return the maximum
    return max(runs) if runs else 0