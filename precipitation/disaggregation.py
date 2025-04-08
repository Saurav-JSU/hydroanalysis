"""
Functions for temporal disaggregation of precipitation data.
"""

import pandas as pd
import numpy as np
import logging
from datetime import timedelta
import os

from hydroanalysis.config import CONFIG

logger = logging.getLogger(__name__)

def get_time_resolution(dataset_df):
    """
    Determine the time resolution of a dataset in minutes.
    
    Parameters
    ----------
    dataset_df : pandas.DataFrame
        DataFrame with datetime column
        
    Returns
    -------
    int
        Time resolution in minutes
    """
    # Check if we have a datetime column
    datetime_col = None
    if 'datetime' in dataset_df.columns:
        datetime_col = 'datetime'
    elif 'Date' in dataset_df.columns and pd.api.types.is_datetime64_dtype(dataset_df['Date']):
        datetime_col = 'Date'
    
    if datetime_col is None or len(dataset_df) < 2:
        logger.warning("Cannot determine time resolution: no datetime column or insufficient data")
        return CONFIG['disaggregation']['default_resolution']
    
    # Sort by datetime
    sorted_df = dataset_df.sort_values(datetime_col)
    
    # Calculate differences between consecutive timestamps
    time_diffs = sorted_df[datetime_col].diff()[1:]  # Skip first which is NaT
    
    # Convert to minutes
    minutes_diffs = time_diffs.dt.total_seconds() / 60
    
    if len(minutes_diffs) > 0:
        # Get the most common difference
        # Use value_counts to find the most frequent time difference
        common_diffs = minutes_diffs.value_counts()
        if len(common_diffs) > 0:
            most_common_diff = common_diffs.index[0]
            
            # Round to nearest common resolution
            if 1 <= most_common_diff < 45:  # Less than 45 minutes
                resolution = round(most_common_diff)
                logger.info(f"Detected time resolution: {resolution} minutes")
                return resolution
            elif 45 <= most_common_diff < 90:  # Hourly
                logger.info("Detected time resolution: 60 minutes (hourly)")
                return 60
            elif 90 <= most_common_diff < 180:  # 2-hourly
                logger.info("Detected time resolution: 120 minutes (2-hourly)")
                return 120
            elif 180 <= most_common_diff < 360:  # 3-hourly
                logger.info("Detected time resolution: 180 minutes (3-hourly)")
                return 180
            else:  # 6-hourly or more
                logger.info("Detected time resolution: > 3 hours")
                return int(most_common_diff)
        
    # Default to hourly if we couldn't determine
    logger.warning("Could not determine time resolution. Defaulting to hourly.")
    return CONFIG['disaggregation']['default_resolution']

def disaggregate_to_half_hourly(hourly_df, pattern_df=None, station=None):
    """
    Disaggregate hourly precipitation to half-hourly using pattern-based or statistical methods.
    
    Parameters
    ----------
    hourly_df : pandas.DataFrame
        DataFrame with hourly precipitation data and 'datetime' column
    pattern_df : pandas.DataFrame, optional
        DataFrame with higher-resolution pattern data to guide disaggregation
    station : str, optional
        Station ID to process (if None, process all precipitation columns)
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with half-hourly precipitation data
    """
    # Ensure we have a datetime column
    if 'datetime' not in hourly_df.columns and 'Date' in hourly_df.columns:
        hourly_df = hourly_df.copy()
        hourly_df['datetime'] = hourly_df['Date']
    
    if 'datetime' not in hourly_df.columns:
        logger.error("Input DataFrame must have a 'datetime' or 'Date' column")
        return pd.DataFrame()
    
    # If no station provided, process all precipitation columns
    if station is None:
        station_columns = [col for col in hourly_df.columns 
                          if col not in ['datetime', 'Date', 'time', 'index']]
        
        if not station_columns:
            logger.error("No precipitation columns found in input DataFrame")
            return pd.DataFrame()
        
        # Process each station
        all_results = []
        for col in station_columns:
            result = disaggregate_to_half_hourly(hourly_df, pattern_df, col)
            all_results.append(result)
        
        # Merge results
        if all_results:
            merged_result = all_results[0]
            for i in range(1, len(all_results)):
                merged_result = pd.merge(merged_result, all_results[i], on='datetime', how='outer')
            return merged_result.sort_values('datetime')
        else:
            return pd.DataFrame()
    
    logger.info(f"Disaggregating hourly data to half-hourly for {station}")
    
    # Make sure dataframes are sorted by datetime
    hourly_df = hourly_df.sort_values('datetime')
    if pattern_df is not None:
        if 'datetime' not in pattern_df.columns and 'Date' in pattern_df.columns:
            pattern_df = pattern_df.copy()
            pattern_df['datetime'] = pattern_df['Date']
        
        if 'datetime' in pattern_df.columns:
            pattern_df = pattern_df.sort_values('datetime')
        else:
            logger.warning("Pattern DataFrame has no datetime column. Cannot use for pattern-based disaggregation.")
            pattern_df = None
    
    # Create a DataFrame to store the results
    half_hourly_data = []
    
    # Process each hour in the hourly dataset
    for _, hour_row in hourly_df.iterrows():
        hour_datetime = hour_row['datetime']
        hour_value = hour_row[station]
        
        # Skip processing if the hour value is zero or NaN (no precipitation)
        if hour_value == 0 or pd.isna(hour_value):
            # Still need to create two half-hour records with zero precipitation
            half_hourly_data.append({
                'datetime': hour_datetime,
                station: 0
            })
            half_hourly_data.append({
                'datetime': hour_datetime + pd.Timedelta(minutes=30),
                station: 0
            })
            continue
        
        # If we have a pattern dataset, try to use it
        if pattern_df is not None and station in pattern_df.columns:
            # Find the matching pattern for this hour
            pattern_hour_start = hour_datetime
            pattern_hour_end = hour_datetime + pd.Timedelta(hours=1)
            
            # Filter pattern data for this hour
            hour_pattern = pattern_df[
                (pattern_df['datetime'] >= pattern_hour_start) & 
                (pattern_df['datetime'] < pattern_hour_end)
            ]
            
            # Check if we have pattern data for this hour with at least two points
            if len(hour_pattern) >= 2:
                # Calculate the total precipitation in the pattern for this hour
                pattern_total = hour_pattern[station].sum()
                
                if pattern_total > 0:
                    # Calculate the distribution in the pattern
                    pattern_fractions = hour_pattern[station] / pattern_total
                    
                    # Apply the pattern fractions to the hourly value
                    disaggregated_values = pattern_fractions * hour_value
                    
                    # Create records for each time point in the pattern
                    for i, pattern_row in hour_pattern.iterrows():
                        half_hourly_data.append({
                            'datetime': pattern_row['datetime'],
                            station: disaggregated_values.iloc[i-hour_pattern.index[0]]
                        })
                    continue
        
        # If we don't have pattern data or it's not usable, use statistical distribution
        # For rainy periods, rainfall is typically more concentrated
        if hour_value > 1:  # More than 1mm is significant rain
            # For significant rainfall, use a more skewed distribution
            # 60-80% in one half-hour, 20-40% in the other
            intense_fraction = np.random.uniform(0.6, 0.8)
            less_intense_fraction = 1 - intense_fraction
            
            # Randomly choose which half-hour is more intense
            if np.random.rand() > 0.5:
                fractions = [intense_fraction, less_intense_fraction]
            else:
                fractions = [less_intense_fraction, intense_fraction]
        else:
            # For light rain, use a more balanced distribution
            # 40-60% in first half-hour, 60-40% in the second
            first_half_fraction = np.random.uniform(0.4, 0.6)
            fractions = [first_half_fraction, 1 - first_half_fraction]
        
        # Create two half-hour records
        half_hourly_data.append({
            'datetime': hour_datetime,
            station: hour_value * fractions[0]
        })
        half_hourly_data.append({
            'datetime': hour_datetime + pd.Timedelta(minutes=30),
            station: hour_value * fractions[1]
        })
    
    # Convert to DataFrame
    half_hourly_df = pd.DataFrame(half_hourly_data)
    
    # Sort by datetime
    half_hourly_df = half_hourly_df.sort_values('datetime')
    
    logger.info(f"Successfully disaggregated {len(hourly_df)} hourly records to {len(half_hourly_df)} half-hourly records for {station}")
    
    return half_hourly_df

def disaggregate_daily_to_hourly(daily_df, hourly_pattern=None, station=None):
    """
    Disaggregate daily precipitation to hourly.
    
    Parameters
    ----------
    daily_df : pandas.DataFrame
        DataFrame with daily precipitation data and 'Date' column
    hourly_pattern : pandas.DataFrame, optional
        DataFrame with hourly pattern data to guide disaggregation
    station : str, optional
        Station ID to process (if None, process all precipitation columns)
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with hourly precipitation data
    """
    # Check input DataFrame
    if 'Date' not in daily_df.columns:
        logger.error("Input DataFrame must have a 'Date' column")
        return pd.DataFrame()
    
    # Ensure Date is datetime
    daily_df = daily_df.copy()
    daily_df['Date'] = pd.to_datetime(daily_df['Date'])
    
    # If no station provided, process all precipitation columns
    if station is None:
        station_columns = [col for col in daily_df.columns 
                          if col not in ['Date', 'datetime', 'time', 'index']]
        
        if not station_columns:
            logger.error("No precipitation columns found in input DataFrame")
            return pd.DataFrame()
        
        # Process each station
        all_results = []
        for col in station_columns:
            result = disaggregate_daily_to_hourly(daily_df, hourly_pattern, col)
            all_results.append(result)
        
        # Merge results
        if all_results:
            merged_result = all_results[0]
            for i in range(1, len(all_results)):
                merged_result = pd.merge(merged_result, all_results[i], on='datetime', how='outer')
            return merged_result.sort_values('datetime')
        else:
            return pd.DataFrame()
    
    logger.info(f"Disaggregating daily data to hourly for {station}")
    
    # Results list
    hourly_data = []
    
    # Process each day
    for _, day_row in daily_df.iterrows():
        day_date = day_row['Date']
        day_value = day_row[station]
        
        # Skip if value is zero or NaN
        if day_value == 0 or pd.isna(day_value):
            # Create 24 hourly records with zero
            for hour in range(24):
                hourly_data.append({
                    'datetime': day_date + pd.Timedelta(hours=hour),
                    station: 0
                })
            continue
        
        # Check if we have hourly pattern data
        if hourly_pattern is not None and station in hourly_pattern.columns:
            # Ensure datetime column in pattern
            if 'datetime' not in hourly_pattern.columns and 'Date' in hourly_pattern.columns:
                hourly_pattern = hourly_pattern.copy()
                hourly_pattern['datetime'] = hourly_pattern['Date']
            
            if 'datetime' in hourly_pattern.columns:
                # Get pattern for this day
                day_start = pd.Timestamp(day_date.year, day_date.month, day_date.day, 0, 0, 0)
                day_end = day_start + pd.Timedelta(days=1)
                
                day_pattern = hourly_pattern[
                    (hourly_pattern['datetime'] >= day_start) & 
                    (hourly_pattern['datetime'] < day_end)
                ]
                
                # Check if we have enough pattern data for this day
                if len(day_pattern) >= 12:  # At least 12 hours of pattern data
                    # Calculate daily sum in pattern
                    pattern_sum = day_pattern[station].sum()
                    
                    if pattern_sum > 0:
                        # Calculate hourly fractions
                        pattern_fractions = day_pattern[station] / pattern_sum
                        
                        # Apply fractions to daily value
                        for _, pattern_row in day_pattern.iterrows():
                            hour_datetime = pattern_row['datetime']
                            hour_value = day_value * pattern_row[station] / pattern_sum
                            
                            hourly_data.append({
                                'datetime': hour_datetime,
                                station: hour_value
                            })
                        
                        # Fill any missing hours with zero
                        hours_in_pattern = set(day_pattern['datetime'].dt.hour)
                        for hour in range(24):
                            hour_datetime = day_start + pd.Timedelta(hours=hour)
                            if hour not in hours_in_pattern:
                                hourly_data.append({
                                    'datetime': hour_datetime,
                                    station: 0
                                })
                        
                        continue
        
        # If no pattern data or insufficient pattern, use statistical distribution
        
        # Check typical rainfall timing (where most rainfall occurs)
        # Morning (6-12), Afternoon (12-18), Evening (18-24), Night (0-6)
        period_weights = [0.15, 0.40, 0.35, 0.10]  # Default weights
        
        # Randomly select which hours will have rain (typically 4-8 hours in a day)
        rainy_hours = np.random.randint(4, 9)
        
        # Allocate total rainfall to hours based on period weights
        all_hours = np.arange(24)
        hour_weights = np.zeros(24)
        
        # Assign weights to each hour based on period
        for i in range(24):
            if i < 6:  # Night
                hour_weights[i] = period_weights[3] / 6
            elif i < 12:  # Morning
                hour_weights[i] = period_weights[0] / 6
            elif i < 18:  # Afternoon
                hour_weights[i] = period_weights[1] / 6
            else:  # Evening
                hour_weights[i] = period_weights[2] / 6
        
        # Add some randomness to weights
        hour_weights = hour_weights * (0.5 + np.random.rand(24))
        
        # Normalize weights
        hour_weights = hour_weights / np.sum(hour_weights)
        
        # Select which hours will have rainfall
        selected_hours = np.random.choice(all_hours, size=rainy_hours, replace=False, p=hour_weights)
        
        # Distribute rainfall to selected hours
        hour_values = np.zeros(24)
        
        # For each selected hour, assign a fraction of daily rainfall
        remaining_value = day_value
        for i, hour in enumerate(selected_hours):
            if i == len(selected_hours) - 1:
                # Last hour gets remaining rainfall
                hour_values[hour] = remaining_value
            else:
                # Assign a random fraction of remaining rainfall
                fraction = np.random.uniform(0.1, 0.4)
                hour_values[hour] = remaining_value * fraction
                remaining_value -= hour_values[hour]
        
        # Create hourly records
        for hour in range(24):
            hourly_data.append({
                'datetime': day_date + pd.Timedelta(hours=hour),
                station: hour_values[hour]
            })
    
    # Convert to DataFrame
    hourly_df = pd.DataFrame(hourly_data)
    
    # Sort by datetime
    hourly_df = hourly_df.sort_values('datetime')
    
    logger.info(f"Successfully disaggregated {len(daily_df)} daily records to {len(hourly_df)} hourly records for {station}")
    
    return hourly_df

def create_high_resolution_precipitation(best_datasets, scaling_factors, datasets_dict, output_dir=None):
    """
    Create high-resolution precipitation data by combining best datasets and applying scaling factors.
    
    Parameters
    ----------
    best_datasets : dict
        Dictionary with best dataset info for each station
    scaling_factors : dict
        Dictionary with scaling factors for each station
    datasets_dict : dict
        Dictionary with dataset name as key and DataFrame as value
    output_dir : str, optional
        Directory to save outputs
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with high-resolution precipitation data
    """
    # Find a high-resolution dataset to use as a pattern
    pattern_dataset = None
    pattern_resolution = float('inf')
    
    for name, df in datasets_dict.items():
        if 'datetime' in df.columns or 'Date' in df.columns:
            # Calculate time resolution
            resolution = get_time_resolution(df)
            
            # If this is a higher resolution dataset, use it as pattern
            if resolution < pattern_resolution:
                pattern_dataset = df
                pattern_resolution = resolution
                logger.info(f"Using {name} (resolution: {resolution} minutes) as pattern dataset")
    
    # Create an empty DataFrame for the high-resolution data
    high_res_data = None
    target_resolution = CONFIG['disaggregation']['target_resolution']
    
    # Process each station
    for station, info in best_datasets.items():
        dataset_name = info["dataset"]
        
        if dataset_name not in datasets_dict:
            logger.warning(f"Dataset {dataset_name} not found. Skipping station {station}.")
            continue
        
        # Get the original data
        original_data = datasets_dict[dataset_name]
        
        if station not in original_data.columns:
            logger.warning(f"Station {station} not found in dataset {dataset_name}. Skipping.")
            continue
        
        # Apply scaling factor if available
        scaling_factor = 1.0
        if station in scaling_factors:
            scaling_factor = scaling_factors[station]["factor"]
            logger.info(f"Using scaling factor {scaling_factor:.3f} for {station}")
        
        # Determine the time resolution of the dataset
        resolution = get_time_resolution(original_data)
        
        # Process based on resolution
        station_high_res = None
        
        if resolution <= target_resolution:
            # Already high-resolution, just apply scaling factor
            if 'datetime' in original_data.columns:
                station_high_res = original_data[['datetime', station]].copy()
            elif 'Date' in original_data.columns:
                station_high_res = original_data[['Date', station]].copy()
                station_high_res = station_high_res.rename(columns={'Date': 'datetime'})
            else:
                logger.warning(f"No datetime column found in dataset {dataset_name}. Skipping station {station}.")
                continue
            
            station_high_res[station] = station_high_res[station] * scaling_factor
            logger.info(f"Applied scaling factor to {station} (already high-resolution)")
            
        elif resolution == 60:  # Hourly to half-hourly
            # Copy the station data
            if 'datetime' in original_data.columns:
                hourly_data = original_data[['datetime', station]].copy()
            elif 'Date' in original_data.columns:
                hourly_data = original_data[['Date', station]].copy()
                hourly_data = hourly_data.rename(columns={'Date': 'datetime'})
            else:
                logger.warning(f"No datetime column found in dataset {dataset_name}. Skipping station {station}.")
                continue
            
            # Apply scaling factor
            hourly_data[station] = hourly_data[station] * scaling_factor
            
            # Disaggregate to half-hourly using pattern dataset
            station_high_res = disaggregate_to_half_hourly(hourly_data, pattern_dataset, station)
            logger.info(f"Disaggregated hourly data to half-hourly for {station}")
            
        elif resolution >= 1440:  # Daily to half-hourly (via hourly)
            # Copy the station data
            if 'Date' in original_data.columns:
                daily_data = original_data[['Date', station]].copy()
            else:
                logger.warning(f"No Date column found in dataset {dataset_name}. Skipping station {station}.")
                continue
            
            # Apply scaling factor
            daily_data[station] = daily_data[station] * scaling_factor
            
            # First disaggregate to hourly
            hourly_data = disaggregate_daily_to_hourly(daily_data, pattern_dataset, station)
            
            # Then disaggregate hourly to half-hourly
            station_high_res = disaggregate_to_half_hourly(hourly_data, pattern_dataset, station)
            logger.info(f"Disaggregated daily data to half-hourly for {station}")
            
        else:  # Other resolutions
            logger.warning(f"Unsupported resolution ({resolution} minutes) for {station}. Skipping.")
            continue
        
        # Add to the high-resolution dataset
        if high_res_data is None:
            high_res_data = station_high_res
        else:
            # Merge with existing data on datetime
            high_res_data = pd.merge(high_res_data, station_high_res, on='datetime', how='outer')
    
    # Sort by datetime
    if high_res_data is not None:
        high_res_data = high_res_data.sort_values('datetime')
    
    # Save the high-resolution dataset if output directory is provided
    if output_dir and high_res_data is not None:
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, "high_resolution_precipitation.csv")
        high_res_data.to_csv(file_path, index=False)
        logger.info(f"Saved high-resolution precipitation data to {file_path}")
    
    return high_res_data

def create_distributed_precipitation(precipitation_df, dem_file=None, idw_power=2, max_distance=50):
    """
    Create spatially distributed precipitation data using inverse distance weighting.
    
    Parameters
    ----------
    precipitation_df : pandas.DataFrame
        DataFrame with precipitation data and 'datetime' column
    dem_file : str, optional
        Path to DEM file for elevation adjustment
    idw_power : float, optional
        Power parameter for inverse distance weighting
    max_distance : float, optional
        Maximum distance (km) to consider for interpolation
        
    Returns
    -------
    numpy.ndarray
        Array with spatially distributed precipitation
    """
    # This is a placeholder for future implementation
    # Spatial interpolation requires additional dependencies like GDAL/rasterio
    logger.warning("Spatially distributed precipitation is not yet implemented")
    return None

def validate_disaggregation(original_df, disaggregated_df, station=None):
    """
    Validate that disaggregation preserves daily/hourly totals.
    
    Parameters
    ----------
    original_df : pandas.DataFrame
        DataFrame with original precipitation data
    disaggregated_df : pandas.DataFrame
        DataFrame with disaggregated precipitation data
    station : str, optional
        Station ID to validate (if None, validate all common stations)
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with validation results
    """
    # Ensure we have datetime columns
    if 'datetime' not in disaggregated_df.columns:
        if 'Date' in disaggregated_df.columns:
            disaggregated_df = disaggregated_df.copy()
            disaggregated_df['datetime'] = disaggregated_df['Date']
        else:
            logger.error("Disaggregated DataFrame must have a 'datetime' or 'Date' column")
            return pd.DataFrame()
    
    datetime_col_orig = None
    if 'datetime' in original_df.columns:
        datetime_col_orig = 'datetime'
    elif 'Date' in original_df.columns:
        datetime_col_orig = 'Date'
    else:
        logger.error("Original DataFrame must have a 'datetime' or 'Date' column")
        return pd.DataFrame()
    
    # Get time resolution of both datasets
    orig_resolution = get_time_resolution(original_df)
    disagg_resolution = get_time_resolution(disaggregated_df)
    
    if disagg_resolution >= orig_resolution:
        logger.error(f"Disaggregated data resolution ({disagg_resolution} min) is not finer than original ({orig_resolution} min)")
        return pd.DataFrame()
    
    # Determine stations to validate
    if station:
        stations = [station]
    else:
        # Find common stations
        stations = [col for col in original_df.columns if col in disaggregated_df.columns 
                   and col not in ['datetime', 'Date', 'time', 'index']]
    
    if not stations:
        logger.error("No common stations found between original and disaggregated data")
        return pd.DataFrame()
    
    # Resample disaggregated data to original resolution
    if orig_resolution == 60:  # Hourly
        # Group by hour
        disaggregated_df['hour'] = disaggregated_df['datetime'].dt.floor('H')
        aggregated = disaggregated_df.groupby('hour').sum()
        
        # Reset index to get hour as column
        aggregated = aggregated.reset_index().rename(columns={'hour': 'datetime'})
        
    elif orig_resolution == 1440:  # Daily
        # Group by day
        disaggregated_df['day'] = disaggregated_df['datetime'].dt.floor('D')
        aggregated = disaggregated_df.groupby('day').sum()
        
        # Reset index to get day as column
        aggregated = aggregated.reset_index().rename(columns={'day': 'datetime'})
        
    else:  # Other resolutions
        # Group by custom resolution
        def round_to_resolution(dt, resolution_minutes):
            minutes = resolution_minutes * (dt.minute // resolution_minutes)
            return dt.replace(minute=minutes, second=0, microsecond=0)
        
        disaggregated_df['period'] = disaggregated_df['datetime'].apply(
            lambda dt: round_to_resolution(dt, orig_resolution))
        
        aggregated = disaggregated_df.groupby('period').sum()
        
        # Reset index to get period as column
        aggregated = aggregated.reset_index().rename(columns={'period': 'datetime'})
    
    # Compare aggregated with original
    validation_results = []
    
    for station in stations:
        # Merge original and aggregated data
        comparison = pd.merge(
            original_df[[datetime_col_orig, station]].rename(columns={datetime_col_orig: 'datetime'}),
            aggregated[['datetime', station]],
            on='datetime', 
            how='inner',
            suffixes=('_orig', '_aggr')
        )
        
        if comparison.empty:
            logger.warning(f"No matching time periods for station {station}")
            continue
        
        # Calculate differences
        orig_col = f"{station}_orig"
        aggr_col = f"{station}_aggr"
        
        comparison['diff'] = comparison[aggr_col] - comparison[orig_col]
        comparison['abs_diff'] = abs(comparison['diff'])
        comparison['rel_diff'] = comparison['diff'] / comparison[orig_col].replace(0, np.nan)
        
        # Calculate statistics
        mean_abs_diff = comparison['abs_diff'].mean()
        max_abs_diff = comparison['abs_diff'].max()
        mean_rel_diff = comparison['rel_diff'].mean()
        
        validation_results.append({
            'Station': station,
            'Original_Resolution': f"{orig_resolution} min",
            'Disaggregated_Resolution': f"{disagg_resolution} min",
            'Mean_Absolute_Difference': mean_abs_diff,
            'Max_Absolute_Difference': max_abs_diff,
            'Mean_Relative_Difference': mean_rel_diff,
            'Match_Rate': sum(comparison['abs_diff'] < 0.01) / len(comparison)
        })
    
    return pd.DataFrame(validation_results)