"""
Functions for temporal disaggregation of precipitation data.
"""

import pandas as pd
import numpy as np
import logging
import os
from datetime import timedelta

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

def create_high_resolution_precipitation(flood_dir, output_dir=None, imerg_file=None):
    """
    Create high-resolution precipitation data from corrected datasets.
    
    Parameters
    ----------
    flood_dir : str
        Directory containing flood event data
    output_dir : str, optional
        Directory to save outputs (defaults to flood_dir/highres)
    imerg_file : str, optional
        Path to IMERG hourly precipitation file to use as pattern
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with high-resolution precipitation data
    """
    logger.info(f"Processing directory: {flood_dir}")
    
    # Set default output directory
    if output_dir is None:
        output_dir = os.path.join(flood_dir, "highres")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get corrected directory
    corrected_dir = os.path.join(flood_dir, "corrected")
    if not os.path.isdir(corrected_dir):
        logger.error(f"Corrected directory not found: {corrected_dir}")
        return None
    
    # Get required files
    corrected_highres_file = os.path.join(corrected_dir, "corrected_highres_precipitation.csv")
    if not os.path.exists(corrected_highres_file):
        logger.error(f"Corrected high-resolution file not found: {corrected_highres_file}")
        return None
    
    # Find IMERG file if not provided
    if imerg_file is None:
        # Try to find IMERG file in standard locations
        potential_paths = [
            os.path.join(flood_dir, "imerg_hourly_precipitation.csv"),
            os.path.join(flood_dir, "gee_imerg", "hourly_precipitation.csv")
        ]
        
        for path in potential_paths:
            if os.path.exists(path):
                imerg_file = path
                logger.info(f"Found IMERG pattern file: {imerg_file}")
                break
    
    if imerg_file is None:
        logger.warning("No IMERG pattern file found. Will use statistical disaggregation.")
    
    # Read corrected data
    logger.info(f"Reading corrected high-resolution data: {corrected_highres_file}")
    corrected_df = pd.read_csv(corrected_highres_file)
    
    # Ensure datetime column
    if 'datetime' not in corrected_df.columns:
        if 'Date' in corrected_df.columns:
            corrected_df['datetime'] = pd.to_datetime(corrected_df['Date'])
        else:
            logger.error("No datetime or Date column found")
            return None
    else:
        corrected_df['datetime'] = pd.to_datetime(corrected_df['datetime'])
    
    # Read IMERG data if available
    imerg_df = None
    if imerg_file:
        try:
            logger.info(f"Reading IMERG pattern file: {imerg_file}")
            imerg_df = pd.read_csv(imerg_file)
            
            # Ensure datetime column
            if 'datetime' not in imerg_df.columns:
                if 'Date' in imerg_df.columns:
                    imerg_df['datetime'] = pd.to_datetime(imerg_df['Date'])
                elif 'Original_Datetime' in imerg_df.columns:
                    imerg_df['datetime'] = pd.to_datetime(imerg_df['Original_Datetime'])
                else:
                    logger.warning("IMERG file has no datetime column. Cannot use for pattern.")
                    imerg_df = None
            else:
                imerg_df['datetime'] = pd.to_datetime(imerg_df['datetime'])
        except Exception as e:
            logger.error(f"Error reading IMERG file: {e}")
            imerg_df = None
    
    # Get station columns
    station_columns = [col for col in corrected_df.columns 
                      if col not in ['datetime', 'Date', 'Original_Datetime']]
    
    logger.info(f"Found {len(station_columns)} stations to process")
    
    # Result DataFrame
    high_res_df = pd.DataFrame()
    
    # Process each station
    for station in station_columns:
        logger.info(f"Processing station: {station}")
        
        # Get station data
        station_data = corrected_df[['datetime', station]].dropna()
        
        # Determine time resolution
        time_diffs = station_data['datetime'].diff()[1:].dt.total_seconds() / 60
        if len(time_diffs) > 0:
            resolution = int(round(time_diffs.median()))
            logger.info(f"Detected time resolution: {resolution} minutes")
        else:
            resolution = 60
            logger.info("Not enough data to determine resolution. Assuming hourly (60 min)")
        
        # Process based on resolution
        if resolution <= 30:
            # Already high-res, use as-is
            logger.info(f"Station {station} already has high-resolution data")
            
            # Initialize result DataFrame if first station
            if high_res_df.empty:
                high_res_df = station_data.copy()
            else:
                # Merge with existing data
                high_res_df = pd.merge(high_res_df, station_data, on='datetime', how='outer')
            
        elif resolution == 60:
            # Hourly data that needs disaggregation
            logger.info(f"Station {station} has hourly data. Disaggregating to half-hourly")
            
            # Create half-hourly data
            half_hourly_rows = []
            
            for idx, row in station_data.iterrows():
                hour_dt = row['datetime']
                hour_value = row[station]
                
                # Skip zero or NaN values
                if hour_value == 0 or pd.isna(hour_value):
                    half_hourly_rows.append({
                        'datetime': hour_dt,
                        station: 0
                    })
                    half_hourly_rows.append({
                        'datetime': hour_dt + pd.Timedelta(minutes=30),
                        station: 0
                    })
                    continue
                
                # Try to find IMERG pattern
                distribution = [0.5, 0.5]  # Default equal distribution
                
                if imerg_df is not None and station in imerg_df.columns:
                    # Get IMERG values for this hour
                    hour_imerg = imerg_df[
                        (imerg_df['datetime'] >= hour_dt) & 
                        (imerg_df['datetime'] < hour_dt + pd.Timedelta(hours=1))
                    ]
                    
                    if len(hour_imerg) == 2:  # Two half-hour values
                        imerg_total = hour_imerg[station].sum()
                        
                        if imerg_total > 0:
                            # Use IMERG's distribution pattern
                            distribution = [
                                hour_imerg[station].iloc[0] / imerg_total,
                                hour_imerg[station].iloc[1] / imerg_total
                            ]
                            logger.debug(f"Using IMERG pattern: {distribution[0]:.2f}/{distribution[1]:.2f}")
                
                # Create two half-hourly records
                half_hourly_rows.append({
                    'datetime': hour_dt,
                    station: hour_value * distribution[0]
                })
                half_hourly_rows.append({
                    'datetime': hour_dt + pd.Timedelta(minutes=30),
                    station: hour_value * distribution[1]
                })
            
            # Convert to DataFrame
            station_half_hourly = pd.DataFrame(half_hourly_rows)
            
            # Add to result
            if high_res_df.empty:
                high_res_df = station_half_hourly
            else:
                high_res_df = pd.merge(high_res_df, station_half_hourly, on='datetime', how='outer')
        
        else:
            logger.warning(f"Unusual time resolution ({resolution} min) for {station}. Using as-is.")
            
            # Initialize result DataFrame if first station
            if high_res_df.empty:
                high_res_df = station_data.copy()
            else:
                # Merge with existing data
                high_res_df = pd.merge(high_res_df, station_data, on='datetime', how='outer')
    
    # Sort by datetime and save
    if not high_res_df.empty:
        high_res_df = high_res_df.sort_values('datetime')
        output_file = os.path.join(output_dir, "high_resolution_precipitation.csv")
        high_res_df.to_csv(output_file, index=False)
        logger.info(f"Saved high-resolution data to {output_file} with {len(high_res_df)} records")
        
        # Try to create visualization
        try:
            from hydroanalysis.visualization.timeseries import plot_high_resolution_precip
            
            # Get top 3 stations by total precipitation
            total_precip = high_res_df.drop('datetime', axis=1).sum()
            top_stations = total_precip.nlargest(3).index.tolist()
            
            if top_stations:
                # Find a good 3-day period to visualize
                high_res_df['date'] = high_res_df['datetime'].dt.date
                daily_sums = high_res_df.groupby('date')[top_stations[0]].sum()
                
                if not daily_sums.empty:
                    max_day = daily_sums.idxmax()
                    
                    # Define 3-day window
                    start_date = pd.Timestamp(max_day) - pd.Timedelta(days=1)
                    end_date = pd.Timestamp(max_day) + pd.Timedelta(days=1)
                    
                    # Plot high-resolution precipitation
                    viz_output = os.path.join(output_dir, "high_resolution_sample.png")
                    plot_high_resolution_precip(
                        high_res_df,
                        start_date=start_date,
                        end_date=end_date,
                        stations=top_stations,
                        title="Sample 3-Day High-Resolution Precipitation",
                        output_path=viz_output
                    )
                    logger.info(f"Created visualization: {viz_output}")
        except Exception as e:
            logger.warning(f"Failed to create visualization: {e}")
        
        return high_res_df
    else:
        logger.error("Failed to create high-resolution data")
        return None

# For backward compatibility
def disaggregate_to_half_hourly(hourly_df, pattern_df=None, station=None):
    """
    Legacy function for backward compatibility.
    
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
    logger.warning("Using legacy disaggregate_to_half_hourly function. Consider using create_high_resolution_precipitation instead.")
    
    if station is None:
        # If no station provided, process all precipitation columns
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
    
    # Ensure we have a datetime column
    if 'datetime' not in hourly_df.columns and 'Date' in hourly_df.columns:
        hourly_df = hourly_df.copy()
        hourly_df['datetime'] = hourly_df['Date']
    
    if 'datetime' not in hourly_df.columns:
        logger.error("Input DataFrame must have a 'datetime' or 'Date' column")
        return pd.DataFrame()
    
    # Process a single station
    station_data = hourly_df[['datetime', station]].dropna()
    
    # Create half-hourly data
    half_hourly_rows = []
    
    for idx, row in station_data.iterrows():
        hour_dt = row['datetime']
        hour_value = row[station]
        
        # Skip zero or NaN values
        if hour_value == 0 or pd.isna(hour_value):
            half_hourly_rows.append({
                'datetime': hour_dt,
                station: 0
            })
            half_hourly_rows.append({
                'datetime': hour_dt + pd.Timedelta(minutes=30),
                station: 0
            })
            continue
        
        # Try to find pattern
        distribution = [0.5, 0.5]  # Default equal distribution
        
        if pattern_df is not None and station in pattern_df.columns:
            # Ensure datetime in pattern_df
            pattern_df_with_datetime = pattern_df.copy()
            if 'datetime' not in pattern_df_with_datetime.columns and 'Date' in pattern_df_with_datetime.columns:
                pattern_df_with_datetime['datetime'] = pd.to_datetime(pattern_df_with_datetime['Date'])
            
            # Get pattern values for this hour
            hour_pattern = pattern_df_with_datetime[
                (pattern_df_with_datetime['datetime'] >= hour_dt) & 
                (pattern_df_with_datetime['datetime'] < hour_dt + pd.Timedelta(hours=1))
            ]
            
            if len(hour_pattern) == 2:  # Two half-hour values
                pattern_total = hour_pattern[station].sum()
                
                if pattern_total > 0:
                    # Use pattern's distribution
                    distribution = [
                        hour_pattern[station].iloc[0] / pattern_total,
                        hour_pattern[station].iloc[1] / pattern_total
                    ]
        
        # Create two half-hourly records
        half_hourly_rows.append({
            'datetime': hour_dt,
            station: hour_value * distribution[0]
        })
        half_hourly_rows.append({
            'datetime': hour_dt + pd.Timedelta(minutes=30),
            station: hour_value * distribution[1]
        })
    
    # Convert to DataFrame
    station_half_hourly = pd.DataFrame(half_hourly_rows)
    station_half_hourly = station_half_hourly.sort_values('datetime')
    
    return station_half_hourly

# More legacy functions for backward compatibility
def create_high_resolution_precipitation_from_corrected(corrected_dir, imerg_file=None, output_dir=None):
    """
    Legacy function for backward compatibility.
    """
    logger.warning("Using legacy create_high_resolution_precipitation_from_corrected function.")
    
    # Get flood directory (parent of corrected directory)
    flood_dir = os.path.dirname(corrected_dir)
    
    return create_high_resolution_precipitation(
        flood_dir=flood_dir,
        output_dir=output_dir,
        imerg_file=imerg_file
    )

def run_create_high_resolution_command(args, flood_dir=None):
    """
    Execute the create-high-resolution command for a specific flood event.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments
    flood_dir : str, optional
        Directory containing flood event data (if None, use args.datasets_dir)
        
    Returns
    -------
    int
        Status code (0 for success, 1 for error)
    """
    logger = logging.getLogger("hydroanalysis.cli")
    
    # Determine datasets directory
    datasets_dir = flood_dir if flood_dir is not None else args.datasets_dir
    
    if not os.path.isdir(datasets_dir):
        logger.error(f"Invalid datasets directory: {datasets_dir}")
        return 1
    
    # Set output directory
    if args.output:
        output_dir = args.output
    else:
        output_dir = os.path.join(datasets_dir, "highres")
    
    # Check for IMERG file if parameter provided
    imerg_file = args.imerg_file if hasattr(args, 'imerg_file') else None
    
    # Create high-resolution precipitation data
    high_res_data = create_high_resolution_precipitation(
        flood_dir=datasets_dir,
        output_dir=output_dir,
        imerg_file=imerg_file
    )
    
    if high_res_data is None:
        logger.error("Failed to create high-resolution precipitation data.")
        return 1
    
    logger.info(f"Successfully created high-resolution precipitation data with {len(high_res_data)} records.")
    return 0

# More legacy functions to maintain backwards compatibility
def disaggregate_daily_to_hourly(daily_df, hourly_pattern=None, station=None):
    """Legacy function kept for backward compatibility only."""
    logger.warning("disaggregate_daily_to_hourly is deprecated and not fully implemented.")
    return pd.DataFrame()  # Return empty DataFrame