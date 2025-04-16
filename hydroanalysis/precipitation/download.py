"""
Functions for downloading precipitation data from Google Earth Engine.

This module provides functionality to:
1. Download precipitation data from different GEE datasets (ERA5-Land, GSMaP, IMERG)
2. Process the data correctly for Nepal's timezone and reporting convention
3. Format the data to be compatible with the HydroAnalysis toolkit
"""

import ee
import pandas as pd
import numpy as np
import datetime
import time
import os
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def initialize_gee(project=None):
    """Initialize Google Earth Engine with optional project ID"""
    try:
        if project:
            ee.Initialize(project=project)
        else:
            ee.Initialize()
        logger.info("Earth Engine initialized successfully.")
        return True
    except Exception as e:
        logger.error(f"Error initializing Earth Engine: {e}")
        logger.error("Please authenticate with Google Earth Engine first.")
        logger.error("Run 'earthengine authenticate' in your terminal")
        return False

def get_precipitation_datasets():
    """Returns information about available precipitation datasets in GEE"""
    # List of precipitation datasets in GEE with their temporal resolution
    datasets = [
        {
            'id': 'JAXA/GPM_L3/GSMaP/v8/operational',
            'name': 'GSMaP Operational V8',
            'temporal_resolution': '1 hour',
            'spatial_resolution': '0.1 degrees (~10 km)',
            'coverage': 'Global (60°N-60°S)',
            'period': '1998-present',
            'band': 'hourlyPrecipRate',
            'unit': 'mm/hr',
            'description': 'JAXA Global Satellite Mapping of Precipitation',
            'accumulation_type': 'hourly_rate'  # Not accumulation - reports rainfall rate
        },
        {
            'id': 'ECMWF/ERA5_LAND/HOURLY',
            'name': 'ERA5-Land Hourly',
            'temporal_resolution': '1 hour',
            'spatial_resolution': '0.1 degrees (~10 km)',
            'coverage': 'Global',
            'period': '1950-present (with delay)',
            'band': 'total_precipitation',
            'unit': 'meters',
            'description': 'ECMWF ERA5-Land Reanalysis',
            'accumulation_type': 'daily_accumulation'  # Accumulates within day, resets at midnight
        },
        {
            'id': 'NASA/GPM_L3/IMERG_V07',
            'name': 'GPM IMERG V07',
            'temporal_resolution': '30 minutes',
            'spatial_resolution': '0.1 degrees (~10 km)',
            'coverage': 'Global (60°N-60°S)',
            'period': '2000-present',
            'band': 'precipitation',
            'unit': 'mm/hr',
            'description': 'NASA Global Precipitation Measurement (GPM) Integrated Multi-satellitE Retrievals for GPM (IMERG)',
            'accumulation_type': 'hourly_rate'  # Not accumulation - reports rainfall rate
        }
    ]
    
    return datasets

def validate_precipitation_data(df, station_id, dataset_name):
    """
    Performs validation checks on precipitation data and applies fixes
    
    Parameters:
    df: DataFrame with precipitation data
    station_id: Column name for the station
    dataset_name: Name of the dataset for logging
    
    Returns:
    Tuple of (cleaned_df, validation_info)
    """
    # Create a copy to avoid modifying the original data
    df_clean = df.copy()
    
    validation_info = {
        "original_min": df_clean[station_id].min(),
        "original_max": df_clean[station_id].max(),
        "original_mean": df_clean[station_id].mean(),
        "negative_values": (df_clean[station_id] < 0).sum(),
        "extreme_values": (df_clean[station_id] > 1000).sum(),  # Values over 1000mm are extreme
        "zero_values": (df_clean[station_id] == 0).sum(),
        "nan_values": df_clean[station_id].isna().sum(),
        "needs_rescaling": False,
        "scaling_factor": 1.0
    }
    
    # Examine for unit issues and extreme values
    max_val = validation_info["original_max"]
    mean_val = validation_info["original_mean"]
    
    # Check for extreme values that would suggest unit issues
    if max_val > 10000:  # Extreme values - likely a unit issue
        # Values in millions might be from excessive conversions
        if max_val > 1000000:
            validation_info["scaling_factor"] = 1000000.0
            validation_info["needs_rescaling"] = True
            logger.warning(f"{dataset_name} - {station_id}: Extreme values detected (max: {max_val:.2f}). " +
                           f"Scaling by 1/{validation_info['scaling_factor']:.0f}")
        # Values in thousands might be mistakenly converted already
        elif max_val > 10000:
            validation_info["scaling_factor"] = 10000.0
            validation_info["needs_rescaling"] = True
            logger.warning(f"{dataset_name} - {station_id}: Very large values detected (max: {max_val:.2f}). " +
                           f"Scaling by 1/{validation_info['scaling_factor']:.0f}")
    # Mean value over 100 suggests the data might already be in mm rather than m
    elif mean_val > 100:
        validation_info["scaling_factor"] = 100.0
        validation_info["needs_rescaling"] = True
        logger.warning(f"{dataset_name} - {station_id}: Mean value ({mean_val:.2f}) suggests data is already in mm. " +
                       f"Scaling by 1/{validation_info['scaling_factor']:.0f}")
    
    # Apply scaling if needed
    if validation_info["needs_rescaling"]:
        df_clean[station_id] = df_clean[station_id] / validation_info["scaling_factor"]
        logger.info(f"{dataset_name} - {station_id}: Applied scaling factor of 1/{validation_info['scaling_factor']:.0f}")
    
    # Handle negative values
    neg_mask = df_clean[station_id] < 0
    if neg_mask.sum() > 0:
        logger.warning(f"{dataset_name} - {station_id}: Found {neg_mask.sum()} negative precipitation values. Setting to zero.")
        df_clean.loc[neg_mask, station_id] = 0
    
    # Check for NaN values
    nan_mask = df_clean[station_id].isna()
    if nan_mask.sum() > 0:
        logger.warning(f"{dataset_name} - {station_id}: Found {nan_mask.sum()} NaN precipitation values.")
    
    # Check extreme precipitation (anything over 200mm/hour is very extreme)
    extreme_mask = df_clean[station_id] > 200
    if extreme_mask.sum() > 0:
        logger.warning(f"{dataset_name} - {station_id}: Found {extreme_mask.sum()} extreme precipitation values (>200mm). " +
                       "Consider checking these values.")
    
    # Update validation info with cleaned data stats
    validation_info["cleaned_min"] = df_clean[station_id].min()
    validation_info["cleaned_max"] = df_clean[station_id].max() 
    validation_info["cleaned_mean"] = df_clean[station_id].mean()
    
    return df_clean, validation_info

def process_era5_precipitation(hourly_df, station_id, dataset_name):
    """
    Process ERA5-Land hourly precipitation data correctly
    Handles the accumulated precipitation format and unit conversion issues
    
    Parameters:
    hourly_df: DataFrame with ERA5-Land hourly precipitation data
    station_id: The station ID column to process
    dataset_name: Name of the dataset for logging
    
    Returns:
    DataFrame with corrected hourly precipitation
    """
    logger.info(f"Processing ERA5-Land hourly precipitation for {station_id}")
    
    # Make a copy to avoid SettingWithCopyWarning
    df = hourly_df.copy()
    
    # Check data and apply corrections if needed
    df, validation_info = validate_precipitation_data(df, station_id, dataset_name)
    
    # Unit conversion only if not already done and not already in mm
    if not validation_info["needs_rescaling"] and validation_info["cleaned_max"] < 1:
        logger.info(f"{station_id}: Data appears to be in meters. Converting to mm.")
        df[station_id] = df[station_id] * 1000  # Convert m to mm
    else:
        logger.info(f"{station_id}: Data likely already in mm or has been scaled appropriately.")
    
    # Process by day to handle accumulation correctly
    logger.info(f"{station_id}: Processing daily accumulation")
    
    # Create a datetime with date only for grouping by day
    df['day'] = df['datetime'].dt.date
    df['hour'] = df['datetime'].dt.hour
    
    # Process each day separately
    days = df['day'].unique()
    result_rows = []
    hourly_stats = {"total_days": len(days), "processed_days": 0, "total_hours": 0}
    
    for day in days:
        # Get data for this day
        day_data = df[df['day'] == day].copy()
        hourly_stats["total_hours"] += len(day_data)
        
        if len(day_data) > 1:  # Need at least 2 points to calculate differences
            # Sort by hour
            day_data = day_data.sort_values('hour')
            
            # Calculate differences between consecutive hours to get hourly values
            # This is necessary because ERA5 precipitation is accumulated within the day
            day_data[f'{station_id}_hourly'] = day_data[station_id].diff()
            
            # The first hour of the day is already an hourly value
            # (it's the amount since midnight)
            first_idx = day_data.index[0]
            day_data.loc[first_idx, f'{station_id}_hourly'] = day_data.loc[first_idx, station_id]
            
            # Replace accumulated values with hourly values
            day_data[station_id] = day_data[f'{station_id}_hourly']
            
            # Handle negative hourly values (physically impossible)
            neg_mask = day_data[station_id] < 0
            if neg_mask.sum() > 0:
                logger.debug(f"{station_id}: Found {neg_mask.sum()} negative hourly values for {day}. Setting to zero.")
                day_data.loc[neg_mask, station_id] = 0
            
            # Add to results
            result_rows.append(day_data)
            hourly_stats["processed_days"] += 1
        else:
            # Only one record for this day - can't calculate differences
            # Keep as is
            logger.debug(f"{station_id}: Only one record for {day}, keeping as is.")
            result_rows.append(day_data)
    
    # Combine all processed daily data
    if result_rows:
        result_df = pd.concat(result_rows)
        # Drop temporary columns
        result_df = result_df.drop(columns=['day', 'hour', f'{station_id}_hourly'], errors='ignore')
        logger.info(f"{station_id}: Successfully processed {hourly_stats['processed_days']} days with {hourly_stats['total_hours']} total hours")
    else:
        logger.warning(f"{station_id}: No data to process!")
        result_df = df
    
    # Final validation
    final_stats = {
        "min": result_df[station_id].min(),
        "max": result_df[station_id].max(),
        "mean": result_df[station_id].mean(),
        "zeros": (result_df[station_id] == 0).sum(),
        "non_zeros": (result_df[station_id] > 0).sum()
    }
    
    logger.info(f"{station_id}: Final data - Min: {final_stats['min']:.2f}, Max: {final_stats['max']:.2f}, " +
                f"Mean: {final_stats['mean']:.2f}, Zeros: {final_stats['zeros']}, Non-zeros: {final_stats['non_zeros']}")
    
    return result_df

def process_rate_precipitation(hourly_df, station_id, dataset_name):
    """
    Process hourly precipitation rate data (GSMaP, IMERG)
    
    Parameters:
    hourly_df: DataFrame with hourly precipitation rate data
    station_id: The station ID column to process
    dataset_name: Name of the dataset for logging
    
    Returns:
    DataFrame with validated precipitation rates
    """
    logger.info(f"Processing precipitation rate data for {station_id} from {dataset_name}")
    
    # Make a copy to avoid SettingWithCopyWarning
    df = hourly_df.copy()
    
    # Validate and clean the data
    df, validation_info = validate_precipitation_data(df, station_id, dataset_name)
    
    # For rate-based datasets (GSMaP, IMERG), values are already rainfall rates (mm/hr)
    # Just need to check for invalid values
    
    # Calculate time intervals between measurements
    df = df.sort_values('datetime')
    df['time_diff'] = df['datetime'].diff().dt.total_seconds() / 3600  # Convert to hours
    
    # For IMERG, check if it's 30-min data and adjust accordingly
    if 'IMERG' in dataset_name:
        # Calculate median time difference
        valid_diffs = df['time_diff'].dropna()
        if len(valid_diffs) > 0:
            median_diff = valid_diffs.median()
            logger.info(f"{station_id}: Median time difference: {median_diff:.2f} hours")
            
            # If it's around 30 minutes (0.5 hours), adjust the rate
            if 0.4 <= median_diff <= 0.6:
                logger.info(f"{station_id}: Detected half-hourly data, adjusting rates")
                df[station_id] = df[station_id] * median_diff
    
    # Drop the temporary column
    df = df.drop(columns=['time_diff'], errors='ignore')
    
    # Final validation
    final_stats = {
        "min": df[station_id].min(),
        "max": df[station_id].max(),
        "mean": df[station_id].mean(),
        "zeros": (df[station_id] == 0).sum(),
        "non_zeros": (df[station_id] > 0).sum()
    }
    
    logger.info(f"{station_id}: Final data - Min: {final_stats['min']:.2f}, Max: {final_stats['max']:.2f}, " +
                f"Mean: {final_stats['mean']:.2f}, Zeros: {final_stats['zeros']}, Non-zeros: {final_stats['non_zeros']}")
    
    return df

def get_hourly_precipitation_from_gee(metadata_df, start_date, end_date, dataset_id, band, 
                                     adjust_nepal_timezone=True, station_id_col='Station_ID'):
    """
    Get hourly precipitation data from Google Earth Engine for each station location
    
    Parameters:
    metadata_df: DataFrame containing station locations with columns for Station_ID, Latitude, and Longitude
    start_date: Start date (datetime object) for data collection
    end_date: End date (datetime object) for data collection
    dataset_id: GEE dataset ID to use
    band: Band name for precipitation
    adjust_nepal_timezone: Whether to adjust times to Nepal local time (+5:45)
    station_id_col: Column name in metadata_df that contains station IDs
    
    Returns:
    DataFrame containing hourly precipitation for each station
    """
    # Create an empty DataFrame to store hourly precipitation data
    hourly_data = pd.DataFrame()
    
    # Format dates for GEE - we need to adjust the request to account for shifts
    # We'll fetch one day earlier and one day later to ensure we have enough data
    # after the Nepal time zone and recording adjustments if needed
    request_start_date = start_date - datetime.timedelta(days=1 if adjust_nepal_timezone else 0)
    request_end_date = end_date + datetime.timedelta(days=2 if adjust_nepal_timezone else 1)
    
    start_date_str = request_start_date.strftime('%Y-%m-%d')
    end_date_str = request_end_date.strftime('%Y-%m-%d')
    
    logger.info(f"Fetching GEE data from {start_date_str} to {end_date_str}")
    if adjust_nepal_timezone:
        logger.info(f"(Expanded range to account for Nepal time zone adjustments)")
    logger.info(f"Using dataset: {dataset_id}")
    
    # Get information about available datasets
    datasets = get_precipitation_datasets()
    dataset_info = next((d for d in datasets if d['id'] == dataset_id), None)
    
    if dataset_info:
        logger.info(f"Dataset: {dataset_info['name']}")
        logger.info(f"Temporal resolution: {dataset_info['temporal_resolution']}")
        logger.info(f"Spatial resolution: {dataset_info['spatial_resolution']}")
        logger.info(f"Accumulation type: {dataset_info['accumulation_type']}")
    
    # Loop through each station in the metadata
    for index, row in metadata_df.iterrows():
        station_id = row[station_id_col]
        lat = row['Latitude']
        lon = row['Longitude']
        
        # If station_id doesn't start with 'Station_', add the prefix
        if not str(station_id).startswith('Station_'):
            station_id = f"Station_{station_id}"
        
        logger.info(f"Processing {station_id} at location ({lat}, {lon})...")
        
        try:
            # Define the point of interest
            point = ee.Geometry.Point([float(lon), float(lat)])
            
            # Get the precipitation data from GEE
            collection = ee.ImageCollection(dataset_id) \
                .filterDate(start_date_str, end_date_str) \
                .select(band)
            
            # Extract data at the point location
            logger.info(f"Extracting GEE data for {station_id}...")
            
            # Define a function to extract values at the point
            def extract_value(image):
                value = image.sample(point, 500).first().get(band)
                return ee.Feature(None, {
                    'precipitation': value,
                    'timestamp': image.date().millis()
                })
            
            # Map the function over the collection
            features = collection.map(extract_value)
            
            # Get the data
            station_data = features.getInfo()['features']
            
            # Convert to DataFrame
            if station_data:
                station_df = pd.DataFrame([
                    {
                        'datetime': datetime.datetime.fromtimestamp(feature['properties']['timestamp'] / 1000),
                        station_id: feature['properties']['precipitation']
                    }
                    for feature in station_data
                ])
                
                # Handle case where no data was returned
                if station_df.empty:
                    logger.warning(f"No data returned for {station_id}. Skipping...")
                    continue
                
                # Record original data for validation
                original_time_range = (station_df['datetime'].min(), station_df['datetime'].max())
                original_precip_range = (station_df[station_id].min(), station_df[station_id].max())
                
                # Apply Nepal time adjustment if requested
                if adjust_nepal_timezone:
                    # STEP 1: TIME ZONE CONVERSION - Convert from UTC to Nepal time (UTC+5:45)
                    nepal_offset = datetime.timedelta(hours=5, minutes=45)
                    station_df['datetime'] = station_df['datetime'] + nepal_offset
                    logger.info(f"  Converted timestamps from UTC to Nepal time (UTC+5:45)")
                    
                    # Print updated time range for validation
                    logger.info(f"  Original time range: {original_time_range[0]} to {original_time_range[1]} (UTC)")
                    logger.info(f"  Adjusted time range: {station_df['datetime'].min()} to {station_df['datetime'].max()} (Nepal time)")
                
                # STEP 3: PROCESS BASED ON DATASET TYPE
                if dataset_id == 'ECMWF/ERA5_LAND/HOURLY' and band == 'total_precipitation':
                    # Process ERA5-Land accumulated precipitation
                    station_df = process_era5_precipitation(station_df, station_id, dataset_info['name'])
                else:
                    # Process rate-based precipitation (GSMaP, IMERG)
                    station_df = process_rate_precipitation(station_df, station_id, dataset_info['name'])
                
                # Merge with the main DataFrame
                if hourly_data.empty:
                    hourly_data = station_df
                else:
                    hourly_data = pd.merge(hourly_data, station_df, on='datetime', how='outer')
            else:
                logger.warning(f"No data returned for {station_id}. Skipping...")
            
            # Add a small delay to avoid rate limiting
            time.sleep(1)
        
        except Exception as e:
            logger.error(f"Error processing station {station_id}: {e}")
            logger.error("Continuing with next station...")
    
    if hourly_data.empty:
        logger.warning("Warning: No data was retrieved from GEE.")
        return hourly_data
    
    # Sort by datetime
    hourly_data = hourly_data.sort_values('datetime')
    
    # Check for missing data
    missing_count = hourly_data.isnull().sum()
    logger.info("\nMissing data summary:")
    for col, count in missing_count.items():
        if col != 'datetime':
            logger.info(f"  {col}: {count} missing values ({count/len(hourly_data)*100:.1f}%)")
    
    # Filter to match the original date range, accounting for adjustments
    if adjust_nepal_timezone:
        # If we've adjusted for Nepal timezone, filter based on Nepal time
        adjusted_start = start_date
        adjusted_end = end_date
    else:
        # Otherwise, just use the original dates
        adjusted_start = start_date
        adjusted_end = end_date
    
    # Add a buffer to ensure we capture the full day
    adjusted_start = adjusted_start.replace(hour=0, minute=0, second=0)
    adjusted_end = adjusted_end.replace(hour=23, minute=59, second=59)
    
    logger.info(f"\nFiltering data to match original request date range:")
    logger.info(f"  Date range: {adjusted_start} to {adjusted_end}")
    
    filtered_hourly_data = hourly_data[(hourly_data['datetime'] >= adjusted_start) & 
                                       (hourly_data['datetime'] <= adjusted_end)]
    
    logger.info(f"  Kept {len(filtered_hourly_data)} of {len(hourly_data)} records after filtering")
    
    return filtered_hourly_data

def aggregate_hourly_to_daily(hourly_df, dataset_id=None, nepal_convention=False):
    """
    Aggregate hourly precipitation data to daily totals
    
    Parameters:
    hourly_df: DataFrame with hourly data, with a 'datetime' column
    dataset_id: Optional - the GEE dataset ID for logging
    nepal_convention: Whether to use Nepal's recording convention (shift daily totals forward by one day)
    
    Returns:
    DataFrame with daily precipitation totals
    """
    if hourly_df.empty:
        logger.warning("No hourly data to aggregate.")
        return pd.DataFrame()
    
    # Make a copy to avoid SettingWithCopyWarning
    df = hourly_df.copy()
    
    # Create a date column based on the convention
    if nepal_convention:
        # For Nepal's recording convention, ALL precipitation should be 
        # attributed to the NEXT day to match station reporting
        df['Date'] = df['datetime'].dt.date + pd.Timedelta(days=1)
        logger.info("Applied Nepal's recording convention: ALL precipitation shifted to next day")
    else:
        # Standard calendar day aggregation (midnight to midnight)
        df['Date'] = df['datetime'].dt.date
    
    # Get station columns
    station_columns = [col for col in df.columns if col not in ['datetime', 'Date']]
    
    if dataset_id:
        # Get dataset information for logging
        datasets = get_precipitation_datasets()
        dataset_info = next((d for d in datasets if d['id'] == dataset_id), None)
        accumulation_type = dataset_info['accumulation_type'] if dataset_info else None
        
        logger.info(f"\nAggregating hourly data to daily for dataset: {dataset_id}")
        logger.info(f"Accumulation type: {accumulation_type}")
    else:
        logger.info("\nAggregating hourly data to daily")
    
    # Drop the datetime column
    df = df.drop('datetime', axis=1)
    
    # Group by date and sum
    daily_df = df.groupby('Date').sum().reset_index()
    
    # Convert Date to datetime
    daily_df['Date'] = pd.to_datetime(daily_df['Date'])
    
    logger.info(f"Aggregated hourly data to {len(daily_df)} daily records.")
    
    # Print summary statistics for each station
    for col in station_columns:
        min_val = daily_df[col].min()
        max_val = daily_df[col].max()
        mean_val = daily_df[col].mean()
        logger.info(f"  {col}: Mean={mean_val:.2f}, Max={max_val:.2f}, Min={min_val:.2f} mm/day")
    
    return daily_df

def download_precipitation_data(metadata_df, dataset_name, start_date, end_date, output_dir, 
                               adjust_nepal_timezone=False, nepal_convention=False, resolution='daily', 
                               station_id_col='Station_ID'):
    """
    Download precipitation data from Earth Engine for all stations in metadata
    and save in formats compatible with HydroAnalysis toolkit
    
    Parameters:
    metadata_df: DataFrame with station metadata (must have Station_ID, Latitude, Longitude columns)
    dataset_name: Name of dataset to download ('era5', 'gsmap', or 'imerg')
    start_date: Start date as string 'YYYY-MM-DD' or datetime object
    end_date: End date as string 'YYYY-MM-DD' or datetime object
    output_dir: Directory to save output files
    adjust_nepal_timezone: Whether to adjust for Nepal timezone (+5:45)
    nepal_convention: Whether to use Nepal's convention (shift to next day)
    resolution: 'daily' or 'hourly'
    station_id_col: Column name in metadata_df that contains station IDs
    
    Returns:
    Tuple of (hourly_df, daily_df) containing downloaded data
    """
    # Convert dates to datetime objects if they're strings
    if isinstance(start_date, str):
        start_date = datetime.datetime.strptime(start_date, '%Y-%m-%d')
    if isinstance(end_date, str):
        end_date = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Map dataset name to GEE dataset ID and band
    dataset_mapping = {
        'era5': ('ECMWF/ERA5_LAND/HOURLY', 'total_precipitation'),
        'gsmap': ('JAXA/GPM_L3/GSMaP/v8/operational', 'hourlyPrecipRate'),
        'imerg': ('NASA/GPM_L3/IMERG_V07', 'precipitation')
    }
    
    # Verify dataset name is valid
    if dataset_name.lower() not in dataset_mapping:
        valid_names = ', '.join(dataset_mapping.keys())
        logger.error(f"Invalid dataset name '{dataset_name}'. Valid options are: {valid_names}")
        return None, None
    
    # Get dataset ID and band name
    dataset_id, band = dataset_mapping[dataset_name.lower()]
    
    # Get the data from Earth Engine
    logger.info(f"Downloading {dataset_name} precipitation data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Check if there's a project ID set in the environment
    project_id = 'ee-jsuhydrolabenb'
    
    # Initialize Earth Engine
    if not initialize_gee(project=project_id):
        logger.error("Failed to initialize Earth Engine. Aborting download.")
        return None, None
    
    # Get hourly data
    hourly_df = get_hourly_precipitation_from_gee(
        metadata_df, 
        start_date, 
        end_date, 
        dataset_id, 
        band,
        adjust_nepal_timezone=adjust_nepal_timezone,
        station_id_col=station_id_col
    )
    
    if hourly_df.empty:
        logger.error("No data retrieved from Earth Engine.")
        return None, None
    
    # Save hourly data if requested
    if resolution.lower() == 'hourly' or resolution.lower() == 'both':
        # Create a separate datetime column formatted as string for output
        hourly_output = hourly_df.copy()
        hourly_output['Date'] = hourly_output['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Rename column for consistency
        hourly_output = hourly_output.rename(columns={'datetime': 'Original_Datetime'})
        
        # Reorder columns to put Date first
        cols = ['Date'] + [col for col in hourly_output.columns if col != 'Date']
        hourly_output = hourly_output[cols]
        
        # Save to CSV
        hourly_filename = os.path.join(output_dir, f"{dataset_name}_hourly_precipitation.csv")
        hourly_output.to_csv(hourly_filename, index=False)
        logger.info(f"Hourly data saved to {hourly_filename}")
    
    # Aggregate to daily and save
    daily_df = aggregate_hourly_to_daily(hourly_df, dataset_id, nepal_convention=nepal_convention)
    
    if not daily_df.empty:
        # Format date for output
        daily_output = daily_df.copy()
        daily_output['Date'] = daily_output['Date'].dt.strftime('%Y-%m-%d')
        
        # Save to CSV in the format expected by the HydroAnalysis toolkit
        daily_filename = os.path.join(output_dir, f"{dataset_name}_daily_precipitation.csv")
        daily_output.to_csv(daily_filename, index=False)
        logger.info(f"Daily data saved to {daily_filename}")
    
    # Return both dataframes for further use if needed
    return hourly_df, daily_df

def download_flood_precipitation(flood_dir, metadata_df, dataset_name, output_dir=None, 
                               adjust_nepal_timezone=False, nepal_convention=False, 
                               resolution='daily', station_id_col='Station_ID'):
    """
    Download precipitation data from Earth Engine for a specific flood event
    
    Parameters:
    flood_dir: Directory containing flood event data (discharge.csv and precipitation.csv)
    metadata_df: DataFrame with station metadata (must have Station_ID, Latitude, Longitude columns)
    dataset_name: Name of dataset to download ('era5', 'gsmap', or 'imerg')
    output_dir: Directory to save output files (defaults to flood_dir if None)
    adjust_nepal_timezone: Whether to adjust for Nepal timezone (+5:45)
    nepal_convention: Whether to use Nepal's recording convention (shift to next day)
    resolution: 'daily' or 'hourly'
    station_id_col: Column name in metadata_df that contains station IDs
    
    Returns:
    Tuple of (hourly_df, daily_df) containing downloaded data
    """
    # Set output directory to flood directory if not specified
    if output_dir is None:
        output_dir = flood_dir
    
    # Read discharge data to get the date range
    discharge_file = os.path.join(flood_dir, 'discharge.csv')
    
    if not os.path.exists(discharge_file):
        logger.error(f"Discharge file not found in {flood_dir}")
        return None, None
    
    try:
        discharge_df = pd.read_csv(discharge_file)
        discharge_df['Date'] = pd.to_datetime(discharge_df['Date'])
        
        # Get start and end dates from discharge data
        start_date = discharge_df['Date'].min()
        end_date = discharge_df['Date'].max()
        
        # Add buffer days to capture more context (optional)
        # start_date = start_date - datetime.timedelta(days=1)
        # end_date = end_date + datetime.timedelta(days=1)
        
        logger.info(f"Flood event period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Download precipitation data for this period
        hourly_df, daily_df = download_precipitation_data(
            metadata_df=metadata_df,
            dataset_name=dataset_name,
            start_date=start_date,
            end_date=end_date,
            output_dir=output_dir,
            adjust_nepal_timezone=adjust_nepal_timezone,
            nepal_convention=nepal_convention,
            resolution=resolution,
            station_id_col=station_id_col
        )
        
        # Create a directory for GEE data in the flood directory
        gee_dir = os.path.join(output_dir, f'gee_{dataset_name}')
        os.makedirs(gee_dir, exist_ok=True)
        
        # Save files with more descriptive names in the GEE directory
        if hourly_df is not None and not hourly_df.empty and (resolution == 'hourly' or resolution == 'both'):
            hourly_file = os.path.join(gee_dir, f'hourly_precipitation.csv')
            
            # Create a copy for saving with appropriate formatting
            hourly_output = hourly_df.copy()
            hourly_output['Date'] = hourly_output['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
            hourly_output = hourly_output.rename(columns={'datetime': 'Original_Datetime'})
            
            # Reorder columns to put Date first
            cols = ['Date'] + [col for col in hourly_output.columns if col != 'Date']
            hourly_output = hourly_output[cols]
            
            hourly_output.to_csv(hourly_file, index=False)
            logger.info(f"Hourly data saved to {hourly_file}")
        
        if daily_df is not None and not daily_df.empty:
            daily_file = os.path.join(gee_dir, f'daily_precipitation.csv')
            
            # Create a copy for saving with appropriate formatting
            daily_output = daily_df.copy()
            daily_output['Date'] = daily_output['Date'].dt.strftime('%Y-%m-%d')
            
            daily_output.to_csv(daily_file, index=False)
            logger.info(f"Daily data saved to {daily_file}")
        
        return hourly_df, daily_df
    
    except Exception as e:
        logger.error(f"Error processing flood data: {e}")
        return None, None

def download_all_flood_precipitation(floods_dir, metadata_df, dataset_name, 
                                   adjust_nepal_timezone=False, nepal_convention=False, 
                                   resolution='daily', station_id_col='Station_ID'):
    """
    Download precipitation data for all flood events
    
    Parameters:
    floods_dir: Directory containing flood event subdirectories
    metadata_df: DataFrame with station metadata
    dataset_name: Name of dataset to download ('era5', 'gsmap', or 'imerg')
    adjust_nepal_timezone: Whether to adjust for Nepal timezone (+5:45)
    nepal_convention: Whether to use Nepal's recording convention (shift to next day)
    resolution: 'daily' or 'hourly'
    station_id_col: Column name in metadata_df that contains station IDs
    
    Returns:
    Dictionary mapping flood directories to daily dataframes
    """
    results = {}
    
    # Find all flood directories
    flood_dirs = []
    for item in os.listdir(floods_dir):
        item_path = os.path.join(floods_dir, item)
        if os.path.isdir(item_path) and item.startswith('flood_'):
            flood_dirs.append(item_path)
    
    # Sort numerically
    flood_dirs.sort(key=lambda x: int(os.path.basename(x).split('_')[1]))
    
    logger.info(f"Found {len(flood_dirs)} flood event directories")
    
    # Process each flood directory
    for flood_dir in flood_dirs:
        flood_name = os.path.basename(flood_dir)
        logger.info(f"Processing {flood_name}...")
        
        hourly_df, daily_df = download_flood_precipitation(
            flood_dir=flood_dir,
            metadata_df=metadata_df,
            dataset_name=dataset_name,
            adjust_nepal_timezone=adjust_nepal_timezone,
            nepal_convention=nepal_convention,
            resolution=resolution,
            station_id_col=station_id_col
        )
        
        if daily_df is not None:
            results[flood_dir] = daily_df
    
    return results