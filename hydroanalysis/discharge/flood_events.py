"""
Functions for identifying and analyzing flood events in discharge data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import logging

from hydroanalysis.config import CONFIG
from hydroanalysis.core.utils import calculate_percentile_threshold

logger = logging.getLogger(__name__)

def identify_flood_events(discharge_data, percentile_threshold=None, min_duration=None, buffer_days=None):
    """
    Identify flood events in discharge data.
    
    Parameters
    ----------
    discharge_data : pandas.DataFrame
        DataFrame with 'Date' and 'Discharge' columns
    percentile_threshold : float, optional
        Percentile threshold for defining a flood (default from config)
    min_duration : int, optional
        Minimum number of days above threshold to be considered a flood event (default from config)
    buffer_days : int, optional
        Number of days before and after the flood to include in analysis (default from config)
        
    Returns
    -------
    list of dict
        List of flood events with details
    """
    # Use default values from config if not specified
    if percentile_threshold is None:
        percentile_threshold = CONFIG['flood_events']['percentile_threshold']
    
    if min_duration is None:
        min_duration = CONFIG['flood_events']['min_duration']
    
    if buffer_days is None:
        buffer_days = CONFIG['flood_events']['buffer_days']
    
    # Make sure date is datetime
    discharge_data = discharge_data.copy()
    discharge_data['Date'] = pd.to_datetime(discharge_data['Date'])
    
    # Sort data by date
    discharge_data = discharge_data.sort_values('Date')
    
    # Calculate threshold based on percentile
    threshold = calculate_percentile_threshold(discharge_data['Discharge'].dropna(), percentile_threshold)
    logger.info(f"Using flood threshold of {threshold:.2f} m³/s ({percentile_threshold}th percentile)")
    
    # Find periods where discharge exceeds threshold
    discharge_data['is_flood'] = discharge_data['Discharge'] >= threshold
    
    # Identify flood events
    flood_events = []
    in_flood = False
    current_flood = None
    
    for i, row in discharge_data.reset_index().iterrows():
        if row['is_flood'] and not in_flood:
            # Start of a new flood
            in_flood = True
            current_flood = {
                'start_index': i,
                'start_date': row['Date'],
                'peak_index': i,
                'peak_date': row['Date'],
                'peak_discharge': row['Discharge'],
                'end_index': i,
                'end_date': row['Date'],
                'duration': 1,
                'discharge_values': [row['Discharge']]
            }
        elif row['is_flood'] and in_flood:
            # Continuing flood
            current_flood['end_index'] = i
            current_flood['end_date'] = row['Date']
            current_flood['duration'] += 1
            current_flood['discharge_values'].append(row['Discharge'])
            
            # Update peak if this discharge is higher
            if row['Discharge'] > current_flood['peak_discharge']:
                current_flood['peak_index'] = i
                current_flood['peak_date'] = row['Date']
                current_flood['peak_discharge'] = row['Discharge']
        elif not row['is_flood'] and in_flood:
            # End of flood
            in_flood = False
            
            # Only consider events with minimum duration
            if current_flood['duration'] >= min_duration:
                # Add buffer days for analysis
                current_flood['buffer_start_index'] = max(0, current_flood['start_index'] - buffer_days)
                current_flood['buffer_end_index'] = min(len(discharge_data)-1, current_flood['end_index'] + buffer_days)
                current_flood['buffer_start_date'] = discharge_data.iloc[current_flood['buffer_start_index']]['Date']
                current_flood['buffer_end_date'] = discharge_data.iloc[current_flood['buffer_end_index']]['Date']
                current_flood['total_duration'] = current_flood['buffer_end_index'] - current_flood['buffer_start_index'] + 1
                
                # Calculate additional flood characteristics
                current_flood['threshold'] = threshold
                current_flood['volume_above_threshold'] = sum(max(0, q - threshold) for q in current_flood['discharge_values'])
                current_flood['mean_discharge'] = np.mean(current_flood['discharge_values'])
                
                flood_events.append(current_flood)
            current_flood = None
    
    # Handle case where data ends during a flood
    if in_flood and current_flood['duration'] >= min_duration:
        current_flood['buffer_start_index'] = max(0, current_flood['start_index'] - buffer_days)
        current_flood['buffer_end_index'] = min(len(discharge_data)-1, current_flood['end_index'] + buffer_days)
        current_flood['buffer_start_date'] = discharge_data.iloc[current_flood['buffer_start_index']]['Date']
        current_flood['buffer_end_date'] = discharge_data.iloc[current_flood['buffer_end_index']]['Date']
        current_flood['total_duration'] = current_flood['buffer_end_index'] - current_flood['buffer_start_index'] + 1
        
        # Calculate additional flood characteristics
        current_flood['threshold'] = threshold
        current_flood['volume_above_threshold'] = sum(max(0, q - threshold) for q in current_flood['discharge_values'])
        current_flood['mean_discharge'] = np.mean(current_flood['discharge_values'])
        
        flood_events.append(current_flood)
    
    # Sort flood events by peak discharge (descending)
    flood_events.sort(key=lambda x: x['peak_discharge'], reverse=True)
    
    logger.info(f"Identified {len(flood_events)} flood events with at least {min_duration} days above threshold")
    
    return flood_events

def extract_precipitation_for_flood(precip_data, flood_event):
    """
    Extract precipitation data for a specific flood event.
    
    Parameters
    ----------
    precip_data : pandas.DataFrame
        DataFrame with 'Date' and precipitation columns
    flood_event : dict
        Flood event details from identify_flood_events
        
    Returns
    -------
    pandas.DataFrame
        Precipitation data during the flood period
    """
    # Make sure date is datetime
    precip_data = precip_data.copy()
    
    # Check for date column with flexible naming
    date_col = None
    
    # First, try standard date column names
    for col_name in ['Date', 'datetime', 'date', 'Datetime', 'DATE']:
        if col_name in precip_data.columns:
            date_col = col_name
            break
    
    # If no standard date column found, try to identify a date-like column
    if date_col is None:
        # Look for columns that might be dates
        for col in precip_data.columns:
            # Skip columns we've identified as precipitation data
            if str(col).startswith('Station_'):
                continue
                
            # Try to convert to datetime to check if it's a date column
            try:
                # Check a sample to see if it can be converted to datetime
                sample = precip_data[col].iloc[0]
                if pd.to_datetime(sample, errors='coerce') is not pd.NaT:
                    date_col = col
                    logger.info(f"Identified column '{col}' as potential date column")
                    break
            except:
                continue
    
    # If we still don't have a date column, check if there are columns called 'year', 'month', 'day'
    if date_col is None:
        year_col = None
        month_col = None
        day_col = None
        
        for col in precip_data.columns:
            col_str = str(col).lower()
            if 'year' in col_str:
                year_col = col
            elif 'month' in col_str:
                month_col = col
            elif 'day' in col_str:
                day_col = col
        
        # If we have year, month, and day, create a date column
        if year_col is not None and month_col is not None and day_col is not None:
            logger.info(f"Creating Date column from {year_col}, {month_col}, and {day_col}")
            
            try:
                # Convert components to numeric
                year = pd.to_numeric(precip_data[year_col], errors='coerce')
                month = pd.to_numeric(precip_data[month_col], errors='coerce')
                day = pd.to_numeric(precip_data[day_col], errors='coerce')
                
                # Create date column
                precip_data['Date'] = pd.to_datetime(
                    {'year': year, 'month': month, 'day': day}, 
                    errors='coerce'
                )
                date_col = 'Date'
            except Exception as e:
                logger.error(f"Failed to create Date column: {e}")
    
    if date_col is None:
        logger.error("Precipitation data must have a 'Date' or 'datetime' column")
        return pd.DataFrame()
    
    # Ensure date is datetime
    precip_data[date_col] = pd.to_datetime(precip_data[date_col])
    
    # Filter precipitation data for this flood period
    flood_precip = precip_data[(precip_data[date_col] >= flood_event['buffer_start_date']) & 
                              (precip_data[date_col] <= flood_event['buffer_end_date'])]
    
    # If 'Date' isn't already a column name, add it for consistency
    if date_col != 'Date':
        flood_precip = flood_precip.rename(columns={date_col: 'Date'})
    
    return flood_precip

def calculate_flood_return_period(flood_events, years_of_data):
    """
    Calculate estimated return periods for flood events.
    
    Parameters
    ----------
    flood_events : list of dict
        List of flood events from identify_flood_events
    years_of_data : int or float
        Number of years in the dataset
        
    Returns
    -------
    list of dict
        List of flood events with added return period estimates
    """
    if not flood_events:
        return []
    
    # Sort flood events by peak discharge (descending)
    sorted_events = sorted(flood_events, key=lambda x: x['peak_discharge'], reverse=True)
    
    # Calculate return periods
    for i, event in enumerate(sorted_events):
        # Weibull formula: Return Period = (n+1) / m
        # Where n is the number of years of data and m is the rank of the event
        rank = i + 1
        return_period = (years_of_data + 1) / rank
        
        # Add return period to the event
        event['return_period'] = return_period
    
    return sorted_events

def analyze_flood_events(discharge_data, precip_data, output_dir=None, max_events=10, years_of_data=None):
    """
    Analyze flood events and corresponding precipitation.
    
    Parameters
    ----------
    discharge_data : pandas.DataFrame
        DataFrame with discharge data
    precip_data : pandas.DataFrame
        DataFrame with precipitation data
    output_dir : str, optional
        Directory to save output files
    max_events : int, optional
        Maximum number of flood events to analyze
    years_of_data : int or float, optional
        Number of years in the dataset (for return period calculation)
        
    Returns
    -------
    dict
        Dictionary with analysis results
    """
    # Identify flood events
    flood_events = identify_flood_events(discharge_data)
    
    # Calculate return periods if years_of_data is provided
    if years_of_data:
        flood_events = calculate_flood_return_period(flood_events, years_of_data)
    
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Analyze each flood event
    flood_analysis = {}
    
    for i, flood in enumerate(flood_events[:max_events]):
        flood_num = i + 1
        logger.info(f"\nAnalyzing Flood #{flood_num}: Peak {flood['peak_discharge']:.2f} m³/s on {flood['peak_date']}")
        logger.info(f"  Duration: {flood['duration']} days, Analysis window: {flood['total_duration']} days")
        logger.info(f"  From {flood['buffer_start_date']} to {flood['buffer_end_date']}")
        
        # Get discharge data for this flood
        flood_discharge = discharge_data[(discharge_data['Date'] >= flood['buffer_start_date']) & 
                                        (discharge_data['Date'] <= flood['buffer_end_date'])]
        
        # Get precipitation data for this flood
        flood_precip = extract_precipitation_for_flood(precip_data, flood)
        
        # Calculate total precipitation during the flood period
        precip_columns = [col for col in flood_precip.columns if col.startswith('Station_')]
        
        total_precip = {}
        for col in precip_columns:
            total_precip[col] = flood_precip[col].sum()
        
        # Save analysis for this flood
        flood_analysis[flood_num] = {
            'flood_event': flood,
            'discharge_data': flood_discharge,
            'precipitation_data': flood_precip,
            'total_precipitation': total_precip
        }
        
        # Save to files if output directory is specified
        if output_dir:
            flood_dir = os.path.join(output_dir, f"flood_{flood_num}")
            os.makedirs(flood_dir, exist_ok=True)
            
            # Save discharge data
            flood_discharge.to_csv(os.path.join(flood_dir, 'discharge.csv'), index=False)
            
            # Save precipitation data
            flood_precip.to_csv(os.path.join(flood_dir, 'precipitation.csv'), index=False)
            
            # Save summary information
            with open(os.path.join(flood_dir, 'summary.txt'), 'w') as f:
                f.write(f"Flood #{flood_num}\n")
                f.write(f"==========\n\n")
                f.write(f"Start date: {flood['start_date']}\n")
                f.write(f"Peak date: {flood['peak_date']}\n")
                f.write(f"Peak discharge: {flood['peak_discharge']:.2f} m³/s\n")
                f.write(f"End date: {flood['end_date']}\n")
                f.write(f"Duration: {flood['duration']} days\n\n")
                
                if 'return_period' in flood:
                    f.write(f"Estimated return period: {flood['return_period']:.1f} years\n\n")
                
                f.write(f"Analysis period: {flood['buffer_start_date']} to {flood['buffer_end_date']} ({flood['total_duration']} days)\n\n")
                
                f.write(f"Total precipitation (mm):\n")
                for col, total in total_precip.items():
                    station_id = col.replace('Station_', '')
                    # Make sure we handle the case where total might not be a number
                    if isinstance(total, (int, float)):
                        f.write(f"  Station {station_id}: {total:.1f} mm\n")
                    else:
                        f.write(f"  Station {station_id}: {total} mm\n")
    
    logger.info(f"\nAnalysis complete. Results saved to {output_dir if output_dir else 'memory'}")
    return flood_analysis

def calculate_flood_frequency(discharge_data, thresholds=None):
    """
    Calculate flood frequency for different thresholds.
    
    Parameters
    ----------
    discharge_data : pandas.DataFrame
        DataFrame with 'Date' and 'Discharge' columns
    thresholds : list of float, optional
        List of discharge thresholds for flood definition
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with flood frequency statistics
    """
    if thresholds is None:
        # If no thresholds provided, use percentiles
        percentiles = [50, 75, 90, 95, 99]
        thresholds = [calculate_percentile_threshold(discharge_data['Discharge'], p) for p in percentiles]
    
    # Calculate flood frequency for each threshold
    frequency_stats = []
    
    for threshold in thresholds:
        # Identify floods above this threshold
        discharge_data['above_threshold'] = discharge_data['Discharge'] >= threshold
        
        # Count flood events (sequences of days above threshold)
        flood_count = 0
        in_flood = False
        
        for is_above in discharge_data['above_threshold']:
            if is_above and not in_flood:
                # Start of a new flood
                flood_count += 1
                in_flood = True
            elif not is_above:
                # End of flood
                in_flood = False
        
        # Calculate days above threshold
        days_above = discharge_data['above_threshold'].sum()
        
        # Calculate percent of time above threshold
        total_days = len(discharge_data)
        percent_time = (days_above / total_days) * 100
        
        # Add to statistics
        frequency_stats.append({
            'Threshold': threshold,
            'Flood_Count': flood_count,
            'Days_Above_Threshold': days_above,
            'Percent_Time_Above_Threshold': percent_time
        })
    
    return pd.DataFrame(frequency_stats)