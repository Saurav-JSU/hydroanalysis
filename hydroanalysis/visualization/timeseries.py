"""
Functions for creating time series visualizations.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import logging

from hydroanalysis.config import CONFIG

logger = logging.getLogger(__name__)

def plot_discharge(discharge_df, flood_event=None, title=None, output_path=None):
    """
    Plot discharge time series with optional flood event highlighting.
    
    Parameters
    ----------
    discharge_df : pandas.DataFrame
        DataFrame with 'Date' and 'Discharge' columns
    flood_event : dict, optional
        Flood event details from identify_flood_events
    title : str, optional
        Plot title
    output_path : str, optional
        Path to save the plot
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    # Set figure style
    plt.style.use(CONFIG['visualization']['style'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=CONFIG['visualization']['figsize'])
    
    # Ensure Date is datetime
    if 'Date' not in discharge_df.columns:
        logger.error("Discharge DataFrame must have a 'Date' column")
        return fig
    
    discharge_df = discharge_df.copy()
    discharge_df['Date'] = pd.to_datetime(discharge_df['Date'])
    
    # Plot discharge data
    ax.plot(discharge_df['Date'], discharge_df['Discharge'], 'b-', linewidth=1.5, label='Discharge')
    
    # If flood event details are provided, highlight the period
    if flood_event:
        # Highlight flood period
        if 'start_date' in flood_event and 'end_date' in flood_event:
            start_date = pd.to_datetime(flood_event['start_date'])
            end_date = pd.to_datetime(flood_event['end_date'])
            
            # Find data points within flood period
            flood_data = discharge_df[
                (discharge_df['Date'] >= start_date) & 
                (discharge_df['Date'] <= end_date)
            ]
            
            if not flood_data.empty:
                ax.plot(flood_data['Date'], flood_data['Discharge'], 'r-', linewidth=2.5, label='Flood Period')
        
        # Mark peak if available
        if 'peak_date' in flood_event and 'peak_discharge' in flood_event:
            peak_date = pd.to_datetime(flood_event['peak_date'])
            peak_discharge = flood_event['peak_discharge']
            
            ax.plot(peak_date, peak_discharge, 'ro', markersize=8, label='Peak')
            
            # Add peak value text
            ax.annotate(f"Peak: {peak_discharge:.1f} m³/s",
                        xy=(peak_date, peak_discharge),
                        xytext=(10, 10),
                        textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
        
        # Show threshold if available
        if 'threshold' in flood_event:
            threshold = flood_event['threshold']
            ax.axhline(y=threshold, color='g', linestyle='--', 
                      label=f'Threshold: {threshold:.1f} m³/s')
    
    # Set labels and title
    ax.set_xlabel('Date')
    ax.set_ylabel('Discharge (m³/s)')
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Discharge Time Series')
    
    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Adjust layout
    fig.tight_layout()
    
    # Save if output path is provided
    if output_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=CONFIG['visualization']['dpi'])
        logger.info(f"Discharge plot saved to {output_path}")
    
    return fig

def plot_precipitation(precip_df, start_date=None, end_date=None, stations=None, title=None, output_path=None):
    """
    Plot precipitation time series for one or more stations.
    
    Parameters
    ----------
    precip_df : pandas.DataFrame
        DataFrame with 'Date' or 'datetime' and precipitation columns
    start_date : datetime.datetime, optional
        Start date for plot
    end_date : datetime.datetime, optional
        End date for plot
    stations : list, optional
        List of station columns to plot (if None, plot all stations)
    title : str, optional
        Plot title
    output_path : str, optional
        Path to save the plot
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    # Set figure style
    plt.style.use(CONFIG['visualization']['style'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=CONFIG['visualization']['figsize'])
    
    # Check for date column
    date_col = None
    if 'Date' in precip_df.columns:
        date_col = 'Date'
    elif 'datetime' in precip_df.columns:
        date_col = 'datetime'
    else:
        logger.error("Precipitation DataFrame must have a 'Date' or 'datetime' column")
        return fig
    
    # Ensure date is datetime
    precip_df = precip_df.copy()
    precip_df[date_col] = pd.to_datetime(precip_df[date_col])
    
    # Filter by date range if provided
    if start_date or end_date:
        if start_date:
            precip_df = precip_df[precip_df[date_col] >= pd.to_datetime(start_date)]
        if end_date:
            precip_df = precip_df[precip_df[date_col] <= pd.to_datetime(end_date)]
    
    # Determine stations to plot
    if stations is None:
        # Find all precipitation columns (exclude date columns)
        stations = [col for col in precip_df.columns 
                   if col not in [date_col, 'time', 'index']]
    
    # Limit to a reasonable number of stations for clarity
    max_stations = 8
    if len(stations) > max_stations:
        logger.warning(f"Too many stations to plot clearly. Showing only the first {max_stations}.")
        stations = stations[:max_stations]
    
    # Plot each station
    for station in stations:
        if station in precip_df.columns:
            # Plot as bars
            ax.bar(precip_df[date_col], precip_df[station], alpha=0.7, 
                  width=pd.Timedelta(hours=12) if len(precip_df) < 100 else pd.Timedelta(hours=2),
                  label=station.replace('Station_', 'Station '))
    
    # Set labels and title
    ax.set_xlabel('Date')
    ax.set_ylabel('Precipitation (mm)')
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title('Precipitation Time Series')
    
    # Add grid and legend
    ax.grid(True, alpha=0.3)
    
    if len(stations) <= 10:  # Only show legend if not too many stations
        ax.legend()
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Adjust layout
    fig.tight_layout()
    
    # Save if output path is provided
    if output_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=CONFIG['visualization']['dpi'])
        logger.info(f"Precipitation plot saved to {output_path}")
    
    return fig

def plot_flood_analysis(flood_event, discharge_df, precip_df, output_path=None):
    """
    Create a comprehensive flood event analysis plot with discharge and precipitation.
    
    Parameters
    ----------
    flood_event : dict
        Flood event details from identify_flood_events
    discharge_df : pandas.DataFrame
        DataFrame with discharge data
    precip_df : pandas.DataFrame
        DataFrame with precipitation data
    output_path : str, optional
        Path to save the plot
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    # Set figure style
    plt.style.use(CONFIG['visualization']['style'])
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Ensure dates are datetime
    discharge_df = discharge_df.copy()
    discharge_df['Date'] = pd.to_datetime(discharge_df['Date'])
    
    # Check for date column in precip_df
    date_col = None
    if 'Date' in precip_df.columns:
        date_col = 'Date'
    elif 'datetime' in precip_df.columns:
        date_col = 'datetime'
    else:
        logger.error("Precipitation DataFrame must have a 'Date' or 'datetime' column")
        return fig
    
    precip_df = precip_df.copy()
    precip_df[date_col] = pd.to_datetime(precip_df[date_col])
    
    # Plot discharge
    ax1.plot(discharge_df['Date'], discharge_df['Discharge'], 'b-', linewidth=1.5)
    
    # Highlight flood period
    if 'start_date' in flood_event and 'end_date' in flood_event:
        start_date = pd.to_datetime(flood_event['start_date'])
        end_date = pd.to_datetime(flood_event['end_date'])
        
        # Find data points within flood period
        flood_data = discharge_df[
            (discharge_df['Date'] >= start_date) & 
            (discharge_df['Date'] <= end_date)
        ]
        
        if not flood_data.empty:
            ax1.plot(flood_data['Date'], flood_data['Discharge'], 'r-', linewidth=2.5)
            
            # Shade the flood area
            min_y = ax1.get_ylim()[0]
            for i in range(len(flood_data) - 1):
                ax1.fill_between(
                    [flood_data['Date'].iloc[i], flood_data['Date'].iloc[i+1]],
                    [min_y, min_y],
                    [flood_data['Discharge'].iloc[i], flood_data['Discharge'].iloc[i+1]],
                    color='red', alpha=0.2
                )
    
    # Mark peak if available
    if 'peak_date' in flood_event and 'peak_discharge' in flood_event:
        peak_date = pd.to_datetime(flood_event['peak_date'])
        peak_discharge = flood_event['peak_discharge']
        
        ax1.plot(peak_date, peak_discharge, 'ro', markersize=8)
        
        # Add peak value text
        ax1.annotate(f"Peak: {peak_discharge:.1f} m³/s",
                    xy=(peak_date, peak_discharge),
                    xytext=(10, 10),
                    textcoords='offset points',
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
    
    # Show threshold if available
    if 'threshold' in flood_event:
        threshold = flood_event['threshold']
        ax1.axhline(y=threshold, color='g', linestyle='--', 
                   label=f'Threshold: {threshold:.1f} m³/s')
    
    # Set labels for discharge plot
    ax1.set_ylabel('Discharge (m³/s)')
    ax1.set_title(f"Flood Analysis: {flood_event.get('start_date', 'Unknown')} to {flood_event.get('end_date', 'Unknown')}")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot precipitation for each station
    precip_columns = [col for col in precip_df.columns 
                     if col.startswith('Station_')]
    
    # Limit to a reasonable number of stations
    max_stations = 5
    if len(precip_columns) > max_stations:
        # Use the stations with the most precipitation during flood
        station_totals = {}
        for station in precip_columns:
            station_data = precip_df[
                (precip_df[date_col] >= pd.to_datetime(flood_event.get('buffer_start_date', start_date))) & 
                (precip_df[date_col] <= pd.to_datetime(flood_event.get('buffer_end_date', end_date)))
            ]
            station_totals[station] = station_data[station].sum()
        
        # Sort by total precipitation
        sorted_stations = sorted(station_totals.items(), key=lambda x: x[1], reverse=True)
        precip_columns = [station for station, _ in sorted_stations[:max_stations]]
    
    # Plot each station
    for station in precip_columns:
        ax2.bar(precip_df[date_col], precip_df[station], alpha=0.7, 
              width=pd.Timedelta(hours=12) if len(precip_df) < 100 else pd.Timedelta(hours=2),
              label=station.replace('Station_', 'Station '))
    
    # Set labels for precipitation plot
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Precipitation (mm)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Adjust layout
    fig.tight_layout()
    
    # Save if output path is provided
    if output_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=CONFIG['visualization']['dpi'])
        logger.info(f"Flood analysis plot saved to {output_path}")
    
    return fig

def plot_high_resolution_precip(high_res_df, start_date=None, end_date=None, stations=None, title=None, output_path=None, comparison_df=None):
    """
    Plot high-resolution precipitation time series with optional comparison to original data.
    
    Parameters
    ----------
    high_res_df : pandas.DataFrame
        DataFrame with high-resolution precipitation data
    start_date : datetime.datetime, optional
        Start date for plot
    end_date : datetime.datetime, optional
        End date for plot
    stations : list, optional
        List of station columns to plot (if None, plot all stations up to a maximum)
    title : str, optional
        Plot title
    output_path : str, optional
        Path to save the plot
    comparison_df : pandas.DataFrame, optional
        DataFrame with original precipitation data for comparison
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    # Set figure style
    plt.style.use(CONFIG['visualization']['style'])
    
    # Determine if we need comparison plots
    has_comparison = comparison_df is not None
    
    # Create figure
    if has_comparison:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    else:
        fig, ax1 = plt.subplots(figsize=CONFIG['visualization']['figsize'])
    
    # Check for date column in high_res_df
    date_col = None
    if 'datetime' in high_res_df.columns:
        date_col = 'datetime'
    elif 'Date' in high_res_df.columns:
        date_col = 'Date'
    else:
        logger.error("High-resolution DataFrame must have a 'datetime' or 'Date' column")
        return fig
    
    # Ensure date is datetime
    high_res_df = high_res_df.copy()
    high_res_df[date_col] = pd.to_datetime(high_res_df[date_col])
    
    # Filter by date range if provided
    if start_date or end_date:
        if start_date:
            high_res_df = high_res_df[high_res_df[date_col] >= pd.to_datetime(start_date)]
        if end_date:
            high_res_df = high_res_df[high_res_df[date_col] <= pd.to_datetime(end_date)]
    
    # Determine stations to plot
    if stations is None:
        # Find all precipitation columns (exclude date columns)
        stations = [col for col in high_res_df.columns 
                   if col not in [date_col, 'time', 'index']]
        
        # Limit to a reasonable number of stations
        max_stations = 5
        if len(stations) > max_stations:
            logger.warning(f"Too many stations to plot clearly. Showing only the first {max_stations}.")
            stations = stations[:max_stations]
    
    # Plot high-resolution data
    for station in stations:
        if station in high_res_df.columns:
            # Plot as bars
            width = pd.Timedelta(minutes=30) if len(high_res_df) < 200 else pd.Timedelta(minutes=15)
            ax1.bar(high_res_df[date_col], high_res_df[station], alpha=0.7, 
                  width=width, label=station.replace('Station_', 'Station '))
    
    # Set labels and title
    ax1.set_ylabel('Precipitation (mm)')
    
    if title:
        ax1.set_title(title)
    else:
        ax1.set_title('High-Resolution Precipitation Time Series')
    
    # Add grid and legend
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot comparison if provided
    if has_comparison:
        # Check for date column in comparison_df
        comp_date_col = None
        if 'Date' in comparison_df.columns:
            comp_date_col = 'Date'
        elif 'datetime' in comparison_df.columns:
            comp_date_col = 'datetime'
        else:
            logger.error("Comparison DataFrame must have a 'Date' or 'datetime' column")
            return fig
        
        # Ensure date is datetime
        comparison_df = comparison_df.copy()
        comparison_df[comp_date_col] = pd.to_datetime(comparison_df[comp_date_col])
        
        # Filter by date range if provided
        if start_date or end_date:
            if start_date:
                comparison_df = comparison_df[comparison_df[comp_date_col] >= pd.to_datetime(start_date)]
            if end_date:
                comparison_df = comparison_df[comparison_df[comp_date_col] <= pd.to_datetime(end_date)]
        
        # Plot each station in comparison data
        for station in stations:
            if station in comparison_df.columns:
                # Plot as lines
                ax2.plot(comparison_df[comp_date_col], comparison_df[station], 'o-', 
                        label=f"{station.replace('Station_', 'Station ')} (Original)")
        
        # Set labels
        ax2.set_xlabel('Date/Time')
        ax2.set_ylabel('Precipitation (mm)')
        ax2.set_title('Original Precipitation Data for Comparison')
        
        # Add grid and legend
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    else:
        # Set x-label on the main plot if no comparison
        ax1.set_xlabel('Date/Time')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Adjust layout
    fig.tight_layout()
    
    # Save if output path is provided
    if output_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=CONFIG['visualization']['dpi'])
        logger.info(f"High-resolution precipitation plot saved to {output_path}")
    
    return fig

def plot_daily_statistics(precip_df, station, output_path=None):
    """
    Plot daily precipitation statistics for a station.
    
    Parameters
    ----------
    precip_df : pandas.DataFrame
        DataFrame with precipitation data
    station : str
        Station ID to plot
    output_path : str, optional
        Path to save the plot
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    # Set figure style
    plt.style.use(CONFIG['visualization']['style'])
    
    # Check for date column
    date_col = None
    if 'Date' in precip_df.columns:
        date_col = 'Date'
    elif 'datetime' in precip_df.columns:
        date_col = 'datetime'
    else:
        logger.error("Precipitation DataFrame must have a 'Date' or 'datetime' column")
        return plt.figure()
    
    # Check if station exists
    if station not in precip_df.columns:
        logger.error(f"Station {station} not found in DataFrame")
        return plt.figure()
    
    # Ensure date is datetime
    precip_df = precip_df.copy()
    precip_df[date_col] = pd.to_datetime(precip_df[date_col])
    
    # Add month and year columns
    precip_df['Month'] = precip_df[date_col].dt.month
    precip_df['Year'] = precip_df[date_col].dt.year
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Monthly average precipitation
    monthly_avg = precip_df.groupby('Month')[station].mean().reindex(range(1, 13))
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    ax1.bar(month_names, monthly_avg, color='steelblue')
    ax1.set_title(f'Monthly Average Precipitation - {station}')
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Average Precipitation (mm/day)')
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Annual precipitation
    annual_sum = precip_df.groupby('Year')[station].sum()
    ax2.bar(annual_sum.index, annual_sum, color='darkgreen')
    ax2.set_title(f'Annual Precipitation - {station}')
    ax2.set_xlabel('Year')
    ax2.set_ylabel('Total Precipitation (mm)')
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Precipitation frequency by month
    # Define wet day as >= 1mm
    precip_df['Wet_Day'] = precip_df[station] >= 1
    monthly_freq = precip_df.groupby('Month')['Wet_Day'].mean() * 100
    
    ax3.bar(month_names, monthly_freq.reindex(range(1, 13)), color='indianred')
    ax3.set_title(f'Wet Day Frequency (≥ 1mm) by Month - {station}')
    ax3.set_xlabel('Month')
    ax3.set_ylabel('Wet Day Frequency (%)')
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Precipitation intensity on wet days
    wet_days = precip_df[precip_df[station] >= 1]
    if not wet_days.empty:
        monthly_intensity = wet_days.groupby('Month')[station].mean()
        
        ax4.bar(month_names, monthly_intensity.reindex(range(1, 13)), color='darkviolet')
        ax4.set_title(f'Wet Day Intensity by Month - {station}')
        ax4.set_xlabel('Month')
        ax4.set_ylabel('Average Intensity (mm/wet day)')
        ax4.grid(axis='y', alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No wet days in the dataset', 
                ha='center', va='center', transform=ax4.transAxes)
    
    # Add overall title
    plt.suptitle(f'Precipitation Statistics for {station}', fontsize=16, y=0.99)
    
    # Adjust layout
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Save if output path is provided
    if output_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=CONFIG['visualization']['dpi'])
        logger.info(f"Precipitation statistics plot saved to {output_path}")
    
    return fig

def plot_flow_duration_curve(discharge_df, percentile_steps=100, log_scale=True, output_path=None):
    """
    Create a flow duration curve from discharge data.
    
    Parameters
    ----------
    discharge_df : pandas.DataFrame
        DataFrame with 'Discharge' column
    percentile_steps : int, optional
        Number of percentile steps to calculate
    log_scale : bool, optional
        Whether to use log scale for y-axis
    output_path : str, optional
        Path to save the plot
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    # Set figure style
    plt.style.use(CONFIG['visualization']['style'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=CONFIG['visualization']['figsize'])
    
    # Check for discharge column
    if 'Discharge' not in discharge_df.columns:
        logger.error("Discharge DataFrame must have a 'Discharge' column")
        return fig
    
    # Calculate flow duration curve
    percentiles = np.linspace(0, 100, percentile_steps)
    flow_values = np.percentile(discharge_df['Discharge'].dropna(), 100 - percentiles)
    
    # Plot the flow duration curve
    ax.plot(percentiles, flow_values, 'b-', linewidth=2)
    
    # Set labels and title
    ax.set_xlabel('Exceedance Probability (%)')
    ax.set_ylabel('Discharge (m³/s)')
    ax.set_title('Flow Duration Curve')
    
    # Set log scale if requested
    if log_scale and np.min(flow_values) > 0:
        ax.set_yscale('log')
    
    # Add grid
    ax.grid(True, which='both', alpha=0.3)
    
    # Adjust layout
    fig.tight_layout()
    
    # Save if output path is provided
    if output_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=CONFIG['visualization']['dpi'])
        logger.info(f"Flow duration curve saved to {output_path}")
    
    return fig