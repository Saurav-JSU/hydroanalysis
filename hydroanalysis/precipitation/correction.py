"""
Functions for correcting precipitation datasets using scaling factors.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import logging
import os
import json

from hydroanalysis.config import CONFIG

logger = logging.getLogger(__name__)

def calculate_scaling_factors(best_datasets, comparison_data, event_based=True):
    """
    Calculate scaling factor for each station based on regression through origin.
    
    Parameters
    ----------
    best_datasets : dict
        Dictionary with best dataset info for each station
    comparison_data : dict
        Dictionary with comparison DataFrames for each dataset
    event_based : bool, optional
        Whether to calculate event-specific scaling factors (default: True)
        
    Returns
    -------
    dict
        Dictionary with scaling factors for each station
    """
    scaling_factors = {}
    
    # Get configuration values
    min_scaling_factor = CONFIG['correction']['min_scaling_factor']
    max_scaling_factor = CONFIG['correction']['max_scaling_factor']
    default_factor = CONFIG['correction']['default_factor']
    
    for station, info in best_datasets.items():
        dataset_name = info["dataset"]
        
        # Get comparison data for this dataset
        if dataset_name not in comparison_data:
            logger.warning(f"No comparison data found for dataset {dataset_name}. Skipping station {station}.")
            continue
            
        comp_df = comparison_data[dataset_name]
        
        obs_col = f"{station}_obs"
        pred_col = f"{station}_pred"
        
        if obs_col in comp_df.columns and pred_col in comp_df.columns:
            # Get valid data pairs
            valid_data = comp_df[[obs_col, pred_col]].dropna()
            
            if len(valid_data) >= 5:  # Ensure enough data points (reduced from 10 for event-based)
                # Use regression through origin (forces intercept=0)
                # This ensures zero prediction = zero observation
                model = LinearRegression(fit_intercept=False)
                X = valid_data[[pred_col]]
                y = valid_data[obs_col]
                model.fit(X, y)
                scaling_factor = model.coef_[0]
                
                # Ensure the scaling factor is reasonable
                if scaling_factor <= 0:
                    logger.warning(f"Invalid scaling factor ({scaling_factor}) for {station}. Setting to {default_factor}.")
                    scaling_factor = default_factor
                elif scaling_factor < min_scaling_factor:
                    logger.warning(f"Very small scaling factor ({scaling_factor}) for {station}. Setting to {min_scaling_factor}.")
                    scaling_factor = min_scaling_factor
                elif scaling_factor > max_scaling_factor:
                    logger.warning(f"Extreme scaling factor ({scaling_factor}) for {station}. Capping at {max_scaling_factor}.")
                    scaling_factor = max_scaling_factor
                
                # Calculate additional statistics
                rmse_orig = np.sqrt(np.mean((valid_data[obs_col] - valid_data[pred_col])**2))
                scaled_pred = valid_data[pred_col] * scaling_factor
                rmse_scaled = np.sqrt(np.mean((valid_data[obs_col] - scaled_pred)**2))
                
                scaling_factors[station] = {
                    "factor": scaling_factor,
                    "dataset": dataset_name,
                    "validation_data": valid_data,
                    "original_rmse": rmse_orig,
                    "scaled_rmse": rmse_scaled,
                    "improvement_percent": ((rmse_orig - rmse_scaled) / rmse_orig * 100) if rmse_orig > 0 else 0,
                    "sample_size": len(valid_data)
                }
                
                logger.info(f"Station {station}: Scaling factor = {scaling_factor:.3f} " +
                          f"(RMSE improvement: {scaling_factors[station]['improvement_percent']:.1f}%, n={len(valid_data)})")
            else:
                logger.warning(f"Not enough valid data pairs for station {station} (found {len(valid_data)}).")
                # For event-based analysis, use default factor if not enough data
                if event_based and len(valid_data) > 0:
                    # Use mean ratio as a simple scaling factor
                    simple_factor = valid_data[obs_col].sum() / valid_data[pred_col].sum() if valid_data[pred_col].sum() > 0 else default_factor
                    
                    # Apply constraints
                    if simple_factor <= 0:
                        simple_factor = default_factor
                    elif simple_factor < min_scaling_factor:
                        simple_factor = min_scaling_factor
                    elif simple_factor > max_scaling_factor:
                        simple_factor = max_scaling_factor
                    
                    scaling_factors[station] = {
                        "factor": simple_factor,
                        "dataset": dataset_name,
                        "validation_data": valid_data,
                        "original_rmse": np.nan,
                        "scaled_rmse": np.nan,
                        "improvement_percent": 0,
                        "sample_size": len(valid_data),
                        "factor_method": "ratio"
                    }
                    logger.warning(f"Using simple ratio method for station {station} due to small sample size. Factor = {simple_factor:.3f}")
        else:
            logger.warning(f"Required columns {obs_col} and {pred_col} not found for station {station}")
    
    return scaling_factors

def apply_scaling_factors(precipitation_data, scaling_factors):
    """
    Apply scaling factors to precipitation data.
    
    Parameters
    ----------
    precipitation_data : pandas.DataFrame
        DataFrame with precipitation data
    scaling_factors : dict
        Dictionary with scaling factors for each station
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with scaled precipitation data
    """
    # Create a copy to avoid modifying the original
    scaled_data = precipitation_data.copy()
    
    # Apply scaling factors to each station
    for station, info in scaling_factors.items():
        if station in scaled_data.columns:
            factor = info["factor"]
            scaled_data[station] = scaled_data[station] * factor
            
            # Calculate statistics before and after scaling
            orig_sum = precipitation_data[station].sum()
            scaled_sum = scaled_data[station].sum()
            
            logger.info(f"Applied scaling factor {factor:.3f} to {station}")
            logger.info(f"  Original total: {orig_sum:.2f} mm, Scaled total: {scaled_sum:.2f} mm")
    
    return scaled_data

def extract_high_resolution_data(flood_dir, dataset_name, station):
    """
    Extract high-resolution precipitation data for a specific flood event.
    
    Parameters
    ----------
    flood_dir : str
        Directory containing flood event data
    dataset_name : str
        Name of the dataset to extract
    station : str
        Station ID to extract
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with high-resolution precipitation data
    """
    # Path to high-resolution data file
    hourly_file = os.path.join(flood_dir, f"gee_{dataset_name.lower()}", "hourly_precipitation.csv")
    
    if not os.path.exists(hourly_file):
        logger.warning(f"High-resolution file not found: {hourly_file}")
        return None
    
    try:
        # Read high-resolution data
        high_res_df = pd.read_csv(hourly_file)
        
        # Ensure datetime column
        if 'datetime' in high_res_df.columns:
            high_res_df['datetime'] = pd.to_datetime(high_res_df['datetime'])
        elif 'Date' in high_res_df.columns:
            high_res_df['datetime'] = pd.to_datetime(high_res_df['Date'])
            
        # Check if station exists
        if station in high_res_df.columns:
            # Return only necessary columns
            return high_res_df[['datetime', station]]
        else:
            logger.warning(f"Station {station} not found in high-resolution data")
            return None
    except Exception as e:
        logger.error(f"Error reading high-resolution data: {e}")
        return None

def create_corrected_dataset(best_datasets, scaling_factors, flood_dir, output_dir=None, high_res=True):
    """
    Create a corrected dataset using the best dataset for each station.
    
    Parameters
    ----------
    best_datasets : dict
        Dictionary with best dataset info for each station
    scaling_factors : dict
        Dictionary with scaling factors for each station
    flood_dir : str
        Directory containing flood event data
    output_dir : str, optional
        Directory to save outputs (defaults to flood_dir/corrected)
    high_res : bool, optional
        Whether to create corrected high-resolution data (default: True)
        
    Returns
    -------
    dict
        Dictionary with corrected datasets (daily and high-resolution)
    """
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = os.path.join(flood_dir, "corrected")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize result dictionaries
    corrected_data = {'daily': None, 'high_res': {}}
    
    # Process each station
    for station, info in best_datasets.items():
        dataset_name = info["dataset"]
        
        # Get scaling factor if available
        scaling_factor = 1.0
        if station in scaling_factors:
            scaling_factor = scaling_factors[station]["factor"]
            
        # 1. Create corrected daily data
        
        # Path to daily comparison data
        comparison_file = os.path.join(flood_dir, "comparison", dataset_name, "comparison_data.csv")
        
        if os.path.exists(comparison_file):
            try:
                # Read comparison data
                comparison_df = pd.read_csv(comparison_file)
                comparison_df['Date'] = pd.to_datetime(comparison_df['Date'])
                
                # Get predicted column
                pred_col = f"{station}_pred"
                
                if pred_col in comparison_df.columns:
                    # Create daily data with Date and station
                    daily_data = comparison_df[['Date', pred_col]].copy()
                    
                    # Rename column to station name
                    daily_data = daily_data.rename(columns={pred_col: station})
                    
                    # Apply scaling factor
                    daily_data[station] = daily_data[station] * scaling_factor
                    
                    # Merge with existing corrected data
                    if corrected_data['daily'] is None:
                        corrected_data['daily'] = daily_data
                    else:
                        corrected_data['daily'] = pd.merge(
                            corrected_data['daily'], 
                            daily_data, 
                            on='Date', 
                            how='outer'
                        )
                    
                    logger.info(f"Created corrected daily data for {station} using {dataset_name}")
                else:
                    logger.warning(f"Predicted column {pred_col} not found in comparison data")
            except Exception as e:
                logger.error(f"Error creating corrected daily data: {e}")
        else:
            logger.warning(f"Comparison file not found: {comparison_file}")
        
        # 2. Create corrected high-resolution data (if requested)
        if high_res:
            # Extract high-resolution data
            high_res_df = extract_high_resolution_data(flood_dir, dataset_name, station)
            
            if high_res_df is not None:
                # Apply scaling factor
                high_res_df[station] = high_res_df[station] * scaling_factor
                
                # Store in result dictionary
                corrected_data['high_res'][station] = {
                    'data': high_res_df,
                    'dataset': dataset_name,
                    'factor': scaling_factor
                }
                
                logger.info(f"Created corrected high-resolution data for {station} using {dataset_name}")
            else:
                logger.warning(f"Could not create high-resolution data for {station}")
    
    # Save corrected daily data
    if corrected_data['daily'] is not None:
        daily_file = os.path.join(output_dir, "corrected_daily_precipitation.csv")
        corrected_data['daily'].to_csv(daily_file, index=False)
        logger.info(f"Saved corrected daily precipitation to {daily_file}")
    
    # Save corrected high-resolution data
    if high_res and corrected_data['high_res']:
        # Create a merged high-resolution dataset with all stations
        merged_high_res = None
        
        for station, hr_info in corrected_data['high_res'].items():
            if merged_high_res is None:
                merged_high_res = hr_info['data']
            else:
                merged_high_res = pd.merge(
                    merged_high_res,
                    hr_info['data'],
                    on='datetime',
                    how='outer'
                )
        
        if merged_high_res is not None:
            # Sort by datetime
            merged_high_res = merged_high_res.sort_values('datetime')
            
            # Save to file
            high_res_file = os.path.join(output_dir, "corrected_highres_precipitation.csv")
            merged_high_res.to_csv(high_res_file, index=False)
            logger.info(f"Saved corrected high-resolution precipitation to {high_res_file}")
    
    # Save a summary of the corrections
    summary_data = []
    
    for station in best_datasets.keys():
        if station in scaling_factors:
            summary_data.append({
                'Station': station,
                'Best_Dataset': best_datasets[station]['dataset'],
                'Scaling_Factor': scaling_factors[station]['factor'],
                'Original_RMSE': scaling_factors[station].get('original_rmse', None),
                'Scaled_RMSE': scaling_factors[station].get('scaled_rmse', None),
                'Improvement_Percent': scaling_factors[station].get('improvement_percent', None),
                'Sample_Size': scaling_factors[station].get('sample_size', None)
            })
    
    if summary_data:
        # Save as CSV
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(os.path.join(output_dir, "correction_summary.csv"), index=False)
        
        # Save as JSON for easier loading by other tools
        with open(os.path.join(output_dir, "correction_factors.json"), 'w') as f:
            json.dump({
                'best_datasets': {
                    row['Station']: {
                        'dataset': row['Best_Dataset'],
                        'factor': row['Scaling_Factor']
                    } for _, row in summary_df.iterrows()
                }
            }, f, indent=2)
        
        logger.info(f"Saved correction summary to {output_dir}")
    
    return corrected_data

def run_correct_precipitation_command(args, flood_dir=None):
    """
    Execute the correct-precipitation command for a specific flood event.
    
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
        output_dir = os.path.join(datasets_dir, "corrected")
    
    # Load comparison results
    comparison_dir = os.path.join(datasets_dir, "comparison")
    if not os.path.isdir(comparison_dir):
        logger.error(f"Comparison directory not found: {comparison_dir}")
        return 1
    
    # Load metrics for each dataset
    all_metrics = {}
    comparison_dfs = {}
    
    dataset_dirs = [d for d in os.listdir(comparison_dir) 
                   if os.path.isdir(os.path.join(comparison_dir, d))]
    
    for dataset_name in dataset_dirs:
        dataset_path = os.path.join(comparison_dir, dataset_name)
        metrics_file = os.path.join(dataset_path, "accuracy_metrics.csv")
        comparison_file = os.path.join(dataset_path, "comparison_data.csv")
        
        if os.path.exists(metrics_file) and os.path.exists(comparison_file):
            # Read metrics
            metrics_df = pd.read_csv(metrics_file)
            metrics = {row['Station']: {
                'RMSE': row['RMSE'],
                'MAE': row['MAE'],
                'Correlation': row['Correlation'],
                'Bias': row['Bias'],
                'Percent_Bias': row['Percent_Bias'],
                'Count': row['Count']
            } for _, row in metrics_df.iterrows()}
            
            all_metrics[dataset_name] = metrics
            
            # Read comparison data
            comparison_dfs[dataset_name] = pd.read_csv(comparison_file)
            comparison_dfs[dataset_name]['Date'] = pd.to_datetime(comparison_dfs[dataset_name]['Date'])
    
    if not all_metrics:
        logger.error("No valid metrics found in comparison directory")
        return 1
    
    # Read station metadata if provided
    station_metadata = None
    if args.metadata:
        from hydroanalysis.core.data_io import read_station_metadata
        station_metadata = read_station_metadata(args.metadata)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Identify best dataset for each station
    from hydroanalysis.precipitation.comparison import identify_best_dataset_per_station
    best_datasets = identify_best_dataset_per_station(all_metrics)
    
    # Calculate scaling factors
    scaling_factors = calculate_scaling_factors(best_datasets, comparison_dfs, event_based=True)
    
    # Create corrected dataset
    corrected_data = create_corrected_dataset(
        best_datasets,
        scaling_factors,
        datasets_dir,
        output_dir=output_dir,
        high_res=True
    )
    
    # If metadata is available, create maps
    if station_metadata is not None:
        from hydroanalysis.visualization.maps import (
            plot_station_map, 
            plot_best_datasets_map,
            plot_scaling_factors_map
        )
        
        # Create station map
        station_map = plot_station_map(
            station_metadata, 
            output_path=os.path.join(output_dir, "station_locations.png")
        )
        
        # Create best datasets map
        best_datasets_map = plot_best_datasets_map(
            best_datasets,
            station_metadata,
            output_path=os.path.join(output_dir, "best_datasets_map.png")
        )
        
        # Create scaling factors map
        scaling_map = plot_scaling_factors_map(
            scaling_factors,
            station_metadata,
            output_path=os.path.join(output_dir, "scaling_factors_map.png")
        )
    
    logger.info(f"Precipitation correction completed. Results saved to {output_dir}")
    return 0