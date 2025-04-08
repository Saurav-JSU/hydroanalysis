"""
Functions for correcting precipitation datasets using scaling factors.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import logging
import os

from hydroanalysis.config import CONFIG

logger = logging.getLogger(__name__)

def calculate_scaling_factors(best_datasets, comparison_data):
    """
    Calculate scaling factor for each station based on regression through origin.
    
    Parameters
    ----------
    best_datasets : dict
        Dictionary with best dataset info for each station
    comparison_data : dict
        Dictionary with comparison DataFrames for each dataset
        
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
            
            if len(valid_data) >= 10:  # Ensure enough data points
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
                    "improvement_percent": ((rmse_orig - rmse_scaled) / rmse_orig * 100) if rmse_orig > 0 else 0
                }
                
                logger.info(f"Station {station}: Scaling factor = {scaling_factor:.3f} " +
                          f"(RMSE improvement: {scaling_factors[station]['improvement_percent']:.1f}%)")
            else:
                logger.warning(f"Not enough valid data pairs for station {station} (found {len(valid_data)})")
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
            logger.info(f"Applied scaling factor {factor:.3f} to {station}")
    
    return scaled_data

def calculate_monthly_scaling_factors(best_datasets, comparison_data):
    """
    Calculate monthly scaling factors to account for seasonal biases.
    
    Parameters
    ----------
    best_datasets : dict
        Dictionary with best dataset info for each station
    comparison_data : dict
        Dictionary with comparison DataFrames for each dataset
        
    Returns
    -------
    dict
        Dictionary with monthly scaling factors for each station
    """
    monthly_scaling_factors = {}
    
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
        
        # Ensure we have Date column
        if 'Date' not in comp_df.columns:
            logger.warning(f"No Date column found in comparison data for {dataset_name}. Skipping station {station}.")
            continue
        
        obs_col = f"{station}_obs"
        pred_col = f"{station}_pred"
        
        if obs_col in comp_df.columns and pred_col in comp_df.columns:
            # Add month column
            comp_df['Month'] = pd.to_datetime(comp_df['Date']).dt.month
            
            # Calculate scaling factor for each month
            station_monthly_factors = {}
            
            for month in range(1, 13):
                month_data = comp_df[comp_df['Month'] == month][[obs_col, pred_col]].dropna()
                
                if len(month_data) >= 5:  # Need at least 5 data points for reliable factor
                    # Use regression through origin
                    model = LinearRegression(fit_intercept=False)
                    X = month_data[[pred_col]]
                    y = month_data[obs_col]
                    model.fit(X, y)
                    scaling_factor = model.coef_[0]
                    
                    # Ensure the scaling factor is reasonable
                    if scaling_factor <= 0:
                        scaling_factor = default_factor
                    elif scaling_factor < min_scaling_factor:
                        scaling_factor = min_scaling_factor
                    elif scaling_factor > max_scaling_factor:
                        scaling_factor = max_scaling_factor
                    
                    station_monthly_factors[month] = {
                        "factor": scaling_factor,
                        "count": len(month_data)
                    }
                else:
                    logger.debug(f"Not enough data for {station} in month {month} (found {len(month_data)})")
            
            # If we have factors for at least 6 months, include this station
            if len(station_monthly_factors) >= 6:
                # For missing months, use interpolation or the annual factor
                annual_factor = info.get("factor", default_factor)
                
                for month in range(1, 13):
                    if month not in station_monthly_factors:
                        # Try to interpolate from adjacent months
                        prev_month = month - 1 if month > 1 else 12
                        next_month = month + 1 if month < 12 else 1
                        
                        if prev_month in station_monthly_factors and next_month in station_monthly_factors:
                            prev_factor = station_monthly_factors[prev_month]["factor"]
                            next_factor = station_monthly_factors[next_month]["factor"]
                            interpolated_factor = (prev_factor + next_factor) / 2
                            station_monthly_factors[month] = {
                                "factor": interpolated_factor,
                                "count": 0,
                                "interpolated": True
                            }
                        else:
                            # Use annual factor as fallback
                            station_monthly_factors[month] = {
                                "factor": annual_factor,
                                "count": 0,
                                "interpolated": True
                            }
                
                monthly_scaling_factors[station] = {
                    "dataset": dataset_name,
                    "monthly_factors": station_monthly_factors
                }
                
                logger.info(f"Calculated monthly scaling factors for {station} (dataset: {dataset_name})")
            else:
                logger.warning(f"Not enough monthly data for {station}. Using annual scaling factor instead.")
        else:
            logger.warning(f"Required columns {obs_col} and {pred_col} not found for station {station}")
    
    return monthly_scaling_factors

def apply_monthly_scaling_factors(precipitation_data, monthly_scaling_factors):
    """
    Apply monthly scaling factors to precipitation data.
    
    Parameters
    ----------
    precipitation_data : pandas.DataFrame
        DataFrame with precipitation data and a Date column
    monthly_scaling_factors : dict
        Dictionary with monthly scaling factors for each station
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with scaled precipitation data
    """
    # Create a copy to avoid modifying the original
    scaled_data = precipitation_data.copy()
    
    # Ensure we have a Date column
    if 'Date' not in scaled_data.columns and 'datetime' in scaled_data.columns:
        scaled_data['Date'] = scaled_data['datetime']
    
    if 'Date' not in scaled_data.columns:
        logger.error("Precipitation data must have a 'Date' or 'datetime' column")
        return scaled_data
    
    # Add month column
    scaled_data['Month'] = pd.to_datetime(scaled_data['Date']).dt.month
    
    # Apply scaling factors to each station
    for station, info in monthly_scaling_factors.items():
        if station in scaled_data.columns:
            monthly_factors = info["monthly_factors"]
            
            # Apply the appropriate factor for each month
            for month, month_info in monthly_factors.items():
                factor = month_info["factor"]
                month_mask = scaled_data['Month'] == month
                scaled_data.loc[month_mask, station] = scaled_data.loc[month_mask, station] * factor
            
            logger.info(f"Applied monthly scaling factors to {station}")
    
    # Remove the temporary month column
    scaled_data = scaled_data.drop('Month', axis=1)
    
    return scaled_data

def create_corrected_dataset(best_datasets, scaling_factors, datasets_dict, output_dir=None):
    """
    Create a corrected dataset using the best dataset for each station.
    
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
        DataFrame with corrected precipitation data
    """
    # Create an empty DataFrame for the corrected data
    corrected_data = None
    
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
        
        # Create a DataFrame with just date and station data
        if 'Date' in original_data.columns:
            station_data = original_data[['Date', station]].copy()
        elif 'datetime' in original_data.columns:
            station_data = original_data[['datetime', station]].copy()
            station_data = station_data.rename(columns={'datetime': 'Date'})
        else:
            logger.warning(f"No date column found in dataset {dataset_name}. Skipping station {station}.")
            continue
        
        # Apply scaling factor if available
        if station in scaling_factors:
            factor = scaling_factors[station]["factor"]
            station_data[station] = station_data[station] * factor
            logger.info(f"Applied scaling factor {factor:.3f} to {station}")
        
        # Add to the corrected dataset
        if corrected_data is None:
            corrected_data = station_data
        else:
            # Merge with existing data on Date
            corrected_data = pd.merge(corrected_data, station_data, on='Date', how='outer')
    
    # Sort by date
    if corrected_data is not None:
        corrected_data = corrected_data.sort_values('Date')
    
    # Save the corrected dataset if output directory is provided
    if output_dir and corrected_data is not None:
        os.makedirs(output_dir, exist_ok=True)
        corrected_data.to_csv(os.path.join(output_dir, "corrected_precipitation.csv"), index=False)
        
        # Save a summary of the corrections
        summary_data = []
        
        for station in corrected_data.columns:
            if station != 'Date':
                summary_data.append({
                    'Station': station,
                    'Best_Dataset': best_datasets[station]['dataset'] if station in best_datasets else 'Unknown',
                    'Scaling_Factor': scaling_factors[station]['factor'] if station in scaling_factors else 1.0,
                    'Original_RMSE': scaling_factors[station]['original_rmse'] if station in scaling_factors else None,
                    'Scaled_RMSE': scaling_factors[station]['scaled_rmse'] if station in scaling_factors else None,
                    'Improvement_Percent': scaling_factors[station]['improvement_percent'] if station in scaling_factors else None
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(os.path.join(output_dir, "correction_summary.csv"), index=False)
    
    return corrected_data

def cross_validate_scaling_factors(observed_df, predicted_df, n_folds=5, random_seed=42):
    """
    Perform cross-validation to assess the robustness of scaling factors.
    
    Parameters
    ----------
    observed_df : pandas.DataFrame
        DataFrame with observed precipitation
    predicted_df : pandas.DataFrame
        DataFrame with predicted precipitation
    n_folds : int, optional
        Number of cross-validation folds
    random_seed : int, optional
        Random seed for reproducibility
        
    Returns
    -------
    dict
        Dictionary with cross-validation results for each station
    """
    if 'Date' not in observed_df.columns or 'Date' not in predicted_df.columns:
        logger.error("Both DataFrames must have a 'Date' column")
        return {}
    
    # Make copies and ensure Date is datetime
    obs_df = observed_df.copy()
    pred_df = predicted_df.copy()
    
    obs_df['Date'] = pd.to_datetime(obs_df['Date'])
    pred_df['Date'] = pd.to_datetime(pred_df['Date'])
    
    # Merge datasets
    merged_df = pd.merge(obs_df, pred_df, on='Date', how='inner')
    
    # Get configuration values
    min_scaling_factor = CONFIG['correction']['min_scaling_factor']
    max_scaling_factor = CONFIG['correction']['max_scaling_factor']
    default_factor = CONFIG['correction']['default_factor']
    
    # Initialize results dictionary
    cv_results = {}
    
    # Process each station
    obs_station_columns = [col for col in obs_df.columns if 'Station_' in col]
    
    for station in obs_station_columns:
        if station not in pred_df.columns:
            continue
            
        # Get observed and predicted column names after merge
        obs_col = f"{station}_x"
        pred_col = f"{station}_y"
        
        if obs_col not in merged_df.columns or pred_col not in merged_df.columns:
            continue
        
        # Get valid data
        valid_data = merged_df[[obs_col, pred_col]].dropna()
        
        if len(valid_data) < n_folds * 5:  # Need at least 5 points per fold
            logger.debug(f"Not enough data for {station} (found {len(valid_data)})")
            continue
        
        # Create fold indices
        np.random.seed(random_seed)
        all_indices = np.arange(len(valid_data))
        np.random.shuffle(all_indices)
        fold_indices = np.array_split(all_indices, n_folds)
        
        # Perform cross-validation
        fold_results = []
        
        for fold in range(n_folds):
            # Get train and test indices
            test_indices = fold_indices[fold]
            train_indices = np.concatenate([fold_indices[i] for i in range(n_folds) if i != fold])
            
            # Split data
            train_data = valid_data.iloc[train_indices]
            test_data = valid_data.iloc[test_indices]
            
            # Calculate scaling factor on training data
            model = LinearRegression(fit_intercept=False)
            X_train = train_data[[pred_col]]
            y_train = train_data[obs_col]
            model.fit(X_train, y_train)
            scaling_factor = model.coef_[0]
            
            # Ensure the scaling factor is reasonable
            if scaling_factor <= 0:
                scaling_factor = default_factor
            elif scaling_factor < min_scaling_factor:
                scaling_factor = min_scaling_factor
            elif scaling_factor > max_scaling_factor:
                scaling_factor = max_scaling_factor
            
            # Apply scaling factor to test data
            predicted_scaled = test_data[pred_col] * scaling_factor
            
            # Calculate metrics
            rmse_orig = np.sqrt(np.mean((test_data[obs_col] - test_data[pred_col])**2))
            rmse_scaled = np.sqrt(np.mean((test_data[obs_col] - predicted_scaled)**2))
            improvement = ((rmse_orig - rmse_scaled) / rmse_orig) * 100 if rmse_orig > 0 else 0
            
            fold_results.append({
                'Fold': fold + 1,
                'Scaling_Factor': scaling_factor,
                'Original_RMSE': rmse_orig,
                'Scaled_RMSE': rmse_scaled,
                'Improvement_Percent': improvement
            })
        
        # Calculate overall statistics
        factors = [result['Scaling_Factor'] for result in fold_results]
        mean_factor = np.mean(factors)
        std_factor = np.std(factors)
        cv_percent = (std_factor / mean_factor) * 100 if mean_factor > 0 else 0
        
        improvements = [result['Improvement_Percent'] for result in fold_results]
        mean_improvement = np.mean(improvements)
        std_improvement = np.std(improvements)
        
        cv_results[station] = {
            'fold_results': fold_results,
            'mean_factor': mean_factor,
            'std_factor': std_factor,
            'cv_percent': cv_percent,
            'mean_improvement': mean_improvement,
            'std_improvement': std_improvement
        }
        
        logger.info(f"Cross-validation for {station}: Mean factor = {mean_factor:.3f} ± {std_factor:.3f} " +
                   f"(CV = {cv_percent:.1f}%), Mean improvement = {mean_improvement:.1f}% ± {std_improvement:.1f}%")
    
    return cv_results

def apply_bias_correction_method(observed_df, predicted_df, method='scaling', params=None):
    """
    Apply different bias correction methods to precipitation data.
    
    Parameters
    ----------
    observed_df : pandas.DataFrame
        DataFrame with observed precipitation
    predicted_df : pandas.DataFrame
        DataFrame with predicted precipitation
    method : str, optional
        Bias correction method: 'scaling', 'delta', 'quantile_mapping'
    params : dict, optional
        Additional parameters for the correction method
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with bias-corrected precipitation
    """
    if 'Date' not in observed_df.columns or 'Date' not in predicted_df.columns:
        logger.error("Both DataFrames must have a 'Date' column")
        return None
    
    # Make copies and ensure Date is datetime
    obs_df = observed_df.copy()
    pred_df = predicted_df.copy()
    
    obs_df['Date'] = pd.to_datetime(obs_df['Date'])
    pred_df['Date'] = pd.to_datetime(pred_df['Date'])
    
    # Default parameters
    if params is None:
        params = {}
    
    # Prepare corrected DataFrame
    corrected_df = pred_df.copy()
    
    # Get station columns
    obs_station_columns = [col for col in obs_df.columns if 'Station_' in col]
    
    # Apply correction method for each station
    for station in obs_station_columns:
        if station not in pred_df.columns:
            continue
        
        # Get calibration period data (observed and predicted in same time period)
        calib_df = pd.merge(obs_df[['Date', station]], pred_df[['Date', station]], 
                           on='Date', how='inner', suffixes=('_obs', '_pred'))
        
        # Ensure we have enough data
        if len(calib_df) < 10:
            logger.warning(f"Not enough calibration data for {station}. Skipping.")
            continue
        
        # Apply correction method
        if method.lower() == 'scaling':
            # Linear scaling method
            # Get scaling factor through regression or ratio
            if params.get('use_regression', True):
                # Regression through origin
                model = LinearRegression(fit_intercept=False)
                X = calib_df[[f"{station}_pred"]]
                y = calib_df[f"{station}_obs"]
                model.fit(X, y)
                factor = model.coef_[0]
            else:
                # Simple ratio of means
                mean_obs = calib_df[f"{station}_obs"].mean()
                mean_pred = calib_df[f"{station}_pred"].mean()
                factor = mean_obs / mean_pred if mean_pred > 0 else 1.0
            
            # Apply scaling factor
            corrected_df[station] = corrected_df[station] * factor
            logger.info(f"Applied scaling factor {factor:.3f} to {station}")
            
        elif method.lower() == 'delta':
            # Delta method (additive correction)
            mean_diff = calib_df[f"{station}_obs"].mean() - calib_df[f"{station}_pred"].mean()
            corrected_df[station] = corrected_df[station] + mean_diff
            logger.info(f"Applied delta correction {mean_diff:.3f} to {station}")
            
        elif method.lower() == 'quantile_mapping':
            # Quantile mapping (empirical CDF transformation)
            n_quantiles = params.get('n_quantiles', 100)
            
            # Calculate empirical CDFs
            obs_sorted = np.sort(calib_df[f"{station}_obs"])
            pred_sorted = np.sort(calib_df[f"{station}_pred"])
            
            # Apply correction to each value
            for i in range(len(corrected_df)):
                if pd.notna(corrected_df.loc[i, station]):
                    # Find closest quantile in predicted CDF
                    pred_value = corrected_df.loc[i, station]
                    quantile_idx = np.searchsorted(pred_sorted, pred_value)
                    
                    if quantile_idx == 0:
                        # Value is below minimum in calibration
                        corrected_value = obs_sorted[0] * (pred_value / pred_sorted[0]) if pred_sorted[0] > 0 else pred_value
                    elif quantile_idx >= len(pred_sorted):
                        # Value is above maximum in calibration
                        corrected_value = obs_sorted[-1] * (pred_value / pred_sorted[-1]) if pred_sorted[-1] > 0 else pred_value
                    else:
                        # Interpolate between quantiles
                        q = (pred_value - pred_sorted[quantile_idx-1]) / (pred_sorted[quantile_idx] - pred_sorted[quantile_idx-1])
                        corrected_value = obs_sorted[quantile_idx-1] + q * (obs_sorted[quantile_idx] - obs_sorted[quantile_idx-1])
                    
                    corrected_df.loc[i, station] = corrected_value
            
            logger.info(f"Applied quantile mapping to {station}")
            
        else:
            logger.error(f"Unknown correction method: {method}")
            return None
    
    return corrected_df