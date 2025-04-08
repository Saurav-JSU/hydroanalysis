"""
Functions for comparing precipitation datasets.
"""

import pandas as pd
import numpy as np
import logging
import os

from hydroanalysis.config import CONFIG

logger = logging.getLogger(__name__)

def calculate_accuracy_metrics(observed_df, predicted_df):
    """
    Calculate accuracy metrics between observed and predicted precipitation.
    
    Parameters
    ----------
    observed_df : pandas.DataFrame
        DataFrame with observed precipitation
    predicted_df : pandas.DataFrame
        DataFrame with predicted precipitation
    
    Returns
    -------
    tuple
        (metrics dictionary, comparison DataFrame)
    """
    if observed_df.empty or predicted_df.empty:
        logger.warning("Cannot calculate accuracy: One or both datasets are empty.")
        return {}, pd.DataFrame()
    
    # Make copies to avoid modification warnings
    obs_df = observed_df.copy()
    pred_df = predicted_df.copy()
    
    # Ensure both DataFrames have the same dates
    if 'Date' in obs_df.columns and 'Date' in pred_df.columns:
        obs_df['Date'] = pd.to_datetime(obs_df['Date'])
        pred_df['Date'] = pd.to_datetime(pred_df['Date'])
        merged_df = pd.merge(obs_df, pred_df, on='Date', how='inner')
    else:
        logger.error("Both DataFrames must have a 'Date' column")
        return {}, pd.DataFrame()
    
    logger.info(f"Found {len(merged_df)} dates in common for comparison.")
    
    if len(merged_df) == 0:
        logger.warning("No overlapping dates found. Cannot calculate accuracy metrics.")
        return {}, merged_df
    
    # Calculate metrics for each station
    metrics = {}
    obs_station_columns = [col for col in obs_df.columns if 'Station_' in col]
    
    logger.info("\nAccuracy Metrics:")
    logger.info("-" * 80)
    logger.info(f"{'Station':<15} {'RMSE':>8} {'MAE':>8} {'Corr':>8} {'Bias':>8} {'%Bias':>8} {'Count':>8}")
    logger.info("-" * 80)
    
    for station in obs_station_columns:
        if station not in pred_df.columns:
            logger.warning(f"Station {station} not found in predicted data. Skipping...")
            continue
            
        # Collect observed and predicted values
        obs_values = merged_df[station + '_x']  # Pandas adds _x suffix on merge
        pred_values = merged_df[station + '_y']  # Pandas adds _y suffix on merge
        
        # Rename columns for clarity in the merged DataFrame
        merged_df = merged_df.rename(columns={
            station + '_x': station + '_obs',
            station + '_y': station + '_pred'
        })
        
        # Remove NaN values
        valid_data = pd.DataFrame({
            'obs': obs_values,
            'pred': pred_values
        }).dropna()
        
        # Ensure we have enough valid data points
        min_data_points = CONFIG['comparison']['min_data_points']
        if len(valid_data) < min_data_points:
            logger.warning(f"{station:<15} Not enough valid data points ({len(valid_data)}/{min_data_points}). Skipping...")
            continue
            
        # Calculate metrics
        rmse = np.sqrt(np.mean((valid_data['obs'] - valid_data['pred'])**2))
        mae = np.mean(np.abs(valid_data['obs'] - valid_data['pred']))
        
        # Handle case where all observed values are the same (correlation undefined)
        if np.std(valid_data['obs']) == 0 or np.std(valid_data['pred']) == 0:
            corr = np.nan
        else:
            corr = np.corrcoef(valid_data['obs'], valid_data['pred'])[0, 1]
        
        # Calculate bias
        bias = np.mean(valid_data['pred'] - valid_data['obs'])
        
        # Calculate percent bias
        if np.sum(valid_data['obs']) > 0:
            pbias = 100 * np.sum(valid_data['pred'] - valid_data['obs']) / np.sum(valid_data['obs'])
        else:
            pbias = np.nan
        
        # Calculate additional metrics
        r_squared = corr**2 if not np.isnan(corr) else np.nan
        
        # Calculate KGE (Kling-Gupta Efficiency)
        mean_obs = np.mean(valid_data['obs'])
        mean_pred = np.mean(valid_data['pred'])
        std_obs = np.std(valid_data['obs'])
        std_pred = np.std(valid_data['pred'])
        
        if mean_obs > 0 and std_obs > 0:
            r_term = corr
            beta_term = mean_pred / mean_obs
            gamma_term = (std_pred / mean_pred) / (std_obs / mean_obs)
            kge = 1 - np.sqrt((r_term - 1)**2 + (beta_term - 1)**2 + (gamma_term - 1)**2)
        else:
            kge = np.nan
        
        metrics[station] = {
            'RMSE': rmse,
            'MAE': mae,
            'Correlation': corr,
            'R_squared': r_squared,
            'Bias': bias,
            'Percent_Bias': pbias,
            'KGE': kge,
            'Count': len(valid_data)
        }
        
        logger.info(f"{station:<15} {rmse:>8.2f} {mae:>8.2f} {corr:>8.2f} {bias:>8.2f} {pbias:>8.2f} {len(valid_data):>8}")
    
    logger.info("-" * 80)
    
    return metrics, merged_df

def rank_datasets(all_metrics):
    """
    Rank multiple datasets based on their accuracy metrics.
    
    Parameters
    ----------
    all_metrics : dict
        Dictionary with metrics for each dataset
        
    Returns
    -------
    pandas.DataFrame
        DataFrame with dataset rankings
    """
    if not all_metrics:
        logger.warning("No metrics available for ranking.")
        return None
    
    # Create summary DataFrames for each metric
    summary = {}
    
    # Get a list of all stations across all datasets
    all_stations = set()
    for dataset_name, metrics in all_metrics.items():
        all_stations.update(metrics.keys())
    
    # Initialize summary dictionaries for each metric
    metrics_to_compare = ['RMSE', 'MAE', 'Correlation', 'Bias', 'Percent_Bias', 'KGE']
    for metric in metrics_to_compare:
        summary[metric] = pd.DataFrame(index=sorted(all_stations))
    
    # Fill in the DataFrames
    for dataset_name, metrics in all_metrics.items():
        for station, station_metrics in metrics.items():
            for metric in metrics_to_compare:
                if metric in station_metrics:
                    summary[metric].loc[station, dataset_name] = station_metrics[metric]
    
    # Calculate overall scores
    overall_scores = pd.DataFrame(index=all_metrics.keys())
    
    # Normalize scores for each metric
    for metric in metrics_to_compare:
        # For correlation and KGE, higher is better, so normalize to 0-1 where 1 is best
        if metric in ['Correlation', 'KGE']:
            min_val = summary[metric].mean().min()
            max_val = summary[metric].mean().max()
            if max_val != min_val:  # Avoid division by zero
                overall_scores[f'{metric}_score'] = (summary[metric].mean() - min_val) / (max_val - min_val)
            else:
                overall_scores[f'{metric}_score'] = 1
        
        # For error metrics, lower is better, so normalize to 0-1 where 1 is best
        else:
            min_val = summary[metric].abs().mean().min()
            max_val = summary[metric].abs().mean().max()
            if max_val != min_val:  # Avoid division by zero
                overall_scores[f'{metric}_score'] = 1 - ((summary[metric].abs().mean() - min_val) / (max_val - min_val))
            else:
                overall_scores[f'{metric}_score'] = 1
    
    # Calculate overall score (average of all metric scores)
    overall_scores['Overall_score'] = overall_scores.mean(axis=1)
    
    # Sort by overall score (descending)
    overall_scores = overall_scores.sort_values('Overall_score', ascending=False)
    
    return overall_scores

def identify_best_dataset_per_station(datasets_metrics):
    """
    Determine the best dataset for each individual station based on metrics.
    
    Parameters
    ----------
    datasets_metrics : dict
        Dictionary with metrics for each dataset
        
    Returns
    -------
    dict
        Dictionary with best dataset info for each station
    """
    best_datasets = {}
    
    # Get all stations from the metrics
    all_stations = set()
    for dataset_name, metrics in datasets_metrics.items():
        all_stations.update(metrics.keys())
    
    logger.info(f"Identifying best dataset for {len(all_stations)} stations...")
    
    for station in all_stations:
        # Collect metrics for this station across all datasets
        station_metrics = {}
        
        for dataset_name, metrics in datasets_metrics.items():
            if station in metrics:
                station_metrics[dataset_name] = metrics[station]
        
        if not station_metrics:
            logger.warning(f"No metrics found for station {station}")
            continue
        
        # Calculate composite score for each dataset
        # Weight correlation more heavily as it's important for capturing temporal patterns
        scores = {}
        for dataset_name, metrics in station_metrics.items():
            # Skip datasets with negative correlation - they're not useful
            if metrics.get("Correlation", 0) <= 0:
                continue
                
            # Create composite score using multiple metrics
            rmse_score = 1 / (metrics.get("RMSE", float('inf')) + 0.1)       # Lower RMSE is better
            corr_score = metrics.get("Correlation", 0) * 2                   # Higher correlation is better (weighted x2)
            bias_score = 1 / (abs(metrics.get("Bias", float('inf'))) + 0.1)  # Lower absolute bias is better
            kge_score = metrics.get("KGE", 0) * 1.5 if "KGE" in metrics else 0  # Higher KGE is better (weighted x1.5)
            
            scores[dataset_name] = rmse_score + corr_score + bias_score + kge_score
        
        if not scores:
            logger.warning(f"No suitable datasets found for station {station}")
            continue
        
        # Select best dataset
        best_dataset = max(scores, key=scores.get)
        
        best_datasets[station] = {
            "dataset": best_dataset,
            "metrics": station_metrics[best_dataset],
            "all_metrics": station_metrics
        }
        
        logger.info(f"Station {station}: Best dataset is {best_dataset} " +
                   f"(RMSE: {station_metrics[best_dataset].get('RMSE', 'N/A'):.2f}, " +
                   f"Corr: {station_metrics[best_dataset].get('Correlation', 'N/A'):.2f})")
    
    return best_datasets

def compare_all_datasets(observed_df, datasets_dict, output_dir=None):
    """
    Compare multiple precipitation datasets against observed data.
    
    Parameters
    ----------
    observed_df : pandas.DataFrame
        DataFrame with observed precipitation
    datasets_dict : dict
        Dictionary with dataset name as key and DataFrame as value
    output_dir : str, optional
        Directory to save outputs
        
    Returns
    -------
    tuple
        (metrics dict, rankings DataFrame, best_datasets dict)
    """
    # Calculate metrics for each dataset
    all_metrics = {}
    comparison_dfs = {}
    
    for dataset_name, dataset_df in datasets_dict.items():
        logger.info(f"Calculating metrics for {dataset_name}")
        metrics, comparison_df = calculate_accuracy_metrics(observed_df, dataset_df)
        
        if metrics:  # Only include if we have valid metrics
            all_metrics[dataset_name] = metrics
            comparison_dfs[dataset_name] = comparison_df
        
        # Save results if output directory is provided
        if output_dir:
            dataset_dir = os.path.join(output_dir, dataset_name.replace(" ", "_"))
            os.makedirs(dataset_dir, exist_ok=True)
            
            # Save comparison data
            if not comparison_df.empty:
                comparison_df.to_csv(os.path.join(dataset_dir, "comparison_data.csv"), index=False)
            
            # Save metrics
            if metrics:
                metrics_df = pd.DataFrame([
                    {
                        'Station': station,
                        'RMSE': m.get('RMSE', np.nan),
                        'MAE': m.get('MAE', np.nan),
                        'Correlation': m.get('Correlation', np.nan),
                        'R_squared': m.get('R_squared', np.nan),
                        'Bias': m.get('Bias', np.nan),
                        'Percent_Bias': m.get('Percent_Bias', np.nan),
                        'KGE': m.get('KGE', np.nan),
                        'Count': m.get('Count', 0)
                    }
                    for station, m in metrics.items()
                ])
                metrics_df.to_csv(os.path.join(dataset_dir, "accuracy_metrics.csv"), index=False)
    
    # Rank datasets
    rankings = rank_datasets(all_metrics)
    
    # Save rankings if output directory is provided
    if output_dir and rankings is not None:
        rankings.to_csv(os.path.join(output_dir, "dataset_rankings.csv"))
    
    # Identify best dataset for each station
    best_datasets = identify_best_dataset_per_station(all_metrics)
    
    # Save best dataset info if output directory is provided
    if output_dir and best_datasets:
        best_dataset_info = [
            {
                "Station": station,
                "Best_Dataset": info["dataset"],
                "RMSE": info["metrics"].get("RMSE", np.nan),
                "Correlation": info["metrics"].get("Correlation", np.nan),
                "Bias": info["metrics"].get("Bias", np.nan),
                "KGE": info["metrics"].get("KGE", np.nan)
            }
            for station, info in best_datasets.items()
        ]
        best_dataset_df = pd.DataFrame(best_dataset_info)
        best_dataset_df.to_csv(os.path.join(output_dir, "best_datasets_summary.csv"), index=False)
    
    return all_metrics, rankings, best_datasets, comparison_dfs

def analyze_seasonal_performance(observed_df, predicted_df, output_dir=None):
    """
    Analyze seasonal performance of a precipitation dataset.
    
    Parameters
    ----------
    observed_df : pandas.DataFrame
        DataFrame with observed precipitation
    predicted_df : pandas.DataFrame
        DataFrame with predicted precipitation
    output_dir : str, optional
        Directory to save outputs
        
    Returns
    -------
    dict
        Dictionary with seasonal metrics
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
    
    # Add season column
    merged_df['Month'] = merged_df['Date'].dt.month
    merged_df['Season'] = 'Unknown'
    
    # Define seasons (adjust as needed for different regions)
    merged_df.loc[merged_df['Month'].isin([12, 1, 2]), 'Season'] = 'Winter'
    merged_df.loc[merged_df['Month'].isin([3, 4, 5]), 'Season'] = 'Spring'
    merged_df.loc[merged_df['Month'].isin([6, 7, 8]), 'Season'] = 'Summer'
    merged_df.loc[merged_df['Month'].isin([9, 10, 11]), 'Season'] = 'Fall'
    
    # Calculate metrics for each season and station
    seasonal_metrics = {}
    obs_station_columns = [col for col in obs_df.columns if 'Station_' in col]
    
    for season in ['Winter', 'Spring', 'Summer', 'Fall']:
        season_df = merged_df[merged_df['Season'] == season]
        season_metrics = {}
        
        if len(season_df) == 0:
            logger.warning(f"No data for season: {season}")
            continue
        
        for station in obs_station_columns:
            if station not in pred_df.columns:
                continue
                
            # Get observed and predicted column names after merge
            obs_col = f"{station}_x"
            pred_col = f"{station}_y"
            
            if obs_col not in season_df.columns or pred_col not in season_df.columns:
                continue
            
            # Get valid data
            valid_data = pd.DataFrame({
                'obs': season_df[obs_col],
                'pred': season_df[pred_col]
            }).dropna()
            
            if len(valid_data) < 10:  # Minimum data points for reliable metrics
                logger.debug(f"Not enough data for {station} in {season} (found {len(valid_data)})")
                continue
            
            # Calculate metrics
            rmse = np.sqrt(np.mean((valid_data['obs'] - valid_data['pred'])**2))
            corr = np.corrcoef(valid_data['obs'], valid_data['pred'])[0, 1] if np.std(valid_data['obs']) > 0 and np.std(valid_data['pred']) > 0 else np.nan
            bias = np.mean(valid_data['pred'] - valid_data['obs'])
            
            season_metrics[station] = {
                'RMSE': rmse,
                'Correlation': corr,
                'Bias': bias,
                'Count': len(valid_data)
            }
        
        seasonal_metrics[season] = season_metrics
    
    # Save results if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a summary DataFrame
        summary_rows = []
        
        for season, metrics in seasonal_metrics.items():
            for station, station_metrics in metrics.items():
                summary_rows.append({
                    'Season': season,
                    'Station': station,
                    'RMSE': station_metrics['RMSE'],
                    'Correlation': station_metrics['Correlation'],
                    'Bias': station_metrics['Bias'],
                    'Count': station_metrics['Count']
                })
        
        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            summary_df.to_csv(os.path.join(output_dir, "seasonal_metrics.csv"), index=False)
    
    return seasonal_metrics

def analyze_extreme_events(observed_df, predicted_df, percentile_threshold=95, output_dir=None):
    """
    Analyze performance for extreme precipitation events.
    
    Parameters
    ----------
    observed_df : pandas.DataFrame
        DataFrame with observed precipitation
    predicted_df : pandas.DataFrame
        DataFrame with predicted precipitation
    percentile_threshold : float, optional
        Percentile threshold for defining extreme events
    output_dir : str, optional
        Directory to save outputs
        
    Returns
    -------
    dict
        Dictionary with extreme event metrics
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
    
    # Calculate metrics for extreme events by station
    extreme_metrics = {}
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
        valid_data = pd.DataFrame({
            'obs': merged_df[obs_col],
            'pred': merged_df[pred_col]
        }).dropna()
        
        if len(valid_data) < 20:  # Need enough data for reliable percentile
            logger.debug(f"Not enough data for {station} (found {len(valid_data)})")
            continue
        
        # Calculate threshold for extreme events
        threshold = np.percentile(valid_data['obs'], percentile_threshold)
        
        # Filter for extreme events
        extreme_data = valid_data[valid_data['obs'] >= threshold]
        
        if len(extreme_data) < 5:  # Need enough extreme events
            logger.debug(f"Not enough extreme events for {station} (found {len(extreme_data)})")
            continue
        
        # Calculate metrics for extreme events
        rmse = np.sqrt(np.mean((extreme_data['obs'] - extreme_data['pred'])**2))
        corr = np.corrcoef(extreme_data['obs'], extreme_data['pred'])[0, 1] if np.std(extreme_data['obs']) > 0 and np.std(extreme_data['pred']) > 0 else np.nan
        bias = np.mean(extreme_data['pred'] - extreme_data['obs'])
        percent_bias = 100 * bias / np.mean(extreme_data['obs']) if np.mean(extreme_data['obs']) > 0 else np.nan
        
        # Count missed events (observed extreme but predicted < threshold)
        missed_events = np.sum((valid_data['obs'] >= threshold) & (valid_data['pred'] < threshold))
        
        # Count false alarms (predicted extreme but observed < threshold)
        false_alarms = np.sum((valid_data['pred'] >= threshold) & (valid_data['obs'] < threshold))
        
        # Calculate hit rate and false alarm rate
        total_observed_extremes = np.sum(valid_data['obs'] >= threshold)
        total_non_extremes = np.sum(valid_data['obs'] < threshold)
        
        hit_rate = 1 - (missed_events / total_observed_extremes) if total_observed_extremes > 0 else np.nan
        false_alarm_rate = false_alarms / total_non_extremes if total_non_extremes > 0 else np.nan
        
        extreme_metrics[station] = {
            'Threshold': threshold,
            'RMSE': rmse,
            'Correlation': corr,
            'Bias': bias,
            'Percent_Bias': percent_bias,
            'Extreme_Count': len(extreme_data),
            'Missed_Events': missed_events,
            'False_Alarms': false_alarms,
            'Hit_Rate': hit_rate,
            'False_Alarm_Rate': false_alarm_rate
        }
    
    # Save results if output directory is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a summary DataFrame
        summary_rows = []
        
        for station, metrics in extreme_metrics.items():
            summary_rows.append({
                'Station': station,
                'Threshold': metrics['Threshold'],
                'RMSE': metrics['RMSE'],
                'Correlation': metrics['Correlation'],
                'Bias': metrics['Bias'],
                'Percent_Bias': metrics['Percent_Bias'],
                'Extreme_Count': metrics['Extreme_Count'],
                'Missed_Events': metrics['Missed_Events'],
                'False_Alarms': metrics['False_Alarms'],
                'Hit_Rate': metrics['Hit_Rate'],
                'False_Alarm_Rate': metrics['False_Alarm_Rate']
            })
        
        if summary_rows:
            summary_df = pd.DataFrame(summary_rows)
            summary_df.to_csv(os.path.join(output_dir, "extreme_event_metrics.csv"), index=False)
    
    return extreme_metrics