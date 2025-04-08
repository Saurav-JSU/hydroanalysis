"""
Functions for creating comparison visualizations.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import logging
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

from hydroanalysis.config import CONFIG

logger = logging.getLogger(__name__)

def plot_dataset_comparison(observed_df, predicted_df, station, dataset_name=None, output_path=None):
    """
    Create comparison plots between observed and predicted precipitation for a station.
    
    Parameters
    ----------
    observed_df : pandas.DataFrame
        DataFrame with observed precipitation
    predicted_df : pandas.DataFrame
        DataFrame with predicted precipitation
    station : str
        Station ID to plot
    dataset_name : str, optional
        Name of the predicted dataset
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
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Check input DataFrames
    if 'Date' not in observed_df.columns or 'Date' not in predicted_df.columns:
        logger.error("Both DataFrames must have a 'Date' column")
        return fig
    
    if station not in observed_df.columns or station not in predicted_df.columns:
        logger.error(f"Station {station} not found in one or both DataFrames")
        return fig
    
    # Copy and ensure dates are datetime
    obs_df = observed_df.copy()
    pred_df = predicted_df.copy()
    
    obs_df['Date'] = pd.to_datetime(obs_df['Date'])
    pred_df['Date'] = pd.to_datetime(pred_df['Date'])
    
    # Merge datasets
    merged_df = pd.merge(
        obs_df[['Date', station]],
        pred_df[['Date', station]],
        on='Date',
        how='inner',
        suffixes=('_obs', '_pred')
    )
    
    if merged_df.empty:
        logger.error("No matching dates found between observed and predicted data")
        return fig
    
    # Column names after merge
    obs_col = f"{station}_obs"
    pred_col = f"{station}_pred"
    
    # Remove NaN values
    valid_data = merged_df[[obs_col, pred_col]].dropna()
    
    # Calculate statistics
    if len(valid_data) >= 2:
        rmse = np.sqrt(np.mean((valid_data[obs_col] - valid_data[pred_col])**2))
        mae = np.mean(np.abs(valid_data[obs_col] - valid_data[pred_col]))
        
        # Handle case where all observed values are the same (correlation undefined)
        if np.std(valid_data[obs_col]) == 0 or np.std(valid_data[pred_col]) == 0:
            corr = np.nan
            r2 = np.nan
        else:
            corr = np.corrcoef(valid_data[obs_col], valid_data[pred_col])[0, 1]
            r2 = corr**2
        
        bias = np.mean(valid_data[pred_col] - valid_data[obs_col])
        
        # Calculate percent bias
        if np.sum(valid_data[obs_col]) > 0:
            pbias = 100 * np.sum(valid_data[pred_col] - valid_data[obs_col]) / np.sum(valid_data[obs_col])
        else:
            pbias = np.nan
        
        # Calculate KGE (Kling-Gupta Efficiency)
        mean_obs = np.mean(valid_data[obs_col])
        mean_pred = np.mean(valid_data[pred_col])
        std_obs = np.std(valid_data[obs_col])
        std_pred = np.std(valid_data[pred_col])
        
        if mean_obs > 0 and std_obs > 0:
            r_term = corr
            beta_term = mean_pred / mean_obs
            gamma_term = (std_pred / mean_pred) / (std_obs / mean_obs)
            kge = 1 - np.sqrt((r_term - 1)**2 + (beta_term - 1)**2 + (gamma_term - 1)**2)
        else:
            kge = np.nan
        
        stats_text = (f"RMSE: {rmse:.2f} mm\n"
                     f"MAE: {mae:.2f} mm\n"
                     f"Correlation: {corr:.2f}\n"
                     f"R²: {r2:.2f}\n"
                     f"Bias: {bias:.2f} mm\n"
                     f"Percent Bias: {pbias:.1f}%\n"
                     f"KGE: {kge:.2f}\n"
                     f"N: {len(valid_data)}")
    else:
        stats_text = "Insufficient data for statistics"
    
    # 1. Scatter plot
    ax1.scatter(valid_data[obs_col], valid_data[pred_col], alpha=0.6)
    
    # Add perfect fit line
    max_val = max(valid_data[obs_col].max(), valid_data[pred_col].max()) * 1.1
    min_val = 0  # Start from 0 for precipitation
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    # Add stats to plot
    ax1.text(0.03, 0.97, stats_text, transform=ax1.transAxes, 
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Labels and title
    title = f'Observed vs {dataset_name} Precipitation' if dataset_name else 'Observed vs Predicted Precipitation'
    ax1.set_title(f'{title} for {station}')
    ax1.set_xlabel('Observed (mm/day)')
    ax1.set_ylabel('Predicted (mm/day)')
    ax1.grid(True, alpha=0.3)
    
    # Equal aspect ratio for scatter plot
    ax1.set_aspect('equal', adjustable='box')
    
    # 2. Time series plot
    ax2.plot(merged_df['Date'], merged_df[obs_col], 'b-', label='Observed')
    ax2.plot(merged_df['Date'], merged_df[pred_col], 'r-', label='Predicted')
    
    # Calculate difference
    merged_df['Difference'] = merged_df[pred_col] - merged_df[obs_col]
    
    # Add difference subplot
    divider = make_axes_locatable(ax2)
    ax3 = divider.append_axes("bottom", size="30%", pad=0.3)
    
    # Plot difference
    ax3.bar(merged_df['Date'], merged_df['Difference'], color='gray', alpha=0.5)
    ax3.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax3.set_ylabel('Difference\n(Pred - Obs)')
    
    # Labels and title for time series
    ax2.set_title(f'Time Series Comparison for {station}')
    ax2.set_ylabel('Precipitation (mm/day)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Final axis gets the x-label
    ax3.set_xlabel('Date')
    
    # Rotate x-axis labels for better readability
    for ax in [ax2, ax3]:
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    # Adjust layout
    fig.tight_layout()
    
    # Save if output path is provided
    if output_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=CONFIG['visualization']['dpi'])
        logger.info(f"Dataset comparison plot saved to {output_path}")
    
    return fig

def plot_scaling_factor(station, best_dataset, scaling_factor, validation_data, output_path=None):
    """
    Plot observed vs predicted with scaling factor regression line.
    
    Parameters
    ----------
    station : str
        Station ID
    best_dataset : str
        Name of the best dataset
    scaling_factor : float
        Calculated scaling factor
    validation_data : pandas.DataFrame
        DataFrame with observed and predicted values
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
    
    # Check input data
    if validation_data is None or validation_data.empty:
        logger.error("No validation data provided")
        return fig
    
    # Define columns
    obs_col = validation_data.columns[0]  # First column should be observed
    pred_col = validation_data.columns[1]  # Second column should be predicted
    
    # Scatter plot of original data
    ax.scatter(validation_data[pred_col], validation_data[obs_col], alpha=0.6, color='blue', label='Original Data')
    
    # Plot 1:1 line for reference
    max_val = max(validation_data[obs_col].max(), validation_data[pred_col].max()) * 1.1
    min_val = 0  # Start from 0 for precipitation
    ax.plot([min_val, max_val], [min_val, max_val], 'k:', label='1:1 Line')
    
    # Plot scaled line
    ax.plot([min_val, max_val], [min_val * scaling_factor, max_val * scaling_factor], 'r-', linewidth=2, 
            label=f'Scaling Factor: {scaling_factor:.3f}')
    
    # Calculate statistics
    rmse_orig = np.sqrt(np.mean((validation_data[obs_col] - validation_data[pred_col])**2))
    scaled_pred = validation_data[pred_col] * scaling_factor
    rmse_scaled = np.sqrt(np.mean((validation_data[obs_col] - scaled_pred)**2))
    improvement = ((rmse_orig - rmse_scaled) / rmse_orig) * 100 if rmse_orig > 0 else 0
    
    # Add stats to plot
    stats_text = (f"Original RMSE: {rmse_orig:.2f} mm\n"
                 f"Scaled RMSE: {rmse_scaled:.2f} mm\n"
                 f"Improvement: {improvement:.1f}%")
    
    ax.text(0.03, 0.97, stats_text, transform=ax.transAxes, 
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Labels and title
    ax.set_title(f'Scaling Factor Determination for {station}\nBest Dataset: {best_dataset}')
    ax.set_xlabel('Original Predicted Precipitation (mm/day)')
    ax.set_ylabel('Observed Precipitation (mm/day)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Equal aspect ratio (square plot)
    ax.set_aspect('equal', adjustable='box')
    
    # Adjust layout
    fig.tight_layout()
    
    # Save if output path is provided
    if output_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=CONFIG['visualization']['dpi'])
        logger.info(f"Scaling factor plot saved to {output_path}")
    
    return fig

def plot_best_datasets(best_datasets, metrics_dict, output_path=None):
    """
    Create a summary plot of best datasets and their metrics.
    
    Parameters
    ----------
    best_datasets : dict
        Dictionary with best dataset info for each station
    metrics_dict : dict
        Dictionary with metrics for each dataset and station
    output_path : str, optional
        Path to save the plot
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    # Set figure style
    plt.style.use(CONFIG['visualization']['style'])
    
    # Extract data
    stations = list(best_datasets.keys())
    best_dataset_names = [info["dataset"] for info in best_datasets.values()]
    
    # Limit to a reasonable number of stations
    max_stations = 15
    if len(stations) > max_stations:
        logger.warning(f"Too many stations to plot clearly. Showing only the first {max_stations}.")
        stations = stations[:max_stations]
        best_dataset_names = best_dataset_names[:max_stations]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1.5])
    
    # 1. Best dataset for each station (pie chart)
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Count occurrences of each dataset
    dataset_counts = {}
    for dataset in best_dataset_names:
        dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
    
    # Sort by count
    sorted_datasets = sorted(dataset_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Plot pie chart
    datasets = [d[0] for d in sorted_datasets]
    counts = [d[1] for d in sorted_datasets]
    
    # Calculate percentages for labels
    total = sum(counts)
    percentages = [count/total*100 for count in counts]
    labels = [f"{d} ({p:.1f}%)" for d, p in zip(datasets, percentages)]
    
    ax1.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    ax1.set_title('Distribution of Best Datasets')
    
    # 2. Map of datasets to colors for consistent coloring
    dataset_colors = {}
    colormap = plt.cm.tab10
    for i, dataset in enumerate(datasets):
        dataset_colors[dataset] = colormap(i % 10)
    
    # 3. Performance metrics by station (horizontal bar chart)
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Extract RMSE and correlation values
    rmse_values = []
    corr_values = []
    
    for station in stations:
        dataset = best_datasets[station]["dataset"]
        if dataset in metrics_dict and station in metrics_dict[dataset]:
            rmse = metrics_dict[dataset][station].get("RMSE", np.nan)
            corr = metrics_dict[dataset][station].get("Correlation", np.nan)
            rmse_values.append(rmse)
            corr_values.append(corr)
        else:
            rmse_values.append(np.nan)
            corr_values.append(np.nan)
    
    # Show metrics by station on a horizontal bar chart
    station_labels = [s.replace('Station_', '') for s in stations]
    
    # Plot RMSE and correlation side by side
    y_pos = np.arange(len(station_labels))
    
    # RMSE bars (primary axis)
    bars1 = ax2.barh(y_pos - 0.2, rmse_values, 0.4, label='RMSE (mm/day)', color='steelblue')
    
    # Create a second y-axis to show correlation
    ax2b = ax2.twiny()
    
    # Plot correlation bars with different color
    bars2 = ax2b.barh(y_pos + 0.2, corr_values, 0.4, label='Correlation', color='indianred')
    
    # Set the limits and labels for the correlation axis
    ax2b.set_xlim(0, 1)
    ax2b.set_xlabel('Correlation')
    
    # Add labels, title and legend for the RMSE axis
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(station_labels)
    ax2.set_xlabel('RMSE (mm/day)')
    ax2.set_title('Performance Metrics by Station')
    
    # Add a legend
    ax2.legend(handles=[bars1, bars2], loc='upper right')
    
    # 3. Station-dataset matrix (colored grid)
    ax3 = fig.add_subplot(gs[1, :])
    
    # Create a matrix showing all metrics for each station-dataset combination
    metric_to_visualize = "RMSE"  # Can be changed to other metrics
    
    # Get all unique datasets
    all_datasets = sorted(set(d for metrics in metrics_dict.values() for d in metrics.keys()))
    
    # Create a DataFrame to hold the metric values
    matrix_data = np.zeros((len(stations), len(all_datasets)))
    matrix_data.fill(np.nan)
    
    # Fill the matrix with metric values
    for i, station in enumerate(stations):
        for j, dataset in enumerate(all_datasets):
            if dataset in metrics_dict and station in metrics_dict[dataset]:
                matrix_data[i, j] = metrics_dict[dataset][station].get(metric_to_visualize, np.nan)
    
    # Create a custom colormap for RMSE (lower is better)
    if metric_to_visualize == "RMSE":
        # RdYlGn colormap (red for high values, green for low values)
        cmap = LinearSegmentedColormap.from_list('custom_cmap', 
                                               [(0, 'darkgreen'), (0.5, 'yellow'), (1, 'darkred')])
        norm = plt.Normalize(vmin=0, vmax=np.nanmax(matrix_data))
    elif metric_to_visualize in ["Correlation", "KGE"]:
        # RdYlGn colormap but inverted (green for high values, red for low values)
        cmap = LinearSegmentedColormap.from_list('custom_cmap', 
                                               [(0, 'darkred'), (0.5, 'yellow'), (1, 'darkgreen')])
        norm = plt.Normalize(vmin=0, vmax=1)
    else:
        # Default colormap
        cmap = 'viridis'
        norm = None
    
    # Create the heatmap
    im = ax3.imshow(matrix_data, aspect='auto', cmap=cmap, norm=norm)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax3)
    cbar.set_label(metric_to_visualize)
    
    # Add labels
    ax3.set_xticks(np.arange(len(all_datasets)))
    ax3.set_xticklabels(all_datasets, rotation=45, ha='right')
    ax3.set_yticks(np.arange(len(stations)))
    ax3.set_yticklabels(station_labels)
    
    # Add title
    ax3.set_title(f'{metric_to_visualize} for Each Station-Dataset Combination')
    
    # Add grid lines
    ax3.set_xticks(np.arange(len(all_datasets)+1)-0.5, minor=True)
    ax3.set_yticks(np.arange(len(stations)+1)-0.5, minor=True)
    ax3.grid(which="minor", color="w", linestyle='-', linewidth=2)
    
    # Highlight best dataset for each station
    for i, station in enumerate(stations):
        best_dataset = best_datasets[station]["dataset"]
        j = all_datasets.index(best_dataset) if best_dataset in all_datasets else -1
        if j >= 0:
            # Add a black border around the best dataset cell
            rect = plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False, edgecolor='black', lw=2)
            ax3.add_patch(rect)
    
    # Adjust layout
    fig.tight_layout()
    
    # Save if output path is provided
    if output_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=CONFIG['visualization']['dpi'])
        logger.info(f"Best datasets summary plot saved to {output_path}")
    
    return fig

def plot_dataset_rankings(rankings_df, output_path=None):
    """
    Create a plot showing the rankings of different precipitation datasets.
    
    Parameters
    ----------
    rankings_df : pandas.DataFrame
        DataFrame with dataset rankings
    output_path : str, optional
        Path to save the plot
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    # Set figure style
    plt.style.use(CONFIG['visualization']['style'])
    
    # Check input data
    if rankings_df is None or rankings_df.empty:
        logger.error("No rankings data provided")
        return plt.figure()
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 1. Overall score
    if 'Overall_score' in rankings_df.columns:
        # Sort by overall score
        sorted_df = rankings_df.sort_values('Overall_score', ascending=False)
        
        # Plot overall score
        ax1.bar(sorted_df.index, sorted_df['Overall_score'], color='skyblue')
        ax1.set_title('Overall Dataset Score (Higher is Better)')
        ax1.set_xlabel('Dataset')
        ax1.set_ylabel('Score')
        ax1.set_ylim(0, 1.05)
        ax1.grid(axis='y', alpha=0.3)
        
        # Add values on top of bars
        for i, score in enumerate(sorted_df['Overall_score']):
            ax1.text(i, score + 0.02, f'{score:.2f}', ha='center')
        
        # Rotate x-axis labels
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    else:
        ax1.text(0.5, 0.5, 'No overall score available', 
                ha='center', va='center', transform=ax1.transAxes)
    
    # 2. Metric-specific scores
    metric_columns = [col for col in rankings_df.columns if col.endswith('_score') and col != 'Overall_score']
    
    if metric_columns:
        # Create a new DataFrame with more readable column names
        metric_names = {col: col.replace('_score', '') for col in metric_columns}
        plot_df = rankings_df[metric_columns].rename(columns=metric_names)
        
        # Plot metric-specific scores
        plot_df.plot(kind='bar', ax=ax2)
        ax2.set_title('Detailed Dataset Scores by Metric (Higher is Better)')
        ax2.set_xlabel('Dataset')
        ax2.set_ylabel('Score')
        ax2.set_ylim(0, 1.05)
        ax2.grid(axis='y', alpha=0.3)
        ax2.legend(title='Metric')
        
        # Rotate x-axis labels
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    else:
        ax2.text(0.5, 0.5, 'No metric-specific scores available', 
                ha='center', va='center', transform=ax2.transAxes)
    
    # Adjust layout
    fig.tight_layout()
    
    # Save if output path is provided
    if output_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=CONFIG['visualization']['dpi'])
        logger.info(f"Dataset rankings plot saved to {output_path}")
    
    return fig

def plot_metric_comparison(metrics_dict, metric_name='RMSE', output_path=None):
    """
    Compare a specific metric across datasets and stations.
    
    Parameters
    ----------
    metrics_dict : dict
        Dictionary with metrics for each dataset
    metric_name : str, optional
        Name of the metric to compare
    output_path : str, optional
        Path to save the plot
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    # Set figure style
    plt.style.use(CONFIG['visualization']['style'])
    
    # Check input data
    if not metrics_dict:
        logger.error("No metrics data provided")
        return plt.figure()
    
    # Create a DataFrame with the metric values
    datasets = list(metrics_dict.keys())
    
    # Get all stations across all datasets
    all_stations = set()
    for dataset_metrics in metrics_dict.values():
        all_stations.update(dataset_metrics.keys())
    
    # Sort stations
    all_stations = sorted(all_stations)
    
    # Create DataFrame with metric values
    metric_df = pd.DataFrame(index=all_stations)
    
    for dataset in datasets:
        metric_values = []
        for station in all_stations:
            if station in metrics_dict[dataset]:
                value = metrics_dict[dataset][station].get(metric_name, np.nan)
                metric_values.append(value)
            else:
                metric_values.append(np.nan)
        
        metric_df[dataset] = metric_values
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot the metric comparison
    metric_df.plot(kind='bar', ax=ax)
    
    # Set title and labels based on metric type
    if metric_name in ['Correlation', 'R_squared', 'KGE']:
        # For these metrics, higher is better
        plt.title(f'Comparison of {metric_name} Across Datasets (Higher is Better)')
    else:
        # For error metrics, lower is better
        plt.title(f'Comparison of {metric_name} Across Datasets (Lower is Better)')
    
    plt.xlabel('Station')
    plt.ylabel(metric_name)
    
    # Add grid
    plt.grid(axis='y', alpha=0.3)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if output path is provided
    if output_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=CONFIG['visualization']['dpi'])
        logger.info(f"Metric comparison plot saved to {output_path}")
    
    return fig

def plot_scaling_factors_distribution(scaling_factors, output_path=None):
    """
    Plot the distribution of scaling factors.
    
    Parameters
    ----------
    scaling_factors : dict
        Dictionary with scaling factors for each station
    output_path : str, optional
        Path to save the plot
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    # Set figure style
    plt.style.use(CONFIG['visualization']['style'])
    
    # Check input data
    if not scaling_factors:
        logger.error("No scaling factors provided")
        return plt.figure()
    
    # Extract scaling factor values
    stations = list(scaling_factors.keys())
    factors = [info["factor"] for info in scaling_factors.values()]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 1. Histogram of scaling factors
    ax1.hist(factors, bins=15, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.set_title('Distribution of Scaling Factors')
    ax1.set_xlabel('Scaling Factor')
    ax1.set_ylabel('Frequency')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add mean and median lines
    mean_factor = np.mean(factors)
    median_factor = np.median(factors)
    
    ax1.axvline(mean_factor, color='r', linestyle='-', label=f'Mean: {mean_factor:.2f}')
    ax1.axvline(median_factor, color='g', linestyle='--', label=f'Median: {median_factor:.2f}')
    ax1.axvline(1.0, color='k', linestyle=':', label='No Correction (1.0)')
    ax1.legend()
    
    # 2. Scaling factors by station
    # Sort by factor value
    sorted_indices = np.argsort(factors)
    sorted_stations = [stations[i] for i in sorted_indices]
    sorted_factors = [factors[i] for i in sorted_indices]
    
    # Bar colors
    colors = ['red' if f < 1 else 'green' for f in sorted_factors]
    
    ax2.bar(range(len(sorted_stations)), sorted_factors, color=colors, alpha=0.7)
    ax2.set_title('Scaling Factors by Station')
    ax2.set_xlabel('Station')
    ax2.set_ylabel('Scaling Factor')
    ax2.set_xticks(range(len(sorted_stations)))
    ax2.set_xticklabels([s.replace('Station_', '') for s in sorted_stations], rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add a horizontal line at 1.0
    ax2.axhline(1.0, color='k', linestyle=':', label='No Correction (1.0)')
    
    # Add legend explaining colors
    red_patch = mpatches.Patch(color='red', alpha=0.7, label='Overestimation (< 1.0)')
    green_patch = mpatches.Patch(color='green', alpha=0.7, label='Underestimation (> 1.0)')
    ax2.legend(handles=[red_patch, green_patch])
    
    # Adjust layout
    fig.tight_layout()
    
    # Save if output path is provided
    if output_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=CONFIG['visualization']['dpi'])
        logger.info(f"Scaling factors distribution plot saved to {output_path}")
    
    return fig

def plot_seasonal_performance(seasonal_metrics, station, output_path=None):
    """
    Plot seasonal performance metrics for a station.
    
    Parameters
    ----------
    seasonal_metrics : dict
        Dictionary with metrics for each season
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
    
    # Check input data
    if not seasonal_metrics:
        logger.error("No seasonal metrics provided")
        return plt.figure()
    
    # Check if station exists in metrics
    station_exists = False
    for season, metrics in seasonal_metrics.items():
        if station in metrics:
            station_exists = True
            break
    
    if not station_exists:
        logger.error(f"Station {station} not found in seasonal metrics")
        return plt.figure()
    
    # Extract data for station
    seasons = sorted(seasonal_metrics.keys())  # Sort for consistent order
    rmse_values = []
    corr_values = []
    bias_values = []
    
    for season in seasons:
        if station in seasonal_metrics[season]:
            rmse = seasonal_metrics[season][station].get('RMSE', np.nan)
            corr = seasonal_metrics[season][station].get('Correlation', np.nan)
            bias = seasonal_metrics[season][station].get('Bias', np.nan)
            
            rmse_values.append(rmse)
            corr_values.append(corr)
            bias_values.append(bias)
        else:
            rmse_values.append(np.nan)
            corr_values.append(np.nan)
            bias_values.append(np.nan)
    
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # 1. RMSE by season
    ax1.bar(seasons, rmse_values, color='steelblue', alpha=0.7)
    ax1.set_title(f'RMSE by Season - {station}')
    ax1.set_xlabel('Season')
    ax1.set_ylabel('RMSE (mm/day)')
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Correlation by season
    ax2.bar(seasons, corr_values, color='darkgreen', alpha=0.7)
    ax2.set_title(f'Correlation by Season - {station}')
    ax2.set_xlabel('Season')
    ax2.set_ylabel('Correlation')
    ax2.grid(axis='y', alpha=0.3)
    
    # Set y-limits for correlation
    ax2.set_ylim(0, 1.0)
    
    # 3. Bias by season
    # Use different colors for positive and negative bias
    colors = ['red' if b < 0 else 'green' for b in bias_values]
    ax3.bar(seasons, bias_values, color=colors, alpha=0.7)
    ax3.set_title(f'Bias by Season - {station}')
    ax3.set_xlabel('Season')
    ax3.set_ylabel('Bias (mm/day)')
    ax3.grid(axis='y', alpha=0.3)
    
    # Add a horizontal line at 0
    ax3.axhline(0, color='k', linestyle=':')
    
    # Add legend explaining colors
    red_patch = mpatches.Patch(color='red', alpha=0.7, label='Underestimation (< 0)')
    green_patch = mpatches.Patch(color='green', alpha=0.7, label='Overestimation (> 0)')
    ax3.legend(handles=[red_patch, green_patch])
    
    # Add overall title
    plt.suptitle(f'Seasonal Performance Metrics for {station}', fontsize=16, y=1.05)
    
    # Adjust layout
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save if output path is provided
    if output_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=CONFIG['visualization']['dpi'])
        logger.info(f"Seasonal performance plot saved to {output_path}")
    
    return fig

def plot_extreme_events_analysis(extreme_metrics, station, output_path=None):
    """
    Plot performance metrics for extreme precipitation events.
    
    Parameters
    ----------
    extreme_metrics : dict
        Dictionary with extreme event metrics for each station
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
    
    # Check input data
    if not extreme_metrics or station not in extreme_metrics:
        logger.error(f"No extreme event metrics for station {station}")
        return plt.figure()
    
    # Get metrics for this station
    metrics = extreme_metrics[station]
    
    # Create figure with subplots in a 2x2 grid
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Scatter plot of extreme events
    if 'validation_data' in metrics:
        validation_data = metrics['validation_data']
        obs_col = validation_data.columns[0]  # Observed column
        pred_col = validation_data.columns[1]  # Predicted column
        
        # Identify extreme events
        threshold = metrics['Threshold']
        extreme_mask = validation_data[obs_col] >= threshold
        extreme_data = validation_data[extreme_mask]
        non_extreme_data = validation_data[~extreme_mask]
        
        # Plot non-extreme events
        ax1.scatter(non_extreme_data[obs_col], non_extreme_data[pred_col], 
                  alpha=0.4, color='gray', label='Normal Events')
        
        # Plot extreme events
        ax1.scatter(extreme_data[obs_col], extreme_data[pred_col], 
                  alpha=0.8, color='red', label='Extreme Events')
        
        # Add 1:1 line
        max_val = max(validation_data[obs_col].max(), validation_data[pred_col].max()) * 1.1
        ax1.plot([0, max_val], [0, max_val], 'k--', alpha=0.7, label='1:1 Line')
        
        # Add threshold lines
        ax1.axvline(x=threshold, color='r', linestyle=':', alpha=0.7,
                  label=f'Threshold: {threshold:.1f} mm')
        ax1.axhline(y=threshold, color='r', linestyle=':', alpha=0.7)
        
        # Add statistics
        stats_text = (f"RMSE: {metrics.get('RMSE', 'N/A')}\n"
                     f"Correlation: {metrics.get('Correlation', 'N/A'):.2f}\n"
                     f"Bias: {metrics.get('Bias', 'N/A'):.2f} mm\n"
                     f"Extreme Count: {metrics.get('Extreme_Count', 'N/A')}")
        
        ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Set labels and title
        ax1.set_xlabel('Observed Precipitation (mm)')
        ax1.set_ylabel('Predicted Precipitation (mm)')
        ax1.set_title('Extreme Events Analysis')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, "No validation data available",
                ha='center', va='center', transform=ax1.transAxes)
    
    # 2. Hit Rate / False Alarm Rate
    hit_rate = metrics.get('Hit_Rate', np.nan)
    false_alarm_rate = metrics.get('False_Alarm_Rate', np.nan)
    
    # Create a simple bar chart
    metric_names = ['Hit Rate', 'False Alarm Rate']
    metric_values = [hit_rate, false_alarm_rate]
    
    # Define colors (green for hit rate, red for false alarm rate)
    colors = ['green', 'red']
    
    ax2.bar(metric_names, metric_values, color=colors, alpha=0.7)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel('Rate')
    ax2.set_title('Detection Performance')
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(metric_values):
        if not np.isnan(v):
            ax2.text(i, v + 0.05, f'{v:.2f}', ha='center')
    
    # 3. Contingency Table
    missed_events = metrics.get('Missed_Events', 0)
    false_alarms = metrics.get('False_Alarms', 0)
    hits = metrics.get('Extreme_Count', 0) - missed_events
    
    # Create a 2x2 contingency table
    table_data = [
        [hits, missed_events],
        [false_alarms, 'N/A']  # We don't know correct negatives
    ]
    
    # Create a table
    table = ax3.table(cellText=[[str(hits), str(missed_events)], 
                              [str(false_alarms), 'N/A']],
                     rowLabels=['Predicted\nYes', 'Predicted\nNo'],
                     colLabels=['Observed\nYes', 'Observed\nNo'],
                     cellLoc='center',
                     loc='center')
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    # Set title
    ax3.set_title('Contingency Table')
    
    # Remove axis ticks
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.axis('off')
    
    # 4. Bias distribution
    if 'validation_data' in metrics:
        validation_data = metrics['validation_data']
        obs_col = validation_data.columns[0]
        pred_col = validation_data.columns[1]
        
        # Calculate bias for each event
        validation_data['bias'] = validation_data[pred_col] - validation_data[obs_col]
        
        # Separate extreme and non-extreme events
        extreme_mask = validation_data[obs_col] >= threshold
        extreme_bias = validation_data.loc[extreme_mask, 'bias']
        non_extreme_bias = validation_data.loc[~extreme_mask, 'bias']
        
        # Create histogram of bias
        ax4.hist(non_extreme_bias, bins=20, alpha=0.5, color='gray', 
                label='Normal Events')
        ax4.hist(extreme_bias, bins=20, alpha=0.7, color='red', 
                label='Extreme Events')
        
        # Add vertical line at zero bias
        ax4.axvline(x=0, color='k', linestyle='--')
        
        # Set labels and title
        ax4.set_xlabel('Bias (Predicted - Observed) [mm]')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Bias Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, "No validation data available",
                ha='center', va='center', transform=ax4.transAxes)
    
    # Add overall title
    plt.suptitle(f'Extreme Precipitation Events Analysis for {station}', 
                fontsize=16, y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save if output path is provided
    if output_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=CONFIG['visualization']['dpi'])
        logger.info(f"Extreme events analysis plot saved to {output_path}")
    
    return fig

def plot_multiple_dataset_comparison(observed_df, datasets_dict, station, output_path=None):
    """
    Create comparison plots showing multiple datasets against observed data for a station.
    
    Parameters
    ----------
    observed_df : pandas.DataFrame
        DataFrame with observed precipitation
    datasets_dict : dict
        Dictionary with dataset name as key and DataFrame as value
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
    
    # Check inputs
    if station not in observed_df.columns:
        logger.error(f"Station {station} not found in observed data")
        return plt.figure()
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
    
    # Create a color cycle for consistent coloring
    colors = plt.cm.tab10(np.linspace(0, 1, len(datasets_dict)))
    
    # 1. Scatter plot for each dataset
    for i, (dataset_name, dataset_df) in enumerate(datasets_dict.items()):
        if station not in dataset_df.columns:
            logger.warning(f"Station {station} not found in dataset {dataset_name}")
            continue
        
        # Merge with observed data
        merged_df = pd.merge(
            observed_df[['Date', station]],
            dataset_df[['Date', station]],
            on='Date',
            how='inner',
            suffixes=('_obs', f'_{dataset_name}')
        )
        
        if merged_df.empty:
            logger.warning(f"No matching dates between observed and {dataset_name}")
            continue
        
        # Column names
        obs_col = f"{station}_obs"
        pred_col = f"{station}_{dataset_name}"
        
        # Filter out NaN values
        valid_data = merged_df[[obs_col, pred_col]].dropna()
        
        if len(valid_data) < 5:
            logger.warning(f"Not enough valid data points for {dataset_name}")
            continue
        
        # Calculate metrics
        rmse = np.sqrt(np.mean((valid_data[obs_col] - valid_data[pred_col])**2))
        corr = np.corrcoef(valid_data[obs_col], valid_data[pred_col])[0, 1] if np.std(valid_data[obs_col]) > 0 and np.std(valid_data[pred_col]) > 0 else np.nan
        
        # Plot scatter
        ax1.scatter(valid_data[obs_col], valid_data[pred_col], 
                  alpha=0.6, color=colors[i],
                  label=f"{dataset_name} (RMSE={rmse:.2f}, r={corr:.2f})")
    
    # Add 1:1 line
    max_val = ax1.get_xlim()[1]
    ax1.plot([0, max_val], [0, max_val], 'k--', label='1:1 Line')
    
    # Set labels and title
    ax1.set_xlabel('Observed Precipitation (mm)')
    ax1.set_ylabel('Predicted Precipitation (mm)')
    ax1.set_title(f'Multiple Dataset Comparison for {station}')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. Time series plot
    # Find a common time period with significant precipitation
    merged_data = observed_df[['Date', station]].rename(columns={station: f"{station}_observed"})
    merged_data['Date'] = pd.to_datetime(merged_data['Date'])
    
    for dataset_name, dataset_df in datasets_dict.items():
        if station in dataset_df.columns:
            dataset_df_copy = dataset_df[['Date', station]].copy()
            dataset_df_copy['Date'] = pd.to_datetime(dataset_df_copy['Date'])
            dataset_df_copy = dataset_df_copy.rename(columns={station: f"{station}_{dataset_name}"})
            merged_data = pd.merge(merged_data, dataset_df_copy, on='Date', how='outer')
    
    # Sort by date
    merged_data = merged_data.sort_values('Date')
    
    # Find a period with significant precipitation
    observed_col = f"{station}_observed"
    if observed_col in merged_data.columns:
        # Calculate running sum to find rainy periods
        window_size = 10  # Look for 10-day windows
        merged_data['rolling_sum'] = merged_data[observed_col].rolling(window=window_size, center=True).sum()
        
        # Find the period with maximum precipitation
        if len(merged_data) > window_size:
            max_period_idx = merged_data['rolling_sum'].idxmax()
            if not pd.isna(max_period_idx):
                center_date = merged_data.loc[max_period_idx, 'Date']
                start_date = center_date - pd.Timedelta(days=window_size//2)
                end_date = center_date + pd.Timedelta(days=window_size//2)
                
                # Filter for this period
                plot_data = merged_data[(merged_data['Date'] >= start_date) & 
                                      (merged_data['Date'] <= end_date)]
                
                # Plot observed data
                ax2.plot(plot_data['Date'], plot_data[observed_col], 
                       'k-', linewidth=2, label='Observed')
                
                # Plot each dataset
                for i, dataset_name in enumerate(datasets_dict.keys()):
                    dataset_col = f"{station}_{dataset_name}"
                    if dataset_col in plot_data.columns:
                        ax2.plot(plot_data['Date'], plot_data[dataset_col], 
                               '-', color=colors[i], label=dataset_name)
                
                # Set labels and title
                ax2.set_xlabel('Date')
                ax2.set_ylabel('Precipitation (mm)')
                ax2.set_title(f'Time Series Comparison ({start_date.date()} to {end_date.date()})')
                ax2.grid(True, alpha=0.3)
                ax2.legend()
            else:
                ax2.text(0.5, 0.5, "No suitable period found for time series comparison",
                       ha='center', va='center', transform=ax2.transAxes)
        else:
            ax2.text(0.5, 0.5, "Not enough data for time series comparison",
                   ha='center', va='center', transform=ax2.transAxes)
    else:
        ax2.text(0.5, 0.5, "No observed data available for time series comparison",
               ha='center', va='center', transform=ax2.transAxes)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if output path is provided
    if output_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=CONFIG['visualization']['dpi'])
        logger.info(f"Multiple dataset comparison plot saved to {output_path}")
    
    return fig

def plot_monthly_scaling_factors(monthly_factors, station, output_path=None):
    """
    Plot monthly scaling factors for a station.
    
    Parameters
    ----------
    monthly_factors : dict
        Dictionary with monthly scaling factors
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
    
    # Check input data
    if station not in monthly_factors:
        logger.error(f"No monthly factors for station {station}")
        return plt.figure()
    
    # Get monthly factors for this station
    station_factors = monthly_factors[station]['monthly_factors']
    
    # Create figure
    fig, ax = plt.subplots(figsize=CONFIG['visualization']['figsize'])
    
    # Extract data
    months = range(1, 13)
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    factors = []
    counts = []
    interpolated = []
    
    for month in months:
        if month in station_factors:
            factors.append(station_factors[month]['factor'])
            counts.append(station_factors[month].get('count', 0))
            interpolated.append(station_factors[month].get('interpolated', False))
        else:
            factors.append(np.nan)
            counts.append(0)
            interpolated.append(True)
    
    # Define colors based on interpolation status
    colors = ['lightgray' if interp else 'steelblue' for interp in interpolated]
    
    # Plot monthly factors
    bars = ax.bar(month_names, factors, color=colors, alpha=0.7)
    
    # Add count labels
    for i, (count, interp) in enumerate(zip(counts, interpolated)):
        if not interp and count > 0:
            ax.text(i, factors[i] + 0.05, f"n={count}", ha='center', fontsize=8)
    
    # Add a horizontal line at 1.0
    ax.axhline(y=1.0, color='k', linestyle='--', label='No Correction')
    
    # Set labels and title
    ax.set_xlabel('Month')
    ax.set_ylabel('Scaling Factor')
    ax.set_title(f'Monthly Scaling Factors for {station}')
    ax.grid(True, alpha=0.3)
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color='steelblue', alpha=0.7, label='Data-based'),
        mpatches.Patch(color='lightgray', alpha=0.7, label='Interpolated')
    ]
    ax.legend(handles=legend_elements)
    
    # Adjust layout
    fig.tight_layout()
    
    # Save if output path is provided
    if output_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=CONFIG['visualization']['dpi'])
        logger.info(f"Monthly scaling factors plot saved to {output_path}")
    
    return fig

def plot_disaggregation_example(hourly_data, half_hourly_data, station, date=None, output_path=None):
    """
    Create a plot showing the disaggregation from hourly to half-hourly precipitation.
    
    Parameters
    ----------
    hourly_data : pandas.DataFrame
        DataFrame with hourly precipitation data
    half_hourly_data : pandas.DataFrame
        DataFrame with half-hourly precipitation data
    station : str
        Station ID to plot
    date : datetime.datetime, optional
        Specific date to plot (if None, a rainy day will be chosen)
    output_path : str, optional
        Path to save the plot
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    # Set figure style
    plt.style.use(CONFIG['visualization']['style'])
    
    # Check input data
    if station not in hourly_data.columns or station not in half_hourly_data.columns:
        logger.error(f"Station {station} not found in input data")
        return plt.figure()
    
    # Ensure datetime columns
    datetime_col_hourly = 'datetime' if 'datetime' in hourly_data.columns else 'Date'
    datetime_col_half = 'datetime' if 'datetime' in half_hourly_data.columns else 'Date'
    
    if datetime_col_hourly not in hourly_data.columns or datetime_col_half not in half_hourly_data.columns:
        logger.error("Both dataframes must have a datetime or Date column")
        return plt.figure()
    
    # Copy data to avoid modifying originals
    hourly_df = hourly_data.copy()
    half_hourly_df = half_hourly_data.copy()
    
    # Ensure datetime types
    hourly_df[datetime_col_hourly] = pd.to_datetime(hourly_df[datetime_col_hourly])
    half_hourly_df[datetime_col_half] = pd.to_datetime(half_hourly_df[datetime_col_half])
    
    # Add date columns for grouping
    hourly_df['date'] = hourly_df[datetime_col_hourly].dt.date
    half_hourly_df['date'] = half_hourly_df[datetime_col_half].dt.date
    
    # If no specific date provided, find a rainy day
    if date is None:
        # Group by day and sum precipitation
        daily_precip = hourly_df.groupby('date')[station].sum()
        
        # Find days with significant precipitation (more than 5mm)
        rainy_days = daily_precip[daily_precip > 5]
        
        if not rainy_days.empty:
            # Use the day with most precipitation
            date = rainy_days.idxmax()
        else:
            # If no significant precipitation days, use day with maximum precipitation
            date = daily_precip.idxmax()
    
    # Filter data for selected date
    hourly_day = hourly_df[hourly_df['date'] == date]
    half_hourly_day = half_hourly_df[half_hourly_df['date'] == date]
    
    if hourly_day.empty or half_hourly_day.empty:
        logger.error(f"No data available for date {date}")
        return plt.figure()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot hourly data as steps
    hourly_x = []
    hourly_y = []
    
    # Sort by datetime
    hourly_day = hourly_day.sort_values(datetime_col_hourly)
    
    # Create step data
    for _, row in hourly_day.iterrows():
        dt = row[datetime_col_hourly]
        val = row[station]
        hourly_x.extend([dt, dt + pd.Timedelta(hours=1)])
        hourly_y.extend([val, val])
    
    # Plot hourly data (removing last point which is just for step visualization)
# Plot hourly data (removing last point which is just for step visualization)
    if len(hourly_x) > 1:
        ax.step(hourly_x[:-1], hourly_y[:-1], where='post', color='blue', alpha=0.6, 
               linewidth=2, label='Hourly Precipitation')
    
    # Plot half-hourly data as bars
    half_hourly_day = half_hourly_day.sort_values(datetime_col_half)
    bar_width = (pd.Timedelta(minutes=30)).total_seconds() / 86400  # Width in days
    
    ax.bar(half_hourly_day[datetime_col_half], half_hourly_day[station], 
          width=bar_width, alpha=0.7, color='red', label='Half-hourly Precipitation')
    
    # Set labels and title
    date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
    ax.set_title(f'Hourly to Half-hourly Disaggregation Example for {station} on {date_str}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Precipitation (mm)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Format x-axis to show times better
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
    
    # Add annotations to explain the disaggregation
    hourly_total = hourly_day[station].sum()
    half_hourly_total = half_hourly_day[station].sum()
    
    ax.text(0.02, 0.95, 
           f"Daily Precipitation Totals:\n" +
           f"Hourly: {hourly_total:.2f} mm\n" +
           f"Half-hourly: {half_hourly_total:.2f} mm\n\n" +
           f"Pattern-based disaggregation preserves\n" +
           f"the total precipitation amount while\n" +
           f"distributing it to create a more\n" +
           f"realistic temporal pattern.",
           transform=ax.transAxes, 
           bbox=dict(facecolor='white', alpha=0.8),
           verticalalignment='top')
    
    # Adjust layout
    fig.tight_layout()
    
    # Save if output path is provided
    if output_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=CONFIG['visualization']['dpi'])
        logger.info(f"Disaggregation example plot saved to {output_path}")
    
    return fig

def plot_validation_results(validation_df, output_path=None):
    """
    Plot validation results for disaggregation.
    
    Parameters
    ----------
    validation_df : pandas.DataFrame
        DataFrame with validation results
    output_path : str, optional
        Path to save the plot
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure object
    """
    # Set figure style
    plt.style.use(CONFIG['visualization']['style'])
    
    # Check input data
    if validation_df is None or validation_df.empty:
        logger.error("No validation data provided")
        return plt.figure()
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 1. Plot mean absolute difference by station
    if 'Station' in validation_df.columns and 'Mean_Absolute_Difference' in validation_df.columns:
        stations = validation_df['Station']
        mean_abs_diff = validation_df['Mean_Absolute_Difference']
        
        # Sort by mean absolute difference
        sorted_indices = np.argsort(mean_abs_diff)[::-1]  # Descending
        sorted_stations = [stations.iloc[i] for i in sorted_indices]
        sorted_diffs = [mean_abs_diff.iloc[i] for i in sorted_indices]
        
        # Plot as bar chart
        ax1.bar(range(len(sorted_stations)), sorted_diffs, color='indianred', alpha=0.7)
        ax1.set_xticks(range(len(sorted_stations)))
        ax1.set_xticklabels([s.replace('Station_', '') for s in sorted_stations], rotation=45, ha='right')
        ax1.set_title('Mean Absolute Difference by Station')
        ax1.set_xlabel('Station')
        ax1.set_ylabel('Mean Absolute Difference (mm)')
        ax1.grid(axis='y', alpha=0.3)
    else:
        ax1.text(0.5, 0.5, 'Required columns not found',
                ha='center', va='center', transform=ax1.transAxes)
    
    # 2. Plot match rate (percentage of days with less than 0.01mm difference)
    if 'Station' in validation_df.columns and 'Match_Rate' in validation_df.columns:
        stations = validation_df['Station']
        match_rates = validation_df['Match_Rate'] * 100  # Convert to percentage
        
        # Sort by match rate
        sorted_indices = np.argsort(match_rates)  # Ascending
        sorted_stations = [stations.iloc[i] for i in sorted_indices]
        sorted_rates = [match_rates.iloc[i] for i in sorted_indices]
        
        # Plot as bar chart
        ax2.bar(range(len(sorted_stations)), sorted_rates, color='seagreen', alpha=0.7)
        ax2.set_xticks(range(len(sorted_stations)))
        ax2.set_xticklabels([s.replace('Station_', '') for s in sorted_stations], rotation=45, ha='right')
        ax2.set_title('Match Rate by Station')
        ax2.set_xlabel('Station')
        ax2.set_ylabel('Match Rate (%)')
        ax2.set_ylim(0, 100)
        ax2.grid(axis='y', alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'Required columns not found',
                ha='center', va='center', transform=ax2.transAxes)
    
    # Add overall title
    plt.suptitle('Validation Results for Temporal Disaggregation', fontsize=16, y=1.05)
    
    # Adjust layout
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save if output path is provided
    if output_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=CONFIG['visualization']['dpi'])
        logger.info(f"Validation results plot saved to {output_path}")
    
    return fig

def plot_cross_validation_results(cv_results, station, output_path=None):
    """
    Plot cross-validation results for scaling factors.
    
    Parameters
    ----------
    cv_results : dict
        Dictionary with cross-validation results
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
    
    # Check input data
    if station not in cv_results:
        logger.error(f"No cross-validation results for station {station}")
        return plt.figure()
    
    # Get results for this station
    station_cv = cv_results[station]
    fold_results = station_cv.get('fold_results', [])
    
    if not fold_results:
        logger.error("No fold results found")
        return plt.figure()
    
    # Extract results
    folds = [result['Fold'] for result in fold_results]
    scaling_factors = [result['Scaling_Factor'] for result in fold_results]
    improvements = [result['Improvement_Percent'] for result in fold_results]
    
    # Get overall statistics
    mean_factor = station_cv.get('mean_factor', np.mean(scaling_factors))
    std_factor = station_cv.get('std_factor', np.std(scaling_factors))
    cv_percent = station_cv.get('cv_percent', (std_factor / mean_factor) * 100 if mean_factor > 0 else 0)
    mean_improvement = station_cv.get('mean_improvement', np.mean(improvements))
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 1. Plot scaling factors by fold
    ax1.bar(folds, scaling_factors, color='steelblue', alpha=0.7)
    ax1.axhline(y=mean_factor, color='r', linestyle='--', 
               label=f'Mean: {mean_factor:.3f} ± {std_factor:.3f} (CV: {cv_percent:.1f}%)')
    ax1.axhline(y=1.0, color='k', linestyle=':', label='No Correction (1.0)')
    
    ax1.set_title(f'Scaling Factors by Fold - {station}')
    ax1.set_xlabel('Fold')
    ax1.set_ylabel('Scaling Factor')
    ax1.grid(axis='y', alpha=0.3)
    ax1.legend()
    
    # 2. Plot improvement percentage by fold
    ax2.bar(folds, improvements, color='darkgreen', alpha=0.7)
    ax2.axhline(y=mean_improvement, color='r', linestyle='--', 
               label=f'Mean Improvement: {mean_improvement:.1f}%')
    ax2.axhline(y=0, color='k', linestyle=':', label='No Improvement')
    
    ax2.set_title(f'RMSE Improvement by Fold - {station}')
    ax2.set_xlabel('Fold')
    ax2.set_ylabel('Improvement (%)')
    ax2.grid(axis='y', alpha=0.3)
    ax2.legend()
    
    # Add overall title
    plt.suptitle(f'Cross-Validation Results for {station}', fontsize=16, y=1.05)
    
    # Adjust layout
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save if output path is provided
    if output_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=CONFIG['visualization']['dpi'])
        logger.info(f"Cross-validation results plot saved to {output_path}")
    
    return fig