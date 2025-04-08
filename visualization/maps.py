"""
Functions for creating spatial visualizations and maps.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import logging

from hydroanalysis.config import CONFIG

logger = logging.getLogger(__name__)

def plot_station_map(station_metadata, highlight_stations=None, output_path=None):
    """
    Create a map showing the locations of precipitation/discharge stations.
    
    Parameters
    ----------
    station_metadata : pandas.DataFrame
        DataFrame with station metadata including Latitude and Longitude columns
    highlight_stations : list, optional
        List of station IDs to highlight
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
    if 'Latitude' not in station_metadata.columns or 'Longitude' not in station_metadata.columns:
        logger.error("Station metadata must include 'Latitude' and 'Longitude' columns")
        return fig
    
    # Check for station ID column
    id_col = None
    for col_name in ['Station_ID', 'Station ID', 'StationID', 'Station', 'ID']:
        if col_name in station_metadata.columns:
            id_col = col_name
            break
    
    if id_col is None:
        logger.warning("No station ID column found in metadata. Using index as ID.")
        station_metadata = station_metadata.copy()
        station_metadata['Station_ID'] = [f"Station_{i}" for i in range(len(station_metadata))]
        id_col = 'Station_ID'
    
    # Check for station type column
    type_col = None
    for col_name in ['Type', 'Station_Type', 'StationType']:
        if col_name in station_metadata.columns:
            type_col = col_name
            break
    
    # Filter out rows with missing coordinates
    valid_stations = station_metadata.dropna(subset=['Latitude', 'Longitude'])
    
    if len(valid_stations) == 0:
        logger.error("No valid station coordinates found")
        return fig
    
    # Plot all stations
    if type_col and type_col in valid_stations.columns:
        # Plot by station type
        station_types = valid_stations[type_col].unique()
        
        for i, stype in enumerate(station_types):
            type_stations = valid_stations[valid_stations[type_col] == stype]
            ax.scatter(type_stations['Longitude'], type_stations['Latitude'], 
                      label=stype, alpha=0.7, s=50)
    else:
        # Plot all stations with same style
        ax.scatter(valid_stations['Longitude'], valid_stations['Latitude'], 
                  alpha=0.7, s=50, label='Stations')
    
    # Highlight specific stations if requested
    if highlight_stations:
        highlight_df = valid_stations[valid_stations[id_col].isin(highlight_stations)]
        
        if len(highlight_df) > 0:
            ax.scatter(highlight_df['Longitude'], highlight_df['Latitude'], 
                      c='red', s=100, alpha=0.9, edgecolor='k', label='Highlighted Stations')
    
    # Add station labels if not too many
    if len(valid_stations) <= 25:
        for _, row in valid_stations.iterrows():
            station_id = str(row[id_col]).replace('Station_', '')
            ax.annotate(station_id, 
                       (row['Longitude'], row['Latitude']),
                       xytext=(3, 3), textcoords='offset points',
                       fontsize=8)
    
    # Set labels and title
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Station Locations')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add legend if needed
    if type_col or highlight_stations:
        ax.legend()
    
    # Adjust axis limits to add a margin
    x_margin = (valid_stations['Longitude'].max() - valid_stations['Longitude'].min()) * 0.1
    y_margin = (valid_stations['Latitude'].max() - valid_stations['Latitude'].min()) * 0.1
    
    ax.set_xlim([valid_stations['Longitude'].min() - x_margin, valid_stations['Longitude'].max() + x_margin])
    ax.set_ylim([valid_stations['Latitude'].min() - y_margin, valid_stations['Latitude'].max() + y_margin])
    
    # Adjust aspect ratio to be equal
    ax.set_aspect('equal')
    
    # Adjust layout
    fig.tight_layout()
    
    # Save if output path is provided
    if output_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=CONFIG['visualization']['dpi'])
        logger.info(f"Station map saved to {output_path}")
    
    return fig

def plot_scaling_factors_map(scaling_factors, station_metadata, output_path=None):
    """
    Create a map showing the spatial distribution of scaling factors.
    
    Parameters
    ----------
    scaling_factors : dict
        Dictionary with scaling factors for each station
    station_metadata : pandas.DataFrame
        DataFrame with station metadata including Latitude and Longitude columns
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
    if not scaling_factors:
        logger.error("No scaling factors provided")
        return fig
    
    if 'Latitude' not in station_metadata.columns or 'Longitude' not in station_metadata.columns:
        logger.error("Station metadata must include 'Latitude' and 'Longitude' columns")
        return fig
    
    # Check for station ID column
    id_col = None
    for col_name in ['Station_ID', 'Station ID', 'StationID', 'Station', 'ID']:
        if col_name in station_metadata.columns:
            id_col = col_name
            break
    
    if id_col is None:
        logger.error("No station ID column found in metadata")
        return fig
    
    # Create a dataframe with station coordinates and scaling factors
    map_data = []
    
    for station, info in scaling_factors.items():
        # Find station in metadata
        station_row = station_metadata[station_metadata[id_col] == station]
        
        if len(station_row) == 0:
            logger.warning(f"Station {station} not found in metadata")
            continue
        
        # Get coordinates and scaling factor
        lat = station_row['Latitude'].values[0]
        lon = station_row['Longitude'].values[0]
        factor = info["factor"]
        
        if np.isnan(lat) or np.isnan(lon):
            logger.warning(f"Invalid coordinates for station {station}")
            continue
        
        map_data.append({
            'Station': station,
            'Latitude': lat,
            'Longitude': lon,
            'Scaling_Factor': factor
        })
    
    if not map_data:
        logger.error("No valid scaling factor data for mapping")
        return fig
    
    map_df = pd.DataFrame(map_data)
    
    # Create a colormap for scaling factors
    # Define a custom colormap: red for values < 1, green for values > 1
    import matplotlib.colors as mcolors
    
    # Get min and max scaling factors
    vmin = min(0.5, map_df['Scaling_Factor'].min())
    vmax = max(1.5, map_df['Scaling_Factor'].max())
    
    # Center the colormap at 1.0
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)
    
    # Plot the scaling factors as colored points
    scatter = ax.scatter(map_df['Longitude'], map_df['Latitude'], 
                        c=map_df['Scaling_Factor'], norm=norm,
                        cmap='RdYlGn', s=100, alpha=0.8, edgecolor='k')
    
    # Add station labels
    for _, row in map_df.iterrows():
        station_id = str(row['Station']).replace('Station_', '')
        ax.annotate(station_id, 
                   (row['Longitude'], row['Latitude']),
                   xytext=(3, 3), textcoords='offset points',
                   fontsize=8)
    
    # Add a colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Scaling Factor')
    
    # Add reference lines on colorbar
    cbar.ax.axhline(y=norm(1.0), color='k', linestyle='--', linewidth=1)
    
    # Set labels and title
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Spatial Distribution of Scaling Factors')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add explanatory text
    ax.text(0.02, 0.02, 
            "Scaling factors represent the ratio between observed and predicted precipitation.\n" +
            "Values > 1: Model underestimates precipitation\n" +
            "Values < 1: Model overestimates precipitation", 
            transform=ax.transAxes, fontsize=9,
            bbox=dict(facecolor='white', alpha=0.8))
    
    # Adjust axis limits to add a margin
    x_margin = (map_df['Longitude'].max() - map_df['Longitude'].min()) * 0.1
    y_margin = (map_df['Latitude'].max() - map_df['Latitude'].min()) * 0.1
    
    ax.set_xlim([map_df['Longitude'].min() - x_margin, map_df['Longitude'].max() + x_margin])
    ax.set_ylim([map_df['Latitude'].min() - y_margin, map_df['Latitude'].max() + y_margin])
    
    # Adjust aspect ratio to be equal
    ax.set_aspect('equal')
    
    # Adjust layout
    fig.tight_layout()
    
    # Save if output path is provided
    if output_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=CONFIG['visualization']['dpi'])
        logger.info(f"Scaling factors map saved to {output_path}")
    
    return fig

def plot_best_datasets_map(best_datasets, station_metadata, output_path=None):
    """
    Create a map showing the best dataset for each station.
    
    Parameters
    ----------
    best_datasets : dict
        Dictionary with best dataset info for each station
    station_metadata : pandas.DataFrame
        DataFrame with station metadata including Latitude and Longitude columns
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
    if not best_datasets:
        logger.error("No best datasets info provided")
        return fig
    
    if 'Latitude' not in station_metadata.columns or 'Longitude' not in station_metadata.columns:
        logger.error("Station metadata must include 'Latitude' and 'Longitude' columns")
        return fig
    
    # Check for station ID column
    id_col = None
    for col_name in ['Station_ID', 'Station ID', 'StationID', 'Station', 'ID']:
        if col_name in station_metadata.columns:
            id_col = col_name
            break
    
    if id_col is None:
        logger.error("No station ID column found in metadata")
        return fig
    
    # Create a dataframe with station coordinates and best datasets
    map_data = []
    
    for station, info in best_datasets.items():
        # Find station in metadata
        station_row = station_metadata[station_metadata[id_col] == station]
        
        if len(station_row) == 0:
            logger.warning(f"Station {station} not found in metadata")
            continue
        
        # Get coordinates and best dataset
        lat = station_row['Latitude'].values[0]
        lon = station_row['Longitude'].values[0]
        dataset = info["dataset"]
        
        if np.isnan(lat) or np.isnan(lon):
            logger.warning(f"Invalid coordinates for station {station}")
            continue
        
        map_data.append({
            'Station': station,
            'Latitude': lat,
            'Longitude': lon,
            'Dataset': dataset
        })
    
    if not map_data:
        logger.error("No valid best dataset info for mapping")
        return fig
    
    map_df = pd.DataFrame(map_data)
    
    # Get unique datasets and assign colors
    unique_datasets = map_df['Dataset'].unique()
    import matplotlib.cm as cm
    colors = cm.tab10(np.linspace(0, 1, len(unique_datasets)))
    dataset_colors = dict(zip(unique_datasets, colors))
    
    # Plot each dataset with a different color
    for dataset in unique_datasets:
        dataset_df = map_df[map_df['Dataset'] == dataset]
        ax.scatter(dataset_df['Longitude'], dataset_df['Latitude'], 
                  c=[dataset_colors[dataset]], 
                  s=100, alpha=0.8, edgecolor='black',
                  label=dataset)
    
    # Add station labels
    for _, row in map_df.iterrows():
        station_id = str(row['Station']).replace('Station_', '')
        ax.annotate(station_id, 
                   (row['Longitude'], row['Latitude']),
                   xytext=(3, 3), textcoords='offset points',
                   fontsize=8)
    
    # Set labels and title
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Spatial Distribution of Best Precipitation Datasets')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add legend
    ax.legend(title="Best Dataset")
    
    # Adjust axis limits to add a margin
    x_margin = (map_df['Longitude'].max() - map_df['Longitude'].min()) * 0.1
    y_margin = (map_df['Latitude'].max() - map_df['Latitude'].min()) * 0.1
    
    ax.set_xlim([map_df['Longitude'].min() - x_margin, map_df['Longitude'].max() + x_margin])
    ax.set_ylim([map_df['Latitude'].min() - y_margin, map_df['Latitude'].max() + y_margin])
    
    # Adjust aspect ratio to be equal
    ax.set_aspect('equal')
    
    # Adjust layout
    fig.tight_layout()
    
    # Save if output path is provided
    if output_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        plt.savefig(output_path, dpi=CONFIG['visualization']['dpi'])
        logger.info(f"Best datasets map saved to {output_path}")
    
    return fig