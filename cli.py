"""
Command-line interface for the HydroAnalysis package.
"""

import argparse
import os
import sys
import logging
import json
import pandas as pd

from hydroanalysis.config import CONFIG
from hydroanalysis.core.utils import setup_logging
from hydroanalysis.core.data_io import (
    read_discharge_data, 
    read_precipitation_data, 
    read_station_metadata, 
    save_data
)
from hydroanalysis.discharge.flood_events import (
    identify_flood_events, 
    analyze_flood_events
)
from hydroanalysis.precipitation.comparison import (
    calculate_accuracy_metrics, 
    rank_datasets, 
    identify_best_dataset_per_station, 
    compare_all_datasets
)
from hydroanalysis.precipitation.correction import (
    calculate_scaling_factors, 
    apply_scaling_factors, 
    create_corrected_dataset
)
from hydroanalysis.precipitation.disaggregation import (
    disaggregate_to_half_hourly, 
    create_high_resolution_precipitation
)
from hydroanalysis.precipitation.download import (
    download_precipitation_data,
    download_flood_precipitation,
    download_all_flood_precipitation
)

def setup_cli():
    """Set up command-line interface for HydroAnalysis package."""
    parser = argparse.ArgumentParser(
        description="HydroAnalysis - Python package for hydrological data analysis",
        epilog="For more information, visit: https://github.com/username/hydroanalysis"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # flood-events command
    flood_parser = subparsers.add_parser('flood-events', help='Identify and analyze flood events')
    flood_parser.add_argument('--discharge', required=True, help='Path to discharge data file')
    flood_parser.add_argument('--precipitation', required=True, help='Path to precipitation data file')
    flood_parser.add_argument('--station', help='Station ID to analyze')
    flood_parser.add_argument('--percentile', type=float, default=95, help='Percentile threshold for flood detection')
    flood_parser.add_argument('--duration', type=int, default=2, help='Minimum flood duration in days')
    flood_parser.add_argument('--buffer', type=int, default=7, help='Number of buffer days before/after flood')
    flood_parser.add_argument('--output', help='Output directory for flood analysis')
    flood_parser.add_argument('--max-events', type=int, default=10, help='Maximum number of flood events to analyze')
    flood_parser.add_argument('--years', type=float, help='Number of years in dataset (for return period calculation)')
    
    # compare-datasets command
    compare_parser = subparsers.add_parser('compare-datasets', help='Compare precipitation datasets')
    compare_parser.add_argument('--observed', required=True, help='Path to observed precipitation data')
    compare_parser.add_argument('--datasets', required=True, nargs='+', help='Paths to predicted datasets')
    compare_parser.add_argument('--dataset-names', nargs='+', help='Names for the datasets (same order as --datasets)')
    compare_parser.add_argument('--metadata', help='Path to station metadata')
    compare_parser.add_argument('--output', help='Output directory for comparison results')
    
    # correct-precipitation command
    correct_parser = subparsers.add_parser('correct-precipitation', help='Apply corrections to precipitation data')
    correct_parser.add_argument('--datasets-dir', required=True, help='Directory with comparison results')
    correct_parser.add_argument('--metadata', help='Path to station metadata')
    correct_parser.add_argument('--output', help='Output directory for corrected precipitation')
    correct_parser.add_argument('--monthly-factors', action='store_true', help='Use monthly correction factors')
    
    # create-high-resolution command
    highres_parser = subparsers.add_parser('create-high-resolution', help='Create high-resolution precipitation data')
    highres_parser.add_argument('--datasets-dir', required=True, help='Directory with comparison results')
    highres_parser.add_argument('--dataset-files', required=True, nargs='+', help='Paths to original dataset files')
    highres_parser.add_argument('--dataset-names', required=True, nargs='+', help='Names for the datasets (same order as --dataset-files)')
    highres_parser.add_argument('--metadata', help='Path to station metadata')
    highres_parser.add_argument('--output', required=True, help='Output directory for high-resolution data')
    
    # download-precipitation command
    download_parser = subparsers.add_parser('download-precipitation', 
                                            help='Download precipitation data from Earth Engine')
    download_parser.add_argument('--metadata', required=True, 
                                help='Path to station metadata file (CSV or Excel)')
    download_parser.add_argument('--dataset', required=True, choices=['era5', 'gsmap', 'imerg'],
                                help='Dataset to download (ERA5-Land, GSMaP, or IMERG)')
    download_parser.add_argument('--start-date', required=True, 
                                help='Start date for data download (YYYY-MM-DD)')
    download_parser.add_argument('--end-date', required=True, 
                                help='End date for data download (YYYY-MM-DD)')
    download_parser.add_argument('--output', default='data', 
                                help='Output directory for downloaded data')
    download_parser.add_argument('--resolution', choices=['daily', 'hourly', 'both'], default='daily',
                                help='Time resolution of output (daily, hourly, or both)')
    download_parser.add_argument('--adjust-timezone', action='store_true',
                                help='Adjust times for Nepal timezone (UTC+5:45)')
    download_parser.add_argument('--nepal-convention', action='store_true',
                                help='Use Nepal 8:45 AM recording convention for daily aggregation')
    download_parser.add_argument('--station-id-column', default='Station_ID',
                                help='Column name in metadata that contains station IDs')
    
    # download-flood-precipitation command
    download_flood_parser = subparsers.add_parser('download-flood-precipitation', 
                                                help='Download precipitation data for identified flood events')
    download_flood_parser.add_argument('--floods-dir', required=True, 
                                    help='Directory containing flood event results')
    download_flood_parser.add_argument('--metadata', required=True, 
                                    help='Path to station metadata file (CSV or Excel)')
    download_flood_parser.add_argument('--dataset', required=True, choices=['era5', 'gsmap', 'imerg'],
                                    help='Dataset to download (ERA5-Land, GSMaP, or IMERG)')
    download_flood_parser.add_argument('--resolution', choices=['daily', 'hourly', 'both'], default='daily',
                                    help='Time resolution of output (daily, hourly, or both)')
    download_flood_parser.add_argument('--adjust-timezone', action='store_true',
                                    help='Adjust times for Nepal timezone (UTC+5:45)')
    download_flood_parser.add_argument('--nepal-convention', action='store_true',
                                    help='Use Nepal 8:45 AM recording convention for daily aggregation')
    download_flood_parser.add_argument('--station-id-column', default='Station_ID',
                                    help='Column name in metadata that contains station IDs')
    
    return parser

def run_flood_events_command(args):
    """Execute the flood-events command."""
    logger = logging.getLogger("hydroanalysis.cli")
    
    # Read discharge data
    if args.station:
        discharge_data = read_discharge_data(args.discharge, station_id=args.station)
    else:
        discharge_data = read_discharge_data(args.discharge)
    
    if discharge_data is None:
        logger.error("Failed to read discharge data")
        return 1
    
    # Read precipitation data
    precip_data = read_precipitation_data(args.precipitation)
    if precip_data is None:
        logger.error("Failed to read precipitation data")
        return 1
    
    # Create output directory
    output_dir = args.output if args.output else "flood_events_analysis"
    
    # Run analysis
    flood_analysis = analyze_flood_events(
        discharge_data, 
        precip_data, 
        output_dir=output_dir,
        max_events=args.max_events,
        years_of_data=args.years
    )
    
    logger.info(f"Flood events analysis completed. Results saved to {output_dir}")
    return 0

def run_compare_datasets_command(args):
    """Execute the compare-datasets command."""
    logger = logging.getLogger("hydroanalysis.cli")
    
    # Read observed precipitation data
    observed_df = read_precipitation_data(args.observed)
    if observed_df is None:
        logger.error("Failed to read observed precipitation data")
        return 1
    
    # Read each predicted dataset
    predicted_dfs = {}
    dataset_names = args.dataset_names if args.dataset_names else None
    
    for i, dataset_path in enumerate(args.datasets):
        if dataset_names and i < len(dataset_names):
            dataset_name = dataset_names[i]
        else:
            dataset_name = os.path.basename(dataset_path).split('.')[0]
        
        predicted_df = read_precipitation_data(dataset_path)
        if predicted_df is not None:
            predicted_dfs[dataset_name] = predicted_df
        else:
            logger.warning(f"Failed to read dataset: {dataset_path}")
    
    if not predicted_dfs:
        logger.error("No valid predicted datasets could be read")
        return 1
    
    # Read station metadata if provided
    station_metadata = None
    if args.metadata:
        station_metadata = read_station_metadata(args.metadata)
    
    # Create output directory
    output_dir = args.output if args.output else "dataset_comparison_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Compare all datasets
    all_metrics, rankings, best_datasets, comparison_dfs = compare_all_datasets(
        observed_df, predicted_dfs, output_dir=output_dir
    )
    
    # If metadata is available, create maps
    if station_metadata is not None:
        from hydroanalysis.visualization.maps import (
            plot_station_map, 
            plot_best_datasets_map
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
    
    logger.info(f"Dataset comparison completed. Results saved to {output_dir}")
    return 0

def run_correct_precipitation_command(args):
    """Execute the correct-precipitation command."""
    logger = logging.getLogger("hydroanalysis.cli")
    
    # Load dataset comparison results
    datasets_dir = args.datasets_dir
    if not os.path.isdir(datasets_dir):
        logger.error(f"Invalid datasets directory: {datasets_dir}")
        return 1
    
    # Check if this is an individual flood directory
    is_flood_dir = os.path.basename(os.path.normpath(datasets_dir)).startswith("flood_")
    
    # Create output directory
    output_dir = args.output if args.output else "corrected_precipitation"
    os.makedirs(output_dir, exist_ok=True)
    
    # If this is a flood directory, use the new event-based approach
    if is_flood_dir:
        from hydroanalysis.precipitation.correction import run_correct_precipitation_command as run_event_correction
        return run_event_correction(args, flood_dir=datasets_dir)
    
    # Otherwise, use the original approach for general correction
    # Load metrics for each dataset
    all_metrics = {}
    comparison_dfs = {}
    
    dataset_dirs = [d for d in os.listdir(datasets_dir) 
                   if os.path.isdir(os.path.join(datasets_dir, d))]
    
    for dataset_name in dataset_dirs:
        dataset_path = os.path.join(datasets_dir, dataset_name)
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
        logger.error("No valid metrics found in datasets directory")
        return 1
    
    # Read station metadata if provided
    station_metadata = None
    if args.metadata:
        station_metadata = read_station_metadata(args.metadata)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Identify best dataset for each station
    best_datasets = identify_best_dataset_per_station(all_metrics)
    
    # Calculate scaling factors
    if args.monthly_factors:
        # Use monthly scaling factors
        scaling_factors = calculate_monthly_scaling_factors(best_datasets, comparison_dfs)
        
        # Save results
        results = {
            'best_datasets': {
                station: {
                    'dataset': info['dataset'],
                    'metrics': info['metrics']
                } for station, info in best_datasets.items()
            },
            'monthly_scaling_factors': {
                station: {
                    'dataset': info['dataset'],
                    'monthly_factors': info['monthly_factors']
                } for station, info in scaling_factors.items()
            }
        }
    else:
        # Use annual scaling factors
        scaling_factors = calculate_scaling_factors(best_datasets, comparison_dfs)
        
        # Save results
        results = {
            'best_datasets': {
                station: {
                    'dataset': info['dataset'],
                    'metrics': info['metrics']
                } for station, info in best_datasets.items()
            },
            'scaling_factors': {
                station: {
                    'factor': info['factor']
                } for station, info in scaling_factors.items()
            }
        }
    
    with open(os.path.join(output_dir, 'correction_factors.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # If metadata is available, create maps
    if station_metadata is not None:
        from hydroanalysis.visualization.maps import (
            plot_station_map, 
            plot_best_datasets_map,
            plot_scaling_factors_map
        )
        
        # Create scaling factors map
        scaling_map = plot_scaling_factors_map(
            scaling_factors,
            station_metadata,
            output_path=os.path.join(output_dir, "scaling_factors_map.png")
        )
    
    logger.info(f"Precipitation correction completed. Results saved to {output_dir}")
    return 0

def run_create_high_resolution_command(args):
    """Execute the create-high-resolution command."""
    logger = logging.getLogger("hydroanalysis.cli")
    
    # Load dataset comparison results
    datasets_dir = args.datasets_dir
    if not os.path.isdir(datasets_dir):
        logger.error(f"Invalid datasets directory: {datasets_dir}")
        return 1
    
    # Load correction factors
    correction_file = os.path.join(datasets_dir, "correction_factors.json")
    if not os.path.exists(correction_file):
        correction_file = os.path.join(datasets_dir, "corrected_precipitation", "correction_factors.json")
    
    if not os.path.exists(correction_file):
        logger.error(f"Correction factors file not found. Please run 'correct-precipitation' first.")
        return 1
    
    with open(correction_file, 'r') as f:
        correction_data = json.load(f)
    
    best_datasets_info = correction_data.get('best_datasets', {})
    scaling_factors_info = correction_data.get('scaling_factors', {})
    
    # Prepare best datasets dictionary
    best_datasets = {}
    for station, info in best_datasets_info.items():
        best_datasets[station] = {
            "dataset": info['dataset'],
            "metrics": info['metrics']
        }
    
    # Prepare scaling factors dictionary
    scaling_factors = {}
    for station, info in scaling_factors_info.items():
        scaling_factors[station] = {
            "factor": info['factor']
        }
    
    # Read dataset files
    datasets_dict = {}
    dataset_names = args.dataset_names if args.dataset_names else None
    
    for i, dataset_path in enumerate(args.dataset_files):
        if dataset_names and i < len(dataset_names):
            dataset_name = dataset_names[i]
        else:
            dataset_name = os.path.basename(dataset_path).split('.')[0]
        
        dataset_df = read_precipitation_data(dataset_path)
        if dataset_df is not None:
            datasets_dict[dataset_name] = dataset_df
        else:
            logger.warning(f"Failed to read dataset: {dataset_path}")
    
    if not datasets_dict:
        logger.error("No valid datasets could be read")
        return 1
    
    # Read station metadata if provided
    station_metadata = None
    if args.metadata:
        station_metadata = read_station_metadata(args.metadata)
    
    # Create output directory
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)
    
    # Create high-resolution precipitation data
    high_res_data = create_high_resolution_precipitation(
        best_datasets, 
        scaling_factors, 
        datasets_dict, 
        output_dir=output_dir
    )
    
    # Create visualization
    if high_res_data is not None:
        from hydroanalysis.visualization.timeseries import plot_high_resolution_precip
        
        # Sample a 3-day period with significant precipitation
        total_precip = high_res_data.drop('datetime', axis=1).sum()
        station_with_most = total_precip.idxmax()
        
        # Group by day and find the day with most precipitation
        high_res_data['date'] = high_res_data['datetime'].dt.date
        daily_sums = high_res_data.groupby('date')[station_with_most].sum()
        
        if not daily_sums.empty:
            max_day = daily_sums.idxmax()
            
            # Define 3-day window
            start_date = pd.Timestamp(max_day) - pd.Timedelta(days=1)
            end_date = pd.Timestamp(max_day) + pd.Timedelta(days=1)
            
            # Create visualization for the top 3 stations
            top_stations = total_precip.nlargest(3).index.tolist()
            
            # Plot high-resolution precipitation
            plot_high_resolution_precip(
                high_res_data,
                start_date=start_date,
                end_date=end_date,
                stations=top_stations,
                title="Sample 3-Day High-Resolution Precipitation",
                output_path=os.path.join(output_dir, "high_resolution_sample.png")
            )
    
    logger.info(f"High-resolution precipitation data created. Results saved to {output_dir}")
    return 0

def run_download_precipitation_command(args):
    """Execute the download-precipitation command."""
    logger = logging.getLogger("hydroanalysis.cli")
    
    # Read station metadata
    metadata_file = args.metadata
    station_id_col = args.station_id_column
    
    # Use our existing read_station_metadata function
    metadata_df = read_station_metadata(metadata_file)
    if metadata_df is None:
        logger.error(f"Failed to read station metadata from {metadata_file}")
        return 1
    
    # Call the download function
    hourly_df, daily_df = download_precipitation_data(
        metadata_df=metadata_df,
        dataset_name=args.dataset,
        start_date=args.start_date,
        end_date=args.end_date,
        output_dir=args.output,
        adjust_nepal_timezone=args.adjust_timezone,
        nepal_convention=args.nepal_convention,
        resolution=args.resolution,
        station_id_col=station_id_col
    )
    
    if daily_df is None:
        logger.error("Failed to download precipitation data")
        return 1
    
    # Success message
    if args.resolution == 'both':
        logger.info(f"Downloaded {args.dataset} precipitation data at hourly and daily resolution")
    else:
        logger.info(f"Downloaded {args.dataset} precipitation data at {args.resolution} resolution")
    
    # Show path to output files
    dataset_name = args.dataset
    if args.resolution in ['daily', 'both']:
        logger.info(f"Daily data saved to {os.path.join(args.output, f'{dataset_name}_daily_precipitation.csv')}")
    if args.resolution in ['hourly', 'both']:
        logger.info(f"Hourly data saved to {os.path.join(args.output, f'{dataset_name}_hourly_precipitation.csv')}")
    
    return 0

def run_download_flood_precipitation_command(args):
    """Execute the download-flood-precipitation command."""
    logger = logging.getLogger("hydroanalysis.cli")
    
    # Read station metadata
    metadata_file = args.metadata
    station_id_col = args.station_id_column
    
    # Use our existing read_station_metadata function
    metadata_df = read_station_metadata(metadata_file)
    if metadata_df is None:
        logger.error(f"Failed to read station metadata from {metadata_file}")
        return 1
    
    # Download data for all flood events
    results = download_all_flood_precipitation(
        floods_dir=args.floods_dir,
        metadata_df=metadata_df,
        dataset_name=args.dataset,
        adjust_nepal_timezone=args.adjust_timezone,
        nepal_convention=args.nepal_convention,
        resolution=args.resolution,
        station_id_col=station_id_col
    )
    
    # Check results
    if not results:
        logger.error("Failed to download precipitation data for any flood events")
        return 1
    
    logger.info(f"Successfully downloaded {args.dataset} precipitation data for {len(results)} flood events")
    logger.info(f"Data saved in each flood directory under 'gee_{args.dataset}' subfolder")
    
    return 0

def main():
    """Main entry point for the HydroAnalysis CLI."""
    # Set up logging
    logger = setup_logging()
    
    # Parse command-line arguments
    parser = setup_cli()
    args = parser.parse_args()
    
    if args.command == 'flood-events':
        return run_flood_events_command(args)
    elif args.command == 'compare-datasets':
        return run_compare_datasets_command(args)
    elif args.command == 'correct-precipitation':
        return run_correct_precipitation_command(args)
    elif args.command == 'create-high-resolution':
        return run_create_high_resolution_command(args)
    elif args.command == 'download-precipitation':
        return run_download_precipitation_command(args)
    elif args.command == 'download-flood-precipitation':
        return run_download_flood_precipitation_command(args)
    else:
        parser.print_help()
        return 0

if __name__ == "__main__":
    sys.exit(main())