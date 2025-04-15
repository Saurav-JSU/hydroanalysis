# create_highres.py
import os
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("create_highres")

def process_flood_directory(flood_dir, output_dir=None):
    """Process a single flood directory to create high-resolution data."""
    logger.info(f"Processing flood directory: {flood_dir}")
    
    if output_dir is None:
        output_dir = os.path.join(flood_dir, "highres")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Corrected directory
    corrected_dir = os.path.join(flood_dir, "corrected")
    if not os.path.isdir(corrected_dir):
        logger.error(f"Corrected directory not found: {corrected_dir}")
        return False
    
    # Get required files
    corrected_highres_file = os.path.join(corrected_dir, "corrected_highres_precipitation.csv")
    if not os.path.exists(corrected_highres_file):
        logger.error(f"Corrected high-resolution file not found: {corrected_highres_file}")
        return False
    
    # Look for IMERG file to use as pattern
    imerg_file = None
    for potential_path in [
        os.path.join(flood_dir, "imerg_hourly_precipitation.csv"),
        os.path.join(flood_dir, "gee_imerg", "hourly_precipitation.csv")
    ]:
        if os.path.exists(potential_path):
            imerg_file = potential_path
            logger.info(f"Found IMERG pattern file: {imerg_file}")
            break
    
    # Read corrected data
    logger.info(f"Reading corrected high-resolution data: {corrected_highres_file}")
    corrected_df = pd.read_csv(corrected_highres_file)
    
    # Ensure datetime column
    if 'datetime' not in corrected_df.columns:
        if 'Date' in corrected_df.columns:
            corrected_df['datetime'] = pd.to_datetime(corrected_df['Date'])
        else:
            logger.error("No datetime or Date column found")
            return False
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
        return True
    else:
        logger.error("Failed to create high-resolution data")
        return False

def process_all_flood_dirs():
    """Process all flood directories in the results/floods folder."""
    base_dir = "results/floods"
    if not os.path.isdir(base_dir):
        logger.error(f"Base directory not found: {base_dir}")
        return
    
    # Find all flood directories
    flood_dirs = []
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path) and item.startswith("flood_"):
            flood_dirs.append(item_path)
    
    flood_dirs.sort()  # Sort to process in order
    
    logger.info(f"Found {len(flood_dirs)} flood directories to process")
    
    # Process each flood directory
    for flood_dir in flood_dirs:
        logger.info(f"Processing {os.path.basename(flood_dir)}")
        process_flood_directory(flood_dir)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Process specific flood directory
        flood_dir = sys.argv[1]
        process_flood_directory(flood_dir)
    else:
        # Process all flood directories
        process_all_flood_dirs()