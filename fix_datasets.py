import pandas as pd
import os
import glob

print("Starting file format fixes...")

# Fix 1: Add Date column to hourly precipitation files that only have datetime
for dataset_dir in ['era5', 'imerg', 'gsmap']:
    pattern = f"results/floods/flood_*/gee_{dataset_dir}/hourly_precipitation.csv"
    for filepath in glob.glob(pattern):
        print(f"Processing {filepath}...")
        try:
            # Read the file
            df = pd.read_csv(filepath)
            
            # Check if it has datetime but no Date column
            if 'datetime' in df.columns and 'Date' not in df.columns:
                # Add Date column
                df['Date'] = pd.to_datetime(df['datetime'])
                print(f"  Added Date column to {filepath}")
                # Save back
                df.to_csv(filepath, index=False)
        except Exception as e:
            print(f"  Error processing {filepath}: {e}")

# Fix 2: Convert station_metadata.csv to Excel format if needed
metadata_file = "data/station_metadata.csv"
if os.path.exists(metadata_file):
    try:
        # Read the CSV file
        df = pd.read_csv(metadata_file)
        
        # Save as Excel
        excel_path = metadata_file.replace('.csv', '.xlsx')
        df.to_excel(excel_path, index=False)
        print(f"Converted {metadata_file} to Excel format: {excel_path}")
    except Exception as e:
        print(f"Error converting metadata file: {e}")

print("File fixes completed!")