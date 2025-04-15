"""
Fix for missing stations issue in high-resolution precipitation generation.
This script ensures all stations in the corrected data are preserved in high-resolution output.
"""

import os
import shutil
import re

# Backup disaggregation.py if not already backed up
disaggregation_path = 'hydroanalysis/precipitation/disaggregation.py'
if os.path.exists(disaggregation_path) and not os.path.exists(f'{disaggregation_path}.bak'):
    shutil.copy(disaggregation_path, f'{disaggregation_path}.bak')
    print(f"Created backup of {disaggregation_path} as {disaggregation_path}.bak")

# Read the current file
with open(disaggregation_path, 'r') as f:
    content = f.read()

# Find the high-resolution function and add code to handle all stations in corrected data
# Look for where we finalize high_res_data
pattern = r'# Sort by datetime\s+if high_res_data is not None:\s+high_res_data = high_res_data.sort_values\(\'datetime\'\)'

# Add code to handle missing stations from corrected data
replacement = """# Add any missing stations from corrected data
    if corrected_data is not None and high_res_data is not None:
        # Find stations in corrected data that aren't in high_res_data
        missing_stations = [col for col in corrected_data.columns 
                           if col not in high_res_data.columns 
                           and col != 'datetime'
                           and col != 'Date']
        
        if missing_stations:
            logger.info(f"Found {len(missing_stations)} stations in corrected data that weren't processed: {', '.join(missing_stations)}")
            
            # Add these stations from corrected data
            for station in missing_stations:
                logger.info(f"Adding {station} directly from corrected data")
                # Create a temporary dataframe with just this station
                station_data = corrected_data[['datetime', station]]
                
                # Merge with high_res_data
                high_res_data = pd.merge(high_res_data, station_data, on='datetime', how='outer')

    # Sort by datetime
    if high_res_data is not None:
        high_res_data = high_res_data.sort_values('datetime')"""

# Replace the pattern with the new code
new_content = re.sub(pattern, replacement, content)

# Write back to file
with open(disaggregation_path, 'w') as f:
    f.write(new_content)

print(f"Updated {disaggregation_path} to handle stations present in corrected data but missing from best_datasets")
print("This fix ensures all stations from the corrected data are preserved in the high-resolution output")
print("\nTo apply this fix, re-run the high-resolution generation process:")
print("python create_high_resolution.bat")