import os
import glob
import xarray as xr
import pandas as pd
from pathlib import Path

location = "fauskane2"

def extract_nc_to_csv(raw_data_folder, output_folder):
    
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    nc_files = sorted(glob.glob(os.path.join(raw_data_folder, '*.nc')))
    
    if not nc_files:
        print(f"No .nc files found in {raw_data_folder}")
        return
    
    print(f"Found {len(nc_files)} .nc files")
    
    for nc_file in nc_files:
        try:
            filename = os.path.basename(nc_file)
            print(f"Processing: {filename}")
            ds = xr.open_dataset(nc_file)
            df = ds.to_dataframe().reset_index()
            
            # Save as CSV
            csv_filename = filename.replace('.nc', '.csv')
            csv_path = os.path.join(output_folder, csv_filename)
            df.to_csv(csv_path, index=False)
            
            print(f"Saved to {csv_path}")
            print(f"Shape: {df.shape}")
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    raw_data_folder = f"data/raw_data/met_norway/{location}"
    output_folder = f"data/csv_raw_data/met_norway/{location}"
    
    extract_nc_to_csv(raw_data_folder, output_folder)
    print("\nExtraction complete!")