import xarray as xr
import pandas as pd
from pathlib import Path

def extract_nc_to_csv(nc_file, output_folder):
    
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
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
    import os
    
    nc_file = "raw_data/waverys/cmems_mod_glo_wav_my_0.2deg_PT3H-i_1768390036755.nc"
    output_folder = "data/waverys"
    
    extract_nc_to_csv(nc_file, output_folder)
    print("\nExtraction complete!")