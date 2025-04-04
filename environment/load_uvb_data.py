# environment/load_uvb_data.py

import numpy as np
import os

def load_uvb_ascii(file_path, crop_start=(100, 100), crop_size=(10, 10)):
    """
    Load and crop UVB3 ASCII data for a custom grid.
    
    Args:
        file_path (str): Path to the .asc file.
        crop_start (tuple): (row_start, col_start) for cropping.
        crop_size (tuple): (num_rows, num_cols) for the grid.
    
    Returns:
        np.ndarray: Normalized UVB values in selected region.
    """
    with open(file_path, 'r') as f:
        # Read header
        header = {}
        for _ in range(6):
            key, value = f.readline().strip().split()
            header[key] = float(value)
        
        nodata_val = header['NODATA_value']
        
        # Read the full grid
        data = np.loadtxt(f)
        data[data == nodata_val] = np.nan  # Replace NODATA with NaN

        # Normalize
        valid_data = data[~np.isnan(data)]
        norm_data = (data - np.nanmin(valid_data)) / (np.nanmax(valid_data) - np.nanmin(valid_data))

        # Crop a region (e.g., 10x10 grid from row 100 to 110, col 100 to 110)
        row_start, col_start = crop_start
        row_end, col_end = row_start + crop_size[0], col_start + crop_size[1]
        cropped = norm_data[row_start:row_end, col_start:col_end]

        return cropped
