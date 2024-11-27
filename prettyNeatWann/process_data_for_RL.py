import sys, math
from collections import namedtuple
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
import random
import lzma
import os
from tqdm import tqdm

DATA_DIRECTORY = "data/2023_2/"
INPUT_FILE = 'KA050_processed_10cm_5h_20230614.pkl.xz'

def load_data(source_dir, input_file, scale = None, arena_dim = None):
    data = None
    with lzma.open(os.path.join(source_dir, input_file)) as file:
        data = pd.read_pickle(file)
    return data.iloc[::int(scale)] if scale else data


def smooth_trajectories(df, sigma=1.0):
    """
    Smooth ant trajectories using Gaussian filter while preserving NaN values.
    Args:
        df: DataFrame with ant positions (MultiIndex columns with ant number and x,y coordinates)
        sigma: Standard deviation for Gaussian kernel (higher = more smoothing)
    Returns:
        DataFrame with smoothed trajectories
    """
    smoothed_df = df.copy()
    ant_numbers = df.columns.get_level_values(0).unique()
    
    for ant in ant_numbers:
        # Get x,y coordinates for this ant
        ant_data = df[ant]
        
        # Only smooth non-NaN values
        mask = ~ant_data.isna()
        if mask.any().any():  # Only process if there's valid data
            x_valid = ant_data['x'][mask['x']].values
            y_valid = ant_data['y'][mask['y']].values
            
            if len(x_valid) > 0:
                # Smooth valid values
                x_smoothed = gaussian_filter(x_valid, sigma=sigma)
                y_smoothed = gaussian_filter(y_valid, sigma=sigma)
                
                # Round to nearest pixel coordinates
                x_smoothed = np.round(x_smoothed)
                y_smoothed = np.round(y_smoothed)
                
                # Put smoothed values back
                smoothed_df.loc[mask['x'], (ant, 'x')] = x_smoothed
                smoothed_df.loc[mask['y'], (ant, 'y')] = y_smoothed
    
    return smoothed_df

def quantize_to_pixels(df):
    """
    Ensure all coordinates are properly quantized to pixel positions.
    Args:
        df: DataFrame with ant positions
    Returns:
        DataFrame with pixel-quantized positions
    """
    quantized_df = df.copy()
    ant_numbers = df.columns.get_level_values(0).unique()
    
    for ant in ant_numbers:
        # Round all non-NaN values to nearest integer
        mask = ~df[ant].isna()
        quantized_df.loc[mask[['x']], (ant, 'x')] = np.round(df.loc[mask[['x']], (ant, 'x')])
        quantized_df.loc[mask[['y']], (ant, 'y')] = np.round(df.loc[mask[['y']], (ant, 'y')])
    
    return quantized_df



# Load and clean the data
data = load_data(DATA_DIRECTORY, INPUT_FILE)

# Apply smoothing and quantization
smoothed_data = smooth_trajectories(data, sigma=1.0)
pixel_data = quantize_to_pixels(smoothed_data)

print("Original data:")
print(data.head())
print("\nSmoothed and quantized data:")
print(pixel_data.head())

# Calculate and print the difference to see the effect
print("\nMean absolute difference from original:")
diff = (data - pixel_data).abs().mean()
print(diff)


# Example output
# -----------------
# print(data.head())
# Output:
# ----------
#       0             1             2             3          ...  53      54      55      56    
#        x      y      x      y      x      y      x      y  ...   x   y   x   y   x   y   x   y
# 0  180.0  225.0  339.0  591.0  326.0  614.0  308.0  750.0  ... NaN NaN NaN NaN NaN NaN NaN NaN
# 1  180.0  225.0  340.0  592.0  325.0  614.0  308.0  750.0  ... NaN NaN NaN NaN NaN NaN NaN NaN
# 2  180.0  225.0  340.0  592.0  325.0  614.0  308.0  749.0  ... NaN NaN NaN NaN NaN NaN NaN NaN
# 3  180.0  224.0  340.0  592.0  324.0  614.0  308.0  749.0  ... NaN NaN NaN NaN NaN NaN NaN NaN
# 4  180.0  224.0  340.0  592.0  324.0  614.0  308.0  749.0  ... NaN NaN NaN NaN NaN NaN NaN NaN
# print(data[0].head())
# Output:
# ----------
# [5 rows x 114 columns]
#        x      y
# 0  180.0  225.0
# 1  180.0  225.0
# 2  180.0  225.0
# 3  180.0  224.0
# 4  180.0  224.0





# --------------------------------------
# Explanation:
#
# The first part of this file loads a dataset of positional information of individual ants (x, y coordinates
# corresponding to the pixel space of the video recordings) which has been extract and translated into the
# simulation space I am working in (typically 900x900 pixels in a pygame environment).
# I then asked ChatGPT to help me come up with a plan for processing this dataset so that I could use it in my research.
# My plan is to develop a way to use neuroevolution (probably a NEAT-type setup) to evolve neural networks that
# reproduce realistic looking collective behaviours of the ants by mimicking the real behaviours observable in the data.
# ChatGPT suggested I should adopt a behavioural cloning approach based in a Reinforcement Learning framework.
# Here is what the suggested process looks like:

# ## Step 1: Data Preprocessing
# ### A. Data Cleaning
# **Handle Missing Data:**

# Identify Missing Points: Check for any missing (𝑥,𝑦)(x,y) positions in your dataset.
# Impute or Remove: Depending on the extent, either interpolate missing positions or remove incomplete trajectories.

# **Smoothing Trajectories:**

# Purpose: Reduce noise in positional data to accurately compute derived features.
# Methods: Apply moving averages or Gaussian filters.

# ### B. Temporal Alignment
# **Consistent Sampling Rate:**
# Ensure that positional data is sampled at regular time intervals (Δt).
# Resample or interpolate if necessary.

# Code ex:
# from scipy.ndimage import gaussian_filter1d

# # Apply Gaussian smoothing to x and y positions
# data['x_smooth'] = gaussian_filter1d(data['x'], sigma=2)
# data['y_smooth'] = gaussian_filter1d(data['y'], sigma=2)

