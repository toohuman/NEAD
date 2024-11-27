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

def handle_missing_data(df, max_gap_seconds=0.5, fps=60):
    """
    Handle missing data in ant trajectories.
    Args:
        df: DataFrame with ant positions (MultiIndex columns with ant number and x,y coordinates)
        max_gap_seconds: Maximum gap in seconds to interpolate across
        fps: Frames per second of the recording
    Returns:
        Cleaned DataFrame with interpolated values
    """
    max_gap = int(max_gap_seconds * fps)  # Convert seconds to frames
    cleaned_df = df.copy()
    
    # Get unique ant numbers from first level of MultiIndex
    ant_numbers = df.columns.get_level_values(0).unique()
    
    for ant in ant_numbers:
        # Get x,y coordinates for this ant
        ant_data = df[ant]
        
        # Only process if there are any NaN values
        if ant_data.isna().any().any():
            # Get mask of NaN values (True where either x or y is NaN)
            mask = ant_data.isna().any(axis=1)
            gap_starts = np.where(mask[1:].values & ~mask[:-1].values)[0] + 1
            gap_ends = np.where(~mask[1:].values & mask[:-1].values)[0] + 1
            
            if len(gap_starts) > 0 and len(gap_ends) > 0:
                # Handle case where first frame is NaN
                if mask.iloc[0]:
                    gap_starts = np.insert(gap_starts, 0, 0)
                # Handle case where last frame is NaN
                if mask.iloc[-1]:
                    gap_ends = np.append(gap_ends, len(mask))
                
                for start, end in zip(gap_starts, gap_ends):
                    gap_length = end - start
                    if gap_length <= max_gap:
                        # Interpolate if gap is small enough
                        cleaned_df.loc[start:end-1, ant] = \
                            df.loc[start:end-1, ant].interpolate(method='linear')
                    else:
                        # Set to NaN if gap is too large
                        cleaned_df.loc[start:end-1, ant] = np.nan

    return cleaned_df

# Load and clean the data
data = load_data(DATA_DIRECTORY, INPUT_FILE)
print(data.head())
# Output:
# ----------
#       0             1             2             3          ...  53      54      55      56    
#        x      y      x      y      x      y      x      y  ...   x   y   x   y   x   y   x   y
# 0  180.0  225.0  339.0  591.0  326.0  614.0  308.0  750.0  ... NaN NaN NaN NaN NaN NaN NaN NaN
# 1  180.0  225.0  340.0  592.0  325.0  614.0  308.0  750.0  ... NaN NaN NaN NaN NaN NaN NaN NaN
# 2  180.0  225.0  340.0  592.0  325.0  614.0  308.0  749.0  ... NaN NaN NaN NaN NaN NaN NaN NaN
# 3  180.0  224.0  340.0  592.0  324.0  614.0  308.0  749.0  ... NaN NaN NaN NaN NaN NaN NaN NaN
# 4  180.0  224.0  340.0  592.0  324.0  614.0  308.0  749.0  ... NaN NaN NaN NaN NaN NaN NaN NaN

print(data[0].head())
# Output:
# ----------
# [5 rows x 114 columns]
#        x      y
# 0  180.0  225.0
# 1  180.0  225.0
# 2  180.0  225.0
# 3  180.0  224.0
# 4  180.0  224.0


# Print diagnostics about NaN values
print(f"NaN values before cleaning: {data.isna().sum().sum()}")
data = handle_missing_data(data)
print(f"NaN values after cleaning: {data.isna().sum().sum()}")

# Print sample of cleaned data
print("\nSample of cleaned data:")
print(data.head())



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

