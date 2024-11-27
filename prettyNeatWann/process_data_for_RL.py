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

