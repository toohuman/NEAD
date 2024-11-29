import sys, math
from collections import namedtuple
import numpy as np
import pandas as pd
from filterpy.kalman import KalmanFilter
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt  # Corrected import
import lzma
import os
from tqdm import tqdm

DATA_DIRECTORY = "data/2023_2/"
INPUT_FILE = 'KA050_processed_10cm_5h_20230614.pkl.xz'

def load_data(source_dir, input_file, scale=None, arena_dim=None):
    """Load data from a compressed pickle file."""
    data = None
    with lzma.open(os.path.join(source_dir, input_file)) as file:
        data = pd.read_pickle(file)
    return data.iloc[::int(scale)] if scale else data

def smooth_and_round(df, sigma=2, threshold=0.5):
    """
    Apply Gaussian smoothing to 'x' and 'y' columns and round them to integers.
    
    Parameters:
    - df: pandas DataFrame with MultiIndex columns (entity, 'x'/'y')
    - sigma: Standard deviation for Gaussian kernel
    - threshold: Threshold for rounding (currently not used for conditional rounding)
    
    Returns:
    - smoothed_df: DataFrame with smoothed and rounded 'x' and 'y' columns
    """
    smoothed_df = df.copy()
    for col in df.columns:
        if col[1] in ['x', 'y']:
            # Apply Gaussian smoothing
            smoothed_values = gaussian_filter1d(df[col], sigma=sigma)
            
            # Keep NaN values and round only non-NaN values
            mask = np.isnan(smoothed_values)
            rounded_values = np.copy(smoothed_values)
            rounded_values[~mask] = np.round(smoothed_values[~mask])
            
            # Calculate the difference (optional, currently not used for conditional logic)
            diff = np.abs(smoothed_values - rounded_values)
            
            # Assign rounded values back, preserving NaN
            smoothed_df[col] = rounded_values
    return smoothed_df

def apply_kalman_filter(df, entity, delta_t=0.1, measurement_noise=5.0, process_noise=0.1):
    """
    Apply a Kalman Filter to smooth 'x' and 'y' positions for a given entity.
    
    Parameters:
    - df: pandas DataFrame with MultiIndex columns (entity, 'x'/'y')
    - entity: Entity ID (e.g., 0, 1, 2, ...)
    - delta_t: Time step interval
    - measurement_noise: Measurement noise parameter (R)
    - process_noise: Process noise parameter (Q)
    
    Returns:
    - x_smoothed: Numpy array of smoothed 'x' positions
    - y_smoothed: Numpy array of smoothed 'y' positions
    - states: Dictionary containing filter states for analysis
    """
    kf = KalmanFilter(dim_x=4, dim_z=2)
    
    # State transition matrix
    kf.F = np.array([[1, 0, delta_t, 0],
                     [0, 1, 0, delta_t],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])
    
    # Measurement matrix
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])
    
    # Measurement noise
    kf.R = np.eye(2) * measurement_noise
    
    # Initial state covariance
    kf.P *= 1000.
    
    # Process noise - tuned for ant motion
    kf.Q = np.eye(4) * process_noise
    kf.Q[2:, 2:] *= 10  # Higher noise for velocity components
    
    measurements = df[(entity, 'x')].values, df[(entity, 'y')].values
    measurements = np.column_stack(measurements)
    
    # Store all states for analysis
    states = {
        'positions': [],
        'velocities': [],
        'uncertainties': [],
        'measurement_residuals': []
    }
    
    last_valid_state = None
    
    for z in measurements:
        # Handle missing measurements (NaN values)
        if np.any(np.isnan(z)):
            if last_valid_state is not None:
                # Only predict if we have a valid previous state
                kf.predict()
                state = (kf.x[0], kf.x[1])
            else:
                # Skip update if no valid previous state
                state = (np.nan, np.nan)
        else:
            kf.predict()
            kf.update(z)
            state = (kf.x[0], kf.x[1])
            last_valid_state = state
            
            # Store state information
            states['velocities'].append((kf.x[2], kf.x[3]))
            states['uncertainties'].append(np.diag(kf.P))
            states['measurement_residuals'].append(kf.y)
        
        states['positions'].append(state)
    
    # Convert to array while explicitly handling NaN values
    positions = np.array([(float(p[0]) if p[0] is not None else np.nan,
                          float(p[1]) if p[1] is not None else np.nan)
                         for p in states['positions']], dtype=float)
    x_smoothed = positions[:, 0]
    y_smoothed = positions[:, 1]
    
    return x_smoothed, y_smoothed, states

def limit_step_size(x, y, max_step=5):
    """
    Limit the step size between consecutive positions to prevent large jumps.
    Preserves NaN values in the output.
    """
    # Convert inputs to numpy arrays and ensure they're 1D
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    
    x_limited = np.full_like(x, np.nan)
    y_limited = np.full_like(y, np.nan)
    
    # Find first non-NaN point
    valid_mask = ~(np.isnan(x) | np.isnan(y))
    if not np.any(valid_mask):
        return x_limited, y_limited
        
    # Process points
    last_valid_x = None
    last_valid_y = None
    
    for i in range(len(x)):
        if np.isnan(x[i]) or np.isnan(y[i]):
            continue
            
        if last_valid_x is None:
            x_limited[i] = x[i]
            y_limited[i] = y[i]
            last_valid_x = x[i]
            last_valid_y = y[i]
            continue
            
        dx = x[i] - last_valid_x
        dy = y[i] - last_valid_y
        distance = np.sqrt(dx**2 + dy**2)
        
        if distance > max_step:
            scale = max_step / distance
            dx = dx * scale
            dy = dy * scale
            new_x = last_valid_x + dx
            new_y = last_valid_y + dy
            x_limited[i] = new_x
            y_limited[i] = new_y
        else:
            x_limited[i] = x[i]
            y_limited[i] = y[i]
            
        last_valid_x = x_limited[i]
        last_valid_y = y_limited[i]
    
    return x_limited, y_limited

def smooth_entity(df, entity, delta_t=0.1, max_step=5, measurement_noise=5.0, process_noise=0.1):
    """
    Apply Kalman Filter and step size limiting to smooth and round positions for an entity.
    
    Parameters:
    - df: pandas DataFrame with MultiIndex columns (entity, 'x'/'y')
    - entity: Entity ID
    - delta_t: Time step interval
    - max_step: Maximum allowed step size
    - measurement_noise: Measurement noise parameter for Kalman filter
    - process_noise: Process noise parameter for Kalman filter
    
    Returns:
    - x_limited: Numpy array of smoothed and rounded 'x' positions
    - y_limited: Numpy array of smoothed and rounded 'y' positions
    - filter_states: Dictionary containing Kalman filter states for analysis
    """
    x_kf, y_kf, filter_states = apply_kalman_filter(
        df, entity, delta_t, 
        measurement_noise=measurement_noise,
        process_noise=process_noise
    )
    x_limited, y_limited = limit_step_size(x_kf, y_kf, max_step)
    return x_limited, y_limited, filter_states

def calculate_mean_absolute_difference(original_df, processed_df, entities):
    """
    Calculate the mean absolute difference between original and processed DataFrames.
    
    Parameters:
    - original_df: Original pandas DataFrame with MultiIndex columns (entity, 'x'/'y')
    - processed_df: Processed pandas DataFrame with MultiIndex columns (entity, 'x'/'y')
    - entities: List of entity IDs
    
    Returns:
    - mean_absolute_diff: pandas Series with mean absolute difference per column
    - overall_mean_absolute_diff: Float representing overall mean absolute difference
    """
    columns_to_compare = []
    for entity in entities:
        columns_to_compare.extend([
            (entity, 'x'),
            (entity, 'y')
        ])

    original_positions = original_df[columns_to_compare]
    processed_positions = processed_df[columns_to_compare]

    # Ensure numeric types
    original_positions = original_positions.apply(pd.to_numeric, errors='coerce')
    processed_positions = processed_positions.apply(pd.to_numeric, errors='coerce')

    # Handle NaNs by filling with 0 (or choose another method if appropriate)
    difference = (original_positions - processed_positions).abs().fillna(0)

    # Compute mean absolute difference per column
    mean_absolute_diff = difference.mean()
    
    # Compute overall mean absolute difference
    overall_mean_absolute_diff = mean_absolute_diff.mean()
    
    return mean_absolute_diff, overall_mean_absolute_diff

# ----------------------------
# Main Processing Workflow
# ----------------------------

# Step 1: Load and clean the data
data = load_data(DATA_DIRECTORY, INPUT_FILE)

# Step 2: Apply smoothing and quantization
smoothed_data = smooth_and_round(data)

print("Original data:")
print(data.head())
print("\nSmoothed data:")
print(smoothed_data.head())

# Step 3: Initialize smoothed_and_rounded_data
smoothed_and_rounded_data = smoothed_data.copy()

# Step 4: Apply smoothing and rounding to all entities
entities = data.columns.levels[0]  # Assuming first level is entity ID
filter_states_by_entity = {}

for entity in tqdm(entities, desc='Smoothing Entities'):
    x_smoothed, y_smoothed, filter_states = smooth_entity(
        smoothed_and_rounded_data, 
        entity, 
        delta_t=0.1, 
        max_step=5,
        measurement_noise=5.0,  # Tune these parameters based on your data
        process_noise=0.1
    )
    smoothed_and_rounded_data[(entity, 'x')] = x_smoothed
    smoothed_and_rounded_data[(entity, 'y')] = y_smoothed
    filter_states_by_entity[entity] = filter_states

# Optional: Analyze Kalman filter performance
print("\nAnalyzing Kalman Filter Performance:")
for entity in entities:
    states = filter_states_by_entity[entity]
    if states['velocities']:  # Check if we have data
        mean_velocity = np.mean(states['velocities'], axis=0)
        mean_uncertainty = np.mean(states['uncertainties'], axis=0)
        print(f"\nEntity {entity}:")
        print(f"Mean velocity (x,y): ({mean_velocity[0]:.2f}, {mean_velocity[1]:.2f})")
        print(f"Mean position uncertainty: {mean_uncertainty[0]:.2f}, {mean_uncertainty[1]:.2f}")

# Step 5: Calculate Mean Absolute Difference
mean_diff_per_column, overall_mean_diff = calculate_mean_absolute_difference(data, smoothed_and_rounded_data, entities)
print("\nMean Absolute Difference per Column:")
print(mean_diff_per_column)
print(f"\nOverall Mean Absolute Difference: {overall_mean_diff:.2f} pixels")

# Step 6: Example Visualization for Entity 0
entity = 0
plt.figure(figsize=(10, 8))
plt.plot(smoothed_data[(entity, 'x')], smoothed_data[(entity, 'y')], label='Original Smoothed', alpha=0.5)
plt.plot(smoothed_and_rounded_data[(entity, 'x')], smoothed_and_rounded_data[(entity, 'y')], label='Smoothed & Rounded', alpha=0.8)
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title(f'Smoothed and Rounded Trajectory for Entity {entity}')
plt.legend()
plt.show()


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

# Raw Dataset, prior to any processing (notice the multi-indexed columns)
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
