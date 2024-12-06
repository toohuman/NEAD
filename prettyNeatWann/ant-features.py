import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass, asdict
from tqdm import tqdm
from typing import Any, Dict, List, Tuple, Optional
import numpy.typing as npt
import lzma
import pickle
import os
import time
import json
import argparse
from pathlib import Path

DATA_DIRECTORY = "data/2023_2/"
INPUT_FILE = 'KA050_processed_10cm_5h_20230614.pkl.xz'

def load_data(source_dir, input_file, scale=None, arena_dim=None, debug=False, debug_ants=5, debug_timesteps=10000):
    """
    Load data from a compressed pickle file.
    
    Args:
        source_dir: Directory containing the data file
        input_file: Name of the data file
        scale: Sampling rate for the data
        arena_dim: Arena dimensions
        debug: If True, only load subset of ants
        debug_ants: Number of ants to load in debug mode
        debug_timesteps: Number of timesteps to load in debug mode
    """
    data = None
    with lzma.open(os.path.join(source_dir, input_file)) as file:
        data = pd.read_pickle(file)
    
    if debug:
        # Limit timesteps first
        data = data.iloc[:debug_timesteps]
        # Then select first N ants
        ant_ids = sorted(list(data.columns.levels[0]))[:debug_ants]
        # Keep only the selected ant columns
        cols_to_keep = [(ant, coord) for ant in ant_ids for coord in ['x', 'y']]
        data = data[cols_to_keep]
    
    return data.iloc[::int(scale)] if scale else data


@dataclass
class TrajectoryFeatures:
    """Container for extracted trajectory features."""
    velocities: npt.NDArray[np.float64]
    accelerations: npt.NDArray[np.float64]
    angular_velocities: npt.NDArray[np.float64]
    curvatures: npt.NDArray[np.float64]
    stop_segments: List[Tuple[int, int]]  # start and end indices of stops
    move_segments: List[Tuple[int, int]]  # start and end indices of movements
    bout_durations: Dict[str, List[float]]  # durations of different behavioural bouts


class AntFeatureExtractor:
    """Extract behavioural features from ant trajectory data using vectorized operations."""
    
    def __init__(self, fps: float = 60.0, velocity_threshold: float = 0.5, max_position_change: float = 10.0):
        """
        Initialize the feature extractor.
        
        Args:
            fps: Frame rate of the data
            velocity_threshold: Threshold for determining stop/move states (units/second)
            max_position_change: Maximum allowable position change between consecutive frames (units)
        """
        self.dt = 1.0 / fps
        self.velocity_threshold = velocity_threshold
        self.max_position_change = max_position_change
    
    def extract_features(self, x: np.ndarray, y: np.ndarray) -> TrajectoryFeatures:
        """
        Extract all trajectory features from position data using vectorized operations.
        
        Args:
            x: Array of x positions
            y: Array of y positions
            
        Returns:
            TrajectoryFeatures object containing all computed features
        """
        # Find valid positions (not NaN)
        valid = ~(np.isnan(x) | np.isnan(y))
        x, y = x[valid], y[valid]
        n_points = len(x)
        
        # Calculate position changes between consecutive frames
        dx = np.diff(x)
        dy = np.diff(y)
        position_changes = np.sqrt(dx**2 + dy**2)
        
        # Identify large jumps (treating them as breaks in the trajectory)
        jump_indices = np.where(position_changes > self.max_position_change)[0]
        
        # Insert NaN values at jump points to break the trajectory
        x_segmented = np.insert(x, jump_indices + 1, np.nan)
        y_segmented = np.insert(y, jump_indices + 1, np.nan)
        
        # Recalculate valid positions after inserting breaks
        valid = ~(np.isnan(x_segmented) | np.isnan(y_segmented))
        x_clean = x_segmented[valid]
        y_clean = y_segmented[valid]
        n_points = len(x_clean)
        
        # Pre-allocate arrays for kinematic features
        velocities = np.zeros((n_points, 2), dtype=np.float64)
        accelerations = np.zeros((n_points, 2), dtype=np.float64)
        angular_velocities = np.zeros(n_points, dtype=np.float64)
        curvatures = np.zeros(n_points, dtype=np.float64)
        
        # Compute velocities with bounded gradient
        dx = np.gradient(x_clean, self.dt)
        dy = np.gradient(y_clean, self.dt)
        
        # Bound the velocities by max_position_change/dt
        max_velocity = self.max_position_change / self.dt
        dx = np.clip(dx, -max_velocity, max_velocity)
        dy = np.clip(dy, -max_velocity, max_velocity)
        
        velocities[:, 0] = dx
        velocities[:, 1] = dy
        velocity_mag = np.sqrt(dx**2 + dy**2)
        
        # Compute accelerations
        accelerations[:, 0] = np.gradient(dx, self.dt)
        accelerations[:, 1] = np.gradient(dy, self.dt)
        
        # Compute angular velocity
        angles = np.arctan2(dy, dx)
        angular_velocities = np.gradient(np.unwrap(angles), self.dt)
        
        # Compute curvature
        ddx = np.gradient(dx, self.dt)
        ddy = np.gradient(dy, self.dt)
        denominator = (dx * dx + dy * dy) ** 1.5
        valid_denom = denominator > 1e-10
        curvatures[valid_denom] = np.abs(
            dx[valid_denom] * ddy[valid_denom] - 
            dy[valid_denom] * ddx[valid_denom]
        ) / denominator[valid_denom]
        
        # Identify movement bouts more efficiently
        is_moving = velocity_mag > self.velocity_threshold
        state_changes = np.diff(is_moving.astype(int))
        move_starts = np.where(state_changes == 1)[0] + 1
        move_ends = np.where(state_changes == -1)[0] + 1
        
        # Handle edge cases vectorially
        if is_moving[0]:
            move_starts = np.insert(move_starts, 0, 0)
        if is_moving[-1]:
            move_ends = np.append(move_ends, len(is_moving))
            
        move_segments = list(zip(move_starts, move_ends))
        
        # Calculate stop segments vectorially
        if len(move_segments) > 0:
            stop_starts = np.array([0] + [end for _, end in move_segments[:-1]])
            stop_ends = np.array([start for start, _ in move_segments] + [len(is_moving)])
            stop_segments = list(zip(stop_starts, stop_ends))
        else:
            stop_segments = [(0, len(is_moving))]
        
        # Calculate bout durations vectorially
        bout_durations = {
            'move': self.dt * np.array([end - start for start, end in move_segments]),
            'stop': self.dt * np.array([end - start for start, end in stop_segments])
        }
        
        return TrajectoryFeatures(
            velocities=velocities,
            accelerations=accelerations,
            angular_velocities=angular_velocities,
            curvatures=curvatures,
            stop_segments=stop_segments,
            move_segments=move_segments,
            bout_durations=bout_durations
        )


class SocialContextExtractor:
    """Extract features related to social interactions using vectorized operations."""
    
    def __init__(self, n_sectors: int = 8, max_distance: float = 100.0):
        """
        Initialize the social context extractor.
        
        Args:
            n_sectors: Number of angular sectors for density calculation
            max_distance: Maximum distance to consider for neighbour interactions
        """
        self.n_sectors = n_sectors
        self.max_distance = max_distance
        self.sector_angles = np.linspace(0, 2*np.pi, n_sectors+1)
    
    def compute_local_density(self, focal_x: float, focal_y: float, 
                            neighbour_x: np.ndarray, neighbour_y: np.ndarray) -> np.ndarray:
        """
        Compute ant density in different sectors around focal ant using vectorized operations.
        
        Args:
            focal_x: x-coordinate of focal ant
            focal_y: y-coordinate of focal ant
            neighbour_x: Array of neighbour x-coordinates
            neighbour_y: Array of neighbour y-coordinates
            
        Returns:
            Array of ant densities in each sector
        """
        # Vectorized relative position calculation
        rel_positions = np.column_stack([
            neighbour_x - focal_x,
            neighbour_y - focal_y
        ])
        
        # Vectorized polar coordinate conversion
        distances = np.linalg.norm(rel_positions, axis=1)
        angles = np.arctan2(rel_positions[:, 1], rel_positions[:, 0]) % (2*np.pi)
        
        # Pre-allocate density array
        densities = np.zeros(self.n_sectors, dtype=np.float64)
        
        # Vectorized sector counting
        in_range = distances <= self.max_distance
        valid_angles = angles[in_range]
        
        # Use numpy's histogram function for efficient binning
        densities, _ = np.histogram(
            valid_angles, 
            bins=self.sector_angles,
            range=(0, 2*np.pi)
        )
        
        return densities
    
    def compute_nearest_neighbour_stats(self, focal_x: float, focal_y: float,
                                     neighbour_x: np.ndarray, neighbour_y: np.ndarray,
                                     k: int = 3) -> Dict[str, float]:
        """Compute statistics about k nearest neighbours."""
        distances = np.sqrt((neighbour_x - focal_x)**2 + (neighbour_y - focal_y)**2)
        sorted_distances = np.sort(distances)
        
        return {
            f'nn_dist_{i+1}': dist for i, dist in enumerate(sorted_distances[:k])
        }

def process_ant_data(data: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
    """
    Process all ant trajectories using vectorized operations and efficient memory management.
    
    Args:
        data: DataFrame with MultiIndex columns (ant_id, coordinate)
    
    Returns:
        Dictionary mapping ant IDs to their extracted features
    """
    feature_extractor = AntFeatureExtractor()
    social_extractor = SocialContextExtractor()
    
    results = {}
    ant_ids = data.columns.levels[0]
    n_timesteps = len(data)
    
    # Get the actual ant IDs from the filtered data
    ant_ids = sorted(list(set(idx[0] for idx in data.columns)))
    n_ants = len(ant_ids)
    
    # Pre-allocate arrays for positions
    positions = np.zeros((n_ants, n_timesteps, 2))
    
    # Extract positions more efficiently
    for i, ant_id in enumerate(ant_ids):
        try:
            positions[i, :, 0] = data[ant_id, 'x'].values
            positions[i, :, 1] = data[ant_id, 'y'].values
        except KeyError:
            # Handle missing columns gracefully
            positions[i, :, :] = np.nan
    
    for i, ant_id in tqdm(enumerate(ant_ids), 
                         desc="Processing ants",
                         total=len(ant_ids),
                         bar_format='{desc}: {n}/{total} [{elapsed}<{remaining}, {rate_fmt}]'):
        # Extract trajectory features
        traj_features = feature_extractor.extract_features(
            positions[i, :, 0],
            positions[i, :, 1]
        )
        
        # Process all social features
        social_features = []
        
        for t in tqdm(range(n_timesteps), desc=f"Processing timesteps for ant {ant_id}", leave=False):
            if np.isnan(positions[i, t, 0]):
                continue
            
            # Get other ants' positions efficiently
            other_positions = np.delete(positions[:, t, :], i, axis=0)
            valid_mask = ~np.isnan(other_positions).any(axis=1)
            if not np.any(valid_mask):
                continue
            
            other_positions = other_positions[valid_mask]
            
            densities = social_extractor.compute_local_density(
                positions[i, t, 0],
                positions[i, t, 1],
                other_positions[:, 0],
                other_positions[:, 1]
            )
            
            nn_stats = social_extractor.compute_nearest_neighbour_stats(
                positions[i, t, 0],
                positions[i, t, 1],
                other_positions[:, 0],
                other_positions[:, 1]
            )
            
            social_features.append({
                'densities': densities,
                'nn_stats': nn_stats
            })
        
        results[ant_id] = {
            'trajectory_features': traj_features,
            'social_features': social_features
        }
    
    return results


def analyze_colony_clustering(data, eps_mm=10, min_samples=3):
    """
    Analyze clustering behaviour of the colony over time using DBSCAN.
    
    Args:
        data: DataFrame with MultiIndex columns (ant_id, coordinate)
        eps_mm: Clustering radius in millimeters
        min_samples: Minimum samples to form a cluster
    
    Returns:
        Dictionary containing clustering statistics over time
    """
    from sklearn.cluster import DBSCAN
    
    PIXELS_PER_MM = 8.1
    eps_pixels = eps_mm * PIXELS_PER_MM
    
    clustering_stats = {
        'n_clusters': [],
        'cluster_sizes': [],
        'isolated_ants': [],
        'mean_cluster_density': [],
        'positions': [],  # Store positions for visualization
        'labels': []     # Store cluster labels for visualization
    }
    
    # Get the actual ant IDs from the filtered data
    ant_ids = sorted(list(set(idx[0] for idx in data.columns)))
    
    # Analyze each timestep
    for t in tqdm(range(len(data)), desc="Analyzing clustering behaviour"):
        # Get positions of all ants at this timestep
        positions = []
        tracked_ant_ids = []  # Track which ant is at each position
        
        for ant_id in ant_ids:
            x = data.loc[t, (ant_id, 'x')]
            y = data.loc[t, (ant_id, 'y')]
            if not (np.isnan(x) or np.isnan(y)):
                positions.append([x, y])
                ant_ids.append(ant_id)
        
        if len(positions) < min_samples:
            clustering_stats['positions'].append([])
            clustering_stats['labels'].append([])
            continue
            
        positions = np.array(positions)
        
        # Perform DBSCAN clustering
        clustering = DBSCAN(eps=eps_pixels, min_samples=min_samples).fit(positions)
        labels = clustering.labels_
        
        # Store positions and labels for visualization
        clustering_stats['positions'].append(positions.tolist())
        clustering_stats['labels'].append(labels.tolist())
        
        # Count unique clusters (excluding noise points labeled as -1)
        unique_clusters = len(set(labels[labels >= 0]))
        clustering_stats['n_clusters'].append(unique_clusters)
        
        # Count ants per cluster
        if unique_clusters > 0:
            cluster_sizes = [np.sum(labels == i) for i in range(unique_clusters)]
            clustering_stats['cluster_sizes'].append(cluster_sizes)
            
            # Calculate cluster densities (ants per unit area)
            cluster_areas = [np.pi * (eps_pixels ** 2) for _ in range(unique_clusters)]
            densities = [size/area for size, area in zip(cluster_sizes, cluster_areas)]
            clustering_stats['mean_cluster_density'].append(np.mean(densities))
        else:
            clustering_stats['cluster_sizes'].append([])
            clustering_stats['mean_cluster_density'].append(0)
        
        # Count isolated ants
        clustering_stats['isolated_ants'].append(np.sum(labels == -1))
    
    return clustering_stats


def save_processed_data(processed_data, processed_dir):
    """Save processed ant data to directory structure."""
    print("\nSaving processed data...")
    # Save specific features to their respective directories
    for ant_id, ant_data in processed_data.items():
        # Save kinematic features
        kinematic_data = {
            'velocities': ant_data['trajectory_features'].velocities,
            'accelerations': ant_data['trajectory_features'].accelerations,
            'angular_velocities': ant_data['trajectory_features'].angular_velocities,
            'curvatures': ant_data['trajectory_features'].curvatures
        }
        with open(os.path.join(processed_dir, "kinematic", f"ant_{ant_id}_kinematic.pkl"), 'wb') as f:
            pickle.dump(kinematic_data, f)
        
        # Save behavioural features
        behavioural_data = {
            'stop_segments': ant_data['trajectory_features'].stop_segments,
            'move_segments': ant_data['trajectory_features'].move_segments,
            'bout_durations': ant_data['trajectory_features'].bout_durations
        }
        with open(os.path.join(processed_dir, "behavioural", f"ant_{ant_id}_behavioural.pkl"), 'wb') as f:
            pickle.dump(behavioural_data, f)
        
        # Save social features
        with open(os.path.join(processed_dir, "social", f"ant_{ant_id}_social.pkl"), 'wb') as f:
            pickle.dump(ant_data['social_features'], f)
    
    print("Data saved successfully!")

def load_processed_data(processed_dir):
    """Load previously processed ant data from directory structure."""
    print("Loading previously processed data...")
    processed_data = {}
    
    try:
        # Get list of all ant IDs from the files in behavioural directory
        behavioural_files = [f for f in os.listdir(os.path.join(processed_dir, "behavioural")) 
                           if f.startswith("ant_") and f.endswith("_behavioural.pkl")]
        ant_ids = sorted([f.split("_")[1] for f in behavioural_files])
        
        print(f"Found ant IDs: {ant_ids}")
        
        # Load data for each ant
        for ant_id in ant_ids:
            print(f"\nProcessing ant {ant_id}")
            try:
                # Initialize the data structure for this ant
                processed_data[ant_id] = {
                    'trajectory_features': TrajectoryFeatures(
                        velocities=None,
                        accelerations=None,
                        angular_velocities=None,
                        curvatures=None,
                        stop_segments=None,
                        move_segments=None,
                        bout_durations=None
                    ),
                    'social_features': None
                }
                
                # Load and verify kinematic features
                kinematic_path = os.path.join(processed_dir, "kinematic", f"ant_{ant_id}_kinematic.pkl")
                print(f"Loading kinematic data from {kinematic_path}")
                with open(kinematic_path, 'rb') as f:
                    kinematic_data = pickle.load(f)
                
                # Load and verify behavioural features 
                behavioural_path = os.path.join(processed_dir, "behavioural", f"ant_{ant_id}_behavioural.pkl")
                print(f"Loading behavioural data from {behavioural_path}")
                with open(behavioural_path, 'rb') as f:
                    behavioural_data = pickle.load(f)
                
                # Load and verify social features
                social_path = os.path.join(processed_dir, "social", f"ant_{ant_id}_social.pkl")
                print(f"Loading social data from {social_path}")
                with open(social_path, 'rb') as f:
                    social_data = pickle.load(f)
                
                # Update processed_data with verified data
                processed_data[ant_id]['trajectory_features'].velocities = kinematic_data['velocities']
                processed_data[ant_id]['trajectory_features'].accelerations = kinematic_data['accelerations']
                processed_data[ant_id]['trajectory_features'].angular_velocities = kinematic_data['angular_velocities']
                processed_data[ant_id]['trajectory_features'].curvatures = kinematic_data['curvatures']
                
                processed_data[ant_id]['trajectory_features'].stop_segments = behavioural_data['stop_segments']
                processed_data[ant_id]['trajectory_features'].move_segments = behavioural_data['move_segments']
                processed_data[ant_id]['trajectory_features'].bout_durations = behavioural_data['bout_durations']
                
                processed_data[ant_id]['social_features'] = social_data
            except Exception as e:
                print(f"Error processing ant {ant_id}: {str(e)}")
                continue
        
        print("Data loaded successfully!")
        return processed_data
    except FileNotFoundError:
        print("No processed data found. Please run with --save first.")
        exit(1)
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        exit(1)

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process ant trajectory data')
    parser.add_argument('--debug', action='store_true', 
                      help='Run in debug mode (process subset of ants and timesteps)')
    parser.add_argument('--ants', type=int, default=5,
                      help='Number of ants to process in debug mode (default: 5)')
    parser.add_argument('--timesteps', type=int, default=10000,
                      help='Number of timesteps to process in debug mode (default: 10000)')
    parser.add_argument('--save', action='store_true',
                      help='Save processed data to processed_data directory')
    parser.add_argument('--load', action='store_true',
                      help='Load previously processed data instead of processing raw data')
    args = parser.parse_args()

    # Constants for biological sanity checks
    PIXELS_PER_MM = 8.1  # Based on 900px = 100mm arena diameter
    ARENA_CENTER = np.array([450, 450])
    ARENA_RADIUS = 405  # pixels
    MAX_EXPECTED_VELOCITY = 50  # mm/s
    
    # Create processed data directories if they don't exist
    processed_dir = os.path.join(DATA_DIRECTORY, "processed_data")
    for subdir in ["behavioural", "kinematic", "social"]:
        os.makedirs(os.path.join(processed_dir, subdir), exist_ok=True)

    if args.load:
        processed_data = load_processed_data(processed_dir)
    else:
        print("Loading raw data...")
        if args.debug:
            print(f"DEBUG MODE: Processing first {args.ants} ants and {args.timesteps} timesteps")
        data = load_data(DATA_DIRECTORY, INPUT_FILE, debug=args.debug, 
                        debug_ants=args.ants, debug_timesteps=args.timesteps)
        
        print("\nData Overview:")
        print(f"Total timesteps: {len(data):,}")
        print(f"Number of ants: {len(data.columns.levels[0])}")
        print(f"Recording duration: {len(data)/60:.1f} minutes")
        
        print("\nProcessing ant trajectories...")
        start_time = time.time()
        processed_data = process_ant_data(data)
        print(f"Processing completed in {(time.time() - start_time)/60:.1f} minutes")
        
        if args.save:
            save_processed_data(processed_data, processed_dir)
    
    # --- Analysis code goes here ---

    
