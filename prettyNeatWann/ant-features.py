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

# Constants for biological sanity checks
PIXELS_PER_MM = 8.64  # Based on 864px = 100mm arena diameter (accounting for 2% margin)
ARENA_CENTER = np.array([900/2.0, 900/2.0])
ARENA_RADIUS = min(900, 900)/2.0 - min(900, 900) * 0.02
MAX_EXPECTED_VELOCITY = 50  # mm/s

DATA_DIRECTORY = "data/2023_2/"
INPUT_FILE = 'KA050_processed_10cm_5h_20230614.pkl.xz'
SCALE = 2

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
    
    def __init__(self, fps: float = 60.0, velocity_threshold: float = 0.5, max_position_change: float = 10.0, max_velocity_pixels: float = 100 * PIXELS_PER_MM):
        """
        Initialize the feature extractor.
        
        Args:
            fps: Frame rate of the data (will be adjusted by SCALE if provided)
            velocity_threshold: Threshold for determining stop/move states (units/second)
            max_position_change: Maximum allowable position change between consecutive frames (units)
            max_velocity_pixels: Maximum allowable velocity in pixels/second
        """
        # Adjust fps if SCALE is provided
        self.fps = fps / SCALE if SCALE else fps
        self.dt = 1.0 / self.fps
        self.velocity_threshold = velocity_threshold
        self.max_position_change = max_position_change * (SCALE if SCALE else 1)  # Scale position change threshold
        self.max_velocity_pixels = max_velocity_pixels
    
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
        
        # Compute velocities
        dx = np.gradient(x_clean, self.dt)
        dy = np.gradient(y_clean, self.dt)
        
        # Calculate velocity magnitude and identify unrealistic velocities
        velocity_magnitude = np.sqrt(dx**2 + dy**2)
        large_velocity_jumps = velocity_magnitude > self.max_velocity_pixels
        
        # Replace unrealistic velocities with NaN
        dx[large_velocity_jumps] = np.nan
        dy[large_velocity_jumps] = np.nan
        
        # Optional: Log instances of large velocity jumps for debugging
        # if np.any(large_velocity_jumps):
        #     jump_locations = np.where(large_velocity_jumps)[0]
        #     for idx in jump_locations:
        #         start_idx = max(0, idx - 2)
        #         end_idx = min(len(x_clean), idx + 3)
        #         print(f"\nLarge velocity detected at index {idx}:")
        #         print(f"Velocity: {velocity_magnitude[idx]:.2f} pixels/second")
        #         print("Position data around the jump:")
        #         for i in range(start_idx, end_idx):
        #             print(f"Index {i}: x = {x_clean[i]:.2f}, y = {y_clean[i]:.2f}")
        
        velocities[:, 0] = dx
        velocities[:, 1] = dy
        
        # Recompute velocity magnitude after filtering
        velocity_mag = np.sqrt(dx**2 + dy**2)
        
        # Compute accelerations (using masked arrays to handle NaN values)
        masked_dx = np.ma.masked_invalid(dx)
        masked_dy = np.ma.masked_invalid(dy)
        accelerations[:, 0] = np.gradient(masked_dx, self.dt)
        accelerations[:, 1] = np.gradient(masked_dy, self.dt)
        
        # Compute angular velocity (accounting for NaN values)
        angles = np.arctan2(masked_dy, masked_dx)
        angular_velocities = np.gradient(np.ma.masked_invalid(np.unwrap(angles)), self.dt)
        
        # Compute curvature (handling NaN values)
        ddx = np.gradient(masked_dx, self.dt)
        ddy = np.gradient(masked_dy, self.dt)
        denominator = (masked_dx * masked_dx + masked_dy * masked_dy) ** 1.5
        valid_denom = ~denominator.mask & (denominator > 1e-10)
        curvatures[valid_denom] = np.abs(
            masked_dx[valid_denom] * ddy[valid_denom] - 
            masked_dy[valid_denom] * ddx[valid_denom]
        ) / denominator[valid_denom]
        
        # Identify movement bouts using filtered velocity
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
    # Adjust fps based on SCALE
    base_fps = 60.0
    effective_fps = base_fps / SCALE if SCALE else base_fps
    
    feature_extractor = AntFeatureExtractor(fps=effective_fps)
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


def analyse_colony_clustering(data, eps_mm=10, min_samples=3, max_centroid_distance=50):
    """
    Analyse clustering behaviour of the colony over time using DBSCAN with consistent cluster tracking.
    
    Args:
        data: DataFrame with MultiIndex columns (ant_id, coordinate)
        eps_mm: Clustering radius in millimeters
        min_samples: Minimum samples to form a cluster
        max_centroid_distance: Maximum distance between centroids to consider it the same cluster
    
    Returns:
        Dictionary containing clustering statistics over time
    """
    from sklearn.cluster import DBSCAN
    
    eps_pixels = eps_mm * PIXELS_PER_MM
    
    clustering_stats = {
        'n_clusters': [],
        'cluster_sizes': [],
        'isolated_ants': [],
        'mean_cluster_density': [],
        'positions': [],
        'labels': [],
        'cluster_ids': [],  # Store consistent cluster IDs
        'centroids': []     # Store cluster centroids
    }
    
    # Get the actual ant IDs from the filtered data
    ant_ids = sorted(list(set(idx[0] for idx in data.columns)))
    
    # Get actual frame indices from the data
    frame_indices = data.index.values
    
    # Analyse each timestep
    for t in tqdm(frame_indices, desc="Analysing clustering behaviour"):
        # Get positions of all ants at this timestep
        positions = []
        tracked_positions = []  # Track which positions are valid
        
        for ant_id in ant_ids:
            x = data.loc[t, (ant_id, 'x')]
            y = data.loc[t, (ant_id, 'y')]
            if not (np.isnan(x) or np.isnan(y)):
                positions.append([x, y])
                tracked_positions.append(ant_id)
        
        if len(positions) < min_samples:
            clustering_stats['n_clusters'].append(0)
            clustering_stats['cluster_sizes'].append([])
            clustering_stats['mean_cluster_density'].append(0)
            clustering_stats['isolated_ants'].append(len(positions))
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
        
        # print(f"Found ant IDs: {ant_ids}")
        
        # Load data for each ant
        for ant_id in ant_ids:
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
                with open(kinematic_path, 'rb') as f:
                    kinematic_data = pickle.load(f)
                
                # Load and verify behavioural features 
                behavioural_path = os.path.join(processed_dir, "behavioural", f"ant_{ant_id}_behavioural.pkl")
                with open(behavioural_path, 'rb') as f:
                    behavioural_data = pickle.load(f)
                
                # Load and verify social features
                social_path = os.path.join(processed_dir, "social", f"ant_{ant_id}_social.pkl")
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


def save_cluster_data(clustering_stats, processed_dir):
    """Save colony clustering data to directory."""
    print("\nSaving clustering data...")
    clustering_path = os.path.join(processed_dir, "cluster", "colony_clustering.pkl")
    with open(clustering_path, 'wb') as f:
        pickle.dump(clustering_stats, f)
    print("Clustering data saved successfully!")


def load_cluster_data(processed_dir):
    """Load previously saved colony clustering data.
    
    Args:
        processed_dir: Directory containing the processed data
        
    Returns:
        Dictionary containing clustering statistics over time
    """
    print("Loading clustering data...")
    clustering_path = os.path.join(processed_dir, "cluster", "colony_clustering.pkl")
    
    try:
        with open(clustering_path, 'rb') as f:
            clustering_stats = pickle.load(f)
        print("Clustering data loaded successfully!")
        return clustering_stats
    except FileNotFoundError:
        print("No clustering data found. Please run with --save first.")
        exit(1)
    except Exception as e:
        print(f"Error loading clustering data: {str(e)}")
        exit(1)


def animate_clustering(clustering_stats: Dict[str, List], 
                    save_path: str = None,
                    target_duration: float = 60.0,  # Target duration in seconds
                    fps: int = 30,
                    start_frame: int = 0,
                    end_frame: Optional[int] = None,
                    arena_radius: float = ARENA_RADIUS) -> None:
    """
    Create a time-lapsed animation of ant clustering behaviour.
    
    Args:
        clustering_stats: Dictionary containing clustering statistics over time
        save_path: Path to save the animation (optional)
        target_duration: Desired duration in seconds
        fps: Frames per second for the output animation
        arena_radius: Radius of the arena in pixels
        start_frame: First frame to include in animation
        end_frame: Last frame to include in animation (if None, uses last frame)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.colors import LinearSegmentedColormap

    # Adjust time calculations based on SCALE
    base_fps = 60.0
    effective_fps = base_fps / SCALE if SCALE else base_fps

    # Calculate frame sampling
    total_frames = len(clustering_stats['positions'])
    end_frame = end_frame if end_frame is not None else total_frames
    
    # Validate frame range
    start_frame = max(0, min(start_frame, total_frames-1))
    end_frame = max(start_frame+1, min(end_frame, total_frames))
    
    frame_range = end_frame - start_frame
    total_output_frames = int(target_duration * fps)
    frame_step = max(1, frame_range // total_output_frames)
    sampled_frames = range(start_frame, end_frame, frame_step)
    
    # Print animation details
    print(f"Frame range: {start_frame:,} to {end_frame:,}")
    print(f"Frame sampling rate: {frame_step}")
    print(f"Output frames: {len(sampled_frames)}")
    print(f"Estimated duration: {len(sampled_frames)/fps:.1f} seconds")
    print(f"Time compression: {frame_range/(len(sampled_frames)*1/fps):.1f}x speed")
    
    # Set up the figure and animation
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Create a custom colormap for clusters
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    # Add timestamp text
    timestamp_text = ax.text(0.02, 0.82, '', transform=ax.transAxes, 
                        verticalalignment='top', fontsize=10)
    
    def update(frame_idx):
        ax.clear()
        frame = sampled_frames[frame_idx]
        
        positions = np.array(clustering_stats['positions'][frame])
        labels = np.array(clustering_stats['labels'][frame])
        cluster_ids = clustering_stats['cluster_ids'][frame]
        
        if len(positions) == 0:
            return
        
        # Plot arena boundary
        circle = plt.Circle((450, 450), arena_radius, fill=False, 
                        color='gray', linestyle='--')
        ax.add_artist(circle)
        
        # Plot isolated ants
        noise_points = positions[labels == -1]
        if len(noise_points) > 0:
            ax.scatter(noise_points[:, 0], noise_points[:, 1], 
                    c='gray', marker='o', s=50, alpha=0.5, 
                    label='Isolated')
        
        # Plot clustered ants using consistent colors based on cluster_ids
        for i, cluster_id in enumerate(cluster_ids):
            mask = labels == cluster_id
            cluster_points = positions[mask]
            color = colors[cluster_id % len(colors)]  # Use cluster_id for consistent coloring
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1],
                    c=[color], marker='o', s=50, 
                    label=f'Cluster {cluster_id}')
        
        # Add information and statistics
        minutes_elapsed = (frame / effective_fps) / 60  # Convert frames to minutes
        hours_elapsed = minutes_elapsed / 60  # Convert minutes to hours
        
        ax.text(0.02, 0.98, f'Time: {hours_elapsed:.1f} hours', 
                transform=ax.transAxes, verticalalignment='top')
        ax.text(0.02, 0.94, f'Frame: {frame:,}',
                transform=ax.transAxes, verticalalignment='top')
        ax.text(0.02, 0.90, f'Total ants: {len(positions)}',
                transform=ax.transAxes, verticalalignment='top')
        ax.text(0.02, 0.86, 
                f'Clusters: {clustering_stats["n_clusters"][frame]}',
                transform=ax.transAxes, verticalalignment='top')
        
        # Set axis properties
        ax.set_xlim(0, 900)
        ax.set_ylim(0, 900)
        ax.set_aspect('equal')
        ax.set_title('Ant Clustering Analysis (Time-Lapsed)')
        
        # Add scale bar (10mm = 86.4 pixels)
        scale_start = 50
        scale_end = scale_start + (10 * 8.64)  # 10mm * 8.64 pixels/mm
        ax.plot([scale_start, scale_end], [50, 50], 'k-', lw=2)
        ax.text(scale_start + (scale_end - scale_start)/2, 70, '10mm', ha='center')
            
        # Add legend outside the plot
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Create the animation
    anim = animation.FuncAnimation(
        fig, update,
        frames=len(sampled_frames),
        interval=1000/fps,  # interval in milliseconds
        blit=False
    )
    
    # Save animation if path provided
    if save_path:
        writer = animation.PillowWriter(fps=fps)
        anim.save(save_path, writer=writer)
        plt.close()
    else:
        plt.show()
    
    return anim


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
    
    # Create processed data directories if they don't exist
    processed_dir = os.path.join(DATA_DIRECTORY, "processed_data")
    for subdir in ["behavioural", "kinematic", "social", "cluster"]:
        os.makedirs(os.path.join(processed_dir, subdir), exist_ok=True)

    if args.load:
        processed_data = load_processed_data(processed_dir)
        clustering_stats = load_cluster_data(processed_dir)
    else:
        print("Loading raw data...")
        if args.debug:
            print(f"DEBUG MODE: Processing first {args.ants} ants and {args.timesteps} timesteps")
        data = load_data(DATA_DIRECTORY, INPUT_FILE, scale=SCALE,
                        debug=args.debug, debug_ants=args.ants,
                        debug_timesteps=args.timesteps)
        
        print("\nData Overview:")
        print(f"Total timesteps: {len(data):,}")
        print(f"Number of ants: {len(data.columns.levels[0])}")
        print(f"Recording duration: {len(data)/(60*60):.1f} minutes")
        
        print("\nProcessing ant trajectories...")
        start_time = time.time()
        processed_data = process_ant_data(data)
        print(f"Processing completed in {(time.time() - start_time)/60:.1f} minutes")
        
        print("\nAnalysing colony clustering...")
        clustering_stats = analyse_colony_clustering(data)
        
        if args.save:
            save_processed_data(processed_data, processed_dir)
            save_cluster_data(clustering_stats, processed_dir)
    

    
    # --- Analysis and Output ---
    print("\n=== Colony-Level Analysis ===")
    
    def analyse_movement_patterns(processed_data):
        """analyse movement patterns across all ants."""
        total_ants = len(processed_data)
        all_move_durations = []
        all_stop_durations = []
        all_velocities = []
        
        for ant_id, ant_data in processed_data.items():
            move_durations = ant_data['trajectory_features'].bout_durations['move']
            stop_durations = ant_data['trajectory_features'].bout_durations['stop']
            velocities = np.linalg.norm(ant_data['trajectory_features'].velocities, axis=1)
            
            all_move_durations.extend(move_durations)
            all_stop_durations.extend(stop_durations)
            all_velocities.extend(velocities[~np.isnan(velocities)])
        
        return {
            'total_ants': total_ants,
            'avg_move_duration': np.mean(all_move_durations),
            'avg_stop_duration': np.mean(all_stop_durations),
            'avg_velocity': np.mean(all_velocities),
            'max_velocity': np.max(all_velocities),
            'movement_ratio': sum(all_move_durations) / (sum(all_move_durations) + sum(all_stop_durations))
        }

    def analyse_social_interactions(processed_data):
        """analyse social interaction patterns."""
        all_nn_distances = []
        
        for ant_id, ant_data in processed_data.items():
            for timestep in ant_data['social_features']:
                if timestep and 'nn_stats' in timestep:
                    nn_stats = timestep['nn_stats']
                    if 'nn_dist_1' in nn_stats:  # First nearest neighbor
                        all_nn_distances.append(nn_stats['nn_dist_1'])
        
        return {
            'avg_nn_distance': np.mean(all_nn_distances),
            'min_nn_distance': np.min(all_nn_distances),
            'max_nn_distance': np.max(all_nn_distances)
        }

    # Perform analyses
    movement_stats = analyse_movement_patterns(processed_data)
    social_stats = analyse_social_interactions(processed_data)

    # Print results in a formatted way
    print("\nMovement Analysis:")
    print(f"{'='*50}")
    print(f"Total ants analysed: {movement_stats['total_ants']}")
    print(f"Average movement bout duration: {movement_stats['avg_move_duration']:.2f} seconds")
    print(f"Average stop bout duration: {movement_stats['avg_stop_duration']:.2f} seconds")
    print(f"Average velocity: {movement_stats['avg_velocity']:.2f} pixels/second")
    print(f"Maximum velocity: {movement_stats['max_velocity']:.2f} pixels/second")
    print(f"Movement ratio: {movement_stats['movement_ratio']*100:.1f}% of time spent moving")

    print("\nSocial Interaction Analysis:")
    print(f"{'='*50}")
    print(f"Average nearest neighbor distance: {social_stats['avg_nn_distance']:.2f} pixels")
    print(f"Minimum nearest neighbor distance: {social_stats['min_nn_distance']:.2f} pixels")
    print(f"Maximum nearest neighbor distance: {social_stats['max_nn_distance']:.2f} pixels")

    # Convert some metrics to biological units
    print("\nBiological Metrics:")
    print(f"{'='*50}")
    print(f"Average velocity: {movement_stats['avg_velocity']/PIXELS_PER_MM:.2f} mm/second")
    print(f"Maximum velocity: {movement_stats['max_velocity']/PIXELS_PER_MM:.2f} mm/second")
    print(f"Average nearest neighbor distance: {social_stats['avg_nn_distance']/PIXELS_PER_MM:.2f} mm")

    print("\nClustering Analysis:")
    print(f"{'='*50}")
    
    # Calculate and print clustering statistics
    n_clusters = np.array(clustering_stats['n_clusters'])
    cluster_sizes = clustering_stats['cluster_sizes']
    isolated_ants = np.array(clustering_stats['isolated_ants'])
    
    # Overall clustering statistics
    print(f"Average number of clusters: {np.mean(n_clusters):.2f}")
    print(f"Maximum number of clusters: {np.max(n_clusters)}")
    
    # Calculate average cluster size (excluding empty timesteps)
    non_empty_sizes = [sizes for sizes in cluster_sizes if sizes]
    if non_empty_sizes:
        all_sizes = [size for sizes in non_empty_sizes for size in sizes]
        print(f"Average cluster size: {np.mean(all_sizes):.2f} ants")
        print(f"Maximum cluster size: {np.max(all_sizes)} ants")
    
    # Isolation statistics
    print(f"Average number of isolated ants: {np.mean(isolated_ants):.2f}")
    print(f"Percentage of ants typically isolated: {100 * np.mean(isolated_ants) / movement_stats['total_ants']:.1f}%")
    
    # Time-based statistics
    clustered_frames = np.sum(n_clusters > 0)
    total_frames = len(n_clusters)
    print(f"\nTemporal patterns:")
    print(f"Percentage of time with clusters present: {100 * clustered_frames / total_frames:.1f}%")

    # Animate clustering
    animate_clustering(clustering_stats, 
                    save_path='ant_clustering.gif',  # optional
                    target_duration=120,  # 2 minutes
                    fps=30,
                    start_frame=0,  # Start from beginning
                    end_frame=10000  # Only animate first 1000 frames
    )
