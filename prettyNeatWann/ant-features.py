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
from pathlib import Path

DATA_DIRECTORY = "data/2023_2/"
INPUT_FILE = 'KA050_processed_10cm_5h_20230614.pkl.xz'

def load_data(source_dir, input_file, scale=None, arena_dim=None):
    """Load data from a compressed pickle file."""
    data = None
    with lzma.open(os.path.join(source_dir, input_file)) as file:
        data = pd.read_pickle(file)
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
    bout_durations: Dict[str, List[float]]  # durations of different behavioral bouts


class AntFeatureExtractor:
    """Extract behavioral features from ant trajectory data using vectorized operations."""
    
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
        
        # Compute velocities (rest of the computation remains the same)
        dx = np.gradient(x_clean, self.dt)
        dy = np.gradient(y_clean, self.dt)
        velocities[:, 0] = dx
        velocities[:, 1] = dy
        velocity_mag = np.sqrt(dx**2 + dy**2)
        
        # Pre-allocate arrays and use vectorized operations
        valid = ~(np.isnan(x) | np.isnan(y))
        x, y = x[valid], y[valid]
        n_points = len(x)
        
        # Pre-allocate arrays for kinematic features
        velocities = np.zeros((n_points, 2), dtype=np.float64)
        accelerations = np.zeros((n_points, 2), dtype=np.float64)
        angular_velocities = np.zeros(n_points, dtype=np.float64)
        curvatures = np.zeros(n_points, dtype=np.float64)
        
        # Vectorized computation of velocities
        dx = np.gradient(x, self.dt)
        dy = np.gradient(y, self.dt)
        velocities[:, 0] = dx
        velocities[:, 1] = dy
        velocity_mag = np.sqrt(dx**2 + dy**2)
        
        # Vectorized computation of accelerations
        accelerations[:, 0] = np.gradient(dx, self.dt)
        accelerations[:, 1] = np.gradient(dy, self.dt)
        
        # Vectorized computation of angular velocity
        angles = np.arctan2(dy, dx)
        angular_velocities = np.gradient(np.unwrap(angles), self.dt)
        
        # Vectorized computation of curvature
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
    
    # Pre-allocate arrays for positions
    positions = np.zeros((len(ant_ids), n_timesteps, 2))
    
    # Extract positions more efficiently
    for i, ant_id in enumerate(ant_ids):
        positions[i, :, 0] = data[ant_id, 'x'].values
        positions[i, :, 1] = data[ant_id, 'y'].values
    
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


if __name__ == "__main__":
# Constants for biological sanity checks
    PIXELS_PER_MM = 8.1  # Based on 900px = 100mm arena diameter
    ARENA_CENTER = np.array([450, 450])
    ARENA_RADIUS = 405  # pixels
    MAX_EXPECTED_VELOCITY = 50  # mm/s
    
    print("Loading data...")
    data = load_data(DATA_DIRECTORY, INPUT_FILE)
    
    print("\nData Overview:")
    print(f"Total timesteps: {len(data):,}")
    print(f"Number of ants: {len(data.columns.levels[0])}")
    print(f"Recording duration: {len(data)/60:.1f} minutes")
    
    print("\nProcessing ant trajectories...")
    start_time = time.time()
    processed_data = process_ant_data(data)
    print(f"Processing completed in {(time.time() - start_time)/60:.1f} minutes")
    
    # Colony-wide statistics
    print("\n=== Colony-wide Analysis ===")
    
    # Initialize containers for colony statistics
    colony_velocities = []
    colony_move_durations = []
    colony_stop_durations = []
    colony_distances_from_center = []
    colony_nn_distances = []
    colony_tracking_segments = []
    colony_activity_ratios = []
    
    # Collect statistics from all ants
    for ant_id, ant_data in processed_data.items():
        features = ant_data['trajectory_features']
        
        # Velocities
        velocities_mm_s = np.linalg.norm(features.velocities, axis=1) / PIXELS_PER_MM
        colony_velocities.extend(velocities_mm_s)
        
        # Movement patterns
        colony_move_durations.extend(features.bout_durations['move'])
        colony_stop_durations.extend(features.bout_durations['stop'])
        colony_tracking_segments.append(len(features.move_segments))
        
        # Activity ratio
        total_time = len(features.velocities) / 60  # in seconds
        moving_time = sum(features.bout_durations['move'])
        colony_activity_ratios.append(moving_time/total_time if total_time > 0 else 0)
        
        # Spatial distribution
        positions = np.column_stack([
            data[ant_id, 'x'].values,
            data[ant_id, 'y'].values
        ])
        distances = np.sqrt(np.sum((positions - ARENA_CENTER)**2, axis=1))
        colony_distances_from_center.extend(distances[~np.isnan(distances)])
        
        # Social distances
        if ant_data['social_features']:
            nn_distances = [sf['nn_stats']['nn_dist_1'] for sf in ant_data['social_features']]
            colony_nn_distances.extend(nn_distances)
    
    print("\nMovement Statistics (Colony-wide):")
    print(f"Velocity (mm/s):")
    print(f"  Maximum: {np.max(colony_velocities):.1f}")
    print(f"  Mean ± SD: {np.mean(colony_velocities):.1f} ± {np.std(colony_velocities):.1f}")
    print(f"  Median: {np.median(colony_velocities):.1f}")
    
    print(f"\nBout Durations (seconds):")
    print(f"Movement bouts:")
    print(f"  Mean ± SD: {np.mean(colony_move_durations):.2f} ± {np.std(colony_move_durations):.2f}")
    print(f"  Median: {np.median(colony_move_durations):.2f}")
    print(f"Stop bouts:")
    print(f"  Mean ± SD: {np.mean(colony_stop_durations):.2f} ± {np.std(colony_stop_durations):.2f}")
    print(f"  Median: {np.median(colony_stop_durations):.2f}")
    
    print(f"\nActivity Patterns:")
    print(f"  Mean activity ratio: {np.mean(colony_activity_ratios):.2%}")
    print(f"  Activity ratio range: {np.min(colony_activity_ratios):.2%} - {np.max(colony_activity_ratios):.2%}")
    print(f"  Tracking segments per ant: {np.mean(colony_tracking_segments):.1f} ± {np.std(colony_tracking_segments):.1f}")
    
    print("\nSpatial Distribution:")
    distances_mm = np.array(colony_distances_from_center) / PIXELS_PER_MM
    print(f"  Mean distance from center: {np.mean(distances_mm):.1f} mm")
    print(f"  Distance range: {np.min(distances_mm):.1f} - {np.max(distances_mm):.1f} mm")
    outside_points = np.sum(distances_mm > (ARENA_RADIUS/PIXELS_PER_MM))
    print(f"  Points outside arena: {100 * outside_points/len(distances_mm):.2f}%")
    
    print("\nSocial Interactions:")
    nn_distances_mm = np.array(colony_nn_distances) / PIXELS_PER_MM
    print(f"  Mean nearest neighbour distance: {np.mean(nn_distances_mm):.1f} mm")
    print(f"  Nearest neighbour range: {np.min(nn_distances_mm):.1f} - {np.max(nn_distances_mm):.1f} mm")
    
    print("\nData Quality Metrics:")
    high_velocity_count = np.sum(np.array(colony_velocities) > MAX_EXPECTED_VELOCITY)
    print(f"  Suspicious velocities: {100 * high_velocity_count/len(colony_velocities):.2f}%")
    print(f"  Missing position data: {100 * np.sum(np.isnan(data.values))/data.size:.2f}%")