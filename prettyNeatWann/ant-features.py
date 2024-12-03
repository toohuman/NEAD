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
from pathlib import Path

DATA_DIRECTORY = "data/2023_2/"
INPUT_FILE = 'KA050_processed_10cm_5h_20230614.pkl.xz'

def load_data(source_dir, input_file, scale=None, arena_dim=None):
    """Load data from a compressed pickle file."""
    data = None
    with lzma.open(os.path.join(source_dir, input_file)) as file:
        data = pd.read_pickle(file)
    return data.iloc[::int(scale)] if scale else data


def save_processed_data_chunked(data: Dict[int, Dict[str, Any]], 
                              output_dir: str,
                              base_filename: str) -> None:
    """
    Save processed ant behavioural data as separate files by feature type.
    
    Structure:
    output_dir/processed_data/
    ├── kinematic/
    │   ├── velocities_ant_{id}.pkl.xz
    │   ├── accelerations_ant_{id}.pkl.xz
    │   ├── angular_velocities_ant_{id}.pkl.xz
    │   └── curvatures_ant_{id}.pkl.xz
    ├── behavioural/
    │   ├── segments_ant_{id}.pkl.xz  (contains stop/move segments)
    │   └── bout_durations_ant_{id}.pkl.xz
    └── social/
        └── social_features_ant_{id}.pkl.xz
    """
    # Create directory structure
    base_dir = os.path.join(output_dir, 'processed_data')
    for subdir in ['kinematic', 'behavioural', 'social']:
        Path(os.path.join(base_dir, subdir)).mkdir(parents=True, exist_ok=True)
    
    print("Saving data in chunks...")
    
    # Process each ant separately
    for ant_id, ant_data in tqdm(data.items(), desc="Processing ants"):
        traj_features = ant_data['trajectory_features']
        
        # Save kinematic features
        kinematic_features = {
            'velocities': traj_features.velocities,
            'accelerations': traj_features.accelerations,
            'angular_velocities': traj_features.angular_velocities,
            'curvatures': traj_features.curvatures
        }
        for feature_name, feature_data in kinematic_features.items():
            filename = f"{feature_name}_ant_{ant_id}.pkl.xz"
            filepath = os.path.join(base_dir, 'kinematic', filename)
            with lzma.open(filepath, 'wb') as f:
                pickle.dump(feature_data, f)
        
        # Save behavioural features
        behavioural_features = {
            'segments': {
                'stop_segments': traj_features.stop_segments,
                'move_segments': traj_features.move_segments
            },
            'bout_durations': traj_features.bout_durations
        }
        for feature_name, feature_data in behavioural_features.items():
            filename = f"{feature_name}_ant_{ant_id}.pkl.xz"
            filepath = os.path.join(base_dir, 'behavioural', filename)
            with lzma.open(filepath, 'wb') as f:
                pickle.dump(feature_data, f)
        
        # Save social features
        filename = f"social_features_ant_{ant_id}.pkl.xz"
        filepath = os.path.join(base_dir, 'social', filename)
        with lzma.open(filepath, 'wb') as f:
            pickle.dump(ant_data['social_features'], f)

def load_processed_data(filepath: str) -> Dict[int, Dict[str, Any]]:
    """
    Load processed ant behavioral data from a compressed file.
    
    Args:
        filepath: Path to the compressed data file
    
    Returns:
        Dictionary containing processed ant data
    """
    with lzma.open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    # Reconstruct TrajectoryFeatures objects
    reconstructed_data = {}
    for ant_id, ant_data in data.items():
        reconstructed_data[ant_id] = {
            'trajectory_features': TrajectoryFeatures(**ant_data['trajectory_features']),
            'social_features': ant_data['social_features']
        }


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
    
    def __init__(self, fps: float = 60.0, velocity_threshold: float = 0.5, max_position_change: float = 100.0):
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
        
        # Process all social features at once instead of chunking
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

# Example usage:
if __name__ == "__main__":
    # Load and process original data
    print("Loading data...")
    data = load_data(DATA_DIRECTORY, INPUT_FILE)
    
    # Basic data validation
    print("\nInitial data validation:")
    print(f"Number of timesteps: {len(data)}")
    print(f"Number of ants: {len(data.columns.levels[0])}")
    print(f"Data shape: {data.shape}")
    
    # Process data and time the execution
    print("\nProcessing ant data...")
    import time
    start_time = time.time()
    processed_data = process_ant_data(data)
    processing_time = time.time() - start_time
    print(f"Processing completed in {processing_time:.2f} seconds")
    
    # Validate processed data structure
    print("\nValidating processed data structure:")
    first_ant = list(processed_data.keys())[0]
    features = processed_data[first_ant]['trajectory_features']
    
    print("\nFeature shapes:")
    print(f"Velocities: {features.velocities.shape}")
    print(f"Accelerations: {features.accelerations.shape}")
    print(f"Angular velocities: {features.angular_velocities.shape}")
    print(f"Curvatures: {features.curvatures.shape}")
    
    # Validate movement segmentation
    print("\nMovement segmentation validation:")
    print(f"Number of move segments: {len(features.move_segments)}")
    print(f"Number of stop segments: {len(features.stop_segments)}")
    print(f"Total move duration: {sum(features.bout_durations['move']):.2f} seconds")
    print(f"Total stop duration: {sum(features.bout_durations['stop']):.2f} seconds")
    
    # Validate social features
    print("\nSocial features validation:")
    social_features = processed_data[first_ant]['social_features']
    print(f"Number of social feature timestamps: {len(social_features)}")
    if social_features:
        print(f"Number of density sectors: {len(social_features[0]['densities'])}")
        print(f"Available nearest neighbour stats: {list(social_features[0]['nn_stats'].keys())}")
    
    # Basic sanity checks
    print("\nPerforming sanity checks:")
    
    # Check for NaN values
    velocities = features.velocities
    nan_count = np.isnan(velocities).sum()
    print(f"NaN values in velocities: {nan_count}")
    
    # Check for reasonable velocity ranges
    max_velocity = np.max(np.linalg.norm(velocities, axis=1))
    mean_velocity = np.mean(np.linalg.norm(velocities, axis=1))
    print(f"Maximum velocity: {max_velocity:.2f} units/second")
    print(f"Mean velocity: {mean_velocity:.2f} units/second")
    
    # Verify segment continuity
    move_segments = features.move_segments
    if move_segments:
        segments_ordered = all(seg[0] < seg[1] for seg in move_segments)
        segments_continuous = all(move_segments[i][1] <= move_segments[i+1][0] 
                                for i in range(len(move_segments)-1))
        print(f"Move segments properly ordered: {segments_ordered}")
        print(f"Move segments continuous: {segments_continuous}")
    
    # Save processed data in chunks
    print("\nSaving processed data...")
    save_start_time = time.time()
    output_directory = DATA_DIRECTORY
    base_filename = "ant_features"
    save_processed_data_chunked(processed_data, output_directory, base_filename)
    save_time = time.time() - save_start_time
    print(f"Data saved in {save_time:.2f} seconds")
    
    # Test loading the saved data
    print("\nTesting data loading...")
    load_start_time = time.time()
    loaded_data = load_processed_data_chunked(
        os.path.join(output_directory, 'processed_data'),
        ant_ids=[first_ant]
    )
    load_time = time.time() - load_start_time
    print(f"Data loaded in {load_time:.2f} seconds")
    
    # Verify loaded data matches original
    print("\nVerifying loaded data integrity:")
    original_features = processed_data[first_ant]['trajectory_features']
    loaded_features = loaded_data[first_ant]['trajectory_features']
    
    velocities_match = np.allclose(
        original_features.velocities, 
        loaded_features.velocities,
        equal_nan=True
    )
    segments_match = (original_features.move_segments == loaded_features.move_segments)
    
    print(f"Velocities match: {velocities_match}")
    print(f"Movement segments match: {segments_match}")
    
    # Memory usage statistics
    import psutil
    process = psutil.Process()
    memory_usage = process.memory_info().rss / 1024 / 1024  # Convert to MB
    print(f"\nCurrent memory usage: {memory_usage:.2f} MB")