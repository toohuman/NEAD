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


def save_processed_data(data: Dict[int, Dict[str, Any]], 
                       output_dir: str,
                       filename: str,
                       compress_level: int = 6) -> None:
    """
    Save processed ant behavioral data to a compressed file.
    
    Args:
        data: Dictionary containing processed ant data
        output_dir: Directory to save the file
        filename: Name of the output file
        compress_level: LZMA compression level (0-9, higher = better compression but slower)
    """
    print("Starting data serialization...")
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Convert TrajectoryFeatures objects to dictionaries for serialization
    print("Converting data structures...")
    serializable_data = {}
    for ant_id, ant_data in tqdm(data.items(), desc="Preparing data"):
        serializable_data[ant_id] = {
            'trajectory_features': asdict(ant_data['trajectory_features']),
            'social_features': ant_data['social_features']
        }
    
    # Ensure filename has proper extension
    if not filename.endswith('.pkl.xz'):
        filename = f"{filename}.pkl.xz"
    
    output_path = os.path.join(output_dir, filename)
    
    # Save with compression
    print(f"Saving compressed data to {output_path}")
    print("This may take a while for large datasets...")
    
    try:
        with lzma.open(output_path, 'wb', preset=compress_level) as f:
            pickle.dump(serializable_data, f)
        
        # Print file size information
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # Convert to MB
        print(f"Data successfully saved!")
        print(f"Compressed file size: {file_size:.2f} MB")
        
    except Exception as e:
        print(f"Error saving data: {str(e)}")
        raise

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
    """Extract behavioral features from ant trajectory data."""
    
    def __init__(self, fps: float = 60.0, velocity_threshold: float = 0.5):
        """
        Initialize the feature extractor.
        
        Args:
            fps: Frame rate of the data
            velocity_threshold: Threshold for determining stop/move states (units/second)
        """
        self.dt = 1.0 / fps
        self.velocity_threshold = velocity_threshold
    
    def compute_velocities(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute velocity components and magnitude."""
        dx = np.gradient(x, self.dt)
        dy = np.gradient(y, self.dt)
        velocity_mag = np.sqrt(dx**2 + dy**2)
        return np.stack([dx, dy], axis=1), velocity_mag
    
    def compute_accelerations(self, velocities: np.ndarray) -> np.ndarray:
        """Compute acceleration components and magnitude."""
        return np.gradient(velocities, self.dt, axis=0)
    
    def compute_angular_velocity(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute angular velocity from position data."""
        # Calculate heading angles
        dx = np.gradient(x, self.dt)
        dy = np.gradient(y, self.dt)
        angles = np.arctan2(dy, dx)
        
        # Compute angular velocity (handling circular wrapping)
        angular_vel = np.gradient(np.unwrap(angles), self.dt)
        return angular_vel
    
    def compute_curvature(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute path curvature."""
        dx = np.gradient(x, self.dt)
        dy = np.gradient(y, self.dt)
        ddx = np.gradient(dx, self.dt)
        ddy = np.gradient(dy, self.dt)
        
        # Avoid division by zero in curvature calculation
        denominator = (dx * dx + dy * dy) ** 1.5
        curvature = np.zeros_like(dx)
        valid_denom = denominator > 1e-10  # Threshold for numerical stability
        curvature[valid_denom] = np.abs(dx[valid_denom] * ddy[valid_denom] - 
                                      dy[valid_denom] * ddx[valid_denom]) / denominator[valid_denom]
        return curvature
    
    def identify_movement_bouts(self, velocity_mag: np.ndarray) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Identify periods of movement and stopping."""
        is_moving = velocity_mag > self.velocity_threshold
        
        # Find transitions
        state_changes = np.diff(is_moving.astype(int))
        start_moves = np.where(state_changes == 1)[0] + 1
        end_moves = np.where(state_changes == -1)[0] + 1
        
        # Handle edge cases
        if is_moving[0]:
            start_moves = np.insert(start_moves, 0, 0)
        if is_moving[-1]:
            end_moves = np.append(end_moves, len(is_moving))
            
        move_segments = list(zip(start_moves, end_moves))
        
        # Calculate stop segments as the complement of move segments
        stop_segments = []
        if len(move_segments) > 0:
            if move_segments[0][0] > 0:
                stop_segments.append((0, move_segments[0][0]))
            for i in range(len(move_segments)-1):
                stop_segments.append((move_segments[i][1], move_segments[i+1][0]))
            if move_segments[-1][1] < len(is_moving):
                stop_segments.append((move_segments[-1][1], len(is_moving)))
                
        return move_segments, stop_segments
    
    def extract_features(self, x: np.ndarray, y: np.ndarray) -> TrajectoryFeatures:
        """Extract all trajectory features from position data."""
        # Remove any NaN values
        valid = ~(np.isnan(x) | np.isnan(y))
        x, y = x[valid], y[valid]
        
        # Compute basic kinematic features
        velocities, velocity_mag = self.compute_velocities(x, y)
        accelerations = self.compute_accelerations(velocities)
        angular_velocities = self.compute_angular_velocity(x, y)
        curvatures = self.compute_curvature(x, y)
        
        # Identify behavioral segments
        move_segments, stop_segments = self.identify_movement_bouts(velocity_mag)
        
        # Calculate bout durations
        bout_durations = {
            'move': [self.dt * (end - start) for start, end in move_segments],
            'stop': [self.dt * (end - start) for start, end in stop_segments]
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
    """Extract features related to social interactions."""
    
    def __init__(self, n_sectors: int = 8, max_distance: float = 100.0):
        """
        Initialize the social context extractor.
        
        Args:
            n_sectors: Number of angular sectors for density calculation
            max_distance: Maximum distance to consider for neighbor interactions
        """
        self.n_sectors = n_sectors
        self.max_distance = max_distance
        self.sector_angles = np.linspace(0, 2*np.pi, n_sectors+1)
    
    def compute_local_density(self, focal_x: float, focal_y: float, 
                            neighbor_x: np.ndarray, neighbor_y: np.ndarray) -> np.ndarray:
        """Compute ant density in different sectors around focal ant."""
        # Calculate relative positions
        rel_x = neighbor_x - focal_x
        rel_y = neighbor_y - focal_y
        
        # Convert to polar coordinates
        distances = np.sqrt(rel_x**2 + rel_y**2)
        angles = np.arctan2(rel_y, rel_x) % (2*np.pi)
        
        # Count ants in each sector within max_distance
        densities = np.zeros(self.n_sectors)
        for i in range(self.n_sectors):
            in_sector = (angles >= self.sector_angles[i]) & (angles < self.sector_angles[i+1])
            in_range = distances <= self.max_distance
            densities[i] = np.sum(in_sector & in_range)
            
        return densities
    
    def compute_nearest_neighbor_stats(self, focal_x: float, focal_y: float,
                                     neighbor_x: np.ndarray, neighbor_y: np.ndarray,
                                     k: int = 3) -> Dict[str, float]:
        """Compute statistics about k nearest neighbors."""
        distances = np.sqrt((neighbor_x - focal_x)**2 + (neighbor_y - focal_y)**2)
        sorted_distances = np.sort(distances)
        
        return {
            f'nn_dist_{i+1}': dist for i, dist in enumerate(sorted_distances[:k])
        }

def process_ant_data(data: pd.DataFrame) -> Dict[int, Dict[str, any]]:
    """
    Process all ant trajectories and extract features.
    
    Args:
        data: DataFrame with MultiIndex columns (ant_id, coordinate)
    
    Returns:
        Dictionary mapping ant IDs to their extracted features
    """
    feature_extractor = AntFeatureExtractor()
    social_extractor = SocialContextExtractor()
    
    results = {}
    ant_ids = data.columns.levels[0]
    
    for ant_id in tqdm(ant_ids, desc="Processing ants"):
        # Extract trajectory features
        x = data[ant_id, 'x'].values
        y = data[ant_id, 'y'].values
        
        traj_features = feature_extractor.extract_features(x, y)
        
        # Extract social features for each timestep with progress bar
        social_features = []
        for t in tqdm(range(len(x)), desc=f"Processing timesteps for ant {ant_id}", leave=False):
            if np.isnan(x[t]) or np.isnan(y[t]):
                continue
                
            # Get positions of other ants at this timestep more efficiently
            other_positions = data.loc[t].drop((ant_id, 'x')).drop((ant_id, 'y')).values.reshape(-1, 2)
            other_x = other_positions[:, 0]
            other_y = other_positions[:, 1]
            
            # Remove NaN values
            valid = ~(np.isnan(other_x) | np.isnan(other_y))
            if np.any(valid):  # Only process if we have valid neighbours
                other_x = other_x[valid]
                other_y = other_y[valid]
            
            if len(other_x) > 0:
                densities = social_extractor.compute_local_density(x[t], y[t], other_x, other_y)
                nn_stats = social_extractor.compute_nearest_neighbor_stats(x[t], y[t], other_x, other_y)
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
    data = load_data(DATA_DIRECTORY, INPUT_FILE)
    processed_data = process_ant_data(data)
    
    # Save processed data
    output_directory = DATA_DIRECTORY
    output_filename = "ant_features_processed.pkl.xz"
    save_processed_data(processed_data, output_directory, output_filename)
    
    # Test loading the saved data
    loaded_data = load_processed_data(os.path.join(output_directory, output_filename))
    
    # Verify the data
    print("\nVerifying loaded data:")
    print(f"Number of ants in original processed data: {len(processed_data)}")
    print(f"Number of ants in loaded data: {len(loaded_data)}")
    
    # Compare some values
    first_ant_id = list(processed_data.keys())[0]
    original_features = processed_data[first_ant_id]['trajectory_features']
    loaded_features = loaded_data[first_ant_id]['trajectory_features']
    
    print(f"\nComparing features for ant {first_ant_id}:")
    print(f"Original number of move segments: {len(original_features.move_segments)}")
    print(f"Loaded number of move segments: {len(loaded_features.move_segments)}")