import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import numpy.typing as npt
import lzma
import os

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
        
        curvature = np.abs(dx * ddy - dy * ddx) / (dx * dx + dy * dy) ** 1.5
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
        
        # Extract social features for each timestep
        social_features = []
        for t in range(len(x)):
            if np.isnan(x[t]) or np.isnan(y[t]):
                continue
                
            # Get positions of other ants at this timestep
            other_x = np.array([data[other_id, 'x'].iloc[t] 
                              for other_id in ant_ids if other_id != ant_id])
            other_y = np.array([data[other_id, 'y'].iloc[t] 
                              for other_id in ant_ids if other_id != ant_id])
            
            # Remove NaN values
            valid = ~(np.isnan(other_x) | np.isnan(other_y))
            other_x, other_y = other_x[valid], other_y[valid]
            
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

# Step 1: Load and clean the data
data = load_data(DATA_DIRECTORY, INPUT_FILE)

print("Original data:")
print(data.head())

data = process_ant_data(data)
data.head()
