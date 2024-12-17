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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')

# Constants for biological sanity checks
PIXELS_PER_MM = 8.64  # Based on 864px = 100mm arena diameter (accounting for 2% margin)
ARENA_CENTER = np.array([900/2.0, 900/2.0])
ARENA_RADIUS = min(900, 900)/2.0 - min(900, 900) * 0.02
MAX_EXPECTED_VELOCITY = 50  # mm/s

DATA_DIRECTORY = "data/2023_2/"
INPUT_FILE = 'KA050_processed_10cm_5h_20230614.pkl.xz'
VIDEO_FPS = 60
SCALE = 2

def load_data(source_dir, input_file, scale=None, arena_dim=None, debug=False, debug_ants=5, debug_timesteps=10000,
              time_window=None):
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
        time_window: Tuple of (start_min, end_min) to filter data by time window
    """
    data = None
    with lzma.open(os.path.join(source_dir, input_file)) as file:
        data = pd.read_pickle(file)
    
    if time_window:
        start_min, end_min = time_window
        # Convert minutes to frames
        fps = VIDEO_FPS / (scale if scale else 1)
        start_frame = int(start_min * 60 * fps)
        end_frame = int(end_min * 60 * fps)
        data = data.iloc[start_frame:end_frame]
    
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
            fps: Frame rate of the data (already adjusted for SCALE if provided)
            velocity_threshold: Threshold for determining stop/move states (units/second)
            max_position_change: Maximum allowable position change between consecutive frames (units)
            max_velocity_pixels: Maximum allowable velocity in pixels/second
        """
        self.fps = fps  # fps is already scaled when passed in
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
        
        # Check if we have enough points for gradient calculation
        if len(x_clean) < 2:  # Need at least 2 points for gradient
            return TrajectoryFeatures(
                velocities=np.zeros((0, 2)),
                accelerations=np.zeros((0, 2)),
                angular_velocities=np.zeros(0),
                curvatures=np.zeros(0),
                stop_segments=[],
                move_segments=[],
                bout_durations={'move': np.array([]), 'stop': np.array([])}
            )
            
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
    effective_fps = VIDEO_FPS / SCALE if SCALE else VIDEO_FPS
    
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


@dataclass
class BehaviouralState:
    """Container for the complete behavioural state at a given timestep."""
    # Individual motion states
    velocities: np.ndarray  # Shape: (n_ants, 2)
    accelerations: np.ndarray  # Shape: (n_ants, 2)
    turn_rates: np.ndarray  # Shape: (n_ants,)
    curvatures: np.ndarray  # Shape: (n_ants,)
    
    # Recent history features
    movement_history: np.ndarray  # Shape: (n_ants, history_length)
    stop_go_patterns: np.ndarray  # Shape: (n_ants, history_length)
    
    # Colony-level states
    cluster_config: Dict[str, any]  # Current clustering configuration
    neighbor_distances: np.ndarray  # Shape: (n_ants, n_nearest)
    activity_level: float  # Overall colony activity
    spatial_distribution: Dict[str, float]  # Metrics of spatial distribution
    
    # Temporal context
    state_changes: np.ndarray  # Recent state transitions
    transition_rates: Dict[str, float]  # Rates of different transitions
    time_features: Dict[str, float]  # Time-dependent features

class BehaviouralStateExtractor:
    """Extract high-dimensional behavioural state features from ant trajectory data."""
    
    def __init__(self, 
                 seconds_of_history: float = 1.0,  # How many seconds of history to maintain
                 n_nearest: int = 5,  # Number of nearest neighbours to track
                 fps: float = 60.0):
        self.history_length = int(seconds_of_history * fps)
        self.n_nearest = n_nearest
        self.fps = fps
        self.dt = 1.0 / fps
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=10)  # Reduce to 10 principal components
        
    def extract_individual_motion(self, 
                                positions: np.ndarray,  # Shape: (n_ants, 2)
                                prev_positions: np.ndarray,  # Shape: (n_ants, history_length, 2)
                                prev_velocities: np.ndarray  # Shape: (n_ants, history_length, 2)
                                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extract individual motion features for each ant."""
        n_ants = positions.shape[0]
        
        # Calculate velocities and accelerations
        velocities = (positions - prev_positions[-1]) / self.dt
        accelerations = (velocities - prev_velocities[-1]) / self.dt
        
        # Calculate turn rates and path curvatures
        prev_headings = np.arctan2(prev_velocities[:, -1, 1], prev_velocities[:, -1, 0])
        current_headings = np.arctan2(velocities[:, 1], velocities[:, 0])
        
        # Ensure arrays have matching shapes by padding or truncating
        min_length = min(len(prev_headings), len(current_headings))
        prev_headings = prev_headings[:min_length]
        current_headings = current_headings[:min_length]
        
        # Calculate turn rates (ensuring matching array lengths)
        angle_diff = current_headings - prev_headings
        # Pad the result with the first value to maintain array length
        turn_rates = np.pad(np.abs(np.unwrap(angle_diff)), (0, 1), mode='edge') / self.dt
        
        # Pad turn_rates back to original size if needed
        if len(turn_rates) < n_ants:
            turn_rates = np.pad(turn_rates, (0, n_ants - len(turn_rates)), mode='edge')
        
        # Calculate path curvatures using three-point method
        curvatures = np.zeros(n_ants)
        for i in range(n_ants):
            # Get available history points
            if i < len(prev_positions):
                history_points = prev_positions[i]
                valid_points = history_points[~np.isnan(history_points).any(axis=1)]
                
                if len(valid_points) >= 2:
                    points = np.vstack([valid_points[-2:], positions[i]])
                else:
                    # If insufficient history, use current position with small offset
                    points = np.vstack([positions[i] - [0.1, 0.1], positions[i]])
            else:
                # Handle case where i is out of bounds
                points = np.vstack([positions[i] - [0.1, 0.1], positions[i]])
            # Calculate curvature using three points
            if len(points) >= 3:
                dx = np.gradient(points[:, 0])
                dy = np.gradient(points[:, 1])
                ddx = np.gradient(dx)
                ddy = np.gradient(dy)
                curvature = np.abs(dx * ddy - dy * ddx) / (dx * dx + dy * dy) ** 1.5
                curvatures[i] = np.mean(curvature)
        
        return velocities, accelerations, turn_rates, curvatures
    
    def extract_movement_history(self,
                               positions: np.ndarray,  # Shape: (n_ants, history_length, 2)
                               velocity_threshold: float = 0.5  # mm/s
                               ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features from recent movement history."""
        n_ants = positions.shape[0]
        
        # Calculate velocities over history
        velocities = np.diff(positions, axis=1) / self.dt
        speeds = np.linalg.norm(velocities, axis=2)
        
        # Create movement history features
        movement_history = speeds
        
        # Create stop-go patterns (binary)
        stop_go_patterns = (speeds > velocity_threshold).astype(float)
        
        return movement_history, stop_go_patterns
    
    def extract_colony_state(self,
                           positions: np.ndarray,  # Shape: (n_ants, 2)
                           cluster_info: Dict[str, any]
                           ) -> Tuple[Dict[str, any], np.ndarray, float, Dict[str, float]]:
        """Extract colony-level state features."""
        n_ants = positions.shape[0]
        
        # Process clustering configuration
        cluster_config = {
            'n_clusters': cluster_info['n_clusters'],
            'cluster_sizes': cluster_info['cluster_sizes'],
            'mean_density': cluster_info['mean_cluster_density']
        }
        
        # Calculate nearest neighbour distances
        neighbor_distances = np.zeros((n_ants, self.n_nearest))
        for i in range(n_ants):
            distances = np.linalg.norm(positions - positions[i], axis=1)
            distances[i] = np.inf  # Exclude self
            # Adjust n_nearest if there aren't enough points
            k = min(self.n_nearest, len(distances) - 1)  # -1 to account for self
            if k > 0:
                nearest_indices = np.argpartition(distances, k)[:k]
                neighbor_distances[i, :k] = distances[nearest_indices]
                # Fill remaining slots with max distance if any
                if k < self.n_nearest:
                    neighbor_distances[i, k:] = np.max(distances)
            else:
                # If no neighbors, fill with max possible distance
                neighbor_distances[i, :] = np.sqrt((2 * ARENA_RADIUS) ** 2)
        
        # Calculate overall activity level
        activity_level = np.mean(cluster_info['mean_cluster_density'])
        
        # Calculate spatial distribution metrics
        if len(positions) >= 4:  # Need at least 4 points for ConvexHull
            try:
                hull = ConvexHull(positions)
                spatial_distribution = {
                    'area': hull.area,
                    'perimeter': hull.area / hull.volume if hull.volume > 0 else 0,
                    'density': n_ants / hull.area if hull.area > 0 else 0
                }
            except Exception:
                spatial_distribution = {'area': 0, 'perimeter': 0, 'density': 0}
        else:
            spatial_distribution = {'area': 0, 'perimeter': 0, 'density': 0}
        
        return cluster_config, neighbor_distances, activity_level, spatial_distribution
    
    def extract_temporal_context(self,
                               current_state: BehaviouralState,
                               prev_states: List[BehaviouralState],
                               window_size: int = 60  # 1 second at 60fps
                               ) -> Tuple[np.ndarray, Dict[str, float], Dict[str, float]]:
        """Extract temporal context features."""
        # Track recent state changes
        state_changes = np.zeros(len(prev_states))
        for i in range(1, len(prev_states)):
            # Calculate state difference metric
            state_changes[i] = np.abs(
                prev_states[i].activity_level - 
                prev_states[i-1].activity_level
            )
        
        # Calculate transition rates
        transition_rates = {
            'activity_change_rate': np.mean(np.abs(np.diff(
                [state.activity_level for state in prev_states]
            ))),
            'cluster_change_rate': np.mean(np.abs(np.diff(
                [state.cluster_config['n_clusters'] for state in prev_states]
            )))
        }
        
        # Extract time-dependent features
        time_features = {
            'activity_trend': np.polyfit(
                np.arange(len(prev_states)),
                [state.activity_level for state in prev_states],
                1
            )[0],
            'clustering_trend': np.polyfit(
                np.arange(len(prev_states)),
                [state.cluster_config['n_clusters'] for state in prev_states],
                1
            )[0]
        }
        
        return state_changes, transition_rates, time_features
    
    def extract_state(self,
                     positions: np.ndarray,  # Current positions
                     prev_positions: np.ndarray,  # Historical positions
                     prev_velocities: np.ndarray,  # Historical velocities
                     cluster_info: Dict[str, any],
                     prev_states: List[Dict[str, any]] = None
                     ) -> BehaviouralState:
        """Extract complete behavioural state."""
        # Extract individual motion features
        velocities, accelerations, turn_rates, curvatures = self.extract_individual_motion(
            positions, prev_positions, prev_velocities
        )
        
        # Extract movement history features
        movement_history, stop_go_patterns = self.extract_movement_history(
            prev_positions
        )
        
        # Extract colony-level features
        cluster_config, neighbor_distances, activity_level, spatial_distribution = \
            self.extract_colony_state(positions, cluster_info)
        
        # Extract temporal context if previous states available
        if prev_states:
            state_changes, transition_rates, time_features = self.extract_temporal_context(
                cluster_config, prev_states
            )
        else:
            state_changes = np.zeros(1)
            transition_rates = {'activity_change_rate': 0, 'cluster_change_rate': 0}
            time_features = {'activity_trend': 0, 'clustering_trend': 0}
        
        return BehaviouralState(
            velocities=velocities,
            accelerations=accelerations,
            turn_rates=turn_rates,
            curvatures=curvatures,
            movement_history=movement_history,
            stop_go_patterns=stop_go_patterns,
            cluster_config=cluster_config,
            neighbor_distances=neighbor_distances,
            activity_level=activity_level,
            spatial_distribution=spatial_distribution,
            state_changes=state_changes,
            transition_rates=transition_rates,
            time_features=time_features
        )
    
    def state_to_vector(self, state: BehaviouralState) -> np.ndarray:
        """Convert behavioural state to feature vector for dimensionality reduction."""
        features = []
        
        # Helper function to safely compute statistics
        def safe_stat(func, arr, axis=None):
            if isinstance(arr, np.ndarray) and arr.size > 0:
                # Replace inf and -inf with nan
                arr = np.where(np.isinf(arr), np.nan, arr)
                # Compute stat, defaulting to 0 if all values are nan
                result = func(arr[~np.isnan(arr)], axis=axis) if np.any(~np.isnan(arr)) else 0
                # Ensure result is finite
                return 0 if np.isinf(result) else result
            return 0
        
        # Individual motion features
        features.extend([
            safe_stat(np.mean, np.linalg.norm(state.velocities, axis=1)),
            safe_stat(np.std, np.linalg.norm(state.velocities, axis=1)),
            safe_stat(np.mean, np.linalg.norm(state.accelerations, axis=1)),
            safe_stat(np.std, np.linalg.norm(state.accelerations, axis=1)),
            safe_stat(np.mean, state.turn_rates),
            safe_stat(np.std, state.turn_rates),
            safe_stat(np.mean, state.curvatures),
            safe_stat(np.std, state.curvatures)
        ])
        
        # Movement history features
        features.extend([
            safe_stat(np.mean, state.movement_history),
            safe_stat(np.std, state.movement_history),
            safe_stat(np.mean, state.stop_go_patterns),
            safe_stat(np.std, state.stop_go_patterns)
        ])
        
        # Colony-level features
        features.extend([
            safe_stat(float, state.cluster_config['n_clusters']),
            safe_stat(np.mean, state.cluster_config['cluster_sizes']),
            safe_stat(float, state.cluster_config['mean_density']),
            safe_stat(float, state.activity_level),
            safe_stat(float, state.spatial_distribution['area']),
            safe_stat(float, state.spatial_distribution['density'])
        ])
        
        # Neighbor distance features
        features.extend([
            safe_stat(np.mean, state.neighbor_distances),
            safe_stat(np.std, state.neighbor_distances),
            safe_stat(np.min, state.neighbor_distances),
            safe_stat(np.max, state.neighbor_distances)
        ])
        
        # Temporal features
        features.extend([
            safe_stat(np.mean, state.state_changes),
            safe_stat(float, state.transition_rates['activity_change_rate']),
            safe_stat(float, state.transition_rates['cluster_change_rate']),
            safe_stat(float, state.time_features['activity_trend']),
            safe_stat(float, state.time_features['clustering_trend'])
        ])
        
        return np.array(features)
    
    def reduce_dimensionality(self, state_vectors: List[np.ndarray]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reduce dimensionality of state vectors using PCA."""
        # Scale the features
        scaled_vectors = self.scaler.fit_transform(state_vectors)
        
        # Apply PCA
        reduced_vectors = self.pca.fit_transform(scaled_vectors)
        
        # Analyze PCA components
        feature_names = [
            'mean_velocity', 'std_velocity', 
            'mean_acceleration', 'std_acceleration',
            'mean_turn_rate', 'std_turn_rate',
            'mean_curvature', 'std_curvature',
            'mean_movement', 'std_movement',
            'mean_stop_go', 'std_stop_go',
            'n_clusters', 'mean_cluster_size', 'cluster_density',
            'activity_level', 'spatial_area', 'spatial_density',
            'mean_neighbor_dist', 'std_neighbor_dist',
            'min_neighbor_dist', 'max_neighbor_dist',
            'mean_state_change', 'activity_change_rate',
            'cluster_change_rate', 'activity_trend', 'clustering_trend'
        ]
        
        # Get component loadings and explained variance
        loadings = self.pca.components_
        explained_variance = self.pca.explained_variance_ratio_
        
        # Analyze top contributors for each component
        pca_analysis = {
            'feature_names': feature_names,
            'loadings': loadings,
            'explained_variance': explained_variance,
            'top_contributors': {}
        }
        
        for i, component in enumerate(loadings[:3]):  # Analyze first 3 components
            sorted_idx = np.argsort(np.abs(component))[::-1]
            pca_analysis['top_contributors'][f'PC{i+1}'] = [
                (feature_names[idx], component[idx])
                for idx in sorted_idx[:5]  # Get top 5 contributors
            ]
        
        return reduced_vectors, pca_analysis
    
    def analyse_state_space(self, 
                          reduced_states: np.ndarray,
                          n_regions: int = 5
                          ) -> Tuple[np.ndarray, List[List[int]]]:
        """Analyse the structure of the behavioural state space."""
        from sklearn.cluster import DBSCAN
        
        # Identify dense regions in state space
        clustering = DBSCAN(eps=0.5, min_samples=5).fit(reduced_states)
        labels = clustering.labels_
        
        # Group points by region
        regions = [[] for _ in range(max(labels) + 1)]
        for i, label in enumerate(labels):
            if label >= 0:
                regions[label].append(i)
        
        # Calculate transition probabilities between regions
        n_regions = len(regions)
        transitions = np.zeros((n_regions, n_regions))
        
        for t in range(len(reduced_states) - 1):
            current_label = labels[t]
            next_label = labels[t + 1]
            if current_label >= 0 and next_label >= 0:
                transitions[current_label, next_label] += 1
        
        # Normalize transitions
        row_sums = transitions.sum(axis=1, keepdims=True)
        transitions = np.divide(transitions, row_sums, where=row_sums!=0)
        
        return transitions, regions

class BehaviouralTrajectoryAnalyser:
    """Analyse trajectories through behavioural state space."""
    
    def __init__(self, reduced_states: np.ndarray):
        self.reduced_states = reduced_states
    
    def compute_density(self) -> gaussian_kde:
        """Compute the density of states in the reduced space."""
        return gaussian_kde(self.reduced_states.T)
    
    def find_common_paths(self, 
                         window_size: int = 60,  # 1 second at 60fps
                         min_occurrences: int = 3,
                         batch_size: int = 1000) -> List[np.ndarray]:
        """
        Find common trajectories through state space using sliding windows and batched processing.
        
        Args:
            window_size: Size of window for trajectory segments
            min_occurrences: Minimum number of times a pattern must occur
            batch_size: Size of batches for processing
            
        Returns:
            List of common trajectory segments
        """
        from sklearn.cluster import DBSCAN
        import gc  # For garbage collection
        
        n_timesteps = len(self.reduced_states)
        common_paths = []
        
        # Process in batches with progress bar
        n_batches = (n_timesteps - window_size) // batch_size + 1
        for batch_start in tqdm(range(0, n_timesteps - window_size, batch_size),
                               desc="Finding common paths",
                               total=n_batches):
            batch_end = min(batch_start + batch_size, n_timesteps - window_size)
            
            # Extract segments for this batch
            batch_segments = []
            for i in range(batch_start, batch_end):
                segment = self.reduced_states[i:i + window_size]
                batch_segments.append(segment)
            
            # Convert to array and reshape for clustering
            segments_array = np.array(batch_segments)
            n_segments = len(segments_array)
            reshaped_segments = segments_array.reshape(n_segments, -1)
            
            # Cluster segments in this batch
            clustering = DBSCAN(eps=0.5, min_samples=min_occurrences).fit(reshaped_segments)
            labels = clustering.labels_
            
            # Extract representatives from this batch
            unique_labels = set(labels)
            for label in unique_labels:
                if label >= 0:  # Ignore noise points
                    cluster_segments = segments_array[labels == label]
                    representative = np.mean(cluster_segments, axis=0)
                    common_paths.append(representative)
            
            # Clean up memory
            del segments_array, reshaped_segments
            gc.collect()
        
        return common_paths
    
    def compute_transition_probabilities(self, n_bins: int = 5, max_dims: int = 3) -> np.ndarray:
        """
        Compute transition probabilities between discretized regions of state space.
        
        Args:
            n_bins: Number of bins for discretizing each dimension
            max_dims: Maximum number of dimensions to consider
            
        Returns:
            Transition probability matrix
        """
        # Use only first max_dims dimensions
        states_subset = self.reduced_states[:, :max_dims]
        
        # Discretize the state space
        discretized_states = []
        for dim in range(states_subset.shape[1]):
            bins = np.linspace(
                states_subset[:, dim].min(),
                states_subset[:, dim].max(),
                n_bins + 1
            )
            discretized_dim = np.digitize(states_subset[:, dim], bins) - 1
            discretized_states.append(discretized_dim)
        
        discretized_states = np.array(discretized_states).T
        
        # Calculate state transitions
        n_states = n_bins ** states_subset.shape[1]
        transitions = np.zeros((n_states, n_states))
        
        for t in range(len(discretized_states) - 1):
            try:
                current_state = np.ravel_multi_index(
                    discretized_states[t], 
                    [n_bins] * states_subset.shape[1]
                )
                next_state = np.ravel_multi_index(
                    discretized_states[t + 1],
                    [n_bins] * states_subset.shape[1]
                )
                transitions[current_state, next_state] += 1
            except ValueError:
                # Skip invalid state transitions
                continue
        
        # Normalize to get probabilities
        row_sums = transitions.sum(axis=1, keepdims=True)
        transition_probs = np.divide(transitions, row_sums, 
                                   where=row_sums != 0)
        
        return transition_probs
    
    def identify_behavioral_motifs(self, 
                                 window_size: int = 60,
                                 n_motifs: int = 5,
                                 batch_size: int = 1000) -> List[np.ndarray]:
        """
        Identify recurring behavioral motifs using time series motif discovery with batched processing.
        
        Args:
            window_size: Size of window for motif detection
            n_motifs: Number of motifs to identify
            batch_size: Size of batches for processing
            
        Returns:
            List of behavioral motifs
        """
        from scipy.signal import find_peaks
        
        motifs = []
        n_timesteps = len(self.reduced_states)
        
        # Process each dimension separately to reduce memory usage
        for dim in range(self.reduced_states.shape[1]):
            dim_motifs = []
            covered_indices = set()
            
            # Process in batches
            for batch_start in range(0, n_timesteps - window_size, batch_size):
                batch_end = min(batch_start + batch_size, n_timesteps - window_size)
                
                # Extract segments for this batch
                batch_segments = np.array([
                    self.reduced_states[i:i + window_size, dim] 
                    for i in range(batch_start, batch_end)
                ])
                
                # Find similar segments within batch
                for i in range(len(batch_segments)):
                    if len(dim_motifs) >= n_motifs:
                        break
                        
                    if (batch_start + i) in covered_indices:
                        continue
                    
                    # Compare with other segments
                    for j in range(i + window_size, len(batch_segments)):
                        if (batch_start + j) in covered_indices:
                            continue
                            
                        distance = np.linalg.norm(batch_segments[i] - batch_segments[j])
                        if distance < 0.5:  # Similarity threshold
                            # Add motif pair
                            motif_pair = (batch_segments[i], batch_segments[j])
                            dim_motifs.append(motif_pair)
                            
                            # Mark indices as covered
                            for idx in range(batch_start + i, batch_start + i + window_size):
                                covered_indices.add(idx)
                            for idx in range(batch_start + j, batch_start + j + window_size):
                                covered_indices.add(idx)
                            
                            break
                    
                    if len(dim_motifs) >= n_motifs:
                        break
            
            motifs.append(dim_motifs)
        
        return motifs


def analyse_colony_clustering(data, eps_mm=10, min_samples=3, max_centroid_distance=50):
    """
    Optimized clustering analysis using vectorized operations.
    
    Args:
        data: DataFrame with MultiIndex columns (ant_id, coordinate)
        eps_mm: Clustering radius in millimeters
        min_samples: Minimum samples to form a cluster
        max_centroid_distance: Maximum distance between centroids to consider it the same cluster
    
    Returns:
        Dictionary containing clustering statistics over time
    """
    from sklearn.cluster import DBSCAN
    from scipy.spatial.distance import cdist
    
    eps_pixels = eps_mm * PIXELS_PER_MM
    
    # Pre-allocate results dictionary
    clustering_stats = {
        'n_clusters': [],
        'cluster_sizes': [],
        'isolated_ants': [],
        'mean_cluster_density': [],
        'positions': [],
        'labels': [],
        'cluster_ids': [],
        'centroids': []
    }
    
    # Extract all positions at once using vectorized operations
    ant_ids = sorted(list(set(idx[0] for idx in data.columns)))
    frame_indices = data.index.values
    
    # Create position array with shape (n_frames, n_ants, 2)
    positions_array = np.full((len(frame_indices), len(ant_ids), 2), np.nan)
    for i, ant_id in enumerate(ant_ids):
        positions_array[:, i, 0] = data[(ant_id, 'x')].values
        positions_array[:, i, 1] = data[(ant_id, 'y')].values
    
    # Track cluster IDs
    next_cluster_id = 0
    previous_centroids = {}
    
    # Process each frame
    for t in tqdm(range(len(frame_indices)), desc="Analysing clustering behaviour"):
        # Get valid positions for this frame
        frame_positions = positions_array[t]
        valid_mask = ~np.isnan(frame_positions).any(axis=1)
        valid_positions = frame_positions[valid_mask]
        
        if len(valid_positions) < min_samples:
            # Handle empty frame case vectorially
            clustering_stats['n_clusters'].append(0)
            clustering_stats['cluster_sizes'].append([])
            clustering_stats['mean_cluster_density'].append(0)
            clustering_stats['isolated_ants'].append(len(valid_positions))
            clustering_stats['positions'].append([])
            clustering_stats['labels'].append([])
            clustering_stats['cluster_ids'].append([])
            clustering_stats['centroids'].append({})
            continue
        
        # Perform DBSCAN clustering
        clustering = DBSCAN(eps=eps_pixels, min_samples=min_samples).fit(valid_positions)
        labels = clustering.labels_
        
        # Calculate centroids using vectorized operations
        unique_labels = np.unique(labels[labels >= 0])
        current_centroids = {}
        current_id_map = {}
        
        if len(unique_labels) > 0:
            # Vectorized centroid calculation
            centroids = np.array([
                np.mean(valid_positions[labels == label], axis=0)
                for label in unique_labels
            ])
            
            # Vectorized distance calculation between all centroids
            if previous_centroids:
                prev_centroid_array = np.array(list(previous_centroids.values()))
                distances = cdist(centroids, prev_centroid_array)
                
                # Find best matches using vectorized operations
                for i, label in enumerate(unique_labels):
                    if distances[i].size > 0:
                        min_dist_idx = np.argmin(distances[i])
                        min_distance = distances[i, min_dist_idx]
                        
                        if min_distance < max_centroid_distance:
                            matched_id = list(previous_centroids.keys())[min_dist_idx]
                            current_id_map[label] = matched_id
                            current_centroids[matched_id] = centroids[i]
                            continue
                    
                    # No match found, assign new ID
                    current_id_map[label] = next_cluster_id
                    current_centroids[next_cluster_id] = centroids[i]
                    next_cluster_id += 1
        
        # Update labels with consistent IDs using vectorized operation
        consistent_labels = np.array([current_id_map.get(label, -1) if label >= 0 else -1 
                                    for label in labels])
        
        # Calculate statistics using vectorized operations
        unique_clusters = len(current_centroids)
        if unique_clusters > 0:
            cluster_sizes = np.bincount(consistent_labels[consistent_labels >= 0])
            cluster_areas = np.full(unique_clusters, np.pi * (eps_pixels ** 2))
            densities = cluster_sizes / cluster_areas
            mean_density = np.mean(densities)
        else:
            cluster_sizes = []
            mean_density = 0
        
        # Store results
        clustering_stats['positions'].append(valid_positions.tolist())
        clustering_stats['labels'].append(consistent_labels.tolist())
        clustering_stats['cluster_ids'].append(list(current_centroids.keys()))
        clustering_stats['centroids'].append(current_centroids)
        clustering_stats['n_clusters'].append(unique_clusters)
        clustering_stats['cluster_sizes'].append(cluster_sizes.tolist() if unique_clusters > 0 else [])
        clustering_stats['mean_cluster_density'].append(mean_density)
        clustering_stats['isolated_ants'].append(np.sum(consistent_labels == -1))
        
        # Update previous centroids
        previous_centroids = current_centroids.copy()
    
    return clustering_stats


def generate_pc_interpretation(component_loadings: List[Tuple[str, float]]) -> str:
    """Generate interpretation of a principal component based on its top loadings."""
    # Group features by type
    spatial_features = {'mean_neighbor_dist', 'max_neighbor_dist', 'min_neighbor_dist', 'std_neighbor_dist'}
    movement_features = {'mean_movement', 'std_movement', 'mean_velocity', 'mean_acceleration'}
    turning_features = {'mean_curvature', 'std_curvature', 'mean_turn_rate'}
    clustering_features = {'n_clusters', 'mean_cluster_size', 'cluster_density'}
    
    # Get absolute loadings and their signs
    feature_contributions = {
        'spatial': [],
        'movement': [],
        'turning': [],
        'clustering': []
    }
    
    for feature, loading in component_loadings:
        if feature in spatial_features:
            feature_contributions['spatial'].append((feature, loading))
        elif feature in movement_features:
            feature_contributions['movement'].append((feature, loading))
        elif feature in turning_features:
            feature_contributions['turning'].append((feature, loading))
        elif feature in clustering_features:
            feature_contributions['clustering'].append((feature, loading))
    
    # Determine dominant aspects
    interpretation_parts = []
    for aspect, contributions in feature_contributions.items():
        if contributions:
            # Calculate mean absolute contribution for this aspect
            mean_contrib = np.mean([abs(loading) for _, loading in contributions])
            if mean_contrib > 0.2:  # Threshold for significance
                # Determine if this aspect generally increases or decreases
                direction = "increases" if np.mean([loading for _, loading in contributions]) > 0 else "decreases"
                interpretation_parts.append(f"{aspect} {direction}")
    
    return " and ".join(interpretation_parts) if interpretation_parts else "mixed effects"

def integrate_state_space_analysis(processed_data: Dict,
                                clustering_stats: Dict,
                                fps: float = 60.0,
                                scale: int = 2) -> Dict:
    """
    Integrate behavioural state space analysis with existing processed data.
    
    Args:
        processed_data: Dictionary of processed ant data
        clustering_stats: Dictionary of clustering statistics
        fps: Frame rate of the original video
        scale: Temporal downsampling factor
    
    Returns:
        Dictionary containing state space analysis results
    """
    print("Initializing state space analysis...")
    
    # Initialize state extractor
    effective_fps = fps / scale if scale else fps
    state_extractor = BehaviouralStateExtractor(fps=effective_fps)
    
    # Extract states for each timestep
    states = []
    state_vectors = []
    
    # Get all ant IDs and timesteps
    ant_ids = list(processed_data.keys())
    n_ants = len(ant_ids)
    
    # Determine number of timesteps from clustering stats
    n_timesteps = len(clustering_stats['positions'])
    
    print("Pre-processing position and velocity data...")
    # Pre-allocate arrays for positions and velocities
    positions_history = np.zeros((n_timesteps, n_ants, 2))
    velocities_history = np.zeros((n_timesteps, n_ants, 2))
    
    # Fill position and velocity histories with progress bar
    for t in tqdm(range(n_timesteps), desc="Building position/velocity history"):
        for i, ant_id in enumerate(ant_ids):
            # Get position data
            if t < len(clustering_stats['positions']):
                positions = np.array(clustering_stats['positions'][t])
                if i < len(positions):
                    positions_history[t, i] = positions[i]
            
            # Calculate velocities if possible
            if t > 0:
                velocities_history[t, i] = (positions_history[t, i] - 
                                          positions_history[t-1, i]) / (1/effective_fps)
    
    # Process each timestep
    window_size = int(effective_fps)  # 1 second window
    
    print("Extracting behavioral states...")
    for t in tqdm(range(window_size, n_timesteps), desc="Processing timesteps"):
        # Extract current positions and history
        current_positions = positions_history[t]
        position_window = positions_history[t-window_size:t]
        velocity_window = velocities_history[t-window_size:t]
        
        # Get clustering info for this timestep
        cluster_info = {
            'n_clusters': clustering_stats['n_clusters'][t],
            'cluster_sizes': clustering_stats['cluster_sizes'][t],
            'mean_cluster_density': clustering_stats['mean_cluster_density'][t]
        }
        
        # Extract state
        state = state_extractor.extract_state(
            current_positions,
            position_window,
            velocity_window,
            cluster_info,
            prev_states=states[-window_size:] if len(states) >= window_size else None
        )
        
        states.append(state)
        state_vectors.append(state_extractor.state_to_vector(state))
    
    print("Reducing dimensionality...")
    # Reduce dimensionality of state vectors
    state_vectors = np.array(state_vectors)
    reduced_states, pca_analysis = state_extractor.reduce_dimensionality(state_vectors)
    
    # Print PCA analysis results
    print("\nPCA Analysis Results:")
    print(f"Total variance explained by first 3 components: {sum(pca_analysis['explained_variance'][:3])*100:.1f}%")
    print("\nComponent-wise explained variance:")
    for i, var in enumerate(pca_analysis['explained_variance'][:3]):
        print(f"PC{i+1}: {var*100:.1f}%")
    
    # Generate dynamic interpretations based on PCA results
    interpretations = {
        f'PC{i+1}': generate_pc_interpretation(contributors)
        for i, (pc, contributors) in enumerate(pca_analysis['top_contributors'].items())
    }
    
    for pc, contributors in pca_analysis['top_contributors'].items():
        print(f"\n{pc} - {interpretations[pc]}:")
        print("Feature                  Loading    Effect")
        print("-" * 45)
        for feature, loading in contributors:
            direction = "increases" if loading > 0 else "decreases"
            print(f"{feature:<22} {abs(loading):>7.3f}    {direction}")
    
    print("Analyzing trajectories...")
    # Analyse trajectories through state space
    trajectory_analyser = BehaviouralTrajectoryAnalyser(reduced_states)
    
    print("Finding common behavioral patterns...")
    # Find common behavioural patterns
    common_paths = trajectory_analyser.find_common_paths()
    
    print("Computing transition probabilities...")
    transition_probs = trajectory_analyser.compute_transition_probabilities()
    
    print("Identifying behavioral motifs...")
    behavioral_motifs = trajectory_analyser.identify_behavioral_motifs()
    
    print("Calculating state space density...")
    # Calculate state space density
    state_density = trajectory_analyser.compute_density()
    
    return {
        'states': states,
        'state_vectors': state_vectors,
        'reduced_states': reduced_states,
        'common_paths': common_paths,
        'transition_probabilities': transition_probs,
        'behavioral_motifs': behavioral_motifs,
        'state_density': state_density,
        'pca_analysis': pca_analysis
    }


def visualise_state_space(analysis_results: Dict,
                         save_path: Optional[str] = None):
    """
    Visualise the behavioural state space and trajectories.
    
    Args:
        analysis_results: Results from state space analysis
        save_path: Optional path to save visualizations
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 10))
    
    # 3D state space plot
    ax1 = fig.add_subplot(121, projection='3d')
    reduced_states = analysis_results['reduced_states']
    
    # Plot points colored by density (using only first 3 dimensions for consistency)
    density_estimator = gaussian_kde(reduced_states[:, :3].T)
    density = density_estimator(reduced_states[:, :3].T)
    scatter = ax1.scatter(reduced_states[:, 0],
                         reduced_states[:, 1],
                         reduced_states[:, 2],
                         c=density,
                         cmap='viridis',
                         alpha=0.6)
    plt.colorbar(scatter, label='State Density')
    
    # Plot common paths
    common_paths = analysis_results['common_paths']
    for path in common_paths:
        ax1.plot(path[:, 0],
                path[:, 1],
                path[:, 2],
                'r-',
                linewidth=2,
                alpha=0.8)
    
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.set_zlabel('PC3')
    ax1.set_title('Behavioural State Space')
    
    # Transition probability heatmap
    ax2 = fig.add_subplot(122)
    transition_probs = analysis_results['transition_probabilities']
    
    # Handle NaN and Inf values
    transition_probs = np.nan_to_num(transition_probs, nan=0.0, posinf=1.0, neginf=0.0)
    
    # Ensure proper normalization
    row_sums = transition_probs.sum(axis=1, keepdims=True)
    transition_probs = np.divide(transition_probs, row_sums, 
                               where=row_sums!=0, 
                               out=np.zeros_like(transition_probs))
    
    im = ax2.imshow(transition_probs,
                    cmap='Blues',
                    interpolation='nearest',
                    vmin=0.0,
                    vmax=1.0)
    plt.colorbar(im, label='Transition Probability')
    ax2.set_title('State Transition Probabilities')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()


def evaluate_behavioral_similarity(real_trajectories: np.ndarray,
                                artificial_trajectories: np.ndarray,
                                n_components: int = 3) -> Dict[str, float]:
    """
    Evaluate similarity between real and artificial ant trajectories in state space.
    
    Args:
        real_trajectories: State space trajectories of real ants
        artificial_trajectories: State space trajectories of artificial ants
        n_components: Number of principal components to use
        
    Returns:
        Dictionary of similarity metrics
    """
    from scipy.stats import wasserstein_distance
    from sklearn.metrics import pairwise_distances
    
    metrics = {}
    
    # Truncate to use only specified number of components
    real = real_trajectories[:, :n_components]
    artificial = artificial_trajectories[:, :n_components]
    
    # Calculate state space coverage
    real_density = gaussian_kde(real.T)
    artificial_density = gaussian_kde(artificial.T)
    
    # Evaluate on a grid of points
    grid_points = np.vstack([r.flatten() for r in np.meshgrid(
        *[np.linspace(min(real.min(), artificial.min()),
                      max(real.max(), artificial.max()),
                      20)] * n_components
    )]).T
    
    real_density_values = real_density(grid_points.T)
    artificial_density_values = artificial_density(grid_points.T)
    
    # Calculate Earth Mover's Distance
    metrics['emd'] = wasserstein_distance(
        real_density_values, artificial_density_values)
    
    # Calculate average minimum distance between trajectories
    real_artificial_distances = pairwise_distances(real, artificial)
    metrics['avg_min_distance'] = np.mean([
        min(row) for row in real_artificial_distances
    ])
    
    # Calculate behavioral coverage ratio
    real_hull = ConvexHull(real)
    artificial_hull = ConvexHull(artificial)
    metrics['coverage_ratio'] = artificial_hull.volume / real_hull.volume
    
    # Calculate transition similarity
    real_transitions = np.zeros((20, 20))
    artificial_transitions = np.zeros((20, 20))
    
    for trajectories, transitions in [(real, real_transitions),
                                    (artificial, artificial_transitions)]:
        for t in range(len(trajectories) - 1):
            current_state = np.digitize(trajectories[t], 
                                      bins=20) - 1
            next_state = np.digitize(trajectories[t + 1],
                                   bins=20) - 1
            transitions[current_state[0], next_state[0]] += 1
    
    # Normalize transition matrices
    real_transitions /= real_transitions.sum()
    artificial_transitions /= artificial_transitions.sum()
    
    # Calculate transition matrix difference
    metrics['transition_diff'] = np.linalg.norm(
        real_transitions - artificial_transitions)
    
    return metrics


def main():
    """Main function to run the complete behavioural state space analysis."""
    import argparse
    import os
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description='Analyse ant behaviour using state space approach')
    parser.add_argument('--data_dir', type=str, default='data/2023_2/',
                      help='Directory containing the data')
    parser.add_argument('--input_file', type=str, 
                      default='KA050_processed_10cm_5h_20230614.pkl.xz',
                      help='Input data file name')
    parser.add_argument('--save_dir', type=str, default='results/state_space',
                      help='Directory to save results')
    parser.add_argument('--fps', type=float, default=60.0,
                      help='Frame rate of original video')
    parser.add_argument('--scale', type=int, default=2,
                      help='Temporal downsampling factor')
    parser.add_argument('--debug', action='store_true',
                      help='Run in debug mode with subset of data')
    parser.add_argument('--time_window_start', type=float, default=None,
                      help='Start time of analysis window in minutes')
    parser.add_argument('--time_window_end', type=float, default=None,
                      help='End time of analysis window in minutes')
    args = parser.parse_args()
    
    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading data...")
    time_window = None
    if args.time_window_start is not None and args.time_window_end is not None:
        time_window = (args.time_window_start, args.time_window_end)
        print(f"Analyzing time window: {args.time_window_start}-{args.time_window_end} minutes")
    
    data = load_data(args.data_dir, args.input_file, 
                    scale=args.scale, debug=args.debug,
                    time_window=time_window)
    
    # Process ant data
    print("\nProcessing ant trajectories...")
    processed_data = process_ant_data(data)
    
    # Analyse colony clustering
    print("\nAnalysing colony clustering...")
    clustering_stats = analyse_colony_clustering(data)
    
    # Perform state space analysis
    print("\nPerforming state space analysis...")
    analysis_results = integrate_state_space_analysis(
        processed_data,
        clustering_stats,
        fps=args.fps,
        scale=args.scale
    )
    
    # Save results
    print("\nSaving results...")
    
    # Create time window specific filename suffix
    suffix = ''
    if args.time_window_start is not None and args.time_window_end is not None:
        suffix = f'_{int(args.time_window_start)}-{int(args.time_window_end)}min'
    
    # Save state vectors and reduced states
    np.save(save_dir / f'state_vectors{suffix}.npy', 
            analysis_results['state_vectors'])
    np.save(save_dir / f'reduced_states{suffix}.npy', 
            analysis_results['reduced_states'])
    
    # Save transition probabilities
    np.save(save_dir / 'transition_probabilities.npy',
            analysis_results['transition_probabilities'])
    
    # Save common paths and motifs
    np.save(save_dir / 'common_paths.npy',
            analysis_results['common_paths'])
    np.save(save_dir / 'behavioral_motifs.npy',
            analysis_results['behavioral_motifs'])
    
    # Create visualizations
    print("\nGenerating visualizations...")
    visualise_state_space(analysis_results,
                         save_path=str(save_dir / 'state_space_visualization.png'))
    
    print("\nAnalysis complete! Results saved to:", save_dir)


if __name__ == "__main__":
    main()
