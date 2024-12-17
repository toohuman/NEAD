import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import seaborn as sns
from scipy.stats import gaussian_kde

@dataclass
class StateCharacteristics:
    """Container for state-specific characteristics"""
    velocity_stats: Dict[str, float]  # mean, std, median
    acceleration_stats: Dict[str, float]
    angular_velocity_stats: Dict[str, float]
    social_stats: Dict[str, float]  # nn distances, local density
    duration: float  # average duration in this state
    transition_probs: Dict[int, float]  # probabilities of transitioning to other states
    
class StateAnalyser:
    """Analyse and classify behavioural states"""
    
    def __init__(self, 
                 velocity_thresholds: Tuple[float, float] = (0.5, 2.0),  # mm/s
                 nn_thresholds: Tuple[float, float] = (10, 30),  # mm
                 density_thresholds: Tuple[float, float] = (0.1, 0.3)):  # ants per mm^2
        """
        Initialize the state analyser with thresholds for classification
        
        Args:
            velocity_thresholds: (low, high) thresholds for velocity classification
            nn_thresholds: (close, far) thresholds for nearest neighbour classification
            density_thresholds: (sparse, dense) thresholds for local density classification
        """
        self.velocity_thresholds = velocity_thresholds
        self.nn_thresholds = nn_thresholds
        self.density_thresholds = density_thresholds
        self.state_characteristics = {}
        
    def compute_state_characteristics(self, 
                                   processed_data: Dict,
                                   clustering_stats: Dict) -> Dict[int, StateCharacteristics]:
        """
        Compute detailed characteristics for each observed state
        
        Args:
            processed_data: Dictionary containing processed ant trajectories
            clustering_stats: Dictionary containing clustering analysis results
            
        Returns:
            Dictionary mapping state IDs to their characteristics
        """
        state_chars = {}
        
        # For each ant's data
        for ant_id, ant_data in processed_data.items():
            traj_features = ant_data['trajectory_features']
            social_features = ant_data['social_features']
            
            # Compute velocity magnitudes
            velocities = np.linalg.norm(traj_features.velocities, axis=1)
            accelerations = np.linalg.norm(traj_features.accelerations, axis=1)
            
            # Group data into states based on characteristics
            for i in range(len(velocities)):
                if np.isnan(velocities[i]):
                    continue
                    
                state_id = self._classify_state(
                    velocity=velocities[i],
                    acceleration=accelerations[i],
                    angular_velocity=traj_features.angular_velocities[i],
                    social_features=social_features[i] if i < len(social_features) else None
                )
                
                if state_id not in state_chars:
                    state_chars[state_id] = []
                    
                state_chars[state_id].append({
                    'velocity': velocities[i],
                    'acceleration': accelerations[i],
                    'angular_velocity': traj_features.angular_velocities[i],
                    'social_features': social_features[i] if i < len(social_features) else None
                })
        
        # Compute summary statistics for each state
        return {
            state_id: self._compute_summary_stats(instances)
            for state_id, instances in state_chars.items()
        }
    
    def _classify_state(self,
                       velocity: float,
                       acceleration: float,
                       angular_velocity: float,
                       social_features: Optional[Dict] = None) -> int:
        """
        Classify a behavioural state based on its characteristics
        
        Returns:
            Integer state ID
        """
        # Movement component (0-2)
        if velocity < self.velocity_thresholds[0]:
            movement_state = 0  # stationary
        elif velocity < self.velocity_thresholds[1]:
            movement_state = 1  # slow
        else:
            movement_state = 2  # fast
            
        # Social component (0-2)
        social_state = 0  # default isolated
        if social_features and 'nn_stats' in social_features:
            nn_dist = social_features['nn_stats'].get('nn_dist_1', float('inf'))
            if nn_dist < self.nn_thresholds[0]:
                social_state = 2  # clustered
            elif nn_dist < self.nn_thresholds[1]:
                social_state = 1  # semi-isolated
        
        # Combine into single state ID (0-8)
        return movement_state * 3 + social_state
    
    def _compute_summary_stats(self, instances: List[Dict]) -> StateCharacteristics:
        """Compute summary statistics for a state"""
        velocities = [inst['velocity'] for inst in instances]
        accelerations = [inst['acceleration'] for inst in instances]
        angular_velocities = [inst['angular_velocity'] for inst in instances]
        
        # Social statistics if available
        nn_distances = []
        densities = []
        for inst in instances:
            if inst['social_features'] and 'nn_stats' in inst['social_features']:
                nn_distances.append(inst['social_features']['nn_stats'].get('nn_dist_1', np.nan))
            if inst['social_features'] and 'densities' in inst['social_features']:
                densities.append(np.mean(inst['social_features']['densities']))
                
        return StateCharacteristics(
            velocity_stats={
                'mean': np.mean(velocities),
                'std': np.std(velocities),
                'median': np.median(velocities)
            },
            acceleration_stats={
                'mean': np.mean(accelerations),
                'std': np.std(accelerations),
                'median': np.median(accelerations)
            },
            angular_velocity_stats={
                'mean': np.mean(angular_velocities),
                'std': np.std(angular_velocities),
                'median': np.median(angular_velocities)
            },
            social_stats={
                'mean_nn_dist': np.nanmean(nn_distances) if nn_distances else np.nan,
                'mean_density': np.nanmean(densities) if densities else np.nan
            },
            duration=len(instances) / 60.0,  # approximate duration in seconds
            transition_probs={}  # to be computed separately
        )
    
    def visualise_state_characteristics(self, save_path: Optional[str] = None):
        """Create comprehensive visualisation of state characteristics"""
        n_states = len(self.state_characteristics)
        
        # Create a grid of subplots
        fig = plt.figure(figsize=(15, 10))
        gs = plt.GridSpec(3, 3)
        
        # 1. Velocity distribution by state
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_state_distribution(ax1, 'velocity_stats', 'mean', 'Velocity (mm/s)')
        
        # 2. Social distribution by state
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_state_distribution(ax2, 'social_stats', 'mean_nn_dist', 'Mean NN Distance (mm)')
        
        # 3. State duration distribution
        ax3 = fig.add_subplot(gs[0, 2])
        durations = [chars.duration for chars in self.state_characteristics.values()]
        ax3.hist(durations, bins=20)
        ax3.set_xlabel('State Duration (s)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('State Duration Distribution')
        
        # 4. State transition network
        ax4 = fig.add_subplot(gs[1:, :])
        self._plot_transition_network(ax4)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def _plot_state_distribution(self, ax, stat_category: str, stat_name: str, xlabel: str):
        """Plot distribution of a particular statistic across states"""
        values = [getattr(chars, stat_category)[stat_name] 
                 for chars in self.state_characteristics.values()]
        states = list(self.state_characteristics.keys())
        
        ax.scatter(states, values)
        ax.set_xlabel('State ID')
        ax.set_ylabel(xlabel)
        ax.set_title(f'{xlabel} by State')
    
    def _plot_transition_network(self, ax):
        """Plot state transition network"""
        # Get transition probabilities
        transitions = np.zeros((len(self.state_characteristics), 
                              len(self.state_characteristics)))
        
        for state_id, chars in self.state_characteristics.items():
            for next_state, prob in chars.transition_probs.items():
                transitions[state_id, next_state] = prob
        
        # Plot heatmap
        sns.heatmap(transitions, ax=ax, cmap='YlOrRd')
        ax.set_xlabel('Next State')
        ax.set_ylabel('Current State')
        ax.set_title('State Transition Probabilities')

def compute_state_transitions(processed_data: Dict,
                            state_analyser: StateAnalyser,
                            time_window: float = 1.0) -> Dict[Tuple[int, int], float]:
    """
    Compute transition probabilities between states
    
    Args:
        processed_data: Dictionary containing processed ant trajectories
        state_analyser: Initialized StateAnalyser instance
        time_window: Time window (in seconds) for considering transitions
    
    Returns:
        Dictionary mapping (state1, state2) pairs to transition probabilities
    """
    transitions = {}
    
    for ant_id, ant_data in processed_data.items():
        traj_features = ant_data['trajectory_features']
        social_features = ant_data['social_features']
        
        # Compute state sequence
        velocities = np.linalg.norm(traj_features.velocities, axis=1)
        accelerations = np.linalg.norm(traj_features.accelerations, axis=1)
        
        # Get state sequence
        states = []
        for i in range(len(velocities)):
            if np.isnan(velocities[i]):
                continue
                
            state = state_analyser._classify_state(
                velocity=velocities[i],
                acceleration=accelerations[i],
                angular_velocity=traj_features.angular_velocities[i],
                social_features=social_features[i] if i < len(social_features) else None
            )
            states.append(state)
        
        # Count transitions
        for i in range(len(states)-1):
            transition = (states[i], states[i+1])
            transitions[transition] = transitions.get(transition, 0) + 1
    
    # Convert counts to probabilities
    total_transitions = sum(transitions.values())
    return {k: v/total_transitions for k, v in transitions.items()}


# Example usage:
# analyser = StateAnalyser()
# characteristics = analyser.compute_state_characteristics(processed_data, clustering_stats)
# transitions = compute_state_transitions(processed_data, analyser)
# analyser.visualize_state_characteristics('state_analysis.png')