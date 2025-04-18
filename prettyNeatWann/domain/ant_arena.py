"""
AntArenaEnv: A single-agent ant environment with a circular arena using pybox2d and gymnasium.
- The ant moves in a circular arena and cannot escape the boundary.
- Other ants (obstacles) are loaded from data or move on scripted paths.
- Collision detection prevents overlap with the arena boundary and other ants.

To expand to multi-agent RL, refactor to PettingZoo or use Gym multi-agent wrappers.
"""

import lzma
import os
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from Box2D import (b2World, b2CircleShape, b2FixtureDef, b2BodyDef, b2_dynamicBody, b2_staticBody)

# Optional: for quick rendering (matplotlib)
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

if __name__ == "__main__":
    # Relative path during development
    DATA_DIRECTORY = "../data/2023_2/"
else:
    # Relative path when training
    DATA_DIRECTORY = "data/2023_2/"
FILE_PREFIX = "KA050_10cm_5h_20230614"
FILE_SUFFIX = "smoothed"
PP_FILE_PREFIX = "KA050_processed"
TRAJECTORIES_FILE = '_'.join([PP_FILE_PREFIX, *FILE_PREFIX.split('_')[1:], FILE_SUFFIX]) + '.pkl.xz'

# Add imports for data loading and transformation
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


def translate_data_to_sim_space(data, arena_dim):
    data_len = len(data)
    logger.info(msg=f"Ant trail data loaded. Total records: {data_len}")
    arena_bb = find_bounding_box(data)
    origin_arena = calculate_circle(*arena_bb)

    translation, scale = circle_transformation(origin_arena, arena_dim)

    logger.info(msg=f"Processing data now. This will take a while...")
    apply_transform_scale(data, translation, scale)
    logger.info(msg=f"Finished processing.")

    logger.info(msg=f"Translation: {translation}, scale: {scale}")
    logger.info(msg=f"Original: ({origin_arena[0][0] + translation[0]}, {origin_arena[0][1] + translation[1]}), scale: {origin_arena[1]*scale}")
    logger.info(msg=f"Simulated: {arena_dim[0]}, scale: {arena_dim[1]}")

    return data

# --- Simulation scale (mm) and rendering scale (pixels/mm) ---
ARENA_DIAMETER_MM = 100.0
ARENA_RADIUS_MM = ARENA_DIAMETER_MM / 2
ANT_RADIUS_MM = 0.7  # More realistic for ant body size
SCALE = 8  # pixels per mm (will be dynamically scaled in render)


def mm_per_sec_to_box2d_velocity(mm_per_s):
    """Convert mm/sec to Box2D world units (which are mm in this env)."""
    return mm_per_s  # 1:1 mapping


class AntArenaEnv(gym.Env):
    """
    Single-agent ant environment in a circular arena using pybox2d.
    World units are millimeters (mm).
    """
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode=None, n_ants=10, arena_radius=ARENA_RADIUS_MM, ant_radius=ANT_RADIUS_MM, data_dir=None, trajectories_file=None):
        super().__init__()
        self.render_mode = render_mode
        self.n_ants = n_ants
        self.arena_radius = arena_radius
        self.ant_radius = ant_radius
        self._setup_spaces()
        self.world = None
        self.agent_body = None
        self.other_ants = []
        self.viewer = None
        self.other_ant_trajectories = None  # Will hold transformed trajectories
        self.data_dir = data_dir or os.path.dirname(os.path.abspath(__file__))
        self.trajectories_file = trajectories_file
        self._load_and_prepare_trajectories()
        self.reset()

    def _setup_spaces(self):
        # Action: [forward_speed, turn_rate] (continuous)
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )
        # Observation: [x, y, theta, ...] (agent position, heading, plus others later)
        self.observation_space = spaces.Box(
            low=np.array([-self.arena_radius, -self.arena_radius, -np.pi], dtype=np.float32),
            high=np.array([self.arena_radius, self.arena_radius, np.pi], dtype=np.float32),
            dtype=np.float32
        )

    def _load_and_prepare_trajectories(self):
        """
        Load and transform real ant trajectories for use in the simulation.
        Only called once at initialization.
        """
        # Load the data
        try:
            df = load_data(self.data_dir, self.trajectories_file)
        except Exception as e:
            print(f"Could not load ant trajectory data: {e}")
            self.other_ant_trajectories = None
            return
        # Inspect the first few rows to understand the coordinate system
        # print(df.head())  # Uncomment to debug
        # Apply coordinate transformation
        # NOTE: You may need to adjust the transform if the data assumes (0,0) is top-left.
        #       translate_data_to_sim_space should be checked for its assumptions.
        try:
            self.other_ant_trajectories = translate_data_to_sim_space(
                df,
                arena_radius=self.arena_radius,
                arena_center=(0, 0),
                flip_y=True  # Set to True if original data uses top-left origin
            )
        except Exception as e:
            print(f"Could not transform ant trajectories: {e}")
            self.other_ant_trajectories = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.world is not None:
            del self.world
        self.world = b2World(gravity=(0, 0), doSleep=True)
        self._create_ants()
        obs = self._get_obs()
        info = {}
        return obs, info

    def _create_ants(self):
        # Create agent ant
        self.agent_body = self.world.CreateDynamicBody(
            position=(0, -self.arena_radius * 0.7),
            angle=np.pi / 2,
            fixtures=b2FixtureDef(
                shape=b2CircleShape(radius=self.ant_radius),
                density=1.0,
                friction=0.3,
                restitution=0.0,
            ),
            linearDamping=0.5,
            angularDamping=0.5,
        )
        # Create obstacle ants (random positions for now)
        self.other_ants = []
        rng = np.random.default_rng()
        for _ in range(self.n_ants - 1):
            while True:
                x, y = rng.uniform(-self.arena_radius * 0.85, self.arena_radius * 0.85, size=2)
                if np.hypot(x, y) < self.arena_radius - 2 * self.ant_radius:
                    break
            body = self.world.CreateDynamicBody(
                position=(x, y),
                angle=rng.uniform(0, 2 * np.pi),
                fixtures=b2FixtureDef(
                    shape=b2CircleShape(radius=self.ant_radius),
                    density=1.0,
                    friction=0.3,
                    restitution=0.0,
                ),
                linearDamping=0.5,
                angularDamping=0.5,
            )
            self.other_ants.append(body)

    def _get_obs(self):
        # For now, just agent position and heading
        pos = self.agent_body.position
        theta = self.agent_body.angle
        return np.array([pos.x, pos.y, theta], dtype=np.float32)

    def _constrain_to_arena(self, body):
        pos = body.position
        dist = np.hypot(pos.x, pos.y)
        max_dist = self.arena_radius - self.ant_radius
        if dist > max_dist:
            # Project back to boundary
            new_x = pos.x * max_dist / dist
            new_y = pos.y * max_dist / dist
            body.position = (new_x, new_y)
            # Remove outward velocity
            vel = body.linearVelocity
            outward = np.array([pos.x, pos.y]) / dist
            v_out = np.dot([vel.x, vel.y], outward)
            if v_out > 0:
                v_tan = np.array([vel.x, vel.y]) - v_out * outward
                body.linearVelocity = (v_tan[0], v_tan[1])

    def step(self, action):
        # Action: [forward_speed, turn_rate] in [-1, 1]
        max_speed = 50.0  # Increased for more visible movement
        max_turn = 2 * np.pi  # Increased for more visible rotation
        forward = float(np.clip(action[0], -1, 1)) * max_speed
        turn = float(np.clip(action[1], -1, 1)) * max_turn
        # Apply force and torque to agent
        angle = self.agent_body.angle
        fx = forward * np.cos(angle)
        fy = forward * np.sin(angle)
        self.agent_body.ApplyForceToCenter((fx, fy), wake=True)
        self.agent_body.ApplyTorque(turn, wake=True)
        # Step physics
        self.world.Step(1.0 / 30.0, 6, 2)
        # Wall constraint: prevent any ant from leaving arena
        self._constrain_to_arena(self.agent_body)
        for body in self.other_ants:
            self._constrain_to_arena(body)
        obs = self._get_obs()
        reward = 0.0  # Placeholder
        terminated = False
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    def render(self):
        if not HAS_MPL:
            print("matplotlib not installed: cannot render")
            return
        # Dynamically scale so the fixed 100mm arena fills the window
        fig_size = 10  # Window size in inches
        pixel_window = 1000  # Window size in pixels
        arena_pixel_radius = int(pixel_window * 0.5)  # 0.98 of window diameter, for minimal margin
        scale = arena_pixel_radius / self.arena_radius  # pixels per mm
        if self.viewer is None:
            self.viewer = plt.figure(figsize=(fig_size, fig_size))
            self.ax = self.viewer.add_subplot(1, 1, 1)
        self.ax.clear()
        # Draw arena
        circle = plt.Circle((0, 0), self.arena_radius * scale, color='lavender', fill=True, zorder=0)
        self.ax.add_patch(circle)
        # Draw agent
        pos = self.agent_body.position
        agent_circle = plt.Circle((pos.x * scale, pos.y * scale), self.ant_radius * scale, color='blue', zorder=2)
        self.ax.add_patch(agent_circle)
        # Draw obstacles
        for body in self.other_ants:
            p = body.position
            c = plt.Circle((p.x * scale, p.y * scale), self.ant_radius * scale, color='gray', zorder=1)
            self.ax.add_patch(c)
        lim = self.arena_radius * scale
        self.ax.set_xlim(-lim, lim)
        self.ax.set_ylim(-lim, lim)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        self.ax.set_position([0, 0, 1, 1])  # Fill the figure area
        self.viewer.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        plt.pause(0.001)
 
    def close(self):
        if self.viewer is not None:
            plt.close(self.viewer)
            self.viewer = None


def main():
    """
    Run a simple test loop for the AntArenaEnv with matplotlib visualization.
    The agent moves forward with a small random turn.
    """
    env = AntArenaEnv(
        render_mode='human',
        data_dir=DATA_DIRECTORY,
        trajectories_file=TRAJECTORIES_FILE
    )
    obs, info = env.reset()
    for step in range(300):
        # Move forward, small random turn
        action = np.array([1.0, np.random.uniform(-0.2, 0.2)], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            print(f"Episode finished after {step+1} steps.")
            break
    env.close()


if __name__ == "__main__":
    main()