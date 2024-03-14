import logging
import sys, math
from collections import namedtuple
import numpy as np
import pygame
import random

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import colorize, seeding

from data_generator import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] In %(pathname)s:%(lineno)d:\n%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

VIDEO_FPS = 60     # Source data FPS (60Hz)
SIM_FPS = 30.0    # Simulation environment FPS

INITIAL_RANDOM = 5

SCREEN_W = 900
SCREEN_H = 900
BOUNDARY_SCALE = 0.02


vec2d = namedtuple('vec2d', ['x', 'y'])


# Global parameters for agent control
TIMESTEP = 2./SIM_FPS       # Not sure if this will be necessary, given the fixed FPS?
ANT_DIM = vec2d(5, 5)
AGENT_SPEED = 10*1.75       # Taken from slimevolley, will need to adjust based on feeling
TURN_RATE = 65 * 2 * math.pi / 360 
VISION_RANGE = 100  # No idea what is a reasonable value for this.

TRACK_TRAIL = 'all' # 'all', 'fade', 'none'
FADE_DURATION = 15 # seconds

# Helper functions
def is_rectangle_in_circle(x, y, circle_center, circle_radius):
    """
    Check if a pygame.Rect is completely contained within a circle.

    Parameters:
    rect (pygame.Rect): The rectangle to check.
    circle_center (tuple): The (x, y) coordinates of the center of the circle.
    circle_radius (float): The radius of the circle.

    Returns:
    bool: True if the rectangle is completely contained within the circle, False otherwise.
    """
    rect = pygame.Rect(x - ANT_DIM.x / 2., y - ANT_DIM.y / 2.,
                       ANT_DIM.x, ANT_DIM.y)
    rect_corners = [
        (rect.left, rect.top),    # Top-left
        (rect.left, rect.bottom), # Bottom-left
        (rect.right, rect.top),   # Top-right
        (rect.right, rect.bottom) # Bottom-right
    ]

    for x, y in rect_corners:
        dx = x - circle_center[0]
        dy = y - circle_center[1]
        distance = math.sqrt(dx ** 2 + dy ** 2)

        if distance > circle_radius:
            return False

    return True


class Ant():
    """Agent class for the ant"""

    def __init__(self, pos):
        self.pos = vec2d(*pos)
        self.speed = 0.0
        self.theta = 0.0
        self.theta_dot = 0.0
        self.trail = []

        # Detection scalar:
        # num of ants in cone, or distance to closes ant
        self.V_f = None
        self.V_r = None
        self.V_b = None
        self.V_l = None
        self.vision_range = VISION_RANGE

    def _detect_vision(self, detected_ants: dict):
        v_f = len(detected_ants['forward'])
        v_r = len(detected_ants['right'])
        v_b = len(detected_ants['back'])
        v_l = len(detected_ants['left'])

        vision = [v_f, v_r, v_b, v_l]
        denominator = np.sum([v_f, v_r, v_b, v_l])
        if denominator != 0: vision /= denominator

        self.V_f, self.V_r, self.V_b, self.V_l = vision


    def _detect_nearby_ants(self, other_ants):
        """
        Detects other ants withsin a specified radius and identifies their relative
        position quadrant based on this ant's orientation.

        Parameters:
        - other_ants (list of tuples): The (x, y) positions of other ants.
        - radius (float): The radius within which to detect other ants.

        Returns:
        - dict: A dictionary mapping 'forward', 'right', 'back', 'left' to a list
                of ants (represented by their positions) that are within the specified
                radius and fall into that relative quadrant.
        """
        detected_ants = {'forward': [],  'left': [], 'back': [], 'right': []}

        for other_ant in other_ants:
            dx = other_ant[0] - self.pos.x
            dy = other_ant[1] - self.pos.y
            distance = math.sqrt(dx**2 + dy**2)

            if distance <= self.vision_range:
                # Calculate angle from self.pos to other_ant.pos, adjusting with self.theta
                angle_to_ant = math.atan2(dy, dx)
                # Adjusting by self.theta to align with the direction the agent is facing
                relative_angle = angle_to_ant - self.theta

                # Normalise the relative angle to be between -pi and pi
                relative_angle = (relative_angle + math.pi) % (2 * math.pi) - math.pi

                # Determine quadrant based on relative_angle
                if -math.pi/2 < relative_angle <= math.pi/2:
                    if 0 <= relative_angle <= math.pi/2:
                        direction = 'forward'
                    else:
                        direction = 'left'
                else:
                    if -math.pi/2 >= relative_angle:
                        direction = 'back'
                    else:
                        direction = 'right'

                detected_ants[direction].append(other_ant)

        return detected_ants


    def _turn(self):
        self.theta += (self.theta_dot * TIMESTEP)
        self.theta = self.theta % (2 * np.pi)


    def _move(self, arena):
        """
        Move an agent from its current position (x, y) according to desired_speed
        and angle theta using matrix multiplication.
        """
        # Define the unit vector with angle theta.
        # unit_direction = [np.cos(self.theta), np.sin(self.theta)]
        # unit_direction /= np.linalg.norm(unit_direction)
        # desired_pos = np.add(
        #     np.array(self.pos),
        #     np.array(unit_direction) * self.desired_speed * TIMESTEP
        # )
        # Calculate the desired direction of travel (rotate to angle theta)
        direction = np.array([
            np.cos(self.theta) - np.sin(self.theta),
            np.sin(self.theta) + np.cos(self.theta),
        ]) * self.desired_speed * TIMESTEP
        # Set the desired position based on direction and speed relative to timestep
        desired_pos = np.add(np.array(self.pos), direction)

        # If leaving the cirle, push agent back into circle.
        if is_rectangle_in_circle(desired_pos[0], desired_pos[1], arena[0], arena[1]):
            self.pos = vec2d(desired_pos[0], desired_pos[1])
        # Otherwise, slightly adjust the agent's angle theta towards tangent at the
        # circle's circumference.
        # else:
        #     # Calculate the angle from the center of the circle to the agent
        #     angle_to_center = math.atan2(self.pos.y - arena[0][1], self.pos.x - arena[0][0])
        #     if angle_to_center < 0: angle_to_center += np.pi
        #     if (self.theta >= angle_to_center) and (self.theta < angle_to_center + np.pi):
        #         d_theta = np.pi / 2 - self.desired_turn_speed
        #     elif (self.theta < angle_to_center) or (self.theta >= angle_to_center - np.pi):
        #         d_theta = -np.pi / 2 + self.desired_turn_speed
                
        #     theta = self.theta + d_theta * TIMESTEP
        #     theta = theta % (2 * np.pi)
        #     self.theta = theta


    def set_action(self, action):
        forward    = False
        backward   = False
        turn_left  = False
        turn_right = False

        if action[0] > 0.75: forward    = True
        if action[1] > 0.75: backward   = True
        if action[2] > 0.75: turn_left  = True
        if action[3] > 0.75: turn_right = True

        self.desired_speed = 0
        self.desired_turn_speed = 0

        if (forward and (not backward)):
            self.desired_speed = AGENT_SPEED
        if (backward and (not forward)):
            self.desired_speed = -AGENT_SPEED
        if (turn_left and (not turn_right)):
            self.desired_turn_speed = -TURN_RATE
        if (turn_right and (not turn_left)):
            self.desired_turn_speed = TURN_RATE


    def get_obs(self, others=None):
        if others is not None:
            self._detect_vision(self._detect_nearby_ants(others))
        result = {
            'x': self.pos.x, 'y': self.pos.y,
            'speed': self.speed,
            'theta': self.theta, 'theta_dot': self.theta_dot,
            'V_f': self.V_f,
            'V_r': self.V_r,
            'V_b': self.V_b,
            'V_l': self.V_l
        }
        return result


    def update(self, arena, noise=0.0):
        self.pos   = vec2d(
            self.pos.x + np.random.randn() * noise,
            self.pos.y - np.random.randn() * noise
        )
        self.theta += (np.random.randn() * noise)
        self.theta = self.theta % (2 * np.pi)

        self.speed = self.desired_speed
        self.theta_dot = self.desired_turn_speed

        self._turn()
        self._move(arena)


class AntDynamicsEnv(gym.Env):
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps' : SIM_FPS
    }

    ant_trail_data = None

    def __init__(self, render_mode=None):
        self.force_mag = 10.0
        self.ant = None
        self.ant_trail = None

        self.target_trail = None

        self.other_ants = None

        self.seed()
        self.viewer = None
        self.state = None
        self.noise = 0

        self.t = 0
        self.t_limit = SIM_FPS * 60 # 60 seconds

        assert render_mode is None or render_mode in type(self).metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

        # circular arena
        self.ant_arena = (
            (SCREEN_W/2.0, SCREEN_H/2.0),
            min(SCREEN_W, SCREEN_H)/2.0 - min(SCREEN_W, SCREEN_H) * BOUNDARY_SCALE
        )

        high = np.array([
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max,
            np.finfo(np.float32).max
        ])

        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=float)
        self.observation_space = spaces.Box(-high, high, dtype=float)

        # Load the ant trail dataset
        if not type(self).ant_trail_data:
            self._get_ant_trails()


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


    def _get_ant_trails(self):
        type(self).ant_trail_data = load_data(
            "../../../data/2023_2/",     
            VIDEO_FPS / SIM_FPS,
            self.ant_arena
        )


    def _select_target(self, others=False):
        """
        Select an ant trail as the target trail for the current trial.
        At the moment, we will just select a single target trail, but we should
        also provide positions of other ants within a given radius for feeding
        into the ant's internal state.
        """
        trail_length = int(SIM_FPS * 60)
        s = np.zeros((trail_length, 2), dtype=float)
        num_ants = len(type(self).ant_trail_data.columns.levels[0])
        # If showing positions of other ants during the trail
        other_ants = None
        if others:
            other_ants = np.zeros(
                (num_ants - 1, trail_length, 2),
                dtype=float
            )

        start = np.random.randint(len(type(self).ant_trail_data) - trail_length)
        indices = list(np.random.permutation(num_ants))
        indices_set = set(indices)
        contains_null = True
        while contains_null and len(indices) > 0:
            ant_index = indices.pop()
            if np.isnan(np.array(type(self).ant_trail_data[ant_index][start:start + trail_length])).any():
                continue
            else:
                s[0:trail_length] = type(self).ant_trail_data[ant_index][start:start + trail_length]
                contains_null = False
                indices_set.discard(ant_index)
        if others and not contains_null:
            trail_index = 0
            for other_ant_index in indices_set:
                if np.isnan(np.array(type(self).ant_trail_data[other_ant_index][start:start + trail_length])).any():
                    np.resize(other_ants, (np.shape(other_ants)[1]-1, trail_length, 2))
                    continue
                other_ants[trail_index][0:trail_length] = type(self).ant_trail_data[other_ant_index][start:start + trail_length]
                trail_index += 1

        return Ant(s[0]), s, other_ants


    def _track_trail(self, pos: tuple, prev_pos: list):
        trail = []
        if TRACK_TRAIL == 'all':
            trail.append(pos)
            trail.append(prev_pos)
        if TRACK_TRAIL == 'fade':
            trail.append(pos)
            trail.append(prev_pos)
            trail = trail[:FADE_DURATION * VIDEO_FPS]
        if TRACK_TRAIL == 'none':
            pass
        return trail


    def _reward_function(self):
        """
        Calculate the reward given the focal ant and the accuracy of its behaviour
        over the trial, given the source data as the ground truth.
        """


    def get_observations(self, others=None):
        return self.ant.get_obs(others)


    def _destroy(self):
        self.ant = None
        self.ant_trail = []
        self.target_trail = []
        self.other_ants = None

        self.viewer = None
        self.state = None

        self.window = None
        self.clock = None


    def reset(self):
        self._destroy()

        self.t = 0      # timestep reset
        self.steps_beyond_done = None

        self.ant, self.target_trail, self.other_ants = self._select_target(others=True)
        self.state = self.get_observations(self.other_ants[:,self.t])
        obs = [
            # x, y position and speed
            self.state['x'], self.state['y'], self.state['speed'],
            # cos and sine of agent's angle theta
            np.cos(self.state['theta']), np.sin(self.state['theta']),
            self.state['theta_dot'],
            # Vision (directional detection of other ants)
            self.state['V_f'],
            self.state['V_r'],
            self.state['V_b'],
            self.state['V_l']
        ]

        if self.render_mode == 'human':
            self._render_frame()

        return obs


    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


    def step(self, action):
        """
        Each step, take the given action and return observations, reward, done (bool)
        and any other additional information if necessary.
        """
        done = False
        self.t += 1

        # Pygame controls and resources if render_mode == 'human'
        if self.render_mode == "human": self._render_frame()

        self.ant.set_action(action)
        self.ant.update(self.ant_arena)
        obs = self.get_observations(self.other_ants[:,self.t])
        
        if self.t >= self.t_limit:
            done = True

        info = {}

        reward = self._reward_function()

        return obs, reward, done, info


    def render(self):
        if self.render_mode == 'rgb_array':
            return self._render_frame()


    def _render_frame(self):
        if self.window is None and self.render_mode == 'human':
            pygame.init()
            pygame.display.init()
            pygame.display.set_caption('WANNts')
            self.window = pygame.display.set_mode(
                (SCREEN_W, SCREEN_H)
            )

        if self.clock is None and self.render_mode == 'human':
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((SCREEN_W, SCREEN_H))
        canvas.fill((150, 150, 170))

        # Project the circular arena
        pygame.draw.circle(
            canvas,
            (200, 200, 200),
            self.ant_arena[0],
            self.ant_arena[1]
        )

        if self.other_ants is not None:
            try:
                for other_ant in self.other_ants[:,self.t]:
                    pygame.draw.rect(canvas, (180, 180, 180),
                                    (other_ant[0] - ANT_DIM.x/2.,
                                    other_ant[1] - ANT_DIM.y/2.,
                                    ANT_DIM.x, ANT_DIM.y))

            except IndexError:
                print(other_ant)
                logger.error("End of time series reached for other ants.")
            except TypeError:
                print(other_ant)
                logger.error("Cannot draw ant with provided coordinates.")

        # Draw projected target trail
        for pos in self.target_trail:
            pygame.draw.rect(canvas, (220, 180, 180),
                            (pos[0] - ANT_DIM.x/2.,
                             pos[1] - ANT_DIM.y/2.,
                             ANT_DIM.x, ANT_DIM.y))
        
        # Draw target ant
        pygame.draw.rect(canvas, (180, 0, 0),
                        (self.target_trail[-1][0] - ANT_DIM.x/2.,
                         self.target_trail[-1][1] - ANT_DIM.y/2.,
                         ANT_DIM.x, ANT_DIM.y))

        # Draw ant trail
        for pos in self.ant_trail:
            pygame.draw.rect(canvas, (150, 150, 255), (*pos, ANT_DIM.x, ANT_DIM.y))

        # Draw ant last; to ensure visibility.
        pygame.draw.rect(canvas, (0, 0, 180),
                        (self.ant.pos.x - ANT_DIM.x/2.,
                         self.ant.pos.y - ANT_DIM.y/2.,
                         ANT_DIM.x, ANT_DIM.y))


        if self.render_mode == 'human':
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:   # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1,0,2)
            )



if __name__ == "__main__":
    env = AntDynamicsEnv(render_mode='human')
    total_reward = 0
    obs = env.reset()

    manual_mode = True
    manual_action = [0, 0, 0, 0]

    done = False
    while not done:
        if manual_mode:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    done = True
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        done = True
                    if event.key == pygame.K_r:
                        env.reset()
                    if event.key == pygame.K_UP:    manual_action[0] = 1
                    if event.key == pygame.K_DOWN:  manual_action[1] = 1
                    if event.key == pygame.K_LEFT:  manual_action[2] = 1
                    if event.key == pygame.K_RIGHT: manual_action[3] = 1
                elif event.type == pygame.KEYUP:
                    if event.key == pygame.K_UP:    manual_action[0] = 0
                    if event.key == pygame.K_DOWN:  manual_action[1] = 0
                    if event.key == pygame.K_LEFT:  manual_action[2] = 0
                    if event.key == pygame.K_RIGHT: manual_action[3] = 0
            action = manual_action

        if done: break

        obs, reward, done, _ = env.step(action)


    env.close()
    print('Cumulative score:', total_reward)