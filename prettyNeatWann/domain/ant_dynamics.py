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

FPS = 60
SCALE = 30.0   # affects how fast-paced the game is, forces should be adjusted as well
DT = 1.0/SCALE

INITIAL_RANDOM = 5

SCREEN_W = 900
SCREEN_H = 900
BOUNDARY_SCALE = 0.02

TRACK_TRAIL = 'all' # 'all', 'fade', 'none'
FADE_DURATION = 15 # seconds

vec2d = namedtuple('vec2d', ['x', 'y'])

class Ant():
    """Agent class for the ant"""
    def __init__(self, pos):
        self.x, self.y = pos
        self.v = 0.0
        self.trail = []
    
    def _move(self):
        self.x += self.vx * DT
        self.y += self.vy * DT

        s = math.sin(self.theta)
        c = math.cos(self.theta)


    def update(self, action, state, noise):
        self.x     += np.random.randn() * noise
        self.theta += np.random.randn() * noise

        self.x += self.x_dot * DT
        self.theta += self.theta_dot * DT

        self.x_dot += action.x_dot * DT
        self.theta_dot += action.theta_dot * DT  

        self._move()

class AntDynamicsEnv(gym.Env):
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps' : FPS
    }

    ant_trail_data = None

    def __init__(self, render_mode=None):
        self.force_mag = 10.0
        self.ant = None
        self.ant_dim = vec2d(5, 5)
        self.ant_trail = None

        self.target_pos = None
        self.target_trail = None

        self.seed()
        self.viewer = None
        self.state = None
        self.noise = 0

        self.t = 0
        self.t_limit = 3600

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

        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _get_ant_trails(self):
        type(self).ant_trail_data = load_data(
            "../../../data/2023_2/",
            FPS / SCALE,
            self.ant_arena
        )

    def _select_target(self):
        """
        Select an ant trail as the target trail for the current trial.
        At the moment, we will just select a single target trail, but we should
        also provide positions of other ants within a given radius for feeding
        into the ant's internal state.
        """
        num_trails = 1
        trail_length = int(SCALE * 60)
        s = np.zeros((num_trails, trail_length, 2), dtype=float)

        for i in range(num_trails):
            start = np.random.randint(len(type(self).ant_trail_data) - trail_length)
            pos_indices = list(np.random.permutation(len(type(self).ant_trail_data.columns.levels[0])))
            contains_null = True
            while contains_null and len(pos_indices) > 0:
                ant_index = pos_indices.pop()
                if np.isnan(np.array(type(self).ant_trail_data[ant_index][start:start + trail_length])).any():
                    continue
                else:
                    s[i][0:trail_length] = type(self).ant_trail_data[ant_index][start:start + trail_length]
                    contains_null = False

        return Ant(s[0][0]), s[0], s[0][-1]
    
    def _track_trail(self, pos: tuple, prev_pos: list):
        trail = []
        if TRACK_TRAIL == 'all':
            trail.append(pos)
            trail.append(prev_pos)
        if TRACK_TRAIL == 'fade':
            trail.append(pos)
            trail.append(prev_pos)
            trail = trail[:FADE_DURATION * FPS]
        if TRACK_TRAIL == 'none':
            pass
        return trail

    def _is_rectangle_in_circle(self, rect, circle_center, circle_radius):
        """
        Check if a pygame.Rect is completely contained within a circle.
        
        Parameters:
        rect (pygame.Rect): The rectangle to check.
        circle_center (tuple): The (x, y) coordinates of the center of the circle.
        circle_radius (float): The radius of the circle.

        Returns:
        bool: True if the rectangle is completely contained within the circle, False otherwise.
        """
        
        rect_corners = [
            (rect.left, rect.top),
            (rect.left, rect.bottom),
            (rect.right, rect.top),
            (rect.right, rect.bottom)
        ]
        
        for x, y in rect_corners:
            dx = x - circle_center[0]
            dy = y - circle_center[1]
            distance = math.sqrt(dx ** 2 + dy ** 2)
            
            if distance > circle_radius:
                return False
                
        return True
    
    def _update_state(self, action, state, noise=0):
        x, x_dot, theta, theta_dot = self.ant.update(action, state, noise)

        return (x, x_dot, theta, theta_dot)


    def _destroy(self):
        self.ant = None
        self.ant_trail = []

        self.target_pos = None
        self.target_trail = []

        self.viewer = None
        self.state = None

        self.window = None
        self.clock = None

    def reset(self):
        self._destroy()

        self.t = 0      # timestep reset
        self.running = True
        self.steps_beyond_done = None

        self.ant, self.target_trail, self.target_pos = self._select_target()
        self.state = np.random.normal(loc=np.array([0.0, 0.0, np.pi, 0.0]), scale=np.array([0.2, 0.2, 0.2, 0.2]))
        x, x_dot, theta, theta_dot = self.state
        obs = np.array([x, x_dot, np.cos(theta), np.sin(theta), theta_dot])

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
        # Pygame controls and resources if render_mode == 'human'
        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        self.running = False
                    # if event.key == pygame.K_LEFT:
                    #     pass
                    # if event.key == pygame.K_RIGHT:
                    #     pass
                    # if event.key == pygame.K_UP:
                    #     pass
                    # if event.key == pygame.K_DOWN:
                    #     pass
                    if event.key == pygame.K_r:
                        self.reset()

                self._render_frame()




        if self.t >= self.t_limit:
            self.running = False
        
        if not self.running:
            self.close()

        
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

        pygame.draw.circle(
            canvas,
            (200, 200, 200),
            self.ant_arena[0],
            self.ant_arena[1]
        )

        # Draw projected target trail
        for pos in self.target_trail:
            pygame.draw.rect(canvas, (180, 180, 180), (*pos, self.ant_dim.x, self.ant_dim.y))
        # Draw target ant
        pygame.draw.rect(canvas, (0, 0, 0), (*self.target_trail[-1], self.ant_dim.x, self.ant_dim.y))    

        # Draw ant trail
        for pos in self.ant_trail:
            pygame.draw.rect(canvas, (150, 150, 255), (*pos, self.ant_dim.x, self.ant_dim.y))

        # Draw ant
        pygame.draw.rect(canvas, (0, 0, 255), (self.ant.x, self.ant.y, self.ant_dim.x, self.ant_dim.y))

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
    a = np.array([0.0, 0.0])

    while env.running:
        env.step()

