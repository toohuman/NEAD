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

FPS = 50
SCALE = 30.0   # affects how fast-paced the game is, forces should be adjusted as well

INITIAL_RANDOM = 5

VIEWPORT_W = 900
VIEWPORT_H = 900
BOUNDARY_SCALE = 0.02

TRACK_TRAIL = 'all' # 'all', 'fade', 'none'
FADE_DURATION = 15 # seconds

vec2d = namedtuple('vec2d', ['x', 'y'])

class AntArena(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self):
        self.seed()
        self.viewer = None
        self.state = None
        self.noise = 0

        pygame.init()

        # circular arena
        self.ant_arena = (
            (VIEWPORT_W/2.0, VIEWPORT_H/2.0),
            min(VIEWPORT_W, VIEWPORT_H)/2.0 - min(VIEWPORT_W, VIEWPORT_H) * BOUNDARY_SCALE
        )
        
        self.t = 0
        self.t_limit = 1000

        # ant_pos = tuple()
        self.ant = None
        self.ant_dim = vec2d(5, 5)
        self.ant_trail = None

        self.target_pos = None
        self.target_trail = None

        # Load the ant trail dataset
        self._get_ant_trails()

        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
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
    
    def _destroy(self):
        self.ant = None
        self.ant_trail = []

        self.target_pos = tuple()
        self.target_trail = []

    def _get_ant_trails(self):
        self.ant_trail_data = load_combined_files("../../../data/2023_2/")
        self.arena_bb = find_bounding_box(self.ant_trail_data)
        origin_arena = calculate_circle(*self.arena_bb)

        translation, scaling = circle_transformation(
            origin_arena, self.ant_arena
        )
        print(f"Translation: {translation}, Scaling: {scaling}")

    def reset(self):
        self._destroy()

        pygame.display.set_caption("WANNts")
        self.display = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
        self.clock = pygame.time.Clock()
        self.running = True
        self.score = 0
        self.t = 0

        self.game_over = False

        if self.ant is None:
            self.ant = pygame.Rect(
                random.uniform(VIEWPORT_W * BOUNDARY_SCALE, VIEWPORT_W - (VIEWPORT_W * BOUNDARY_SCALE)),
                random.uniform(VIEWPORT_H * BOUNDARY_SCALE, VIEWPORT_H - (VIEWPORT_H * BOUNDARY_SCALE)),
                self.ant_dim.x,
                self.ant_dim.y
            )
            while not self._is_rectangle_in_circle(
                self.ant,
                (VIEWPORT_W/2.0, VIEWPORT_H/2.0),
                min(VIEWPORT_W, VIEWPORT_H)/2.0 - min(VIEWPORT_W, VIEWPORT_H) * BOUNDARY_SCALE
            ):
                self.ant = pygame.Rect(
                    random.uniform(VIEWPORT_W * BOUNDARY_SCALE, VIEWPORT_W - (VIEWPORT_W * BOUNDARY_SCALE)),
                    random.uniform(VIEWPORT_H * BOUNDARY_SCALE, VIEWPORT_H - (VIEWPORT_H * BOUNDARY_SCALE)),
                    self.ant_dim.x,
                    self.ant_dim.y
                )

        if len(self.target_trail) == 0:
            # Load target data
            self.target_trail = [
                vec2d(200, 200),
                vec2d(201, 200),
                vec2d(203, 200),
                vec2d(205, 202),
                vec2d(209, 202),
                vec2d(215, 205),
                vec2d(217, 207),
                vec2d(220, 205)
            ]

    def step(self):

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                if event.key == pygame.K_LEFT:
                    pass
                if event.key == pygame.K_RIGHT:
                    pass
                if event.key == pygame.K_UP:
                    pass
                if event.key == pygame.K_DOWN:
                    pass

        self.clock.tick(FPS)

    def render(self):
        # draw
        env.display.fill((150, 150, 170))
        
        pygame.draw.circle(env.display, (200, 200, 200), self.ant_arena[0], self.ant_arena[1])

        # projected trail
        for pos in env.target_trail:
            # pygame.Rect()
            pygame.draw.rect(env.display, (180, 180, 180), (*pos, env.ant_dim.x, env.ant_dim.y))
        pygame.draw.rect(env.display, (0, 0, 0), (*env.target_trail[-1], env.ant_dim.x, env.ant_dim.y))    

        # ant trail
        for pos in env.ant_trail:
            pygame.draw.rect(env.display, (150, 150, 255), pos)

        # ant
        pygame.draw.rect(env.display, (0, 0, 255), env.ant)

        pygame.display.update()


if __name__ == "__main__":

    env = AntArena()
    steps = 0
    total_reward = 0
    a = np.array([0.0, 0.0])

    while env.running:
        env.step()
        env.render()

    pygame.quit()