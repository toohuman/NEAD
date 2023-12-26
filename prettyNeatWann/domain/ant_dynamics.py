import sys, math
from collections import namedtuple
import numpy as np
import pygame
import random

import gymnasium as gym

WIDTH, HEIGHT = 900, 900
BOUNDARY_SCALE = 0.02
FPS = 60

SCALE = 30.0

TRACK_TRAIL = 'all' # 'all', 'fade', 'none'
FADE_DURATION = 15 # seconds

vec2d = namedtuple('vec2d', ['x', 'y'])

# ant_pos = tuple()
ant = None
ant_dim = vec2d(5, 5)
ant_trail = []

target_pos = tuple()
target_trail = []

def track_trail(pos: tuple, prev_pos: list):
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

def is_rectangle_in_circle(rect, circle_center, circle_radius):
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

pygame.init()
pygame.display.set_caption("WANNts")
display = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
running = True
score = 0

while running:
    
    if ant is None:
        ant = pygame.Rect(
            random.uniform(WIDTH * BOUNDARY_SCALE, WIDTH - (WIDTH * BOUNDARY_SCALE)),
            random.uniform(HEIGHT * BOUNDARY_SCALE, HEIGHT - (HEIGHT * BOUNDARY_SCALE)),
            ant_dim.x,
            ant_dim.y
        )
        while not is_rectangle_in_circle(
            ant,
            (WIDTH/2.0, HEIGHT/2.0),
            min(WIDTH, HEIGHT)/2.0 - min(WIDTH, HEIGHT) * BOUNDARY_SCALE
        ):
            ant = pygame.Rect(
                random.uniform(WIDTH * BOUNDARY_SCALE, WIDTH - (WIDTH * BOUNDARY_SCALE)),
                random.uniform(HEIGHT * BOUNDARY_SCALE, HEIGHT - (HEIGHT * BOUNDARY_SCALE)),
                ant_dim.x,
                ant_dim.y
            )

    if len(target_trail) == 0:
        # Load target data
        target_trail = [
            vec2d(200, 200),
            vec2d(201, 200),
            vec2d(202, 200)
        ]

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            break

    if event.type == pygame.KEYDOWN:
        if event.key == pygame.K_ESCAPE:
            running = False
            break
        if event.key == pygame.K_LEFT:
            pass
        if event.key == pygame.K_RIGHT:
            pass
        if event.key == pygame.K_UP:
            pass
        if event.key == pygame.K_DOWN:
            pass

    # draw
    display.fill((150, 150, 170))
    
    # circular border
    ant_arena = (
        (WIDTH/2.0, HEIGHT/2.0),
        min(WIDTH, HEIGHT)/2.0 - min(WIDTH, HEIGHT) * BOUNDARY_SCALE
    )
    print(ant_arena)
    pygame.draw.circle(display, (200, 200, 200), ant_arena[0], ant_arena[1])

    # projected trail
    for pos in target_trail:
        # pygame.Rect()
        pygame.draw.rect(display, (180, 180, 180), (*pos, ant_dim.x, ant_dim.y))
    pygame.draw.rect(display, (0, 0, 0), (*target_trail[-1], ant_dim.x, ant_dim.y))    

    # ant trail
    for pos in ant_trail:
        pygame.draw.rect(display, (150, 150, 255), pos)

    # ant
    pygame.draw.rect(display, (0, 0, 255), ant)

    pygame.display.update()
    clock.tick(FPS)

pygame.quit()