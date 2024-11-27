import sys, math
from collections import namedtuple
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
import pygame
import random
import lzma
import os
from tqdm import tqdm
import warnings

DATA_DIRECTORY = "data/2023_2/"
INPUT_FILE = 'KA050_processed_10cm_5h_20230614.pkl.xz'

def load_data(source_dir, input_file, scale = None, arena_dim = None):
    data = None
    with lzma.open(os.path.join(source_dir, input_file)) as file:
        data = pd.read_pickle(file)
    return data.iloc[::int(scale)] if scale else data


def process_data(data, arena_dim):
    data_len = len(data)
    arena_bb = find_bounding_box(data)
    origin_arena = calculate_circle(*arena_bb)

    translation, scale = circle_transformation(origin_arena, arena_dim)

    apply_transform_scale(data, translation, scale)

    return data

data = load_data(DATA_DIRECTORY, INPUT_FILE)

print(data.head())
print(data[0].head())
