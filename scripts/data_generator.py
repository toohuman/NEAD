import argparse
import lzma, pickle
import numpy as np
import pandas as pd
import os


def load_input_files(source_dir):
    input_files = []
    data = []

    for file in os.listdir(source_dir):
        if file.endswith('.pkl.xz'):
            input_files.append(file)

    for input_file in input_files:
        with lzma.open(os.path.join(source_dir, input_file)) as file:
            data.append(pd.read_pickle(file))

    return pd.concat(data, ignore_index=True)

def find_valid_trail(data, num_trails, trail_length):
    num_trails = 10_000
    trail_length = 60 * 60
    trails = np.zeros((num_trails, trail_length, 2), dtype=float)

    for i in range(num_trails):
        start = np.random.randint(0, len(data[0]) - trail_length)
        ant_index = np.random.randint(0, len(data.T)/2)
        not_null = False
        while not not_null:
            ant_index = np.random.randint(0, len(data.T)/2)
            if np.isnan(np.array(data[ant_index][start:start + trail_length])).any():
                continue
            else:
                trails[i][0:trail_length] = data[ant_index][start:start + trail_length]
                not_null = True
    
    return trails

def find_bounding_box(data):
    # Concatenating all x and y values into separate Series
    all_x_values = pd.concat([data[col] for col in data.columns if col[1] == 'x'], ignore_index=True)
    all_y_values = pd.concat([data[col] for col in data.columns if col[1] == 'y'], ignore_index=True)

    # Calculating the minimum and maximum for x and y values efficiently
    min_x = all_x_values.min()
    max_x = all_x_values.max()
    min_y = all_y_values.min()
    max_y = all_y_values.max()

    return min_x, min_y, max_x, max_y

def calculate_circle(min_x, max_x, min_y, max_y):
    """
    Calculate the circle that fits perfectly within a bounding box.

    Parameters:
    min_x (float): The minimum x value of the bounding box.
    max_x (float): The maximum x value of the bounding box.
    min_y (float): The minimum y value of the bounding box.
    max_y (float): The maximum y value of the bounding box.

    Returns:
    tuple: A tuple containing the center coordinates (x, y) and the radius of the circle.
    """
    # Calculate the center of the bounding box
    x_centre = (min_x + max_x) / 2
    y_centre = (min_y + max_y) / 2

    # Calculate the radius of the circle
    radius = min(max_x - min_x, max_y - min_y) / 2

    return ((x_centre, y_centre), radius)

def circle_transformation(circle_a, circle_b):
    """
    Calculate the transformation from one circle to another.

    Parameters:
    circle_a (tuple): A tuple (x_a, y_a, r_a) representing Circle A's center and radius.
    circle_b (tuple): A tuple (x_b, y_b, r_b) representing Circle B's center and radius.

    Returns:
    tuple: A tuple containing the translation vector (dx, dy) and the scaling factor.
    """
    (x_a, y_a), r_a = circle_a
    (x_b, y_b), r_b = circle_b

    # Translation vector
    dx = x_b - x_a
    dy = y_b - y_a

    # Scaling factor
    scale = r_b / r_a

    return (dx, dy), scale

def main(args):
    with lzma.open(args.input_file, 'r') as file:
        input_data = pickle.load(file)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=('Train policy on OpenAI Gym environment '
                                                'using pepg, ses, openes, ga, cma'))
    parser.add_argument('-i', '--input_file', type=str, help='Input data file of ant dynamics [numpy array].', default='../data/2023_2/KA050_10cm_5h_20230614_1h-2h.pkl.xz')
    # parser.add_argument('-o', '--output_dir', type=str, default="../data", help='num episodes per trial')
    # parser.add_argument('-s', '--seed_start', type=int, default=111, help='initial seed')
    # parser.add_argument('--sigma_init', type=float, default=0.10, help='sigma_init')
    # parser.add_argument('--sigma_decay', type=float, default=0.999, help='sigma_decay')

    args = parser.parse_args()

    main(args)