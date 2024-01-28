import argparse
import lzma, pickle
import numpy as np
import pandas as pd
import os


def load_combined_files(source_dir, scale = None):
    input_files = []
    data = []

    for file in os.listdir(source_dir):
        if file.endswith('.pkl.xz'):
            input_files.append(file)

    for input_file in input_files:
        with lzma.open(os.path.join(source_dir, input_file)) as file:
            data.append(pd.read_pickle(file))

    if scale:
        return pd.concat(data, ignore_index=True).iloc[::int(scale)]
    else:
        return pd.concat(data, ignore_index=True)

def find_valid_trail(data, num_trails, trail_length):
    num_trails = 10_000
    trail_length = 60 * 60
    trails = np.zeros((num_trails, trail_length, 2), dtype=float)

    for i in range(num_trails):
        start = np.random.randint(0, len(data[0]) - trail_length)
        ant_index = np.random.randint(0, len(data.columns.levels[0]))
        not_null = False
        while not not_null:
            ant_index = np.random.randint(0, len(data.columns.levels[0]))
            if np.isnan(np.array(data[ant_index][start:start + trail_length])).any():
                continue
            else:
                trails[i][0:trail_length] = data[ant_index][start:start + trail_length]
                not_null = True
    
    return trails

def find_bounding_box(data):
    # Separate all x and y values into slices
    all_x_values = data[[col for col in data.columns if 'x' in col]]
    all_y_values = data[[col for col in data.columns if 'y' in col]]
    # Calculating the minimum and maximum for x and y values efficiently
    min_x = all_x_values.min(axis=None)
    max_x = all_x_values.max(axis=None)
    min_y = all_y_values.min(axis=None)
    max_y = all_y_values.max(axis=None)

    return min_x, min_y, max_x, max_y

def calculate_circle(min_x, min_y, max_x, max_y):
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
    SF = 0.99
    (x_a, y_a), r_a = circle_a
    (x_b, y_b), r_b = circle_b

    # Scaling factor
    scale = r_b / r_a

    x_a *= scale*SF
    y_a *= scale*SF

    # Translation vector
    dx = x_b - x_a
    dy = y_b - y_a

    return (dx, dy), scale*SF

def apply_transform_scale(data, trans, scale):
    data[[col for col in data.columns if 'x' in col]] = data[[col for col in data.columns if 'x' in col]].transform(
        np.vectorize(lambda x : np.round((x * scale) + trans[0]))
    )
    data[[col for col in data.columns if 'y' in col]] = data[[col for col in data.columns if 'y' in col]].transform(
        np.vectorize(lambda x : np.round((x * scale) + trans[1]))
    )

def euclidean_distances(data):
    a = np.array(data)
    b = a.reshape(a.shape[0], 1, a.shape[1])
    distances = np.sqrt(np.einsum('ijk, ijk->ij', a-b, a-b))
    np.fill_diagonal(distances, np.NaN)

    return distances


# def main(args):
#     with lzma.open(args.input_file, 'r') as file:
#         input_data = pickle.load(file)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description=('Train policy on OpenAI Gym environment '
#                                                 'using pepg, ses, openes, ga, cma'))
#     parser.add_argument('-i', '--input_file', type=str, help='Input data file of ant dynamics [numpy array].', default='../data/2023_2/KA050_10cm_5h_20230614_1h-2h.pkl.xz')
#     # parser.add_argument('-o', '--output_dir', type=str, default="../data", help='num episodes per trial')
#     # parser.add_argument('-s', '--seed_start', type=int, default=111, help='initial seed')
#     # parser.add_argument('--sigma_init', type=float, default=0.10, help='sigma_init')
#     # parser.add_argument('--sigma_decay', type=float, default=0.999, help='sigma_decay')

#     args = parser.parse_args()

#     main(args)