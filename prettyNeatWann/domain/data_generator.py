import argparse
import logging
import lzma, pickle
import numpy as np
import pandas as pd
import os, sys

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] In %(pathname)s:%(lineno)d:\n%(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

FILE_PREFIX = "KA050_10cm_5h_20230614"
PP_FILE_PREFIX = "KA050_processed"
OUTPUT_FILE = '_'.join([PP_FILE_PREFIX, *FILE_PREFIX.split('_')[1:]]) + '.pkl.xz'

def load_data(source_dir, scale = None, arena_dim = None):
    data = None
    if os.path.exists(os.path.join(source_dir, OUTPUT_FILE)):
        with lzma.open(os.path.join(source_dir, OUTPUT_FILE)) as file:
            data = pd.read_pickle(file)
            logger.info(msg=f"Processed data file found: {OUTPUT_FILE}")
        return data.iloc[::int(scale)] if scale else data
    else:
        logger.info(msg=f"No processed file found. Looking for ")
        return load_combined_files(source_dir, arena_dim, scale)


def process_data(data, arena_dim):
    data_len = len(data)
    logger.info(msg=f"Ant trail data loaded. Total records: {data_len}")
    arena_bb = find_bounding_box(data)
    origin_arena = calculate_circle(*arena_bb)

    translation, scale = circle_transformation(
        origin_arena, arena_dim
    )

    logger.info(msg=f"Processing data now. This will take a while...")
    apply_transform_scale(data, translation, scale)
    logger.info(msg=f"Finished processing.")

    logger.info(msg=f"Translation: {translation}, scale: {scale}")
    logger.info(msg=f"Original: ({origin_arena[0][0] + translation[0]}, {origin_arena[0][1] + translation[1]}), scale: {origin_arena[1]*scale}")
    logger.info(msg=f"Simulated: {arena_dim[0]}, scale: {arena_dim[1]}")

    return data


def load_combined_files(source_dir, arena_dim, scale = None):
    input_files = []
    data = []

    for file in os.listdir(source_dir):
        if FILE_PREFIX in file and file.endswith('.pkl.xz'):
            input_files.append(file)

    for input_file in input_files:
        with lzma.open(os.path.join(source_dir, input_file)) as file:
            data.append(pd.read_pickle(file))

    data = process_data(pd.concat(data, ignore_index=True), arena_dim)
    data.to_pickle(os.path.join(source_dir, OUTPUT_FILE), compression='xz')

    return data.iloc[::int(scale)] if scale else data


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