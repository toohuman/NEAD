import argparse
import lzma, pickle
import numpy as np
import os

def main():
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=('Train policy on OpenAI Gym environment '
                                                'using pepg, ses, openes, ga, cma'))
    parser.add_argument('input_file', type=str, help='cartpole_swingup, biped, etc.')
    # parser.add_argument('-i', '--input_file', type=str, help='Input data file of ant dynamics [numpy array].', default='pepg')
    parser.add_argument('-o', '--output_dir', type=str, default="../data", help='num episodes per trial')
    parser.add_argument('-s', '--seed_start', type=int, default=111, help='initial seed')
    parser.add_argument('--sigma_init', type=float, default=0.10, help='sigma_init')
    parser.add_argument('--sigma_decay', type=float, default=0.999, help='sigma_decay')

    args = parser.parse_args()

    main(args)