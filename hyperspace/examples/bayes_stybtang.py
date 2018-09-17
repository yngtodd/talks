import os
import argparse
import numpy as np

import skopt
from skopt import gp_minimize
from hyperspace.benchmarks import StyblinskiTang


def main():
    parser = argparse.ArgumentParser(description='Setup experiment.')
    parser.add_argument('--ndims', type=int, help='Number of dimensions for Styblinski-Tang function')
    parser.add_argument('--results_dir', type=str, help='Path to results directory.')
    args = parser.parse_args()

    stybtang = StyblinskiTang(args.ndims)
    bounds = np.tile((-5., 5.), (args.ndims, 1))

    results = gp_minimize(stybtang,
                          bounds,
                          n_calls=20,
                          verbose=True,
                          random_state=0)

    savefile = os.path.join(args.results_dir, 'smbo_results')
    skopt.dump(results, savefile)


if __name__ == '__main__':
    main()
