import os
import argparse
import numpy as np

import skopt
from skopt import dummy_minimize
from hyperspace.benchmarks import StyblinskiTang

from mpi4py import MPI


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def main():
    parser = argparse.ArgumentParser(description='Setup experiment.')
    parser.add_argument('--ndims', type=int, help='Number of dimensions for Styblinski-Tang function')
    parser.add_argument('--results_dir', type=str, help='Path to results directory.')
    args = parser.parse_args()

    stybtang = StyblinskiTang(args.ndims)
    bounds = np.tile((-5., 5.), (args.ndims, 1))

    rand_state = 0 + rank
    results = dummy_minimize(stybtang,
                             bounds,
                             n_calls=20,
                             verbose=True,
                             random_state=rand_state)

    savefile = os.path.join(args.results_dir, 'parallel_rand' + str(rank))
    skopt.dump(results, savefile)


if __name__ == '__main__':
    main()
