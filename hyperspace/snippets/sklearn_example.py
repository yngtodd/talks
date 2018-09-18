from sklearn.datasets import load_boston
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
import numpy as np
import argparse

from hyperspace import hyperdrive


def objective(params):
    """
    Objective function to be minimized.
    """
    param0, param1, param2 = params
    model = Model(param0, param1, param3)

    train_loss = train(model)
    val_loss = validate(model)

    return val_loss


def main():
    hparams = [(2, 10),             # param0
               (10.0**-2, 10.0**0), # param1
               (1, 10)]             # param2

    hyperdrive(objective=objective,
               hyperparameters=hparams,
               results_path='/path/to/save/results',
               model="GP",
               n_iterations=15,
               verbose=True,
               random_state=0,
               sampler="lhs",
               n_samples=5,
               checkpoints=True)


if __name__ == '__main__':
    main()
