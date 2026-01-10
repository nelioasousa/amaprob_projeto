import random
from time import monotonic
import numpy as np
from exploration import exploration


class Naive:

    def __init__(self, projection_dim: int):
        pass

    def fit(self, X: np.ndarray):
        pass

    def evaluate(self, X: np.ndarray):
        return np.random.rand(X.shape[0])

    def project(self, X: np.ndarray):
        return np.empty((X.shape[0], 0))


def _run():
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description='Perform a naive/random data exploration.'
    )

    parser.add_argument(
        '-z', '--latent-dim', type=int, default=0,
        help='Does nothing. Naive do not use any projection.'
    )

    parser.add_argument(
        '-i', '--iterations', type=int, default=20,
        help='Number of exploration iterations to perform.'
    )

    parser.add_argument(
        '-s', '--selection-size', type=int, default=50,
        help='Number of samples to select in each iteration.'
    )

    parser.add_argument(
        '-rs', '--random_seed', type=int, default=42,
        help='Random seed for reproducibility.'
    )

    args = parser.parse_args()

    try:
        start = monotonic()
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        exploration(
            model_class=Naive,
            latent_dim=args.latent_dim,
            selection_size=args.selection_size,
            num_iterations=args.iterations,
        )
        print(f'[3] Naive exploration completed successfully ({monotonic() - start:.2f} secs)')
        sys.exit(0)
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(2)


if __name__ == '__main__':
    _run()
