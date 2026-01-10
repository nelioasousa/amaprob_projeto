import random
from time import monotonic
import numpy as np
from ppca import PPCA
from exploration import exploration


def _run():
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description='Perform data exploration using PPCA.'
    )

    parser.add_argument(
        '-z', '--latent-dim', type=int, default=16,
        help='Dimensionality of the latent space.'
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
            model_class=PPCA,
            latent_dim=args.latent_dim,
            selection_size=args.selection_size,
            num_iterations=args.iterations,
        )
        print(f'[2] Exploration using PPCA completed successfully ({monotonic() - start:.2f} secs)')
        sys.exit(0)
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(2)


if __name__ == '__main__':
    _run()
