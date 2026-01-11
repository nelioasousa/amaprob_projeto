import random
from pathlib import Path
from time import monotonic
import numpy as np


def get_image_groups(images):
    groups = []
    actual_group = []
    last_group_id = None
    for img_idx, img_path in images:
        _, bus, date, _, frame = Path(img_path).stem.split('_')
        frame = int(frame.split('.')[0])
        group_id = (bus, date, frame)
        belongs_to = (
            last_group_id is None
            or (bus, date, frame - 1) == last_group_id
        )
        if belongs_to:
            actual_group.append(img_idx)
        else:
            groups.append(actual_group)
            actual_group = [img_idx]
        last_group_id = group_id
    return groups


def train_val_split(data_dir, train_ratio):
    with open(data_dir / 'image_paths.txt', 'r') as f:
        image_paths = f.readlines()
    defec_images = [(i, p) for i, p in enumerate(image_paths) if 'defec' in p]
    clean_images = [(i, p) for i, p in enumerate(image_paths) if 'clean' in p]
    defec_groups = get_image_groups(defec_images)
    clean_groups = get_image_groups(clean_images)
    num_train_defec = int(len(defec_groups) * train_ratio)
    num_train_clean = int(len(clean_groups) * train_ratio)
    random.shuffle(defec_groups)
    random.shuffle(clean_groups)
    train_indexes = []
    for group in defec_groups[:num_train_defec]:
        train_indexes.extend(group)
    for group in clean_groups[:num_train_clean]:
        train_indexes.extend(group)
    train_indexes.sort()
    features = np.load(data_dir / 'features.npy')
    y = np.array(
        [('defec' in p) for p in image_paths], dtype=bool,
    ).reshape(-1, 1)
    train_mask = np.zeros(len(image_paths), dtype=bool)
    train_mask[train_indexes] = True
    X_train = features[train_mask]
    y_train = y[train_mask]
    train_split = np.concatenate((X_train, y_train), axis=1)
    X_valid = features[~train_mask]
    y_valid = y[~train_mask]
    valid_split = np.concatenate((X_valid, y_valid), axis=1)
    np.save(data_dir / 'train_split.npy', train_split, allow_pickle=False)
    np.save(data_dir / 'valid_split.npy', valid_split, allow_pickle=False)
    valid_indexes = set(range(len(image_paths))) - set(train_indexes)
    train_image_paths = [image_paths[i] for i in train_indexes]
    valid_image_paths = [image_paths[i] for i in sorted(valid_indexes)]
    with open(data_dir / 'train_split_image_paths.txt', 'w') as f:
        f.writelines(train_image_paths)
    with open(data_dir / 'valid_split_image_paths.txt', 'w') as f:
        f.writelines(valid_image_paths)


def _run():
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description='Split dataset into training and validation sets.'
    )

    parser.add_argument(
        '-d', '--data_dir', type=Path, default=Path('datasets/pav_features/'),
        help='Directory containing extracted features and image paths.'
    )

    parser.add_argument(
        '-t', '--train_ratio', type=float, default=0.7,
        help='Proportion of data to use for training.'
    )

    parser.add_argument(
        '-rs', '--random_seed', type=int, default=42,
        help='Random seed for shuffling data before splitting.'
    )

    args = parser.parse_args()
    args.data_dir = args.data_dir.resolve()

    try:
        start = monotonic()
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        train_val_split(
            data_dir=args.data_dir,
            train_ratio=args.train_ratio,
        )
        print(f'[1] Train-val split completed successfully ({monotonic() - start:.2f} secs)')
        sys.exit(0)
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(2)


if __name__ == '__main__':
    _run()
