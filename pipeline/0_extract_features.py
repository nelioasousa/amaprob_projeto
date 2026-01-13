import torch
import random
from pathlib import Path
from time import monotonic
from PIL import Image
from tqdm import tqdm
import numpy as np
import ultralytics


FMTS = {'.png', '.jpg', '.jpeg'}


def get_features(image_path, model, num_partitions):
    img = Image.open(image_path)
    width, height = img.size
    partition_width = width // num_partitions
    features = []
    for i in range(num_partitions):
        left = i * partition_width
        right = (i + 1) * partition_width if i < num_partitions - 1 else width
        img_partition = img.crop((left, 0, right, height))
        feats = model.embed(img_partition, verbose=False)[0]
        features.append(feats.cpu().numpy())
    return np.concatenate(features, axis=0)


def extract_features(images_paths, model_name, num_partitions, save_dir):
    images_paths.sort()
    model = ultralytics.YOLO(model_name)
    save_dir.mkdir(parents=True, exist_ok=True)
    root = Path('.').resolve()
    # Feature extraction
    features = [
        get_features(img_path, model, num_partitions)
        for img_path
        in tqdm(images_paths, desc='Feature Extraction', unit='image')
    ]
    features = np.stack(features)
    np.save(save_dir / 'features.npy', features, allow_pickle=False)
    # Save image paths
    with open(save_dir / 'image_paths.txt', 'w') as f:
        for img_path in images_paths:
            f.write(f'{img_path.relative_to(root)}\n')


def _run():
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description='Extract features from images using a pre-trained model.'
    )
    
    parser.add_argument(
        '-d', '--data_dir', type=Path, default=Path('datasets/pav_images/'),
        help='Directory containing images to extract features from.'
    )

    parser.add_argument(
        '-o', '--output_dir', type=Path, default=Path('datasets/pav_features/'),
        help='Directory to save the extracted features.'
    )

    parser.add_argument(
        '-m', '--model_name', type=str, default='yolo11m.pt',
        help='Pre-trained model name.'
    )

    parser.add_argument(
        '-p', '--num_partitions', type=int, default=1,
        help='Number of partitions to split each image for feature extraction.'
    )

    parser.add_argument(
        '-rs', '--random_seed', type=int, default=42,
        help='Random seed for reproducibility.'
    )

    args = parser.parse_args()

    args.data_dir = args.data_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    imgs = [
        f for f in args.data_dir.rglob('*.*')
        if f.suffix.lower() in FMTS
    ]

    try:
        start = monotonic()
        print(f'[Script 0] Extracting features from {len(imgs)} images...')
        torch.manual_seed(args.random_seed)
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        extract_features(
            images_paths=imgs,
            model_name=args.model_name,
            num_partitions=args.num_partitions,
            save_dir=args.output_dir,
        )
        print(f'[Script 0] Feature extraction completed successfully ({monotonic() - start:.2f} secs)')
        sys.exit(0)
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(2)


if __name__ == '__main__':
    _run()
