import pickle
from pathlib import Path
import numpy as np
from tqdm import trange


class DataExplorer:
    BASE_DIR = Path('datasets/pav_features/')

    def __init__(self):
        with (self.BASE_DIR / 'train_split_image_paths.txt').open('r') as f:
            self.images = f.read().splitlines()
        self.train_data = np.load(self.BASE_DIR / 'train_split.npy')
        self.valid_data = np.load(self.BASE_DIR / 'valid_split.npy')
        self.total_iterations = []
        self.clean_iterations = []
        self.defec_iterations = []
        self.selected = np.zeros(self.train_data.shape[0], dtype=bool)

    def select(self, size: int = 50, weights: np.ndarray = None):
        if weights is None:
            idxs = np.random.choice(
                np.arange(self.train_data.shape[0]),
                size=size,
                replace=False,
            )
        else:
            weights = weights.flatten()
            weights[self.selected] = np.inf
            idxs = np.argpartition(weights, size)[:size]
        self.total_iterations.append(idxs)
        clean_idxs, defec_idxs = self._split_by_class(idxs)
        self.clean_iterations.append(clean_idxs)
        self.defec_iterations.append(defec_idxs)
        self.selected[idxs] = True
        return clean_idxs, defec_idxs

    def select_all_remaining(self):
        idxs = np.nonzero(~self.selected)[0]
        self.total_iterations.append(idxs)
        clean_idxs, defec_idxs = self._split_by_class(idxs)
        self.clean_iterations.append(clean_idxs)
        self.defec_iterations.append(defec_idxs)
        self.selected[idxs] = True
        return clean_idxs, defec_idxs

    def _split_by_class(self, idxs: np.ndarray):
        labels = self.train_data[idxs, -1].astype(bool)
        defec_idxs = idxs[labels]
        clean_idxs = idxs[~labels]
        return clean_idxs, defec_idxs

    def get_clean_selected(self, iteration: int = None):
        if iteration is None:
            clean_idxs = np.concatenate(self.clean_iterations)
        else:
            clean_idxs = self.clean_iterations[iteration]
        return self.train_data[clean_idxs, :-1], [self.images[i] for i in clean_idxs]

    def get_defec_selected(self, iteration: int = None):
        if iteration is None:
            defec_idxs = np.concatenate(self.defec_iterations)
        else:
            defec_idxs = self.defec_iterations[iteration]
        return self.train_data[defec_idxs, :-1], [self.images[i] for i in defec_idxs]

    def get_selected(self, iteration: int = None):
        if iteration is None:
            total_idxs = np.concatenate(self.total_iterations)
        else:
            total_idxs = self.total_iterations[iteration]
        return self.train_data[total_idxs], [self.images[i] for i in total_idxs]


def save_result(
    model,
    latent_dim: int,
    explorer: DataExplorer,
    iteration: int,
    save_dir: Path = Path('results/'),
):
    # Validation results
    valid_proj = model.project(explorer.valid_data[:, :-1])
    valid_logpdf = model.evaluate(explorer.valid_data[:, :-1])
    valid_eval = np.column_stack((valid_proj, explorer.valid_data[:, -1], valid_logpdf))
    # Training results
    train_proj = model.project(explorer.train_data[:, :-1])
    train_logpdf = model.evaluate(explorer.train_data[:, :-1])
    train_eval = np.column_stack((train_proj, explorer.train_data[:, -1], train_logpdf))
    # Save location
    model_name = f'{type(model).__name__.lower()}_{latent_dim:02d}'
    save_folder = save_dir / model_name / f'iteration_{iteration:02d}'
    save_folder.mkdir(parents=True, exist_ok=True)
    # Results saving
    _, clean_image_paths = explorer.get_clean_selected(iteration)
    _, defec_image_paths = explorer.get_defec_selected(iteration)
    if valid_proj.shape[1]:
        np.save(save_folder / 'valid.npy', valid_eval)
        np.save(save_folder / 'train.npy', train_eval)
    with (save_folder / 'clean_images.txt').open('w') as f:
        for path in clean_image_paths:
            f.write(f'{path}\n')
    with (save_folder / 'defec_images.txt').open('w') as f:
        for path in defec_image_paths:
            f.write(f'{path}\n')


def exploration(model_class, latent_dim, selection_size, num_iterations):
    explorer = DataExplorer()
    weights = None
    for i in trange(num_iterations + 1):
        if i == num_iterations:
            explorer.select_all_remaining()
        else:
            explorer.select(selection_size, weights=weights)
        clean_feats, _ = explorer.get_clean_selected()
        model = model_class(latent_dim)
        model.fit(clean_feats)
        save_result(model, latent_dim, explorer, i)
        weights = model.evaluate(explorer.train_data[:, :-1])
