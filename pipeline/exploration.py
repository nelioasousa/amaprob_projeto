import pickle
from pathlib import Path
import numpy as np
from tqdm import trange


class DataExplorer:
    BASE_DIR = Path('datasets/pav_features/')

    def __init__(self):
        with (self.BASE_DIR / 'train_split_image_paths.txt').open('r') as f:
            self.images = [l.strip() for l in f.readlines()]
        self.data = np.load(self.BASE_DIR / 'train_split.npy')
        self.total_iterations = []
        self.clean_iterations = []
        self.defec_iterations = []
        self.selected = np.zeros(self.data.shape[0], dtype=bool)

    def select(self, size: int = 50, weights: np.ndarray = None):
        if weights is None:
            idxs = np.random.choice(np.arange(self.data.shape[0]), size=size, replace=False)
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

    def _split_by_class(self, idxs: np.ndarray):
        labels = self.data[idxs, -1].astype(bool)
        defec_idxs = idxs[labels]
        clean_idxs = idxs[~labels]
        return clean_idxs, defec_idxs

    def get_clean_selected(self, iteration: int = None):
        if iteration is None:
            clean_idxs = np.concatenate(self.clean_iterations)
        else:
            clean_idxs = self.clean_iterations[iteration]
        return self.data[clean_idxs, :-1], [self.images[i] for i in clean_idxs]

    def get_defec_selected(self, iteration: int = None):
        if iteration is None:
            defec_idxs = np.concatenate(self.defec_iterations)
        else:
            defec_idxs = self.defec_iterations[iteration]
        return self.data[defec_idxs, :-1], [self.images[i] for i in defec_idxs]

    def get_selected(self, iteration: int = None):
        if iteration is None:
            total_idxs = np.concatenate(self.total_iterations)
        else:
            total_idxs = self.total_iterations[iteration]
        return self.data[total_idxs], [self.images[i] for i in total_idxs]


def save_result(
    model,
    explorer: DataExplorer,
    iteration: int = None,
    valid_eval: np.ndarray = None,
    save_dir: Path = Path('results/'),
):
    model_name = type(model).__name__.lower()
    result_str = 'final' if iteration is None else f'iteration_{iteration:02d}'
    save_folder = save_dir / model_name / result_str
    save_folder.mkdir(parents=True, exist_ok=True)
    clean_features, clean_image_paths = explorer.get_clean_selected(iteration)
    defec_features, defec_image_paths = explorer.get_defec_selected(iteration)
    np.save(save_folder / 'clean.npy', clean_features)
    np.save(save_folder / 'defec.npy', defec_features)
    if valid_eval is not None:
        np.save(save_folder / 'valid.npy', valid_eval)
    with (save_folder / 'model.pkl').open('wb') as f:
        pickle.dump(model, f)
    with (save_folder / 'clean_image_paths.txt').open('w') as f:
        for path in clean_image_paths:
            f.write(f'{path}\n')
    with (save_folder / 'defec_image_paths.txt').open('w') as f:
        for path in defec_image_paths:
            f.write(f'{path}\n')


def exploration(model_class, latent_dim, selection_size, num_iterations):
    valid_data = np.load(DataExplorer.BASE_DIR / 'valid_split.npy')
    explorer = DataExplorer()
    weights = None
    for i in trange(num_iterations):
        explorer.select(selection_size, weights=weights)
        clean_feats, _ = explorer.get_clean_selected()
        model = model_class(latent_dim)
        model.fit(clean_feats)
        valid_logpdf = model.evaluate(valid_data[:, :-1])
        save_result(model, explorer, i, valid_logpdf)
        weights = model.evaluate(explorer.data[:, :-1])
    save_result(model, explorer)
