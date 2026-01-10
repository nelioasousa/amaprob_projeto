import numpy as np
from numbers import Real
from scipy.stats import multivariate_normal


def _get_M_inverse(W_proj: np.ndarray, sigma_var: Real):
    M = W_proj.transpose().dot(W_proj)
    M[:] += sigma_var * np.identity(M.shape[0], dtype=M.dtype)
    try:
        M_inv = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        M[:] += 1e-15 * np.identity(M.shape[0], dtype=M.dtype)
        M_inv = np.linalg.inv(M)
    return M_inv


def _get_marginal_cov_and_inv(
    W_proj: np.ndarray,
    sigma_var: Real,
):
    M_inv = _get_M_inverse(W_proj, sigma_var)
    cov = W_proj.dot(W_proj.transpose())
    cov[:] += sigma_var * np.identity(cov.shape[0], dtype=cov.dtype)
    cov_inv = - W_proj.dot(M_inv).dot(W_proj.transpose()) / sigma_var
    cov_inv[:] += np.identity(cov_inv.shape[0], dtype=cov_inv.dtype) / (sigma_var ** 0.5)
    return cov, cov_inv, M_inv


def _params_update(
    W_proj: np.ndarray,
    sigma_var: Real,
    X_to_mean: np.ndarray,
    M_inv: np.ndarray,
):
    N, D = X_to_mean.shape
    expec_z = X_to_mean.dot(W_proj).dot(M_inv.transpose())
    expec_z = expec_z[..., np.newaxis]
    expec_z_zt = np.matmul(expec_z, expec_z.transpose((0, 2, 1)))
    expec_z_zt[:] += sigma_var * M_inv
    # Update W_proj
    sum_z_zt = expec_z_zt.sum(axis=0)
    try:
        sum_z_zt_inv = np.linalg.inv(sum_z_zt)
    except np.linalg.LinAlgError:
        sum_z_zt[:] += 1e-15 * np.identity(sum_z_zt.shape[0], dtype=sum_z_zt.dtype)
        sum_z_zt_inv = np.linalg.inv(sum_z_zt)
    batch_X_to_mean = X_to_mean[..., np.newaxis]
    new_W_proj = np.matmul(batch_X_to_mean, expec_z.transpose((0, 2, 1)))
    new_W_proj = new_W_proj.sum(axis=0).dot(sum_z_zt_inv)
    # Update sigma_var
    s_dist = np.pow(np.linalg.norm(X_to_mean, axis=1), 2)
    s_z = np.matmul(expec_z.transpose((0, 2, 1)).dot(new_W_proj.transpose()), batch_X_to_mean)
    s_z = -2 * s_z.ravel()
    s_z_zt = expec_z_zt.dot(new_W_proj.transpose().dot(new_W_proj))
    s_z_zt = np.linalg.trace(s_z_zt)
    new_sigma_var = (np.sum(s_dist + s_z + s_z_zt) / (N * D)).item()
    return new_W_proj, new_sigma_var


def _params_init(projection_dim: int, X: np.ndarray):
    D = X.shape[1]
    L = projection_dim
    mean = X.mean(axis=0)
    W_proj = np.random.normal(size=(D, L))
    sigma_var = 100.0 ** 2
    return mean, W_proj, sigma_var


class PPCA:

    def __init__(
        self,
        projection_dim: int,
        precision_percentage: float = 1e-12,
        plateau_size: int = 10,
        max_iterations: int = 100,
    ):
        self.mean = None
        self.proj = None
        self.var = None
        self.projection_dim = projection_dim
        self.precision_percentage = precision_percentage
        self.plateau_size = plateau_size
        self.max_iterations = max_iterations
    
    def fit(self, X: np.ndarray):
        if self.projection_dim > X.shape[1]:
            raise ValueError('Projection dimension greater than data dimension')
        self.D = X.shape[1]
        self.L = self.projection_dim
        self.mean, self.proj, self.var = _params_init(self.projection_dim, X)
        self.marginal_cov, self.marginal_cov_inv, self.M_inv = _get_marginal_cov_and_inv(
            self.proj,
            self.var,
        )
        X_to_mean = X - self.mean
        old_metric = self.get_log_likelihood(X_to_mean, subtract_mean=False)
        stabilization_countdown = self.plateau_size
        metric_history = [old_metric]
        for _ in range(1, self.max_iterations + 1):
            self.proj, self.var = _params_update(self.proj, self.var, X_to_mean, self.M_inv)
            self.marginal_cov, self.marginal_cov_inv, self.M_inv = _get_marginal_cov_and_inv(
                self.proj,
                self.var,
            )
            new_metric = self.get_log_likelihood(X_to_mean, subtract_mean=False)
            metric_history.append(new_metric)
            if (old_metric - new_metric) / old_metric < self.precision_percentage:
                stabilization_countdown -= 1
                if not stabilization_countdown:
                    break
            else:
                stabilization_countdown = self.plateau_size
            old_metric = new_metric
        self.marginal_dist = multivariate_normal(self.mean, self.marginal_cov)
        return metric_history
    
    def get_log_likelihood(self, X: np.ndarray, subtract_mean: bool = True):
        if self.mean is None:
            raise RuntimeError('Model not fitted')
        if subtract_mean:
            X -= self.mean
        N, D = X.shape
        log_like = -0.5 * (X * X.dot(self.marginal_cov_inv.transpose())).sum().item()
        _, abs_log_det = np.linalg.slogdet(self.marginal_cov)
        log_like -= (N * abs_log_det / 2).item()
        log_like -= (N * D * np.log(2 * np.pi) / 2).item()
        return log_like

    def get_marginal_logprob(self, X: np.ndarray):
        return self.marginal_dist.logpdf(X)

    def evaluate(self, X: np.ndarray):
        return self.get_marginal_logprob(X)

    def generate(self, num_samples: int):
        if self.mean is None:
            raise RuntimeError('Model not fitted')
        latents = multivariate_normal.rvs(
            mean=np.zeros(self.L),
            cov=np.identity(self.L),
            size=num_samples,
        )
        clean = latents.dot(self.proj.transpose()) + self.mean
        noisy = clean + multivariate_normal.rvs(
            mean=np.zeros_like(self.mean),
            cov=(self.var * np.identity(self.D)),
            size=num_samples,
        )
        return clean, noisy
    
    def reconstruct(self, X: np.ndarray):
        if self.mean is None:
            raise RuntimeError('Model not fitted')
        if X.shape[1] != self.D:
            raise ValueError('Samples dimension not compatible with fitted model')
        X_to_mean = X - self.mean
        projections = X_to_mean.dot(self.proj)
        reconstructions = projections.dot(self.proj.transpose()) + self.mean
        error = np.pow(np.linalg.norm(X - reconstructions, axis=1), 2).sum().item()
        return reconstructions, error

    def project(self, X: np.ndarray):
        posterior_mean = self.M_inv.dot(self.proj.transpose()).dot((X - self.mean).transpose())
        return posterior_mean.transpose()
