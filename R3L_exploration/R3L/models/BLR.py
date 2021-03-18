import math
import torch

from R3L.models.Model import StreamingModel
from R3L.utils import batch_generator, smw_inv_correction


class VanillaBLR(StreamingModel):
    """
    Vanilla Bayesian Linear Regression. Uses matrix inversion to compute Sn.
    """

    def __init__(self, m, alpha=0.1, beta=1.0, d_out=1, dtype=torch.float):
        """
        :param m: input dimension.
        :param alpha: Prior precision.
        "param beta: Noise precision.
        :param d_out: Number of output dimentions.
        """
        self.m = m
        self.alpha = alpha
        self.beta = beta
        self.d_out = d_out
        self.mn = None
        self.sn = None
        self.invsn = None
        self.sn_inv_tgt = None
        self.tgt0 = None
        self.is_model_init = False
        self.dtype = dtype
        self.need_recompute_params = True
        self.reset()

    def reset(self):
        self.sn_inv_tgt = torch.zeros(size=(self.m, self.d_out),
                                      dtype=self.dtype)
        self.mn = torch.zeros(size=(self.m, self.d_out), dtype=self.dtype)
        self.tgt0 = torch.zeros(size=(self.m, self.d_out), dtype=self.dtype)
        self.invsn = torch.eye(self.m, dtype=self.dtype) * self.alpha
        self.sn = torch.eye(self.m, dtype=self.dtype) / self.alpha

        self.need_recompute_params = True
        self.is_model_init = False

    def is_init(self):
        return self.is_model_init

    def update(self, phi, y):
        self.need_recompute_params = True
        self.is_model_init = True

        # BLR update
        self.invsn += self.beta * phi.t().mm(phi)
        self.sn_inv_tgt += self.beta * phi.t().mm(y)

    def fit(self, phi, y):
        self.update(phi, y)

    def _recompute(self):
        self.sn = torch.inverse(self.invsn)
        self.mn = self.sn.mm(self.sn_inv_tgt)
        self.need_recompute_params = False

    def predict_mean(self, phi):
        assert len(phi.shape) == 2
        if self.need_recompute_params:
            self._recompute()
        return self._predict_mean(phi)

    def _predict_mean(self, phi):
        return phi.mm(self.mn).reshape(-1, self.d_out)

    def predict_var(self, phi):
        assert len(phi.shape) == 2
        if self.need_recompute_params:
            self._recompute()
        return self._predict_var(phi)

    def _predict_var(self, phi, include_beta_var=True):
        var = torch.sum(phi.mm(self.sn) * phi, dim=1, keepdim=True)
        if include_beta_var:
            var += 1.0 / self.beta
        return var.reshape(-1, 1)

    def predict(self, phi):
        if self.need_recompute_params:
            self._recompute()
        mean = self._predict_mean(phi)
        var = self._predict_var(phi)
        return mean, var

    def optimise(self, max_evals=200):
        pass

    @property
    def model_params(self):
        if self.need_recompute_params:
            self._recompute()
        return self.mn

    def update_targets(self, all_phi, all_t):
        """
        Update target for all datapoints.
        :param all_phi: Feature map for all data points
        :param all_t: Targets for all data points
        """
        self.sn_inv_tgt = self.tgt0 + self.beta * all_phi.t().mm(all_t)
        self.need_recompute_params = True


class BLR(VanillaBLR):
    """
    Purely analytic BLR with rank-k streaming updates
    """

    def __init__(self, m, alpha=1.0, beta=1.0, d_out=1, compute_sninv=True,
                 dtype=torch.float):
        """
        Initialize model.
        :param m: Number of weights
        :param alpha: Weight prior precision
        :param beta: Noise precision
        :param d_out: Number of output dimensions
        :param compute_sninv: Whether to compute the inverse of Sn. Useful for
        NLML computations (default=True).
        """
        super().__init__(m, alpha, beta, d_out, dtype)
        self.compute_sninv = compute_sninv

    def update(self, phi, y):
        """
        Update BLR model with one a set of data points, performing a rank-k
        sherman-morisson-woodburry update.
        :param phi: Feature map for new points
        :param y: Target value for new points
        """
        self.sn -= smw_inv_correction(a_inv=self.sn,
                                      u=math.sqrt(self.beta) * phi.t(),
                                      v=math.sqrt(self.beta) * phi)

        self.sn_inv_tgt += self.beta * phi.t().mm(y)
        if self.compute_sninv:
            self.invsn += self.beta * phi.t().mm(phi)
        self.need_recompute_mean = True
        self.is_model_init = True

    def fit(self, all_phi, all_y, batch_size=None):
        """
        Fits model on dataset by cutting it into batches for faster learning.
        :param all_phi: data features
        :param all_y: data targets
        :param batch_size: size of batches data set is cut into. Set to None
        for automatically computing optimal batch size (default=None).
        """
        # Define the batch data generator. This maintains an internal counter
        # and also allows wraparound for multiple epochs
        # Compute optimal batch size
        if batch_size is None:
            batch_size = int((self.m ** 2 / 2) ** (1. / 3.))

        data_batch_gen = batch_generator(arrays=[all_phi, all_y],
                                         batch_size=batch_size,
                                         wrap_last_batch=False)

        n = all_phi.shape[0]  # Alias for the total number of training samples
        n_batches = int(math.ceil(n / batch_size))  # The number of batches

        """ Run the batched inference """
        for _ in range(n_batches):
            phi_batch, y_batch = next(data_batch_gen)
            self.update(phi=phi_batch, y=y_batch)

    def _recompute(self):
        self.mn = self.sn.mm(self.sn_inv_tgt)
        self.need_recompute_mean = False

