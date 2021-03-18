import torch
import math
import numbers
import ghalton
from abc import ABC, abstractmethod

from R3L.models.Model import Features


class RFF(Features):
    """
    Vanilla Random Fourier Features.
    Note: make sure input space is normalised
    """

    def to_features(self, x):
        pass

    def __init__(self, m, d, sigma, cos_only=False, quasi_random=True,
                 kernel="RBF", dtype=torch.float, device=torch.device("cpu")):
        """
        RFF for RBF kernel.
        """
        self.m = int(m)
        self.n_features = self.m
        self.d = int(d)
        self.coeff = None
        self.offset = None
        self.a = 1.0
        self.dtype = dtype
        self.device = device

        # Fix sigma
        if isinstance(sigma, numbers.Number):
            sigma = torch.ones(d, dtype=dtype, device=device) * sigma
        elif isinstance(sigma, list):
            sigma = torch.tensor(sigma, dtype=dtype, device=device)
        self.sigma = sigma.to(dtype=dtype, device=device)

        if kernel == "RBF":
            rff_kernel = RFFKernelRBF(dtype, device)
        elif kernel == "Laplace" or kernel == "Matern12":
            rff_kernel = RFFKernelMatern12(dtype, device)
        elif kernel == "Matern32":
            rff_kernel = RFFKernelMatern32(dtype, device)
        elif kernel == "Matern52":
            rff_kernel = RFFKernelMatern52(dtype, device)
        else:
            raise ValueError("Kernel {} is not recognised.".format(kernel))

        self.quasiRandom = quasi_random
        self.cos_only = cos_only
        if self.cos_only:  # cos only features
            self.coeff = self._draw_coeff(rff_kernel, m)
            self.offset = 2.0 * math.pi * torch.rand(1, m, dtype=self.dtype,
                                                     device=device)
            self.a = math.sqrt(1.0 / float(self.m))
            self.to_features = self._to_cos_only_features
        else:  # "cossin"
            assert m % 2 == 0 and "RFF: Number of fts must be multiple of 2."
            self.coeff = self._draw_coeff(rff_kernel, int(m // 2))
            self.a = math.sqrt(1.0 / float(self.m / 2))
            self.to_features = self._to_cos_sin_features

    def _draw_coeff(self, rff_kernel, m):
        if self.quasiRandom:
            perms = ghalton.EA_PERMS[:self.d]
            sequencer = ghalton.GeneralizedHalton(perms)
            points = torch.tensor(sequencer.get(m + 1), dtype=self.dtype,
                                  device=self.device)[1:]
            freqs = rff_kernel.inv_cdf(points)
            return freqs / self.sigma.reshape(1, len(self.sigma))

        else:
            freqs = rff_kernel.sample_freqs((m, self.d))
            return freqs / self.sigma.reshape(1, len(self.sigma))

    def _to_cos_only_features(self, x):
        inner = x.mm(self.coeff.t())
        return self.a * torch.cos(inner + self.offset)

    def _to_cos_sin_features(self, x):
        inner = x.mm(self.coeff.t())
        return self.a * torch.cat((torch.cos(inner), torch.sin(inner)), 1)


class RFFKernel(ABC):
    def __init__(self, dtype, device):
        self.dtype = dtype
        self.device = device

    @abstractmethod
    def sample_freqs(self, shape):
        pass

    @abstractmethod
    def inv_cdf(self, x):
        pass


class RFFKernelRBF(RFFKernel):
    def sample_freqs(self, shape, dtype=torch.dtype,
                     device=torch.device("cpu")):
        return torch.normal(
            torch.zeros(shape, dtype=self.dtype, device=self.device))

    def inv_cdf(self, x):
        return torch.erfinv(2 * x - 1) * math.sqrt(2)


class RFFKernelMatern12(RFFKernel):
    def sample_freqs(self, shape):
        m = torch.distributions.chi2.Chi2(
            torch.tensor([1.0], dtype=self.dtype, device=self.device))
        sqm = torch.sqrt(1.0 / m.sample(shape))
        return torch.normal(0, 1, shape, dtype=self.dtype) * sqm

    def inv_cdf(self, x):
        # This formula comes from the inv cdf of a standard cauchy
        # distribution (see Laplace RFF).
        return torch.tan(math.pi * (x - 0.5))


class RFFKernelMatern32(RFFKernel):
    def sample_freqs(self, shape):
        m = torch.distributions.chi2.Chi2(
            torch.tensor([3.0], dtype=self.dtype, device=self.device))
        return torch.normal(
            0, 1, shape, dtype=self.dtype, device=self.device) * torch.sqrt(
            3.0 / m.sample(shape))

    def inv_cdf(self, x):
        # From https://www.researchgate.net/profile/William_Shaw9/publication/
        # 247441442_Sampling_Student%27%27s_T_distribution-use_of_the_inverse_
        # cumulative_distribution_function/links/55bbbc7908ae9289a09574f6/
        # Sampling-Students-T-distribution-use-of-the-inverse-cumulative-
        # distribution-function.pdf
        return (2 * x - 1) / torch.sqrt(2 * x * (1 - x))


class RFFKernelMatern52(RFFKernel):
    def sample_freqs(self, shape):
        m = torch.distributions.chi2.Chi2(
            torch.tensor([5.0], dtype=self.dtype, device=self.device))
        return torch.normal(
            0, 1, shape, dtype=self.dtype, device=self.device) * torch.sqrt(
            5.0 / m.sample(shape))

    def inv_cdf(self, x):
        # From https://www.researchgate.net/profile/William_Shaw9/publication/
        # 247441442_Sampling_Student%27%27s_T_distribution-use_of_the_inverse_
        # cumulative_distribution_function/links/55bbbc7908ae9289a09574f6/
        # Sampling-Students-T-distribution-use-of-the-inverse-cumulative-
        # distribution-function.pdf
        alpha = 4 * x * (1 - x)
        p = 4 * torch.cos(torch.acos(torch.sqrt(alpha)) / 3) / torch.sqrt(alpha)
        return torch.sign(x - 0.5) * torch.sqrt(p - 4)
