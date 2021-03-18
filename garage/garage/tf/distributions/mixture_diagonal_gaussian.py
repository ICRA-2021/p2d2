import numpy as np
import tensorflow as tf
from scipy.special import logsumexp

from garage.tf.distributions.base import Distribution
from garage.tf.distributions import DiagonalGaussian


class MixtureDiagonalGaussian(Distribution):
    def __init__(self, dim, size, name="MixtureDiagonalGaussian"):
        self._dim = dim
        self._size = size
        self._name = name
        self._diag = DiagonalGaussian(dim=dim)

    @property
    def dim(self):
        return self._dim

    @property
    def size(self):
        return self._size

    def kl(self, old_dist_info, new_dist_info):
        old_means = old_dist_info["mean"]
        old_log_stds = old_dist_info["log_std"]
        new_means = new_dist_info["mean"]
        new_log_stds = new_dist_info["log_std"]
        """
        Compute the KL divergence of two multivariate Gaussian distribution
        with diagonal covariance matrices
        """
        old_std = np.exp(old_log_stds)
        new_std = np.exp(new_log_stds)
        # means: (N*A)
        # std: (N*A)
        # formula:
        # { (\mu_1 - \mu_2)^2 + \sigma_1^2 - \sigma_2^2 } / (2\sigma_2^2) +
        # ln(\sigma_2/\sigma_1)
        numerator = np.square(old_means - new_means) + \
            np.square(old_std) - np.square(new_std)
        denominator = 2 * np.square(new_std) + 1e-8
        return np.sum(
            numerator / denominator + new_log_stds - old_log_stds, axis=-1)

    def kl_sym(self, old_dist_info_vars, new_dist_info_vars, name=None):
        with tf.name_scope(name, "kl_sym",
                           [old_dist_info_vars, new_dist_info_vars]):
            old_means = old_dist_info_vars["mean"]
            old_log_stds = old_dist_info_vars["log_std"]
            new_means = new_dist_info_vars["mean"]
            new_log_stds = new_dist_info_vars["log_std"]
            """
            Compute the KL divergence of two multivariate Gaussian distribution
            with diagonal covariance matrices
            """
            old_std = tf.exp(old_log_stds)
            new_std = tf.exp(new_log_stds)
            # means: (N*A)
            # std: (N*A)
            # formula:
            # { (\mu_1 - \mu_2)^2 + \sigma_1^2 - \sigma_2^2 } / (2\sigma_2^2) +
            # ln(\sigma_2/\sigma_1)
            numerator = tf.square(old_means - new_means) + \
                tf.square(old_std) - tf.square(new_std)
            denominator = 2 * tf.square(new_std) + 1e-8
            return tf.reduce_sum(
                numerator / denominator + new_log_stds - old_log_stds, axis=-1)

    def likelihood_ratio_sym(self,
                             x_var,
                             old_dist_info_vars,
                             new_dist_info_vars,
                             name=None):
        with tf.name_scope(name, "likelihood_ratio_sym",
                           [x_var, old_dist_info_vars, new_dist_info_vars]):
            logli_new = self.log_likelihood_sym(x_var, new_dist_info_vars)
            logli_old = self.log_likelihood_sym(x_var, old_dist_info_vars)
            return tf.exp(logli_new - logli_old)

    def log_likelihood_sym(self, xs_var, dist_info_vars, name=None):
        with tf.name_scope(name, "log_likelihood_sym",
                           [xs_var, dist_info_vars]):
            size, dim = self._size, self._dim
            weights = dist_info_vars["weight"]
            means = dist_info_vars["mean"]
            log_stds = dist_info_vars["log_std"]
            norm_weights = tf.ones_like(weights) / self.size
            #norm_weights = weights / tf.reduce_sum(weights, axis=-1, keepdims=True)
            log_weights = tf.log(norm_weights)
            log_probs = self._diag.log_likelihood_sym(
                tf.reshape(xs_var, (-1,1,dim)),
                dict(
                    mean=tf.reshape(means, (-1,size,dim)),
                    log_std=tf.reshape(log_stds, (-1,size,dim))
                )
            )
            return tf.reduce_logsumexp(log_weights + log_probs, axis=-1)


    def log_likelihood(self, xs, dist_info):
        size, dim = self._size, self._dim
        weights = dist_info["weight"]
        means = dist_info["mean"]
        log_stds = dist_info["log_std"]
        norm_weights = weights / weights.sum(axis=-1).reshape(-1,1)
        log_weights = np.log(norm_weights)
        log_probs = self._diag.log_likelihood(
            xs.reshape(-1,1,dim),
            dict(
                mean=means.reshape(-1,size,dim),
                log_std=log_stds.reshape(-1,size,dim)
            )
        )
        return logsumexp(log_weights + log_probs, axis=-1)

    def sample(self, dist_info):
        weights = dist_info["weight"]
        norm_weights = weights / weights.sum(axis=-1)
        idx = np.random.choice(np.arange(self._size), p=norm_weights)
        means = dist_info["mean"][idx]
        log_stds = dist_info["log_std"][idx]
        rnd = np.random.normal(size=means.shape)
        return rnd * np.exp(log_stds) + means

    def sample_n(self, dist_info, n):
        weights = dist_info["weight"]
        norm_weights = weights / weights.sum(axis=-1)
        idx = np.random.choice(np.arange(self._size), size=n, p=norm_weights)

    def explained_variance(self, dist_info):
        weights = dist_info_var["weight"]
        means = dist_info_var["mean"]
        norm_weights=weights
        #norm_weights = ( weights / weights.sum(axis=-1).reshape(-1,1)
        #    ).reshape(-1, self._size, 1)
        uni_means = np.sum(norm_weights * means, axis=1)
        epistemic = np.sum(
                norm_weights * np.square(means - uni_means), axis=1
            )
        return epistemic

    def explained_variance_sym(self, dist_info_var, name=None):
        with tf.name_scope(name, "epistemic_sym",
                           [dist_info_var]):
            weights_var = dist_info_var["weight"]
            means_var = dist_info_var["mean"]
            norm_weights = tf.reshape(weights_var, (-1,self.size,1))
            norm_weights = tf.ones_like(norm_weights) / self.size
            #norm_weights = tf.reshape(
            #    weights_var / tf.reduce_sum(weights_var,axis=-1,keepdims=True),
            #    (-1, self._size, 1)
            #)
            uni_means = tf.reduce_sum(norm_weights * means_var, axis=1)
            epistemic = tf.reduce_sum(
                    norm_weights * tf.square(means_var), axis=1
                ) \
                - tf.square(uni_means)
            return epistemic

    def approximate_gaussian(self, dist_info):
        weights = dist_info["weight"]
        means = dist_info["mean"]
        log_stds = dist_info["log_std"]
        norm_weights = ( weights / weights.sum(axis=-1).reshape(-1,1)
            ).reshape(-1, self._size, 1)
        uni_means = np.sum(norm_weights * means, axis=1)
        uni_vars = np.sum(
                norm_weights * (np.exp(2*log_stds) + means**2), axis=1
            ) \
            - uni_means**2
        uni_log_stds = 0.5*np.log(uni_vars)
        return dict(mean=uni_means, log_std=uni_log_stds)

    def approximate_gaussian_sym(self, dist_info_var, name=None):
        with tf.name_scope(name, "approximate_gaussian_sym",
                           [dist_info_var]):
            weights_var = dist_info_var["weight"]
            means_var = dist_info_var["mean"]
            log_stds_var = dist_info_var["log_std"]
            norm_weights = tf.reshape(
                weights_var / tf.reduce_sum(weights_var,axis=-1,keepdims=True),
                (-1, self._size, 1)
            )
            uni_means = tf.reduce_sum(norm_weights * means_var, axis=1)
            uni_vars = tf.reduce_sum(
                    norm_weights * (tf.exp(2*log_stds_var) + means_var**2), axis=1
                ) \
                - uni_means**2
            uni_log_stds = 0.5*tf.log(uni_vars)
            return dict(mean=uni_means, log_std=uni_log_stds)

    def entropy(self, dist_info):
        log_stds = approximate_gaussian(dist_info)["log_std"]
        return np.sum(log_stds + np.log(np.sqrt(2 * np.pi * np.e)), axis=-1)

    def entropy_sym(self, dist_info_var, name=None):
        with tf.name_scope(name, "entropy_sym", [dist_info_var]):
            log_stds_var = approximate_gaussian_sym(dist_info_var)["log_std"]
            return tf.reduce_sum(
                log_stds_var + np.log(np.sqrt(2 * np.pi * np.e)), axis=-1)

    @property
    def dist_info_specs(self):
        return [("mean", (self.dim, )), ("log_std", (self.dim, ))]
