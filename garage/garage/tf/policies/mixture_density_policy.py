from akro.tf import Box
import numpy as np
import tensorflow as tf

from garage.core import Serializable
from garage.misc import logger
from garage.misc.overrides import overrides
from garage.tf.core import LayersPowered
import garage.tf.core.layers as L
from garage.tf.core.network import MLP
from garage.tf.distributions import MixtureDiagonalGaussian
from garage.tf.misc import tensor_utils
from garage.tf.policies.base import StochasticPolicy


class MixtureDensityPolicy(StochasticPolicy, LayersPowered, Serializable):
    def __init__(self,
                 env_spec,
                 mixture_network,
                 name="MixtureDensityPolicy",
                 min_std=1e-6,):
        """
        :param env_spec:
        :param mixture_network: a MDN
        :return:
        """
        assert isinstance(env_spec.action_space, Box)

        Serializable.quick_init(self, locals())
        self.name = name
        self._mixture_network_name = "mixture_network"

        with tf.variable_scope(name, "MixtureDensityPolicy"):

            obs_dim = env_spec.observation_space.flat_dim
            action_dim = env_spec.action_space.flat_dim
            assert action_dim == mixture_network.dim

            l_mean, l_log_std, l_w = mixture_network.output_layer
            self._mixture_network = mixture_network

            obs_var = mixture_network.input_layer.input_var
            self._obs_var = obs_var
            min_std_param = np.log(min_std)
            self.min_std_param = min_std_param
            self._l_mean = l_mean
            self._l_log_std = l_log_std
            self._l_w = l_w
            self._outputs = dict(mean=l_mean, log_std=l_log_std, weight=l_w)

            self.sig_rate = tf.placeholder_with_default(
                tf.convert_to_tensor(1.0, dtype=tf.float32), shape=[],name='sig_rate')
            self._dist = MixtureDiagonalGaussian(action_dim, mixture_network.M)

            LayersPowered.__init__(self, mixture_network.output_layer)
            super(MixtureDensityPolicy, self).__init__(env_spec)

            dist_info_sym = self.dist_info_sym(
                mixture_network.input_layer.input_var, dict())
            mean_var = tf.identity(dist_info_sym["mean"], name="mean")
            log_std_var = tf.identity(
                dist_info_sym["log_std"], name="standard_dev")
            weight_var = tf.identity(dist_info_sym["weight"], name="weight")

            self._f_dist = tensor_utils.compile_function(
                inputs=[obs_var],
                outputs=[mean_var, log_std_var, weight_var],
            )

    @property
    def vectorized(self):
        return True

    def dist_info_sym(self, obs_var, state_info_vars=None, name=None):
        with tf.name_scope(name, "dist_info_sym", [obs_var]):
            with tf.name_scope(self._mixture_network_name, values=[obs_var]):
                mean_var = L.get_output(self._l_mean, obs_var)
                std_param_var = L.get_output(self._l_log_std, obs_var)
                weight_var = L.get_output(self._l_w, obs_var)
            if self.min_std_param is not None:
                std_param_var = tf.maximum(std_param_var, self.min_std_param)
            log_std_var = tf.log(self.sig_rate * std_param_var)
            return dict(mean=mean_var, log_std=log_std_var, weight=weight_var)

    def predict(self, observation):
        return self._dist.approximate_gaussian_sym( self.dist_info_sym(observation) )["mean"]

    @overrides
    def get_action(self, observation):
        flat_obs = self.observation_space.flatten(observation)
        dist_info = dict(zip(self._outputs.keys(), [x[0] for x in self._f_dist([flat_obs])]))
        action = self._dist.sample(dist_info)
        return action, dist_info

    def get_actions(self, observations):
        actions = []
        dists = dict(mean=[], log_std=[], weight=[])
        for obs in observations:
            action, dist_info = get_action(obs)
            actions.append(action)
            for key in dist_info.keys():
                dists[key].append(dist_info[key])
        return actions, dists

    def log_diagnostics(self, paths):
        log_stds = [e.trim_zeros() for e in paths["agent_infos"]['log_std']]
        means = paths["agent_infos"]['mean']
        weights = paths["agent_infos"]['weight']
        means = np.concatenate([means[i][:len(log_stds[i])] for i in len(log_stds)], axis=0)
        weights = np.concatenate([weights[i][:len(log_stds[i])] for i in len(log_stds)], axis=0)
        log_stds = np.concatenate(log_stds, axis=0)
        dist_info = self._dist.approximate_gaussian(dict(mean=means, log_std=log_stds, weight=weights))
        log_stds = dist_info["log_std"]
        logger.record_tabular("{}/AveragePolicyStd".format(self.name),
                              list(np.mean(np.exp(log_stds), axis=0)))

    @property
    def distribution(self):
        return self._dist
