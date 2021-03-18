from garage.tf.baselines import GaussianMLPBaseline
from garage.experiment import LocalRunner, run_experiment
from garage.tf.algos import TRPO
from garage.envs import PendulumEnv, normalize
from garage.tf.envs import TfEnv
from garage.tf.policies import GaussianMLPPolicy
from garage.baselines import LinearFeatureBaseline
from garage.tf.core.network import ConvMergeNetwork
import garage.tf.core.layers as L
import garage.tf.core.nonlinearities as NL
import numpy as np
import tensorflow as tf
import joblib

seeds = [661643, 14495, 118501, 970775, 780238, 977055, 647323, 328208, 960850, 899519]

exp_id = 1
seed = seeds[exp_id - 1]

init_filepath = None
#init_filepath = f"/home/tbla2725/garage/data/local/experiment/networks/trained_sst_pendulum{exp_id}.pkl"


def run_task(*_):
    with LocalRunner() as runner:
        env = TfEnv(normalize(PendulumEnv(seed=seed), normalize_obs=True))
        if init_filepath:
            print("loading model from %s" % init_filepath)
            data = joblib.load(init_filepath)
            policy = data['policy']
            baseline = data['baseline']
        else:
            policy = GaussianMLPPolicy(
            env_spec=env.spec,
            # The neural network policy should have two hidden layers, each with 32 hidden units.
            hidden_sizes=(32, 32),
            init_std=0.3
            )
            baseline = LinearFeatureBaseline(env_spec=env.spec)
        algo = TRPO(
            env=env,
            policy=policy,
            baseline=baseline,
            max_path_length=env.horizon,
            discount=0.99,
            max_kl_step=0.01,
            plot=False,
        )
        runner.setup(algo, env)
        runner.train(n_epochs=int(1e3+10), batch_size=40 * env.horizon)


run_experiment(run_task, snapshot_mode="gap", snapshot_gap=100, seed=seed)
