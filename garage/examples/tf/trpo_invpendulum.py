#!/usr/bin/env python3
"""
This is an example to train a task with VPG algorithm.

Here it runs CartPole-v1 environment with 100 iterations.

Results:
    AverageReturn: 100
    RiseTime: itr 13
"""
import gym
from garage.envs import normalize
from garage.baselines import LinearFeatureBaseline
from garage.experiment import LocalRunner, run_experiment, deterministic
from garage.tf.algos import TRPO
from garage.tf.envs import TfEnv
from garage.tf.policies import GaussianMLPPolicy

exp_id = 10
seed = exp_id
gpu_id = exp_id % 4


def run_task(*_):
    with LocalRunner(gpu_id=gpu_id) as runner:
        deterministic.set_seed(seed)
        env = TfEnv(normalize(gym.make('InvertedPendulum-v2')))

        policy = GaussianMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=(32, 32),
            init_std=0.05)
        baseline = LinearFeatureBaseline(env_spec=env.spec)

        algo = TRPO(
            env=env,
            policy=policy,
            baseline=baseline,
            max_path_length=100,
            discount=0.99,
            max_kl_step=0.01)

        runner.setup(algo, env)
        runner.train(n_epochs=100, batch_size=4000)


run_experiment(
    run_task,
    snapshot_mode="last",
    seed=seed,
    n_parallel=1
)

