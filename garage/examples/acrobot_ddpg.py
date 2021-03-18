import joblib
import tensorflow as tf

from garage.experiment import LocalRunner, run_experiment
from garage.exploration_strategies import OUStrategy, GaussianStrategy
from garage.replay_buffer import SimpleReplayBuffer
from garage.tf.algos import DDPG
from garage.tf.envs import TfEnv
from garage.envs import AcrobotEnv, normalize
from garage.tf.policies import DeterministicMLPPolicy
from garage.tf.q_functions import ContinuousMLPQFunction


seeds = [229453, 40706, 980035, 447630, 347531, 101121, 696140, 306079, 378436, 371290]

exp_id = 1
gpu_id = 0
seed = seeds[exp_id - 1]

init_filepath = None
#init_filepath = "/home/tbla2725/garage/data/local/experiment/networks/trained_ddpg_acrobot%d.pkl" %exp_id


def run_task(*_):
    with LocalRunner(gpu_id=gpu_id) as runner:
        env = TfEnv(normalize(AcrobotEnv(seed=seed), normalize_obs=True))
        if init_filepath:
            print("loading model from %s" % init_filepath)
            data = joblib.load(init_filepath)
            policy = data['policy']
            qf = data['qf']
        else:
            policy = DeterministicMLPPolicy(
                env_spec=env.spec,
                hidden_sizes=[32, 32],
                hidden_nonlinearity=tf.nn.tanh,
                output_nonlinearity=tf.nn.tanh
                )
            qf = ContinuousMLPQFunction(
                env_spec=env.spec,
                hidden_sizes=[32, 32],
                hidden_nonlinearity=tf.nn.relu
                )

        action_noise = GaussianStrategy(env.spec, max_sigma=0.3, min_sigma=0.05, decay_period=2.5e6)
        replay_buffer = SimpleReplayBuffer(
            env_spec=env.spec, size_in_transitions=int(1e6), time_horizon=env.horizon)
        n_epoch_cycles = 10
        ddpg = DDPG(
            env,
            policy=policy,
            policy_lr=1e-5,
            qf=qf,
            qf_lr=1e-3,
            max_path_length=env.horizon,
            replay_buffer=replay_buffer,
            n_epoch_cycles=n_epoch_cycles,
            buffer_batch_size=256,
            smooth_return=False,
            target_update_tau=1e-3,
            n_train_steps=50,
            discount=0.999,
            min_buffer_size=int(1e4),
            exploration_strategy=action_noise,
            policy_optimizer=tf.train.AdamOptimizer,
            qf_optimizer=tf.train.AdamOptimizer)

        runner.setup(algo=ddpg, env=env)

        runner.train(n_epochs=int(1e3+10), n_epoch_cycles=n_epoch_cycles, batch_size=env.horizon)


run_experiment(run_task, snapshot_mode="gap", snapshot_gap=100, seed=seed)
