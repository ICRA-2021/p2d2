from garage.experiment import LocalRunner, run_experiment
from garage.tf.algos import TRPO
from garage.envs import FetchReachEnv, normalize
from garage.tf.envs import TfEnv
from garage.tf.policies import GaussianMLPPolicy
from garage.baselines import LinearFeatureBaseline
import joblib
import tensorflow as tf

seeds = [605427, 682773, 522064, 752367, 148214, 884873, 389805, 210238, 433716, 727977]

exp_id = 1
gpu_id = 0
seed = seeds[exp_id - 1]

init_filepath = None
#init_filepath = "/home/tbla2725/garage/data/local/experiment/networks/trained_rrt_fetch%d.pkl" %exp_id


def run_task(*_):
    with LocalRunner(gpu_id=gpu_id) as runner:
        env = TfEnv(normalize(FetchReachEnv(seed=seed), normalize_obs=True))
        if init_filepath:
            print("loading model from %s" % init_filepath)
            data = joblib.load(init_filepath)
            policy = data['policy']
            baseline = data['baseline']
        else:
            policy = GaussianMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=(128, 128, 128, 128),
            hidden_nonlinearity=tf.nn.relu,
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
        runner.train(n_epochs=int(5e3+10), batch_size=40 * env.horizon)


run_experiment(run_task, snapshot_mode="gap", snapshot_gap=100, seed=seed)
