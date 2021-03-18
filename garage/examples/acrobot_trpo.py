from garage.experiment import LocalRunner, run_experiment
from garage.tf.algos import TRPO
from garage.envs import AcrobotEnv, normalize
from garage.tf.envs import TfEnv
from garage.tf.policies import GaussianMLPPolicy
from garage.baselines import LinearFeatureBaseline
import joblib
import tensorflow as tf

seeds = [753668, 548206, 422113, 412189, 391972, 455067, 123759, 646556, 217726, 671394]

exp_id = 1
seed = seeds[exp_id - 1]

init_filepath = None
#init_filepath = f"/home/tbla2725/garage/data/local/experiment/networks/trained_solver_acrobot{exp_id}.pkl"

def run_task(*_):
    with LocalRunner() as runner:
        env = TfEnv(normalize(AcrobotEnv(seed=seed), normalize_obs=True))
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
            hidden_nonlinearity=tf.nn.tanh,
            init_std=0.3
            )
            baseline = LinearFeatureBaseline(env_spec=env.spec)
        algo = TRPO(
            env=env,
            policy=policy,
            baseline=baseline,
            max_path_length=env.horizon,
            discount=0.999,
            max_kl_step=0.01,
            plot=False,
        )
        runner.setup(algo, env)
        runner.train(n_epochs=int(1e3+10), batch_size=10 * env.horizon)


run_experiment(run_task, snapshot_mode="gap", snapshot_gap=100, seed=seed)
