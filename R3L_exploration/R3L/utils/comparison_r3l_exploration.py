import numpy as np
import torch
from tqdm import tqdm

from R3L.R3L import R3L
from R3L.models.BLR import BLR
from R3L.models.Model import FeatureWrap
from R3L.models.RandomFourierFeatures import RFF
from R3L.policies import LocalPolicy, RandomPolicy
from R3L.rl.AgentHelper import AgentHelper
from R3L.rl.EnvGlue import EnvGlue
from R3L.rl.Envs import GymEnv

# Functions to sample a goal for Gym environments
goal_sample_fn = {  # Samples from the original state space
    "PendulumSparse-v0": lambda: ([0], [0.0]),
    "MountainCarSparse-v0": lambda: ([0], [0.47]),
    "AcrobotContinuous-v0": lambda: ([0, 1], [np.pi, 0.0]),
    "CartpoleSwingup-v0": lambda: ([0, 2], [0., 0.]),
    "ReacherSparse-v0": lambda: ([2, 3, 6], [0., 0., 0.]),
}

# Functions to check whether task goal is reached for Gym environments
goal_check_fn = {  # These goals are done in the original state space
    "PendulumSparse-v0": lambda s: s[0, 0].abs() < 0.1,
    "MountainCarSparse-v0": lambda s: s[0, 0] > 0.45,
    "AcrobotContinuous-v0":
        lambda s: -np.cos(s[0, 0]) - np.cos(s[0, 1] + s[0, 0]) > 1.9,
    "CartpoleSwingup-v0":
        lambda s: np.cos(s[0, 2]) > 0.99 and s[0, 3].abs() < 1.,
    "ReacherSparse-v0":
        lambda s: np.linalg.norm(s[0, 6:8]) < 0.01 and s[0, 2].abs() < .5 and s[
            0, 3].abs() < .5,
}

# State lower and upper bounds for Gym environments
bounds = {  # None means using original bounds provided by Gym
    "PendulumSparse-v0": (None, None),
    "MountainCarSparse-v0": (None, None),
    "AcrobotContinuous-v0": (None, None),
    "CartpoleSwingup-v0": (None, None),
    "ReacherSparse-v0": (None, None),
}


def main():
    """
    This script is used to compare R3L exploration with and without learned
    policy, and with/without goal biasing. This was used to get results in
    Table 1.
    """
    env_name = "ReacherSparse-v0"
    env_orig = GymEnv(env_name)
    o_shape = env_orig.observation_space.shape
    env_bounds = bounds[env_name]
    if env_bounds[0] is not None and isinstance(env_bounds[0], (int, float)):
        env_bounds = (env_bounds[0] * torch.ones(o_shape),
                      env_bounds[1] * torch.ones(o_shape))
    env = EnvGlue(env_orig, env_bounds[0], env_bounds[1])
    agent_helper = AgentHelper(env)
    n_iter = 50000
    rrt_expand_dis = 0.1
    rrt_goal_sample_rates = [0.0, 0.05]

    # Define model for local policy
    m = 300
    d, d_out = 2 * agent_helper.d_s, agent_helper.d_a
    model = FeatureWrap(BLR(m, alpha=0.1, beta=1., d_out=d_out),
                        RFF(m, d, 0.3))
    local_policies = [LocalPolicy(agent_helper, model),
                      RandomPolicy(agent_helper)]

    n_runs = 20
    lengths = np.ones((n_runs, len(rrt_goal_sample_rates), len(local_policies)))
    itrs = np.ones((n_runs, len(rrt_goal_sample_rates), len(local_policies)))
    for i_run in tqdm(range(n_runs)):
        for i_rate, rrt_goal_sample_rate in enumerate(rrt_goal_sample_rates):
            for i_policy, local_policy in enumerate(local_policies):
                model.reset()
                env.reset()

                # Run RRT
                r3l = R3L(agent_helper, env, local_policy, rrt_expand_dis,
                          rrt_goal_sample_rate, render_env=False)
                r3l_result = r3l.run(goal_sample_fn[env_name],
                                     goal_check_fn[env_name], n_iter, verbose=0)
                lengths[i_run, i_rate, i_policy] = len(r3l_result[0])
                itrs[i_run, i_rate, i_policy] = r3l_result[3]

    np.savez("../results/comp_r3l_exploration_{}".format(env_name),
             lengths, itrs)

    print("mean lengths")
    print(np.mean(lengths, axis=0))
    print("std lengths")
    print(np.std(lengths, axis=0))
    print("mean itrs")
    print(np.mean(itrs, axis=0))
    print("std itrs")
    print(np.std(itrs, axis=0))

    env_orig.env.close()


if __name__ == "__main__":
    main()
