import numpy as np
import torch

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
    env_name = "MountainCarSparse-v0"
    env_orig = GymEnv(env_name)
    o_shape = env_orig.observation_space.shape
    env_bounds = bounds[env_name]
    if env_bounds[0] is not None and isinstance(env_bounds[0], (int, float)):
        env_bounds = (env_bounds[0] * torch.ones(o_shape),
                      env_bounds[1] * torch.ones(o_shape))
    env = EnvGlue(env_orig, env_bounds[0], env_bounds[1])
    agent_helper = AgentHelper(env)
    n_iter = 10000
    rrt_expand_dis = 0.1
    rrt_goal_sample_rate = 0.05  # Set to 0.0 to disable goal biasing

    plan_lengths = []
    for _ in range(1):
        # Define model for local policy
        m = 300  # Number of random Fourier features
        d, d_out = 2 * agent_helper.d_s, agent_helper.d_a
        model = FeatureWrap(BLR(m, alpha=0.1, beta=1., d_out=d_out),
                            RFF(m, d, 0.3))  # 0.3 is the RBF lengthscale

        # Local policy
        local_policy = LocalPolicy(agent_helper, model)
        # local_policy = RandomPolicy(agent_helper)

        # Run RRT
        r3l = R3L(agent_helper, env, local_policy, rrt_expand_dis,
                  rrt_goal_sample_rate, render_env=False)
        r3l_result = r3l.run(goal_sample_fn[env_name],
                             goal_check_fn[env_name], n_iter, verbose=2)

        # Visualise
        r3l.draw_tree(r3l_result[0])
        r3l.replay_trajectory(r3l_result[0], r3l_result[1])

        plan_lengths.append(r3l_result[3])

    print("Mean:", np.mean(plan_lengths), "\t\tStd:", np.std(plan_lengths))
    env_orig.env.close()


if __name__ == "__main__":
    main()
