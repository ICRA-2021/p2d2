import numpy as np
import torch
import matplotlib.pylab as plt

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

img_id = 0


def draw_tree_frame(r3l, path=None):
    global img_id
    r3l.rrt.draw_graph_2d()
    if path is not None:
        plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')
    plt.grid(True)
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xlabel("position")
    plt.ylabel("velocity")
    plt.tight_layout()
    plt.savefig("/tmp/video/frame{:05}.png".format(img_id), dpi=300)
    plt.close()
    img_id += 1


def main():
    """
    This script is used to generate a video (frames{xxx}.png) of the tree
    built by R3L during exploration.
    """
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
    rrt_goal_sample_rate = 0.05
    _goal_check_fn = goal_check_fn[env_name]

    # Define model for local policy
    m = 300
    d, d_out = 2 * agent_helper.d_s, agent_helper.d_a
    model = FeatureWrap(BLR(m, alpha=0.1, beta=1., d_out=d_out),
                        RFF(m, d, 0.3))

    # Local policy
    local_policy = LocalPolicy(agent_helper, model)
    # local_policy = RandomPolicy(agent_helper)

    # Run RRT
    r3l = R3L(agent_helper, env, local_policy, rrt_expand_dis,
              rrt_goal_sample_rate, render_env=True)

    def abused_goal_check_fn(s):
        draw_tree_frame(r3l)
        return _goal_check_fn(s)

    r3l_result = r3l.run(goal_sample_fn[env_name],
                         abused_goal_check_fn, n_iter, verbose=2)

    draw_tree_frame(r3l, r3l_result[0])

    env_orig.env.close()


if __name__ == "__main__":
    main()
