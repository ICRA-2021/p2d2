import os
import numpy as np
import torch
import argparse

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


def parse_path(_args):
    filepath = _args.filepath
    filename = "trajectories.npz"
    if filepath is None:
        file_dir = os.path.join("R3L", "data", args.env)
        try:
            os.mkdir(file_dir)
        except FileExistsError:
            pass
        return os.path.join(file_dir, filename)
    else:
        return filepath


def main(args):
    demo_file_path = parse_path(args)
    env_name = args.env
    env_orig = GymEnv(env_name)
    env_bounds = bounds[env_name]
    o_shape = env_orig.observation_space.shape
    if env_bounds[0] is not None and isinstance(env_bounds[0], (int, float)):
        env_bounds = (env_bounds[0] * torch.ones(o_shape),
                      env_bounds[1] * torch.ones(o_shape))
    env = EnvGlue(env_orig, env_bounds[0], env_bounds[1])
    agent_helper = AgentHelper(env)
    n_iter = args.n_iter

    rrt_expand_dis = 0.1
    rrt_goal_sample_rate = 0.05

    plan_lengths = []
    obs_list = []
    act_list = []

    n_success = 0
    while n_success < args.n_traj:
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
                  rrt_goal_sample_rate, render_env=False)
        r3l_result = r3l.run(goal_sample_fn[env_name],
                             goal_check_fn[env_name], n_iter, verbose=1)
        if args.skip_unsucessful and not r3l_result[2]:
            continue
        n_success += 1
        print("## {}/{} trajectories generated".format(n_success, args.n_traj))

        obs_list.append(np.array(r3l_result[0])[:-1])
        act_list.append(np.array(r3l_result[1]))
        # r3l.replay_trajectory(r3l_result[0], r3l_result[1])
        plan_lengths.append(r3l_result[3])

    # save demo
    obs = np.concatenate(obs_list, axis=0)
    act = np.concatenate(act_list, axis=0)
    np.savez_compressed(demo_file_path, obs=obs, act=act)

    print("Mean:", np.mean(plan_lengths), "\t\tStd:", np.std(plan_lengths))
    env_orig.env.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', help="id of the environment to work on",
                        default="MountainCarSparse-v0", type=str)
    parser.add_argument('--n_traj', help="number of trajectories", default=20,
                        type=int)
    parser.add_argument('--n_iter', help="number of RRT iterations",
                        default=20000, type=int)
    parser.add_argument('--filepath',
                        help="location to store recorded trajectories",
                        default=None, type=str)
    parser.add_argument('--skip_unsucessful',
                        help="Whether to only keep successful trajectories",
                        default=True, type=bool)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
