import numpy as np
import matplotlib.pylab as plt
import torch
from tqdm import tqdm

from R3L.rl.EnvGlue import EnvGlue
from R3L.rl.Envs import GymEnv


class LinearPolicy:
    def __init__(self, d_s, d_a, bias_term=False, ftmap=None):
        """
        Policy mapping observations to actions linearly.
        :param bias_term: Whether to use a bias term for linear policy
        """
        self.w = None
        self.ftmap = ftmap
        self.bias_term = bias_term
        self.d_in = d_s + (1 if bias_term else 0)
        self.d_out = d_a

    def set_weights(self, w):
        """
        Replaces policy weights
        :param w: policy weights
        """
        self.w = w.reshape(-1, self.d_out).t()

    def pick(self, obs):
        """
        Returns an action given an observation.
        :param obs: Observation
        :return: Action
        """
        if self.ftmap is not None:
            obs = self.ftmap.to_features(obs.view(1, -1)).view(-1)
        if self.bias_term:
            return self.w[:, 1:].mv(obs) + self.w[:, 0]
        else:
            return self.w.mv(obs)


def run_episodes(_env, _policy, _starting_states, _n_steps, render_env=False):
    all_observations, all_actions, all_rewards = [], [], []
    for starting_state in _starting_states:
        o = _env.reset()
        _env.set_state(starting_state.view(-1))
        all_observations.append([o])
        all_actions.append([])
        all_rewards.append([])
        for i_step in range(_n_steps):
            a = _policy.pick(o.reshape(-1)).reshape(-1)
            o_new, r, done, info = _env.step(a)
            all_observations[-1].append([o_new])
            all_actions[-1].append([a])
            all_rewards[-1].append([r])
            if render_env:
                _env.render()
            if done:
                break
            o = o_new
    return all_observations, all_actions, all_rewards

"""
This script is used to generate the cost surface of MountainCar when using 
a linear (2 parameters), running a few RL episodes for a grid of values of 
policy parameters.
"""

env_name = "MountainCarSparse-v0"
n_steps = 200
n_eps_avg = 5
env_orig = GymEnv(env_name)
env = EnvGlue(env_orig, normalised=False)
policy = LinearPolicy(env.d_s, env.d_a)
starting_states = [env.reset() for _ in range(n_eps_avg)]

# Construct cost surface
bound_low, bound_high, res = (-17, -120), (15, 320), 64
x, y = np.meshgrid(np.linspace(bound_low[0], bound_high[0], res),
                   np.linspace(bound_low[1], bound_high[1], res))
xy = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))
rets = np.zeros((xy.shape[0], 1))
for i, weights in enumerate(tqdm(xy)):
    policy.set_weights(torch.tensor(weights, dtype=torch.float32))
    rr = run_episodes(env, policy, starting_states, n_steps, render_env=False)[
        2]
    sm = torch.tensor([np.sum(r) for r in rr]).sum()
    rets[i] = sm.detach().numpy() / float(n_eps_avg)

print(rets.T)
np.savez("../results/plotting_chain_and_loss/loss_surface2.npz",
         x=x, y=y, z=rets.reshape(x.shape))

plt.figure(dpi=300)
cs = plt.contourf(x, y, rets.reshape(x.shape), 64)
plt.colorbar(cs)
plt.xlabel("policy parameter 0")
plt.ylabel("policy parameter 1")
plt.tight_layout(rect=(0, 0.01, 1, 1))
plt.savefig("toto2.png")
plt.show()
