"""
A version of OpenAI Gym's Fetch Reaching problem that is less of a mess.
Controls are 3-d delta in gripper position. Rewards are sparse.

State/observation space is:

3d Cartesian gripper position
2d gripper finger position
3d Cartesian gripper velocity
2d gripper finger velocity
3d Cartesian goal position
"""


import os
import numpy as np
from gym import utils, spaces
from gym.envs.robotics import fetch_env


MODEL_XML_PATH = os.path.join('fetch', 'reach.xml')


class FetchReachEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse', seed=None):
        initial_qpos = {
            'robot0:slide0': 0.4049,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
        }
        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=False, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)

        high = np.array([2., 2., 1.1,
                         1e-4, 1e-4,
                         1., 1., 1.,
                         0.1, 0.1,
                         1.5, 0.9, 0.69])
        low = np.array([0.25, -1., 0.,
                        -1e-4, -1e-4,
                        -1., -1., -1.,
                        -0.1, -0.1,
                        1.19, 0.59, 0.38])
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.seed(seed)

    #Necessary Gym functions

    def step(self, action):
        if action.shape[0] == 1:
            action = action.reshape(-1)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        self.sim.step()
        self._step_callback()
        obs = self.observe()
        done = self._is_success(obs[:self.goal.size], self.goal)
        reward = -1. + float(done)
        return obs, reward, done, {"success": done}

    def reset(self):
        obs = super(FetchReachEnv, self).reset()
        offset = self.np_random.uniform(0.1, 0.15, size=3) *\
                 np.random.choice([-1,1], size=3)
        self.goal = self.initial_gripper_xpos[:3] + offset
        return self.observe()


    def observe(self):
        obs = self._get_obs()
        return np.concatenate([obs['observation'], self.goal])

    @property
    def horizon(self):
        return 50

    #Optional helper functions
    def get_state(self):
        obs = self.observe()
        state = self.sim.get_state()
        return np.concatenate([state.qpos, state.qvel, obs])
