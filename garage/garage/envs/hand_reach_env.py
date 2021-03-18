"""
A version of OpenAI Gym's Hand Reaching problem that is less of a mess.
Controls are 20-d absolute angle of hand joints. Rewards are sparse

State/observation space is 78-d:

24d joint angles
24d joint angular velocities
15d cartesian coordinates of fingertips (5x3)
15d cartesian coordinates of targets (5x3)
"""


import os
import numpy as np
from gym import utils, spaces
from gym.envs.robotics.hand.reach import HandReachEnv as SrcEnv
from gym.envs.robotics.hand.reach import FINGERTIP_SITE_NAMES
from gym.envs.robotics.utils import robot_get_obs


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm((goal_a - goal_b).reshape(5,3), axis=-1).max()

class HandReachEnv(SrcEnv):
    def __init__(self, distance_threshold=0.02, n_substeps=20,
        relative_control=False, reward_type='sparse', seed=None
    ):
        SrcEnv.__init__(
            self, distance_threshold=distance_threshold, n_substeps=n_substeps,
            relative_control=relative_control, reward_type=reward_type)

        high = np.array([
            0.14 , 0.489,
            0.349, 1.571, 1.571, 1.571,
            0.349, 1.571, 1.571, 1.571,
            0.349, 1.571, 1.571, 1.571,
            0.785, 0.349, 1.571, 1.571, 1.571,
            1.047, 1.222, 0.209, 0.524, 0.,
            10., 10.,
            20., 20., 20., 20.,
            20., 20., 20., 20.,
            20., 20., 20., 20.,
            20., 20., 20., 20., 20.,
            20., 20., 20., 20., 20.,
            1.12482863, 0.97250619, 0.22791091,
            1.12809776, 0.97401904, 0.22692807,
            1.1464395 , 0.97503628, 0.22856588,
            1.17281652, 0.97677939, 0.22697715,
            1.12582987, 0.97694104, 0.22869581,
            1.12482863, 0.97250619, 0.22791091,
            1.12809776, 0.97401904, 0.22692807,
            1.1464395 , 0.97503628, 0.22856588,
            1.17281652, 0.97677939, 0.22697715,
            1.12582987, 0.97694104, 0.22869581,
        ])
        low = np.array([
            -0.489, -0.698,
            -0.349, 0., 0., 0.,
            -0.349, 0., 0., 0.,
            -0.349, 0., 0., 0., 
            0., -0.349, 0., 0., 0.,
            -1.047, 0., -0.209, -0.524, -1.571,
            -10., -10.,
            -20., -20., -20., -20.,
            -20., -20., -20., -20.,
            -20., -20., -20., -20.,
            -20., -20., -20., -20., -20.,
            -20., -20., -20., -20., -20.,
            0.88900791, 0.7177926 , 0.13700981,
            0.88836738, 0.72156632, 0.13666836,
            0.89055303, 0.72673447, 0.13971812,
            0.89206152, 0.740787  , 0.14786129,
            0.85544797, 0.75626273, 0.14554927,
            0.88900791, 0.7177926 , 0.13700981,
            0.88836738, 0.72156632, 0.13666836,
            0.89055303, 0.72673447, 0.13971812,
            0.89206152, 0.740787  , 0.14786129,
            0.85544797, 0.75626273, 0.14554927,
        ])
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.goal_size = self.goal.size
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
        done = self._is_success(obs[-2 * self.goal_size: -self.goal_size],
                                self.goal)
        if self.reward_type == 'dense':
            reward = -goal_distance(obs[-2 * self.goal.size: -self.goal.size],
                                   self.goal)
        else:
            reward = -1. + float(done)
        return obs, reward, done, {"success": done}

    def reset(self):
        obs = super(SrcEnv, self).reset()
        return self.observe()

    def observe(self):
        obs = self._get_obs()
        return np.concatenate([obs['observation'], obs['desired_goal']])

    def _sample_goal(self):
        thumb_name = 'robot0:S_thtip'
        finger_names = [name for name in FINGERTIP_SITE_NAMES if name != thumb_name]
        finger_name = self.np_random.choice(finger_names)

        thumb_idx = FINGERTIP_SITE_NAMES.index(thumb_name)
        finger_idx = FINGERTIP_SITE_NAMES.index(finger_name)
        assert thumb_idx != finger_idx

        # Pick a meeting point above the hand.
        meeting_pos = self.palm_xpos + np.array([0.0, -0.09, 0.05])
        meeting_pos += self.np_random.normal(scale=0.005, size=meeting_pos.shape)

        # Slightly move meeting goal towards the respective finger to avoid that they
        # overlap.
        goal = self.initial_goal.copy().reshape(-1, 3)
        for idx in [thumb_idx, finger_idx]:
            offset_direction = (meeting_pos - goal[idx])
            offset_direction /= np.linalg.norm(offset_direction)
            goal[idx] = meeting_pos - 0.005 * offset_direction

        return goal.flatten()

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    @property
    def horizon(self):
        return 50
