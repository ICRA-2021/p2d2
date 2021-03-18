"""
A swingup version of OpenAI Gym's MuJoCo Reacher problem.
Controls are continuous. Rewards are sparse.

qpos is (j0 angle, j1 angle, target x, target y)
"""

import numpy as np
from gym import utils, spaces
from gym.envs.mujoco import mujoco_env

from R3L.utils import angle_modulus as th_mod


class ReacherEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 20
    }

    goal_dist = 0.01

    def __init__(self, seed=None):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, 'reacher.xml', 2)
        high = np.array([np.pi, np.pi, 0.2, 0.2, 200., 200., 0.5, 0.5])
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.seed(seed)

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = self.terminal()
        reward = -1. + float(done)
        return ob, reward, done, {"success": done}

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        qpos = self.init_qpos + \
            self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq)
        while True:
            self.goal = self.np_random.uniform(low=-.2, high=.2, size=2)
            if np.linalg.norm(self.goal) < 0.2:
                break
        qpos[-2:] = self.goal
        qvel = self.init_qvel + \
            self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        #(theta_0, theta_1, x_target, y_target, theta_0_dot, theta_1_dot, dist)
        return np.concatenate([
            #bound angles to [-pi,pi]
            th_mod(self.sim.data.qpos.flat[:2]),
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat[:2],
            (self.get_body_com("fingertip") - self.get_body_com("target"))[:2]
        ])

    def terminal(self):
        vec = self.get_body_com("fingertip")-self.get_body_com("target")
        dist = np.linalg.norm(vec)
        return dist < ReacherEnv.goal_dist

    @property
    def horizon(self):
        return 50