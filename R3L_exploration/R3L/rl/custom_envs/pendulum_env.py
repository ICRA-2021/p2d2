"""
A version of OpenAI Gym's continuous Pendulum environment that isn't
such a gigantic mess. Rewards are sparse.
"""

from gym import spaces, Env
from gym.utils import seeding
import numpy as np

from R3L.utils import angle_modulus as th_mod


class PendulumEnv(Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    max_vel = 8.
    max_torque = 2.
    dt = .05
    grav = 10.
    mass = 1.
    length = 1.

    def __init__(self, seed=None):
        high = np.array([np.pi, PendulumEnv.max_vel])
        low = -high
        self.observation_space = spaces.Box(low=low, high=high,
            dtype=np.float32)
        self.action_space = spaces.Box(low=-1., high=1., shape=(1,),
            dtype=np.float32)
        self.viewer = None
        self.seed(seed)
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        th, thdot = self.state
        action = np.clip(action, self.action_space.low, self.action_space.high)[0]
        torque = PendulumEnv.max_torque * action
        self.last_t = torque # for rendering

        g = PendulumEnv.grav
        m = PendulumEnv.mass
        l = PendulumEnv.length
        dt = PendulumEnv.dt

        newthdot = thdot + \
            (-3*g/(2*l) * np.sin(th + np.pi) + 3./(m*l**2)*torque) * dt
        newthdot = np.clip(newthdot, -PendulumEnv.max_vel, PendulumEnv.max_vel)
        newth = th_mod(th + newthdot*dt)

        cos =  np.cos(newth)
        if cos > 0.99:
            reward = cos
        else:
            reward = -1.

        self.state = np.array([newth, newthdot]).astype(np.float32)
        return self.state, reward, False, {}

    def reset(self):
        low = np.array([np.pi - 1., -1.])
        high = np.array([np.pi + 1., 1.])
        self.state = self.np_random.uniform(low=low, high=high)
        self.last_t = None
        return self.state

    @property
    def horizon(self):
        return 100

    def render(self, mode='human'):
        if self.viewer is None:
            import gym.envs.classic_control as cc
            from gym.envs.classic_control import rendering
            from os import path
            self.viewer = rendering.Viewer(500,500)
            self.viewer.set_bounds(-2.2,2.2,-2.2,2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0,0,0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(cc.__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi/2)
        if self.last_t:
            self.imgtrans.scale = (-self.last_t/2, np.abs(self.last_t)/2)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None