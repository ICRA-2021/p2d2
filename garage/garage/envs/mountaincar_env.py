"""
A version of OpenAI Gym's continuous MountainCar environment that isn't
such a gigantic mess. Rewards are sparse.
"""

from gym import spaces, Env
from gym.utils import seeding
import numpy as np

class MountainCarEnv(Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    power = 0.0015
    min_vel = -0.07
    max_vel = 0.07
    min_pos = -1.2
    max_pos = 0.6
    goal_pos = 0.45

    def __init__(self, seed=None):
        high = np.array([MountainCarEnv.max_pos, MountainCarEnv.max_vel])
        low = np.array([MountainCarEnv.min_pos, MountainCarEnv.min_vel])
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
        position, velocity = self.state
        action = np.clip(action, self.action_space.low, self.action_space.high)[0]
        engine_force = action * MountainCarEnv.power
        gravity_force = 0.0025 * np.cos(3*position)
        velocity += engine_force - gravity_force
        velocity = np.clip(velocity, MountainCarEnv.min_vel,
            MountainCarEnv.max_vel)
        position += velocity
        position = np.clip(position, MountainCarEnv.min_pos,
            MountainCarEnv.max_pos)
        if (position==MountainCarEnv.min_pos and velocity<0): velocity = 0.

        done = bool(position >= MountainCarEnv.goal_pos)
        reward = -1. if not done else 0.

        self.state = np.array([position, velocity]).astype(np.float32)
        return self.state, reward, done, {}

    def reset(self):
        self.state = np.array([self.np_random.uniform(low=-0.6, high=-0.4), 0.]
            ).astype(np.float32)
        return self.state

    @property
    def horizon(self):
        return 200

    def _height(self, xs):
        return np.sin(3 * xs)*.45+.55

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = MountainCarEnv.max_pos - MountainCarEnv.min_pos
        scale = screen_width/world_width
        carwidth=40
        carheight=20


        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            xs = np.linspace(MountainCarEnv.min_pos, MountainCarEnv.max_pos,
                100)
            ys = self._height(xs)
            xys = list(zip((xs-MountainCarEnv.min_pos)*scale, ys*scale))

            self.track = rendering.make_polyline(xys)
            self.track.set_linewidth(4)
            self.viewer.add_geom(self.track)

            clearance = 10

            l,r,t,b = -carwidth/2, carwidth/2, carheight, 0
            car = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            car.add_attr(rendering.Transform(translation=(0, clearance)))
            self.cartrans = rendering.Transform()
            car.add_attr(self.cartrans)
            self.viewer.add_geom(car)
            frontwheel = rendering.make_circle(carheight/2.5)
            frontwheel.set_color(.5, .5, .5)
            frontwheel.add_attr(
                rendering.Transform(translation=(carwidth/4,clearance)))
            frontwheel.add_attr(self.cartrans)
            self.viewer.add_geom(frontwheel)
            backwheel = rendering.make_circle(carheight/2.5)
            backwheel.add_attr(
                rendering.Transform(translation=(-carwidth/4,clearance)))
            backwheel.add_attr(self.cartrans)
            backwheel.set_color(.5, .5, .5)
            self.viewer.add_geom(backwheel)
            flagx = (MountainCarEnv.goal_pos-MountainCarEnv.min_pos)*scale
            flagy1 = self._height(MountainCarEnv.goal_pos)*scale
            flagy2 = flagy1 + 50
            flagpole = rendering.Line((flagx, flagy1), (flagx, flagy2))
            self.viewer.add_geom(flagpole)
            flag = rendering.FilledPolygon(
                [(flagx, flagy2), (flagx, flagy2-10), (flagx+25, flagy2-5)])
            flag.set_color(.8,.8,0)
            self.viewer.add_geom(flag)

        pos = self.state[0]
        self.cartrans.set_translation(
            (pos-MountainCarEnv.min_pos)*scale, self._height(pos)*scale)
        self.cartrans.set_rotation(np.cos(3 * pos))

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
