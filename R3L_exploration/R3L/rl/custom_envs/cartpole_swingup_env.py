"""
A swingup version of OpenAI Gym's cartpole balancing problem.
Controls are continuous. Rewards are sparse.
"""

from gym import spaces, logger, Env
from gym.utils import seeding
import numpy as np
from R3L.utils import angle_modulus as th_mod

class CartpoleSwingupEnv(Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    gravity = 9.8
    masscart = 1.
    masspole = 0.1
    total_mass = masspole + masscart
    length = 0.5 # actually half the pole's length
    polemass_length = masspole * length
    force_mag = 10.
    dt = 0.01  # seconds between state updates
    #enough timesteps to simulate 5 seconds
    time_horizon = int(5 / dt)
    x_threshold = 3.  # limits of the track
    MAX_LIN_VEL = 20.
    MAX_ANG_VEL = 20.

    def __init__(self, seed=None):
        high = np.array([
            CartpoleSwingupEnv.x_threshold * 2,
            CartpoleSwingupEnv.MAX_LIN_VEL,
            np.pi,
            CartpoleSwingupEnv.MAX_ANG_VEL])

        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.action_space = spaces.Box(low=-1., high=1., shape=(1,),
            dtype=np.float32)
        self.seed(seed)
        self.viewer = None
        self.state = None
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        high = np.array([1., 2., np.pi+1, 3.])
        low = np.array([-1., -2., np.pi-1, -3.])
        self.state = self.np_random.uniform(low, high).astype(np.float32)
        return self.state

    def step(self, action):
        x, x_dot, theta, theta_dot = self.state
        action = np.clip(action, self.action_space.low, self.action_space.high)[0]
        force = CartpoleSwingupEnv.force_mag * action
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        temp = (force + \
            CartpoleSwingupEnv.polemass_length * (theta_dot**2) * sintheta) \
            / CartpoleSwingupEnv.total_mass
        thetaacc = (CartpoleSwingupEnv.gravity * sintheta - costheta * temp) \
            / (CartpoleSwingupEnv.length \
                * (4.0/3.0 - CartpoleSwingupEnv.masspole * (costheta**2) \
                    / CartpoleSwingupEnv.total_mass))
        xacc  = temp \
            - CartpoleSwingupEnv.polemass_length * thetaacc * costheta \
            / CartpoleSwingupEnv.total_mass
        x  = x + CartpoleSwingupEnv.dt * x_dot
        x_dot = x_dot + CartpoleSwingupEnv.dt * xacc
        theta = theta + CartpoleSwingupEnv.dt * theta_dot
        theta_dot = theta_dot + CartpoleSwingupEnv.dt * thetaacc
        
        theta = th_mod(theta)
        x_dot = np.clip(x_dot, -CartpoleSwingupEnv.MAX_LIN_VEL,
            CartpoleSwingupEnv.MAX_LIN_VEL)
        theta_dot = np.clip(theta_dot, -CartpoleSwingupEnv.MAX_ANG_VEL,
            CartpoleSwingupEnv.MAX_ANG_VEL)
        self.state = np.array([x,x_dot,theta,theta_dot]).astype(np.float32)
        if np.abs(x) > CartpoleSwingupEnv.x_threshold:
            x = np.clip(x, -CartpoleSwingupEnv.x_threshold, CartpoleSwingupEnv.x_threshold)
            x_dot = 0.

        cos = np.cos(theta)
        if cos > 0.9:
            reward = cos
        else:
            reward = -1.

        return self.state, reward, False, {}

    def render(self, mode='human'):
        screen_width = 600
        screen_height = 400

        world_width = CartpoleSwingupEnv.x_threshold*2
        scale = screen_width/world_width
        carty = 100 # TOP OF CART
        polewidth = 10.0
        polelen = scale * (2 * CartpoleSwingupEnv.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            l,r,t,b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
            axleoffset =cartheight/4.0
            cart = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)
            l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
            pole = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            pole.set_color(.8,.6,.4)
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            pole.add_attr(self.poletrans)
            pole.add_attr(self.carttrans)
            self.viewer.add_geom(pole)
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(.5,.5,.8)
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0,carty), (screen_width,carty))
            self.track.set_color(0,0,0)
            self.viewer.add_geom(self.track)

            self._pole_geom = pole

        if self.state is None: return None

        # Edit the pole polygon vertex
        pole = self._pole_geom
        l,r,t,b = -polewidth/2,polewidth/2,polelen-polewidth/2,-polewidth/2
        pole.v = [(l,b), (l,t), (r,t), (r,b)]

        x = self.state
        cartx = x[0]*scale+screen_width/2.0 # MIDDLE OF CART
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    @property
    def horizon(self):
        return CartpoleSwingupEnv.time_horizon
