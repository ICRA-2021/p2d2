"""
A version of OpenAI Gym's acrobot task that isn't such a gigantic mess.
Controls are continuous. Rewards are sparse
"""
import numpy as np
from gym import Env, spaces
from gym.utils import seeding
from garage.envs.util import angle_modulus as th_mod

class AcrobotEnv(Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 15
    }

    dt = .2 # [s]
    #enough timesteps to simulate 100 seconds
    time_horizon = int(100 / dt)

    LINK_LENGTH_1 = 1.  # [m]
    LINK_LENGTH_2 = 1.  # [m]
    LINK_MASS_1 = 1.  #: [kg] mass of link 1
    LINK_MASS_2 = 1.  #: [kg] mass of link 2
    LINK_COM_POS_1 = 0.5  #: [m] position of the center of mass of link 1
    LINK_COM_POS_2 = 0.5  #: [m] position of the center of mass of link 2
    LINK_MOI = 1.  #: moments of inertia for both links

    MAX_VEL_1 = 4 * np.pi
    MAX_VEL_2 = 9 * np.pi

    def __init__(self, seed=None, goal_y=1.9):
        self.viewer = None
        high = np.array([
            np.pi,
            np.pi,
            AcrobotEnv.MAX_VEL_1,
            AcrobotEnv.MAX_VEL_2])
        low = -high
        self.observation_space = spaces.Box(low=low, high=high,
                                            dtype=np.float32)
        self.action_space = spaces.Box(low=-1., high=1.,
            shape=(1,), dtype=np.float32)
        self.state = None
        self.goal_y = goal_y
        self.seed(seed)
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.state = self.np_random.uniform(low=-0.1, high=0.1, size=(4,))
        return self.state

    def step(self, action):
        s = self.state
        torque = np.clip(action, self.action_space.low, self.action_space.high)[0]
        # augment the state with the action so it can be passed to _dsdt
        s_augmented = np.append(s, torque)

        ns = rk4(self._dsdt, s_augmented, [0, AcrobotEnv.dt])
        # only care about final timestep of integration returned by integrator
        ns = ns[-1]
        ns = ns[:4]  # omit action

        ns[0] = th_mod(ns[0])
        ns[1] = th_mod(ns[1])
        ns[2] = np.clip(ns[2], -AcrobotEnv.MAX_VEL_1, AcrobotEnv.MAX_VEL_1)
        ns[3] = np.clip(ns[3], -AcrobotEnv.MAX_VEL_2, AcrobotEnv.MAX_VEL_2)
        self.state = ns
        #done if y-coord of far end is above goal_y
        done = bool(-np.cos(ns[0]) - np.cos(ns[1] + ns[0]) > self.goal_y)
        reward = -1. if not done else 0.
        return (self.state, reward, done, {})

    def _dsdt(self, s_augmented, t):
        m1 = AcrobotEnv.LINK_MASS_1
        m2 = AcrobotEnv.LINK_MASS_2
        l1 = AcrobotEnv.LINK_LENGTH_1
        lc1 = AcrobotEnv.LINK_COM_POS_1
        lc2 = AcrobotEnv.LINK_COM_POS_2
        I1 = AcrobotEnv.LINK_MOI
        I2 = AcrobotEnv.LINK_MOI
        g = 9.8
        a = s_augmented[-1]
        s = s_augmented[:-1]
        theta1 = s[0]
        theta2 = s[1]
        dtheta1 = s[2]
        dtheta2 = s[3]
        d1 = m1 * lc1 ** 2 + m2 * \
             (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * np.cos(theta2)) + I1 + I2
        d2 = m2 * (lc2 ** 2 + l1 * lc2 * np.cos(theta2)) + I2
        phi2 = m2 * lc2 * g * np.cos(theta1 + theta2 - np.pi / 2.)
        phi1 = - m2 * l1 * lc2 * dtheta2 ** 2 * np.sin(theta2) \
               - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * np.sin(theta2) \
               + (m1 * lc1 + m2 * l1) * g * np.cos(theta1 - np.pi / 2) + phi2
        ddtheta2 = (a + d2 / d1 * phi1 - m2 * l1 * lc2 * dtheta1 ** 2 * np.sin(
                        theta2) - phi2) \
                    / (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
        ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
        return np.array([dtheta1, dtheta2, ddtheta1, ddtheta2, 0.]
            ).astype(np.float32)

    def render(self, mode='human'):
        from gym.envs.classic_control import rendering

        s = self.state

        if self.viewer is None:
            self.viewer = rendering.Viewer(500, 500)
            bound = AcrobotEnv.LINK_LENGTH_1 + AcrobotEnv.LINK_LENGTH_2 + 0.2
            self.viewer.set_bounds(-bound, bound, -bound, bound)

        if s is None: return None

        p1 = [-AcrobotEnv.LINK_LENGTH_1 *
              np.cos(s[0]), AcrobotEnv.LINK_LENGTH_1 * np.sin(s[0])]

        p2 = [p1[0] - AcrobotEnv.LINK_LENGTH_2 * np.cos(s[0] + s[1]),
              p1[1] + AcrobotEnv.LINK_LENGTH_2 * np.sin(s[0] + s[1])]

        xys = np.array([[0, 0], p1, p2])[:, ::-1]
        thetas = [s[0] - np.pi / 2, s[0] + s[1] - np.pi / 2]
        link_lengths = [AcrobotEnv.LINK_LENGTH_1, AcrobotEnv.LINK_LENGTH_2]

        self.viewer.draw_line((-2.2, self.goal_y), (2.2, self.goal_y))
        for ((x, y), th, llen) in zip(xys, thetas, link_lengths):
            l, r, t, b = 0, llen, .1, -.1
            jtransform = rendering.Transform(rotation=th, translation=(x, y))
            link = self.viewer.draw_polygon([(l, b), (l, t), (r, t), (r, b)])
            link.add_attr(jtransform)
            link.set_color(0, .8, .8)
            circ = self.viewer.draw_circle(.1)
            circ.set_color(.8, .8, 0)
            circ.add_attr(jtransform)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

    @property
    def horizon(self):
        return AcrobotEnv.time_horizon


def rk4(derivs, y0, t, *args, **kwargs):
    """
    Integrate 1D or ND system of ODEs using 4-th order Runge-Kutta.
    This is a toy implementation which may be useful if you find
    yourself stranded on a system w/o scipy.  Otherwise use
    :func:`scipy.integrate`.
    *y0*
        initial state vector
    *t*
        sample times
    *derivs*
        returns the derivative of the system and has the
        signature ``dy = derivs(yi, ti)``
    *args*
        additional arguments passed to the derivative function
    *kwargs*
        additional keyword arguments passed to the derivative function
    Example 1 ::
        ## 2D system
        def derivs6(x,t):
            d1 =  x[0] + 2*x[1]
            d2 =  -3*x[0] + 4*x[1]
            return (d1, d2)
        dt = 0.0005
        t = arange(0.0, 2.0, dt)
        y0 = (1,2)
        yout = rk4(derivs6, y0, t)
    Example 2::
        ## 1D system
        alpha = 2
        def derivs(x,t):
            return -alpha*x + exp(-t)
        y0 = 1
        yout = rk4(derivs, y0, t)
    If you have access to scipy, you should probably be using the
    scipy.integrate tools rather than this function.
    """

    try:
        Ny = len(y0)
    except TypeError:
        yout = np.zeros((len(t),), np.float_)
    else:
        yout = np.zeros((len(t), Ny), np.float_)

    yout[0] = y0

    for i in np.arange(len(t) - 1):
        thist = t[i]
        dt = t[i + 1] - thist
        dt2 = dt / 2.0
        y0 = yout[i]

        k1 = np.asarray(derivs(y0, thist, *args, **kwargs))
        k2 = np.asarray(derivs(y0 + dt2 * k1, thist + dt2, *args, **kwargs))
        k3 = np.asarray(derivs(y0 + dt2 * k2, thist + dt2, *args, **kwargs))
        k4 = np.asarray(derivs(y0 + dt * k3, thist + dt, *args, **kwargs))
        yout[i + 1] = y0 + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    return yout
