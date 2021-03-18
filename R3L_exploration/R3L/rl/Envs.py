from abc import ABC, abstractmethod
import numpy as np
import gym
from gym.envs.mujoco import MujocoEnv
from R3L.utils import angle_modulus as th_mod

"""
Envs provides a classic interface for all RL environments. It also makes it
easy to transform Gym environments to sparse reward problems.
"""


class AbsEnv(ABC):
    def __init__(self, low_s, high_s, low_a, high_a,
                 disc_s=False, disc_a=False):
        self.low_s, self.high_s = low_s, high_s
        self.low_a, self.high_a = low_a, high_a
        self.d_s_, self.d_a_ = len(low_s), len(low_a)
        self.disc_s, self.disc_a = disc_s, disc_a
        self.n_a_ = int(high_a[0]) if disc_a else -1
        self.n_s_ = int(high_s[0]) if disc_s else -1
        self._state = None

    @abstractmethod
    def step(self, action):
        """
        Executes one step of the environment given an action and returns
        transition information as a tuple.
        :param action: action to execute
        :return: a tuple of (next state, reward, done, information)
        """
        pass

    @abstractmethod
    def reset(self):
        """
        Resets environment at the beginning of an episode.
        :return: a state
        """
        pass

    @abstractmethod
    def render(self):
        """
        Renders the environment.
        """
        pass

    @abstractmethod
    def seed(self, seed=None):
        """
        Specify random seed.
        :param seed: random seed
        """
        pass

    def set_state(self, state):
        """
        Overrides environment current state.
        :param state: new current state
        """
        self._state = state

    def state(self):
        """
        Returns the state of the environment
        :return: state
        """
        return self._state

    def action_space(self):
        """
        Returns either Gym.spaces.Box or Gym.spaces.Discrete object
        describing the action space.
        :return: Gym.spaces.Box or Gym.spaces.Discrete
        """
        if self.disc_a:
            return gym.spaces.discrete.Discrete(self.n_a)
        else:
            return gym.spaces.box.Box(self.low_a.detach().cpu().numpy(),
                                      self.high_a.detach().cpu().numpy())

    def observation_space(self):
        """
        Returns either Gym.spaces.Box or Gym.spaces.Discrete object
        describing the observation space.
        :return: Gym.spaces.Box or Gym.spaces.Discrete
        """

        if self.disc_s:
            return gym.spaces.discrete.Discrete(self.n_s)
        else:
            return gym.spaces.box.Box(self.low_s.detach().cpu().numpy(),
                                      self.high_s.detach().cpu().numpy())

    def get_action_shape(self):
        """
        Returns action shape as tuple.
        :return: tuple
        """
        return len(self.low_a)

    @property
    def n_a(self):
        """
        Returns the number of actions (if state action is discrete). Raises
        exception otherwise.
        :return: (int)
        """
        if not self.disc_a:
            raise ValueError("Environment action space is continuous.")
        if self.d_a_ > 1:
            raise ValueError("Environment action space is multi dimensional.")
        return self.n_a_

    @property
    def n_s(self):
        """
        Returns the number of states (if state space is discrete). Raises
        exception otherwise.
        :return: (int)
        """
        if not self.disc_s:
            raise ValueError("Environment state space is continuous.")
        return self.n_s_

    @property
    def d_s(self):
        """
        Returns the dimension of the state space
        :return: (int)
        """
        return self.d_s_

    @property
    def d_a(self):
        """
        Returns the dimension of the action space
        :return: (int)
        """
        return self.d_a_


def clean_gym(env_name, env):
    """
    Cleans inconsistencies in Gym
    :param env_name:
    :return: set_state_fn, state_fn, obs_spc, act_spc
    """

    # Default
    def set_state_fn(s):
        env.env.state = s

    def state_fn():
        return env.env.state

    def reset_fn():
        env.env.reset()

    obs_spc = env.observation_space
    act_spc = env.action_space

    # env.env.reset = reset_fn
    if env_name == "Ant-v3":
        padding = [0, 0]

        def state_fn():
            return env.env._get_obs()[:27]

        def set_state_fn(s):
            pos = np.hstack(([0, 0], s[:13]))
            env.env.set_state(pos, s[13:])

        high = np.inf * np.ones(state_fn().shape[0])
        low = -high
        obs_spc = gym.spaces.Box(low, high, dtype=np.float32)

    elif isinstance(env.env, MujocoEnv):
        def state_fn():
            position = env.env.sim.data.qpos.flat.copy()
            velocity = env.env.sim.data.qvel.flat.copy()
            return np.concatenate((position, velocity)).ravel()
            # return env.env._get_obs()

        def set_state_fn(s):
            pos, vel = s[:env.env.model.nq], s[env.env.model.nq:]
            env.env.set_state(pos, vel)

        high = np.inf * np.ones(state_fn().shape[0])
        low = -high
        obs_spc = gym.spaces.Box(low, high, dtype=np.float32)
        if env_name == "InvertedDoublePendulumCustom-v0":
            def set_state_fn(s):
                pos = s[:3]
                vel = s[3:6]
                env.env.set_state(pos, vel)
        if env_name == "ReacherSparse-v0":
            obs_spc = env.observation_space
            def state_fn():
                return env.env._get_obs()

            def set_state_fn(s):
                pos = s[:4]
                vel = np.concatenate([s[4:6], [0., 0.]])
                env.env.set_state(pos, vel)
    else:
        print("Warning: clean_gym was not defined for environment "
              "{}".format(env_name))

    return set_state_fn, state_fn, reset_fn, obs_spc, act_spc


class GymEnv(AbsEnv):
    def __init__(self, env_name):
        self.env_name = env_name
        self.env = gym.make(env_name)

        # Fix Gym's mess
        set_state_fn, state_fn, reset_fn, obs_spc, act_spc = clean_gym(env_name,
                                                                       self.env)
        self.state = state_fn
        self.set_state = set_state_fn
        self.reset_env = reset_fn
        self.observation_space = obs_spc
        self.action_space = act_spc

        # Get env bounds
        # Discrete states
        if isinstance(self.observation_space, gym.spaces.Discrete):
            disc_s = True
            low_s = [0]
            high_s = [self.observation_space.n]
        # Continuous states
        else:
            disc_s = False
            low_s = self.observation_space.low
            high_s = self.observation_space.high

        # Discrete actions
        if isinstance(self.action_space, gym.spaces.Discrete):
            disc_a = True
            low_a = [0]
            high_a = [self.action_space.n]
        # Continuous actions
        else:
            disc_a = False
            low_a = self.action_space.low
            high_a = self.action_space.high

        super(GymEnv, self).__init__(low_s, high_s, low_a, high_a,
                                     disc_s, disc_a)

    def step(self, a):
        _, r, done, _ = self.env.step(a)
        return self.state(), r, done

    def reset(self):
        self.env.reset()
        self.reset_env()
        return self.state()

    def render(self):
        self.env.render()

    def seed(self, seed=None):
        self.env.seed(seed)
