from abc import ABC, abstractmethod
import random


class AbsPolicy(ABC):
    def pick(self, s):
        """
        Evaluates policy for given state
        :param s: state
        :return: action
        """
        pass

    def __init__(self, is_disc_a):
        if is_disc_a:
            self.pick = self.pick_disc
        else:
            self.pick = self.pick_cont

    @abstractmethod
    def pick_disc(self, s):
        pass

    @abstractmethod
    def pick_cont(self, s):
        pass


class LocalPolicy(AbsPolicy):
    def __init__(self, agent_helper, model):
        super().__init__(agent_helper.is_disc_a)
        self.agent_helper = agent_helper
        self.model = model

    def pick_disc(self, sg):
        if random.random() < 0.2:
            return self.agent_helper.rand_a(1)
        else:
            return self.model.predict(sg)[0]

    def pick_cont(self, sg):
        if random.random() < 0.2:
            return self.agent_helper.rand_a(1)
        else:
            return self.model.predict(sg)[0]

    def update(self, sg, a):
        self.model.update(sg, a)


class RandomPolicy(AbsPolicy):
    def __init__(self, agent_helper):
        """
        Purely random policy
        :param agent_helper: (AgentHelper)
        """
        super().__init__(agent_helper.is_disc_a)
        self.agent_helper = agent_helper

    def pick_disc(self, s):
        return self.agent_helper.rand_disc_a(len(s))

    def pick_cont(self, s):
        return self.agent_helper.sample_cont_a(len(s))
