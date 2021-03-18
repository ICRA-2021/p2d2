import torch


class AgentHelper:
    def __init__(self, glue):
        """
        AgentHelper provides Agents with a series of function to access environment
        state and action spaces.
        """
        self.glue = glue
        self.dtype = self.glue.dtype
        self.device = self.glue.device

    @staticmethod
    def to_state_action_pair(ss, aa):
        ssaa = torch.cat((ss, aa), 1)
        return ssaa

    @property
    def is_disc_s(self):
        return self.glue.env.disc_s

    @property
    def is_disc_a(self):
        return self.glue.env.disc_a

    def all_disc_a(self):
        # This could be extend to multiple dimensional discrete actions,
        # similarly to function "all_disc_s". Not useful at the moment...
        bounds = self.bounds_a
        end = bounds[1].item()
        if not self.glue.normalised:
            end -= 1
        return torch.linspace(
            bounds[0].item(), end, self.n_a,
            dtype=self.dtype, device=self.device).reshape(-1, 1)

    def all_disc_s(self):
        bounds = self.bounds_s
        end = bounds[1]
        if not self.glue.normalised:
            end -= 1
        if len(end) == 1:
            return torch.linspace(bounds[0].item(), end, self.n_a,
                                  dtype=self.dtype, device=self.device)
        else:
            mg = torch.meshgrid([
                torch.range(bounds[0][i], end[i]) for i in range(len(end))]).to(
                dtype=self.dtype, device=self.device)
            return torch.cat([mg[i].reshape(-1, 1) for i in range(len(end))],
                             dim=1)

    def rand_disc_a(self, n=1):
        all_a = self.all_disc_a()
        if n >= all_a.shape[0]:
            return all_a
        else:
            return all_a[torch.randperm(all_a.shape[0])[:n]]

    def rand_disc_s(self, n=1):
        all_s = self.all_disc_s()
        if n >= all_s.shape[0]:
            return all_s
        else:
            return all_s[torch.randperm(all_s.shape[0])[:n]]

    def sample_cont_a(self, n_samps=1):
        r = torch.rand(n_samps, self.d_a).to(self.device, self.dtype)
        low, high = self.glue.bounds_a
        return r * (high - low) + low

    def sample_cont_s(self, n_samps=1):
        r = torch.rand(n_samps, self.d_s).to(self.device, self.dtype)
        low, high = self.glue.bounds_s
        return r * (high - low) + low

    def rand_a(self, n):
        """
        Generates N random actions.
        :param n: number of random actions (int).
        :return: array of N random actions
        """
        if self.is_disc_a:
            return self.rand_disc_a(n)
        else:
            return self.sample_cont_a(n)

    def rand_s(self, n):
        """
        Generates N random states.
        :param n: number of random states (int).
        :return: array of N random states
        """
        if self.is_disc_s:
            return self.rand_disc_s(n)
        else:
            return self.sample_cont_s(n)

    @property
    def n_a(self):
        """
        Get number of actions.
        :return: (int)
        """
        return self.glue.n_a

    @property
    def n_s(self):
        """
        Get number of states.
        :return: (int)
        """
        return self.glue.n_s

    @property
    def bounds_a(self):
        """
        Get actions space bounds.
        :return: tuple of lower and upper bounds
        """
        return self.glue.bounds_a

    @property
    def bounds_s(self):
        """
        Get state space bounds.
        :return: tuple of lower and upper bounds
        """
        return self.glue.bounds_s

    @property
    def d_s(self):
        """
        Get state space dimension.
        :return: (int)
        """
        return self.glue.d_s

    @property
    def d_a(self):
        """
        Get action space dimension.
        :return: (int)
        """
        return self.glue.d_a

    def is_s_valid(self, s):
        """
        Checks whether state is valid.
        :param s: state
        :return: (bool)
        """
        low = torch.all(s >= self.bounds_s[0])
        high = torch.all(s < self.bounds_s[1])
        return low + high == 2

    def is_a_valid(self, a):
        """
        Checks whether action is valid.
        :param a: action
        :return: (bool)
        """
        low = torch.all(a >= self.bounds_a[0])
        high = torch.all(a < self.bounds_a[1])
        return low + high == 2
