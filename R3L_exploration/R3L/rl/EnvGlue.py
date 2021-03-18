import torch

from R3L.rl.Envs import AbsEnv
from R3L.utils import Normaliser

"""
This is the glue (wrapper) between RL environments from Env.py and the runner.
EnvGlue takes care of state and action space normalisation.
"""


class EnvGlue(AbsEnv):
    def __init__(self, env, low_s=None, high_s=None, normalised=True,
                 dtype=torch.float, device=torch.device("cpu")):
        self.dtype = dtype
        self.device = device
        self.env = env
        self.normalised = normalised

        high_a_l = [h - (1 if self.env.disc_a else 0) for h in self.env.high_a]
        high_a = torch.tensor(high_a_l).to(self.device, self.dtype)
        if low_s is None:
            low_s = torch.tensor(self.env.low_s).to(self.device, self.dtype)
        if high_s is None:
            high_s = torch.tensor(self.env.high_s).to(self.device,
                                                      self.dtype)
        low_a = torch.tensor(self.env.low_a).to(self.device, self.dtype)
        if normalised:
            self.nrms = Normaliser(low_s, high_s, False)
            self.nrma = Normaliser(low_a, high_a, False)

        super().__init__(low_s, high_s, low_a, high_a, self.env.disc_s,
                         self.env.disc_a)

    def step(self, a):
        if self.normalised:
            raw_a = self.nrma.unnormalise(a)
        else:
            raw_a = a
        if self.env.disc_a:
            raw_a = max(0, min(int(round(float(raw_a))), int(self.high_a[0])))
        else:
            raw_a = raw_a.max(self.low_a).min(
                self.high_a).detach().cpu().numpy()
        raw_s, r, done = self.env.step(raw_a)
        raw_s = torch.tensor(raw_s).to(self.device, self.dtype).reshape(1, -1)
        raw_a = torch.tensor(raw_a).to(self.device, self.dtype).reshape(1, -1)
        r = torch.tensor([[r]]).to(self.device, self.dtype)
        if self.normalised:
            real_a = self.nrma.normalise(raw_a)
            return self.nrms.normalise(raw_s), r, done, real_a
        else:
            return raw_s, r, done, raw_a

    def steps(self, ss, aa):
        # NOTE: May not handle GPU device
        if self.normalised:
            raw_aa = self.nrma.unnormalise(aa)
            raw_ss = self.nrms.unnormalise(ss)
        else:
            raw_aa = aa
            raw_ss = ss
        if self.env.disc_a:
            raw_aa = torch.clamp(torch.round(raw_aa), 0, self.high_a[0])
        raw_ssp, rr, done = self.env.steps(raw_ss, raw_aa)
        if self.normalised:
            return self.nrms.normalise(raw_ssp), rr, done
        else:
            return raw_ssp, rr, done

    def cost(self, ss):
        # NOTE: May not handle GPU device
        if self.normalised:
            raw_ss = self.nrms.unnormalise(ss)
        else:
            raw_ss = ss
        return self.env.cost(raw_ss)

    def reset(self):
        raw_s = torch.tensor(self.env.reset()).to(self.device,
                                                  self.dtype).reshape(1, -1)
        if self.normalised:
            return self.nrms.normalise(raw_s)
        else:
            return raw_s

    @property
    def bounds_a(self):
        if self.normalised:
            return self.nrma.bounds_normalised()
        else:
            low = torch.tensor(self.env.low_a).to(self.device, self.dtype)
            h = torch.tensor(self.env.high_a).to(self.device, self.dtype)
            return low, h

    @property
    def bounds_s(self):
        if self.normalised:
            return self.nrms.bounds_normalised()
        else:
            low = torch.tensor(self.env.low_s).to(self.device, self.dtype)
            h = torch.tensor(self.env.high_s).to(self.device, self.dtype)
            return low, h

    def render(self):
        self.env.render()

    def seed(self, seed=None):
        self.env.seed(seed)

    def set_state(self, state):
        if self.normalised:
            raw_state = self.nrms.unnormalise(state)
        else:
            raw_state = state
        self.env.set_state(raw_state.clone().detach().cpu().numpy())
