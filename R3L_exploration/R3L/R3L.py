import matplotlib.pyplot as plt
import torch

from R3L.RRT import RRT
import time


class R3L:
    def __init__(self, agent_helper, env, local_policy,
                 expand_dis=0.1, goal_sample_rate=0.05, render_env=False):
        self.agent_helper = agent_helper
        self.env = env
        self.render_env = render_env
        d_a = agent_helper.d_a

        # Model for local policy
        self.policy = local_policy

        # Initialise rrt
        self.rrt = RRT(self.agent_helper.sample_cont_s, self.collision_fn,
                       expand_dis, goal_sample_rate, d_a=d_a)

    def collision_fn(self, state, goal):
        # Try one step towards goal using local policy
        self.env.reset()  # Need to reset Gym, or done=True after 200 steps
        self.env.set_state(state)
        s = state.view(1, -1)
        a = self.policy.pick(self.to_sg(s, goal))
        sp, r, done, a = self.env.step(a)
        if self.render_env:
            self.env.render()
        if hasattr(self.policy, 'update'):
            self.policy.update(self.to_sg(s, sp), a)
        return sp, a, done

    @staticmethod
    def to_sg(s, g):
        ds = (g - s)
        return torch.cat((s, ds), 1)

    def run(self, goal_sample_fn, goal_check_fn, n_iter=10000, verbose=0):
        def rectified_goal_check_fn(s):  # revert state normalisation
            return goal_check_fn(self.env.nrms.unnormalise(s))

        def rectified_sample_fn():  # revert state normalisation
            rnd = self.rrt.sample_fn()
            dims, values = goal_sample_fn()
            rnd[0, dims] = torch.tensor(values).to(rnd.device, rnd.dtype)
            return dims, self.env.nrms.normalise(rnd)[0, dims].tolist()

        result = self.rrt.planning(self.env.reset(), rectified_sample_fn,
                                   rectified_goal_check_fn, n_iter, verbose)
        path, actions, found_goal, n_iter_found = result

        if verbose > 0:
            print("Found goal?", found_goal)
            print("Path length:", len(path))

        return result

    def draw_tree(self, path):
        # Only draw tree in 2D
        if path is not None and path.shape[1] != 2:
            return
        # Draw tree
        self.rrt.draw_graph_2d()
        # Draw final path
        if path is not None:
            plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')
        plt.grid(True)
        plt.show()

    def replay_trajectory(self, states, actions):
        assert states.shape[0] == actions.shape[0] + 1
        self.env.reset()
        self.env.set_state(states[0, :])
        for i in range(actions.shape[0]):
            time.sleep(0.05)
            self.env.step(actions[i, :])
            self.env.render()
