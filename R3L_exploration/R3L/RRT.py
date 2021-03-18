import torch
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

from R3L.utils import DynamicArray


class RRT:
    def __init__(self, sample_fn, collision_fn, expand_dis=1.0,
                 goal_sample_rate=.05, d_a=1):
        """
        Rapidly-exploring Random Trees (RRT) Planning.
        :param sample_fn: sample function to select a random goal
        :param collision_fn: collision function, test whether it is possible
        to go from one node to another.
        :param expand_dis: If not using actions, distance of tree expansions
        :param goal_sample_rate: probability to sample goal
        :param d_a: action space dimension (default: 1)
        """
        self.collision_fn = collision_fn
        self.sample_fn = sample_fn
        self.expand_dis = expand_dis
        self.goal_sample_rate = goal_sample_rate
        self.d_a = d_a

        self.node_pos = None
        self.node_parents = DynamicArray()
        self.node_action = DynamicArray()

    def planning(self, start, goal_sample_fn, goal_check_fn, max_iter=500,
                 verbose=0):
        """
        Run path planning from start to end.
        :param start: starting node as (tensor)
        :param goal_sample_fn: function to sample goals, returns dims and values
        :param goal_check_fn: function to check whether a state achieves the
        final goal or not
        :param max_iter: number of planning iterations (int)
        :param verbose: verbose level (int)
        :return tuple of (path, path_acts, found_goal, iter_i)
            - path: list of nodes to reach goal
            - path_acts: list of actions to reach goal
            - found_goal: whether goal was found
            - iter_i: number of iterations to find successful plan
        """
        found_goal = False
        self.node_pos = DynamicArray(d=start.shape[1])
        self.node_pos.append(start)
        self.node_action.append(torch.tensor([-1.] * self.d_a))
        self.node_parents.append(torch.tensor([-1]))
        gen = range(max_iter)
        if verbose > 1:
            gen = tqdm(gen)
        iter_i = 0
        for iter_i in gen:
            # Random Sampling
            rnd = self.sample_fn()
            if iter_i > 0 and random.random() < self.goal_sample_rate:
                dims, values = goal_sample_fn()
                rnd[0, dims] = torch.tensor(values).to(rnd.device, rnd.dtype)

            # Find nearest node
            n_id = self._get_nearest_list_index(rnd)

            # expand tree
            nearest_node = self.node_pos[n_id]
            new_node = self._expand_tree_classic(rnd, nearest_node)
            new_node, a, done = self.collision_fn(nearest_node, new_node)
            if (new_node - nearest_node).norm(2) < 1e-10:
                continue

            # Check goal reaching
            if goal_check_fn(new_node):
                found_goal = True
            elif done:
                continue  # Obstacle

            # Add state/action to graph
            self.node_pos.append(new_node)
            self.node_parents.append(n_id)
            self.node_action.append(a)
            if found_goal:
                if verbose > 1:
                    print("Goal found")
                break

        # Generate path to found goal (if any), or towards a sampled goal
        if found_goal:
            gen_path_goal = None
        else:
            gen_path_goal = self.sample_fn()
            dims, values = goal_sample_fn()
            gen_path_goal[0, dims] = torch.tensor(values).to(
                gen_path_goal.device, gen_path_goal.dtype)
        path, path_acts = self.generate_path(gen_path_goal)
        return path, path_acts, found_goal, iter_i

    def generate_path(self, goal=None):
        """
        Generated path from start to goal using RRT tree.
        :param goal: goal state for which to generate path. (In practice,
        a path to the closest tree node is returned). goal=None returns a
        path to the node added last in the tree (default: None).
        :return: tensor of states (n+1, d_s), tensor of actions (n, d_a)
        """
        # Generate path from tree
        path = []
        path_acts = []
        if goal is None:  # Start from lastly added node in tree
            last_index = len(self.node_pos) - 1
        else:  # Find closest node to given goal
            last_index = self._get_nearest_list_index(goal)
        # Traverse tree up to root
        n = 0
        while self.node_parents[last_index] != -1:
            path.append(self.node_pos[last_index].view(1, -1))
            path_acts.append(self.node_action[last_index])
            last_index = int(self.node_parents[last_index])
            n += 1
        path.append(self.node_pos[0].view(1, -1))

        # Reverse path (start to end)
        path = torch.cat(path[::-1], dim=0).view(n + 1, -1)
        path_acts = torch.cat(path_acts[::-1], dim=0).view(n, -1)
        return path, path_acts

    def generate_transitions(self):
        obs = self.node_pos[self.node_parents[1:]]
        act = self.node_action[1:]
        obs2 = self.node_pos[1:]
        return obs, act, obs2

    def _get_nearest_list_index(self, point):
        """
        Returns index of the closest node to given point
        :param point: given point
        :return: index
        """
        distances = ((self.node_pos.get() - point) ** 2).sum(1)
        return int(torch.argmin(distances))

    def _expand_tree_classic(self, node, nearest_node):
        unit_vec = node[0, :] - nearest_node
        unit_vec /= torch.norm(unit_vec, 2)
        new_node = torch.clone(nearest_node).view(1, -1)
        new_node[0, :] += self.expand_dis * unit_vec

        return new_node

    def draw_graph_2d(self, start=None, end=None, obstacle_list=None, rnd=None):
        """
        Draw Graph. Only valid for planning in 2D.
        """
        if len(self.node_pos[0]) != 2:
            return
        # plt.clf()
        plt.figure()
        plt.grid(True)
        if rnd is not None:
            plt.plot(rnd[0, 0], rnd[0, 1], "^k")

        # Plot obstacles
        if obstacle_list is not None:
            for (ox, oy, size) in obstacle_list:
                plt.plot(ox, oy, "ok", ms=30 * size)

        # Start and end
        if start is not None:
            plt.plot(start[0, 0], start[0, 1], "xb")
        if end is not None:
            plt.plot(end[0, 0], end[0, 1], "xr")

        for i in range(len(self.node_pos)):
            if self.node_parents[i] != -1:
                parent_id = int(self.node_parents[i])
                from_node, to_node = self.node_pos[i], self.node_pos[parent_id]
                plt.plot([from_node[0], to_node[0]], [from_node[1], to_node[1]],
                         "-g")
                # plt.pause(0.001)
