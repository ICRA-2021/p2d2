import torch
import numpy as np


class DynamicArray:
    """
    Class providing an array of dynamic and increasing size.
    """

    def __init__(self, d=1, init_size=100):
        """
        Initialises the array of dynamic size but fixed dimension.
        :return:
        """
        self.capacity = init_size
        self.dim = d
        self.size = 0
        self.data = None

    def extend(self, rows):
        """
        Appends multiple elements to the array.
        :return:
        """
        for row in rows:
            self.append(row)

    def append(self, x):
        """
        Appends one element to the array.
        :return:
        """
        if self.size == 0:
            self.dim = x.shape[1] if len(x.shape) == 2 else len(x)
            self._clear(x.dtype, x.device)
        elif self.size == self.capacity:
            self.capacity *= 4
            new_data = torch.zeros((self.capacity, self.dim),
                                   dtype=self.data.dtype,
                                   device=self.data.device)
            new_data[:self.size, :] = self.data
            self.data = new_data

        self.data[self.size, :] = x
        self.size += 1

    def get(self):
        """
        Returns the data from the array.
        :return: Tensor with data
        """
        if self.size == 0:
            return None
        return self.data[:self.size, :]

    def _clear(self, dtype, device):
        self.data = torch.zeros((self.capacity, self.dim),
                                dtype=dtype, device=device)
        self.size = 0

    def clear(self):
        """
        Clears the data from the array.
        :return: None
        """
        if self.data is not None:
            self._clear(self.data.dtype, self.data.device)

    def __getitem__(self, key):
        return self.get()[key, :]

    def __setitem__(self, key, value):
        self.data[key, :] = value

    def __len__(self):
        return self.size


def smw_inv_correction(a_inv, u, v):
    """
    Sherman-Morrison-Woodbury update for rank k updates to the inverse of
    a matrix a_inv. This function computes the UPDATE to the matrix inverse
    such that (A+uv)^{-1} = A^{-1} + UPDATE

    ref:     http://mathworld.wolfram.com/WoodburyFormula.html
             https://en.wikipedia.org/wiki/Woodbury_matrix_identity
    :param a_inv: previous matrix inverse as n x n (Tensor)
    :param u: rank k update vector as n x k (Tensor)
    :param v: rank k update vector as k x n (Tensor)
    :return: update to previous matrix inverse as n x n (Tensor)
    """
    dtype = a_inv.dtype
    rank = u.shape[1]
    su = a_inv.mm(u)
    vs = v.mm(a_inv)
    i_plus_vsu_inv = torch.inverse(torch.eye(rank, dtype=dtype) + vs.mm(u))
    su_i_plus_vsu = su.mm(i_plus_vsu_inv)
    return su_i_plus_vsu.mm(vs)


def batch_generator(arrays, batch_size, wrap_last_batch=False):
    """
    Batch generator() function for yielding [x_train, y_train] batch slices for
    numpy arrays
    Appropriately deals with looping back around to the start of the dataset
    Generate batches, one with respect to each array's first axis.
    :param arrays:[array, array]  or [array, None]...
                  e.g. [X_trn, Y_trn] where X_trn and Y_trn are ndarrays
    :param batch_size: batch size
    :param wrap_last_batch: whether the last batch should wrap around dataset
    to include first datapoints (True), or be smaller to stop at the end of
    the dataset (False).
    :return:
    """
    starts = [0] * len(
        arrays)  # pointers to where we are in iteration     --> [0, 0]
    while True:
        batches = []
        for i, array in enumerate(arrays):
            start = starts[i]
            stop = start + batch_size
            diff = stop - array.shape[0]
            if diff <= 0:
                batch = array[start:stop]
                starts[i] += batch_size
            else:
                if wrap_last_batch:
                    batch = torch.cat((array[start:], array[:diff]))
                    starts[i] = diff
                else:
                    batch = array[start:]
                    starts[i] = 0
            batches.append(batch)
        yield batches


class Normaliser:
    """
    Normalise/unnormalise data to [0,1] or [-1,1].
    """

    def __init__(self, low, high, zero_one_interval=True):
        """
        :param low: List of lower-bounds for each dimension
        :param high: List of upper-bounds for each dimension
        :param zero_one_interval: whether normalised interval should be [0,1]
        (default) or [-1,1]
        """
        assert (len(low) == len(high) and
                "Upper and lower bounds much be same dimension.")
        assert (torch.isfinite(torch.sum(low)) and
                "Lower bound elements must be numbers.")
        assert (torch.isfinite(torch.sum(high)) and
                "Upper bound elements must be numbers.")

        space_range = (high - low).detach().clone()

        if torch.sum(space_range > 100) > 0:
            print("Warning: normalising over large space.")

        self.factor = space_range / (1.0 if zero_one_interval else 2.0)
        self.inv_factor = (1.0 if zero_one_interval else 2.0) / space_range
        self.low = low
        self.offset = 0.0 if zero_one_interval else -1.0
        self.bounds_norm = (space_range * 0 - (0 if zero_one_interval else 1),
                            space_range * 0 + 1)
        self.bounds_orig = (low, high)

    def normalise(self, x):
        """
        Normalise x.
        :param x: list with 1 element, or N*D numpy matrix with N elements
        :return: numpy matrix with shape of input
        """
        if len(x.shape) == 1:
            assert (x.shape == self.factor.shape and
                    "Data must be same dimension as lower/upper bounds")
        else:
            assert (x.shape[1] == self.factor.shape[0] and
                    "Data must be same dimension as lower/upper bounds")

        return (x - self.low) * self.inv_factor + self.offset

    def unnormalise(self, x):
        """
        Unnormalise x.
        :param x: list with 1 element, or N*D numpy matrix with N elements
        :return: numpy matrix with shape of input
        """
        if len(x.shape) == 1:
            assert (x.shape == self.factor.shape and
                    "Data must be same dimension as lower/upper bounds")
        else:
            assert (x.shape[1] == self.factor.shape[0] and
                    "Data must be same dimension as lower/upper bounds")

        return (x - self.offset) * self.factor + self.low

    def bounds_normalised(self):
        return self.bounds_norm

    def bounds_original(self):
        return self.bounds_orig


def angle_modulus(th):
    return ((th + np.pi) % (2 * np.pi)) - np.pi
