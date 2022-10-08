import numpy as np
from abc import ABC


class UtilityMatrix(ABC):
    @property
    def u(self):
        return self._u

    @u.setter
    def u(self, val):
        self._u = val

    @property
    def d(self):
        return self._d

    @d.setter
    def d(self, val):
        self._d = val

    def __init__(
            self,
            u,
            d,
    ):
        self.u = u
        self.d = d


class ForwardProp(ABC):
    def __init__(
            self,
            u,
            d,
    ):
        self.u = u
        self.d = d

    def get_forward_prop_matrix(self, from_level):
        return np.r_[
            self.u * np.eye(from_level + 1)[:1],
            self.d * np.eye(from_level + 1),
        ]


class BackwardProp(ABC):

    @property
    def q(self):
        return self._q

    @q.setter
    def q(self, val):
        self._q = val

    def __init__(
            self,
            q,
    ):
        self.q = q

    def get_backward_prop_matrix(self, from_level):
        """
        time range t = 0,1,2,..
        each time level has t+1 states
        :param from_level: level t having t+1 states.
        :return: backward propagation matrix of dimensions (t,t+1)
        """
        # print("frommmmm", from_level)
        return np.c_[
                self.q * np.eye(from_level),
                np.zeros(from_level)
            ] + np.c_[
                np.zeros(from_level),
                (1 - self.q) * np.eye(from_level)
            ]
