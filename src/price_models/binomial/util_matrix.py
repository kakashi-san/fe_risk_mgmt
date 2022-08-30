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


class ForwardPriceMatrix(UtilityMatrix):

    def __init_(
            self,
            u,
            d,
    ):
        super().__init__(
            u=u,
            d=d
        )

    def get_forward_matrix(self, to_level):
        up_con = np.zeros(to_level)
        up_con[0] = self.u
        up_con = up_con.reshape(1, to_level)

        return np.concatenate([up_con, self.d * np.identity(to_level)], axis=0)


class BackwardOptionPriceMatrix(UtilityMatrix):

    @property
    def q(self):
        return self._q

    @q.setter
    def q(self, val):
        self._q = val

    def __init__(
            self,
            q,
            u=0.5,
            d=0.5,

    ):
        super().__init__(
            u=u,
            d=d
        )

        self.q = q

    def get_backward_matrix(self, from_level):
        zeroth_row = np.zeros(shape=(from_level,))
        zeroth_row[-2:] = [1 - self.q, self.q]

        iter_row = zeroth_row.tolist()
        collect_rows = []
        for i in range(from_level - 1):
            collect_rows.append(iter_row[::-1])
            iter_row = self.stride_element(iter_row)

        return np.array(collect_rows)

    @staticmethod
    def stride_element(_list):
        _list += [_list.pop(0)]
        return _list
