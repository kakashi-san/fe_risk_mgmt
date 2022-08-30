from abc import ABC

import numpy as np
from src.price_models.binomial.stock_price_params import StockPriceParams
from src.price_models.binomial.util_matrix import ForwardPriceMatrix
from src.price_models.binomial.util_matrix import BackwardOptionPriceMatrix as bopm


class StockPriceLattice(StockPriceParams):

    def __init__(
            self,
            S0,
            n,
            u,
            d,
    ):
        super().__init__(
            S0=S0,
            n=n,
            u=u,
            d=d,
        )

        self.sfp = ForwardPriceMatrix(
            u=u,
            d=d
        )

    def generate_lattice(
            self,
    ):
        self.tree = self._loop_levels(
            fcn=self._forward_prop_level,
            levels=range(self.n),
            init_level=np.array([[self.S0]])
        )

    def _forward_prop_level(
            self,
            lat_level,
            to_level
    ):
        forward_matrix = self.sfp.get_forward_matrix(
            to_level=to_level
        )

        return np.dot(forward_matrix, lat_level)

    @staticmethod
    def _loop_levels(
            fcn,
            levels,
            init_level
    ):
        lat_levels = [init_level]
        lat_level = init_level

        for level in levels:
            lat_level = fcn(
                lat_level,
                to_level=level + 1
            )
            lat_levels.append(lat_level)

        return lat_levels

    @property
    def tree(self):
        return self._tree

    @tree.setter
    def tree(self, value):
        self._tree = value


class OptionsPricing(StockPriceLattice):
    @property
    def stock_price_tree(self):
        return self.tree

    @stock_price_tree.__init__
    def stock_price_tree(self):
        self.generate_lattice()

    @property
    def K(self):
        return self._K

    @K.setter
    def K(self, val):
        self._K = val

    def __init__(self,
                 K,
                 model,
                 is_call=True,
                 is_american=False
                 ):
        super().__init__(
            S0=model.S0,
            n=model.n,
            u=model.u,
            d=model.d,
        )

        self.K = K

        self.multiplier = 1 if is_call else -1

        self.is_american = is_american

        self.bopm = bopm(
            q=model.q,
            u=model.u,
            d=model.d
        )

    def _get_payoff(
            self,
            diff_level,
            base=0,
    ):
        return np.maximum(
            base,
            self.multiplier * diff_level
        )

    def _generate_option_price_lattice(self):
        exp_cashflow = self._get_payoff(
            diff_level=self.tree[-1] - self.K
        )

        levels = len(self.tree)
        for level in reversed(range(levels)[1:]):
            if self.is_american:
                diff_level = self.tree[level] - self.K

                exp_cashflow = self._get_payoff(
                    diff_level=diff_level,
                    base=exp_cashflow,
                )
                exp_cashflow = self.bopm.get_backward_matrix(
                    from_level=level + 1
                ).dot(exp_cashflow)
            else:
                exp_cashflow = self.bopm.get_backward_matrix(
                    from_level=level + 1
                ).dot(exp_cashflow)
        return exp_cashflow

    def get_price(self):
        return self._generate_option_price_lattice()


class Model(ABC):
    @property
    def S0(self):
        return self._S0

    @S0.setter
    def S0(self, val):
        self._S0 = val

    @property
    def q(self):
        return self._q

    @q.setter
    def q(self, val):
        self._q = val

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, val):
        self._n = val

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
            S0,
            q,
            u,
            d,
            n,
    ):
        self.S0 = S0

        self.q = q

        self.u = u

        self.d = d

        self.n = n


model = Model(
    S0=100,
    q=0.6,
    u=1.01,
    d=1 / 1.01,
    n=5,
)
op = OptionsPricing(
    K=103,
    model=model,
    is_american=True,
)
op.generate_lattice()
# op.print_tree()
print(op.get_price())
