from abc import ABC

import numpy as np
from src.price_models.binomial.stock_price_params import StockPriceParams
from src.price_models.binomial.util_matrix import ForwardProp, BackwardProp


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

        self.sfp = ForwardProp()

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
        forward_matrix = self.sfp.get_forward_prop_matrix(
            from_level=to_level - 1
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

        self.bopm = BackwardProp(
            q=model.q,
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
                exp_cashflow = self.bopm.get_backward_prop_matrix(
                    from_level=level + 1
                ).dot(exp_cashflow)
            else:
                exp_cashflow = self.bopm.get_backward_prop_matrix(
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


class InterestRateModel(ABC):
    @property
    def r0(self):
        return self._r0

    @r0.setter
    def r0(self, val):
        self._r0 = val

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

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, val):
        self._n = val

    @property
    def _lattice(self):
        return self._lat  # [np.array([[self.r0]])]

    @_lattice.setter
    def _lattice(self, val):
        self._lat = val

    def __init__(
            self,
            r0,
            u,
            d,
            n,
    ):
        self.r0 = r0
        self.u = u
        self.d = d
        self.n = n

        self._generate_lattice()

    def _init_lattice(self):
        self._lattice = [np.array([[self.r0]])]

    def _generate_lattice(self):
        self._set_up_forward_prop()
        self._init_lattice()

        for level in list(range(self.n + 1)):
            self._lattice.append(
                self._forward_prop.get_forward_prop_matrix(
                    from_level=level,
                ).dot(
                    self._lattice[-1]
                )
            )

    def _set_up_forward_prop(self):
        self._forward_prop = ForwardProp(
            u=self.u,
            d=self.d,
        )

    def get_lattice(self):
        return self._lattice

    def print_lattice(self):
        for lattice_arr in self._lattice:
            print(lattice_arr)


class ZCBPriceLattice(ABC):
    """
    Generate n period price lattice for Zero Coupon Bond.

    q: risk neutral probability for price movements
    payoff: pay-off received at the end of expiry period
    n: number of periods
    """

    @property
    def q(self):
        return self._q

    @q.setter
    def q(self, val):
        self._q = val

    @property
    def payoff(self):
        return self._payoff

    @payoff.setter
    def payoff(self, val):
        self._payoff = val

    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, val):
        self._n = val

    @property
    def _array_expiry_payoff(
            self,
    ):
        return self.payoff * np.ones(
            shape=(self.n, 1)
        )

    @property
    def _lattice(self):
        return self._lat

    @_lattice.setter
    def _lattice(self, val):
        self._lat = val

    @property
    def _int_rate_lattice(self):
        return self.ir_model.get_lattice()

    def __init__(
            self,
            q,
            payoff,
            n,
            ir_model,
    ):
        self.q = q
        self.payoff = payoff
        self.n = n
        self.ir_model = ir_model
        self._generate_lattice()

    def _set_up_backward_prop(self):
        self._backprop = BackwardProp(
            q=self.q,
        )

    def _init_lattice(self):
        self._lattice = [self._array_expiry_payoff]

    def _back_prop(
            self,
            from_level,
            int_rate_lat,
    ):
        int_rat_mul = (int_rate_lat + 1) ** -1

        self._lattice.insert(
            0,
            self._backprop.get_backward_prop_matrix(
                from_level=from_level
            ).dot(self._lattice[0]) * int_rat_mul

        )

    def _generate_lattice(self):
        self._init_lattice()

        if self._lattice:
            self._set_up_backward_prop()

            while len(self._lattice) < self.n:
                curr_level = len(self._lattice[0]) - 1
                int_rate_lat = self._int_rate_lattice[curr_level - 1]

                self._back_prop(
                    from_level=curr_level,
                    int_rate_lat=int_rate_lat
                )

    def print_lattice(self):
        for lattice_arr in self._lattice:
            print(lattice_arr)

