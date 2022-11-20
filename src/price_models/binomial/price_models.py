from abc import ABC

import numpy as np
from src.price_models.binomial.stock_price_params import StockPriceParams
from src.price_models.binomial.util_matrix import ForwardProp, BackwardProp
from src.price_models.binomial.binomial_tree import BackPropLattice


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

    @property
    def get_lattice(self):
        return self._lattice

    def print_lattice(self):
        for lattice_arr in self._lattice:
            print(lattice_arr)


class OptionsPricingZCB(ABC):
    @property
    def q(self):
        return self._q

    @q.setter
    def q(self, val):
        self._q = val

    @property
    def n_period(self):
        return self._n_period

    @n_period.setter
    def n_period(self, val):
        self._n_period = val

    @property
    def strike_price(self):
        return self._strike_price

    @strike_price.setter
    def strike_price(self, val):
        self._strike_price = val

    def _set_up_backward_prop(self):
        self._backprop = BackwardProp(
            q=self.q,
        )

    def _present_val_fut_cashflow(
            self,
            from_level,
            int_rat_mul,
            fut_cashflow,
    ):
        return self._backprop.get_backward_prop_matrix(
            from_level=from_level
        ).dot(
            fut_cashflow
        ) * int_rat_mul

    def _back_push_lattice(
            self,
            val,
    ):
        self._lattice.insert(
            0,
            val
        )

    def _append_to_lattice(
            self,
            val
    ):
        self._lattice.append(
            val
        )

    def _get_exer_val(
            self,
            val
    ):
        return np.maximum(
            0,
            (val - self.strike_price) * self.option
        )

    def _back_prop(
            self,
            from_level,
            int_rate_lat,
            fut_cashflow,
    ):
        int_rat_mul = (int_rate_lat + 1) ** -1

        present_val_fut_cf = self._present_val_fut_cashflow(
            from_level=from_level,
            int_rat_mul=int_rat_mul,
            fut_cashflow=fut_cashflow,
        )
        if self.is_american:
            exercise_val = self._get_exer_val(
                val=self._zcb_price_lattice.lattice[from_level - 1]
            )
        else:
            exercise_val = None

        return present_val_fut_cf, exercise_val

    def _generate_lattice(self):

        # initialise the payoff
        self._init_lattice()

        # set up back prop matrix
        self._set_up_backward_prop()
        curr_level = self._n_period

        while len(self._lattice) < self._n_period + 1:
            # self.print_lattice()

            present_val_fut_cf, exec_val = self._back_prop(
                from_level=curr_level,
                fut_cashflow=self._lattice[0],
                int_rate_lat=self._int_rate_model.get_lattice[curr_level - 1]
            )

            # American option
            if self.is_american:
                lattice_state = np.maximum(
                    present_val_fut_cf,
                    exec_val,
                )

            # European option
            else:
                assert exec_val is None
                lattice_state = present_val_fut_cf

            self._back_push_lattice(
                val=lattice_state
            )
            curr_level -= 1

    def _init_lattice(self):
        self._zcb_payoff_expiry = self._zcb_price_lattice.lattice[
            self.n_period
        ]

        self._lattice = [
            self._get_exer_val(
                val=self._zcb_payoff_expiry
            ),
        ]

    def print_lattice(self):
        for lattice_arr in self._lattice:
            print(lattice_arr)

    def __init__(
            self,
            zcb_price_lattice,
            int_rate_model,
            n_period,
            q,
            strike_price,
            is_american=True,
            option=1,  # 1 for call option, -1 for put option
            early_exercise=True,
    ):
        self._zcb_price_lattice = zcb_price_lattice
        self._int_rate_model = int_rate_model
        self.n_period = n_period
        self.q = q
        self.strike_price = strike_price
        self.is_american = is_american
        self.option = option
        self.early_exercise = early_exercise

        self._generate_lattice()


class ForwardsPricing(ABC):

    @property
    def mat_t(self):
        return self._mat_t

    @mat_t.setter
    def mat_t(self, val):
        self._mat_t = val

    @property
    def q(self):
        return self._cbb_model.q

    @property
    def coupon_pay(self):
        return self._cbb_model.coupon_pay

    @property
    def _int_rate_lattice(self):
        return self._ir_model.get_lattice

    @property
    def _lattice(self):
        return self._lat

    @_lattice.setter
    def _lattice(self, val):
        self._lat = val

    @property
    def lattice(self):
        return self._lattice

    def __init__(
            self,
            mat_t,
            cbb_model,
            ir_model,
    ):
        self.mat_t = mat_t
        self._cbb_model = cbb_model
        self._ir_model = ir_model

        self._generate_lattice()

    def forward_price(self):
        zcb_model = BondPricing(
            n=self.mat_t,
            q=self.q,
            ir_model=self._ir_model,
            payoff=self._cbb_model.payoff,
            coupon_rate=0.0,
        )
        zcb_price = zcb_model.lattice[0][0]
        forward_ccb = self.lattice[0][0]
        return 100 * forward_ccb / zcb_price

    def _init_lattice(self):
        self._lattice = [
            self._cbb_model.lattice[self.mat_t] - self.coupon_pay
        ]

    def _set_up_backward_prop(self):
        self._backprop_set_up = BackwardProp(
            q=self.q,
        )

    def _back_push_lattice(
            self,
            val,
    ):
        self._lattice.insert(
            0,
            val
        )

    def _append_to_lattice(
            self,
            val
    ):
        self._lattice.append(
            val
        )

    def _back_prop(
            self,
            from_level,
            int_rate_lat,
            futures_mode=False,
    ):

        if futures_mode:
            int_rat_mul = 1
        else:
            int_rat_mul = (int_rate_lat + 1) ** -1

        _lattice_arr = self._backprop_set_up.get_backward_prop_matrix(
            from_level=from_level
        ).dot(self._lattice[0]) * int_rat_mul

        self._back_push_lattice(
            val=_lattice_arr
        )

    def _generate_lattice(self):
        self._init_lattice()
        self._set_up_backward_prop()

        while len(self._lattice) < self.mat_t + 1:
            curr_level = len(self._lattice[0]) - 1
            int_rate_lat = self._int_rate_lattice[curr_level - 1]

            self._back_prop(
                from_level=curr_level,
                int_rate_lat=int_rate_lat,
            )

    def print_lattice(self):
        for lattice_arr in self._lattice:
            print(lattice_arr)


class FuturesPricing(ABC):
    @property
    def mat_t(self):
        return self._mat_t

    @mat_t.setter
    def mat_t(self, val):
        self._mat_t = val

    @property
    def coupon_pay(self):
        return self._cbb_model.coupon_pay

    @property
    def _int_rate_lattice(self):
        return self._ir_model.get_lattice

    @property
    def _lattice(self):
        return self._lat

    @_lattice.setter
    def _lattice(self, val):
        self._lat = val

    @property
    def lattice(self):
        return self._lattice

    def __init__(
            self,
            mat_t,
            cbb_model,
            ir_model,
    ):
        self.mat_t = mat_t
        self._cbb_model = cbb_model
        self._ir_model = ir_model

        self._generate_lattice()

    def _init_lattice(self):
        self._lattice = [
            self._cbb_model.lattice[self.mat_t] - self.coupon_pay
        ]

    def _set_up_backward_prop(self):
        self._backprop_set_up = BackwardProp(
            q=0.5,
        )

    def _back_push_lattice(
            self,
            val,
    ):
        self._lattice.insert(
            0,
            val
        )

    def _append_to_lattice(
            self,
            val
    ):
        self._lattice.append(
            val
        )

    def _back_prop(
            self,
            from_level,
            int_rate_lat,
            futures_mode=False,
    ):

        if futures_mode:
            int_rat_mul = 1
        else:
            int_rat_mul = (int_rate_lat + 1) ** -1

        _lattice_arr = self._backprop_set_up.get_backward_prop_matrix(
            from_level=from_level
        ).dot(self._lattice[0]) * int_rat_mul

        self._back_push_lattice(
            val=_lattice_arr
        )

    def _generate_lattice(self):
        self._init_lattice()
        self._set_up_backward_prop()

        while len(self._lattice) < self.mat_t + 1:
            curr_level = len(self._lattice[0]) - 1
            int_rate_lat = self._int_rate_lattice[curr_level - 1]

            self._back_prop(
                from_level=curr_level,
                int_rate_lat=int_rate_lat,
                futures_mode=True
            )

    def print_lattice(self):
        for lattice_arr in self._lattice:
            print(lattice_arr)


class BondPricing(BackPropLattice):

    @property
    def coupon_rate(self):
        return self._coupon_rate

    @coupon_rate.setter
    def coupon_rate(self, val):
        self._coupon_rate = val

    @property
    def coupon_pay(self):
        return self.coupon_rate * self.payoff

    @property
    def payoff(self):
        return self._payoff

    @payoff.setter
    def payoff(self, val):
        self._payoff = val

    @property
    def _payoff_at_expiry(self):
        return self.payoff + self.coupon_pay

    @property
    def _int_rate_lattice(self):
        return self._ir_model.get_lattice

    @property
    def _latest_prc_states(self):
        return super()._curr_states 

    @property
    def _idx_latest_prc_states(self):
        return super()._idx_curr_states

    @property
    def _latest_int_rate_states(self):
        return self._int_rate_lattice[self._idx_latest_prc_states - 1]

    def __init__(
            self,
            n,
            q,
            ir_model,
            payoff,
            coupon_rate=0.0,
    ):
        super().__init__(
            n=n,
            q=q,
            val=self._payoff_at_expiry * np.ones(
                shape=(self.n + 1, 1)
            )
        )
        self.payoff = payoff
        self.coupon_rate = coupon_rate
        self._ir_model = ir_model

        self._generate_lattice()

    def print_lattice(self):
        for lattice_arr in self.lattice:
            print(lattice_arr)

    def _generate_lattice(self):
        # self._init_lattice(
        #     val=self._payoff_at_expiry * np.ones(
        #         shape=(self.n + 1, 1)
        #     )
        # )

        while len(self._lattice) < self.n + 1:
            self._back_prop(
                from_level=self._idx_latest_prc_states,
                int_rate_states=self._latest_int_rate_states,
            )

    def _init_lattice(
        self,
        val
        ):

        self._lattice = [
            val,

        ]

        self._set_up_backward_prop()

    def _back_prop(
            self,
            from_level,
            int_rate_states,
            futures_mode=False,
    ):

        if futures_mode:
            int_rat_mul = 1
        else:
            int_rat_mul = (int_rate_states + 1) ** -1

        _new_prc_states = self._gen_new_prc_states(
            int_rat_mul=int_rat_mul,
            from_level=from_level
        )
        self._back_push_lattice(
            val=_new_prc_states
        )

    def _gen_new_prc_states(
            self,
            int_rat_mul,
            from_level,
    ):
        return self._back_prop_op(
            from_level=from_level,
        ).dot(self._latest_prc_states) * int_rat_mul + self.coupon_pay

    def _back_prop_op(
            self,
            from_level
    ):
        return self._backprop_set_up.get_backward_prop_matrix(
            from_level=from_level
        )

    def _set_up_backward_prop(
            self,
            use_q=True,
    ):
        self._backprop_set_up = BackwardProp(
            q=self.q if use_q else 0.5,
        )

    def _back_push_lattice(
            self,
            val,
    ):
        self._lattice.insert(
            0,
            val
        )

    def _append_to_lattice(
            self,
            val
    ):
        self._lattice.append(
            val
        )
