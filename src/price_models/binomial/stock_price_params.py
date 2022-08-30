from src.price_models.binomial.binomial_tree import BinomialTree


class StockPriceParams(BinomialTree):

    @property
    def S0(self):
        """
        S0: stock spot price
        getter method for S0
        """
        return self._S0

    @S0.setter
    def S0(self, val):
        """
        setter method for S0
        """
        self._S0 = val

    @property
    def n(self):
        """
        n: number of periods in n-period binomial tree
        getter method for n
        """
        return self._n

    @n.setter
    def n(self, val):
        """
        setter method for n
        """
        self._n = val

    @property
    def u(self):
        """
        u: upward movement factor
        getter method for u
        """
        return self._u

    @u.setter
    def u(self, val):
        """
        setter method for u
        """
        self._u = val

    @property
    def d(self):
        """
        downward movement factor
        getter method for d
        """
        return self._d

    @d.setter
    def d(self, val):
        """
        setter method for d
        """
        self._d = val

    def __init__(
            self,
            S0,  # Spot price of the stock
            n,  # number of periods
            u,  # upward movement factor
            d,  # downward movement factor
    ):
        super().__init__(
            n=n
        )
        self.S0 = S0

        self.n = n

        self.u = u

        self.d = d
