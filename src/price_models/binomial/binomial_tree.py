import numpy as np
from abc import ABC

# from src.price_models.binomial.util_matrix import BackwardProp
from src.price_models.binomial.util_matrix import BackwardProp
# from src.price.mutil_matrix import BackwardProp

class BackPropLattice(ABC):
    
    @property
    def n(self):
        return self._n

    @n.setter
    def n(self, val):
        self._n = val

    @property
    def q(self):
        return self._q

    @q.setter
    def q(self, val):
        self._q = val
    
    @property
    def lattice(self):
        return self._lat

    @lattice.setter
    def lattice(self, val):
        self._lat = val

    @property
    def _curr_states(self):
        return self.lattice[0]

    @property
    def _idx_curr_states(self):
        return len(self._curr_states) - 1
    
    def _init_lattice(self, val):
        self.lattice = [
            val,
        ]
    
    def _back_push_lattice(
            self,
            val,
    ):
        self.lattice.insert(
            0,
            val
        )

    def _set_up_backward_prop(
            self,
            use_q=True,
    ):
        self._backprop_set_up = BackwardProp(
            q=self.q if use_q else 0.5,
        )
    def __init__(
        self,
        n,
        q,
        val,
        ) -> None:

        self.n = n
        self.q = q
        self._init_lattice(
            val = val
        )


class BinomialTree(ABC):
    @property
    def n(self):
        """
        n: n period binomial model defined here
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
    def tree(self):
        """
        tree: binomial tree for price modelling
        getter method for tree
        """
        return self._tree

    @tree.__init__
    def tree(self, val):
        """
        setter method for tree
        """
        self._tree = val

    def print_tree(self):
        """
        print out tree levels
        """
        for lattice_level in self.tree:
            print(
                str(
                    lattice_level
                        .tolist()
                )
            )

    def __init__(
            self,
            n,
    ):
        """
        n: n period binomial model defined here.

        class constructor for Binomial Tree
        """

        self.n = n
        self._tree = [
            np.zeros(
                shape=(i + 1, 1)
            )
            for i in range(n)
        ]
