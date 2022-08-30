import numpy as np
from abc import ABC


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
