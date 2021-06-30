import unittest

import matplotlib.pyplot as plt
import numpy as np

from src.constrained_min import interior_pt
from src.utils import plot_feasible_region_lp, plot_feasible_region_qp
from tests.examples import eq_constraints_mat_qp, eq_constraints_rhs_qp, objective_qp, objective_lp, c1_qp, c3_qp, \
    c2_qp, \
    c1_lp, c2_lp, c3_lp, c4_lp


class TestConstrainedMin(unittest.TestCase):

    def setUp(self):
        plt.cla()

    """
     quadratic programming example
    """
    def test_qp(self):
        x, history = interior_pt(objective_qp, np.array([c1_qp, c2_qp, c3_qp]),
                                 eq_constraints_mat_qp, eq_constraints_rhs_qp,
                                 np.array([0.1, 0.2, 0.7]))

        plot_feasible_region_qp(history)


    """
     linear programming example
    """
    def test_lp(self):
        x, history = interior_pt(objective_lp, np.array([c1_lp, c2_lp, c3_lp, c4_lp]),
                                 None, None,
                                 np.array([0.5, 0.75]))

        plot_feasible_region_lp(history)
