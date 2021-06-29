import numpy as np

from src.unconstrained_min import line_search, gradient_descent
from src.utils import plot_outlines
from tests.examples import func_quadratic1, func_quadratic3, func_quadratic2, func_rosenbrock, func_linear
import unittest
import matplotlib.pyplot as plt

line_search_types = ["gd", "bfgs", "nt"]


class TestUnconstrainedMin(unittest.TestCase):

    def setUp(self):
        plt.cla()

    def test_quad_min1(self):
        for type in line_search_types:
            test_name = "Quadratic 1 " + type
            print(test_name)

            x0 = np.array([1.0, 1.0])
            x, x_history = line_search(f=func_quadratic1, x0=x0, step_size=0.1,
                                       max_iter=100, obj_tol=1e-12, param_tol=1e-8,
                                       dir_selection_method=type)

            plot_outlines(func_quadratic1, x, x0, x_history, test_name)

    def test_quad_min2(self):
        for type in line_search_types:
            test_name = "Quadratic 2 " + type
            print(test_name)

            x0 = np.array([1.0, 1.0])
            x, x_history = line_search(f=func_quadratic2, x0=x0, step_size=0.1, max_iter=100,
                                       obj_tol=1e-12,
                                       param_tol=1e-8, dir_selection_method=type)
            plot_outlines(func_quadratic2, x, x0, x_history, test_name)

    def test_quad_min3(self):

        for type in line_search_types:
            test_name = "Quadratic 3 " + type
            print(test_name)

            x0 = np.array([1.0, 1.0])
            x, x_history = line_search(f=func_quadratic3, x0=x0, step_size=0.1, max_iter=100,
                                       obj_tol=1e-12,
                                       param_tol=1e-8, dir_selection_method=type)
            plot_outlines(func_quadratic3, x, x0, x_history, test_name)

    def test_rosenbrock_min(self):
        for type in line_search_types:
            test_name = "Rosenbrock " + type
            print(test_name)

            x0 = np.array([2.0, 2.0])
            x, x_history = line_search(f=func_rosenbrock, x0=x0, step_size=0.001, max_iter=10000,
                                       obj_tol=1 / np.power(10, 7),
                                       param_tol=1 / np.power(10, 8), dir_selection_method=type)

            plot_outlines(func_rosenbrock, x, x0, x_history, test_name)


    # def test_lin_min(self):
    #     print("Linear ")
    #
    #     x,x_history = gradient_descent(func_linear, x0=np.array([1.0, 1.0]), step_size=0.1, max_iter=100, obj_tol=1e-12,
    #                      param_tol=1e-8)
    #
    #     plot_outlines(func_linear, x, np.array([1.0, 1.0]), x_history, "Linear")

