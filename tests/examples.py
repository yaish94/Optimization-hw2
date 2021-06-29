import numpy as np

Q1 = np.array([[1, 0], [0, 1]])
Q2 = np.array([[5, 0], [0, 1]])
A = np.array([[np.sqrt(3) / 2, -0.5], [0.5, np.sqrt(3) / 2]])
Q3 = np.dot(np.dot(A.T, Q2), A)
a_t = np.array([1, 2])  ### create some vector

eq_constraints_mat_qp = np.array([[1.0, 1.0, 1.0]])
eq_constraints_rhs_qp = np.array([1.0])
zero_hessian1 = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
zero_hessian2 = np.array([[0.0, 0.0], [0.0, 0.0]])
objective_qp_hessian = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]])


def func_quadratic(x, q):
    return np.dot(np.dot(x.T, q), x)


def func_quadratic1(x, hessian=False):
    f0, f1 = func_quadratic(x, Q1), np.dot(2 * Q1, x)
    return (f0, f1) if not hessian else (f0, f1, 2 * Q1)


def func_quadratic2(x, hessian=False):
    f0, f1 = func_quadratic(x, Q2), np.dot(2 * Q2, x)
    return (f0, f1) if not hessian else (f0, f1, 2 * Q2)


def func_quadratic3(x, hessian=False):
    f0, f1 = func_quadratic(x, Q3), np.dot(2 * Q3, x)
    return (f0, f1) if not hessian else (f0, f1, 2 * Q3)


def func_rosenbrock(x, hessian=False):
    x1, x2 = x[0], x[1]
    f0, f1 = 100.0 * (x2 - x1 ** 2.0) ** 2.0 + (1 - x1) ** 2.0, \
             np.array([400 * (x1 ** 3.0) + 2 * x1 - 400 * x1 * x2 - 2, 200 * x2 - 200 * (x1 ** 2.0)])
    return (f0, f1) if not hessian else (f0, f1,
                                         np.array([[1200 * (x1 ** 2.0) - 400 * x2 + 2, - 400 * x1], [- 400 * x1, 200]]))


def objective_qp(x):
    x1, x2, x3 = x[0], x[1], x[2]
    return x1 ** 2.0 + x2 ** 2.0 + (x3 + 1.0) ** 2.0, np.array([2.0 * x1, 2.0 * x2, 2.0 * x3 + 2.0]), objective_qp_hessian


def c1_qp(x):
    return -x[0], np.array([-1.0, 0.0, 0.0]), zero_hessian1


def c2_qp(x):
    return -x[1], np.array([0.0, -1.0, 0.0]), zero_hessian1


def c3_qp(x):
    return -x[2], np.array([0.0, 0.0, -1.0]), zero_hessian1


def objective_lp(x):
    x1, x2 = x[0], x[1]
    return -x1 - x2, np.array([-1.0, -1.0]), zero_hessian2


def c1_lp(x):
    return -x[0] - x[1] + 1.0, np.array([-1.0, -1.0]), zero_hessian2


def c2_lp(x):
    return x[1] - 1, np.array([0.0, 1.0]), zero_hessian2


def c3_lp(x):
    return x[0] - 2, np.array([1.0, 0.0]), zero_hessian2


def c4_lp(x):
    return -x[1], np.array([0.0, -1.0]), zero_hessian2


def func_linear(x, hessian=False):
    return np.dot(a_t, x), np.array([1, 5])
