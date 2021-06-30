import numpy as np

from src.unconstrained_min import newton_dir, _step_length_wolfe_condition


"""
interior_pt minimizes the function func subject to the:
* ineq_constraints - list of inequality constraints, 
* matrix eq_constraints_mat, eq_constraints_rhs- affine equality constraints
* The outer iterations start at x0
"""
def interior_pt(func, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, x0, eps=1e-3):
    t = 1
    m = 10
    x = x0
    total_x_history = None

    while True:
        # log_barrier
        # definition of function - f0, gradient - f1 and hessian - f2
        f0 = lambda a: t * func(a)[0] + phi(a, ineq_constraints)
        f1 = lambda a: t * func(a)[1] + np.add.reduce([(1 / -f(a)[0]) * f(a)[1] for f in ineq_constraints])
        f2 = lambda a: t * func(a)[2] + np.add.reduce(
            [(1 / (f(a)[0] ** 2.0)) * np.outer(f(a)[1], f(a)[1].T) + (1 / -f(a)[0]) * f(a)[2] for f in
             ineq_constraints])
        objective = lambda a, hessian=False: (f0(a), f1(a)) if not hessian else (
            f0(a), f1(a), f2(a))

        if eq_constraints_mat is None:
            # when there are no equality constraints, use newton method for unconstrained problems
            x, x_history = newton_dir(objective, x, 1e-12, 1e-8, 100, 1.0, 1e-4, 0.2)
        else:
            x, x_history = newton_constrained(objective, eq_constraints_mat, eq_constraints_rhs, x, eps)

        total_x_history = x_history if total_x_history is None else np.vstack((total_x_history, x_history))

        if m / t < eps:
            print("last iteration index over all", len(total_x_history)-2) 
            return x, total_x_history
        t = m * t


def newton_constrained(func, eq_constraints_mat, eq_constraints_rhs, x0, eps):
    x_history = x0
    x = x0
    A = eq_constraints_mat
    A_t = eq_constraints_mat.T

    while True:
        hes = func(x, True)[2]
        grad = np.append(-func(x)[1], np.zeros(1))
        large_matrix = np.vstack((np.hstack([hes, A_t]), np.hstack([A, np.zeros((1, 1))])))
        p_k = np.linalg.lstsq(large_matrix, grad, rcond=None)[0][0:-1]
        termination_criteria = 0.5 * np.dot(np.dot(p_k.T, hes), p_k)
        if termination_criteria < eps:
            _print_summary(abs(func(x)[0]-func(x_history[-2])[0]), abs(x_history[-2]-x), func(x)[0], x)
            return x, x_history
        alpha = _step_length_wolfe_condition(1.0, 1e-4, 0.2, func, x, p_k)
        x = x + alpha * p_k
        x_history = np.vstack((x_history, x))


def phi(x, ineq_constraints):
    r = 0
    for f in ineq_constraints:
        fi = f(x)[0]
        if fi >= 0:
            r -= - 1 / .0000001
        else:
            r -= np.log(-fi)
    return r

def _print_summary(diff_f, diff_x, f_prev, x_prev):
    print("Last x Location: ", x_prev)
    print("Last step length: ", diff_x)
    print("Last function value: ", f_prev)
    print("Last function change: ", diff_f)
