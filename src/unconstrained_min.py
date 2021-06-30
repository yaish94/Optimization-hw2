import numpy as np


"""
general method that minimizes unconstrained functions
"""
def line_search(f, x0, step_size, obj_tol, param_tol, max_iter, dir_selection_method,
                init_step_len=1.0, slope_ratio=1e-4, back_track_factor=0.2):
    if dir_selection_method == "gd":
        return gradient_descent(f, x0, step_size, obj_tol, param_tol, max_iter)
    elif dir_selection_method == "bfgs":
        return bfgs_dir(f, x0, obj_tol, param_tol, max_iter, init_step_len, slope_ratio, back_track_factor)
    elif dir_selection_method == "nt":
        return newton_dir(f, x0, obj_tol, param_tol, max_iter, init_step_len, slope_ratio, back_track_factor)
    return None


"""
gradient descent algorithm
"""
def gradient_descent(f, x0, step_size, obj_tol, param_tol, max_iter):
    x_prev = x0
    f_prev, df_prev = f(x0, False)

    x_history = x_prev
    f_history = f_prev

    i = 0
    diff_x = param_tol + 1
    diff_f = obj_tol + 1

    while i < max_iter and diff_x > param_tol and diff_f > obj_tol:
        x_next = x_prev - step_size * df_prev
        f_next, df_next = f(x_next)

        diff_x = np.linalg.norm(x_next - x_prev)  # Change in x
        diff_f = abs(f_next - f_prev)  # Change in f

        x_history = np.vstack((x_history, x_next))
        f_history = np.vstack((f_history, f_next))

        x_prev = x_next
        f_prev = f_next
        df_prev = df_next
        i = i + 1  # iteration count

    success = not (diff_x > param_tol and diff_f > obj_tol)
    _print_summary(diff_f, diff_x, f_prev, i, x_prev, success)

    return x_next, x_history


"""
bfgs direction algorithm
"""
def bfgs_dir(f, x0, obj_tol, param_tol, max_iter, init_step_len, slope_ratio, back_track_factor):
    x_prev = x0
    f_prev, df_prev = f(x0)
    x_len = x0.shape[0]
    I = np.eye(len(x0), dtype=int)  ## identity matrix
    B_k = I

    x_history = x_prev
    f_history = f_prev
    i = 0

    diff_x = param_tol + 1
    diff_f = obj_tol + 1

    while i < max_iter and np.linalg.norm(df_prev) > obj_tol and diff_x > param_tol and diff_f > obj_tol:
        p_k = -np.linalg.solve(B_k, df_prev)

        alpha = _step_length_wolfe_condition(init_step_len, slope_ratio, back_track_factor, f, x_prev, p_k)

        x_next = x_prev + alpha * p_k
        f_next, df_next = f(x_next)
        diff_x = np.linalg.norm(x_next - x_prev)  # Change in x
        diff_f = abs(f_next - f_prev)  # Change in f

        s_k = x_next - x_prev  # Change in x
        y_k = df_next - df_prev

        y_t = y_k.reshape([x_len, 1])
        B_s = np.dot(B_k, s_k)
        s_t_B = np.dot(s_k, B_k)
        sBs = np.dot(np.dot(s_k, B_k), s_k)

        B_k = B_k + y_t * y_k / np.dot(y_k, s_k) - B_s.reshape([x_len, 1]) * s_t_B / sBs

        x_history = np.vstack((x_history, x_next))
        f_history = np.vstack((f_history, f_next))

        x_prev = x_next
        f_prev = f_next
        df_prev = df_next
        i = i + 1  # iteration count

    success = not (np.linalg.norm(df_prev) > obj_tol and diff_x > param_tol and diff_f > obj_tol)
    _print_summary(diff_f, diff_x, f_prev, i, x_prev, success)

    return x_next, x_history


"""
newton direction algorithm
"""
def newton_dir(f, x0, obj_tol, param_tol, max_iter, init_step_len, slope_ratio, back_track_factor):
    x_prev = x0
    f_prev, df_prev, B_k = f(x0, True)

    x_history = x_prev
    f_history = f_prev

    i = 0

    diff_x = param_tol + 1
    diff_f = obj_tol + 1

    while i < max_iter and np.linalg.norm(df_prev) > obj_tol and diff_x > param_tol and diff_f > obj_tol:
        p_k = -np.linalg.solve(B_k, df_prev)

        alpha = _step_length_wolfe_condition(init_step_len, slope_ratio, back_track_factor, f, x_prev, p_k)
        x_next = x_prev + alpha * p_k
        f_next, df_next, B_k = f(x_next, True)
        diff_x = np.linalg.norm(x_next - x_prev)  # Change in x
        diff_f = abs(f_next - f_prev)  # Change in f

        x_history = np.vstack((x_history, x_next))
        f_history = np.vstack((f_history, f_next))

        x_prev = x_next
        f_prev = f_next
        df_prev = df_next
        i = i + 1  # iteration count

    success = not (np.linalg.norm(df_prev) > obj_tol and diff_x > param_tol and diff_f > obj_tol)
    _print_summary(diff_f, diff_x, f_prev, i, x_prev, success)

    return x_next, x_history


def _step_length_wolfe_condition(init_step_len, slope_ratio, back_track_factor, f, x, p):
    f0, f1 = lambda a: f(a)[0], lambda a: f(a)[1]
    step = init_step_len
    while f0(x + step * p) > f0(x) + slope_ratio * step * (np.dot(f1(x).T, p)):
        step = step * back_track_factor
        if step < .0000001: return step # stop for very small values
    return step


def _print_summary(diff_f, diff_x, f_prev, i, x_prev, success):
    print("Last iteration index:", i - 1)
    print("Last x Location: ", x_prev)
    print("Last step length: ", diff_x)
    print("Last function value: ", f_prev)
    print("Last function change: ", diff_f)
    print("status: ", "success" if success else "failure")
