"""Some examples of using the functions im optimize.py"""

import numpy as np; np.seterr(all='raise')
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.linalg import solve

from optimize import steepest_descent, backtracking_line_search,\
    AdaptiveLineSearch

plt.rc('text', usetex=True)
plt.rc('font', family='serif')


def example_gradient_descent001():
    """Compute the analytic center of a randomly generated
    polyhedra in dimension n, subject to the additional constraint
    that x_i \in [-1, 1] for each i"""

    # ---------------Generate some problem data-----------------
    m, n = 1050, 450  # (number of hyperplanes), (dimension)
    A = np.random.normal(size=(m, n))
    b = np.random.gamma(shape=3.0, size=(m,))  # b > 0 ensures feasibility
    x0 = np.zeros(n)
    assert np.all(np.dot(A, x0) < b)

    # ---------------Some helper functions----------------------
    def _psi(x):
        '''-log(1 + x) - log(1 - x)'''
        if np.any(np.abs(x) > 1):
            return np.inf
        else:
            return -(np.log1p(x) + np.log1p(-x))

    def _phi(x):
        '''-log(-x)'''
        if np.any(x > 0):
            return np.inf
        else:
            return -np.sum(np.log(-x))

    def _grad_phi(x):
        return -1. / x

    # ------------- the actual functions we need to do gradient descent -------
    def f(x):  # The function to minimize
        return np.sum(_psi(x)) + _phi(np.dot(A, x) - b)

    # The gradient of f
    def grad_f(x):
        return (1 / (1 - x)) - (1 / (1 + x)) + np.dot(A.T, _grad_phi(
            np.dot(A, x) - b))

    # The descent direction for gradient descent
    def f_dd(x):
        return -grad_f(x)

    def stopping_criteria_closure(eps=1e-5):
        check_count, check_max = 0, 10

        def stopping_criteria(x):
            nonlocal check_count
            check_count += 1
            if check_count >= check_max:
                check_count = 0
                if np.linalg.norm(grad_f(x), ord=np.inf) < eps:
                    return True
            return False
        return stopping_criteria

    def line_search_closure():
        ASL = AdaptiveLineSearch(grad_f)

        def line_search(x, x_dd):
            # Pre compute Ax and Ax_dd
            Ax = np.dot(A, x)
            Ax_dd = np.dot(A, x_dd)

            # Create a function to evaluate f(x + tx_dd) efficiently
            def g(t):
                return np.sum(_psi(x + t * x_dd)) + _phi(Ax + t * Ax_dd - b)

            # Return the step size from the line search
            return ASL.line_search(x, x_dd, g)

        return line_search

    # Store the execution history of the function for plotting
    def callback_closure():
        fx_list = []
        grad_fx_list = []

        def callback(x, n, x_dd, ss):
            nonlocal grad_fx_list
            nonlocal fx_list
            grad_fx_list.append(np.linalg.norm(grad_f(x), ord=np.inf))
            fx_list.append(f(x))
            return
        return callback, grad_fx_list, fx_list

    fig, axes = plt.subplots(2, 1, sharex=True)
    fig.suptitle('Gradient Descent for Polyhedron Analytic Centering')
    ax_grad, ax_f = axes
    ax_grad.set_ylabel(r'$||\nabla f(x^{(n)})||_\infty$')
    ax_grad.set_title(r'$n = %d$ dimensions, $m = %d$ hyperplanes'
                      % (n, m))
    ax_grad.set_yscale('log')
    ax_grad.grid()

    ax_f.set_ylabel(r'$f(x^{(n)}) - p^\star$')
    ax_f.set_xlabel('iteration count $n$')
    ax_f.set_yscale('log')
    ax_f.grid()

    for i, eps in enumerate([1e-7, 1e-6, 1e-3, 1e-1]):
        line_search = line_search_closure()
        callback, grad_fx_list, fx_list = callback_closure()
        stopping_criteria = stopping_criteria_closure(eps)

        # Call the optimizer
        x_star, n_iter, success = steepest_descent(x0, f_dd,
                                                   stopping_criteria,
                                                   line_search,
                                                   maxiter=2000,
                                                   callback=callback)
        if i == 0:
            p_star = fx_list[-1]

        # Plot the results
        ax_grad.plot(grad_fx_list, linewidth=2,
                     label=r'$\epsilon = %.2E$' % eps)
        ax_f.plot(np.array(fx_list[:-1]) - p_star,
                  linewidth=2, label=r'$\epsilon = %.2E$' % eps)

    ax_grad.legend()
    ax_f.legend()
    fig.savefig('./images/analytic_center_gradient_descent.png')
    plt.show()
    return


def example_newton001():
    """Compute the analytic center of a randomly generated
    polyhedra in dimension n, subject to the additional constraint
    that x_i \in [-1, 1] for each i"""

    # ---------------Generate some problem data-----------------
    m, n = 1050, 550  # (number of hyperplanes), (dimension)
    A = np.random.normal(size=(m, n))
    b = np.random.gamma(shape=3.0, size=(m,))  # b > 0 ensures feasibility
    x0 = np.zeros(n)
    assert np.all(np.dot(A, x0) < b)  # {x | Ax < b} is the polyhedron

    # ---------------Some helper functions----------------------
    def _psi(x):
        '''-log(1 + x) - log(1 - x)'''
        if np.any(np.abs(x) > 1):
            return np.inf
        else:
            return -(np.log1p(x) + np.log1p(-x))

    def _psi_pp(x):
        '''2nd derivative of _psi'''
        return (1 / (1 - x**2))**2  # This could be bad for numerical precision

    def _phi(x):
        '''-log(-x)'''
        if np.any(x > 0):
            return np.inf
        else:
            return -np.sum(np.log(-x))

    def _grad_phi(x):
        # The gradient is undefined when x = 0 and rightfully throws an error
        return -1. / x

    # ------------- the actual functions we need to do gradient descent -------
    def f(x):  # The function to minimize
        return np.sum(_psi(x)) + _phi(np.dot(A, x) - b)

    # The gradient of f
    def grad_f(x):
        return (1 / (1 - x)) - (1 / (1 + x)) + np.dot(A.T, _grad_phi(
            np.dot(A, x) - b))

    # The Hessian of f -- This is NOT Lipschitz near x = 1, and hence we
    # should not necessarily expect quadratic convergence.
    def hess_f(x):
        H = _grad_phi(np.dot(A, x) - b)**2  # Hessian of Phi
        HA = H[:, None] * A
        ATHA = np.dot(A.T, HA)
        dg = np.arange(ATHA.shape[0])
        ATHA[dg, dg] += (1 / (1 - x**2))**2
        return ATHA

    def fdd_stopping_closure(eps=1e-5):
        x_dd = x0
        g = grad_f(x0)

        # The descent direction for Newton's method
        def f_dd(x):
            nonlocal x_dd, g
            g = grad_f(x)
            H = hess_f(x)
            x_dd = solve(H, -g, sym_pos=True, check_finite=False)
            return x_dd

        def stopping_criteria(x):
            lmbda = 0.5 * np.dot(g, -x_dd)**0.5  # Newton decrement
            if lmbda <= eps:
                return True
            return False
        return f_dd, stopping_criteria

    def line_search_closure():
        ASL = AdaptiveLineSearch(grad_f, beta0=0.8)
        undamped = False

        def line_search(x, x_dd):
            # Pre compute Ax and Ax_dd
            nonlocal undamped
            if not undamped:
                Ax = np.dot(A, x)
                Ax_dd = np.dot(A, x_dd)

                # Create a function to evaluate f(x + tx_dd) efficiently
                def g(t):
                    return np.sum(_psi(x + t * x_dd)) +\
                        _phi(Ax + t * Ax_dd - b)

                # Return the step size from the line search
                t = ASL.line_search(x, x_dd, g)
            return t

        return line_search

    # Store the execution history of the function for plotting
    def callback_closure():
        fx_list = []
        lmbda_list = []

        def callback(x, n, x_dd, ss):
            nonlocal lmbda_list
            nonlocal fx_list
            lmbda_list.append(0.5 * np.dot(grad_f(x), -x_dd)**0.5)
            fx_list.append(f(x))
            return
        return callback, lmbda_list, fx_list

    fig, axes = plt.subplots(2, 1, sharex=True)
    fig.suptitle("Newton's method for Polyhedron Analytic Centering")
    ax_lmbda, ax_f = axes

    ax_lmbda.set_title(r'$n = %d$ dimensions, $m = %d$ hyperplanes'
                       % (n, m))
    ax_lmbda.set_ylabel(r'$\lambda(x) / 2$')
    ax_lmbda.set_yscale('log')
    ax_lmbda.grid()

    ax_f.set_ylabel(r'$f(x^{(n)}) - p^\star$')
    ax_f.set_xlabel('iteration count $n$')
    ax_f.set_yscale('log')
    ax_f.grid()

    for i, eps in enumerate([1e-12, 1e-9, 1e-6, 1e-3, 1e-1]):
        line_search = line_search_closure()
        callback, lmbda_list, fx_list = callback_closure()
        f_dd, stopping_criteria = fdd_stopping_closure(eps)

        # Call the optimizer
        x_star, n_iter, success = steepest_descent(x0, f_dd,
                                                   stopping_criteria,
                                                   line_search,
                                                   maxiter=50,
                                                   callback=callback)
        if i == 0:
            p_star = fx_list[-1]

        # Plot the results
        ax_lmbda.plot(lmbda_list[:-1], linewidth=2,
                      label=r'$\epsilon = %.2E$' % eps)
        ax_f.plot(np.array(fx_list[:-1]) - p_star,
                  linewidth=2, label=r'$\epsilon = %.2E$' % eps)

    ax_lmbda.legend()
    ax_f.legend()
    fig.savefig('./images/analytic_center_newton.png')
    plt.show()
    return


def example_newton002():
    """
    Minimize f(x) = <c, x> + sum_j log(b_j - <a_j, x>) + sum_i(x_i),
    Which is a log barrier with a linear objective.
    """

    # ---------------Generate some problem data-----------------
    m, n = 750, 800  # (number of hyperplanes), (dimension)
    A = np.random.normal(size=(m, n))
    b = np.random.gamma(shape=3.0, size=(m,))  # b > 0 ensures feasibility
    c = np.random.gamma(shape=3.0, size=(n,))  # c > 0 ensures boundedness
    x0 = np.zeros(n)
    assert np.all(np.dot(A, x0) < b)  # {x | Ax < b} is the polyhedron

    # ---------------Some helper functions----------------------
    def _phi(x):
        '''-log(1 + x)'''
        if any(x < -1):
            return np.inf
        else:
            return -np.sum(np.log1p(x))

    def _psi(x):
        '''-log(-x)'''
        if np.any(x > 0):
            return np.inf
        else:
            return -np.sum(np.log(-x))

    def _grad_psi(x):
        # The gradient is undefined when x = 0 and rightfully throws an error
        return -1. / x

    def _grad_phi(x):
        return -1. / (1 + x)

    # ------------- the actual functions we need to do gradient descent -------
    def f(x):  # The function to minimize
        return np.dot(c, x) + _phi(x) + _psi(np.dot(A, x) - b)

    # The gradient of f
    def grad_f(x):
        return c + _grad_phi(x) + np.dot(A.T, _grad_psi(np.dot(A, x) - b))

    def hess_f(x):
        # H is diagonal and is returned as a vector only
        D = _grad_psi(np.dot(A, x) - b)**2
        HA = D[:, None] * A
        ATHA = np.dot(A.T, HA)
        dg = np.arange(ATHA.shape[0])
        ATHA[dg, dg] = ATHA[dg, dg] + _grad_phi(x)**2
        return ATHA

    def fdd_stopping_closure(eps=1e-5):
        x_dd = x0
        g = grad_f(x0)

        # The descent direction for Newton's method
        def f_dd(x):
            nonlocal x_dd, g
            g = grad_f(x)
            H = hess_f(x)
            x_dd = solve(H, -g, sym_pos=True, check_finite=False)
            return x_dd

        def stopping_criteria(x):
            lmbda = 0.5 * np.dot(g, -x_dd)**0.5  # Newton decrement
            if lmbda <= eps:
                return True
            return False
        return f_dd, stopping_criteria

    def line_search_closure():
        ASL = AdaptiveLineSearch(grad_f, beta0=0.8)
        undamped = False

        def line_search(x, x_dd):
            # Pre compute Ax and Ax_dd
            nonlocal undamped
            if not undamped:
                Ax = np.dot(A, x)
                Ax_dd = np.dot(A, x_dd)
                cx = np.dot(c, x)
                cx_dd = np.dot(c, x_dd)

                # Create a function to evaluate f(x + tx_dd) efficiently
                def g(t):
                    return (cx + t * cx_dd) +\
                        _phi(x + t * x_dd) +\
                        _psi(Ax + t * Ax_dd - b)

                # Return the step size from the line search
                t = ASL.line_search(x, x_dd, g)
            return t

        return line_search

    # Store the execution history of the function for plotting
    def callback_closure():
        fx_list = []
        lmbda_list = []

        def callback(x, n, x_dd, ss):
            nonlocal lmbda_list
            nonlocal fx_list
            lmbda_list.append(0.5 * np.dot(grad_f(x), -x_dd)**0.5)
            fx_list.append(f(x))
            return
        return callback, lmbda_list, fx_list

    fig, axes = plt.subplots(2, 1, sharex=True)
    fig.suptitle("Newton's method for barrier LP")
    ax_lmbda, ax_f = axes

    ax_lmbda.set_title(r'$n = %d$ dimensions, $m = %d$ hyperplanes'
                       % (n, m))
    ax_lmbda.set_ylabel(r'$\lambda(x) / 2$')
    ax_lmbda.set_yscale('log')
    ax_lmbda.grid()

    ax_f.set_ylabel(r'$f(x^{(n)}) - p^\star$')
    ax_f.set_xlabel('iteration count $n$')
    ax_f.set_yscale('log')
    ax_f.grid()

    for i, eps in enumerate([1e-12, 1e-7, 1e-6, 1e-3, 1e-1]):
        line_search = line_search_closure()
        callback, lmbda_list, fx_list = callback_closure()
        f_dd, stopping_criteria = fdd_stopping_closure(eps)

        # Call the optimizer
        x_star, n_iter, success = steepest_descent(x0, f_dd,
                                                   stopping_criteria,
                                                   line_search,
                                                   maxiter=50,
                                                   callback=callback)
        if i == 0:
            p_star = fx_list[-1]

        # Plot the results
        ax_lmbda.plot(lmbda_list[:-1], linewidth=2,
                      label=r'$\epsilon = %.2E$' % eps)
        ax_f.plot(np.array(fx_list[:-1]) - p_star,
                  linewidth=2, label=r'$\epsilon = %.2E$' % eps)

    ax_lmbda.legend()
    ax_f.legend()
    fig.savefig('./images/LP_barrier.png')
    plt.show()
    return


if __name__ == '__main__':
    example_gradient_descent001()
    example_newton001()
    example_newton002()
