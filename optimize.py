"""
Some code implementing algorithms from Boyd's Convex Optimization, mostly for
my own learning and for fun.  These are toy implementations, but they
work and provide a reasonable degree of flexibility.

I have taken a more functional approach where the user needs to
define a few functions for computing the gradient, descent direction,
stopping criteria etc...  This makes the set up a lot more cumbersome,
but provides more flexibility.  Some of these functions may need to share
or save some state, which can be reasonably accomplished via function
closures.  The implication here is that the user needs to have some
view of the inner-workings of the descent method, since it's calling of
user functions may have side effects on the state.
"""

from numpy import dot as inner_product
from math import inf


class AdaptiveLineSearch:
    """
    An algorithm that wraps backtracking_line_search to automatically
    tune the parameters and store the step size from the previous run.
    The heuristics seek to obtain the largest possible step quickly while
    still providing a guarantee that the step size will remain feasible.
    """
    def __init__(self, grad_f, alpha0=0.25, beta0=0.5, maxiter0=5, t0=1.0):
        """
        Note that all of the parameters will be tuned automatically,
        the only necessary parameter is grad_f

        params:
          grad_f: The gradient of the objective function
            x |-> grad_f(x)
          alpha0 (float, optional): The initial value for alpha (expected
            decrease param), must be 0 < alpha < 0.5
          beta (float, optional): Initial beta (backtrack param), must
            satisfy 0 < beta < 1
          maxiter0 (int, optional): The initial max iterations for
            backtracking_line_search before declaring it has failed
          t0 (float, optional): Initial value for the step size.
        """
        self.grad_f = grad_f
        self.alpha = alpha0
        self.beta = beta0
        self.maxiter = maxiter0
        self.t0 = t0
        return

    def line_search(self, x, x_dd, g):
        """
        Perform the backtracking line search on g(t) = f(x + t * x_dd)

        params:
          x (np.array): The "current" point where we step from
          x_dd (np.array): The descent direction
          g (callable): Function which implements g(t) = f(x + t * x_dd)

        returns:
          t (float): The step size
        """
        while True:
            t, success = backtracking_line_search(g, self.grad_f, x, x_dd,
                                                  self.alpha, self.beta,
                                                  maxiter=self.maxiter,
                                                  t0=self.t0)
            # If it reaches the max iters it will return success False
            # In such a case, we must retry else the point we jump to
            # is likely to be outside of the domain of f.
            if not success:
                # If we fail, relax the success requirements
                # then try again from where we left off (t isn't reset)
                self.maxiter += 1
                self.beta *= 0.9
                self.alpha *= 0.9
            else:
                # If we do succeed, try to get a better point next time
                # by increasing alpha and beta and by increasing the
                # starting value of t.
                self.maxiter = max(3, self.maxiter - 1)
                self.t0 = min(1.0, t / 0.9)
                self.alpha = min(0.49, self.alpha / 0.9)
                self.beta = min(0.95, self.beta / 0.9)
                break
        return t


def backtracking_line_search(g, grad_f, x, x_dd, alpha=0.2, beta=0.5,
                             maxiter=inf, t0=1.0):
    """
    Implements a backtracking line search (Algorithm 9.2).  The interface
    is intended to provide flexibility for an efficient implementation
    of the function g(t) = f(x + a * x_dd)

    Larger values of alpha or beta correspond to more accurate searches.
    We must have 0 < alpha < 1/2, 0 < beta < 1 with usually 0.01 < alpha < 0.3,
    0.1 < beta < 0.8.

    while g(t) > g(0) + t * alpha * <grad_f(x), x_dd>:
      t = beta * t

    params:
      g (callable): The function f(x) restricted to the line (x, x_dd)
        signature: g(t) -> f(x + t * x_dd)
      grad_f (callable): The gradient function for f, only evaluated once.
        signature: grad_f(x) -> (np.array)
      x (np.array): The point (current point) we are searching from
      x_dd (np.array): The direction in which we are searching
      alpha (float): Tuning parameter, requires (0 < alpha < 1/2)
      beta (float): Tuning parameter, requires (0 < beta < 1)
      maxiter (int): Maximum number of iterations.  One must be careful
        about this parameter since the default value of inf could possibly
        lead to infinite looping.  On the other hand, if maxiter is small,
        the algorithm could diverge.  These issues can be particularly
        pernicious if the domain of g is not the whole space.
      t0 (float): The initial value of t to start at

    returns:
      (ss, success):
        ss (float): The step size chosen by the line search
        success (bool): Returns False if the maximum number of iterations
          was reached.  This is indicative of a bug.
    """
    success = True
    n, t = 0, t0
    desc = inner_product(grad_f(x), x_dd)  # descent amount for full descent
    g0 = g(0)
    while g(t) > g0 + alpha * t * desc:
        t = beta * t
        n += 1
        if n > maxiter:
            success = False
            break
    return (t, success)


def steepest_descent(x0, f_dd, stopping_criteria, line_search=None,
                     maxiter=100, callback=None):
    """
    Implements a general steepest descent algorithm for the function f
    starting from the point x0.  This is a rather general algorithm which
    supports gradient descent, coordinate descent, gradient conditioning,
    and Newton's method by simply providing different f_dd for computing
    the descent direction.

    Starting at x = x0
    for n = 0, 1, ...
      1. Compute descent direction x_dd = f_dd(x)
      2. Choose step size via ss = line_search(x, x_dd)
      2.5 callback(x, n, x_dd, ss)
      3. Update x = x + ss * x_dd
      4. Update n = n + 1
    while not stopping_criteria(x)

    params:
      x0 (np.array): The starting point
      f_dd (callable): The function to compute a descent direction
        signature: f_dd(x) -> (np.array)
      stopping_criteria (callable): Function which checks a stopping
        criteria.  Returns True if the criteria is met
        signature: stopping_criteria(x) -> (bool)
      line_search (callable, optional): A function for computing a step size.
        If None, a step size of 1 is used by default.
        signature: line_search(x, x_dd) -> ss
      maxiter (int, optional): The maximum number of iterations to perform
      callback (callable, optional): A function called at every iteration
        for the purposes of visualization or debugging.
        signature: callback(x, n, x_dd, ss) -> None

    returns:
      (x_star, n_iter, success):
        x_star (np.array): The point on which the algorithm terminated,
          this is nominally the minimizer of the function f.
        success (bool): True if the stopping criteria was met, otherwise False
        n_iter (int): The total number of iterations performed
    """
    success = False  # No success if maxiter is exceeded
    n, x, ss = 0, x0, 1.0
    while n < maxiter:
        x_dd = f_dd(x)
        if line_search is not None:
            ss = line_search(x, x_dd)
        if callback is not None:
            callback(x, n, x_dd, ss)
        x = x + ss * x_dd
        n += 1

        if stopping_criteria(x):
            success = True  # We satisfied the stopping criteria
            break

    return (x, n, success)
