This repo contains some implementations for doing (mostly convex) optimization.  The work is done mostly for my own learning purposes, but the examples here could conceivably be useful for others.

The basic descent algorithm for unconstrained minimization of a function f, with domain D, is simply

    Starting at x = x0 in D
    while not stopping_criteria(x)
      1. Compute descent direction x_dd
      2. Choose step size ss via line search
      3. Update x = x + ss * x_dd

By asking the user for the basic ingredients: a starting point, a function to choose the descent direction, a function which evaluates a stopping criteria, and a line search routine, this general algorithm can implement various methods including gradient descent, coordinate descent, and Newton's method.  This puts a greater burden on the user to provide these functions, but implies a greater degree of flexibility.  Some examples are provided in examples.py

Gradient descent for finding the analytic center of a polyhedron.

![alt tag](https://github.com/RJTK/convex_optimize/tree/master/images/analytic_center_gradient_descent.png)

Newton's method for the same problem as above.  There is no quadratic convergence in this case, which I believe is due to the fact that the Hessian (involving 1/x terms) is not globally Lipschitz.

![alt tag](https://github.com/RJTK/convex_optimize/tree/master/images/analytic_center_newton.png)

This problem is very similar as above, except there is a linear cost <c, x> in addition to the log barrier log(b - Ax).  In this case, we get quadratic convergence.

![alt tag](https://github.com/RJTK/convex_optimize/tree/master/images/LP_barrier.png)
