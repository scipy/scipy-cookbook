SciPy's optimization package is scipy.optimize. The most basic
non-linear optimization functions are: \*optimize.fmin(func, x0), which
finds the minimum of f(x) starting x with x0 (x can be a vector)
\*optimize.fsolve(func, x0), which finds a solution to func(x) = 0
starting with x = x0 (x can be a vector) \*optimize.fminbound(func, x1,
x2), which finds the minimum of a scalar function func(x) for the range
[x1,x2] (x1,x2 must be a scalar and func(x) must return a scalar) See
the `scipy.optimze
documentation <http://docs.scipy.org/doc/scipy/reference/optimize.html>`__
for details.

This is a quick demonstration of generating data from several Bessel
functions and finding some local maxima using fminbound. This uses
ipython with the -pylab switch.



.. code-block:: python

    from scipy import optimize, special
    from numpy import *
    from pylab import *
    
    x = arange(0,10,0.01)
    
    for k in arange(0.5,5.5):
         y = special.jv(k,x)
         plot(x,y)
         f = lambda x: -special.jv(k,x)
         x_max = optimize.fminbound(f,0,6)
         plot([x_max], [special.jv(k,x_max)],'ro')
    
    title('Different Bessel functions and their local maxima')
    show()
    







.. code-block:: python

    #!figure
    #class left
    .. image:: OptimizationDemo1_attachments/NumPyOptimizationSmall.png
    
    Optimization Example
    



--------------

CategoryCookbook

