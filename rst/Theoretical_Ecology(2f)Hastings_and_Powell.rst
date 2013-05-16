Overview
========

A simple script that recreates the min/max bifurcation diagrams from
Hastings and Powell 1991.

Library Functions
=================

Two useful functions are defined in the module bif.py.



.. code-block:: python

    import numpy
    
    def window(data, size):
        """A generator that returns the moving window of length
        `size` over the `data`
    
        """
        for start in range(len(data) - (size - 1)):
            yield data[start:(start + size)]
    
    
    def min_max(data, tol=1e-14):
        """Return a list of the local min/max found
        in a `data` series, given the relative tolerance `tol`
    
        """
        maxes = []
        mins = []
        for first, second, third in window(data, size=3):
            if first < second and third < second:
                maxes.append(second)
            elif first > second and third > second:
                mins.append(second)
            elif abs(first - second) < tol and abs(second - third) < tol:
                # an equilibrium is both the maximum and minimum
                maxes.append(second)
                mins.append(second)
    
        return {'max': numpy.asarray(maxes),
                'min': numpy.asarray(mins)}
    



The Model
=========

For speed the model is defined in a fortran file and compiled into a
library for use from python. Using this method gives a 100 fold increase
in speed. The file uses Fortran 90, which makes using f2py especially
easy. The file is named hastings.f90.



.. code-block:: python

    module model
        implicit none
    
        real(8), save :: a1, a2, b1, b2, d1, d2
    
    contains 
    
        subroutine fweb(y, t, yprime)
            real(8), dimension(3), intent(in) :: y
            real(8), intent(in) :: t
            real(8), dimension(3), intent(out) :: yprime
    
            yprime(1) = y(1)*(1.0d0 - y(1)) - a1*y(1)*y(2)/(1.0d0 + b1*y(1)) 
            yprime(2) = a1*y(1)*y(2)/(1.0d0 + b1*y(1)) - a2*y(2)*y(3)/(1.0d0 + b2*y(
    2)) - d1*y(2)
            yprime(3) = a2*y(2)*y(3)/(1.0d0 + b2*y(2)) - d2*y(3)
        end subroutine fweb
    
    end module model
    



Which is compiled (using the gfortran compiler) with the command:



.. code-block:: python

    
    = The Script =
    {{{#!python
    import numpy
    from scipy.integrate import odeint
    import bif
    
    import hastings
    
    # setup the food web parameters
    hastings.model.a1 = 5.0 
    hastings.model.a2 = 0.1 
    hastings.model.b2 = 2.0 
    hastings.model.d1 = 0.4 
    hastings.model.d2 = 0.01
    
    # setup the ode solver parameters
    t = numpy.arange(10000)
    y0 = [0.8, 0.2, 10.0]
    
    def print_max(data, maxfile):
        for a_max in data['max']:
            print >> maxfile, hastings.model.b1, a_max
    
    x_maxfile = open('x_maxfile.dat', 'w')
    y_maxfile = open('y_maxfile.dat', 'w')
    z_maxfile = open('z_maxfile.dat', 'w')
    for i, hastings.model.b1 in enumerate(numpy.linspace(2.0, 6.2, 420)):
        print i, hastings.model.b1
        y = odeint(hastings.model.fweb, y0, t)
    
        # use the last 'stationary' solution as an intial guess for the
        # next run. This both speeds up the computations, as well as helps
        # make sure that solver doesn't need to do too much work.
        y0 = y[-1, :]
    
        x_minmax = bif.min_max(y[5000:, 0]) 
        y_minmax = bif.min_max(y[5000:, 1]) 
        z_minmax = bif.min_max(y[5000:, 2]) 
    
        print_max(x_minmax, x_maxfile)
        print_max(y_minmax, y_maxfile)
        print_max(z_minmax, z_maxfile)
    





