TableOfContents

Using B-splines in scipy.signal
===============================

Example showing how to use B-splines in scipy.signal to do
interpolation. The input points must be equally spaced to use these
routine.



.. code-block:: python

    from numpy import r_, sin
    from scipy.signal import cspline1d, cspline1d_eval
    
    x = r_[0:10]
    dx = x[1]-x[0]
    newx = r_[-3:13:0.1]  # notice outside the original domain 
    y = sin(x) 
    cj = cspline1d(y)
    newy = cspline1d_eval(cj, newx, dx=dx,x0=x[0]) 
    from pylab import plot, show
    plot(newx, newy, x, y, 'o') 
    show()
    



.. image:: Interpolation_attachments/interpolate_figure1.png

N-D interpolation for equally-spaced data
=========================================

The scipy.ndimage package also contains spline\_filter and
map\_coordinates which can be used to perform N-dimensional
interpolation for equally-spaced data. A two-dimensional example is
given below:



.. code-block:: python

    from scipy import ogrid, sin, mgrid, ndimage, array
    
    x,y = ogrid[-1:1:5j,-1:1:5j]
    fvals = sin(x)*sin(y)
    newx,newy = mgrid[-1:1:100j,-1:1:100j]
    x0 = x[0,0]
    y0 = y[0,0]
    dx = x[1,0] - x0
    dy = y[0,1] - y0
    ivals = (newx - x0)/dx
    jvals = (newy - y0)/dy
    coords = array([ivals, jvals])
    newf = ndimage.map_coordinates(fvals, coords)
    



To pre-compute the weights (for multiple interpolation results), you
would use



.. code-block:: python

    coeffs = ndimage.spline_filter(fvals)
    newf = ndimage.map_coordinates(coeffs, coords, prefilter=False)
    



.. image:: Interpolation_attachments/interpolate_figure2.png

Interpolation of an N-D curve
=============================

The scipy.interpolate packages wraps the netlib FITPACK routines
(Dierckx) for calculating smoothing splines for various kinds of data
and geometries. Although the data is evenly spaced in this example, it
need not be so to use this routine.



.. code-block:: python

    from numpy import arange, cos, linspace, pi, sin, random
    from scipy.interpolate import splprep, splev
    
    # make ascending spiral in 3-space
    t=linspace(0,1.75*2*pi,100)
    
    x = sin(t)
    y = cos(t)
    z = t
    
    # add noise
    x+= random.normal(scale=0.1, size=x.shape)
    y+= random.normal(scale=0.1, size=y.shape)
    z+= random.normal(scale=0.1, size=z.shape)
    
    # spline parameters
    s=3.0 # smoothness parameter
    k=2 # spline order
    nest=-1 # estimate of number of knots needed (-1 = maximal)
    
    # find the knot points
    tckp,u = splprep([x,y,z],s=s,k=k,nest=-1)
    
    # evaluate spline, including interpolated points
    xnew,ynew,znew = splev(linspace(0,1,400),tckp)
    
    import pylab
    pylab.subplot(2,2,1)
    data,=pylab.plot(x,y,'bo-',label='data')
    fit,=pylab.plot(xnew,ynew,'r-',label='fit')
    pylab.legend()
    pylab.xlabel('x')
    pylab.ylabel('y')
    
    pylab.subplot(2,2,2)
    data,=pylab.plot(x,z,'bo-',label='data')
    fit,=pylab.plot(xnew,znew,'r-',label='fit')
    pylab.legend()
    pylab.xlabel('x')
    pylab.ylabel('z')
    
    pylab.subplot(2,2,3)
    data,=pylab.plot(y,z,'bo-',label='data')
    fit,=pylab.plot(ynew,znew,'r-',label='fit')
    pylab.legend()
    pylab.xlabel('y')
    pylab.ylabel('z')
    
    pylab.savefig('splprep_demo.png')
    



.. image:: Interpolation_attachments/splprep_demo.png

