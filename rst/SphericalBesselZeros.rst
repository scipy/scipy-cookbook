It may be useful to find out the zeros of the spherical Bessel
functions, for instance, if you want to compute the eigenfrequencies of
a spherical electromagnetic cavity (in this case, you'll need also the
zeros of the derivative of (r\*Jn(r))).

The problem is that you have to work out the ranges where you are
supposed to find the zeros.

Happily, the range of a given zero of the n'th spherical Bessel
functions can be computed from the zeros of the (n-1)'th spherical
Bessel function.

Thus, the approach proposed here is recursive, knowing that the
spherical Bessel function of order 0 is equal to sin(r)/r, whose zeros
are well known.

This approach is obviously not efficient at all, but it works ;-).

A sample example is shown, for the 10 first zeros of the spherical
Bessel function of order 5 (and the derivative of (r\*J5(r))), using
[:Cookbook/Matplotlib: matplotlib].



.. code-block:: python

    #! /usr/bin/env python
    
    ### recursive method: computes zeros ranges of Jn(r,n) from zeros of Jn(r,n-1)
    ### (also for zeros of (rJn(r,n))')
    ### pros : you are certain to find the right zeros values;
    ### cons : all zeros of the n-1 previous Jn have to be computed;
    ### note : Jn(r,0) = sin(r)/r
    
    from scipy import arange, pi, sqrt, zeros
    from scipy.special import jv, jvp
    from scipy.optimize import brentq
    from sys import argv
    from pylab import *
    
    def Jn(r,n):
      return (sqrt(pi/(2*r))*jv(n+0.5,r))
    def Jn_zeros(n,nt):
      zerosj = zeros((n+1, nt), dtype=Float32)
      zerosj[0] = arange(1,nt+1)*pi
      points = arange(1,nt+n+1)*pi
      racines = zeros(nt+n, dtype=Float32)
      for i in range(1,n+1):
        for j in range(nt+n-i):
          foo = brentq(Jn, points[j], points[j+1], (i,))
          racines[j] = foo
        points = racines
        zerosj[i][:nt] = racines[:nt]
      return (zerosj)
    
    def rJnp(r,n):
      return (0.5*sqrt(pi/(2*r))*jv(n+0.5,r) + sqrt(pi*r/2)*jvp(n+0.5,r))
    def rJnp_zeros(n,nt):
      zerosj = zeros((n+1, nt), dtype=Float32)
      zerosj[0] = (2.*arange(1,nt+1)-1)*pi/2
      points = (2.*arange(1,nt+n+1)-1)*pi/2
      racines = zeros(nt+n, dtype=Float32)
      for i in range(1,n+1):
        for j in range(nt+n-i):
          foo = brentq(rJnp, points[j], points[j+1], (i,))
          racines[j] = foo
        points = racines
        zerosj[i][:nt] = racines[:nt]
      return (zerosj)
    
    n = int(argv[1])  # n'th spherical bessel function
    nt = int(argv[2]) # number of zeros to be computed
    
    dr = 0.01
    eps = dr/1000
    
    jnz = Jn_zeros(n,nt)[n]
    r1 = arange(eps,jnz[len(jnz)-1],dr)
    jnzp = rJnp_zeros(n,nt)[n]
    r2 = arange(eps,jnzp[len(jnzp)-1],dr)
    
    grid(True)
    plot(r1,Jn(r1,n),'b', r2,rJnp(r2,n),'r')
    title((str(nt)+' first zeros'))
    legend((r'$j_{'+str(n)+'}(r)$', r'$(rj_{'+str(n)+'}(r))\'$'))
    plot(jnz,zeros(len(jnz)),'bo', jnzp,zeros(len(jnzp)),'rd')
    gca().xaxis.set_minor_locator(MultipleLocator(1))
    # gca().xaxis.set_minor_formatter(FormatStrFormatter('%d'))
    show()
    







.. code-block:: python

    bessph_zeros_rec 5 10
    



.. image:: SphericalBesselZeros_attachments/snapshot.png

