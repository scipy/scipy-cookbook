This sample code demonstrates the use of the function
scipy.signal.filtfilt, a linear filter that achieves zero phase delay by
applying an `IIR
filter <http://en.wikipedia.org/wiki/Infinite_impulse_response>`__ to a
signal twice, once forwards and once backwards. The order of the filter
is twice the original filter order. The function also computes the
initial filter parameters in order to provide a more stable response
(via lfilter\_zi).

For comparison, this script also applies the same IIR filter to the
signal using scipy.signal.lfilter; for these calculations, lfilter\_zi
is used to choose appropriate initial conditions for the filter. Without
this, these plots would have long transients near 0. As it is, they have
long transients near the initial value of the signal.

Code
====




.. code-block:: python

    
    from numpy import sin, cos, pi, linspace
    from numpy.random import randn
    from scipy.signal import lfilter, lfilter_zi, filtfilt, butter
    
    from matplotlib.pyplot import plot, legend, show, hold, grid, figure, savefig
    
    
    # Generate a noisy signal to be filtered.
    t = linspace(-1, 1, 201)
    x = (sin(2 * pi * 0.75 * t*(1-t) + 2.1) + 0.1*sin(2 * pi * 1.25 * t + 1) +
        0.18*cos(2 * pi * 3.85 * t))
    xn = x + randn(len(t)) * 0.08
    
    # Create an order 3 lowpass butterworth filter.
    b, a = butter(3, 0.05)
    
    # Apply the filter to xn.  Use lfilter_zi to choose the initial condition
    # of the filter.
    zi = lfilter_zi(b, a)
    z, _ = lfilter(b, a, xn, zi=zi*xn[0])
    
    # Apply the filter again, to have a result filtered at an order
    # the same as filtfilt.
    z2, _ = lfilter(b, a, z, zi=zi*z[0])
    
    # Use filtfilt to apply the filter.
    y = filtfilt(b, a, xn)
    
    
    # Make the plot.
    figure(figsize=(10,5))
    hold(True)
    plot(t, xn, 'b', linewidth=1.75, alpha=0.75)
    plot(t, z, 'r--', linewidth=1.75)
    plot(t, z2, 'r', linewidth=1.75)
    plot(t, y, 'k', linewidth=1.75)
    legend(('noisy signal',
            'lfilter, once',
            'lfilter, twice',
            'filtfilt'),
            loc='best')
    hold(False)
    grid(True)
    show()
    #savefig('plot.png', dpi=65)
    



Figure
======

.. image:: FiltFilt_attachments/filfilt2.jpg

