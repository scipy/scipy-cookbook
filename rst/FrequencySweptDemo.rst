This page demonstrates two functions in scipy.signal for generating
frequency-swept signals: \`chirp\` and \`sweep\_poly\`.

Some of these require SciPy 0.8.

To run the code samples, you will need the following imports:



.. code-block:: python

        import numpy as np
        from scipy.signal import chirp, sweep_poly
    



Linear Chirp
============

Sample code:



.. code-block:: python

        t = np.linspace(0, 10, 5001)
        w = chirp(t, f0=12.5, f1=2.5, t1=10, method='linear')
    



.. image:: FrequencySweptDemo_attachments/chirp_linear.png

Quadratic Chirp
===============

Sample code:



.. code-block:: python

        t = np.linspace(0, 10, 5001)
        w = chirp(t, f0=12.5, f1=2.5, t1=10, method='quadratic')
    



.. image:: FrequencySweptDemo_attachments/chirp_quadratic.png

Sample code using \`vertex\_zero\`:



.. code-block:: python

        t = np.linspace(0, 10, 5001)
        w = chirp(t, f0=12.5, f1=2.5, t1=10, method='quadratic', vertex_zero=False)
    



.. image:: FrequencySweptDemo_attachments/chirp_quadratic_v0false.png

Logarithmic Chirp
=================

Sample code:



.. code-block:: python

        t = np.linspace(0, 10, 5001)
        w = chirp(t, f0=12.5, f1=2.5, t1=10, method='logarithmic')
    



.. image:: FrequencySweptDemo_attachments/chirp_logarithmic.png

Hyperbolic Chirp
================

Sample code:



.. code-block:: python

        t = np.linspace(0, 10, 5001)
        w = chirp(t, f0=12.5, f1=2.5, t1=10, method='hyperbolic')
    



.. image:: FrequencySweptDemo_attachments/chirp_hyperbolic.png

Sweep Poly
==========

Sample code:



.. code-block:: python

        p = poly1d([0.05, -0.75, 2.5, 5.0])
        t = np.linspace(0, 10, 5001)
        w = sweep_poly(t, p)
    



.. image:: FrequencySweptDemo_attachments/sweep_poly.png

The script that generated the plots is here:

.. image:: FrequencySweptDemo_attachments/chirp_plot.py

