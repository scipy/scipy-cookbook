Note to the administrators of scipy/cookbook
============================================

I'm planning to describe a method that could help other people to keep
track of their simulations and provide simple framework. It is available
as `pypi <http://pypi.python.org>`__ package. A tutorial can be found
`here <http://homepage.univie.ac.at/wolfgang.lechner/gmailpm.html>`__.
In the recipe I would just describe what I did in the python code.

Do you think this is appropriate here? The script does not make use of
scipy or numpy but I think the audience of scipy.org might like the
idea! Please let me know if this is an inappropriate recipe, otherwise I
will just start writing next week.

Basics
======

The idea is to use python in combination with gmail as a powerful but
yet simple tool to document runs of computer simulations, their
parameters, starting times, progress and results.



.. code-block:: python

    #!python numbers=disable
    import numpy as np 
    #test
    





