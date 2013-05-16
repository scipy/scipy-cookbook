Stride tricks for the Game of Life
==================================

This is similar to [:../SegmentAxis:Segment axis], but for 2D arrays
with 2D windows.

The Game of Life is a cellular automaton devised by the British
mathematician John Horton Conway in 1970, see [1].

It consists of a rectangular grid of cells which are either dead or
alive, and a transition rule for updating the cells' state. To update
each cell in the grid, the state of the 8 neighbouring cells needs to be
examined, i.e. it would be desirable to have an easy way of accessing
the 8 neighbours of all the cells at once without making unnecessary
copies. The code snippet below shows how to use the devious stride
tricks for that purpose.

[1] `Game of
Life <http://en.wikipedia.org/wiki/Conway%27s_Game_of_Life>`__ at
Wikipedia



.. code-block:: python

    In [1]: import numpy as np
    In [2]: from numpy.lib import stride_tricks
    In [3]: x = np.arange(20).reshape([4, 5])
    In [4]: xx = stride_tricks.as_strided(x, shape=(2, 3, 3, 3), strides=x.strides +
     x.strides)
    In [5]: x
    Out[5]: 
    array([[ 0,  1,  2,  3,  4],
           [ 5,  6,  7,  8,  9],
           [10, 11, 12, 13, 14],
           [15, 16, 17, 18, 19]])
    In [6]: xx
    Out[6]: 
    array([[[[ 0,  1,  2],
             [ 5,  6,  7],
             [10, 11, 12]],
    
            [[ 1,  2,  3],
             [ 6,  7,  8],
             [11, 12, 13]],
    
            [[ 2,  3,  4],
             [ 7,  8,  9],
             [12, 13, 14]]],
    
    
           [[[ 5,  6,  7],
             [10, 11, 12],
             [15, 16, 17]],
    
            [[ 6,  7,  8],
             [11, 12, 13],
             [16, 17, 18]],
    
            [[ 7,  8,  9],
             [12, 13, 14],
             [17, 18, 19]]]])
    In [7]: xx[0, 0]
    Out[7]: 
    array([[ 0,  1,  2],
           [ 5,  6,  7],
           [10, 11, 12]])
    In [8]: xx[1, 2]
    Out[8]: 
    array([[ 7,  8,  9],
           [12, 13, 14],
           [17, 18, 19]])
    In [9]: x.strides
    Out[9]: (20, 4)
    In [10]: xx.strides
    Out[10]: (20, 4, 20, 4)
    





