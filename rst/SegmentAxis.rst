Convolution-like operations
===========================

Users frequently want to break an array up into overlapping chunks, then
apply the same operation to each chunk. You can generate a dynamical
power spectrum, for example, by taking an FFT of each chunk, or you can
construct a convolution using a dot product. Some of these operations
already exist in numpy and scipy, but others don't. One way to attack
the problem would be to make a matrix in which each column was a
starting location, and each row was a chunk. This would normally require
duplicating some data, potentially a lot of data if there's a lot of
overlap, but numpy's striding can be used to do this. The simplification
of striding doesn't come for free; if you modify the array, all shared
elements will be modified. Nevertheless, it's a useful operation. Find
attached the code, .. image:: SegmentAxis_attachments/segmentaxis.py . Example usage:



.. code-block:: python

    In [1]: import numpy as N
    In [2]: import segmentaxis
    In [3]: a = N.zeros(30)
    In [4]: a[15] = 1
    In [5]: filter = N.array([0.1,0.5,1,0.5,0.1])
    In [6]: sa = segmentaxis.segment_axis(a,len(filter),len(filter)-1)
    In [7]: sa
    Out[7]:
    array([[ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  1.],
           [ 0.,  0.,  0.,  1.,  0.],
           [ 0.,  0.,  1.,  0.,  0.],
           [ 0.,  1.,  0.,  0.,  0.],
           [ 1.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.],
           [ 0.,  0.,  0.,  0.,  0.]])
    In [8]: N.dot(sa[::2,:],filter)
    Out[8]:
    array([ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0.5,  0.5,  0. ,  0. ,  0. ,
            0. ,  0. ])
    In [9]: N.dot(sa[1::2,:],filter)
    Out[9]:
    array([ 0. ,  0. ,  0. ,  0. ,  0. ,  0.1,  1. ,  0.1,  0. ,  0. ,  0. ,
            0. ,  0. ])
    In [10]: N.dot(sa,filter)
    Out[10]:
    array([ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
            0.1,  0.5,  1. ,  0.5,  0.1,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,
            0. ,  0. ,  0. ,  0. ])
    





