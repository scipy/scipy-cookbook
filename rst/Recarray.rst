Addressing Array Columns by Name
================================

There are two very closely related ways to access array columns by name:
recarrays and structured arrays. Structured arrays are just ndarrays
with a complicated data type:



.. code-block:: python

    #!python numbers=disable
    In [1]: from numpy import *
    In [2]: ones(3, dtype=dtype([('foo', int), ('bar', float)]))
    Out[2]:
    array([(1, 1.0), (1, 1.0), (1, 1.0)],
          dtype=[('foo', '<i4'), ('bar', '<f8')])
    In [3]: r = _
    In [4]: r['foo']
    Out[4]: array([1, 1, 1])
    



recarray is a subclass of ndarray that just adds attribute access to
structured arrays:



.. code-block:: python

    #!python numbers=disable
    In [10]: r2 = r.view(recarray)
    In [11]: r2
    Out[11]:
    recarray([(1, 1.0), (1, 1.0), (1, 1.0)],
          dtype=[('foo', '<i4'), ('bar', '<f8')])
    In [12]: r2.foo
    Out[12]: array([1, 1, 1])
    



One downside of recarrays is that the attribute access feature slows
down all field accesses, even the r['foo'] form, because it sticks a
bunch of pure Python code in the middle. Much code won't notice this,
but if you end up having to iterate over an array of records, this will
be a hotspot for you.

Structured arrays are sometimes confusingly called record arrays.

``. - lightly adapted from a Robert Kern post of Thu, 26 Jun 2008 15:25:11 -0500``

Converting to regular arrays and reshaping
==========================================

A little script showing how to efficiently reformat structured arrays
into normal ndarrays.

Based on: `printing structured
arrays <http://old.nabble.com/printing-structured-arrays-td27794203.html>`__.



.. code-block:: python

    #!python numbers=disable
    
    import numpy as np
    
    data = [ (1, 2), (3, 4.1), (13, 77) ]
    dtype = [('x', float), ('y', float)]
    
    print('\n ndarray')
    nd = np.array(data)
    print nd
    
    print ('\n structured array')
    
    struct_1dtype = np.array(data, dtype=dtype)
    print struct_1dtype
    
    print('\n structured to ndarray')
    struct_1dtype_float = struct_1dtype.view(np.ndarray).reshape(len(struct_1dtype),
     -1) 
    print struct_1dtype_float
    
    print('\n structured to float: alternative ways')
    struct_1dtype_float_alt = struct_1dtype.view((np.float, len(struct_1dtype.dtype.
    names)))
    print struct_1dtype_float_alt
    
    # with heterogeneous dtype.
    struct_diffdtype = np.array([(1.0, 'string1', 2.0), (3.0, 'string2', 4.1)],
    dtype=[('x', float),('str_var', 'a7'),('y',float)])
    print('\n structured array with different dtypes')
    print struct_diffdtype
    struct_diffdtype_nd = struct_diffdtype[['str_var', 'x', 'y']].view(np.ndarray).r
    eshape(len(struct_diffdtype), -1) 
    
    
    print('\n structured array with different dtypes to reshaped ndarray')
    print struct_diffdtype_nd
    
    
    print('\n structured array with different dtypes to reshaped float array ommitin
    g string columns')
    struct_diffdtype_float = struct_diffdtype[['x', 'y']].view(float).reshape(len(st
    ruct_diffdtype),-1)
    print struct_diffdtype_float
    





