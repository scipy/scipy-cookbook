accum, a function like MATLAB's accumarray
==========================================

NumPy doesn't include a function that is equivalent to MATLAB's
\`accumarray\` function. The following function, \`accum\`, is close.

Note that \`accum\` can handle n-dimensional arrays, and allows the data
type of the result to be specified.



.. code-block:: python

    from itertools import product
    import numpy as np
    
    
    def accum(accmap, a, func=None, size=None, fill_value=0, dtype=None):
        """
        An accumulation function similar to Matlab's `accumarray` function.
    
        Parameters
        ----------
        accmap : ndarray
            This is the "accumulation map".  It maps input (i.e. indices into
            `a`) to their destination in the output array.  The first `a.ndim`
            dimensions of `accmap` must be the same as `a.shape`.  That is,
            `accmap.shape[:a.ndim]` must equal `a.shape`.  For example, if `a`
            has shape (15,4), then `accmap.shape[:2]` must equal (15,4).  In this
            case `accmap[i,j]` gives the index into the output array where
            element (i,j) of `a` is to be accumulated.  If the output is, say,
            a 2D, then `accmap` must have shape (15,4,2).  The value in the
            last dimension give indices into the output array. If the output is
            1D, then the shape of `accmap` can be either (15,4) or (15,4,1) 
        a : ndarray
            The input data to be accumulated.
        func : callable or None
            The accumulation function.  The function will be passed a list
            of values from `a` to be accumulated.
            If None, numpy.sum is assumed.
        size : ndarray or None
            The size of the output array.  If None, the size will be determined
            from `accmap`.
        fill_value : scalar
            The default value for elements of the output array. 
        dtype : numpy data type, or None
            The data type of the output array.  If None, the data type of
            `a` is used.
    
        Returns
        -------
        out : ndarray
            The accumulated results.
    
            The shape of `out` is `size` if `size` is given.  Otherwise the
            shape is determined by the (lexicographically) largest indices of
            the output found in `accmap`.
    
    
        Examples
        --------
        >>> from numpy import array, prod
        >>> a = array([[1,2,3],[4,-1,6],[-1,8,9]])
        >>> a
        array([[ 1,  2,  3],
               [ 4, -1,  6],
               [-1,  8,  9]])
        >>> # Sum the diagonals.
        >>> accmap = array([[0,1,2],[2,0,1],[1,2,0]])
        >>> s = accum(accmap, a)
        array([9, 7, 15])
        >>> # A 2D output, from sub-arrays with shapes and positions like this:
        >>> # [ (2,2) (2,1)]
        >>> # [ (1,2) (1,1)]
        >>> accmap = array([
                [[0,0],[0,0],[0,1]],
                [[0,0],[0,0],[0,1]],
                [[1,0],[1,0],[1,1]],
            ])
        >>> # Accumulate using a product.
        >>> accum(accmap, a, func=prod, dtype=float)
        array([[ -8.,  18.],
               [ -8.,   9.]])
        >>> # Same accmap, but create an array of lists of values.
        >>> accum(accmap, a, func=lambda x: x, dtype='O')
        array([[[1, 2, 4, -1], [3, 6]],
               [[-1, 8], [9]]], dtype=object)
        """
    
        # Check for bad arguments and handle the defaults.
        if accmap.shape[:a.ndim] != a.shape:
            raise ValueError("The initial dimensions of accmap must be the same as a
    .shape")
        if func is None:
            func = np.sum
        if dtype is None:
            dtype = a.dtype
        if accmap.shape == a.shape:
            accmap = np.expand_dims(accmap, -1)
        adims = tuple(range(a.ndim))
        if size is None:
            size = 1 + np.squeeze(np.apply_over_axes(np.max, accmap, axes=adims))
        size = np.atleast_1d(size)
    
        # Create an array of python lists of values.
        vals = np.empty(size, dtype='O')
        for s in product(*[range(k) for k in size]):
            vals[s] = []
        for s in product(*[range(k) for k in a.shape]):
            indx = tuple(accmap[s])
            val = a[s]
            vals[indx].append(val)
    
        # Create the output array.
        out = np.empty(size, dtype=dtype)
        for s in product(*[range(k) for k in size]):
            if vals[s] == []:
                out[s] = fill_value
            else:
                out[s] = func(vals[s])
    
        return out
    



Examples
========

A basic example--sum the diagonals (with wrapping) of a 3 by 3 array:



.. code-block:: python

    In [5]: from numpy import array, prod
    
    In [6]: from accum import accum
    
    In [7]: a = array([[1,2,3],[4,-1,6],[-1,8,9]])
    
    In [8]: a
    Out[8]: 
    array([[ 1,  2,  3],
           [ 4, -1,  6],
           [-1,  8,  9]])
    
    In [9]: accmap = array([[0,1,2],[2,0,1],[1,2,0]])
    
    In [10]: s = accum(accmap, a)
    
    In [11]: s
    Out[11]: array([ 9,  7, 15])
    



Accumulate using multiplication, going from a 3 by 3 array to 2 by 2
array:



.. code-block:: python

    In [12]: accmap = array([
       ....:             [[0,0],[0,0],[0,1]],
       ....:             [[0,0],[0,0],[0,1]],
       ....:             [[1,0],[1,0],[1,1]],
       ....:         ])
    
    In [13]: accum(accmap, a, func=prod, dtype=float)
    Out[13]: 
    array([[ -8.,  18.],
           [ -8.,   9.]])
    



Create an array of lists containing the values to be accumulated in each
position in the output array:



.. code-block:: python

    In [14]: accum(accmap, a, func=lambda x: x, dtype='O')
    Out[14]: 
    array([[[1, 2, 4, -1], [3, 6]],
           [[-1, 8], [9]]], dtype=object)
    



Use \`accum\` to arrange some values from a 1D array in a 2D array (note
that using \`accum\` for this is overkill; fancy indexing would
suffice):



.. code-block:: python

    In [15]: subs = np.array([[k,5-k] for k in range(6)])
    
    In [16]: subs
    Out[16]: 
    array([[0, 5],
           [1, 4],
           [2, 3],
           [3, 2],
           [4, 1],
           [5, 0]])
    
    In [17]: vals = array(range(10,16))
    
    In [18]: accum(subs, vals)
    Out[18]: 
    array([[ 0,  0,  0,  0,  0, 10],
           [ 0,  0,  0,  0, 11,  0],
           [ 0,  0,  0, 12,  0,  0],
           [ 0,  0, 13,  0,  0,  0],
           [ 0, 14,  0,  0,  0,  0],
           [15,  0,  0,  0,  0,  0]])
    





