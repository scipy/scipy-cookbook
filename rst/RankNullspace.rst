Rank and nullspace of a matrix
==============================

The following module, rank\_nullspace.py, provides the functions rank()
and nullspace(). (Note that !NumPy already provides the function
matrix\_rank(); the function given here allows an absolute tolerance to
be specified along with a relative tolerance.)

**rank\_nullspace.py**



.. code-block:: python

    #!python
    import numpy as np
    from numpy.linalg import svd
    
    
    def rank(A, atol=1e-13, rtol=0):
        """Estimate the rank (i.e. the dimension of the nullspace) of a matrix.
    
        The algorithm used by this function is based on the singular value
        decomposition of `A`.
    
        Parameters
        ----------
        A : ndarray
            A should be at most 2-D.  A 1-D array with length n will be treated
            as a 2-D with shape (1, n)
        atol : float
            The absolute tolerance for a zero singular value.  Singular values
            smaller than `atol` are considered to be zero.
        rtol : float
            The relative tolerance.  Singular values less than rtol*smax are
            considered to be zero, where smax is the largest singular value.
    
        If both `atol` and `rtol` are positive, the combined tolerance is the
        maximum of the two; that is::
            tol = max(atol, rtol * smax)
        Singular values smaller than `tol` are considered to be zero.
    
        Return value
        ------------
        r : int
            The estimated rank of the matrix.
    
        See also
        --------
        numpy.linalg.matrix_rank
            matrix_rank is basically the same as this function, but it does not
            provide the option of the absolute tolerance.
        """
    
        A = np.atleast_2d(A)
        s = svd(A, compute_uv=False)
        tol = max(atol, rtol * s[0])
        rank = int((s >= tol).sum())
        return rank
    
    
    def nullspace(A, atol=1e-13, rtol=0):
        """Compute an approximate basis for the nullspace of A.
    
        The algorithm used by this function is based on the singular value
        decomposition of `A`.
    
        Parameters
        ----------
        A : ndarray
            A should be at most 2-D.  A 1-D array with length k will be treated
            as a 2-D with shape (1, k)
        atol : float
            The absolute tolerance for a zero singular value.  Singular values
            smaller than `atol` are considered to be zero.
        rtol : float
            The relative tolerance.  Singular values less than rtol*smax are
            considered to be zero, where smax is the largest singular value.
    
        If both `atol` and `rtol` are positive, the combined tolerance is the
        maximum of the two; that is::
            tol = max(atol, rtol * smax)
        Singular values smaller than `tol` are considered to be zero.
    
        Return value
        ------------
        ns : ndarray
            If `A` is an array with shape (m, k), then `ns` will be an array
            with shape (k, n), where n is the estimated dimension of the
            nullspace of `A`.  The columns of `ns` are a basis for the
            nullspace; each element in numpy.dot(A, ns) will be approximately
            zero.
        """
    
        A = np.atleast_2d(A)
        u, s, vh = svd(A)
        tol = max(atol, rtol * s[0])
        nnz = (s >= tol).sum()
        ns = vh[nnz:].conj().T
        return ns
    



Here's a demonstration script.



.. code-block:: python

    #!python
    import numpy as np
    
    from rank_nullspace import rank, nullspace
    
    
    def checkit(a):
        print "a:"
        print a
        r = rank(a)
        print "rank is", r
        ns = nullspace(a)
        print "nullspace:"
        print ns
        if ns.size > 0:
            res = np.abs(np.dot(a, ns)).max()
            print "max residual is", res    
    
    
    print "-"*25
    
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    checkit(a)
    
    print "-"*25
    
    a = np.array([[0.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    checkit(a)
    
    print "-"*25
    
    a = np.array([[0.0, 1.0, 2.0, 4.0], [1.0, 2.0, 3.0, 4.0]])
    checkit(a)
    
    print "-"*25
    
    a = np.array([[1.0,   1.0j,   2.0+2.0j],
                  [1.0j, -1.0,   -2.0+2.0j],
                  [0.5,   0.5j,   1.0+1.0j]])
    checkit(a)
    
    print "-"*25
    



And here is the output of the script.



.. code-block:: python

    -------------------------
    a:
    [[ 1.  2.  3.]
     [ 4.  5.  6.]
     [ 7.  8.  9.]]
    rank is 2
    nullspace:
    [[-0.40824829]
     [ 0.81649658]
     [-0.40824829]]
    max residual is 4.4408920985e-16
    -------------------------
    a:
    [[ 0.  2.  3.]
     [ 4.  5.  6.]
     [ 7.  8.  9.]]
    rank is 3
    nullspace:
    []
    -------------------------
    a:
    [[ 0.  1.  2.  4.]
     [ 1.  2.  3.  4.]]
    rank is 2
    nullspace:
    [[-0.48666474 -0.61177492]
     [-0.27946883  0.76717915]
     [ 0.76613356 -0.15540423]
     [-0.31319957 -0.11409267]]
    max residual is 3.88578058619e-16
    -------------------------
    a:
    [[ 1.0+0.j   0.0+1.j   2.0+2.j ]
     [ 0.0+1.j  -1.0+0.j  -2.0+2.j ]
     [ 0.5+0.j   0.0+0.5j  1.0+1.j ]]
    rank is 1
    nullspace:
    [[ 0.00000000-0.j         -0.94868330-0.j        ]
     [ 0.13333333+0.93333333j  0.00000000-0.10540926j]
     [ 0.20000000-0.26666667j  0.21081851-0.21081851j]]
    max residual is 1.04295984227e-15
    -------------------------
    





