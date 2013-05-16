#. 

   #. page was renamed from Cookbook/Pyrex And NumPy

Please note that the code described here is slightly out of date, since
today `cython <http://cython.org>`__ is the actively maintained version
of Pyrex, and numpy now ships with Cython examples.

Rather than maintaining both the wiki and the source dir, we'll continue
to update the sources, kept
`here <http://projects.scipy.org/scipy/numpy/browser/trunk/numpy/doc/cython>`__.

Old Pyrex page
--------------

`Pyrex <http://nz.cosc.canterbury.ac.nz/~greg/python/Pyrex/>`__ is a
language for writing C extensions to Python. Its syntax is very similar
to writing Python. A file is compiled to a file, which is then compiled
like a standard C extension module for Python. Many people find writing
extension modules with Pyrex preferable to writing them in C or using
other tools, such as SWIG.

This page is a starting point for accessing numpy arrays natively with
Pyrex. Please note that with current versions of NumPy (SVN), the
directory contains a complete working example with the code in this
page, including also a proper file so you can install it with the
standard Python mechanisms. This should help you get up and running
quickly.

Here's a file I call "c\_python.pxd":



.. code-block:: python

    cdef extern from "Python.h":
        ctypedef int Py_intptr_t
    



and here's "c\_numpy.pxd":



.. code-block:: python

    cimport c_python
    
    cdef extern from "numpy/arrayobject.h":
        ctypedef class numpy.ndarray [object PyArrayObject]:
            cdef char *data
            cdef int nd
            cdef c_python.Py_intptr_t *dimensions
            cdef c_python.Py_intptr_t *strides
            cdef object base
            # descr not implemented yet here...
            cdef int flags
            cdef int itemsize
            cdef object weakreflist
    
        cdef void import_array()
    



Here's an example program, name this something like "test.pyx" suffix.



.. code-block:: python

    cimport c_numpy
    cimport c_python
    import numpy
    
    c_numpy.import_array()
    
    def print_array_info(c_numpy.ndarray arr):
        cdef int i
    
        print '-='*10
        print 'printing array info for ndarray at 0x%0lx'%(<c_python.Py_intptr_t>arr
    ,)
        print 'print number of dimensions:',arr.nd
        print 'address of strides: 0x%0lx'%(<c_python.Py_intptr_t>arr.strides,)
        print 'strides:'
        for i from 0<=i<arr.nd:
            # print each stride
            print '  stride %d:'%i,<c_python.Py_intptr_t>arr.strides[i]
        print 'memory dump:'
        print_elements( arr.data, arr.strides, arr.dimensions, arr.nd, sizeof(double
    ), arr.dtype )
        print '-='*10
        print
        
    cdef print_elements(char *data,
                        c_python.Py_intptr_t* strides,
                        c_python.Py_intptr_t* dimensions,
                        int nd,
                        int elsize,
                        object dtype):
        cdef c_python.Py_intptr_t i,j
        cdef void* elptr
        
        if dtype not in [numpy.dtype(numpy.object_),
                         numpy.dtype(numpy.float64)]:
            print '   print_elements() not (yet) implemented for dtype %s'%dtype.nam
    e
            return
        
        if nd ==0:
            if dtype==numpy.dtype(numpy.object_):
                elptr = (<void**>data)[0] #[0] dereferences pointer in Pyrex
                print '  ',<object>elptr
            elif dtype==numpy.dtype(numpy.float64):
                print '  ',(<double*>data)[0]
        elif nd == 1:
            for i from 0<=i<dimensions[0]:
                if dtype==numpy.dtype(numpy.object_):
                    elptr = (<void**>data)[0]
                    print '  ',<object>elptr
                elif dtype==numpy.dtype(numpy.float64):
                    print '  ',(<double*>data)[0]
                data = data + strides[0]
        else:
            for i from 0<=i<dimensions[0]:
                print_elements(data, strides+1, dimensions+1, nd-1, elsize, dtype)
                data = data + strides[0]
    
    def test():
        """this function is pure Python"""
        arr1 = numpy.array(-1e-30,dtype=numpy.Float64)
        arr2 = numpy.array([1.0,2.0,3.0],dtype=numpy.Float64)
    
        arr3 = numpy.arange(9,dtype=numpy.Float64)
        arr3.shape = 3,3
    
        four = 4
        arr4 = numpy.array(['one','two',3,four],dtype=numpy.object_)
    
        arr5 = numpy.array([1,2,3]) # int types not (yet) supported by print_element
    s
    
        for arr in [arr1,arr2,arr3,arr4,arr5]:
            print_array_info(arr)
    



Now, if you compile and install the above test.pyx, the output of should
be something like the following:



.. code-block:: python

    -=-=-=-=-=-=-=-=-=-=
    printing array info for ndarray at 0x8184508
    print number of dimensions: 0
    address of strides: 0xb764f7ec
    strides:
    memory dump:
       -1e-30
    -=-=-=-=-=-=-=-=-=-=
    
    -=-=-=-=-=-=-=-=-=-=
    printing array info for ndarray at 0x8190060
    print number of dimensions: 1
    address of strides: 0x818453c
    strides:
      stride 0: 8
    memory dump:
       1.0
       2.0
       3.0
    -=-=-=-=-=-=-=-=-=-=
    
    -=-=-=-=-=-=-=-=-=-=
    printing array info for ndarray at 0x82698a0
    print number of dimensions: 2
    address of strides: 0x8190098
    strides:
      stride 0: 24
      stride 1: 8
    memory dump:
       0.0
       1.0
       2.0
       3.0
       4.0
       5.0
       6.0
       7.0
       8.0
    -=-=-=-=-=-=-=-=-=-=
    
    -=-=-=-=-=-=-=-=-=-=
    printing array info for ndarray at 0x821d6e0
    print number of dimensions: 1
    address of strides: 0x818ed74
    strides:
      stride 0: 4
    memory dump:
       one
       two
       3
       4
    -=-=-=-=-=-=-=-=-=-=
    
    -=-=-=-=-=-=-=-=-=-=
    printing array info for ndarray at 0x821d728
    print number of dimensions: 1
    address of strides: 0x821d75c
    strides:
      stride 0: 4
    memory dump:
       print_elements() not (yet) implemented for dtype int32
    -=-=-=-=-=-=-=-=-=-=
    



The `pytables project <http://pytables.sourceforge.net/>`__ makes
extensive use of Pyrex and numarray. See the pytables source code for
more ideas.

See Also
========

``["Cookbook/ArrayStruct_and_Pyrex"]``

--------------

CategoryCookbook

