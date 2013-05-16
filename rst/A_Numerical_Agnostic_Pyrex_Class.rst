Overview
========

Here is presented *!NumInd* (Numerical Independent Extension), an
example of a class written in
`Pyrex <http://nz.cosc.canterbury.ac.nz/~greg/python/Pyrex/>`__. This
class can wrap information from any object coming from Numeric, numarray
or !NumPy without any dependency of those packages for compiling the
extension. It lets you to create a uniform interface in both Python and
C spaces. In addition, you can personalize it by adding new methods or
properties.

For this extension to work, you need a numerical package that supports
the `array interface <http://numpy.scipy.org/array_interface.shtml>`__.
Any of these versions would be good enough:

| ``* !NumPy (all versions)``
| ``* Numeric (>=24.2)``
| ``* numarray (>=1.5.1)``

NumInd: a Numerical Independent Pyrex-based extension
=====================================================

The !NumInd class shown below takes a Numeric/numarray/!NumPy object and
creates another object that can be accessed in an uniform way from both
Python and Pyrex (and hence, C) space. Moreover, it exposes an array
interface so that you can re-wrap this object with any
Numeric/numarray/!NumPy. All of these features are achieved without
actually copying the data itself. This opens the door to the possibility
to develop applications that supports the Numeric/numarray/!NumPy triad
without a need to compile against any of them.

**Warning**: This class supports mainly homogeneous datasets, but it
wouldn't be difficult to support recarrays as well. This is a work
in-progress anyway.

#. 

   #. pyrex should pass through python parser:





.. code-block:: python

    # This Pyrex extension class can take a numpy/numarray/Numeric object
    # as a parameter and wrap it so that its information can be accessed
    # in a standard way, both in Python space and C space.
    #
    # Heavily based on an idea of Andrew Straw. See
    # http://www.scipy.org/Cookbook/ArrayStruct_and_Pyrex
    # Very inspiring! :-)
    #
    # First version: 2006-03-25
    # Last update: 2006-03-25
    # Author: Francesc Altet 
    
    import sys
    
    cdef extern from "Python.h":
        ctypedef int Py_intptr_t
        long PyInt_AsLong(object)
        void Py_INCREF(object)
        void Py_DECREF(object)
        object PyCObject_FromVoidPtrAndDesc(void* cobj, void* desc,
                                            void (*destr)(void *, void *))
    
    cdef extern from "stdlib.h":
        ctypedef long size_t
        ctypedef long intptr_t
        void *malloc(size_t size)
        void free(void* ptr)
    
    # for PyArrayInterface:
    CONTIGUOUS=0x01
    FORTRAN=0x02
    ALIGNED=0x100
    NOTSWAPPED=0x200
    WRITEABLE=0x400
    
    # byteorder dictionary
    byteorder = {'<':'little', '>':'big'}
    
    ctypedef struct PyArrayInterface:
        int version          # contains the integer 2 as a sanity check
        int nd               # number of dimensions
        char typekind        # kind in array --- character code of typestr
        int itemsize         # size of each element
        int flags            # flags indicating how the data should be interpreted
        Py_intptr_t *shape   # A length-nd array of shape information
        Py_intptr_t *strides # A length-nd array of stride information
        void *data           # A pointer to the first element of the array
    
    cdef void free_array_interface(void *ptr, void *arr):
        arrpy = <object>arr
        Py_DECREF(arrpy)
    
    
    cdef class NumInd:
        cdef void *data
        cdef int _nd
        cdef Py_intptr_t *_shape, *_strides
        cdef PyArrayInterface *inter
        cdef object _t_shape, _t_strides, _undarray
    
        def __init__(self, object undarray):
            cdef int i, stride
            cdef object array_shape, array_strides
    
            # Keep a reference to the underlying object
            self._undarray = undarray
            # Get the shape and strides C arrays
            array_shape = undarray.__array_shape__
            self._t_shape = array_shape
            # The number of dimensions
            self._nd = len(array_shape)
            # The shape
            self._shape = <Py_intptr_t *>malloc(self._nd*sizeof(Py_intptr_t))
            for i from 0 <= i < self._nd:
                self._shape[i] = self._t_shape[i]
            # The strides (compute them if needed)
            array_strides = undarray.__array_strides__
            self._t_strides = array_strides
            self._strides = <Py_intptr_t *>malloc(self._nd*sizeof(Py_intptr_t))
            if array_strides:
                for i from 0 <= i < self._nd:
                    self._strides[i] = array_strides[i]
            else:
                # strides is None. Compute them explicitely.
                self._t_strides = [0] * self._nd
                stride = int(self.typestr[2:])
                for i from self._nd > i >= 0:
                    self._strides[i] = stride
                    self._t_strides[i] = stride
                    stride = stride * array_shape[i]
                self._t_strides = tuple(self._t_strides)
            # Populate the C array interface
            self.inter = self._get_array_interface()
    
        # Properties. This are visible from Python space.
        # Add as many as you want.
    
        property undarray:  # Returns the underlying array
            def __get__(self):
                return self._undarray
    
        property shape:
            def __get__(self):
                return self._t_shape
    
        property strides:
            def __get__(self):
                return self._t_strides
    
        property typestr:
            def __get__(self):
                return self._undarray.__array_typestr__
    
        property readonly:
            def __get__(self):
                return self._undarray.__array_data__[1]
     
        property __array_struct__:
            "Allows other numerical packages to obtain a new object."
            def __get__(self):
                if hasattr(self._undarray, "__array_struct__"):
                    return self._undarray.__array_struct__
                else:
                    # No an underlying array with __array_struct__
                    # Deliver an equivalent PyCObject.
                    Py_INCREF(self)
                    return PyCObject_FromVoidPtrAndDesc(<void*>self.inter,
                                                        <void*>self,
                                                        free_array_interface)
    
        cdef PyArrayInterface *_get_array_interface(self):
            "Populates the array interface"
            cdef PyArrayInterface *inter
            cdef object undarray, data_address, typestr
    
            undarray = self._undarray
            typestr = self.typestr
            inter = <PyArrayInterface *>malloc(sizeof(PyArrayInterface))
            if inter is NULL:
                raise MemoryError()
    
            inter.version = 2
            inter.nd = self._nd
            inter.typekind = ord(typestr[1])
            inter.itemsize = int(typestr[2:])
            inter.flags = 0  # initialize flags
            if typestr[0] == '|':
                inter.flags = inter.flags | NOTSWAPPED
            elif byteorder[typestr[0]] == sys.byteorder:
                inter.flags = inter.flags | NOTSWAPPED
            if not self.readonly:
                inter.flags = inter.flags | WRITEABLE
            # XXX how to determine the ALIGNED flag?
            inter.strides = self._strides
            inter.shape = self._shape
            # Get the data address
            data_address = int(undarray.__array_data__[0], 16)
            inter.data = <void*>PyInt_AsLong(data_address)
            return inter
    
    
        # This is just an example on how to modify the data in C space
        # (and at C speed! :-)
        def modify(self):
            "Modify the values of the underlying array"
            cdef int *data, i
    
            data = <int *>self.inter.data
            # Modify just the first row
            for i from 0 <= i < self.inter.shape[self.inter.nd-1]:
                data[i] = data[i] + 1
    
        def __dealloc__(self):
            free(self._shape)
            free(self._strides)
            free(self.inter)
    



An example of use
=================

In order to get an idea of what the above extension offers, try to run
this script against the !NumInd extension:



.. code-block:: python

    import Numeric
    import numarray
    import numpy
    import numind
    
    # Create an arbitrary object for each package
    nu=Numeric.arange(12)
    nu.shape = (4,3)
    na=numarray.arange(12)
    na.shape = (4,3)
    np=numpy.arange(12)
    np.shape = (4,3)
    
    # Wrap the different objects with the NumInd class
    # and execute some actions on it
    for obj in [nu, na, np]:
        ni = numind.NumInd(obj)
        print "original object type-->", type(ni.undarray)
        # Print some values
        print "typestr -->", ni.typestr
        print "shape -->", ni.shape
        print "strides -->", ni.strides
        npa = numpy.asarray(ni)
        print "object after a numpy re-wrapping -->", npa
        ni.modify()
        print "object after modification in C space -->", npa
    



You can check the output here `1 <.. image:: A_Numerical_Agnostic_Pyrex_Class_attachments/test_output.txt>`__.

See also
========

| ``* ["Cookbook/Pyrex_and_NumPy"]``
| ``* ["Cookbook/ArrayStruct_and_Pyrex"] (The inspiring recipe)``

--------------

CategoryCookbook

