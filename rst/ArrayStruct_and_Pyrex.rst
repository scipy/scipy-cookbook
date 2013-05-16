Overview
========

`Pyrex <http://nz.cosc.canterbury.ac.nz/~greg/python/Pyrex/>`__ is a
language for writing C extensions to Python. Its syntax is very similar
to writing Python. A file is compiled to a file, which is then compiled
like a standard C extension module for Python. Many people find writing
extension modules with Pyrex preferable to writing them in C or using
other tools, such as SWIG.

See http://numeric.scipy.org/ for an explanation of the interface. The
following packages support the interface:

| ``* !NumPy (all versions)``
| ``* Numeric (>=24.2)``
| ``* numarray (>=1.5.0)``

Sharing data malloced by a Pyrex-based extension
================================================

Here is a Pyrex file which shows how to export its data using the array
interface. This allows various Python types to have a view of the data
without actually copying the data itself.

#. 

   #. pyrex should pass through python parser:





.. code-block:: python

    cdef extern from "Python.h":
        ctypedef int Py_intptr_t
        void Py_INCREF(object)
        void Py_DECREF(object)
        object PyCObject_FromVoidPtrAndDesc( void* cobj, void* desc, void (*destr)(v
    oid *, void *))
        
    cdef extern from "stdlib.h":
        ctypedef int size_t
        ctypedef long intptr_t
        void *malloc(size_t size)
        void free(void* ptr)
    
    # for PyArrayInterface:
    CONTIGUOUS=0x01
    FORTRAN=0x02
    ALIGNED=0x100
    NOTSWAPPED=0x200
    WRITEABLE=0x400
    
    ctypedef struct PyArrayInterface:
        int version          # contains the integer 2 as a sanity check
        int nd               # number of dimensions
        char typekind        # kind in array --- character code of typestr
        int itemsize         # size of each element
        int flags            # flags indicating how the data should be interpreted
        Py_intptr_t *shape   # A length-nd array of shape information
        Py_intptr_t *strides # A length-nd array of stride information
        void *data           # A pointer to the first element of the array
    
    cdef void free_array_interface( void* ptr, void *arr ):
        cdef PyArrayInterface* inter
    
        inter = <PyArrayInterface*>ptr
        arrpy = <object>arr
        Py_DECREF( arrpy )
        free(inter)
    
    ctypedef unsigned char fi
    ctypedef fi* fiptr
    cdef class Unsigned8Buf:
        cdef fiptr data
        cdef Py_intptr_t shape[2]
        cdef Py_intptr_t strides[2]
        
        def __init__(self, int width, int height):
            cdef int bufsize
            bufsize = width*height*sizeof(fi)
            self.data=<fiptr>malloc( bufsize )
            if self.data==NULL: raise MemoryError("Error allocating memory")
            self.strides[0]=width
            self.strides[1]=1 # 1 byte per element
            
            self.shape[0]=height
            self.shape[1]=width
        
        def __dealloc__(self):
                free(self.data)
                
        property __array_struct__:
            def __get__(self):
                cdef PyArrayInterface* inter
    
                cdef Py_intptr_t *newshape   # A length-nd array of shape informatio
    n
                cdef Py_intptr_t *newstrides # A length-nd array of stride informati
    on
                cdef int nd
    
                nd = 2
    
                inter = <PyArrayInterface*>malloc( sizeof( PyArrayInterface ) )
                if inter is NULL:
                    raise MemoryError()
    
                inter.version = 2
                inter.nd = nd
                inter.typekind = 'u'[0] # unsigned int
                inter.itemsize = 1
                inter.flags = NOTSWAPPED | ALIGNED | WRITEABLE
                inter.strides = self.strides
                inter.shape = self.shape
                inter.data = self.data
                Py_INCREF(self)
                obj = PyCObject_FromVoidPtrAndDesc( <void*>inter, <void*>self, free_
    array_interface)
                return obj
    



Using data malloced elsewhere with a Pyrex-based extension
==========================================================

One example is the get\_data\_copy() function of
`\_cam\_iface\_shm.pyx <http://code.astraw.com/projects/motmot/browser/trunk/cam_iface/src/_cam_iface_shm.pyx>`__
in the `motmot camera
utilities <http://code.astraw.com/projects/motmot>`__ software. In this
use example, an image is copied from shared memory into an externally
malloced buffer supporting the interface. (The shared memory stuff has
only been tested on linux, but the rest should work anywhere.)

See also
========

``* ["Cookbook/Pyrex_and_NumPy"]``

--------------

CategoryCookbook

