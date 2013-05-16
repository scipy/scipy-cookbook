Table of Contents
=================

TableOfContents

Introduction
============

`ctypes <http://starship.python.net/crew/theller/ctypes/>`__ is an
advanced Foreign Function Interface package for Python 2.3 and higher.
It is included in the standard library for Python 2.5.

ctypes allows to call functions exposed from DLLs/shared libraries and
has extensive facilities to create, access and manipulate simple and
complicated C data types in Python - in other words: wrap libraries in
pure Python. It is even possible to implement C callback functions in
pure Python.

ctypes also includes a code generator tool chain which allows automatic
creation of library wrappers from C header files. ctypes works on
Windows, Mac OS X, Linux, Solaris, FreeBSD, OpenBSD and other systems.

**Ensure that you have at least ctypes version 1.0.1 or later.**

Other possibilities to call or run C code in python include:
`SWIG <http://www.scipy.org/Cookbook/SWIG_and_NumPy>`__,
`Cython <http://wiki.cython.org/tutorials/numpy>`__, ["Weave"], etc

Getting Started with ctypes
===========================

The `ctypes
tutorial <http://starship.python.net/crew/theller/ctypes/tutorial.html>`__
and the `ctypes documentation for
Python <http://docs.python.org/lib/module-ctypes.html>`__ provide
extensive information on getting started with ctypes.

Assuming you've built a library called \`foo.dll\` or \`libfoo.so\`
containing a function called \`bar\` that takes a pointer to a buffer of
doubles and an int as arguments and returns an int, the following code
should get you up and running. The following sections cover some
possible build scripts, C code and Python code.

If you would like to build your DLL/shared library with distutils, take
a look at the !SharedLibrary distutils extension included with
`OOF2 <http://www.ctcms.nist.gov/oof/oof2/index.html>`__. This should
probably be included in numpy.distutils at some point.

Nmake Makefile (Windows)
------------------------

Run nmake inside the Visual Studio Command Prompt to build with the
following file.

You should be able to build the DLL with any version of the Visual
Studio compiler regardless of the compiler used to compile Python. Keep
in mind that you shouldn't allocate/deallocate memory across different
debug/release and single-threaded/multi-threaded runtimes or operate on
FILE\*s from different runtimes.



.. code-block:: python

    CXX = cl.exe
    LINK = link.exe
    
    CPPFLAGS = -D_WIN32 -D_USRDLL -DFOO_DLL -DFOO_EXPORTS
    CXXFLAGSALL = -nologo -EHsc -GS -W3 -Wp64 $(CPPFLAGS)
    CXXFLAGSDBG = -MDd -Od -Z7 -RTCcsu
    CXXFLAGSOPT = -MD -O2
    #CXXFLAGS = $(CXXFLAGSALL) $(CXXFLAGSDBG)
    CXXFLAGS = $(CXXFLAGSALL) $(CXXFLAGSOPT)
    
    LINKFLAGSALL = /nologo /DLL
    LINKFLAGSDBG = /DEBUG
    LINKFLAGSOPT =
    #LINKFLAGS = $(LINKFLAGSALL) $(LINKFLAGSDBG)
    LINKFLAGS = $(LINKFLAGSALL) $(LINKFLAGSOPT)
    
    all: foo.dll
    
    foo.dll: foo.obj
        $(LINK) $(LINKFLAGS) foo.obj /OUT:foo.dll
    
    svm.obj: svm.cpp svm.h
        $(CXX) $(CXXFLAGS) -c foo.cpp
    
    clean:
        -erase /Q *.obj *.dll *.exp *.lib
    



SConstruct (GCC)
----------------

You can use the following file with `SCons <http://www.scons.org>`__ to
build a shared library.



.. code-block:: python

    env = Environment()
    env.Replace(CFLAGS=['-O2','-Wall','-ansi','-pedantic'])
    env.SharedLibrary('foo', ['foo.cpp'])
    



foo.cpp
-------




.. code-block:: python

    #include <stdio.h>
    
    #ifdef FOO_DLL
    #ifdef FOO_EXPORTS
    #define FOO_API __declspec(dllexport)
    #else
    #define FOO_API __declspec(dllimport)
    #endif /* FOO_EXPORTS */
    #else
    #define FOO_API extern /* XXX confirm this */
    #endif /* FOO_DLL */
    
    #ifdef __cplusplus
    extern "C" {
    #endif
    
    extern FOO_API int bar(double* data, int len) {
       int i;
       printf("data = %p\n", (void*) data);
       for (i = 0; i < len; i++) {
          printf("data[%d] = %f\n", i, data[i]);
       }
       printf("len = %d\n", len);
       return len + 1;
    }
    
    #ifdef __cplusplus
    }
    #endif
    



When building the DLL for foo on Windows, define \`FOO\_DLL\` and
\`FOO\_EXPORTS\` (this is what you want to do when building a DLL for
use with ctypes). When linking against the DLL, define \`FOO\_DLL\`.
When linking against a static library that contains foo, or when
including foo in an executable, don't define anything.

If you're unclear about what is for, read
[http://www.tldp.org/HOWTO/C\ ++-dlopen/thesolution.html section 3 of
the C++ dlopen mini HOWTO]. This allows you to write function wrappers
with C linkage on top of a bunch of C++ classes so that you can use them
with ctypes. Alternatively, you might prefer to write C code.

foo.py
------




.. code-block:: python

    import numpy as N
    import ctypes as C
    _foo = N.ctypeslib.load_library('libfoo', '.')
    _foo.bar.restype = C.c_int
    _foo.bar.argtypes = [C.POINTER(C.c_double), C.c_int]
    def bar(x):
        return _foo.bar(x.ctypes.data_as(C.POINTER(C.c_double)), len(x))
    x = N.random.randn(10)
    n = bar(x)
    



NumPy arrays' ctypes property
=============================

A ctypes property was recently added to NumPy arrays:



.. code-block:: python

    In [18]: x = N.random.randn(2,3,4)
    
    In [19]: x.ctypes.data
    Out[19]: c_void_p(14394256)
    
    In [21]: x.ctypes.data_as(ctypes.POINTER(c_double))
    
    In [24]: x.ctypes.shape
    Out[24]: <ctypes._endian.c_long_Array_3 object at 0x00DEF2B0>
    
    In [25]: x.ctypes.shape[:3]
    Out[25]: [2, 3, 4]
    
    In [26]: x.ctypes.strides
    Out[26]: <ctypes._endian.c_long_Array_3 object at 0x00DEF300>
    
    In [27]: x.ctypes.strides[:3]
    Out[27]: [96, 32, 8]
    



In general, a C function might take a pointer to the array's data, an
integer indicating the number of array dimensions, (pass the value of
the ndim property here) and two int pointers to the shapes and stride
information.

If your C function assumes contiguous storage, you might want to wrap it
with a Python function that calls !NumPy's \`ascontiguousarray\`
function on all the input arrays.

NumPy's ndpointer with ctypes argtypes
======================================

Starting with ctypes 0.9.9.9, any class implementing the from\_param
method can be used in the argtypes list of a function. Before ctypes
calls a C function, it uses the argtypes list to check each parameter.

Using !NumPy's ndpointer function, some very useful argtypes classes can
be constructed, for example:



.. code-block:: python

    from numpy.ctypeslib import ndpointer
    arg1 = ndpointer(dtype='<f4')
    arg2 = ndpointer(ndim=2)
    arg3 = ndpointer(shape=(10,10))
    arg4 = ndpointer(flags='CONTIGUOUS,ALIGNED')
    # or any combination of the above
    arg5 = ndpointer(dtype='>i4', flags='CONTIGUOUS')
    func.argtypes = [arg1,arg2,arg3,arg4,arg5]
    



Now, if an argument doesn't meet the requirements, a !TypeError is
raised. This allows one to make sure that arrays passed to the C
function is in a form that the function can handle.

See also the mailing list thread on `ctypes and
ndpointer <http://thread.gmane.org/gmane.comp.python.numeric.general/7418/focus=7418>`__.

Dynamic allocation through callbacks
====================================

ctypes supports the idea of
`callbacks <http://docs.python.org/lib/ctypes-callback-functions.html>`__,
allowing C code to call back into Python through a function pointer.
This is possible because ctypes releases the Python Global Interpreter
Lock (GIL) before calling the C function.

We can use this feature to allocate !NumPy arrays if and when we need a
buffer for C code to operate on. This could avoid having to copy data in
certain cases. You also don't have to worry about freeing the C data
after you're done with it. By allocating your buffers as !NumPy arrays,
the Python garbage collector can take care of this.

Python code:



.. code-block:: python

    from ctypes import *
    ALLOCATOR = CFUNCTYPE(c_long, c_int, POINTER(c_int))
    # load your library as lib
    lib.baz.restype = None
    lib.baz.argtypes = [c_float, c_int, ALLOCATOR]
    



This isn't the prettiest way to define the allocator (I'm also not sure
if c\_long is the right return type), but there are a few bugs in ctypes
that seem to make this the only way at present. Eventually, we'd like to
write the allocator like this (but it doesn't work yet):



.. code-block:: python

    from numpy.ctypeslib import ndpointer
    ALLOCATOR = CFUNCTYPE(ndpointer('f4'), c_int, POINTER(c_int))
    



The following also seems to cause problems:



.. code-block:: python

    ALLOCATOR = CFUNCTYPE(POINTER(c_float), c_int, POINTER(c_int))
    ALLOCATOR = CFUNCTYPE(c_void_p, c_int, POINTER(c_int))
    ALLOCATOR = CFUNCTYPE(None, c_int, POINTER(c_int), POINTER(c_void_p))
    



Possible failures include a !SystemError exception being raised, the
interpreter crashing or the interpreter hanging. Check these mailing
list threads for more details: \* `Pointer-to-pointer unchanged when
assigning in
callback <http://thread.gmane.org/gmane.comp.python.ctypes/2979>`__ \*
`Hang with callback returning
POINTER(c\_float) <http://thread.gmane.org/gmane.comp.python.ctypes/2974>`__
\* `Error with callback function and as\_parameter with NumPy
ndpointer <http://thread.gmane.org/gmane.comp.python.ctypes/2972>`__

Time for an example. The C code for the example:



.. code-block:: python

    #ifndef CSPKREC_H
    #define CSPKREC_H
    #ifdef FOO_DLL
    #ifdef FOO_EXPORTS
    #define FOO_API __declspec(dllexport)
    #else
    #define FOO_API __declspec(dllimport)
    #endif
    #else
    #define FOO_API
    #endif
    #endif
    #include <stdio.h>
    #ifdef __cplusplus
    extern "C" {
    #endif
    
    typedef void*(*allocator_t)(int, int*);
    
    extern FOO_API void foo(allocator_t allocator) {
       int dim = 2;
       int shape[] = {2, 3};
       float* data = NULL;
       int i, j;
       printf("foo calling allocator\n");
       data = (float*) allocator(dim, shape);
       printf("allocator returned in foo\n");
       printf("data = 0x%p\n", data);
       for (i = 0; i < shape[0]; i++) {
          for (j = 0; j < shape[1]; j++) {
             *data++ = (i + 1) * (j + 1);
          }
       }
    }
    
    #ifdef __cplusplus
    }
    #endif
    



Check the `The Function Pointer
Tutorials <http://www.newty.de/fpt/index.html>`__ if you're new to
function pointers in C or C++. And the Python code:



.. code-block:: python

    from ctypes import *
    import numpy as N
    
    allocated_arrays = []
    def allocate(dim, shape):
        print 'allocate called'
        x = N.zeros(shape[:dim], 'f4')
        allocated_arrays.append(x)
        ptr = x.ctypes.data_as(c_void_p).value
        print hex(ptr)
        print 'allocate returning'
        return ptr
    
    lib = cdll['callback.dll']
    lib.foo.restype = None
    ALLOCATOR = CFUNCTYPE(c_long, c_int, POINTER(c_int))
    lib.foo.argtypes = [ALLOCATOR]
    
    print 'calling foo'
    lib.foo(ALLOCATOR(allocate))
    print 'foo returned'
    
    print allocated_arrays[0]
    



The allocate function creates a new !NumPy array and puts it in a list
so that we keep a reference to it after the callback function returns.
Expected output:



.. code-block:: python

    calling foo
    foo calling allocator
    allocate called
    0xaf5778
    allocate returning
    allocator returned in foo
    data = 0x00AF5778
    foo returned
    [[ 1.  2.  3.]
     [ 2.  4.  6.]]
    



Here's another idea for an Allocator class to manage this kind of thing.
In addition to dimension and shape, this allocator function takes a char
indicating what type of array to allocate. You can get these typecodes
from the ndarrayobject.h header, in the \`NPY\_TYPECHAR\` enum.



.. code-block:: python

    from ctypes import *
    import numpy as N
    
    class Allocator:
        CFUNCTYPE = CFUNCTYPE(c_long, c_int, POINTER(c_int), c_char)
    
        def __init__(self):
            self.allocated_arrays = []
    
        def __call__(self, dims, shape, dtype):
            x = N.empty(shape[:dims], N.dtype(dtype))
            self.allocated_arrays.append(x)
            return x.ctypes.data_as(c_void_p).value
    
        def getcfunc(self):
            return self.CFUNCTYPE(self)
        cfunc = property(getcfunc)
    



Use it like this in Python:



.. code-block:: python

    lib.func.argtypes = [..., Allocator.CFUNCTYPE]
    def func():
        alloc = Allocator()
        lib.func(..., alloc.cfunc)
        return tuple(alloc.allocated_arrays[:3])
    



Corresponding C code:



.. code-block:: python

    typedef void*(*allocator_t)(int, int*, char);
    
    void func(..., allocator_t allocator) {
       /* ... */
       int dims[] = {2, 3, 4};
       double* data = (double*) allocator(3, dims, 'd');
       /* allocate more arrays here */
    }
    



None of the allocators presented above are thread safe. If you have
multiple Python threads calling the C code that invokes your callbacks,
you will have to do something a bit smarter.

More useful code frags
======================

Suppose you have a C function like the following, which operates on a
pointer-to-pointers data structure.



.. code-block:: python

    void foo(float** data, int len) {
        float** x = data;
        for (int i = 0; i < len; i++, x++) {
            /* do something with *x */
        }
    }
    



You can create the necessary structure from an existing 2-D !NumPy array
using the following code:



.. code-block:: python

    x = N.array([[10,20,30], [40,50,60], [80,90,100]], 'f4')
    f4ptr = POINTER(c_float)
    data = (f4ptr*len(x))(*[row.ctypes.data_as(f4ptr) for row in x])
    



\`f4ptr\*len(x)\` creates a ctypes array type that is just large enough
to contain a pointer to every row of the array.

Heterogeneous Types Example
===========================

Here's a simple example when using heterogeneous dtypes (record arrays).

But, be warned that NumPy recarrays and corresponding structs in C
***may not*** be congruent.

Also structs are not standardized across platforms ...In other words,
**'' be aware of padding issues!**''

sample.c



.. code-block:: python

    #include <stdio.h>
    
    typedef struct Weather_t {
        int timestamp;
        char desc[12];
    } Weather;
    
    void print_weather(Weather* w, int nelems)
    {
        int i;
        for (i=0;i<nelems;++i) {
            printf("timestamp: %d\ndescription: %s\n\n", w[i].timestamp, w[i].desc);
    
        }
    }
    



SConstruct



.. code-block:: python

    env = Environment()
    env.Replace(CFLAGS=['-O2','-Wall','-ansi','-pedantic'])
    env.SharedLibrary('sample', ['sample.c'])
    



sample.py



.. code-block:: python

    import numpy as N
    import ctypes as C
    
    dat = [[1126877361,'sunny'], [1126877371,'rain'], [1126877385,'damn nasty'], [11
    26877387,'sunny']]
    
    dat_dtype = N.dtype([('timestamp','i4'),('desc','|S12')])
    arr = N.rec.fromrecords(dat,dtype=dat_dtype)
    
    _sample = N.ctypeslib.load_library('libsample','.')
    _sample.print_weather.restype = None
    _sample.print_weather.argtypes = [N.ctypeslib.ndpointer(dat_dtype, flags='aligne
    d, contiguous'), C.c_int]
    
    
    def print_weather(x):
        _sample.print_weather(x, x.size)
    
    
    
    if __name__=='__main__':
        print_weather(arr)
    



Fibonacci example (using NumPy arrays, C and Scons)
===================================================

The following was tested and works on Windows (using MinGW) and
GNU/Linux 32-bit OSs (last tested 13-08-2009). Copy all three files to
the same directory.

The C code (this calculates the Fibonacci number recursively):



.. code-block:: python

    /*
        Filename: fibonacci.c
        To be used with fibonacci.py, as an imported library. Use Scons to compile,
        simply type 'scons' in the same directory as this file (see www.scons.org).
    */
    
    /* Function prototypes */
    int fib(int a);
    void fibseries(int *a, int elements, int *series);
    void fibmatrix(int *a, int rows, int columns, int *matrix);
    
    int fib(int a)
    {
        if (a <= 0) /*  Error -- wrong input will return -1. */
            return -1;
        else if (a==1)
            return 0;
        else if ((a==2)||(a==3))
            return 1;
        else
            return fib(a - 2) + fib(a - 1);
    }
    
    void fibseries(int *a, int elements, int *series)
    {
        int i;
        for (i=0; i < elements; i++)
        {
        series[i] = fib(a[i]);
        }
    }
    
    void fibmatrix(int *a, int rows, int columns, int *matrix)
    {
        int i, j;
        for (i=0; i<rows; i++)
            for (j=0; j<columns; j++)
            {
                matrix[i * columns + j] = fib(a[i * columns + j]);
            }
    }
    



The Python code:



.. code-block:: python

    """
    Filename: fibonacci.py
    Demonstrates the use of ctypes with three functions:
    
        (1) fib(a)
        (2) fibseries(b)
        (3) fibmatrix(c)
    """
    
    import numpy as nm
    import ctypes as ct
    
    # Load the library as _libfibonacci.
    # Why the underscore (_) in front of _libfibonacci below?
    # To mimimise namespace pollution -- see PEP 8 (www.python.org).
    _libfibonacci = nm.ctypeslib.load_library('libfibonacci', '.')
    
    _libfibonacci.fib.argtypes = [ct.c_int] #  Declare arg type, same below.
    _libfibonacci.fib.restype  =  ct.c_int  #  Declare result type, same below.
    
    _libfibonacci.fibseries.argtypes = [nm.ctypeslib.ndpointer(dtype = nm.int),\
                                         ct.c_int,\
                                         nm.ctypeslib.ndpointer(dtype = nm.int)]
    _libfibonacci.fibseries.restype  = ct.c_void_p
    
    _libfibonacci.fibmatrix.argtypes = [nm.ctypeslib.ndpointer(dtype = nm.int),\
                                         ct.c_int, ct.c_int,\
                                        nm.ctypeslib.ndpointer(dtype = nm.int)]
    _libfibonacci.fibmatrix.restype  = ct.c_void_p
    
    def fib(a):
        """Compute the n'th Fibonacci number.
    
        ARGUMENT(S):
            An integer.
    
        RESULT(S):
            The n'th Fibonacci number.
    
        EXAMPLE(S):
        >>> fib(8)
        13
        >>> fib(23)
        17711
        >>> fib(0)
        -1
        """
        return _libfibonacci.fib(int(a))
    
    def fibseries(b):
        """Compute an array containing the n'th Fibonacci number of each entry.
    
        ARGUMENT(S):
            A list or NumPy array (dim = 1) of integers.
    
        RESULT(S):
            NumPy array containing the n'th Fibonacci number of each entry.
    
        EXAMPLE(S):
        >>> fibseries([1,2,3,4,5,6,7,8])
        array([ 0,  1,  1,  2,  3,  5,  8, 13])
        >>> fibseries(range(1,12))
        array([ 0,  1,  1,  2,  3,  5,  8, 13, 21, 34, 55])
        """
        b = nm.asarray(b, dtype=nm.intc)
        result = nm.empty(len(b), dtype=nm.intc)
        _libfibonacci.fibseries(b, len(b), result)
        return result
    
    def fibmatrix(c):
        """Compute a matrix containing the n'th Fibonacci number of each entry.
    
        ARGUMENT(S):
            A nested list or NumPy array (dim = 2) of integers.
    
        RESULT(S):
            NumPy array containing the n'th Fibonacci number of each entry.
    
        EXAMPLE(S):
        >>> from numpy import array
        >>> fibmatrix([[3,4],[5,6]])
        array([[1, 2],
               [3, 5]])
        >>> fibmatrix(array([[1,2,3],[4,5,6],[7,8,9]]))
        array([[ 0,  1,  1],
               [ 2,  3,  5],
               [ 8, 13, 21]])
        """
        tmp = nm.asarray(c)
        rows, cols = tmp.shape
        c = tmp.astype(nm.intc)
        result = nm.empty(c.shape, dtype=nm.intc)
        _libfibonacci.fibmatrix(c, rows, cols, result)
        return result
    



Here's the SConstruct file contents (filename: SConstruct):



.. code-block:: python

    env = Environment()
    env.Replace(CFLAGS=['-O2', '-Wall', '-ansi', '-pedantic'])
    env.SharedLibrary('libfibonacci', ['fibonacci.c'])
    



In Python interpreter (or whatever you use), do:



.. code-block:: python

    >>> import fibonacci as fb
    >>> fb.fib(8)
    13
    >>> fb.fibseries([5,13,2,6]
    array([  3, 144,   1,   5])
    



etc.

Pertinent Mailing List Threads
==============================

Some useful threads on the ctypes-users mailing list:

| ``* ``\ ```IndexError`` ``when`` ``indexing`` ``on``
``POINTER(POINTER(ctype))`` <http://aspn.activestate.com/ASPN/Mail/Message/ctypes-users/3119087>`__
| ``* ``\ ```Adding`` ``ctypes`` ``support`` ``to``
``NumPy`` <http://aspn.activestate.com/ASPN/Mail/Message/ctypes-users/3118513>`__
| ``* ``\ ```Determining`` ``if`` ``a`` ``ctype`` ``is`` ``a``
``pointer`` ``type`` ``(was`` ``RE:`` ``Adding`` ``ctypes`` ``support``
``to``
``NumPy)`` <http://aspn.activestate.com/ASPN/Mail/Message/ctypes-users/3118656>`__
| ``* ``\ ```Check`` ``for`` ``NULL`` ``pointer`` ``without``
``ValueError`` <http://aspn.activestate.com/ASPN/Mail/Message/ctypes-users/3117306>`__
| ``* ``\ ```Problem`` ``with`` ``callbacks`` ``from`` ``C`` ``into``
``Python`` <http://aspn.activestate.com/ASPN/Mail/Message/ctypes-users/3205951>`__
| ``* [``\ ```http://thread.gmane.org/gmane.comp.python.numeric.general/7418`` <http://thread.gmane.org/gmane.comp.python.numeric.general/7418>`__\ ``\ ctypes and ndpointer]``
| ``* ``\ ```Problems`` ``with`` ``64`` ``signed``
``integer`` <http://thread.gmane.org/gmane.comp.python.ctypes/3116>`__

Thomas Heller's answers are particularly insightful.

Documentation
=============

| ``* ``\ ```ctypes``
``tutorial`` <http://starship.python.net/crew/theller/ctypes/tutorial.html>`__
| ``* ``\ ```13.14`` ``ctypes`` ``--`` ``A`` ``foreign`` ``function``
``library`` ``for``
``Python.`` <http://docs.python.org/dev/lib/module-ctypes.html>`__

--------------

CategoryCookbook CategoryCookbook

