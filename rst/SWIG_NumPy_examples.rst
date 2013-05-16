Table of Contents
=================

TableOfContents

Introduction
============

These are simple !NumPy and SWIG examples which use the numpy.i
interface file. There is also a MinGW section for people who may want to
use these in a Win32 environment. The following code is C, rather than
C++.

The information contained here was first made available by Bill Spotz in
his article *numpy.i: a SWIG Interface File for !NumPy*, and the !NumPy
SVN which can be checked out using the following command:



.. code-block:: python

    svn co http://scipy.org/svn/numpy/trunk numpy
    



| `` * The !NumPy+SWIG manual is available here: ``\ ```1`` <http://scipy.org/svn/numpy/trunk/doc/swig/doc/numpy_swig.pdf>`__
| `` * The numpy.i file can be downloaded from the SVN: ``\ ```2`` <http://scipy.org/svn/numpy/trunk/doc/swig/numpy.i>`__
| `` * and the pyfragments.swg file, hich is also needed, is available from ``\ ```3`` <http://scipy.org/svn/numpy/trunk/doc/swig/pyfragments.swg>`__
| `` * These two files (and others) are also available in the numpy source tarball: ``\ ```4`` <http://sourceforge.net/project/showfiles.php?group_id=1369&package_id=175103>`__

Initial setup
=============

gcc and SWIG
------------

Check that both gcc and SWIG are available (paths known):



.. code-block:: python

    swig -version
    



and



.. code-block:: python

    gcc -v
    



Both should output some text...

Modifying the pyfragments.swg file (MinGW only)
-----------------------------------------------

This is from my own tests, running SWIG Version 1.3.36 and gcc version
3.4.5 (mingw-vista special r3). I had to remove the 'static' statements
from the source, otherwise your SWIGed sources won't compile. There are
only two 'static' statements in the file, both will need removing. Here
is my modified version:
`pyfragments.swg <http://ezwidgets.googlecode.com/svn/trunk/numpy/pyfragments.swg>`__

Compilation and testing
-----------------------

A setup.py file specific to each module must be written first. I based
mine on the reference setup.py available in
http://scipy.org/svn/numpy/trunk/doc/swig/test/ with added automatic
handling of swig.

On a un\*x like system, the command-line is:



.. code-block:: python

    python setup.py build
    



In a Win32 environment (either cygwin or cmd), the setup command-line is
(for use with MinGW):



.. code-block:: python

    python setup.py build --compiler=mingw32
    



The command handles both the SWIG process (generation of wrapper C and
Python code) and gcc compilation. The resulting module (a pyd file) is
built in the \`build\\lib.XXX\` directory (e.g. for a Python 2.5 install
and on a Win32 machine, the \`build\\lib.win32-2.5\` directory).

A simple ARGOUT\_ARRAY1 example
===============================

This is a re-implementation of the range function. The module is called
ezrange. One thing to remember with \`ARGOUT\_ARRAY1\` is that the
dimension of the array must be passed from Python.

From Bill Spotz's article: *The python user does not pass these arrays
in, they simply get returned. For the case where a dimension is
specified, the python user must provide that dimension as an argument.*

This is useful for functions like \`numpy.arange(N)\`, for which the
size of the returned array is known in advance and passed to the C
function.

For functions that follow \`array\_out = function(array\_in)\` where the
size of array\_out is *not* known in advance and depends on memory
allocated in C, see the example given in [:Cookbook/SWIG Memory
Deallocation].

The C source (ezrange.c and ezrange.h)
--------------------------------------

Here is the
`ezrange.h <http://ezwidgets.googlecode.com/svn/trunk/numpy/ezrange.h>`__
file:



.. code-block:: python

    void range(int *rangevec, int n);
    



Here is the
`ezrange.c <http://ezwidgets.googlecode.com/svn/trunk/numpy/ezrange.c>`__
file:



.. code-block:: python

    void range(int *rangevec, int n)
    {
        int i;
    
        for (i=0; i< n; i++)
            rangevec[i] = i;
    }
    



The interface file (ezrange.i)
------------------------------

Here is the
`ezrange.i <http://ezwidgets.googlecode.com/svn/trunk/numpy/ezrange.i>`__
file.



.. code-block:: python

    %module ezrange
    
    %{
        #define SWIG_FILE_WITH_INIT
        #include "ezrange.h"
    %}
    
    %include "numpy.i"
    
    %init %{
        import_array();
    %}
    
    %apply (int* ARGOUT_ARRAY1, int DIM1) {(int* rangevec, int n)}
    
    %include "ezrange.h"
    



Don't forget that you will also need the
`numpy.i <http://scipy.org/svn/numpy/trunk/doc/swig/numpy.i>`__ file in
the same directory.

Setup file (setup.py)
---------------------

This is my
`setup.py <http://ezwidgets.googlecode.com/svn/trunk/numpy/setup_range.py>`__
file:



.. code-block:: python

    #! /usr/bin/env python
    
    # System imports
    from distutils.core import *
    from distutils      import sysconfig
    
    # Third-party modules - we depend on numpy for everything
    import numpy
    
    # Obtain the numpy include directory.  This logic works across numpy versions.
    try:
        numpy_include = numpy.get_include()
    except AttributeError:
        numpy_include = numpy.get_numpy_include()
    
    # ezrange extension module
    _ezrange = Extension("_ezrange",
                       ["ezrange.i","ezrange.c"],
                       include_dirs = [numpy_include],
                       )
    
    # ezrange setup
    setup(  name        = "range function",
            description = "range takes an integer and returns an n element int array
     where each element is equal to its index",
            author      = "Egor Zindy",
            version     = "1.0",
            ext_modules = [_ezrange]
            )
    



Compiling the module
--------------------

The setup command-line is:



.. code-block:: python

    python setup.py build
    



or



.. code-block:: python

    python setup.py build --compiler=mingw32
    



depending on your environment.

Testing the module
------------------

If everything goes according to plan, there should be a
\`\_ezrange.pyd\` file available in the \`build\\lib.XXX\` directory.
You will need to copy the file in the directory where the \`ezrange.py\`
file is (generated by swig), in which case, the following will work (in
python):



.. code-block:: python

    >>> import ezrange
    >>> ezrange.range(10)
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    



A simple INPLACE\_ARRAY1 example
================================

This example doubles the elements of the 1-D array passed to it. The
operation is done in-place, which means that the array passed to the
function is changed.

The C source (inplace.c and inplace.h)
--------------------------------------

Here is the
`inplace.h <http://ezwidgets.googlecode.com/svn/trunk/numpy/inplace.h>`__
file:



.. code-block:: python

    void inplace(double *invec, int n);
    



Here is the
`inplace.c <http://ezwidgets.googlecode.com/svn/trunk/numpy/inplace.c>`__
file:



.. code-block:: python

    void inplace(double *invec, int n)
    {
        int i;
    
        for (i=0; i<n; i++)
        {
            invec[i] = 2*invec[i];
        }
    }
    



The interface file (inplace.i)
------------------------------

Here is the
`inplace.i <http://ezwidgets.googlecode.com/svn/trunk/numpy/inplace.i>`__
interface file:



.. code-block:: python

    %module inplace
    
    %{
        #define SWIG_FILE_WITH_INIT
        #include "inplace.h"
    %}
    
    %include "numpy.i"
    
    %init %{
        import_array();
    %}
    
    %apply (double* INPLACE_ARRAY1, int DIM1) {(double* invec, int n)}
    %include "inplace.h"
    



Setup file (setup.py)
---------------------

This is my
`setup.py <http://ezwidgets.googlecode.com/svn/trunk/numpy/setup_inplace.py>`__
file:



.. code-block:: python

    #! /usr/bin/env python
    
    # System imports
    from distutils.core import *
    from distutils      import sysconfig
    
    # Third-party modules - we depend on numpy for everything
    import numpy
    
    # Obtain the numpy include directory.  This logic works across numpy versions.
    try:
        numpy_include = numpy.get_include()
    except AttributeError:
        numpy_include = numpy.get_numpy_include()
    
    # inplace extension module
    _inplace = Extension("_inplace",
                       ["inplace.i","inplace.c"],
                       include_dirs = [numpy_include],
                       )
    
    # NumyTypemapTests setup
    setup(  name        = "inplace function",
            description = "inplace takes a double array and doubles each of its elem
    ents in-place.",
    
            author      = "Egor Zindy",
            version     = "1.0",
            ext_modules = [_inplace]
            )
    



Compiling the module
--------------------

The setup command-line is:



.. code-block:: python

    python setup.py build
    



or



.. code-block:: python

    python setup.py build --compiler=mingw32
    



depending on your environment.

Testing the module
------------------

If everything goes according to plan, there should be a
\`\_inplace.pyd\` file available in the \`build\\lib.XXX\` directory.
You will need to copy the file in the directory where the \`inplace.py\`
file is (generated by swig), in which case, the following will work (in
python):



.. code-block:: python

    >>> import numpy
    >>> import inplace
    >>> a = numpy.array([1,2,3],'d')
    >>> inplace.inplace(a)
    >>> a
    array([2., 4., 6.])
    



A simple ARGOUTVIEW\_ARRAY1 example
===================================

Big fat multiple warnings
-------------------------

Please note, Bill Spotz advises against the use of argout\_view arrays,
unless absolutely necessary:

`` Argoutview arrays are for when your C code provides you with a view of its internal data and does not require any memory to be allocated by the user. This can be dangerous. There is almost no way to guarantee that the internal data from the C code will remain in existence for the entire lifetime of the !NumPy array that encapsulates it. If the user destroys the object that provides the view of the data before destroying the !NumPy array, then using that array my result in bad memory references or segmentation faults. Nevertheless, there are situations, working with large data sets, where you simply have no other choice.``

Python does not take care of memory de-allocation, as stated here by
Travis Oliphant: `1 <http://blog.enthought.com/?p=62>`__

`` The tricky part, however, is memory management. How does the memory get deallocated? The suggestions have always been something similar to “make sure the memory doesn’t get deallocated before the !NumPy array disappears.” This is nice advice, but not generally helpful as it basically just tells you to create a memory leak.``

Memory deallocation is also difficult to handle automatically as there
is no easy way to do module "finalization". There is a
\`Py\_InitModule()\` function, but nothing to handle
deletion/destruction/finalization (this will be addressed in Python 3000
as stated in `PEP3121 <http://www.python.org/dev/peps/pep-3121/>`__. In
my example, I use the python `module
atexit <http://www.python.org/doc/2.5.2/lib/module-atexit.html>`__ but
there must be a better way.

Having said all that, if you have no other choice, here is an example
that uses ARGOUTVIEW\_ARRAY1. As usual, comments welcome!

The module declares a block of memory and a couple of functions: \*
ezview.set\_ones() sets all the elements (doubles) in the memory block
to one and returns a numpy array that is a VIEW of the memory block. \*
ezview.get\_view() simply returns a view of the memory block. \*
ezview.finalize() takes care of the memory deallocation (this is the
weak part of this example).

The C source (ezview.c and ezview.h)
------------------------------------

Here is the
`ezview.h <http://ezwidgets.googlecode.com/svn/trunk/numpy/ezview.h>`__
file:



.. code-block:: python

    void set_ones(double *array, int n);
    



Here is the
`ezview.c <http://ezwidgets.googlecode.com/svn/trunk/numpy/ezview.c>`__
file:



.. code-block:: python

    #include <stdio.h>
    #include <stdlib.h> 
    
    #include "ezview.h"
    
    void set_ones(double *array, int n)
    {
        int i;
    
        if (array == NULL)
            return;
    
        for (i=0;i<n;i++)
            array[i] = 1.;
    }
    



The interface file (ezview.i)
-----------------------------

Here is the
`ezview.i <http://ezwidgets.googlecode.com/svn/trunk/numpy/ezview.i>`__
interface file:



.. code-block:: python

    %module ezview
    
    %{
        #define SWIG_FILE_WITH_INIT
        #include "ezview.h"
    
        double *my_array = NULL;
        int my_n = 10;
    
        void __call_at_begining()
        {
            printf("__call_at_begining...\n");
            my_array = (double *)malloc(my_n*sizeof(double));
        }
    
        void __call_at_end(void)
        {
            printf("__call_at_end...\n");
            if (my_array != NULL)
                free(my_array);
        }
    %}
    
    %include "numpy.i"
    
    %init %{
        import_array();
        __call_at_begining();
    %}
    
    %apply (double** ARGOUTVIEW_ARRAY1, int *DIM1) {(double** vec, int* n)}
    
    %include "ezview.h"
    %rename (set_ones) my_set_ones;
    
    %inline %{
    void finalize(void){
        __call_at_end();
    }
    
    void get_view(double **vec, int* n) {
        *vec = my_array;
        *n = my_n;
    }
    
    void my_set_ones(double **vec, int* n) {
        set_ones(my_array,my_n);
        *vec = my_array;
        *n = my_n;
    }
    %}
    



Don't forget that you will also need the numpy.i file in the same
directory.

Setup file (setup.py)
---------------------

This is my
`setup.py <http://ezwidgets.googlecode.com/svn/trunk/numpy/setup_ezview.py>`__
file:



.. code-block:: python

    #! /usr/bin/env python
    
    # System imports
    from distutils.core import *
    from distutils      import sysconfig
    
    # Third-party modules - we depend on numpy for everything
    import numpy
    
    # Obtain the numpy include directory.  This logic works across numpy versions.
    try:
        numpy_include = numpy.get_include()
    except AttributeError:
        numpy_include = numpy.get_numpy_include()
    
    # view extension module
    _ezview = Extension("_ezview",
                       ["ezview.i","ezview.c"],
                       include_dirs = [numpy_include],
                       )
    
    # NumyTypemapTests setup
    setup(  name        = "ezview module",
            description = "ezview provides 3 functions: set_ones(), get_view() and f
    inalize(). set_ones() and get_view() provide a view on a memory block allocated 
    in C, finalize() takes care of the memory deallocation.",
            author      = "Egor Zindy",
            version     = "1.0",
            ext_modules = [_ezview]
            )
    



Compiling the module
--------------------

The setup command-line is:



.. code-block:: python

    python setup.py build
    



or



.. code-block:: python

    python setup.py build --compiler=mingw32
    



depending on your environment.

Testing the module
------------------

If everything goes according to plan, there should be a \`\_ezview.pyd\`
file available in the \`build\\lib.XXX\` directory. You will need to
copy the file in the directory where the \`ezview.py\` file is
(generated by swig), in which case, the following will work (in python):

The test code
`test\_ezview.py <http://ezwidgets.googlecode.com/svn/trunk/numpy/test_ezview.py>`__
follows:



.. code-block:: python

    import atexit
    import numpy
    print "first message is from __call_at_begining()"
    import ezview
    
    #There is no easy way to finalize the module (see PEP3121)
    atexit.register(ezview.finalize)
    
    a = ezview.set_ones()
    print "\ncalling ezview.set_ones() - now the memory block is all ones.\nReturned
     array (a view on the allocated memory block) is:"
    print a
    
    print "\nwe're setting the array using a[:]=arange(a.shape[0])\nThis changes the
     content of the allocated memory block:"
    a[:] = numpy.arange(a.shape[0])
    print a
    
    print "\nwe're now deleting the array  - this only deletes the view,\nnot the al
    located memory!"
    del a
    
    print "\nlet's get a new view on the allocated memory, should STILL contain [0,1
    ,2,3...]"
    b = ezview.get_view()
    print b
    
    print "\nnext message from __call_at_end() - finalize() registered via module at
    exit"
    



Launch test\_ezview.py and the following will hopefully happen:



.. code-block:: python

    ~> python test_ezview.py
    first message is from __call_at_begining()
    __call_at_begining...
    
    calling ezview.set_ones() - now the memory block is all ones.
    Returned array (a view on the allocated memory block) is:
    [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]
    
    we re setting the array using a[:]=arange(a.shape[0])
    This changes the content of the allocated memory block:
    [ 0.  1.  2.  3.  4.  5.  6.  7.  8.  9.]
    
    we re now deleting the array  - this only deletes the view,
    not the allocated memory!
    
    let s get a new view on the allocated memory, should STILL contain [0,1,2,3...]
    [ 0.  1.  2.  3.  4.  5.  6.  7.  8.  9.]
    
    next message from __call_at_end() - finalize() registered via module atexit
    __call_at_end...
    



Error handling using errno and python exceptions
================================================

I have been testing this for a few months now and this is the best I've
come-up with. If anyone knows of a better way, please let me know.

From the opengroup website, the lvalue errno is used by many functions
to return error values. The idea is that the global variable errno is
set to 0 before a function is called (in swig parlance: $action), and
checked afterwards. If errno is non-zero, a python exception with a
meaningful message is generated, depending on the value of errno.

The following example comprises two examples: First example uses errno
when checking whether an array index is valid. Second example uses errno
to notify the user of a malloc() problem.

The C source (ezerr.c and ezerr.h)
----------------------------------

Here is the
`ezerr.h <http://ezwidgets.googlecode.com/svn/trunk/numpy/ezerr.h>`__
file:



.. code-block:: python

    int val(int *array, int n, int index);
    void alloc(int n);
    



Here is the
`ezerr.c <http://ezwidgets.googlecode.com/svn/trunk/numpy/ezerr.c>`__
file:



.. code-block:: python

    #include <stdlib.h>
    #include <errno.h>
    
    #include "ezerr.h"
    
    //return the array element defined by index
    int val(int *array, int n, int index)
    {
        int value=0;
    
        if (index < 0 || index >=n)
        {
            errno = EPERM;
            goto end;
        }
    
        value = array[index];
    
    end:
        return value;
    }
    
    //allocate (and free) a char array of size n
    void alloc(int n)
    {
        char *array;
    
        array = (char *)malloc(n*sizeof(char));
        if (array == NULL)
        {
            errno = ENOMEM;
            goto end;
        }
    
        //don't keep the memory allocated...
        free(array);
    
    end:
        return;
    }
    



The interface file (ezerr.i)
----------------------------

Here is the
`ezerr.i <http://ezwidgets.googlecode.com/svn/trunk/numpy/ezerr.i>`__
interface file:



.. code-block:: python

    %module ezerr
    %{
    #include <errno.h>
    #include "ezerr.h"
    
    #define SWIG_FILE_WITH_INIT
    %}
    
    %include "numpy.i"
    
    %init %{
        import_array();
    %}
    
    %exception
    {
        errno = 0;
        $action
    
        if (errno != 0)
        {
            switch(errno)
            {
                case EPERM:
                    PyErr_Format(PyExc_IndexError, "Index out of range");
                    break;
                case ENOMEM:
                    PyErr_Format(PyExc_MemoryError, "Failed malloc()");
                    break;
                default:
                    PyErr_Format(PyExc_Exception, "Unknown exception");
            }
            SWIG_fail;
        }
    }
    
    %apply (int* IN_ARRAY1, int DIM1) {(int *array, int n)}
    
    %include "ezerr.h"
    



Note the *SWIG\_fail*, which is a macro for *goto fail* in case there is
any other cleanup code to execute (thanks Bill!).

Don't forget that you will also need the numpy.i file in the same
directory.

Setup file (setup.py)
---------------------

This is my
`setup.py <http://ezwidgets.googlecode.com/svn/trunk/numpy/setup_err.py>`__
file:



.. code-block:: python

    #! /usr/bin/env python
    
    # System imports
    from distutils.core import *
    from distutils      import sysconfig
    
    # Third-party modules - we depend on numpy for everything
    import numpy
    
    # Obtain the numpy include directory.  This logic works across numpy versions.
    try:
        numpy_include = numpy.get_include()
    except AttributeError:
        numpy_include = numpy.get_numpy_include()
    
    # err extension module
    ezerr = Extension("_ezerr",
                       ["ezerr.i","ezerr.c"],
                       include_dirs = [numpy_include],
    
                       extra_compile_args = ["--verbose"]
                       )
    
    # NumyTypemapTests setup
    setup(  name        = "err test",
            description = "A simple test to demonstrate the use of errno and python 
    exceptions",
            author      = "Egor Zindy",
            version     = "1.0",
            ext_modules = [ezerr]
            )
    



Compiling the module
--------------------

The setup command-line is:



.. code-block:: python

    python setup.py build
    



or



.. code-block:: python

    python setup.py build --compiler=mingw32
    



depending on your environment.

Testing the module
------------------

If everything goes according to plan, there should be a \`\_ezerr.pyd\`
file available in the \`build\\lib.XXX\` directory. You will need to
copy the file in the directory where the \`ezerr.py\` file is (generated
by swig), in which case, the following will work (in python):

The test code
`test\_err.py <http://ezwidgets.googlecode.com/svn/trunk/numpy/test_err.py>`__
follows:



.. code-block:: python

    
    import traceback,sys
    import numpy
    import ezerr
    
    print "\n--- testing ezerr.val() ---"
    a = numpy.arange(10)
    indexes = [5,20,-1]
    
    for i in indexes:
        try:
            value = ezerr.val(a,i)
        except:
            print ">> failed for index=%d" % i
            traceback.print_exc(file=sys.stdout)
        else:
            print "using ezerr.val() a[%d]=%d - should be %d" % (i,value,a[i])
    
    print "\n--- testing ezerr.alloc() ---"
    amounts = [1,-1] #1 byte, -1 byte
    
    for n in amounts:
        try:
            ezerr.alloc(n)
        except:
            print ">> could not allocate %d bytes" % n
            traceback.print_exc(file=sys.stdout)
        else:
            print "allocated (and deallocated) %d bytes" % n
    



Launch test\_err.py and the following will hopefully happen:



.. code-block:: python

    ~> python test_err.py
    
    --- testing ezerr.val() ---
    using ezerr.val() a[5]=5 - should be 5
    >> failed for index=20
    Traceback (most recent call last):
      File "test_err.py", line 11, in <module>
        value = ezerr.val(a,i)
    IndexError: Index out of range
    >> failed for index=-1
    Traceback (most recent call last):
      File "test_err.py", line 11, in <module>
        value = ezerr.val(a,i)
    IndexError: Index out of range
    
    --- testing ezerr.alloc() ---
    allocated (and deallocated) 1 bytes
    >> could not allocate -1 bytes
    Traceback (most recent call last):
      File "test_err.py", line 23, in <module>
        ezerr.alloc(n)
    MemoryError: Failed malloc()
    



Dot product example (from Bill Spotz's article)
===============================================

The last example given in Bill Spotz's artice is for a dot product
function. Here is a fleshed-out version.

The C source (dot.c and dot.h)
------------------------------

Here is the
`dot.h <http://ezwidgets.googlecode.com/svn/trunk/numpy/dot.h>`__ file:



.. code-block:: python

    double dot(int len, double* vec1, double* vec2);
    



Here is the
`dot.c <http://ezwidgets.googlecode.com/svn/trunk/numpy/dot.c>`__ file:



.. code-block:: python

    #include <stdio.h>
    #include "dot.h"
    
    double dot(int len, double* vec1, double* vec2)
    {
        int i;
        double d;
    
        d = 0;
        for(i=0;i<len;i++)
            d += vec1[i]*vec2[i];
    
        return d;
    }
    



The interface files (dot.i and numpy.i)
---------------------------------------

Here is the complete
`dot.i <http://ezwidgets.googlecode.com/svn/trunk/numpy/dot.i>`__ file:



.. code-block:: python

    %module dot
    
    %{
        #define SWIG_FILE_WITH_INIT
        #include "dot.h"
    %}
    
    %include "numpy.i"
    
    %init %{
        import_array();
    %}
    
    %apply (int DIM1, double* IN_ARRAY1) {(int len1, double* vec1), (int len2, doubl
    e* vec2)}
    
    
    %include "dot.h"
    %rename (dot) my_dot;
    
    %inline %{
        double my_dot(int len1, double* vec1, int len2, double* vec2) {
        if (len1 != len2) {
            PyErr_Format(PyExc_ValueError, "Arrays of lengths (%d,%d) given", len1, 
    len2);
            return 0.0;
        }
        return dot(len1, vec1, vec2);
    }
    %}
    



Setup file (setup.py)
---------------------

This is the
`setup.py <http://ezwidgets.googlecode.com/svn/trunk/numpy/setup_dot.py>`__
file:



.. code-block:: python

    #! /usr/bin/env python
    
    # System imports
    from distutils.core import *
    from distutils      import sysconfig
    
    # Third-party modules - we depend on numpy for everything
    import numpy
    
    # Obtain the numpy include directory.  This logic works across numpy versions.
    try:
        numpy_include = numpy.get_include()
    except AttributeError:
        numpy_include = numpy.get_numpy_include()
    
    # dot extension module
    _dot = Extension("_dot",
                       ["dot.i","dot.c"],
                       include_dirs = [numpy_include],
                       )
    
    # dot setup
    setup(  name        = "Dot product",
            description = "Function that performs a dot product (numpy.i: a SWIG Int
    erface File for NumPy)",
            author      = "Egor Zindy (based on the setup.py file available in the n
    umpy tree)",
            version     = "1.0",
            ext_modules = [_dot]
            )
    



Compiling the module
--------------------

The setup command-line is:



.. code-block:: python

    python setup.py build
    



or



.. code-block:: python

    python setup.py build --compiler=mingw32
    



depending on your environment.

Testing
-------

If everything goes according to plan, there should be a \`\_dot.pyd\`
file available in the \`build\\lib.XXX\` directory. You will need to
copy the file in the directory where the \`dot.py\` file is (generated
by swig), in which case, the following will work (in python):



.. code-block:: python

    >>> import dot
    >>> dot.dot([1,2,3],[1,2,3])
    14.0
    



Conclusion
==========

That's all folks (for now)! As usual, comments welcome!

`` * ``\ **``TODO``**\ ``: Code clean-up and moving the examples over to the !SciPy/!NumPy repository?``

Regards, Egor

--------------

CategoryCookbook

