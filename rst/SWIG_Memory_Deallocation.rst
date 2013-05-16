Table of Contents
=================

TableOfContents

Introduction
============

Recipe description
------------------

This cookbook recipe describes the automatic deallocation of memory
blocks allocated via \`malloc()\` calls in C, when the corresponding
Python numpy array objects are destroyed. The recipe uses SWIG and a
modified \`numpy.i\` helper file.

To be more specific, new fragments were added to the existing
\`numpy.i\` to handle automatic deallocation of arrays, the size of
which is not known in advance. As with the original fragments, a block
of \`malloc()\` memory can be converted into a returned numpy python
object via a call to \`PyArray\_SimpleNewFromData()\`. However, the
returned python object is created using \`PyCObject\_FromVoidPtr()\`,
which ensures that the allocated memory is automatically disposed of
when the Python object is destroyed. Examples below show how using these
new fragments avoids leaking memory.

Since the new fragments are based on the \`\_ARGOUTVIEW\_\` ones, the
name \`\_ARGOUTVIEWM\_\` was chosen, where \`M\` stands for managed. All
managed fragments (ARRAY1, 2 and 3, FARRAY1, 2 and 3) were implemented,
and have now been extensively tested.

Where to get the files
----------------------

At the moment, the modified numpy.i file is available here (last updated
2012-04-22): \* http://ezwidgets.googlecode.com/svn/trunk/numpy/numpy.i
\* http://ezwidgets.googlecode.com/svn/trunk/numpy/pyfragments.swg

How the code came about
-----------------------

The original memory deallocation code was written by Travis Oliphant
(see http://blog.enthought.com/?p=62 ) and as far as I know, these
clever people were the first ones to use it in a swig file (see
http://niftilib.sourceforge.net/pynifti, file nifticlib.i). Lisandro
Dalcin then pointed out a simplified implementation using
`CObjects <http://docs.python.org/c-api/cobject.html>`__, which Travis
details in this `updated blog
post <http://blog.enthought.com/python/numpy/simplified-creation-of-numpy-arrays-from-pre-allocated-memory/>`__.

How to use the new fragments
============================

Important steps
---------------

In yourfile.i, the %init function uses the same \`import\_array()\` call
you already know:



.. code-block:: python

    %init %{
        import_array();
    %}
    



... then just use ARGOUTVIEWM\_ARRAY1 instead of ARGOUTVIEW\_ARRAY1 and
memory deallocation is handled automatically when the python array is
destroyed (see examples below).

A simple ARGOUTVIEWM\_ARRAY1 example
====================================

The SWIG-wrapped function in C creates an N integers array, using
\`malloc()\` to allocate memory. From python, this function is
repetitively called and the array created destroyed (M times).

Using the ARGOUTVIEW\_ARRAY1 provided in numpy.i, this will create
memory leaks (I know ARGOUTVIEW\_ARRAY1 has not been designed for this
purpose but it's tempting!).

Using the ARGOUTVIEWM\_ARRAY1 fragment instead, the memory allocated
with \`malloc()\` will be automatically deallocated when the array is
deleted.

The python test program creates and deletes a 1024^2 ints array 2048
times using both ARGOUTVIEW\_ARRAY1 and ARGOUTVIEWM\_ARRAY1 and when
memory allocation fails, an exception is generated in C and caught in
Python, showing which iteration finally caused the allocation to fail.

The C source (ezalloc.c and ezalloc.h)
--------------------------------------

Here is the
`ezalloc.h <http://ezwidgets.googlecode.com/svn/trunk/numpy/ezalloc.h>`__
file:



.. code-block:: python

    void alloc(int ni, int** veco, int *n);
    



Here is the
`ezalloc.c <http://ezwidgets.googlecode.com/svn/trunk/numpy/ezalloc.c>`__
file:



.. code-block:: python

    #include <stdio.h>
    #include <errno.h>
    #include "ezalloc.h"
    
    void alloc(int ni, int** veco, int *n)
    {
        int *temp;
        temp = (int *)malloc(ni*sizeof(int));
    
        if (temp == NULL)
            errno = ENOMEM;
    
        //veco is either NULL or pointing to the allocated block of memory...
        *veco = temp;
        *n = ni;
    }
    



The interface file (ezalloc.i)
------------------------------

The file (available here:
`ezalloc.i <http://ezwidgets.googlecode.com/svn/trunk/numpy/ezalloc.i>`__)
does a couple of interesting things: \* Like I said in the introduction,
calling the \`import\_array()\` function in the \`%init\` section now
also initialises the memory deallocation code. There is nothing else to
add here. \* An exception is generated if memory allocation fails. After
a few iterations of the code construct, using \`errno\` and
\`SWIG\_fail\` is the simplest I've come-up with. \* In this example,
two inline functions are created, one using ARGOUTVIEW\_ARRAY1 and the
other ARGOUTVIEWM\_ARRAY1. Both function use the \`alloc()\` function
(see ezalloc.h and ezalloc.c).



.. code-block:: python

    %module ezalloc
    %{
    #include <errno.h>
    #include "ezalloc.h"
    
    #define SWIG_FILE_WITH_INIT
    %}
    
    %include "numpy.i"
    
    %init %{
        import_array();
    %}
    
    %apply (int** ARGOUTVIEWM_ARRAY1, int *DIM1) {(int** veco1, int* n1)}
    %apply (int** ARGOUTVIEW_ARRAY1, int *DIM1) {(int** veco2, int* n2)}
    
    %include "ezalloc.h"
    
    %exception
    {
        errno = 0;
        $action
    
        if (errno != 0)
        {
            switch(errno)
            {
                case ENOMEM:
                    PyErr_Format(PyExc_MemoryError, "Failed malloc()");
                    break;
                default:
                    PyErr_Format(PyExc_Exception, "Unknown exception");
            }
            SWIG_fail;
        }
    }
    
    %rename (alloc_managed) my_alloc1;
    %rename (alloc_leaking) my_alloc2;
    
    %inline %{
    
    void my_alloc1(int ni, int** veco1, int *n1)
    {
        /* The function... */
        alloc(ni, veco1, n1);
    }
    
    void my_alloc2(int ni, int** veco2, int *n2)
    {
        /* The function... */
        alloc(ni, veco2, n2);
    }
    
    %}
    



Don't forget that you will need the
`numpy.i <http://ezwidgets.googlecode.com/svn/trunk/numpy/numpy.i>`__
file in the same directory for this to compile.

Setup file (setup\_alloc.py)
----------------------------

This is the
`setup\_alloc.py <http://ezwidgets.googlecode.com/svn/trunk/numpy/setup_alloc.py>`__
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
    
    # alloc extension module
    _ezalloc = Extension("_ezalloc",
                       ["ezalloc.i","ezalloc.c"],
                       include_dirs = [numpy_include],
    
                       extra_compile_args = ["--verbose"]
                       )
    
    # NumyTypemapTests setup
    setup(  name        = "alloc functions",
            description = "Testing managed arrays",
            author      = "Egor Zindy",
            version     = "1.0",
            ext_modules = [_ezalloc]
            )
    



Compiling the module
--------------------

The setup command-line is (in Windows, using mingw):



.. code-block:: python

    $> python setup_alloc.py build --compiler=mingw32
    



or in UN\*X, simply



.. code-block:: python

    $> python setup_alloc.py build
    



Testing the module
------------------

If everything goes according to plan, there should be a
\`\_ezalloc.pyd\` file available in the \`build\\lib.XXX\` directory.
The file needs to be copied in the directory with the \`ezalloc.py\`
file (generated by swig).

A python test program is provided in the SVN repository
(`test\_alloc.py <http://ezwidgets.googlecode.com/svn/trunk/numpy/test_alloc.py>`__)
and reproduced below:



.. code-block:: python

    import ezalloc
    
    n = 2048
    
    # this multiplied by sizeof(int) to get size in bytes...
    #assuming sizeof(int)=4 on a 32bit machine (sorry, it's late!)
    m = 1024 * 1024
    err = 0
    
    print "ARGOUTVIEWM_ARRAY1 (managed arrays) - %d allocations (%d bytes each)" % (
    n,4*m)
    for i in range(n):
        try:
            #allocating some memory
            a = ezalloc.alloc_managed(m)
            #deleting the array
            del a
        except:
            err = 1
            print "Step %d failed" % i
            break
    
    if err == 0:
        print "Done!\n"
    
    print "ARGOUTVIEW_ARRAY1 (unmanaged, leaking) - %d allocations (%d bytes each)" 
    % (n,4*m)
    for i in range(n):
        try:
            #allocating some memory
            a = ezalloc.alloc_leaking(m)
            #deleting the array
            del a
        except:
            err = 1
            print "Step %d failed" % i
            break
    
    if err == 0:
        print "Done? Increase n!\n"
    



Then, a



.. code-block:: python

    $> python test_alloc.py
    



will produce an output similar to this:



.. code-block:: python

    ARGOUTVIEWM_ARRAY1 (managed arrays) - 2048 allocations (4194304 bytes each)
    Done!
    
    ARGOUTVIEW_ARRAY1 (unmanaged, leaking) - 2048 allocations (4194304 bytes each)
    Step 483 failed
    



The unmanaged array leaks memory every time the array view is deleted.
The managed one will delete the memory block seamlessly. This was tested
both in Windows XP and Linux.

A simple ARGOUTVIEWM\_ARRAY2 example
====================================

The following examples shows how to return a two-dimensional array from
C which also benefits from the automatic memory deallocation.

A naive "crop" function is wrapped using SWIG/numpy.i and returns a
slice of the input array. When used as \`array\_out =
crop.crop(array\_in, d1\_0,d1\_1, d2\_0,d2\_1)\`, it is equivalent to
the native numpy slicing \`array\_out = array\_in[d1\_0:d1\_1,
d2\_0:d2\_1]\`.

The C source (crop.c and crop.h)
--------------------------------

Here is the
`crop.h <http://ezwidgets.googlecode.com/svn/trunk/numpy/crop.h>`__
file:



.. code-block:: python

    void crop(int *arr_in, int dim1, int dim2, int d1_0, int d1_1, int d2_0, int d2_
    1, int **arr_out, int *dim1_out, int *dim2_out);
    



Here is the
`crop.c <http://ezwidgets.googlecode.com/svn/trunk/numpy/crop.c>`__
file:



.. code-block:: python

    #include <stdlib.h>
    #include <errno.h>
    
    #include "crop.h"
    
    void crop(int *arr_in, int dim1, int dim2, int d1_0, int d1_1, int d2_0, int d2_
    1, int **arr_out, int *dim1_out, int *dim2_out)
    {
        int *arr=NULL;
        int dim1_o=0;
        int dim2_o=0;
        int i,j;
    
        //value checks
        if ((d1_1 < d1_0) || (d2_1 < d2_0) ||
            (d1_0 >= dim1) || (d1_1 >= dim1) || (d1_0 < 0) || (d1_1 < 0) ||
            (d2_0 >= dim2) || (d2_1 >= dim2) || (d2_0 < 0) || (d2_1 < 0))
        {
            errno = EPERM;
            goto end;
        }
    
        //output sizes
        dim1_o = d1_1-d1_0;
        dim2_o = d2_1-d2_0;
    
        //memory allocation
        arr = (int *)malloc(dim1_o*dim2_o*sizeof(int));
        if (arr == NULL)
        {
            errno = ENOMEM;
            goto end;
        }
    
        //copying the cropped arr_in region to arr (naive implementation)
        printf("\n--- d1_0=%d d1_1=%d (rows)  -- d2_0=%d d2_1=%d (columns)\n",d1_0,d
    1_1,d2_0,d2_1);
        for (j=0; j<dim1_o; j++)
        {
            for (i=0; i<dim2_o; i++)
            {
                arr[j*dim2_o+i] = arr_in[(j+d1_0)*dim2+(i+d2_0)];
                printf("%d ",arr[j*dim2_o+i]);
            }
            printf("\n");
        }
        printf("---\n\n");
    
    end:
        *dim1_out = dim1_o;
        *dim2_out = dim2_o;
        *arr_out = arr;
    }
    



The interface file (crop.i)
---------------------------

The file (available here:
`crop.i <http://ezwidgets.googlecode.com/svn/trunk/numpy/crop.i>`__)
does a couple of interesting things: \* The array dimensions DIM1 and
DIM2 are in the same order as array.shape on the Python side. In a row
major array definition for an image, DIM1 would the number of rows and
DIM2 the number of columns. \* Using the errno library, An exception is
generated when memory allocation fails (ENOMEM) or when a problem occurs
with the indexes (EPERM).



.. code-block:: python

    %module crop
    %{
    #include <errno.h>
    #include "crop.h"
    
    #define SWIG_FILE_WITH_INIT
    %}
    
    %include "numpy.i"
    
    %init %{
        import_array();
    %}
    
    %exception crop
    {
        errno = 0;
        $action
    
        if (errno != 0)
        {
            switch(errno)
            {
                case EPERM:
                    PyErr_Format(PyExc_IndexError, "Index error");
                    break;
                case ENOMEM:
                    PyErr_Format(PyExc_MemoryError, "Not enough memory");
                    break;
                default:
                    PyErr_Format(PyExc_Exception, "Unknown exception");
            }
            SWIG_fail;
        }
    }
    
    %apply (int* IN_ARRAY2, int DIM1, int DIM2) {(int *arr_in, int dim1, int dim2)}
    %apply (int** ARGOUTVIEWM_ARRAY2, int* DIM1, int* DIM2) {(int **arr_out, int *di
    m1_out, int *dim2_out)}
    
    %include "crop.h"
    



Don't forget that you will need the
`numpy.i <http://ezwidgets.googlecode.com/svn/trunk/numpy/numpy.i>`__
file in the same directory for this to compile.

Setup file (setup\_crop.py)
---------------------------

This is the
`setup\_crop.py <http://ezwidgets.googlecode.com/svn/trunk/numpy/setup_crop.py>`__
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
    
    # crop extension module
    _crop = Extension("_crop",
                       ["crop.i","crop.c"],
                       include_dirs = [numpy_include],
    
                       extra_compile_args = ["--verbose"]
                       )
    
    # NumyTypemapTests setup
    setup(  name        = "crop test",
            description = "A simple crop test to demonstrate the use of ARGOUTVIEWM_
    ARRAY2",
            author      = "Egor Zindy",
            version     = "1.0",
            ext_modules = [_crop]
            )
    



Testing the module
------------------

If everything goes according to plan, there should be a \`\_crop.pyd\`
file available in the \`build\\lib.XXX\` directory. The file needs to be
copied in the directory with the \`crop.py\` file (generated by swig).

A python test program is provided in the SVN repository
(`test\_crop.py <http://ezwidgets.googlecode.com/svn/trunk/numpy/test_crop.py>`__)
and reproduced below:



.. code-block:: python

    import crop
    import numpy
    
    a = numpy.zeros((5,10),numpy.int)
    a[numpy.arange(5),:] = numpy.arange(10)
    
    b = numpy.transpose([(10 ** numpy.arange(5))])
    a = (a*b)[:,1:] #this array is most likely NOT contiguous
    
    print a
    print "dim1=%d dim2=%d" % (a.shape[0],a.shape[1])
    
    d1_0 = 2
    d1_1 = 4
    d2_0 = 1
    d2_1 = 5
    
    c = crop.crop(a, d1_0,d1_1, d2_0,d2_1)
    d = a[d1_0:d1_1, d2_0:d2_1]
    
    print "returned array:"
    print c
    
    print "native slicing:"
    print d
    



This is what the output looks like:



.. code-block:: python

    $ python test_crop.py 
    [[    1     2     3     4     5     6     7     8     9]
     [   10    20    30    40    50    60    70    80    90]
     [  100   200   300   400   500   600   700   800   900]
     [ 1000  2000  3000  4000  5000  6000  7000  8000  9000]
     [10000 20000 30000 40000 50000 60000 70000 80000 90000]]
    dim1=5 dim2=9
    
    --- d1_0=2 d1_1=4 (rows)  -- d2_0=1 d2_1=5 (columns)
    200 300 400 500 
    2000 3000 4000 5000 
    ---
    
    returned array:
    [[ 200  300  400  500]
     [2000 3000 4000 5000]]
    native slicing:
    [[ 200  300  400  500]
     [2000 3000 4000 5000]]
    



numpy.i takes care of making the array contiguous if needed, so the only
thing left to take care of is the array orientation.

Conclusion and comments
=======================

That's all folks! Files are available on the `Google code
SVN <http://code.google.com/p/ezwidgets/source/browse/#svn/trunk/numpy>`__.
As usual, comments welcome!

Regards, Egor

--------------

CategoryCookbook

