Table of Contents
=================

TableOfContents

Skeleton
========

\`extmodule.h\`:



.. code-block:: python

    #ifndef EXTMODULE_H
    #define EXTMODULE_H
    
    #ifdef __cplusplus
    extern "C" {
    #endif
    
    /* Python.h must be included before everything else */
    #include "Python.h"
    
    /* include system headers here */
    
    #if !defined(EXTMODULE_IMPORT_ARRAY)
    #define NO_IMPORT_ARRAY
    #endif
    #include "numpy/arrayobject.h"
    
    #ifdef __cplusplus
    }
    #endif
    
    #endif
    



Note that \`PY\_ARRAY\_UNIQUE\_SYMBOL\` must be set to a unique value
for each extension module. But, you actually don't need to set it at all
unless you are going to compile an extension module that uses two
different source files

\`extmodule.c\`:



.. code-block:: python

    #define EXTMODULE_IMPORT_ARRAY
    #include "extmodule.h"
    #undef EXTMODULE_IMPORT_ARRAY
    
    static PyObject* FooError;
    
    static PyObject* Ext_Foo(PyObject* obj, PyObject* args) {
       Py_INCREF(Py_None);
       return Py_None;
    }
    
    static PyMethodDef methods[] = {
       {"foo", (PyCFunction) Ext_Foo, METH_VARARGS, ""},
       {NULL, NULL, 0, NULL}
    };
    
    PyMODINIT_FUNC init_extmodule() {
       PyObject* m;
       m = Py_InitModule("_extmodule", methods);
       import_array();
       SVMError = PyErr_NewException("_extmodule.FooError", NULL, NULL);
       Py_INCREF(FooError);
       PyModule_AddObject(m, "FooError", FooError);
    }
    



If your extension module is contained in a single source file then you
can get rid of extmodule.h entirely and replace the first part of
extmodule.c with



.. code-block:: python

    #inlude "Python.h"
    #include "numpy/arrayobject.h"
    
    /* remainder of extmodule.c after here */
    



Debugging C Extensions on Windows
=================================

Debugging C extensions on Windows can be tricky. If you compile your
extension code in debug mode, you have to link against the debug version
of the Python library, e.g. \`Python24\_d.lib\`. When building with
Visual Studio, this is taken care of by a pragma in \`Python24.h\`. If
you force the compiler to link debug code against the release library,
you will probably get the following errors (especially when compiling
SWIG wrapped codes):



.. code-block:: python

    extmodule.obj : error LNK2019: unresolved external symbol __imp___Py_Dealloc ref
    erenced in function _PySwigObject_format
    extmodule.obj : error LNK2019: unresolved external symbol __imp___Py_NegativeRef
    count referenced in function _PySwigObject_format
    extmodule.obj : error LNK2001: unresolved external symbol __imp___Py_RefTotal
    extmodule.obj : error LNK2019: unresolved external symbol __imp___PyObject_Debug
    Free referenced in function _PySwigObject_dealloc
    extmodule.obj : error LNK2019: unresolved external symbol __imp___PyObject_Debug
    Malloc referenced in function _PySwigObject_New
    extmodule.obj : error LNK2019: unresolved external symbol __imp__Py_InitModule4T
    raceRefs referenced in function _init_extmodule
    



However, now you also need a debug build of the Python interpreter if
you want to import this debuggable extension module. Now you also need
debug builds of every other extension module you use. Clearly, this can
take some time to get sorted out.

An alternative is to build your library code as a debug DLL. This way,
you can at least that your extension module is passing the right data to
the library code you are wrapping.

As an aside, it seems that the MinGW GCC compiler doesn't produce debug
symbols that are understood by the Visual Studio debugger.

Valgrind
========

To develop a stable extension module, it is essential that you check the
memory allocation and memory accesses done by your C code. On Linux, you
can use `Valgrind <http://www.valgrind.org/>`__. On Windows, you could
try a commercial tool such as `Rational
PurifyPlus <http://www-306.ibm.com/software/awdtools/purifyplus/>`__.

Before using Valgrind, make sure your extension module is compiled with
the \`-g\` switch to GCC so that you can get useful stack traces when
errors are detected.

Then put the following in a shell script, say \`valgrind\_py.sh\`:



.. code-block:: python

    #!/bin/sh
    valgrind \
       --tool=memcheck \
       --leak-check=yes \
       --error-limit=no \
       --suppressions=valgrind-python.supp \
       --num-callers=10 \
       -v \
       python $1
    



\`valgrind-python.supp\` suppresses some warnings caused by the Python
code. You can find the suppression file for Python 2.4 `in the Python
SVN
repository <http://svn.python.org/projects/python/branches/release24-maint/Misc/valgrind-python.supp>`__.
See also
`README.valgrind <http://svn.python.org/projects/python/branches/release24-maint/Misc/README.valgrind>`__
in the same location. Some of the suppressions are commented out by
default. Enable them by removing the # comment markers.

Execute \`chmod +x valgrind\_py.sh\` and run it as \`./valgrind\_py.sh
test\_extmodule.py\`.

Documentation
=============

| ``* ``\ ```Extending`` ``and`` ``Embedding`` ``the`` ``Python``
``Interpreter`` <http://docs.python.org/ext/ext.html>`__\ `` (read this first)``
| ``* ``\ ```Python/C`` ``API`` ``Reference``
``Manual`` <http://docs.python.org/api/api.html>`__\ `` (then browse through this)``
| ``* Chapter 13 of ``\ ```Guide`` ``to``
``NumPy`` <http://www.tramy.us/>`__\ `` describes the complete API``
| ``* Chapter 14 deals with extending !NumPy (make sure you have the edition dated March 15, 2006 or later)``

Examples
========

| ``* ``\ ```ndimage`` ``in`` ``the`` ``SciPy`` ``SVN``
``repository`` <http://projects.scipy.org/scipy/scipy/browser/trunk/Lib/ndimage>`__
| ``* [:Cookbook/C_Extensions/NumPy_arrays:NumPy arrays] (collection of small examples)``

Mailing List Threads
====================

| ``* ``\ ```Freeing`` ``memory`` ``allocated`` ``in``
``C`` <http://thread.gmane.org/gmane.comp.python.numeric.general/5206/focus=5206>`__
| ``* ``\ ```port`` ``C/C++`` ``matlab`` ``mexing`` ``code`` ``to``
``numpy`` <http://thread.gmane.org/gmane.comp.python.scientific.user/7392/focus=7400>`__
| ``* ``\ ```newbie`` ``for`` ``writing`` ``numpy/scipy``
``extensions`` <http://thread.gmane.org/gmane.comp.python.numeric.general/5186/focus=5186>`__
| ``* ``\ ```Array`` ``data`` ``and`` ``struct``
``alignment`` <http://thread.gmane.org/gmane.comp.python.numeric.general/5234/focus=5234>`__

--------------

CategoryCookbook

