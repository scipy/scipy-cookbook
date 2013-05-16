Please note that with current versions of NumPy (SVN), the directory
contains a complete working example with simple SWIG typemaps, including
also a proper file so you can install it with the standard Python
mechanisms. This should help you get up and running quickly.

To get the feel how to write a truly minimalist interface, below is a
relevant part of the simple SWIG interface file .. image:: SWIG_and_NumPy_attachments/umfpack.i
(this is for SWIG < version 1.3.29) used to wrap the UMFPACK sparse
linear solver libraries. The full interface can be found in the
directory in the SciPy SVN repository. If you're using SWIG > version
1.3.29, refer to the file in SciPy SVN repository, which is slightly
different.



.. code-block:: python

    
    /*!
      Gets PyArrayObject from a PyObject.
    */
    PyArrayObject *helper_getCArrayObject( PyObject *input, int type,
    				       int minDim, int maxDim ) {
      PyArrayObject *obj;
    
      if (PyArray_Check( input )) {
        obj = (PyArrayObject *) input;
        if (!PyArray_ISCARRAY( obj )) {
          PyErr_SetString( PyExc_TypeError, "not a C array" );
          return NULL;
        }
        obj = (PyArrayObject *)
          PyArray_ContiguousFromAny( input, type, minDim, maxDim );
        if (!obj) return NULL;
      } else {
        PyErr_SetString( PyExc_TypeError, "not an array" );
        return NULL;
      }
      return obj;
    }
    %}
    
    /*!
      Use for arrays as input arguments. Could be also used for changing an array
      in place.
    
      @a rtype ... return this C data type
      @a ctype ... C data type of the C function
      @a atype ... PyArray_* suffix
    */
    #define ARRAY_IN( rtype, ctype, atype ) \
    %typemap( python, in ) (ctype *array) { \
      PyArrayObject *obj; \
      obj = helper_getCArrayObject( $input, PyArray_##atype, 1, 1 ); \
      if (!obj) return NULL; \
      $1 = (rtype *) obj->data; \
      Py_DECREF( obj ); \
    };
    
    ARRAY_IN( int, const int, INT )
    %apply const int *array {
        const int Ap [ ],
        const int Ai [ ]
    };
    
    ARRAY_IN( long, const long, LONG )
    %apply const long *array {
        const long Ap [ ],
        const long Ai [ ]
    };
    
    ARRAY_IN( double, const double, DOUBLE )
    %apply const double *array {
        const double Ax [ ],
        const double Az [ ],
        const double B [ ]
    };
    
    ARRAY_IN( double, double, DOUBLE )
    %apply double *array {
        double X [ ]
    };
    



The function being wrapped then could be like:



.. code-block:: python

    int umfpack_di_solve( int sys, const int Ap [ ], const int Ai [ ],
                          const double Ax [ ], double X [ ], const double B [ ],
                          ... );
    



--------------

CategoryCookbook

