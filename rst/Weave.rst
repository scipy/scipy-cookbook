Inline Weave With Basic Array Conversion (no Blitz)
---------------------------------------------------

Python and Numpy are designed to express general statements that work
transparently on many sizes of incoming data. Using inline Weave with
Blitz conversion can dramatically speed up many numerical operations
(eg, addition of a series of arrays) because in some ways it bypasses
generality. How can you speed up your algorithms with inline C code
while maintaining generality? One tool provided by Numpy is the
**iterator**. Because an iterator keeps track of memory indexing for
you, its operation is very analogous to the concept of iteration in
Python itself. You can write loop in C that simply says "take the next
element in a serial object--the !PyArrayObject--and operate on it, until
there are no more elements."

--------------

This is a very simple example of multi dimensional iterators, and their
power to "broadcast" arrays of compatible shapes. It shows that the very
same code that is entirely ignorant of dimensionality can achieve
completely different computations based on the rules of broadcasting. I
have assumed in this case that ***a*** has at least as many dimensions
as ***b***. It is important to know that the weave array conversion of
***a*** gives you access in C++ to: \*py\_a -- !PyObject \* \*a\_array
-- !PyArrayObject \* \*a -- (c-type \*) py\_array->data



.. code-block:: python

    import numpy as npy
    from scipy.weave import inline
    
    def multi_iter_example():
        a = npy.ones((4,4), npy.float64)
        # for the sake of driving home the "dynamic code" approach...
        dtype2ctype = {
            npy.dtype(npy.float64): 'double',
            npy.dtype(npy.float32): 'float',
            npy.dtype(npy.int32): 'int',
            npy.dtype(npy.int16): 'short',
        }
        dt = dtype2ctype.get(a.dtype)
        
        # this code does a = a*b inplace, broadcasting b to fit the shape of a
        code = \
    """
    %s *p1, *p2;
    PyObject *itr;
    itr = PyArray_MultiIterNew(2, a_array, b_array);
    while(PyArray_MultiIter_NOTDONE(itr)) {
      p1 = (%s *) PyArray_MultiIter_DATA(itr, 0);
      p2 = (%s *) PyArray_MultiIter_DATA(itr, 1);
      *p1 = (*p1) * (*p2);
      PyArray_MultiIter_NEXT(itr);
    }
    """ % (dt, dt, dt)
    
        b = npy.arange(4, dtype=a.dtype)
        print '\n         A                  B     '
        print a, b
        # this reshaping is redundant, it would be the default broadcast
        b.shape = (1,4)
        inline(code, ['a', 'b'])
        print "\ninline version of a*b[None,:],"
        print a
        a = npy.ones((4,4), npy.float64)
        b = npy.arange(4, dtype=a.dtype)
        b.shape = (4,1)
        inline(code, ['a', 'b'])
        print "\ninline version of a*b[:,None],"
        print a
    



There are two other iterator applications in
`iterators_example.py <.. image:: Weave_attachments/iterators_example.py>`__ and
`iterators.py <.. image:: Weave_attachments/iterators.py>`__.

Deeper into the "inline" method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The
`docstring <http://www.scipy.org/scipy/scipy/browser/trunk/scipy/weave/inline_tools.py>`__
for **inline** is enormous, and indicates that all kinds of compiling
options are supported when integrating your inline code. I've taken
advantage of this to make some specialized FFTW calls a lot more simple,
and in only a few additional lines add support for inline FFTs. In this
example, I read in a file of pure C code and use it as
***support\_code*** in my inline statement. I also use a tool from
Numpy's distutils to locate my FFTW libraries and headers.



.. code-block:: python

    import numpy as N
    from scipy.weave import inline
    from os.path import join, split
    from numpy.distutils.system_info import get_info        
    
    fft1_code = \
    """
    char *i, *o;
    i = (char *) a;
    o = inplace ? i : (char *) b;
    if(isfloat) {
      cfft1d(reinterpret_cast<fftwf_complex*>(i),
             reinterpret_cast<fftwf_complex*>(o),
             xdim, len_array, direction, shift);
    } else {
      zfft1d(reinterpret_cast<fftw_complex*>(i),
             reinterpret_cast<fftw_complex*>(o),
             xdim, len_array, direction, shift);
    }
    """
    extra_code = open(join(split(__file__)[0],'src/cmplx_fft.c')).read()
    fftw_info = get_info('fftw3')
    
    def fft1(a, shift=True, inplace=False):
        if inplace:
            _fft1_work(a, -1, shift, inplace)
        else:
            return _fft1_work(a, -1, shift, inplace)
    
    def ifft1(a, shift=True, inplace=False):
        if inplace:
            _fft1_work(a, +1, shift, inplace)
        else:
            return _fft1_work(a, +1, shift, inplace)
    
    def _fft1_work(a, direction, shift, inplace):
        # to get correct C-code, b always must be an array (but if it's
        # not being used, it can be trivially small)
        b = N.empty_like(a) if not inplace else N.array([1j], a.dtype)
        inplace = 1 if inplace else 0
        shift = 1 if shift else 0    
        isfloat = 1 if a.dtype.itemsize==8 else 0
        len_array = N.product(a.shape)
        xdim = a.shape[-1]
        inline(fft1_code, ['a', 'b', 'isfloat', 'inplace',
                           'len_array', 'xdim', 'direction', 'shift'],
               support_code=extra_code,
               headers=['<fftw3.h>'],
               libraries=['fftw3', 'fftw3f'],
               include_dirs=fftw_info['include_dirs'],
               library_dirs=fftw_info['library_dirs'],
               compiler='gcc')
        if not inplace:
            return b
    



This code is available in `fftmod.tar.gz <.. image:: Weave_attachments/fftmod.tar.gz>`__.

--------------

CategoryCookbook

