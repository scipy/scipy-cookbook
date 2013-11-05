#!/usr/bin/env python
import sys
import numpy as npy
import pylab as P
from scipy.weave import inline, converters, blitz
from scipy.testing import measure
# Blitz conversion is terrific, but sometimes you don't have fixed array sizes
# in your problem. Fortunately numpy iterators still make writing inline
# weave code very, very simple. 

def multi_iter_example():
    # This is a very simple example of multi dimensional iterators, and
    # their power to "broadcast" arrays of compatible shapes. It shows that
    # the very same code that is entirely ignorant of dimensionality can
    # achieve completely different computations based on the rules of
    # broadcasting.

    # it is important to know that the weave array conversion of "a"
    # gives you access in C++ to:
    # py_a -- PyObject *
    # a_array -- PyArrayObject *
    # a -- py_array->data
    
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

def data_casting_test():
    # In my MR application, raw data is stored as a file with one or more
    # (block-hdr, block-data) pairs. Block data is one or more
    # rows of Npt complex samples in big-endian integer pairs (real, imag).
    #
    # At the block level, I encounter three different raw data layouts--
    # 1) one plane, or slice: Y rows by 2*Npt samples
    # 2) one volume: Z slices * Y rows by 2*Npt samples
    # 3) one row sliced across the z-axis: Z slices by 2*Npt samples
    #
    # The task is to tease out one volume at a time from any given layout,
    # and cast the integer precision data into a complex64 array.
    # Given that contiguity is not guaranteed, and the number of dimensions
    # can vary, Numpy iterators are useful to provide a single code that can
    # carry out the conversion.
    #
    # Other solutions include:
    # 1) working entirely with the string data from file.read() with string
    #    manipulations (simulated below).
    # 2) letting numpy handle automatic byteorder/dtype conversion
    
    nsl, nline, npt = (20,64,64)
    hdr_dt = npy.dtype('>V28')
    # example 1: a block is one slice of complex samples in short integer pairs
    blk_dt1 = npy.dtype(('>i2', nline*npt*2))
    dat_dt = npy.dtype({'names': ['hdr', 'data'], 'formats': [hdr_dt, blk_dt1]})
    # create an empty volume-- nsl contiguous blocks
    vol = npy.empty((nsl,), dat_dt)
    t = time_casting(vol[:]['data'])
    P.plot(100*t/t.max(), 'b--', label='vol=20 contiguous blocks')
    P.plot(100*t/t.max(), 'bo')
    # example 2: a block is one entire volume
    blk_dt2 = npy.dtype(('>i2', nsl*nline*npt*2))
    dat_dt = npy.dtype({'names': ['hdr', 'data'], 'formats': [hdr_dt, blk_dt2]})
    # create an empty volume-- 1 block
    vol = npy.empty((1,), dat_dt)
    t = time_casting(vol[0]['data'])
    P.plot(100*t/t.max(), 'g--', label='vol=1 contiguous block')
    P.plot(100*t/t.max(), 'go')    
    # example 3: a block slices across the z dimension, long integer precision
    # ALSO--a given volume is sliced discontiguously
    blk_dt3 = npy.dtype(('>i4', nsl*npt*2))
    dat_dt = npy.dtype({'names': ['hdr', 'data'], 'formats': [hdr_dt, blk_dt3]})
    # a real data set has volumes interleaved, so create two volumes here
    vols = npy.empty((2*nline,), dat_dt)
    # and work on casting the first volume
    t = time_casting(vols[0::2]['data'])
    P.plot(100*t/t.max(), 'r--', label='vol=64 discontiguous blocks')
    P.plot(100*t/t.max(), 'ro')    
    P.xticks([0,1,2], ('strings', 'numpy auto', 'inline'))
    P.gca().set_xlim((-0.25, 2.25))
    P.gca().set_ylim((0, 110))
    P.gca().set_ylabel(r"% of slowest time")
    P.legend(loc=8)
    P.title('Casting raw file data to an MR volume')
    P.show()
    

def time_casting(int_data):
    nblk = 1 if len(int_data.shape) < 2 else int_data.shape[0]
    bias = (npy.random.rand(nblk) + \
            1j*npy.random.rand(nblk)).astype(npy.complex64)
    dstr = int_data.tostring()
    dt = npy.int16 if int_data.dtype.itemsize == 2 else npy.int32
    fshape = list(int_data.shape)
    fshape[-1] = fshape[-1]/2
    float_data = npy.empty(fshape, npy.complex64)
    # method 1: string conversion
    float_data.shape = (npy.product(fshape),)
    tstr = measure("float_data[:] = complex_fromstring(dstr, dt)", times=25)
    float_data.shape = fshape
    print "to-/from- string: ", tstr, "shape=",float_data.shape

    # method 2: numpy dtype magic
    sl = [None, slice(None)] if len(fshape)<2 else [slice(None)]*len(fshape)
    # need to loop since int_data need not be contiguous
    tnpy = measure("""
for fline, iline, b in zip(float_data[sl], int_data[sl], bias):
    cast_to_complex_npy(fline, iline, bias=b)""", times=25)
    print"numpy automagic: ", tnpy

    # method 3: plain inline brute force!
    twv = measure("cast_to_complex(float_data, int_data, bias=bias)",
                  times=25)
    print"inline casting: ", twv
    return npy.array([tstr, tnpy, twv], npy.float64)
    
def complex_fromstring(data, numtype):
    if sys.byteorder == "little":
        return npy.fromstring(
            npy.fromstring(data,numtype).byteswap().astype(npy.float32).tostring(),
            npy.complex64)
    else:
        return npy.fromstring(
	    npy.fromstring(data,numtype).astype(npy.float32).tostring(),
            npy.complex64)

def cast_to_complex(cplx_float, cplx_integer, bias=None):
    if cplx_integer.dtype.itemsize == 4:
        replacements = tuple(["l", "long", "SWAPLONG", "l"]*2)
    else:
        replacements = tuple(["s", "short", "SWAPSHORT", "s"]*2)
    if sys.byteorder == "big":
        replacements[-2] = replacements[-6] = "NOP"

    cast_code = """
    #define SWAPSHORT(x) ((short) ((x >> 8) | (x << 8)) )
    #define SWAPLONG(x) ((long) ((x >> 24) | (x << 24) | ((x & 0x00ff0000) >> 8) | ((x & 0x0000ff00) << 8)) )
    #define NOP(x) x
    
    unsigned short *s;
    unsigned long *l;
    float repart, impart;
    PyObject *itr;
    itr = PyArray_IterNew(py_cplx_integer);
    while(PyArray_ITER_NOTDONE(itr)) {

      // get real part
      %s = (unsigned %s *) PyArray_ITER_DATA(itr);
      repart = %s(*%s);
      PyArray_ITER_NEXT(itr);
      // get imag part
      %s = (unsigned %s *) PyArray_ITER_DATA(itr);
      impart = %s(*%s);
      PyArray_ITER_NEXT(itr);
      *(cplx_float++) = std::complex<float>(repart, impart);

    }
    """ % replacements
    
    inline(cast_code, ['cplx_float', 'cplx_integer'])
    if bias is not None:
        if len(cplx_float.shape) > 1:
            bsl = [slice(None)]*(len(cplx_float.shape)-1) + [None]
        else:
            bsl = slice(None)
        npy.subtract(cplx_float, bias[bsl], cplx_float)

def cast_to_complex_npy(cplx_float, cplx_integer, bias=None):
    cplx_float.real[:] = cplx_integer[0::2]
    cplx_float.imag[:] = cplx_integer[1::2]
    if bias is not None:
        npy.subtract(cplx_float, bias, cplx_float)

if __name__=="__main__":
    data_casting_test()
    multi_iter_example()
