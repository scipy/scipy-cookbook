# <markdowncell>

# <TableOfContents>
# 
# Introduction
# ============
# 
# This page gives examples how to read or write a !NumPy array to or from
# a file, be it ascii or binary. The various methods demonstrated all have
# copious and sometimes sophisticated options, call help to get details.
# 
# We will consider a trivial example where we create an array of zeros
# called \`data\`, write it to a file \`myfile.txt\` (myfile.dat for the
# binary case), and read it into \`read\_data\`.
# 
# This documentation could be improved by discussing more sophisticated
# cases (e.g. multiple arrays), and discussing the costs/benefits of the
# various approaches presented.
# 
# Text files
# ==========
# 
# SciPy
# -----
# 
# Writing files can be accomplished using \`savetxt\`. By far the easiest
# way to read text data is via genfromtxt, (or derivative convenience
# functions recfromtxt and recfromcsv).
# 
# <codecell>


>>> from numpy import *
>>> data = zeros((3,3))
>>>#Write data:
>>> savetxt("myfile.txt", data)
>>>#Read:
>>> data = genfromtxt("myfile.txt") }}}

== Matplotlib (pylab) ==

Matplotlib  provides an easy solution which seems to load data faster than read_array:

{{{#!python numbers=disable
>>> from numpy import *
>>> from pylab import load           # warning, the load() function of numpy will be shadowed
>>> from pylab import save
>>> data = zeros((3,3))
>>> save('myfile.txt', data)
>>> read_data = load("myfile.txt")

# <markdowncell>

# numPy
# -----
# 
# <codecell>


>>> savetxt('myfile.txt', data, fmt="%12.6G")    # save to file

# <markdowncell>

# 
# 
# <codecell>


>>> from numpy import *
>>> data = genfromtxt('table.dat', unpack=True)

# <markdowncell>

# csv files
# ---------
# 
# Note that csv stands for "comma separated value". This means that the
# separator (also called a delimiter), i.e. the character which is used to
# separate individual values in a file, is a comma. In the examples above,
# the default delimiter is a space, but all of the above methods have an
# option (see their respective help for details), which can be set to a
# comma in order to read or write a csv file instead.
# 
# A more sophisticated example
# ----------------------------
# 
# Or, assuming you have imported numpy as N, you may want to read
# arbitrary column types. You can also return a recarray, which let's you
# assign 'column headings' to your array.
# 
# <codecell>


def read_array(filename, dtype, separator=','):
    """ Read a file with an arbitrary number of columns.
        The type of data in each column is arbitrary
        It will be cast to the given dtype at runtime
    """
    cast = N.cast
    data = [[] for dummy in xrange(len(dtype))]
    for line in open(filename, 'r'):
        fields = line.strip().split(separator)
        for i, number in enumerate(fields):
            data[i].append(number)
    for i in xrange(len(dtype)):
        data[i] = cast[dtype[i]](data[i])
    return N.rec.array(data, dtype=dtype)

# <markdowncell>

# This can then be called with the corresponding dtype:
# 
# <codecell>


mydescr = N.dtype([('column1', 'int32'), ('column2Name', 'uint32'), ('col3', 'uint64'), ('c4', 'float32')])
myrecarray = read_array('file.csv', mydescr)

# <markdowncell>

# Binary Files
# ============
# 
# The advantage of binary files is the huge reduction in file size. The
# price paid is losing human readability, and in some formats, losing
# portability.
# 
# Let us consider the array in the previous example.
# 
# File format with metadata
# -------------------------
# 
# The simplest possibility is to use 's own binary file format. See , and
# .
# 
# <codecell>


>>> numpy.save('test.npy', data)
>>> data2 = numpy.load('test.npy')

# <markdowncell>

# You can save several arrays in a single file using . When loading an
# file you get an object of type . You can obtain a list of arrays and
# load individual arrays like this:
# 
# <codecell>


>>> numpy.savez('foo.npz', a=a,b=b)
>>> foo = numpy.load('foo.npz')
>>> foo.files
['a', 'b']
>>> a2 = foo['a']
>>> b2 = foo['b']

# <markdowncell>

# On older systems, the standard was to use python's pickle module to
# pickle the arrays.
# 
# Raw binary
# ----------
# 
# These file formats simply write out the internal representation of the
# arrays. This is platform-dependent and includes no information about
# array shape or datatype, but is quick and easy.
# 
# SciPy provides fwrite() from scipy.io.numpyio. You have to set the size
# of your data, and optionally, its type (integer, short, float, etc; see
# [1](http://docs.neuroinf.de/api/scipy/scipy.io.numpyio-module.html)).
# 
# For reading binary files, scipy.io.numpyio provides fread(). You have to
# know the datatype of your array, its size and its shape.
# 
# <codecell>


>>> from scipy.io.numpyio import fwrite, fread
>>> data = zeros((3,3))
>>>#write:  fd = open('myfile.dat', 'wb')
>>> fwrite(fd, data.size, data)
>>> fd.close()
>>>#read:
>>> fd = open('myfile.dat', 'rb')
>>> datatype = 'i'
>>> size = 9
>>> shape = (3,3)
>>> read_data = fread(fd, size, datatype)
>>> read_data = data.reshape(shape)

# <markdowncell>

# Or, you can simply use and . Following the previous example:
# 
# <codecell>


>>> data.tofile('myfile.dat')
>>> fd = open('myfile.dat', 'rb')
>>> read_data = numpy.fromfile(file=fd, dtype=numpy.uint8).reshape(shape)

# <markdowncell>

# 
# 
# <codecell>


numpy data type. The option {{{fromfile(..., count=<number>)}}} specifies the number of data entries of that type you want to read in (the default -1 means read in the whole file, which is what you usually want). However, the method is not recommended for data storage and transfer between different platforms, since no byteorder and datatype information is stored (see also the docstrings).
If you want that, use {{{numpy}}}'s own binary file format. See {{{numpy.save}}}, {{{numpy.savez}}} and {{{numpy.load}}}.
{{{#! python numbers=disable
>>> numpy.save('test.npy', data)
>>> data2 = numpy.load('test.npy')

# <markdowncell>

# Another, but deprecated, way to fully control endianness (byteorder),
# storage order (row-major, column-major) for rank \> 1 arrays and
# datatypes that are written and read back is . Writing:
# 
# <codecell>


>>> from scipy.io import npfile
>>> shape = (3,3)
>>> data = numpy.random.random(shape)
>>> npf = npfile('test.dat', order='F', endian='<', permission='wb') 
>>> npf.write_array(data)
>>> npf.close()

# <markdowncell>

# And reading back:
# 
# <codecell>


>>> npf = npfile('test.dat', order='F', endian='<', permission='rb') 
>>> data2 = npf.read_array(float, shape=shape)
>>> npf.close()

# <markdowncell>

# Write a Fortran or C array to a binary file with metadata
# ---------------------------------------------------------
# 
# [libnpy](http://www.maths.unsw.edu.au/~mclean/libnpy-0.5.tgz) is a small
# library that provides simple routines for saving a C or Fortran array to
# a data file using NumPy's own binary format. For a description of this
# format, do
# 
# <codecell>


>>> from numpy.lib import format
>>> help(format)

# <markdowncell>

# Here is a minimal C example \`cex.c\`:
# 
# <codecell>


#include"npy.h"
int main(){
    double a[2][4] = { { 1, 2, 3, 4 },
                       { 5, 6, 7, 8 } };
    int shape[2] = { 2, 4 }, fortran_order = 0;

    npy_save_double("ca.npy", fortran_order, 2, shape, &a[0][0]);
    return 0;
}

# <markdowncell>

# The program creates a file \`ca.npy\` that you can load into python in
# the usual way.
# 
# <codecell>


>>> ca = np.load('ca.npy')
>>> print ca
[[ 1.  2.  3.  4.]
 [ 5.  6.  7.  8.]]

# <markdowncell>

# The corresponding Fortran program, \`fex.f95\`, looks like
# 
# <codecell>


program fex
    use fnpy
    use iso_c_binding
    implicit none

    integer  :: i
    real(C_DOUBLE) :: a(2,4) = reshape([(i, i=1,8)], [2,4])

    call save_double("fa.npy", shape(a), a)
end program fex

# <markdowncell>

# but the entries of the NumPy array now follow the Fortran (column-major)
# ordering.
# 
# <codecell>


>>> fa = np.load('fa.npy')
>>> print fa
[[ 1.  3.  5.  7.]
 [ 2.  4.  6.  8.]]

# <markdowncell>

# The \`README\` file in the source distribution explains how to compile
# the library using \`make\`.
# 
# If you put \`npy.h\` and \`libnpy.a\` in the same directory as
# \`cex.c\`, then you can build the executable \`cex\` with the command
# 
# <codecell>


gcc -o cex cex.c libnpy.a

# <markdowncell>

# Similarly, with \`npy.mod\` and \`libnpy.a\` in the same directory as
# \`fex.f95\`, build \`fex\` with the command
# 
# <codecell>


gfortran -o fex fex.f95 libnpy.a

# <markdowncell>

# * * * * *
# 
# CategoryCookbook CategoryCookbook
# 