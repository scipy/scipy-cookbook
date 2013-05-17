# <markdowncell>

# Wrapping C codes using f2py
# ===========================
# 
# While initially f2py was developed for wrapping Fortran codes for
# Python, it can be easily used for wrapping C codes as well. Signature
# files describing the interface to wrapped functions must be created
# manually and the functions and their arguments must have the attribute .
# See [f2py
# UsersGuide](http://cens.ioc.ee/projects/f2py2e/usersguide/index.html)
# for more information about the syntax of signature files.
# 
# Here follows as simple C code
# 
# <codecell>


/* File foo.c */
void foo(int n, double *x, double *y) {
  int i;
  for (i=0;i<n;i++) {
    y[i] = x[i] + i;
  }
}

# <markdowncell>

# and the corresponding signature file
# 
# <codecell>


! File m.pyf
python module m
interface
  subroutine foo(n,x,y)
    intent(c) foo                 ! foo is a C function
    intent(c)                     ! all foo arguments are 
                                  ! considered as C based
    integer intent(hide), depend(x) :: n=len(x)  ! n is the length
                                                 ! of input array x
    double precision intent(in) :: x(n)          ! x is input array 
                                                 ! (or  arbitrary sequence)
    double precision intent(out) :: y(n)         ! y is output array, 
                                                 ! see code in foo.c
  end subroutine foo
end interface
end python module m

# <markdowncell>

# To build the wrapper, one can either create a setup.py script
# 
# <codecell>


# File setup.py
def configuration(parent_package='',top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('',parent_package,top_path)

    config.add_extension('m',
                         sources = ['m.pyf','foo.c'])
    return config
if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())

# <markdowncell>

# and execute:
# 
# <codecell>


python setup.py build_src build_ext --inplace

# <markdowncell>

# Or one can call f2py directly in command line to build the wrapper as
# follows:
# 
# <codecell>


f2py m.pyf foo.c -c

# <markdowncell>

# In both cases an extension module will be created to current directory
# that can be imported to python:
# 
# <codecell>


>>> import m
>>> print m.__doc__
This module 'm' is auto-generated with f2py (version:2_2130).
Functions:
  y = foo(x)
.
>>> print m.foo.__doc__
foo - Function signature:
  y = foo(x)
Required arguments:
  x : input rank-1 array('d') with bounds (n)
Return objects:
  y : rank-1 array('d') with bounds (n)

>>> print m.foo([1,2,3,4,5])    
[ 1.  3.  5.  7.  9.]
>>>

# <markdowncell>

# * * * * *
# 
# CategoryCookbook
# 