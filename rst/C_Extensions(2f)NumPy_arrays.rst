C Extensions for Using NumPy Arrays

I've written several C extensions that handle NumPy arrays. They are
simple, but they seem to work well. They will show you how to pass
Python variables and NumPy arrays to your C code. Once you learn how to
do it, it's pretty straight-forward. I suspect they will suffice for
most numerical code. I've written it up as a draft and have made the
code and document file available. I found for my numerical needs I
really only need to pass a limited set of things (integers, floats,
strings, and NumPy arrays). If that's your category, this code might
help you.

I have tested the routines and so far,so good, but I cannot guarantee
anything. I am a bit new to this. If you find any errors put up a
message on the SciPy mailing list.

A link to the tar ball that holds the code and docs is given below.

I have recently updated some information and included more examples. The
document presented below is the original documentation which is still
useful. The link below holds the latest documentation and source code.

-- Lou Pecora

``. ``\ ```1`` <.. image:: C_Extensions(2f)NumPy_arrays_attachments/Cext_v2.tar.gz>`__

--------------

*What follows is the content of Lou\`s word-document originally pasted
here as version 1. I (DavidLinke) have converted this to wiki-markup:*

C Extensions to NumPy and Python
================================

By Lou Pecora - 2006-12-07 (Draft version 0.1)

TableOfContents

Overview
--------

Introduction– a little background
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In my use of Python I came across a typical problem: I needed to speed
up particular parts of my code. I am not a Python guru or any kind of
coding/computer guru. I use Python for numerical calculations and I make
heavy use of Numeric/NumPy. Almost every Python book or tutorial tells
you build C extensions to Python when you need a routine to run fast. C
extensions are C code that can be compiled and linked to a shared
library that can be imported like any Python module and you can call
specified C routines like they were Python functions.

Sounds nice, but I had reservations. It looked non-trivial (it is, to an
extent). So I searched for other solutions. I found them. They are such
approaches as `SWIG <http://www.swig.org/>`__,
`Pyrex <http://www.cosc.canterbury.ac.nz/greg.ewing/python/Pyrex/>`__,
`ctypes <http://python.net/crew/theller/ctypes/>`__,
`Psycho <http://psyco.sourceforge.net/>`__, and
`Weave <http://www.scipy.org/Weave>`__. I often got the simple examples
given to work (not all, however) when I tried these. But I hit a barrier
when I tried to apply them to NumPy. Then one gets into typemaps or
other hybrid constructs. I am not knocking these approaches, but I could
never figure them out and get going on my own code despite lots of
online tutorials and helpful suggestions from various Python support
groups and emailing lists.

So I decided to see if I could just write my own C extensions. I got
help in the form of some simple C extension examples for using Numeric
written about 2000 from Tom Loredo of Cornell university. These sat on
my hard drive until 5 years later out of desperation I pulled them out
and using his examples, I was able to quickly put together several C
extensions that (at least for me) handle all of the cases (so far) where
I want a speedup. These cases mostly involve passing Python integers,
floats (=C doubles), strings, and NumPy 1D and 2D float and integer
arrays. I rarely need to pass anything else to a C routine to do a
calculation. If you are in the same situation as me, then this package I
put together might help you. It turns out to be fairly easy once you get
going.

Please note, Tom Loredo is not responsible for any errors in my code or
instructions although I am deeply indebted to him. Likewise, this code
is for research only. It was tested by only my development and usage. It
is not guaranteed, and comes with no warranty. Do not use this code
where there are any threats of loss of life, limb, property, or money or
anything you or others hold dear.

I developed these C extensions and their Python wrappers on a Macintosh
G4 laptop using system OS X 10.4 (essential BSD Unix), Python 2.4, NumPy
0.9x, and the gnu compiler and linker gcc. I think most of what I tell
you here will be easily translated to Linux and other Unix systems
beyond the Mac. I am not sure about Windows. I hope that my low-level
approach will make it easy for Windows users, too.

The code (both C and Python) for the extensions may look like a lot, but
it is **very** repetitious. Once you get the main scheme for one
extension function you will see that repeated over and over again in all
the others with minor variations to handle different arguments or return
different objects to the calling routine. Don't be put off by the code.
The good news is that for many numerical uses extensions will follow the
same format so you can quickly reuse what you already have written for
new projects. Focus on one extension function and follow it in detail
(in fact, I will do this below). Once you understand it, the other
routines will be almost obvious. The same is true of the several utility
functions that come with the package. They help you create, test, and
manipulate the data and they also have a lot of repetition. The utility
functions are also very short and simple so nothing to fear there.

General Scheme for NumPy Extensions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This will be covered in detail below, but first I wanted to give you a
sense of how each extension is organized.

Three things that must be done before your C extension functions in the
C source file.

``1. You must include Python and NumPy headers.``

``2. Each extension must be named in a defining structure at the beginning of the file. This is a name used to access the extension from a Python function.``

``3. Next an initialization set of calls is made to set up the Python and NumPy calls and interface. It will be the same for all extensions involving NumPy and Python unless you add extensions to access other Python packages or classes beyond NumPy arrays. I will not cover any of that here (because I don't know it). So the init calls can be copied to each extension file.``

Each C extension will have the following form.

``* The arguments will always be the same: (`PyObject *self`, `PyObject *args`) - Don't worry if you don't know what exactly these are. They are pointers to general Python objects and they are automatically provided by the header files you will use from NumPy and Python itself. You need know no more than that.``

``* The args get processed by a function call that parses them and assigns them to C defined objects.``

``* Next the results of that parse might be checked by a utility routine that reaches into the structure representing the object and makes sure the data is the right kind (float or int, 1D or 2D array, etc.). Although I included some of these C-level checks, you will see that I think they are better done in Python functions that are used to wrap the C extensions. They are also a lot easier to do in Python. I have plenty of data checks in my calling Python wrappers. Usually this does not lead to much overhead since you are not calling these extensions billions of times in some loop, but using them as a portal to a C or C++ routine to do a long, complex, repetitive, specialized calculation.``

``* After some possible data checks, C data types are initialized to point to the data part of the NumPy arrays with the help of utility functions.``

``* Next dimension information is extracted so you know the number of columns, rows, vector dimensions, etc.``

``* Now you can use the C arrays to manipulate the data in the NumPy arrays. The C arrays and C data from the above parse point to the original Python/NumPy data so changes you make affect the array values when you go back to Python after the extension returns. You can pass the arrays to other C functions that do calculations, etc. Just remember you are operating on the original NumPy matrices and vectors.``

``* After your calculation you have to free any memory allocated in the construction of your C data for the NumPy arrays. This is done again by Utility functions. This step is only necessary if you allocated memory to handle the arrays (e.g. in the matrix routines), but is not necessary if you have not allocated memory (e.g. in the vector routines).``

``* Finally, you return to the Python calling function, by returning a Python value or NumPy array. I have C extensions which show examples of both.``

Python Wrapper Functions
~~~~~~~~~~~~~~~~~~~~~~~~

It is best to call the C extensions by calling a Python function that
then calls the extension. This is called a Python wrapper function. It
puts a more pythonic look to your code (e.g. you can use keywords
easily) and, as I pointed out above, allows you to easily check that the
function arguments and data are correct before you had them over to the
C extension and other C functions for that big calculation. It may seem
like an unnecessary extra step, but it's worth it.

The Code
--------

In this section I refer to the code in the source files C\_arraytest.h,
C\_arraytest.c, C\_arraytest.py, and C\_arraytest.mak. You should keep
those files handy (probably printed out) so you can follow the
explanations of the code below.

The C Code – one detailed example with utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, I will use the example of code from C\_arraytest.h,
C\_arraytest.c for the routine called matsq. This function takes a
(NumPy) matrix *A*, integer *i*, and (Python) float *y* as input and
outputs a return (NumPy) matrix *B* each of whose components is equal to
the square of the input matrix component times the integer times the
float. Mathematically:

`` ``\ *``B,,ij,,``*\ `` = ``\ *``i``
``y``*\ `` (``\ *``A,,ij,,``*\ ``)^2^``

The Python code to call the matsq routine is \`A=matsq(B,i,y)\`. Here is
the relevant code in one place:

The Header file, C\_arraytest.h:

``.. image:: C_Extensions(2f)NumPy_arrays_attachments/C_arraytest.h``

The Source file, C\_arraytest.c:

``.. image:: C_Extensions(2f)NumPy_arrays_attachments/C_arraytest.c``

Now, lets look at the source code in smaller chunks.

Headers
^^^^^^^

You must include the following headers with Python.h **always** the
first header included.

I also include the header C\_arraytest.h which contains the prototype of
the matsq function:

The static keyword in front of a function declaration makes this
function private to your extension module. The linker just won't see it.
This way you can use the same intuitional function names(i.e. sum,
check, trace) for all extension modules without having name clashes
between them at link time. The type of the function is \`PyObject \*\`
because it will always be returning to a Python calling function so you
can (must, actually) return a Python object. The arguments are always
the same ,

The first one self is never used, but necessary because of how Python
passes arguments. The second args is a pointer to a Python tuple that
contains all of the arguments (B,i,x) of the function.

Method definitions
^^^^^^^^^^^^^^^^^^

This sets up a table of function names that will be the interface from
your Python code to your C extension. The name of the C extension module
will be \`\_C\_arraytest\` (note the leading underscore). It is
important to get the name right each time it is used because there are
strict requirements on using the module name in the code. The name
appears first in the method definitions table as the first part of the
table name:

where I used ellipses (...) to ignore other code not relevant to this
function. The \`METH\_VARARGS\` parameter tells the compiler that you
will pass the arguments the usual way without keywords as in the example
\`A=matsq(B,i,x)\` above. There are ways to use Python keywords, but I
have not tried them out. The table should always end with {NULL, NULL}
which is just a "marker" to note the end of the table.

Initializations
^^^^^^^^^^^^^^^

These functions tell the Python interpreter what to call when the module
is loaded. Note the name of the module (\`\_C\_arraytest\`) must come
directly after the init in the name of the initialization structure.

The order is important and you must call these two initialization
functions first.

The matsqfunction code
^^^^^^^^^^^^^^^^^^^^^^

Now here is the actual function that you will call from Python code. I
will split it up and explain each section.

The function name and type:

You can see they match the prototype in C\_arraytest.h.

The local variables:

The \`PyArrayObjects\` are structures defined in the NumPy header file
and they will be assigned pointers to the actual input and output NumPy
arrays (A and B). The C arrays \`cin\` and \`cout\` are Cpointers that
will point (eventually) to the actual data in the NumPy arrays and allow
you to manipulate it. The variable \`dfactor\` will be the Python float
y, \`ifactor\` will be the Python int i, the variables i,j,n, and m will
be loop variables (i and j) and matrix dimensions (n= number of rows, m=
number of columns) in A and B. The array dims will be used to access n
and m from the NumPy array. All this happens below. First we have to
extract the input variables (A, i, y) from the args tuple. This is done
by the call,

The \`PyArg\_ParseTuple\` function takes the args tuple and using the
format string that appears next ("O!id" ) it assigns each member of the
tuple to a C variable. Note you must pass all C variables by reference.
This is true even if the C variable is a pointer to a string (see code
in vecfcn1 routine). The format string tells the parsing function what
type of variable to use. The common variables for Python all have letter
names (e.g. s for string, i for integer, d for (double - the Python
float)). You can find a list of these and many more in Guido's tutorial
(http://docs.python.org/ext/ext.html). For data types that are not in
standard Python like the NumPy arrays you use the O! notation which
tells the parser to look for a type structure (in this case a NumPy
structure \`PyArray\_Type\`) to help it convert the tuple member that
will be assigned to the local variable ( matin ) pointing to the NumPy
array structure. Note these are also passed by reference. The order must
be maintained and match the calling interface of the Python function you
want. The format string defines the interface and if you do not call the
function from Python so the number of arguments match the number in the
format string, you will get an error. This is good since it will point
to where the problem is.

If this doesn't work we return NULL which will cause a Python exception.

Next we have a check that the input matrix really is a matrix of NumPy
type double. This test is also done in the Python wrapper for this C
extension. It is better to do it there, but I include the test here to
show you that you can do testing in the C extension and you can "reach
into" the NumPy structure to pick out it's parameters. The utility
function \`not\_doublematrix\` is explained later.

Here's an example of reaching into the NumPy structure to get the
dimensions of the matrix matin and assign them to local variables as
mentioned above.

Now we use these matrix parameters to generate a new NumPy matrix matout
(our output) right here in our C extension.
PyArray\_FromDims(2,dims,NPY\_DOUBLE) is a utility function provided by
NumPy (not me) and its arguments tell NumPy the rank of the NumPy object
(2), the size of each dimension (dims), and the data type (NPY\_DOUBLE).
Other examples of creating different NumPy arrays are in the other C
extensions.

To actually do our calculations we need C structures to handle our data
so we generate two C 2-dimensional arrays (cin and cout) which will
point to the data in matin and matout, respectively. Note, here memory
is allocated since we need to create an array of pointers to C doubles
so we can address cin and cout like usual C matrices with two indices.
This memory must be released at the end of this C extension. Memory
allocation like this is not always necessary. See the routines for NumPy
vector manipulation and treating NumPy matrices like contiguous arrays
(as they are in NumPy) in the C extension (the routine contigmat).

Finally, we get to the point where we can manipulate the matrices and do
our calculations. Here is the part where the original equation
operations *B,,ij,,*\ ='' i y *(*\ A,,ij,,'')^2^ are carried out. Note,
we are directly manipulating the data in the original NumPy arrays A and
B passed to this extension. So anything you do here to the components of
cin or cout will be done to the original matrices and will appear there
when you return to the Python code.

We are ready to go back to the Python calling routine, but first we
release the memory we allocated for cin and cout.

Now we return the result of the calculation.

If you look at the other C extensions you can see that you can also
return regular Python variables (like ints) using another
Python-provided function \`Py\_BuildValue("i", 1)\` where the string "i"
tells the function the data type and the second argument (1 here) is the
data value to be returned. If you decide to return nothing, you **must**
return the Python keyword None like this:

The Py\_INCREF function increases the number of references to None
(remember Python collects allocated memory automatically when there are
no more references to the data). You must be careful about this in the C
extensions. For more info see `Guido´s
tutorial <http://docs.python.org/ext/ext.html>`__.

The utility functions
^^^^^^^^^^^^^^^^^^^^^

Here are some quick descriptions of the matrix utility functions. They
are pretty much self-explanatory. The vector and integer array utility
functions are very similar.

The first utility function is not used in any of the C extensions here,
but I include it because a helpful person sent it along with some code
and it does show how one might convert a python object to a NumPy array.
I have not tried it. Use at your own risk.

The next one creates the C arrays that are used to point to the rows of
the NumPy matrices. This allocates arrays of pointers which point into
the NumPy data. The NumPy data is contiguous and strides (m) are used to
access each row. This function calls ptrvector(n) which does the actual
memory allocation. Remember to deallocate memory after using this one.

Here is where the memory for the C arrays of pointers is allocated. It's
a pretty standard memory allocator for arrays.

This is the routine to deallocate the memory.

**Note**: There is a standard C-API for converting from Python objects
to C-style arrays-of-pointers called PyArray\_AsCArray

Here is a utility function that checks to make sure the object produced
by the parse is a NumPy matrix. You can see how it reaches into the
NumPy object structure.

The C Code – other variations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As I mentioned in the introduction the functions are repetitious. All
the other functions follow a very similar pattern. They are given a line
in the methods structure, they have the same arguments, they parse the
arguments, they may check the C structures after the parsing, they set
up C variables to manipulate which point to the input data, they do the
actual calculation, they deallocate memory (if necessary) and they
return something for Python (either None or a Python object). I'll just
mention some of the differences in the code from the above matrix C
extension matsq.

**vecfcn1:**

The format string for the parse function specifies that a variable from
Python is a string (s).

No memory is allocated in the pointer assignments for the local C arrays
because they are already vectors.

The return is an int = 1 if successful. This is returned as a Python
int.

**rowx2:**

In this routine we "pass back" the output using the fact that it is
passed by reference in the argument tuple list and is changed in place
by the manipulations. Compare this to returning an array from the C
extension in matsq. Either one gets the job done.

**contigmat:**

Here the matrix data is treated like a long vector (just like stacking
the rows end to end). This is useful if you have array classes in C++
which store the data as one long vector and then use strides to access
it like an array (two-dimensional, three-dimensional, or whatever). Even
though matin and matout are "matrices" we treat them like vectors and
use the vector utilities to get our C pointers cin and cout.

For other utility functions note that we use different rank, dimensions,
and NumPy parameters (e.g. \`NPY\_LONG\`) to tell the routines we are
calling what the data types are.

The Make File
~~~~~~~~~~~~~

The make file is very simple. It is written for Mac OS X 10.4, as BSD
Unix.

The compile step is pretty standard. You do need to add paths to the
Python headers:

\`-I/Library/Frameworks/Python.framework/Versions/2.4/include/python2.4\`

and paths to NumPy headers:

\`-I/Library/Frameworks/Python.framework/Versions/2.4/lib/python2.4/site-packages/numpy/core/include/numpy/\`

These paths are for a Framework Python 2.4 install on Mac OS X. You need
to supply paths to the headers installed on your computer. They may be
different. My guess is the gcc flags will be the same for the compile.

The link step produces the actual module (\`\_C\_arraytest.so\`) which
can be imported to Python code. This is specific to the Mac OS X system.
On Linux or Windows you will need something different. I have been
searching for generic examples, but I'm not sure what I found would work
for most people so I chose not to display the findings there. I cannot
judge whether the code is good for those systems.

Note, again the name of the produced shared library **must** match the
name in the initialization and methods definition calls in the C
extension source code. Hence the leading underline in the name
\`\_C\_arraytest.so\`.

Here's my modified Makefile which compiles this code under Linux (save
it as Makefile in the same directory, and then run 'make' --PaulIvanov

The Python Wrapper Code
~~~~~~~~~~~~~~~~~~~~~~~

Here as in the C code I will only show detailed description of one
wrapper function and its use. There is so much repetition that the other
wrappers will be clear if you understand one. I will again use the matsq
function. This is the code that will first be called when you invoke the
matsq function in your own Python code after importing the wrapper
module (\`C\_arraytest.py\`) which automatically imports and uses (in a
way hidden from the user) the actual C extensions in
\`\_C\_arraytest.so\` (note the leading underscore which keeps the names
separate).

**imports:**

Import the C extensions, NumPy, and the system module (used for the exit
statement at the end which is optional).

**The definition of the Python matsq function**

Pass a NumPy matrix (matin), a Python int (ifac), and a Python float
(dfac). Check the arguments to make sure they are the right type and
dimensions and size. This is much easier and safer on the Python side
which is why I do it here even though I showed a way to do some of this
in the C extensions.

Finally, call the C extension to do the actual calculation on matin.

You can see that the python part is the easiest.

**Using your C extension**

If the test function \`mattest2\` were in another module (one you were
writing), here's how you would use it to call the wrapped matsq function
in a script.

The output looks like this:

The output of all the test functions is in the file \`C\_arraytest\`
output.

Summary
-------

This is the first draft explaining the C extensions I wrote (with help).
If you have comments on the code, mistakes, etc. Please post them on the
pythonmac email list. I will see them.

``. ``\ *``...or`` ``better`` ``edit`` ``the`` ``wiki!``*

I wrote the C extensions to work with NumPy although they were
originally written for Numeric. If you must use Numeric you should test
them to see if they are compatible. I suspect names like
\`NPY\_DOUBLE\`, for example, will not be. I strongly suggest you
upgrade to the NumPy since it is the future of Numeric in Python. It's
worth the effort.

Comments?!
==========

Note that this line, while in the header file above, is missing from the
.h in the tar-ball.

``-- PaulIvanov``

--------------

The output file name should be \_C\_arraytest\_d.pyd for Debug version
and \_C\_arraytest.pyd for Release version.

``-- Geoffrey Zhu``

--------------

ptrvector() allocates n\*sizeof(double), but should really allocate
pointers to double; so: n\*sizeof(double \*)

``-- Peter Meerwald``

--------------

``* In vecsq(), line 86, since you will be creating a 1-dimensional vector:  ``

\ ``  ``

``should be:  ``

``* In ``\ \ ``, ``\ \ `` is never used.``

``-- FredSpiessens``

--------------

CategoryCookbook

