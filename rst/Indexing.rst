Indexing numpy arrays
=====================

The whole point of numpy is to introduce a multidimensional array object
for holding homogeneously-typed numerical data. This is of course a
useful tool for storing data, but it is also possible to manipulate
large numbers of values without writing inefficient python loops. To
accomplish this, one needs to be able to refer to elements of the arrays
in many different ways, from simple "slices" to using arrays as lookup
tables. The purpose of this page is to go over the various different
types of indexing available. Hopefully the sometimes-peculiar syntax
will also become more clear.

TableOfContents

We will use the same arrays as examples wherever possible:



.. code-block:: python

    #!python numbers=disable
    >>> import numpy as np
    >>>
    >>> A = np.arange(10)
    >>> A
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>>
    >>> B = np.reshape(np.arange(9),(3,3))
    >>> B
    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])
    >>>
    >>> C = np.reshape(np.arange(2*3*4),(2,3,4))
    >>> C
    array([[[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11]],
           [[12, 13, 14, 15],
            [16, 17, 18, 19],
            [20, 21, 22, 23]]])
    >>>
    



Elements
--------

The simplest way to pick one or some elements of an array looks very
similar to python lists:



.. code-block:: python

    #!python numbers=disable
    >>> A[1]
    1
    >>> B[1,0]
    3
    >>> C[1,0,2]
    14
    



That is, to pick out a particular element, you simply put the indices
into square brackets after it. As is standard for python, element
numbers start at zero.

If you want to change an array value in-place, you can simply use the
syntax above in an assignment:



.. code-block:: python

    #!python numbers=disable
    >>> T = A.copy()
    >>> T[3] = -5
    >>> T
    array([ 0,  1,  2, -5,  4,  5,  6,  7,  8,  9])
    >>> T[0] += 7
    >>> T
    array([ 7,  1,  2, -5,  4,  5,  6,  7,  8,  9])
    



(The business with .copy() is to ensure that we don't actually modify A,
since that would make further examples confusing.) Note that numpy also
supports python's "augmented assignment" operators, +=, -=, \*=, and so
on.

Be aware that the type of array elements is a property of the array
itself, so that if you try to assign an element of another type to an
array, it will be silently converted (if possible):



.. code-block:: python

    #!python numbers=disable
    >>> T = A.copy()
    >>> T[3] = -1.5
    >>> T
    array([ 0,  1,  2, -1,  4,  5,  6,  7,  8,  9])
    >>> T[3] = -0.5j
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    TypeError: can't convert complex to int; use int(abs(z))
    >>> T
    array([ 0,  1,  2, -1,  4,  5,  6,  7,  8,  9])
    



Note that the conversion that happens is a default conversion; in the
case of float to int conversion, it's truncation. If you wanted
something different, say taking the floor, you would have to arrange
that yourself (for example with np.floor()). In the case of converting
complex values to integers, there's no resonable default way to do it,
so numpy raises an exception and leaves the array unchanged.

Finally, two slightly more technical matters.

If you want to manipulate indices programmatically, you should know that
when you write something like



.. code-block:: python

    #!python numbers=disable
    >>> C[1,0,1]
    13
    



it is the same as (in fact it is internally converted to)



.. code-block:: python

    #!python numbers=disable
    >>> C[(1,0,1)]
    13
    



This peculiar-looking syntax is constructing a tuple, python's data
structure for immutable sequences, and using that tuple as an index into
the array. (Under the hood, C[1,0,1] is converted to
C.\_\_getitem\_\_((1,0,1)).) This means you can whip up tuples if you
want to:



.. code-block:: python

    #!python numbers=disable
    >>> i = (1,0,1)
    >>> C[i]
    13
    



If it doesn't seem likely you would ever want to do this, consider
iterating over an arbitrarily multidimensional array:



.. code-block:: python

    #!python numbers=disable
    >>> for i in np.ndindex(B.shape):
    ...     print i, B[i]
    ... 
    (0, 0) 0
    (0, 1) 1
    (0, 2) 2
    (1, 0) 3
    (1, 1) 4
    (1, 2) 5
    (2, 0) 6
    (2, 1) 7
    (2, 2) 8
    



Indexing with tuples will also become important when we start looking at
fancy indexing and the function np.where().

The last technical issue I want to mention is that when you select an
element from an array, what you get back has the same type as the array
elements. This may sound obvious, and in a way it is, but keep in mind
that even innocuous numpy arrays like our A, B, and C often contain
types that are not quite the python types:



.. code-block:: python

    #!python numbers=disable
    >>> a = C[1,2,3]
    >>> a
    23
    >>> type(a)
    <type 'numpy.int32'>
    >>> type(int(a))
    <type 'int'>
    >>> a**a
    Warning: overflow encountered in long_scalars
    -1276351769
    >>> int(a)**int(a)
    20880467999847912034355032910567L
    



numpy scalars also support certain indexing operations, for consistency,
but these are somewhat subtle and under discussion.

Slices
------

It is obviously essential to be able to work with single elements of an
array. But one of the selling points of numpy is the ability to do
operations "array-wise":



.. code-block:: python

    #!python numbers=disable
    >>> 2*A
    array([ 0,  2,  4,  6,  8, 10, 12, 14, 16, 18])
    



This is handy, but one very often wants to work with only part of an
array. For example, suppose one wants to compute the array of
differences of A, that is, the array whose elements are A[1]-A[0],
A[2]-A[1], and so on. (In fact, the function np.diff does this, but
let's ignore that for expositional convenience.) numpy makes it possible
to do this using array-wise operations:



.. code-block:: python

    #!python numbers=disable
    >>> A[1:]
    array([1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> A[:-1]
    array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    >>> A[1:] - A[:-1]
    array([1, 1, 1, 1, 1, 1, 1, 1, 1])
    



This is done by making an array that is all but the first element of A,
an array that is all but the last element of A, and subtracting the
corresponding elements. The process of taking subarrays in this way is
called "slicing".

One-dimensional slices
~~~~~~~~~~~~~~~~~~~~~~

The general syntax for a slice is *array*\ [*start*:*stop*:*step*]. Any
or all of the values *start*, *stop*, and *step* may be left out (and if
*step* is left out the colon in front of it may also be left out):



.. code-block:: python

    #!python numbers=disable
    >>> A[5:]
    array([5, 6, 7, 8, 9])
    >>> A[:5]
    array([0, 1, 2, 3, 4])
    >>> A[::2]
    array([0, 2, 4, 6, 8])
    >>> A[1::2]
    array([1, 3, 5, 7, 9])
    >>> A[1:8:2]
    array([1, 3, 5, 7])
    



As usual for python, the *start* index is included and the *stop* index
is not included. Also as usual for python, negative numbers for *start*
or *stop* count backwards from the end of the array:



.. code-block:: python

    #!python numbers=disable
    >>> A[-3:]
    array([7, 8, 9])
    >>> A[:-3]
    array([0, 1, 2, 3, 4, 5, 6])
    



If *stop* comes before *start* in the array, then an array of length
zero is returned:



.. code-block:: python

    #!python numbers=disable
    >>> A[5:3]
    array([], dtype=int32)
    



(The "dtype=int32" is present in the printed form because in an array
with no elements, one cannot tell what type the elements have from their
printed representation. It nevertheless makes sense to keep track of the
type that they would have if the array had any elements.)

If you specify a slice that happens to have only one element, you get an
array in return that happens to have only one element:



.. code-block:: python

    #!python numbers=disable
    >>> A[5:6]
    array([5])
    >>> A[5]
    5
    



This seems fairly obvious and reasonable, but when dealing with fancy
indexing and multidimensional arrays it can be surprising.

If the number *step* is negative, the step through the array is
negative, that is, the new array contains (some of) the elements of the
original in reverse order:



.. code-block:: python

    #!python numbers=disable
    >>> A[::-1]
    array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
    



This is extremely useful, but it can be confusing when *start* and
*stop* are given:



.. code-block:: python

    #!python numbers=disable
    >>> A[5:3:-1]
    array([5, 4])
    >>> A[3:5:1]
    array([3, 4])
    



The rule to remember is: whether *step* is positive or negative, *start*
is always included and *stop* never is.

Just as one can retrieve elements of an array as a subarray rather than
one-by-one, one can modify them as a subarray rather than one-by-one:



.. code-block:: python

    #!python numbers=disable
    >>> T = A.copy()
    >>> T
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> T[1::2]
    array([1, 3, 5, 7, 9])
    >>> T[1::2] = -np.arange(5)
    >>> T[1::2]
    array([ 0, -1, -2, -3, -4])
    >>> T
    array([ 0,  0,  2, -1,  4, -2,  6, -3,  8, -4])
    



If the array you are trying to assign is the wrong shape, an exception
is raised:



.. code-block:: python

    #!python numbers=disable
    >>> T = A.copy()
    >>> T[1::2] = np.arange(6)
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    ValueError: shape mismatch: objects cannot be broadcast to a single shape
    >>> T[:4] = np.array([[0,1],[1,0]])
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    ValueError: shape mismatch: objects cannot be broadcast to a single shape
    



If you think the error message sounds confusing, I have to agree, but
there is a reason. In the first case, we tried to stuff six elements
into five slots, so numpy refused. In the second case, there were the
right number of elements - four - but we tried to stuff a two-by-two
array where there was supposed to be a one-dimensional array of length
four. While numpy could have coerced the two-by-two array into the right
shape, instead the designers chose to follow the python philosophy
"explicit is better than implicit" and leave any coercing up to the
user. Let's do that, though:



.. code-block:: python

    #!python numbers=disable
    >>> T = A.copy()
    >>> T[:4] = np.array([[0,1],[1,0]]).ravel()
    >>> T
    array([0, 1, 1, 0, 4, 5, 6, 7, 8, 9])
    



So in order for assignment to work, it is not simply enough to have the
right number of elements - they must be arranged in an array of the
right shape.

There is another issue complicating the error message: numpy has some
extremely convenient rules for converting lower-dimensional arrays into
higher-dimensional arrays, and for implicitly repeating arrays along
axes. This process is called "broadcasting". We will see more of it
elsewhere, but here it is in its simplest possible form:



.. code-block:: python

    #!python numbers=disable
    >>> T = A.copy()
    >>> T[1::2] = -1
    >>> T
    array([ 0, -1,  2, -1,  4, -1,  6, -1,  8, -1])
    



We told numpy to take a scalar, -1, and put it into an array of length
five. Rather than signal an error, numpy's broadcasting rules tell it to
convert this scalar into an effective array of length five by repeating
the scalar five times. (It does not, of course, actually create a
temporary array of this size; in fact it uses a clever trick of telling
itself that the temporary array has its elements spaced zero bytes
apart.) This particular case of broadcasting gets used all the time:



.. code-block:: python

    #!python numbers=disable
    >>> T = A.copy()
    >>> T[1::2] -= 1
    >>> T
    array([0, 0, 2, 2, 4, 4, 6, 6, 8, 8])
    



Assignment is sometimes a good reason to use the "everything" slice:



.. code-block:: python

    #!python numbers=disable
    >>> T = A.copy()
    >>> T[:] = -1
    >>> T
    array([-1, -1, -1, -1, -1, -1, -1, -1, -1, -1])
    >>> T = A.copy()
    >>> T = -1
    >>> T
    -1
    



What happened here? Well, in the first case we told numpy to assign -1
to all the elements of T, so that's what it did. In the second case, we
told python "T = -1". In python, variables are just names that can be
attached to objects in memory. This is in sharp contrast with languages
like C, where a variable is a named region of memory where data can be
stored. Assignment to a variable name - T in this case - simply changes
which object the name refers to, without altering the underlying object
in any way. (If the name was the only reference to the original object,
it becomes impossible for your program ever to find it again after the
reassignment, so python deletes the original object to free up some
memory.) In a language like C, assigning to a variable changes the value
stored in that memory region. If you really must think in terms of C,
you can think of all python variables as holding pointers to actual
objects; assignment to a python variable is just modification of the
pointer, and doesn't affect the object pointed to (unless garbage
collection deletes it). In any case, if you want to modify the
*contents* of an array, you can't do it by assigning to the name you
gave the array; you must use slice assignment or some other approach.

Finally, a technical point: how can a program work with slices
programmatically? What if you want to, say, save a slice specification
to apply to many arrays later on? The answer is to use a slice object,
which is constructed using slice():



.. code-block:: python

    #!python numbers=disable
    >>> A[1::2]
    array([1, 3, 5, 7, 9])
    >>> s = slice(1,None,2)
    >>> A[s]
    array([1, 3, 5, 7, 9])
    



(Regrettably, you can't just write "s = 1::2". But within square
brackets, 1::2 is converted internally to slice(1,None,2).) You can
leave out arguments to slice() just like you can with the colon
notation, with one exception:



.. code-block:: python

    #!python numbers=disable
    >>> A[slice(-3)]
    array([0, 1, 2, 3, 4, 5, 6])
    >>> A[slice(None,3)]
    array([0, 1, 2])
    >>> A[slice()]
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    TypeError: slice expected at least 1 arguments, got 0
    >>> A[slice(None,None,None)]
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    



Multidimensional slices
~~~~~~~~~~~~~~~~~~~~~~~

One-dimensional arrays are extremely useful, but often one has data that
is naturally multidimensional - image data might be an N by M array of
pixel values, or an N by M by 3 array of colour values, for example.
Just as it is useful to take slices of one-dimensional arrays, it is
useful to take slices of multidimensional arrays. This is fairly
straightforward:



.. code-block:: python

    #!python numbers=disable
    >>> B
    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])
    >>> B[:2,:]
    array([[0, 1, 2],
           [3, 4, 5]])
    >>> B[:,::-1]
    array([[2, 1, 0],
           [5, 4, 3],
           [8, 7, 6]])
    



Essentially one simply specifies a one-dimensional slice for each axis.
One can also supply a number for an axis rather than a slice:



.. code-block:: python

    #!python numbers=disable
    >>> B[0,:]
    array([0, 1, 2])
    >>> B[0,::-1]
    array([2, 1, 0])
    >>> B[:,0]
    array([0, 3, 6])
    



Notice that when one supplies a number for (say) the first axis, the
result is no longer a two-dimensional array; it's now a one-dimensional
array. This makes sense:



.. code-block:: python

    #!python numbers=disable
    >>> B[:,:]
    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])
    >>> B[0,:]
    array([0, 1, 2])
    >>> B[0,0]
    0
    



If you supply no numbers, you get a two-dimensional array; if you supply
one number, the dimension drops by one, and you get a one-dimensional
array; and if you supply two numbers the dimension drops by two and you
get a scalar. (If you think you should get a zero-dimensional array, you
are opening a can of worms. The distinction, or lack thereof, between
scalars and zero-dimensional arrays is an issue under discussion and
development.)

If you are used to working with matrices, you may want to preserve a
distinction between "row vectors" and "column vectors". numpy supports
only one kind of one-dimensional array, but you could represent row and
column vectors as *two*-dimensional arrays, one of whose dimensions
happens to be one. Unfortunately indexing of these objects then becomes
cumbersome.

As with one-dimensional arrays, if you specify a slice that happens to
have only one element, you get an array one of whose axes has length 1 -
the axis doesn't "disappear" the way it would if you had provided an
actual number for that axis:



.. code-block:: python

    #!python numbers=disable
    >>> B[:,0:1]
    array([[0],
           [3],
           [6]])
    >>> B[:,0]
    array([0, 3, 6])
    



numpy also has a few shortcuts well-suited to dealing with arrays with
an indeterminate number of dimensions. If this seems like something
unreasonable, keep in mind that many of numpy's functions (for example
np.sort(), np.sum(), and np.transpose()) must work on arrays of
arbitrary dimension. It is of course possible to extract the number of
dimensions from an array and work with it explicitly, but one's code
tends to fill up with things like (slice(None,None,None),)\*(C.ndim-1),
making it unpleasant to read. So numpy has some shortcuts which often
simplify things.

First the Ellipsis object:



.. code-block:: python

    #!python numbers=disable
    >>> A[...]
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> B[...]
    array([[0, 1, 2],
           [3, 4, 5],
           [6, 7, 8]])
    >>> B[0,...]
    array([0, 1, 2])
    >>> B[0,...,0]
    array(0)
    >>> C[0,...,0]
    array([0, 4, 8])
    >>> C[0,Ellipsis,0]
    array([0, 4, 8])
    



The ellipsis (three dots) indicates "as many ':' as needed". (Its name
for use in index-fiddling code is Ellipsis, and it's not
numpy-specific.) This makes it easy to manipulate only one dimension of
an array, letting numpy do array-wise operations over the "unwanted"
dimensions. You can only really have one ellipsis in any given indexing
expression, or else the expression would be ambiguous about how many ':'
should be put in each. (In fact, for some reason it is allowed to have
something like "C[...,...]"; this is not actually ambiguous.)

In some circumstances, it is convenient to omit the ellipsis entirely:



.. code-block:: python

    #!python numbers=disable
    >>> B[0]
    array([0, 1, 2])
    >>> C[0]
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11]])
    >>> C[0,0]
    array([0, 1, 2, 3])
    >>> B[0:2]
    array([[0, 1, 2],
           [3, 4, 5]])
    



If you don't supply enough indices to an array, an ellipsis is silently
appended. This means that in some sense you can view a two-dimensional
array as an array of one-dimensional arrays. In combination with numpy's
array-wise operations, this means that functions written for
one-dimensional arrays can often just work for two-dimensional arrays.
For example, recall the difference operation we wrote out in the section
on one-dimensional slices:



.. code-block:: python

    #!python numbers=disable
    >>> A[1:] - A[:-1]
    array([1, 1, 1, 1, 1, 1, 1, 1, 1])
    >>> B[1:] - B[:-1]
    array([[3, 3, 3],
           [3, 3, 3]])
    



It works, unmodified, to take the differences along the first axis of a
two-dimensional array.

Writing to multidimensional slices works just the way writing to
one-dimensional slices does:



.. code-block:: python

    >>> T = B.copy()
    >>> T[1,:] = -1
    >>> T
    array([[ 0,  1,  2],
           [-1, -1, -1],
           [ 6,  7,  8]])
    >>> T[:,:2] = -2
    >>> T
    array([[-2, -2,  2],
           [-2, -2, -1],
           [-2, -2,  8]])
    



FIXME: np.newaxis and broadcasting rules.

Views versus copies
~~~~~~~~~~~~~~~~~~~

FIXME: Zero-dimensional arrays, views of a single element.

Fancy indexing
--------------

Slices are very handy, and the fact that they can be created as views
makes them efficient. But some operations cannot really be done with
slices; for example, suppose one wanted to square all the negative
values in an array. Short of writing a loop in python, one wants to be
able to locate the negative values, extract them, square them, and put
the new values where the old ones were:



.. code-block:: python

    #!python numbers=disable
    >>> T = A.copy() - 5
    >>> T[T<0] **= 2
    >>> T
    array([25, 16,  9,  4,  1,  0,  1,  2,  3,  4])
    



Or suppose one wants to use an array as a lookup table, that is, for an
array B, produce an array whose i,j th element is LUT[B[i,j]]: FIXME:
argsort is a better example



.. code-block:: python

    #!python numbers=disable
    >>> LUT = np.sin(A)
    >>> LUT
    array([ 0.        ,  0.84147098,  0.90929743,  0.14112001, -0.7568025 ,
           -0.95892427, -0.2794155 ,  0.6569866 ,  0.98935825,  0.41211849])
    >>> LUT[B]
    array([[ 0.        ,  0.84147098,  0.90929743],
           [ 0.14112001, -0.7568025 , -0.95892427],
           [-0.2794155 ,  0.6569866 ,  0.98935825]])
    



For this sort of thing numpy provides what is called "fancy indexing".
It is not nearly as quick and lightweight as slicing, but it allows one
to do some rather sophisticated things while letting numpy do all the
hard work in C.

Boolean indexing
~~~~~~~~~~~~~~~~

It frequently happens that one wants to select or modify only the
elements of an array satisfying some condition. numpy provides several
tools for working with this sort of situation. The first is boolean
arrays. Comparisons - equal to, less than, and so on - between numpy
arrays produce arrays of boolean values:



.. code-block:: python

    #!python numbers=disable
    >>> A<5
    array([ True,  True,  True,  True,  True, False, False, False, False, False], dt
    ype=bool)
    



These are normal arrays. The actual storage type is normally a single
byte per value, not bits packed into a byte, but boolean arrays offer
the same range of indexing and array-wise operations as other arrays.
Unfortunately, python's "and" and "or" cannot be overridden to do
array-wise operations, so you must use the bitwise operations "&", "\|",
and "^" (for exclusive-or). Similarly python's chained inequalities
cannot be overridden. Also, regrettably, one cannot chage the precence
of the bitwise operators:



.. code-block:: python

    #!python numbers=disable
    >>> c = A<5 & A>1
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    ValueError: The truth value of an array with more than one element is ambiguous.
     Use a.any() or a.all()
    >>> c = (A<5) & (A>1)
    >>> c
    array([False, False,  True,  True,  True, False, False, False, False, False], dt
    ype=bool)
    



Nevertheless, numpy's boolean arrays are extremely powerful.

One can use boolean arrays to extract values from arrays:



.. code-block:: python

    #!python numbers=disable
    >>> c = (A<5) & (A>1)
    >>> A[c]
    array([2, 3, 4])
    



The result is necessarily a copy of the original array, rather than a
view, since it will not normally be the case the the elements of c that
are True select an evenly-strided memory layout. Nevertheless it is also
possible to use boolean arrays to write to specific elements:



.. code-block:: python

    >>> T = A.copy()
    >>> c = (A<5) & (A>1)
    >>> T[c] = -7
    >>> T
    array([ 0,  1, -7, -7, -7,  5,  6,  7,  8,  9])
    



FIXME: mention where()

Multidimensional boolean indexing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Boolean indexing works for multidimensional arrays as well. In its
simplest (and most common) incarnation, you simply supply a single
boolean array as index, the same shape as the original array:



.. code-block:: python

    >>> C[C%5==0]
    array([ 0,  5, 10, 15, 20])
    



You then get back a one-dimensional array of the elements for which the
condition is True. (Note that the array must be one-dimensional, since
the boolean values can be arranged arbitrarily around the array. If you
want to keep track of the arrangement of values in the original array,
look into using numpy's "masked array" tools.) You can also use boolean
indexing for assignment, just as you can for one-dimensional arrays.

Two very useful operations on boolean arrays are np.any() and np.all():



.. code-block:: python

    >>> np.any(B<5)
    True
    >>> np.all(B<5)
    False
    



They do just what they say on the tin, evaluate whether any entry in the
boolean matrix is True, or whether all elements in the boolean matrix
are True. But they can also be used to evaluate "along an axis", for
example, to produce a boolean array saying whether any element in a
given row is True:



.. code-block:: python

    >>> B<5
    array([[ True,  True,  True],
           [ True,  True, False],
           [False, False, False]], dtype=bool)
    >>> np.any(B<5, axis=1)
    array([ True,  True, False], dtype=bool)
    >>> np.all(B<5, axis=1)
    array([ True, False, False], dtype=bool)
    



One can also use boolean indexing to pull out rows or columns meeting
some criterion:



.. code-block:: python

    >>> B[np.any(B<5, axis=1),:]
    array([[0, 1, 2],
           [3, 4, 5]])
    



The result here is two-dimensional because there is one dimension for
the results of the boolean indexing, and one dimension because each row
is one-dimensional.

This works with higher-dimensional boolean arrays as well:



.. code-block:: python

    >>> c = np.any(C<5,axis=2)
    >>> c
    array([[ True,  True, False],
           [False, False, False]], dtype=bool)
    >>> C[c,:]
    array([[0, 1, 2, 3],
           [4, 5, 6, 7]])
    



Here too the result is two-dimensional, though that is perhaps a little
more surprising. The boolean array is two-dimensional, but the part of
the return value corresponding to the boolean array must be
one-dimensional, since the True values may be distributed arbitrarily.
The subarray of C corresponding to each True or False value is
one-dimensional, so we get a return array of dimension two.

Finally, if you want to apply boolean conditions to the rows and columns
simultaneously, beware:



.. code-block:: python

    >>> B[np.array([True, False, True]), np.array([False, True, True])]
    array([1, 8])
    >>> B[np.array([True, False, True]),:][:,np.array([False, True, True])]
    array([[1, 2],
           [7, 8]])
    



The obvious approach doesn't give the right answer. I don't know why
not, or why it produces the value that it does. You can get the right
answer by indexing twice, but that's clumsy and inefficient and doesn't
allow assignment.

FIXME: works with too-small boolean arrays for some reason?

List-of-locations indexing
~~~~~~~~~~~~~~~~~~~~~~~~~~

It happens with some frequency that one wants to pull out values at a
particular location in an array. If one wants a single location, one can
just use simple indexing. But if there are many locations, you need
something a bit more clever. Fortunately numpy supports a mode of fancy
indexing that accomplishes this:



.. code-block:: python

    >>> primes = np.array([2,3,5,7,11,13,17,19,23])
    >>> idx = [3,4,1,2,2]
    >>> primes[idx]
    array([ 7, 11,  3,  5,  5])
    >>> idx = np.array([3,4,1,2,2])
    >>> primes[idx]
    array([ 7, 11,  3,  5,  5])
    



When you index with an array that is not an array of booleans, or with a
list, numpy views it as an array of indices. The array can be any shape,
and the returned array has the same shape:



.. code-block:: python

    >>> primes = np.array([2,3,5,7,11,13,17,19,23,29,31])
    >>> primes[B]
    array([[ 2,  3,  5],
           [ 7, 11, 13],
           [17, 19, 23]])
    



Effectively this uses the original array as a look-up table.

You can also assign to arrays in this way:



.. code-block:: python

    >>> T = A.copy()
    >>> T[ [1,3,5,0] ] = -np.arange(4)
    >>> T
    array([-3,  0,  2, -1,  4, -2,  6,  7,  8,  9])
    



**Warning:** Augmented assignment - the operators like "+=" - works, but
it does not necessarily do what you would expect. In particular,
repeated indices do not result in the value getting added twice:



.. code-block:: python

    >>> T = A.copy()
    >>> T[ [0,1,2,3,3,3] ] += 10
    >>> T
    array([10, 11, 12, 13,  4,  5,  6,  7,  8,  9])
    



This is surprising, inconvenient, and unfortunate, but it is a direct
result of how python implements the "+=" operators. The most common case
for doing this is something histogram-like:



.. code-block:: python

    >>> bins = np.zeros(5,dtype=np.int32)
    >>> pos = [1,0,2,0,3]
    >>> wts = [1,2,1,1,4]
    >>> bins[pos]+=wts
    >>> bins
    array([1, 1, 1, 4, 0])
    



Unfortunately this gives the wrong answer. In older versions of numpy
there was no really satisfactory solution, but as of numpy 1.1, the
histogram function can do this:



.. code-block:: python

    >>> bins = np.zeros(5,dtype=np.int32)
    >>> pos = [1,0,2,0,3]
    >>> wts = [1,2,1,1,4]
    >>> np.histogram(pos,bins=5,range=(0,5),weights=wts,new=True)
    (array([3, 1, 1, 4, 0]), array([ 0.,  1.,  2.,  3.,  4.,  5.]))
    



FIXME: mention put() and take()

Multidimensional list-of-locations indexing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

One can also, not too surprisingly, use list-of-locations indexing on
multidimensional arrays. The syntax is, however, a bit surprising. Let's
suppose we want the list [B[0,0],B[1,2],B[0,1]]. Then we write:



.. code-block:: python

    >>> B[ [0,1,0], [0,2,1] ]
    array([0, 5, 1])
    >>> [B[0,0],B[1,2],B[0,1]]
    [0, 5, 1]
    



This may seem weird - why not provide a list of tuples representing
coordinates? Well, the reason is basically that for large arrays, lists
and tuples are very inefficient, so numpy is designed to work with
arrays only, for indices as well as values. This means that something
like B[ [(0,0),(1,2),(0,1)] ] looks just like indexing B with a
two-dimensional array, which as we saw above just means that B should be
used as a look-up table yielding a two-dimensional array of results
(each of which is one-dimensional, as usual when we supply only one
index to a two-dimensional array).

In summary, in list-of-locations indexing, you supply an array of values
for each coordinate, all the same shape, and numpy returns an array of
the same shape containing the values obtained by looking up each set of
coordinates in the original array. If the coordinate arrays are not the
same shape, numpy's broadcasting rules are applied to them to try to
make their shapes the same. If there are not as many arrays as the
original array has dimensions, the original array is regarded as
containing arrays, and the extra dimensions appear on the result array.

Fortunately, most of the time when one wants to supply a list of
locations to a multidimensional array, one got the list from numpy in
the first place. A normal way to do this is something like:



.. code-block:: python

    >>> idx = np.nonzero(B%2)
    >>> idx
    (array([0, 1, 1, 2]), array([1, 0, 2, 1]))
    >>> B[idx]
    array([1, 3, 5, 7])
    >>> B[B%2 != 0]
    array([1, 3, 5, 7])
    



Here nonzero() takes an array and returns a list of locations (in the
correct format) where the array is nonzero. Of course, one can also
index directly into the array with a boolean array; this will be much
more efficient unless the number of nonzero locations is small and the
indexing is done many times. But sometimes it is valuable to work with
the list of indices directly.

Picking out rows and columns
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

One unfortunate consequence of numpy's list-of-locations indexing syntax
is that users used to other array languages expect it to pick out rows
and columns. After all, it's quite reasonable to want to pull out a list
of rows and columns from a matrix. So numpy provides a convenience
function, ix\_() for doing this:



.. code-block:: python

    >>> B[ np.ix_([0,2],[0,2]) ]
    array([[0, 2],
           [6, 8]])
    >>> np.ix_([0,2],[0,2])
    (array([[0],
           [2]]), array([[0, 2]]))
    



The way it works is by taking advantage of numpy's broadcasting
facilities. You can see that the two arrays used as row and column
indices have different shapes; numpy's broadcasting repeats each along
the too-short axis so that they conform.

Mixed indexing modes
--------------------

What happens when you try to mix slice indexing, element indexing,
boolean indexing, and list-of-locations indexing?

How indexing works under the hood
---------------------------------

A numpy array is a block of memory, a data type for interpreting memory
locations, a list of sizes, and a list of strides. So for example,
C[i,j,k] is the element starting at position
i\*strides[0]+j\*strides[1]+k\*strides[2]. This means, for example, that
transposing amatrix can be done very efficiently: just reverse the
strides and sizes arrays. This is why slices are efficient and can
return views, but fancy indexing is slower and can't.

At a python level, numpy's indexing works by overriding the
\_\_getitem\_\_ and \_\_setitem\_\_ methods in an ndarray object. These
methods are called when arrays are indexed, and they allow arbitrary
implementations:



.. code-block:: python

    >>> class IndexDemo:
    ...     def __getitem__(self, *args):
    ...         print "__getitem__", args
    ...         return 1
    ...     def __setitem__(self, *args):
    ...         print "__setitem__", args
    ...     def __iadd__(self, *args):
    ...         print "__iadd__", args
    ... 
    >>> 
    >>> T = IndexDemo()
    >>> T[1]
    __getitem__ (1,)
    1
    >>> T["fish"]
    __getitem__ ('fish',)
    1
    >>> T[A]
    __getitem__ (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]),)
    1
    >>> T[1,2]
    __getitem__ ((1, 2),)
    1
    >>> T[1] = 7
    __setitem__ (1, 7)
    >>> T[1] += 7
    __getitem__ (1,)
    __setitem__ (1, 8)
    



Array-like objects
------------------

numpy and scipy provide a few other types that behave like arrays, in
particular matrices and sparse matrices. Their indexing can differ from
that of arrays in surprising ways.

