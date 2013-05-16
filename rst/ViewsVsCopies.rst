Views versus copies in NumPy
============================

From time to time, people write to the !NumPy list asking in which cases
a view of an array is created and in which it isn't. This page tries to
clarify some tricky points on this rather subtle subject.

What is a view of a NumPy array?
--------------------------------

As its name is saying, it is simply another way of **viewing** the data
of the array. Technically, that means that the data of both objects is
*shared*. You can create views by selecting a **slice** of the original
array, or also by changing the **dtype** (or a combination of both).
These different kinds of views are described below.

Slice views
-----------

This is probably the most common source of view creations in !NumPy. The
rule of thumb for creating a slice view is that the viewed elements can
be addressed with offsets, strides, and counts in the original array.
For example:



.. code-block:: python

    >>> a = numpy.arange(10)
    >>> a
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> v1 = a[1:2]
    >>> v1
    array([1])
    >>> a[1] = 2
    >>> v1
    array([2])
    >>> v2 = a[1::3]
    >>> v2
    array([2, 4, 7])
    >>> a[7] = 10
    >>> v2
    array([ 2,  4, 10])
    



In the above code snippet, () and () are views of , because if you
update , and will reflect the change.

Dtype views
-----------

Another way to create array views is by assigning another **dtype** to
the same data area. For example:



.. code-block:: python

    >>> b = numpy.arange(10, dtype='int16')
    >>> b
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int16)
    >>> v3 = b.view('int32')
    >>> v3 += 1
    >>> b
    array([1, 1, 3, 3, 5, 5, 7, 7, 9, 9], dtype=int16)
    >>> v4 = b.view('int8')
    >>> v4
    array([1, 0, 1, 0, 3, 0, 3, 0, 5, 0, 5, 0, 7, 0, 7, 0, 9, 0, 9, 0], dtype=int8)
    



In that case, and are views of . As you can see, **dtype views** are not
as useful as **slice views**, but can be handy in some cases (for
example, for quickly looking at the bytes of a generic array).

FAQ
---

I think I understand what a view is, but why fancy indexing is not returning a view?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

One might be tempted to think that doing fancy indexing may lead to
sliced views. But this is not true:



.. code-block:: python

    >>> a = numpy.arange(10)
    >>> c1 = a[[1,3]]
    >>> c2 = a[[3,1,1]]
    >>> a[:] = 100
    >>> c1
    array([1, 3])
    >>> c2
    array([3, 1, 1])
    



The reason why a fancy indexing is not returning a view is that, in
general, it cannot be expressed as a **slice** (in the sense stated
above of being able to be addressed with offsets, strides, and counts).

For example, fancy indexing for could have been expressed by , but it is
not possible to do the same for by means of a slice. So, this is why an
object with a *copy* of the original data is returned instead.

But fancy indexing does seem to return views sometimes, doesn't it?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Many users get fooled and think that fancy indexing returns views
instead of copies when they use this idiom:



.. code-block:: python

    >>> a = numpy.arange(10)
    >>> a[[1,2]] = 100
    >>> a
    array([  0, 100, 100,   3,   4,   5,   6,   7,   8,   9])
    



So, it seems that a\ 1,2 is actually a *view* because elements 1 and 2
have been updated. However, if we try this step by step, it won't work:



.. code-block:: python

    >>> a = numpy.arange(10)
    >>> c1 = a[[1,2]]
    >>> c1[:] = 100
    >>> a
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> c1
    array([100, 100])
    



What happens here is that, in the first idiom (), the python interpreter
translate it to:



.. code-block:: python

    a.__setitem__([1,2], 100)
    



i.e. there is not need to create neither a view or a copy because the
method can be evaluated *inplace* (i.e. no new object creation is
involved).

However, the second idiom () is translated to:



.. code-block:: python

    c1 = a.__getitem__([1,2])
    c1.__setitem__(slice(None, None, None), 100)  # [:] translates into slice(None, 
    None, None)
    



i.e. a new object with a **copy** (remember, fancy indexing does not
return views) of some elements of is created and returned prior to call
.

The rule of thumb here can be: in the context of **lvalue indexing**
(i.e. the indices are placed in the left hand side value of an
assignment), no view or copy of the array is created (because there is
no need to). However, with regular values, the above rules for creating
views does apply.

A final exercise
----------------

Finally, let's put a somewhat advanced problem. The next snippet works
as expected:



.. code-block:: python

    >>> a = numpy.arange(12).reshape(3,4)
    >>> ifancy = [0,2]
    >>> islice = slice(0,3,2)
    >>> a[islice, :][:, ifancy] = 100
    >>> a
    array([[100,   1, 100,   3],
           [  4,   5,   6,   7],
           [100,   9, 100,  11]])
    



but the next one does not:



.. code-block:: python

    >>> a = numpy.arange(12).reshape(3,4)
    >>> ifancy = [0,2]
    >>> islice = slice(0,3,2)
    >>> a[ifancy, :][:, islice] = 100  # note that ifancy and islice are interchange
    d here
    >>> a
    array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11]])
    



Would the reader discover why is the difference? *Hint: think in terms
of the sequence of and calls and what they do on each example.*

--------------

CategoryCookbook

