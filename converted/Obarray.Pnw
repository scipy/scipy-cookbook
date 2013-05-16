Object arrays using record arrays
=================================

numpy supports working with arrays of python objects, but these arrays
lack the type-uniformity of normal numpy arrays, so they can be quite
inefficient in terms of space and time, and they can be quite cumbersome
to work with. However, it would often be useful to be able to store a
user-defined class in an array.

One approach is to take advantage of numpy's record arrays. These are
arrays in which each element can be large, as it has named and typed
fields; essentially they are numpy's equivalent to arrays of C
structures. Thus if one had a class consisting of some data - named
fields, each of a numpy type - and some methods, one could represent the
data for an array of these objects as a record array. Getting the
methods is more tricky.

One approach is to create a custom subclass of the numpy array which
handles conversion to and from your object type. The idea is to store
the data for each instance internally in a record array, but when
indexing returns a scalar, construct a new instance from the data in the
records. Similarly, when assigning to a particular element, the array
subclass would convert an instance to its representation as a record.

Attached is an implementation of the above scheme.

