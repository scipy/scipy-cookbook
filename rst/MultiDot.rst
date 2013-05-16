The matrix multiplication function, numpy.dot(), only takes two
arguments. That means to multiply more than two arrays together you end
up with nested function calls which are hard to read:



.. code-block:: python

      dot(dot(dot(a,b),c),d)
    



versus infix notation where you'd just be able to write



.. code-block:: python

      a*b*c*d
    



There are a couple of ways to define an 'mdot' function that acts like
dot but accepts more than two arguments. Using one of these allows you
to write the above expression as



.. code-block:: python

    mdot(a,b,c,d)
    



Using reduce
------------

The simplest way it to just use reduce.



.. code-block:: python

    def mdot(*args):
        return reduce(numpy.dot, args)
    



Or use the equivalent loop (which is apparently the preferred style `for
Py3K <http://www.python.org/dev/peps/pep-3100/#id53>`__):



.. code-block:: python

    def mdot(*args):
        ret = args[0]
        for a in args[1:]:
            ret = dot(ret,a)
        return ret
    



This will always give you left to right associativity, i.e. the
expression is interpreted as \`(((a\*b)\*c)\*d)\`.

You also can make a right-associative version of the loop:



.. code-block:: python

    def mdotr(*args):
        ret = args[-1]
        for a in reversed(args[:-1]):
            ret = dot(a,ret)
        return ret
    



which evaluates as \`(a\*(b\*(c\*d)))\`. But sometimes you'd like to
have finer control since the order in which matrix multiplies are
performed can have a big impact on performance. The next version gives
that control.

Controlling order of evaluation
-------------------------------

If we're willing to sacrifice Numpy's ability to treat tuples as arrays,
we can use tuples as grouping constructs. This version of \`mdot\`
allows syntax like this:



.. code-block:: python

       mdot(a,((b,c),d))
    



to control the order in which the pairwise \`dot\` calls are made.



.. code-block:: python

    import types
    import numpy
    def mdot(*args):
       """Multiply all the arguments using matrix product rules.
       The output is equivalent to multiplying the arguments one by one
       from left to right using dot().
       Precedence can be controlled by creating tuples of arguments,
       for instance mdot(a,((b,c),d)) multiplies a (a*((b*c)*d)).
       Note that this means the output of dot(a,b) and mdot(a,b) will differ if
       a or b is a pure tuple of numbers.
       """
       if len(args)==1:
           return args[0]
       elif len(args)==2:
           return _mdot_r(args[0],args[1])
       else:
           return _mdot_r(args[:-1],args[-1])
    
    def _mdot_r(a,b):
       """Recursive helper for mdot"""
       if type(a)==types.TupleType:
           if len(a)>1:
               a = mdot(*a)
           else:
               a = a[0]
       if type(b)==types.TupleType:
           if len(b)>1:
               b = mdot(*b)
           else:
               b = b[0]
       return numpy.dot(a,b)
    



Multiply
--------

Note that the elementwise multiplication function \`numpy.multiply\` has
the same two-argument limitation as \`numpy.dot\`. The exact same
generalized forms can be defined for multiply.

Left associative versions:



.. code-block:: python

    def mmultiply(*args):
        return reduce(numpy.multiply, args)
    







.. code-block:: python

    def mmultiply(*args):
        ret = args[0]
        for a in args[1:]:
            ret = multiply(ret,a)
        return ret
    



Right-associative version:



.. code-block:: python

    def mmultiplyr(*args):
        ret = args[-1]
        for a in reversed(args[:-1]):
            ret = multiply(a,ret)
        return ret
    



Version using tuples to control order of evaluation:



.. code-block:: python

    import types
    import numpy
    def mmultiply(*args):
       """Multiply all the arguments using elementwise product.
       The output is equivalent to multiplying the arguments one by one
       from left to right using multiply().
       Precedence can be controlled by creating tuples of arguments,
       for instance mmultiply(a,((b,c),d)) multiplies a (a*((b*c)*d)).
       Note that this means the output of multiply(a,b) and mmultiply(a,b) will diff
    er if
       a or b is a pure tuple of numbers.
       """
       if len(args)==1:
           return args[0]
       elif len(args)==2:
           return _mmultiply_r(args[0],args[1])
       else:
           return _mmultiply_r(args[:-1],args[-1])
    
    def _mmultiply_r(a,b):
       """Recursive helper for mmultiply"""
       if type(a)==types.TupleType:
           if len(a)>1:
               a = mmultiply(*a)
           else:
               a = a[0]
       if type(b)==types.TupleType:
           if len(b)>1:
               b = mmultiply(*b)
           else:
               b = b[0]
       return numpy.multiply(a,b)
    





