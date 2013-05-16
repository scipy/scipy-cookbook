Easy multithreading
===================

Python includes a multithreading package, "threading", but python's
multithreading is seriously limited by the Global Interpreter Lock,
which allows only one thread to be interacting with the interpreter at a
time. For purely interpreted code, this makes multithreading effectively
cooperative and unable to take advantage of multiple cores.

However, numpy code often releases the GIL while it is calculating, so
that simple parallelism can speed up the code. For sophisticated
applications, one should look into MPI or using threading directly, but
surprisingly often one's application is "embarrassingly parallel", that
is, one simply has to do the same operation to many objects, with no
interaction between iterations. This kind of calculation can be easily
parallelized:



.. code-block:: python

    dft = parallel_map(lambda f: sum(exp(2.j*pi*f*times)), frequencies)
    



The code implementing parallel\_map is not too complicated, and is
attached to this entry. Even simpler, if one doesn't want to return
values:



.. code-block:: python

    def compute(n):
        ...do something...
    foreach(compute, range(100))
    



This replaces a for loop.

See attachments for code (written by AMArchibald). AttachList

See also ParallelProgramming for alternatives and more discussion.

--------------

``CategoryCookbook``

