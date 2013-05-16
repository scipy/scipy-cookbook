Each axes instance contains a lines attribute, which is a list of the
data series in the plot, added in chronological order. To delete a
particular data series, one must simply delete the appropriate element
of the lines list and redraw if necessary.

The is illustrated in the following example from an interactive session:



.. code-block:: python

    >>> x = N.arange(10)
    
    >>> fig = P.figure()
    >>> ax = fig.add_subplot(111)
    >>> ax.plot(x)
    [<matplotlib.lines.Line2D instance at 0x427ce7ec>]
    
    >>> ax.plot(x+10)
    [<matplotlib.lines.Line2D instance at 0x427ce88c>]
    
    >>> ax.plot(x+20)
    [<matplotlib.lines.Line2D instance at 0x427ce9ac>]
    
    >>> P.show()
    >>> ax.lines
    [<matplotlib.lines.Line2D instance at 0x427ce7ec>,
     <matplotlib.lines.Line2D instance at 0x427ce88c>,
     <matplotlib.lines.Line2D instance at 0x427ce9ac>]
    
    >>> del ax.lines[1]
    >>> P.show()
    



which will plot three lines, and then delete the second.

--------------

CategoryCookbookMatplotlib

