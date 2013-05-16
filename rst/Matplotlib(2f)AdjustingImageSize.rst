This is a small demo file that helps teach how to adjust figure sizes
for matplotlib

First a little introduction
===========================

There are three parameters define an image size (this is not MPL specific):
---------------------------------------------------------------------------

| ``* Size in length units (inches, cm, pt, etc): e.g. 5"x7"``
| ``* Size in pixels: e.g. 800x600 pixels``
| ``* Dots per inch (dpi) e.g. 100 dpi``
``Only two of these are independent, so if you define two of them, the third can be calculated from the others.``

When displaying on a computer screen (or saved to a PNG), the size in
length units is irrelevant, the pixels are simply displayed. When
printed, or saved to PS, EPS or PDF (all designed to support printing),
then the Size or dpi is used to determine how to scale the image.

Now I'm getting into how MPL works
----------------------------------

| ``. 1) The size of a figure is defined in length units (inches), and can be set by ``\ 
| ``. 2) The layout of the figure is defined in 'figure units' so that as the figure size is changed, the layout (eg axes positions) will update.``
| ``. 3) Size of text, width of lines, etc is defined in terms of length units (points?).``
| ``. 4) When displaying to the screen, or creating an image (PNG) the pixel size of text and line widths, etc is  determined by the dpi setting, which is set by ``\ 
``The trick here is that when printing, it's natural to think in terms of inches, but when creating an image (for a web page, for instance), it is natural to think in terms of pixel size. However, as of 0.84, pixel size can only be set directly in the GTK* back-ends, with the canvas.resize(w,h) method. (remember that you can only set two of the three size parameters, the third must be calculated from the other two).``

Another trick
=============

Figure.savefig() overrides the dpi setting in figure, and uses a default
(which on my system at least is 100 dpi). If you want to overide it, you
can specify the 'dpi' in the savefig call:

The following code will hopefully make this more clear, at least for
generating PNGs for web pages and the like.

.. image:: Matplotlib(2f)AdjustingImageSize_attachments/MPL_size_test.py



.. code-block:: python

    #!python
    
    """
    This is a small demo file that helps teach how to adjust figure sizes
    for matplotlib
    
    """
    
    import matplotlib
    print "using MPL version:", matplotlib.__version__
    matplotlib.use("WXAgg") # do this before pylab so you don'tget the default back 
    end.
    
    import pylab
    import matplotlib.numerix as N
    
    # Generate and plot some simple data:
    x = N.arange(0, 2*N.pi, 0.1)
    y = N.sin(x)
    
    pylab.plot(x,y)
    F = pylab.gcf()
    
    # Now check everything with the defaults:
    DPI = F.get_dpi()
    print "DPI:", DPI
    DefaultSize = F.get_size_inches()
    print "Default size in Inches", DefaultSize
    print "Which should result in a %i x %i Image"%(DPI*DefaultSize[0], DPI*DefaultS
    ize[1])
    # the default is 100dpi for savefig:
    F.savefig("test1.png")
    # this gives me a 797 x 566 pixel image, which is about 100 DPI
    
    # Now make the image twice as big, while keeping the fonts and all the
    # same size
    F.set_figsize_inches( (DefaultSize[0]*2, DefaultSize[1]*2) )
    Size = F.get_size_inches()
    print "Size in Inches", Size
    F.savefig("test2.png")
    # this results in a 1595x1132 image
    
    # Now make the image twice as big, making all the fonts and lines
    # bigger too.
    
    F.set_figsize_inches( DefaultSize )# resetthe size
    Size = F.get_size_inches()
    print "Size in Inches", Size
    F.savefig("test3.png", dpi = (200)) # change the dpi
    # this also results in a 1595x1132 image, but the fonts are larger.
    



Putting more than one image in a figure
=======================================

Suppose you have two images: 100x100 and 100x50 that you want to display
in a figure with a buffer of 20 pixels (relative to image pixels)
between them and a border of 10 pixels all around.

The solution isn't particularly object oriented, but at least it gets to
the practical details.



.. code-block:: python

    #!python
    def _calcsize(matrix1, matrix2, top=10, left=10, right=10, bottom=10, buffer=20,
     height=4, scale = 1.):
       size1 = array(matrix1.shape) * scale
       size2 = array(matrix2.shape) * scale
       _width = float(size1[1] + size2[1] + left + right + buffer)
       _height = float(max(size1[0], size2[0]) + top + bottom)
       x1 = left / _width
       y1 = bottom / _height
       dx1 = size1[1] / _width
       dy1 = size1[0] / _height
       size1 = (x1, y1, dx1, dy1)
       x2 = (size1[1] + left + buffer) / _width
       y2 = bottom / _height
       dx2 = size2[1] / _width
       dy2 = size2[0] / _height
       size2 = (x2, y2, dx2, dy2)
       figure = pylab.figure(figsize=(_width * height / _height, height))
       axis1 = apply(pylab.axes, size1)
       pylab.imshow(X1, aspect='preserve')
       axis2 = apply(pylab.axes, size2)
       pylab.imshow(X2, aspect='preserve')
       return axes1, axes2, figure
    



--------------

``. CategoryCookbookMatplotlib``

