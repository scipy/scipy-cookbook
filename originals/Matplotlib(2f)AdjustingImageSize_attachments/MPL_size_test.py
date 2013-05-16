#!/usr/bin/env python2.4

"""
This is a small demo file that helps teach how to adjust figure sizes
for matplotlib

First a little introduction:

There are three parameters define an image size (this is not MPL specific):

Size in length units (inches, cm, pt, etc): i.e. 5"x7"
Size in pixels: i.e. 800x600 pixels
Dots per inch (dpi) i.e. 100 dpi

Only two of these are independent, so if you define two of them, the
third can be calculated from the others.

When displaying on a computer screen (or saved to a PNG), the size in
length units is irrelevant, the pixels are simply displayed. When
printed, or saved to PS, EPS or PDF (all designed to support printing),
then the Size or dpi is used to determine how to scale the image.

Now I'm getting into how MPL works:

1) The size of a figure is defined in length units (inches), and can be
set by:

Figure.set_figsize_inches( (w,h) )

2) The layout of the figure is defined in 'figure units' so that as the
figure size is changed, the layout (eg axes positions) will update.

3) Size of text, width of lines, etc is defined in terms of length units
(points?).

4) When displaying to the screen, or creating an image (PNG) the pixel
size of text and line widths, etc is determined by the dpi setting,
which is set by:

Figure.set_dpi( val )

The trick here is that when printing, it's natural to think in terms of
inches, but when creating an image (for a web page, for instance), it is
natural to think in terms of pixel size. However, as of 0.84, pixel size
can only be set directly in the GTK* back-ends, with the
canvas.resize(w,h) method. (remember that you can only set two of the
three size parameters, the third must be calculated from the other
two).

Another trick:

Figure.savefig() overrides the ppi setting in figure, and uses a default
(which on my system at least is 100ppi). I you want to overide it, you
can specify the ppi in the savefig call:

Figure.savefig(filename, ppi=value)

The following code will hopefully make this more clear, at least for
generating PNGs for web pages and the like.

"""

import matplotlib
print "using MPL version:", matplotlib.__version__
matplotlib.use("WXAgg") # do this before pylab so you don'tget the default back end.

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
print "Which should result in a %i x %i Image"%(DPI*DefaultSize[0], DPI*DefaultSize[1])
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
