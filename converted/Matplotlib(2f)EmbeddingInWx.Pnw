Matplotlib can be embedded in wxPython applications to provide high
quality data visualization. There are two approaches to this, direct
embedding and using an embedding library.

*Direct embedding* is where your put one of the wxPython backend widgets
(which subclass ) directly into your application and draw plots on it
using matplotlib's object-oriented API. This approach is demonstrated by
the
`embedding\_in\_wx\*.py <http://cvs.sourceforge.net/viewcvs.py/matplotlib/matplotlib/examples/>`__
examples that come with matplotlib. Neither nor provide any facilities
for user interactions like displaying the coordinates under the mouse,
so you'll have to implement such things yourself. The matplotlib example
`wxcursor\_demo.py <http://cvs.sourceforge.net/viewcvs.py/%2Acheckout%2A/matplotlib/matplotlib/examples/wxcursor_demo.py?content-type=text%2Fplain>`__
should help you get started.

An *embedding library* saves you a lot of time and effort by providing
plotting widgets that already support user interactions and other bells
and whistles. There are two such libraries that I am aware of:

``1. Matt Newville's ``\ ```MPlot`` <http://cars9.uchicago.edu/~newville/Python/MPlot/>`__\ `` package supports drawing 2D line plots using pylab-style ``\ \ `` and ``\ \ `` methods.``

``2. Ken !McIvor's ``\ ```WxMpl`` <http://agni.phys.iit.edu/~kmcivor/wxmpl/>`__\ `` module supports drawing all plot types using matplotlib's object-oriented API.``

Each of these libraries has different benefits and drawbacks, so I
encourage you to evaluate each of them and select the one that best
meets your needs.

Learning the Object-Oriented API
================================

If you're embedding matplotlib in a wxPython program, you're probably
going to have to use Matplotlib's Object-Oriented API to at some point.
Take heart, as it matches the pylab API closely and is easy to pick up.
There are more nuts and bolts to deal with, but that's no problem to
someone already programming with wxPython! ;-)

The matplotlib FAQ [http://matplotlib.sourceforge.net/faq.html#OO"
links] to several resources for learning about the OO API. Once you've
got your feet wet, reading the classdocs is the most helpful source of
information. The
[http://matplotlib.sourceforge.net/matplotlib.axes.html#Axes"
matplotlib.axes.Axes] class is where most of the plotting methods live,
so it's a good place to start after you've conquored creating a Figure.

For your edification, a series of pylab examples have been translated to
the OO API. They are available in a demonstration script that must be
run from a command line. You may use any interactive matplotlib backend
to display these plots.

A Simple Application
====================

Here is a simple example of an application written in wx that embeds a
["Matplotlib figure in a wx panel"]. No toolbars, mouse clicks or any of
that, just a plot drawn in a panel. Some work has been put into it to
make sure that the figure is only redrawn once during a resize. For
plots with many points, the redrawing can take some time, so it is best
to only redraw when the figure is released. Have a read of the code.

--------------

CategoryCookbookMatplotlib

