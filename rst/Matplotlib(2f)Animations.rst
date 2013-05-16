***Note:*** Some of the matplotlib code in this cookbook entry may be
deprecated or obsolete. For example, the file **anim.py** mentioned
below no longer exists in matplotlib. Examples of animation in
matplotlib are at
http://matplotlib.sourceforge.net/examples/animation/index.html, and the
main animation API documentation is
http://matplotlib.sourceforge.net/api/animation_api.html.

--------------

matplotlib supports animated plots, and provides a number of demos. An
important question when considering whether to use matplotlib for
animation is what kind of speed you need. matplotlib is not the fastest
plotting library in the west, and may be too slow for some animation
applications. Nonetheless, it is fast enough for many if not most, and
this tutorial is geared to showing you how to make it fast enough for
you. In particular, the section \*Animating selected plot elements\*
shows you how to get top speed out of matplotlib animations.

TableOfContents

Performance
===========

matplotlib supports 5 different graphical user interfaces (GTK, WX, Qt,
Tkinter, FLTK) and for some of those GUIs, there are various ways to
draw to the canvas. For example, for GTK, you can use native `GDK
drawing <http://www.pygtk.org/pygtk2reference/class-gdkdrawable.html>`__,
`antigrain <http://antigrain.com>`__, or
`cairo <http://cairographics.org/>`__. A GUI toolkit combined with some
method of drawing comprises a
`backend <http://matplotlib.sourceforge.net/backends.html>`__. For
example, drawing to a GTK canvas with the antigrain drawing toolkit is
called the GTKAgg backend. This is important because different backends
have different performance characteristics, and the difference can be
considerable.

When considering performance, the typical measure is frames per second.
Television is 30 frames per second, and for many application if you can
get 10 or more frames per second the animation is smooth enough to "look
good". Monitors refresh at 75-80 frames per second typically, and so
this is an upper limit for performance. Any faster is probably wasted
CPU cycles.

Here are some numbers for the animated script
`anim.py <http://matplotlib.sf.net/examples/anim.py>`__, which simply
updates a sine wave on various backends, run under linux on a 3GHz
Pentium IV. To profile a script under different backends, you can use
the "GUI neutral" animation technique described below and then run it
with the flag, as in:



.. code-block:: python

    > python anim.py -dWX
    > python anim.py -dGTKAgg
    



Here are the results. Note that these should be interpreted cautiously,
because some GUIs might call a drawing operation in a separate thread
and return before it is complete, or drop a drawing operation while one
is in the queue. The most important assessment is qualitative.



.. code-block:: python

    Backend  Frames/second
     GTK         43   
     GTKAgg      36 
     TkAgg       20 
     WX          11
     WXAgg       27
    



GUI neutral animation in pylab
==============================

The pylab interface supports animation that does not depend on a
specific GUI toolkit. This is not recommended for production use, but is
often a good way to make a quick-and-dirty, throw away animation. After
importing pylab, you need to turn interaction on with the
[http://matplotlib.sf.net/matplotlib.pylab.html\ #-ion ion] command. You
can then force a draw at any time with
[http://matplotlib.sf.net/matplotlib.pylab.html\ #-draw draw]. In
interactive mode, a new draw command is issued after each pylab command,
and you can also temporarily turn off this behavior for a block of
plotting commands in which you do not want an update with the
[http://matplotlib.sf.net/matplotlib.pylab.html\ #-ioff ioff] commands.
This is described in more detail on the
`interactive <http://matplotlib.sf.net/interactive.html>`__ page.

Here is the anim.py script that was used to generate the profiling
numbers across backends in the table above



.. code-block:: python

    from pylab import *
    import time
    
    ion()  
    
    tstart = time.time()               # for profiling
    x = arange(0,2*pi,0.01)            # x-array
    line, = plot(x,sin(x))
    for i in arange(1,200):
        line.set_ydata(sin(x+i/10.0))  # update the data
        draw()                         # redraw the canvas
    
    print 'FPS:' , 200/(time.time()-tstart)
    



Note the technique of creating a line with the call to
[http://matplotlib.sf.net/matplotlib.pylab.html\ #-plot plot]:



.. code-block:: python

    line, = plot(x,sin(x))
    



and then setting its data with the set\_ydata method and calling draw:



.. code-block:: python

    line.set_ydata(sin(x+i/10.0))  # update the data
    draw()                         # redraw the canvas
    



This can be much faster than clearing the axes and/or creating new
objects for each plot command.

To animate a pcolor plot use:



.. code-block:: python

    p = pcolor(XI,YI,C[0])
    
    for i in range(1,len(C)):
      p.set_array(C[i,0:-1,0:-1].ravel())
      p.autoscale()
      draw()
    



This assumes C is a 3D array where the first dimension is time and that
XI,YI and C[i,:,:] have the same shape. If C[i,:,:] is one row and one
column smaller simply use C.ravel().

Using the GUI timers or idle handlers
=====================================

If you are doing production code or anything semi-serious, you are
advised to use the GUI event handling specific to your toolkit for
animation, because this will give you more control over your animation
than matplotlib can provide through the GUI neutral pylab interface to
animation. How you do this depends on your toolkit, but there are
examples for several of the backends in the matplotlib examples
directory, eg,
`anim\_tk.py <http://matplotlib.sf.net/examples/anim_tk.py>`__ for
Tkinter,
`dynamic\_image\_gtkagg.py <http://matplotlib.sf.net/examples/dynamic_image_gtkagg.py>`__
for GTKAgg and
`dynamic\_image\_wxagg.py <http://matplotlib.sf.net/examples/dynamic_image_wxagg.py>`__
for WXAgg.

The basic idea is to create your figure and a callback function that
updates your figure. You then pass that callback to the GUI idle handler
or timer. A simple example in GTK looks like



.. code-block:: python

    def callback(*args):  
       line.set_ydata(get_next_plot())
       canvas.draw()  # or simply "draw" in pylab
    
    gtk.idle_add(callback)
    



A simple example in WX or WXAgg looks like



.. code-block:: python

    def callback(*args):  
       line.set_ydata(get_next_plot())
       canvas.draw()
       wx.WakeUpIdle() # ensure that the idle event keeps firing
    
    wx.EVT_IDLE(wx.GetApp(), callback)
    



Animating selected plot elements
================================

One limitation of the methods presented above is that all figure
elements are redrawn with every call to draw, but we are only updating a
single element. Often what we want to do is draw a background, and
animate just one or two elements on top of it. As of matplotlib-0.87,
GTKAgg, !TkAgg, WXAgg, and FLTKAgg support the methods discussed here.

The basic idea is to set the 'animated' property of the Artist you want
to animate (all figure elements from Figure to Axes to Line2D to Text
derive from the base class
`Artist <http://matplotlib.sf.net/matplotlib.artist.html>`__). Then,
when the standard canvas draw operation is called, all the artists
except the animated one will be drawn. You can then use the method to
copy a rectangular region (eg the axes bounding box) into a a pixel
buffer. In animation, you restore the background with , and then call to
draw your animated artist onto the clean background, and to blit the
updated axes rectangle to the figure. When I run the example below in
the same environment that produced 36 FPS for GTKAgg above, I measure
327 FPS with the techniques below. See the caveats on performance
numbers mentioned above. Suffice it to say, quantitatively and
qualitiatively it is much faster.



.. code-block:: python

    import sys
    import gtk, gobject
    import pylab as p
    import matplotlib.numerix as nx
    import time
    
    ax = p.subplot(111)
    canvas = ax.figure.canvas
    
    # for profiling
    tstart = time.time()
    
    # create the initial line
    x = nx.arange(0,2*nx.pi,0.01)
    line, = p.plot(x, nx.sin(x), animated=True)
    
    # save the clean slate background -- everything but the animated line
    # is drawn and saved in the pixel buffer background
    background = canvas.copy_from_bbox(ax.bbox)
    
    def update_line(*args):
        # restore the clean slate background
        canvas.restore_region(background)
        # update the data
        line.set_ydata(nx.sin(x+update_line.cnt/10.0))  
        # just draw the animated artist
        ax.draw_artist(line)
        # just redraw the axes rectangle
        canvas.blit(ax.bbox) 
    
        if update_line.cnt==50:
            # print the timing info and quit
            print 'FPS:' , update_line.cnt/(time.time()-tstart)
            sys.exit()
    
        update_line.cnt += 1
        return True
    update_line.cnt = 0
    
    gobject.idle_add(update_line)
    p.show()
    



Example: cursoring
==================

matplotlib 0.83.2 introduced a cursor class which can utilize these blit
methods for no lag cursoring. The class takes a argument in the
constructor. For backends that support the new API (GTKAgg) set :



.. code-block:: python

    from matplotlib.widgets import Cursor
    import pylab
    
    fig = pylab.figure(figsize=(8,6))
    ax = fig.add_axes([0.075, 0.25, 0.9, 0.725], axisbg='#FFFFCC')
    
    x,y = 4*(pylab.rand(2,100)-.5)
    ax.plot(x,y,'o')
    ax.set_xlim(-2,2)
    ax.set_ylim(-2,2)
    
    # set useblit = True on gtkagg for enhanced performance    
    cursor = Cursor(ax, useblit=True, color='red', linewidth=2 )
    
    pylab.show()
    



The 'blit' animation methods
============================

As noted above, only the GTKAgg supports the methods above to to the
animations of selected actors. The following are needed

Figure canvas methods
---------------------

``* ``\ \ `` - copy the region in ax.bbox into a pixel buffer and return it in an object type of your choosing.  bbox is a matplotlib BBox instance from the ``\ ```transforms``
``module`` <http://matplotlib.sf.net/transforms.html>`__\ ``. ``\ \ `` is not used by the matplotlib frontend, but it stores it and passes it back to the backend in the ``\ \ `` method. You will probably want to store not only the pixel buffer but the rectangular region of the canvas from whence it came in the background object.``

``* ``\ \ `` - restore the region copied above to the canvas.  ``

``* ``\ \ `` - transfer the pixel buffer in region bounded by bbox to the canvas.``

For \*Agg backends, there is no need to implement the first two as Agg
will do all the work ( defines them). Thus you only need to be able to
blit the agg buffer from a selected rectangle. One thing that might make
this easier for backends using string methods to transfer the agg pixel
buffer to their respective canvas is to define a method in agg. If you
are working on this and need help, please contact the `matplotlib-devel
list <http://sourceforge.net/mailarchive/forum.php?forum_id=36187>`__.

Once all/most of the backends have implemented these methods, the
matplotlib front end can do all the work of managing the
background/restore/blit opertations, and userland animated code can look
like



.. code-block:: python

    line, = plot(something, animated=True)
    draw()   
    def callback(*args):
        line.set_ydata(somedata)
        ax.draw_animated()
    



and the rest will happen automagically. Since some backends **do not**
currently implement the required methods, I am making them available to
the users to manage themselves but am not assuming them in the axes
drawing code.

--------------

CategoryCookbookMatplotlib

