TableOfContents

Multiple line plots
===================

Often one wants to plot many signals over one another. There are a few
ways to do this. The naive implementation is just to add a constant
offset to each signal:



.. code-block:: python

    
    from pylab import plot, show, ylim, yticks
    from matplotlib.numerix import sin, cos, exp, pi, arange
    
    t = arange(0.0, 2.0, 0.01)
    s1 = sin(2*pi*t)
    s2 = exp(-t)
    s3 = sin(2*pi*t)*exp(-t)
    s4 = sin(2*pi*t)*cos(4*pi*t)
    
    t = arange(0.0, 2.0, 0.01)
    plot(t, s1, t, s2+1, t, s3+2, t, s4+3, color='k')
    ylim(-1,4)
    yticks(arange(4), ['S1', 'S2', 'S3', 'S4']) 
    
    show()
    



but then it is difficult to do change the y scale in a reasonable way.
For example when you zoom in on y, the signals on top and bottom will go
off the screen. Often what one wants is for the y location of each
signal to remain in place and the gain of the signal to be changed.

Using multiple axes
===================

If you have just a few signals, you could make each signal a separate
axes and make the y label horizontal. This works fine for a small number
of signals (4-10 say) except the extra horizontal lines and ticks around
the axes may be annoying. It's on our list of things to change the way
these axes lines are draw so that you can remove it, but it isn't done
yet:



.. code-block:: python

    
    from pylab import figure, show, setp
    from matplotlib.numerix import sin, cos, exp, pi, arange
    
    t = arange(0.0, 2.0, 0.01)
    s1 = sin(2*pi*t)
    s2 = exp(-t)
    s3 = sin(2*pi*t)*exp(-t)
    s4 = sin(2*pi*t)*cos(4*pi*t)
    
    fig = figure()
    t = arange(0.0, 2.0, 0.01)
    
    yprops = dict(rotation=0,  
                  horizontalalignment='right',
                  verticalalignment='center',
                  x=-0.01)
    
    axprops = dict(yticks=[])
    
    ax1 =fig.add_axes([0.1, 0.7, 0.8, 0.2], **axprops)
    ax1.plot(t, s1)
    ax1.set_ylabel('S1', **yprops)
    
    axprops['sharex'] = ax1
    axprops['sharey'] = ax1
    # force x axes to remain in register, even with toolbar navigation
    ax2 = fig.add_axes([0.1, 0.5, 0.8, 0.2], **axprops)
    
    ax2.plot(t, s2)
    ax2.set_ylabel('S2', **yprops)
    
    ax3 = fig.add_axes([0.1, 0.3, 0.8, 0.2], **axprops)
    ax3.plot(t, s4)
    ax3.set_ylabel('S3', **yprops)
    
    ax4 = fig.add_axes([0.1, 0.1, 0.8, 0.2], **axprops)
    ax4.plot(t, s4)
    ax4.set_ylabel('S4', **yprops)
    
    # turn off x ticklabels for all but the lower axes
    for ax in ax1, ax2, ax3:
        setp(ax.get_xticklabels(), visible=False)
    
    show()
    



.. image:: Matplotlib(2f)MultilinePlots_attachments/multipleaxes.png

Manipulating transforms
=======================

For large numbers of lines the approach above is inefficient because
creating a separate axes for each line creates a lot of useless
overhead. The application that gave birth to matplotlib is an `EEG
viewer <http://matplotlib.sourceforge.net/screenshots/eeg_small.png>`__
which must efficiently handle hundreds of lines; this is is available as
part of the `pbrain package <http://pbrain.sf.net>`__.

Here is an example of how that application does multiline plotting with
"in place" gain changes. Note that this will break the y behavior of the
toolbar because we have changed all the default transforms. In my
application I have a custom toolbar to increase or decrease the y scale.
In this example, I bind the plus/minus keys to a function which
increases or decreases the y gain. Perhaps I will take this and wrap it
up into a function called plot\_signals or something like that because
the code is a bit hairy since it makes heavy use of the somewhat arcane
matplotlib transforms. I suggest reading up on the `transforms
module <http://matplotlib.sourceforge.net/matplotlib.transforms.html>`__
before trying to understand this example:



.. code-block:: python

    
    from pylab import figure, show, setp, connect, draw
    from matplotlib.numerix import sin, cos, exp, pi, arange
    from matplotlib.numerix.mlab import mean
    from matplotlib.transforms import Bbox, Value, Point, \
         get_bbox_transform, unit_bbox
    # load the data
    
    t = arange(0.0, 2.0, 0.01)
    s1 = sin(2*pi*t)
    s2 = exp(-t)
    s3 = sin(2*pi*t)*exp(-t)
    s4 = sin(2*pi*t)*cos(4*pi*t)
    s5 = s1*s2
    s6 = s1-s4
    s7 = s3*s4-s1
    
    signals = s1, s2, s3, s4, s5, s6, s7
    for sig in signals:
        sig = sig-mean(sig)
    
    lineprops = dict(linewidth=1, color='black', linestyle='-')
    fig = figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    
    # The normal matplotlib transformation is the view lim bounding box
    # (ax.viewLim) to the axes bounding box (ax.bbox).  Where are going to
    # define a new transform by defining a new input bounding box. See the
    # matplotlib.transforms module helkp for more information on
    # transforms
    
    # This bounding reuses the x data of the viewLim for the xscale -10 to
    # 10 on the y scale.  -10 to 10 means that a signal with a min/max
    # amplitude of 10 will span the entire vertical extent of the axes
    scale = 10
    boxin = Bbox(
        Point(ax.viewLim.ll().x(), Value(-scale)),
        Point(ax.viewLim.ur().x(), Value(scale)))
    
    
    # height is a lazy value
    height = ax.bbox.ur().y() - ax.bbox.ll().y()
    
    boxout = Bbox(
        Point(ax.bbox.ll().x(), Value(-0.5) * height),
        Point(ax.bbox.ur().x(), Value( 0.5) * height))
    
    
    # matplotlib transforms can accepts an offset, which is defined as a
    # point and another transform to map that point to display.  This
    # transform maps x as identity and maps the 0-1 y interval to the
    # vertical extent of the yaxis.  This will be used to offset the lines
    # and ticks vertically
    transOffset = get_bbox_transform(
        unit_bbox(),
        Bbox( Point( Value(0), ax.bbox.ll().y()),
              Point( Value(1), ax.bbox.ur().y())
              ))
    
    # now add the signals, set the transform, and set the offset of each
    # line
    ticklocs = []
    for i, s in enumerate(signals):
        trans = get_bbox_transform(boxin, boxout) 
        offset = (i+1.)/(len(signals)+1.)
        trans.set_offset( (0, offset), transOffset)
    
        ax.plot(t, s, transform=trans, **lineprops)
        ticklocs.append(offset)
    
    
    ax.set_yticks(ticklocs)
    ax.set_yticklabels(['S%d'%(i+1) for i in range(len(signals))])
    
    # place all the y tick attributes in axes coords  
    all = []
    labels = []
    ax.set_yticks(ticklocs)
    for tick in ax.yaxis.get_major_ticks():
        all.extend(( tick.label1, tick.label2, tick.tick1line,
                     tick.tick2line, tick.gridline))
        labels.append(tick.label1)
    
    setp(all, transform=ax.transAxes)
    setp(labels, x=-0.01)
    
    ax.set_xlabel('time (s)')
    
    
    # Because we have hacked the transforms, you need a special method to
    # set the voltage gain; this is a naive implementation of how you
    # might want to do this in real life (eg make the scale changes
    # exponential rather than linear) but it gives you the idea
    def set_ygain(direction):
        set_ygain.scale += direction
        if set_ygain.scale <=0:
            set_ygain.scale -= direction
            return
    
        for line in ax.lines:
            trans = line.get_transform()
            box1 =  trans.get_bbox1()
            box1.intervaly().set_bounds(-set_ygain.scale, set_ygain.scale)
        draw()
    set_ygain.scale = scale    
    
    def keypress(event):
        if event.key in ('+', '='): set_ygain(-1)
        elif event.key in ('-', '_'): set_ygain(1)
    
    connect('key_press_event', keypress)
    ax.set_title('Use + / - to change y gain')    
    show()
    



.. image:: Matplotlib(2f)MultilinePlots_attachments/multiline.png

--------------

CategoryCookbookMatplotlib

