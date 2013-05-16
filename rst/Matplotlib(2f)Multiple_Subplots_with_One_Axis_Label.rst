Using a single axis label to annotate multiple subplot axes
===========================================================

When using multiple subplots with the same axis units, it is redundant
to label each axis individually, and makes the graph overly complex. You
can use a single axis label, centered in the plot frame, to label
multiple subplot axes. Here is how to do it:



.. code-block:: python

    #!python
    # note that this a code fragment...you will have to define your own data to plot
    #
    # Set up a whole-figure axes, with invisible axis, ticks, and ticklabels,
    # which we use to get the xlabel and ylabel in the right place
    bigAxes = pylab.axes(frameon=False)     # hide frame
    pylab.xticks([])                        # don't want to see any ticks on this ax
    is
    pylab.yticks([])
    # I'm using TeX for typesetting the labels--not necessary
    pylab.ylabel(r'\textbf{Surface Concentration $(nmol/m^2)$}', size='medium')
    pylab.xlabel(r'\textbf{Time (hours)}', size='medium')
    # Create subplots and shift them up and to the right to keep tick labels
    # from overlapping the axis labels defined above
    topSubplot = pylab.subplot(2,1,1)
    position = topSubplot.get_position()
    position[0] = 0.15
    position[1] = position[1] + 0.01
    topSubplot.set_position(position)
    pylab.errorbar(times150, average150)
    bottomSubplot = pylab.subplot(2,1,2)
    position = bottomSubplot.get_position()
    position[0] = 0.15
    position[1] = position[1] + 0.03
    bottomSubplot.set_position(position)
    pylab.errorbar(times300, average300)
    



#. 

   #. 

Alternatively, you can use the following snippet to have shared ylabels
on your subplots. Also see the attached `figure
output <.. image:: Matplotlib(2f)Multiple_Subplots_with_One_Axis_Label_attachments/Same_ylabel_subplots.png>`__.



.. code-block:: python

    #!python
    import pylab
    
    figprops = dict(figsize=(8., 8. / 1.618), dpi=128)                              
                # Figure properties
    adjustprops = dict(left=0.1, bottom=0.1, right=0.97, top=0.93, wspace=0.2 hspace
    =0.2)       # Subplot properties
    
    fig = pylab.figure(**figprops)                                                  
                # New figure
    fig.subplots_adjust(**adjustprops)                                              
                # Tunes the subplot layout
    
    ax = fig.add_subplot(3, 1, 1)
    bx = fig.add_subplot(3, 1, 2, sharex=ax, sharey=ax)
    cx = fig.add_subplot(3, 1, 3, sharex=ax, sharey=ax)
    
    ax.plot([0,1,2], [2,3,4], 'k-')
    bx.plot([0,1,2], [2,3,4], 'k-')
    cx.plot([0,1,2], [2,3,4], 'k-')
    
    pylab.setp(ax.get_xticklabels(), visible=False)
    pylab.setp(bx.get_xticklabels(), visible=False)
    
    bx.set_ylabel('This is a long label shared among more axes', fontsize=14)
    cx.set_xlabel('And a shared x label', fontsize=14)
    



Thanks to Sebastian Krieger from matplotlib-users list for this trick.

#. 

   #. 

#. 

   #. 

Simple function to get rid of superfluous xticks but retain the ones on
the bottom (works in pylab). Combine it with the above snippets to get a
nice plot without too much redundance:



.. code-block:: python

    #!python
    def rem_x():
        '''Removes superfluous x ticks when multiple subplots  share
        their axis works only in pylab mode but can easily be rewritten
        for api use'''
        nr_ax=len(gcf().get_axes())
        count=0
        for z in gcf().get_axes():
            if count == nr_ax-1: break
                setp(z.get_xticklabels(),visible=False)
                count+=1
    



#. 

   #. 

The first one above doesn't work for me. The subplot command overwrites
the bigaxes. However, I found a much simpler solution to do a decent job
for two axes and one ylabel:

yyl=plt.ylabel(r'My longish label that I want vertically centred')

yyl.set\_position((yyl.get\_position()[0],1)) # This says use the top of
the bottom axis as the reference point.

yyl.set\_verticalalignment('center')

--------------

CategoryCookbookMatplotlib

