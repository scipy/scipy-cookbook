Example of how to thicken the lines around your plot (axes lines) and to
get big bold fonts on the tick and axis labels.



.. code-block:: python

    from pylab import *
    
    # Thicken the axes lines and labels
    # 
    #   Comment by J. R. Lu:
    #       I couldn't figure out a way to do this on the 
    #       individual plot and have it work with all backends
    #       and in interactive mode. So, used rc instead.
    # 
    rc('axes', linewidth=2)
    
    # Make a dummy plot
    plot([0, 1], [0, 1])
    
    # Change size and font of tick labels
    # Again, this doesn't work in interactive mode.
    fontsize = 14
    ax = gca()
    
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontweight('bold')
    
    xlabel('X Axis', fontsize=16, fontweight='bold')
    ylabel('Y Axis', fontsize=16, fontweight='bold')
    
    # Save figure
    savefig('thick_axes.png')
    



.. image:: Matplotlib(2f)ThickAxes_attachments/thick_axes.png

