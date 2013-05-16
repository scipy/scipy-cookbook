import matplotlib
import numarray as na

def hist(binsIn, dataIn, normed=False):
    """
    Make a histogram that can be plotted with plot() so that
    the histogram just has the outline rather than bars as it
    usually does.

    Example Usage:
    binsIn = numarray.arange(0, 1, 0.1)
    angle = pylab.rand(50)

    (bins, data) = histOutline(binsIn, angle)
    plot(bins, data, 'k-', linewidth=2)

    """
    (en, eb) = matplotlib.mlab.hist(dataIn, bins=binsIn, normed=normed)

    stepSize = binsIn[1] - binsIn[0]

    bins = na.zeros(len(eb)*2 + 2, type=na.Float)
    data = na.zeros(len(eb)*2 + 2, type=na.Float)    
    for bb in range(len(binsIn)):
        bins[2*bb + 1] = binsIn[bb]
        bins[2*bb + 2] = binsIn[bb] + stepSize
        data[2*bb + 1] = en[bb]
        data[2*bb + 2] = en[bb]

    bins[0] = bins[1]
    bins[-1] = bins[-2]
    data[0] = 0
    data[-1] = 0
    
    return (bins, data)

def convertForPlot(binsIn, histIn):
    """
    Take the output from a normal histogram and turn it into
    a histogram that can be plotted (square tops, etc.).

    binsIn - The bins output by matplotlib.mlab.hist()
    histIn - The histogram output by matplotlib.mlab.hist()
    """
    stepSize = binsIn[1] - binsIn[0]

    bins = na.zeros(len(binsIn)*2 + 2, type=na.Float)
    data = na.zeros(len(binsIn)*2 + 2, type=na.Float)    
    for bb in range(len(binsIn)):
        bins[2*bb + 1] = binsIn[bb]
        bins[2*bb + 2] = binsIn[bb] + stepSize
        data[2*bb + 1] = histIn[bb]
        data[2*bb + 2] = histIn[bb]

    bins[0] = bins[1]
    bins[-1] = bins[-2]
    data[0] = 0
    data[-1] = 0
    
    return (bins, data)
