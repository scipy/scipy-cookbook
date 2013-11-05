# code from http://www.scipy.org/Cookbook/Matplotlib/UnfilledHistograms?action=AttachFile&do=get&target=histNofill.py
from matplotlib import pylab
import numpy as np



def histOutline(dataIn, *args, **kwargs):
    """
    Make a histogram that can be plotted with plot() so that
    the histogram just has the outline rather than bars as it
    usually does.

    Example Usage:
    binsIn = numpy.arange(0, 1, 0.1)
    angle = pylab.rand(50)

    (bins, data) = histOutline(binsIn, angle)
    plot(bins, data, 'k-', linewidth=2)

    """

    (histIn, binsIn) = np.histogram(dataIn, *args, **kwargs)

    stepSize = binsIn[1] - binsIn[0]

    bins = np.zeros(len(binsIn)*2 + 2, dtype=np.float)
    data = np.zeros(len(binsIn)*2 + 2, dtype=np.float)    
    for bb in range(len(binsIn)):
        bins[2*bb + 1] = binsIn[bb]
        bins[2*bb + 2] = binsIn[bb] + stepSize
        if bb < len(histIn):
            data[2*bb + 1] = histIn[bb]
            data[2*bb + 2] = histIn[bb]

    bins[0] = bins[1]
    bins[-1] = bins[-2]
    data[0] = 0
    data[-1] = 0
    
    return (bins, data)



if __name__ == "__main__":
    binsIn = np.arange(0, 1, 0.1)
    angle = pylab.rand(50)

    pylab.subplot(121)
    pylab.hist(angle,binsIn)
    pylab.title("regular histogram")
    pylab.axis(xmax=1.0)

    pylab.subplot(122)

    (bins, data) = histOutline(angle, binsIn)
    pylab.plot(bins, data, 'k-', linewidth=2)
    pylab.title("histOutline Demo")
    pylab.show()


