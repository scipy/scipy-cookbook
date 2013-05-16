"""
cookb_signalsmooth.py

from: http://scipy.org/Cookbook/SignalSmooth
"""

import numpy as np
import matplotlib.pyplot as plt

def smooth(x, window_len=10, window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    import numpy as np    
    t = np.linspace(-2,2,0.1)
    x = np.sin(t)+np.random.randn(len(t))*0.1
    y = smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string   
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"

    s=np.r_[2*x[0]-x[window_len:1:-1], x, 2*x[-1]-x[-1:-window_len:-1]]
    #print(len(s))
    
    if window == 'flat': #moving average
        w = np.ones(window_len,'d')
    else:
        w = getattr(np, window)(window_len)
    y = np.convolve(w/w.sum(), s, mode='same')
    return y[window_len-1:-window_len+1]


#*********** part2: 2d

from scipy import signal

def gauss_kern(size, sizey=None):
    """ Returns a normalized 2D gauss kernel array for convolutions """
    size = int(size)
    if not sizey:
        sizey = size
    else:
        sizey = int(sizey)
    x, y = np.mgrid[-size:size+1, -sizey:sizey+1]
    g = np.exp(-(x**2/float(size) + y**2/float(sizey)))
    return g / g.sum()

def blur_image(im, n, ny=None) :
    """ blurs the image by convolving with a gaussian kernel of typical
        size n. The optional keyword argument ny allows for a different
        size in the y direction.
    """
    g = gauss_kern(n, sizey=ny)
    improc = signal.convolve(im, g, mode='valid')
    return(improc)


def smooth_demo():
    t = np.linspace(-4,4,100)
    x = np.sin(t)
    xn = x + np.random.randn(len(t)) * 0.1
    y = smooth(x)
    ws = 31

    plt.subplot(211)
    plt.plot(np.ones(ws))

    windows=['flat', 'hanning', 'hamming', 'bartlett', 'blackman']

    plt.hold(True)
    for w in windows[1:]:
        #eval('plt.plot('+w+'(ws) )')
        plt.plot(getattr(np, w)(ws))

    plt.axis([0,30,0,1.1])

    plt.legend(windows)
    plt.title("The smoothing windows")
    plt.subplot(212)
    plt.plot(x)
    plt.plot(xn)
    for w in windows:
        plt.plot(smooth(xn,10,w))
    l = ['original signal', 'signal with noise']
    l.extend(windows)
    plt.legend(l)
    plt.title("Smoothing a noisy signal")
    #plt.show()


if __name__=='__main__':
    
    # part 1: 1d
    smooth_demo()
    
    # part 2: 2d
    X, Y = np.mgrid[-70:70, -70:70]
    Z = np.cos((X**2+Y**2)/200.)+ np.random.normal(size=X.shape)
    Z2 = blur_image(Z, 3)
    plt.figure()
    plt.imshow(Z)
    plt.figure()
    plt.imshow(Z2)
    plt.show()
    