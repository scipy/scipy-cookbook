# gaussfitter.py
# created by Adam Ginsburg (adam.ginsburg@colorado.edu or keflavich@gmail.com) 3/17/08)
from numpy import *
from scipy import optimize
from scipy import stats

def moments(data,circle,rotate,vheight):
    """Returns (height, amplitude, x, y, width_x, width_y, rotation angle)
    the gaussian parameters of a 2D distribution by calculating its
    moments.  Depending on the input parameters, will only output 
    a subset of the above"""
    total = data.sum()
    X, Y = indices(data.shape)
    x = (X*data).sum()/total
    y = (Y*data).sum()/total
    col = data[:, int(y)]
    width_x = sqrt(abs((arange(col.size)-y)**2*col).sum()/col.sum())
    row = data[int(x), :]
    width_y = sqrt(abs((arange(row.size)-x)**2*row).sum()/row.sum())
    width = ( width_x + width_y ) / 2.
    height = stats.mode(data.ravel())[0][0]
    amplitude = data.max()-height
    mylist = [amplitude,x,y]
    if vheight==1:
        mylist = [height] + mylist
    if circle==0:
        mylist = mylist + [width_x,width_y]
    else:
        mylist = mylist + [width]
    if rotate==1:
        mylist = mylist + [0.] #rotation "moment" is just zero...
    return tuple(mylist)

def twodgaussian(inpars, circle, rotate, vheight):
    """Returns a 2d gaussian function of the form:
        x' = cos(rota) * x - sin(rota) * y
        y' = sin(rota) * x + cos(rota) * y
        (rota should be in degrees)
        g = b + a exp ( - ( ((x-center_x)/width_x)**2 +
        ((y-center_y)/width_y)**2 ) / 2 )

        where x and y are the input parameters of the returned function,
        and all other parameters are specified by this function

        However, the above values are passed by list.  The list should be:
        inpars = (height,amplitude,center_x,center_y,width_x,width_y,rota)

        You can choose to ignore / neglect some of the above input parameters using the following options:
            circle=0 - default is an elliptical gaussian (different x, y widths), but can reduce
                the input by one parameter if it's a circular gaussian
            rotate=1 - default allows rotation of the gaussian ellipse.  Can remove last parameter
                by setting rotate=0
            vheight=1 - default allows a variable height-above-zero, i.e. an additive constant
                for the Gaussian function.  Can remove first parameter by setting this to 0
        """
    inpars_old = inpars
    inpars = list(inpars)
    if vheight == 1:
        height = inpars.pop(0)
        height = float(height)
    else:
        height = float(0)
    amplitude, center_x, center_y = inpars.pop(0),inpars.pop(0),inpars.pop(0)
    amplitude = float(amplitude)
    center_x = float(center_x)
    center_y = float(center_y)
    if circle == 1:
        width = inpars.pop(0)
        width_x = float(width)
        width_y = float(width)
    else:
        width_x, width_y = inpars.pop(0),inpars.pop(0)
        width_x = float(width_x)
        width_y = float(width_y)
    if rotate == 1:
        rota = inpars.pop(0)
        rota = pi/180. * float(rota)
        rcen_x = center_x * cos(rota) - center_y * sin(rota)
        rcen_y = center_x * sin(rota) + center_y * cos(rota)
    else:
        rcen_x = center_x
        rcen_y = center_y
    if len(inpars) > 0:
        raise ValueError("There are still input parameters:" + str(inpars) + \
                " and you've input: " + str(inpars_old) + " circle=%d, rotate=%d, vheight=%d" % (circle,rotate,vheight) )
            
    def rotgauss(x,y):
        if rotate==1:
            xp = x * cos(rota) - y * sin(rota)
            yp = x * sin(rota) + y * cos(rota)
        else:
            xp = x
            yp = y
        g = height+amplitude*exp(
            -(((rcen_x-xp)/width_x)**2+
            ((rcen_y-yp)/width_y)**2)/2.)
        return g
    return rotgauss

def gaussfit(data,err=None,params=[],autoderiv=1,return_all=0,circle=0,rotate=1,vheight=1):
    """
    Gaussian fitter with the ability to fit a variety of different forms of 2-dimensional gaussian.
    
    Input Parameters:
        data - 2-dimensional data array
        err=None - error array with same size as data array
        params=[] - initial input parameters for Gaussian function.
            (height, amplitude, x, y, width_x, width_y, rota)
            if not input, these will be determined from the moments of the system, 
            assuming no rotation
        autoderiv=1 - use the autoderiv provided in the lmder.f function (the alternative
            is to us an analytic derivative with lmdif.f: this method is less robust)
        return_all=0 - Default is to return only the Gaussian parameters.  See below for
            detail on output
        circle=0 - default is an elliptical gaussian (different x, y widths), but can reduce
            the input by one parameter if it's a circular gaussian
        rotate=1 - default allows rotation of the gaussian ellipse.  Can remove last parameter
            by setting rotate=0
        vheight=1 - default allows a variable height-above-zero, i.e. an additive constant
            for the Gaussian function.  Can remove first parameter by setting this to 0

    Output:
        Default output is a set of Gaussian parameters with the same shape as the input parameters
        Can also output the covariance matrix, 'infodict' that contains a lot more detail about
            the fit (see scipy.optimize.leastsq), and a message from leastsq telling what the exit
            status of the fitting routine was

        Warning: Does NOT necessarily output a rotation angle between 0 and 360 degrees.
    """
    if params == []:
        params = (moments(data,circle,rotate,vheight))
    if err == None:
        errorfunction = lambda p: ravel((twodgaussian(p,circle,rotate,vheight)(*indices(data.shape)) - data))
    else:
        errorfunction = lambda p: ravel((twodgaussian(p,circle,rotate,vheight)(*indices(data.shape)) - data)/err)
    if autoderiv == 0:
        # the analytic derivative, while not terribly difficult, is less efficient and useful.  I only bothered
        # putting it here because I was instructed to do so for a class project - please ask if you would like 
        # this feature implemented
        raise ValueError("I'm sorry, I haven't implemented this feature yet.")
    else:
        p, cov, infodict, errmsg, success = optimize.leastsq(errorfunction, params, full_output=1)
    if  return_all == 0:
        return p
    elif return_all == 1:
        return p,cov,infodict,errmsg
