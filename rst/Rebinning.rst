Examples showing how to rebin data to produce a smaller or bigger array
without (and with) using interpolation.

Example 1
---------

Here we deal with the simplest case where any desired new shape is valid
and no interpolation is done on the data to determine the new values. \*
First, floating slices objects are created for each dimension. \*
Second, the coordinates of the new bins are computed from the slices
using mgrid. \* Then, coordinates are transformed to integer indices. \*
And, finally, 'fancy indexing' is used to evaluate the original array at
the desired indices.



.. code-block:: python

    def rebin( a, newshape ):
            '''Rebin an array to a new shape.
            '''
            assert len(a.shape) == len(newshape)
    
            slices = [ slice(0,old, float(old)/new) for old,new in zip(a.shape,newsh
    ape) ]
            coordinates = mgrid[slices]
            indices = coordinates.astype('i')   #choose the biggest smaller integer 
    index
            return a[tuple(indices)]
    



If we were only interested in reducing the sizes by some integer factor
then we could use:



.. code-block:: python

    def rebin_factor( a, newshape ):
            '''Rebin an array to a new shape.
            newshape must be a factor of a.shape.
            '''
            assert len(a.shape) == len(newshape)
            assert not sometrue(mod( a.shape, newshape ))
    
            slices = [ slice(None,None, old/new) for old,new in zip(a.shape,newshape
    ) ]
            return a[slices]
    



Example 2
---------

Here is an other way to deal with the reducing case for ndarrays. This
acts identically to IDL's rebin command where all values in the original
array are summed and divided amongst the entries in the new array. As in
IDL, the new shape must be a factor of the old one. The ugly 'evList
trick' builds and executes a python command of the form

a.reshape(args[0],factor[0],).sum(1)/factor[0]
a.reshape(args[0],factor[0],args[1],factor[1],).sum(1).sum(2)/factor[0]/factor[1]

etc. This general form is extended to cover the number of required
dimensions.



.. code-block:: python

    def rebin(a, *args):
        '''rebin ndarray data into a smaller ndarray of the same rank whose dimensio
    ns
        are factors of the original dimensions. eg. An array with 6 columns and 4 ro
    ws
        can be reduced to have 6,3,2 or 1 columns and 4,2 or 1 rows.
        example usages:
        >>> a=rand(6,4); b=rebin(a,3,2)
        >>> a=rand(6); b=rebin(a,2)
        '''
        shape = a.shape
        lenShape = len(shape)
        factor = asarray(shape)/asarray(args)
        evList = ['a.reshape('] + \
                 ['args[%d],factor[%d],'%(i,i) for i in range(lenShape)] + \
                 [')'] + ['.sum(%d)'%(i+1) for i in range(lenShape)] + \
                 ['/factor[%d]'%i for i in range(lenShape)]
        print ''.join(evList)
        return eval(''.join(evList))
    



The above code returns an array of the same type as the input array. If
the input is an integer array, the output values will be rounded down.
If you want a float array which correctly averages the input values
without rounding, you can do the following instead.

a.reshape(args[0],factor[0],).mean(1) BR
a.reshape(args[0],factor[0],args[1],factor[1],).mean(1).mean(2)



.. code-block:: python

    def rebin(a, *args):
        shape = a.shape
        lenShape = len(shape)
        factor = asarray(shape)/asarray(args)
        evList = ['a.reshape('] + \
                 ['args[%d],factor[%d],'%(i,i) for i in range(lenShape)] + \
                 [')'] + ['.mean(%d)'%(i+1) for i in range(lenShape)]
        print ''.join(evList)
        return eval(''.join(evList))
    



Some test cases:



.. code-block:: python

    # 2-D case
    a=rand(6,4)
    print a
    b=rebin(a,6,4)
    print b
    b=rebin(a,6,2)
    print b
    b=rebin(a,3,2)
    print b
    b=rebin(a,1,1)
    
    # 1-D case
    print b
    a=rand(4)
    print a
    b=rebin(a,4)
    print b
    b=rebin(a,2)
    print b
    b=rebin(a,1)
    print b
    



Example 3
---------

A python version of congrid, used in IDL, for resampling of data to
arbitrary sizes, using a variety of nearest-neighbour and interpolation
routines.



.. code-block:: python

    import numpy as n
    import scipy.interpolate
    import scipy.ndimage
    
    def congrid(a, newdims, method='linear', centre=False, minusone=False):
        '''Arbitrary resampling of source array to new dimension sizes.
        Currently only supports maintaining the same number of dimensions.
        To use 1-D arrays, first promote them to shape (x,1).
        
        Uses the same parameters and creates the same co-ordinate lookup points
        as IDL''s congrid routine, which apparently originally came from a VAX/VMS
        routine of the same name.
    
        method:
        neighbour - closest value from original data
        nearest and linear - uses n x 1-D interpolations using
                             scipy.interpolate.interp1d
        (see Numerical Recipes for validity of use of n 1-D interpolations)
        spline - uses ndimage.map_coordinates
    
        centre:
        True - interpolation points are at the centres of the bins
        False - points are at the front edge of the bin
    
        minusone:
        For example- inarray.shape = (i,j) & new dimensions = (x,y)
        False - inarray is resampled by factors of (i/x) * (j/y)
        True - inarray is resampled by(i-1)/(x-1) * (j-1)/(y-1)
        This prevents extrapolation one element beyond bounds of input array.
        '''
        if not a.dtype in [n.float64, n.float32]:
            a = n.cast[float](a)
        
        m1 = n.cast[int](minusone)
        ofs = n.cast[int](centre) * 0.5
        old = n.array( a.shape )
        ndims = len( a.shape )
        if len( newdims ) != ndims:
            print "[congrid] dimensions error. " \
                  "This routine currently only support " \
                  "rebinning to the same number of dimensions."
            return None
        newdims = n.asarray( newdims, dtype=float )    
        dimlist = []
    
        if method == 'neighbour':
            for i in range( ndims ):
                base = n.indices(newdims)[i]
                dimlist.append( (old[i] - m1) / (newdims[i] - m1) \
                                * (base + ofs) - ofs )
            cd = n.array( dimlist ).round().astype(int)
            newa = a[list( cd )]
            return newa
        
        elif method in ['nearest','linear']:
            # calculate new dims
            for i in range( ndims ):
                base = n.arange( newdims[i] )
                dimlist.append( (old[i] - m1) / (newdims[i] - m1) \
                                * (base + ofs) - ofs )
            # specify old dims
            olddims = [n.arange(i, dtype = n.float) for i in list( a.shape )]
    
            # first interpolation - for ndims = any
            mint = scipy.interpolate.interp1d( olddims[-1], a, kind=method )
            newa = mint( dimlist[-1] )
    
            trorder = [ndims - 1] + range( ndims - 1 )
            for i in range( ndims - 2, -1, -1 ):
                newa = newa.transpose( trorder )
    
                mint = scipy.interpolate.interp1d( olddims[i], newa, kind=method )
                newa = mint( dimlist[i] )
    
            if ndims > 1:
                # need one more transpose to return to original dimensions
                newa = newa.transpose( trorder )
    
            return newa
        elif method in ['spline']:
            oslices = [ slice(0,j) for j in old ]
            oldcoords = n.ogrid[oslices]
            nslices = [ slice(0,j) for j in list(newdims) ]
            newcoords = n.mgrid[nslices]
    
            newcoords_dims = range(n.rank(newcoords))
            #make first index last
            newcoords_dims.append(newcoords_dims.pop(0))
            newcoords_tr = newcoords.transpose(newcoords_dims)
            # makes a view that affects newcoords
    
            newcoords_tr += ofs        
    
            deltas = (n.asarray(old) - m1) / (newdims - m1)
            newcoords_tr *= deltas
    
            newcoords_tr -= ofs
    
            newa = scipy.ndimage.map_coordinates(a, newcoords)
            return newa
        else:
            print "Congrid error: Unrecognized interpolation type.\n", \
                  "Currently only \'neighbour\', \'nearest\',\'linear\',", \
                  "and \'spline\' are supported."
            return None
    



--------------

CategoryCookbook

