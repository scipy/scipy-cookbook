import numpy as N
import unittest
from numpy.testing import NumpyTestCase, assert_array_almost_equal,             assert_almost_equal, assert_equal
import warnings

def segment_axis(a, length, overlap=0, axis=None, end='cut', endvalue=0):
    """Generate a new array that chops the given array along the given axis into overlapping frames.

    example:
    >>> segment_axis(arange(10), 4, 2)
    array([[0, 1, 2, 3],
           [2, 3, 4, 5],
           [4, 5, 6, 7],
           [6, 7, 8, 9]])

    arguments:
    a       The array to segment
    length  The length of each frame
    overlap The number of array elements by which the frames should overlap
    axis    The axis to operate on; if None, act on the flattened array
    end     What to do with the last frame, if the array is not evenly
            divisible into pieces. Options are:

            'cut'   Simply discard the extra values
            'wrap'  Copy values from the beginning of the array
            'pad'   Pad with a constant value

    endvalue    The value to use for end='pad'

    The array is not copied unless necessary (either because it is 
    unevenly strided and being flattened or because end is set to 
    'pad' or 'wrap').
    """

    if axis is None:
        a = N.ravel(a) # may copy
        axis = 0

    l = a.shape[axis]

    if overlap>=length:
        raise ValueError, "frames cannot overlap by more than 100%"
    if overlap<0 or length<=0:
        raise ValueError, "overlap must be nonnegative and length must be positive"

    if l<length or (l-length)%(length-overlap):
        if l>length:
            roundup = length + (1+(l-length)//(length-overlap))*(length-overlap)
            rounddown = length + ((l-length)//(length-overlap))*(length-overlap)
        else:
            roundup = length
            rounddown = 0
        assert rounddown<l<roundup
        assert roundup==rounddown+(length-overlap) or (roundup==length and rounddown==0)
        a = a.swapaxes(-1,axis)

        if end=='cut':
            a = a[...,:rounddown]
        elif end in ['pad','wrap']: # copying will be necessary
            s = list(a.shape)
            s[-1]=roundup
            b = N.empty(s,dtype=a.dtype)
            b[...,:l] = a
            if end=='pad':
                b[...,l:] = endvalue
            elif end=='wrap':
                b[...,l:] = a[...,:roundup-l]
            a = b
        
        a = a.swapaxes(-1,axis)


    l = a.shape[axis]
    if l==0:
        raise ValueError, "Not enough data points to segment array in 'cut' mode; try 'pad' or 'wrap'"
    assert l>=length
    assert (l-length)%(length-overlap) == 0
    n = 1+(l-length)//(length-overlap)
    s = a.strides[axis]
    newshape = a.shape[:axis]+(n,length)+a.shape[axis+1:]
    newstrides = a.strides[:axis]+((length-overlap)*s,s) + a.strides[axis+1:]

    try: 
        return N.ndarray.__new__(N.ndarray,strides=newstrides,shape=newshape,buffer=a,dtype=a.dtype)
    except TypeError:
        warnings.warn("Problem with ndarray creation forces copy.")
        a = a.copy()
        # Shape doesn't change but strides does
        newstrides = a.strides[:axis]+((length-overlap)*s,s) + a.strides[axis+1:]
        return N.ndarray.__new__(N.ndarray,strides=newstrides,shape=newshape,buffer=a,dtype=a.dtype)
        


class TestSegment(NumpyTestCase):
    def test_simple(self):
        assert_equal(segment_axis(N.arange(6),length=3,overlap=0),
                         N.array([[0,1,2],[3,4,5]]))

        assert_equal(segment_axis(N.arange(7),length=3,overlap=1),
                         N.array([[0,1,2],[2,3,4],[4,5,6]]))

        assert_equal(segment_axis(N.arange(7),length=3,overlap=2),
                         N.array([[0,1,2],[1,2,3],[2,3,4],[3,4,5],[4,5,6]]))

    def test_error_checking(self):
        self.assertRaises(ValueError,
                lambda: segment_axis(N.arange(7),length=3,overlap=-1))
        self.assertRaises(ValueError,
                lambda: segment_axis(N.arange(7),length=0,overlap=0))
        self.assertRaises(ValueError,
                lambda: segment_axis(N.arange(7),length=3,overlap=3))
        self.assertRaises(ValueError,
                lambda: segment_axis(N.arange(7),length=8,overlap=3))

    def test_ending(self):
        assert_equal(segment_axis(N.arange(6),length=3,overlap=1,end='cut'),
                         N.array([[0,1,2],[2,3,4]]))
        assert_equal(segment_axis(N.arange(6),length=3,overlap=1,end='wrap'),
                         N.array([[0,1,2],[2,3,4],[4,5,0]]))
        assert_equal(segment_axis(N.arange(6),length=3,overlap=1,end='pad',endvalue=-17),
                         N.array([[0,1,2],[2,3,4],[4,5,-17]]))

    def test_multidimensional(self):
        
        assert_equal(segment_axis(N.ones((2,3,4,5,6)),axis=3,length=3,overlap=1).shape,
                     (2,3,4,2,3,6))

        assert_equal(segment_axis(N.ones((2,5,4,3,6)).swapaxes(1,3),axis=3,length=3,overlap=1).shape,
                     (2,3,4,2,3,6))

        assert_equal(segment_axis(N.ones((2,3,4,5,6)),axis=2,length=3,overlap=1,end='cut').shape,
                     (2,3,1,3,5,6))

        assert_equal(segment_axis(N.ones((2,3,4,5,6)),axis=2,length=3,overlap=1,end='wrap').shape,
                     (2,3,2,3,5,6))

        assert_equal(segment_axis(N.ones((2,3,4,5,6)),axis=2,length=3,overlap=1,end='pad').shape,
                     (2,3,2,3,5,6))

if __name__=='__main__':
    unittest.main()
