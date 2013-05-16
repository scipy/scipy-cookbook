#. format wiki
#. language fr

``(!) see also [:Cookbook/Matplotlib/LoadImage] to load a PNG image``

Python Imaging Library
----------------------

Apply this patch to make PIL Image objects both export and consume the
array interface (from Travis Oliphant):



.. code-block:: python

    Index: PIL/Image.py
    ===================================================================
    --- PIL/Image.py	(revision 358)
    +++ PIL/Image.py	(working copy)
    @@ -187,6 +187,42 @@
     
     }
     
    +if sys.byteorder == 'little':
    +    _ENDIAN = '<'
    +else:
    +    _ENDIAN = '>'
    +
    +_MODE_CONV = {
    +
    +    # official modes
    +    "1": ('|b1', None),
    +    "L": ('|u1', None),
    +    "I": ('%si4' % _ENDIAN, None),
    +    "F": ('%sf4' % _ENDIAN, None),
    +    "P": ('|u1', None),
    +    "RGB": ('|u1', 3),
    +    "RGBX": ('|u1', 4),
    +    "RGBA": ('|u1', 4),
    +    "CMYK": ('|u1', 4),
    +    "YCbCr": ('|u1', 4),
    +
    +    # Experimental modes include I;16, I;16B, RGBa, BGR;15,
    +    # and BGR;24.  Use these modes only if you know exactly
    +    # what you're doing...
    +
    +}
    +
    +def _conv_type_shape(im):
    +    shape = im.size[::-1]
    +    typ, extra = _MODE_CONV[im.mode]
    +    if extra is None:
    +        return shape, typ
    +    else:
    +        return shape+(extra,), typ
    +
    +
    +
    +
     MODES = _MODEINFO.keys()
     MODES.sort()
     
    @@ -491,6 +527,22 @@
             return string.join(data, "")
     
         ##
    +    # Returns the array_interface dictionary
    +    #
    +    # @return A dictionary with keys 'shape', 'typestr', 'data'
    +
    +    def __get_array_interface__(self):
    +        new = {}
    +        shape, typestr = _conv_type_shape(self)
    +        new['shape'] = shape
    +        new['typestr'] = typestr
    +        new['data'] = self.tostring()
    +        return new
    +
    +    __array_interface__ = property(__get_array_interface__, None, doc="array in
    terface")
    +
    +
    +    ##
         # Returns the image converted to an X11 bitmap.  This method
         # only works for mode "1" images.
         #
    @@ -1749,7 +1801,61 @@
     
         return apply(fromstring, (mode, size, data, decoder_name, args))
     
    +
     ##
    +# (New in 1.1.6) Create an image memory from an object exporting
    +#  the array interface (using the buffer protocol). 
    +#
    +#  If obj is not contiguous, then the tostring method is called
    +#  and frombuffer is used
    +#
    +# @param obj Object with array interface
    +# @param mode Mode to use (will be determined from type if None)
    +
    +def fromarray(obj, mode=None):
    +    arr = obj.__array_interface__
    +    shape = arr['shape']
    +    ndim = len(shape)
    +    try:
    +        strides = arr['strides']
    +    except KeyError:
    +        strides = None
    +    if mode is None:
    +        typestr = arr['typestr']
    +        if not (typestr[0] == '|' or typestr[0] == _ENDIAN or
    +                typestr[1:] not in ['u1', 'b1', 'i4', 'f4']):
    +            raise TypeError, "cannot handle data-type"
    +        typestr = typestr[:2]
    +        if typestr == 'i4':
    +            mode = 'I'
    +        elif typestr == 'f4':
    +            mode = 'F'
    +        elif typestr == 'b1':
    +            mode = '1'
    +        elif ndim == 2:
    +            mode = 'L'
    +        elif ndim == 3:
    +            mode = 'RGB'
    +        elif ndim == 4:
    +            mode = 'RGBA'
    +        else:
    +            raise TypeError, "Do not understand data."
    +    ndmax = 4
    +    bad_dims=0
    +    if mode in ['1','L','I','P','F']:
    +        ndmax = 2
    +    elif mode == 'RGB':
    +        ndmax = 3
    +    if ndim > ndmax:
    +        raise ValueError, "Too many dimensions."
    +
    +    size = shape[:2][::-1]
    +    if strides is not None:
    +        obj = obj.tostring()
    +        
    +    return frombuffer(mode, size, obj)
    +    
    +##
     # Opens and identifies the given image file.
     # <p>
     # This is a lazy operation; this function identifies the file, but the
    



Exemple
~~~~~~~




.. code-block:: python

    >>> import Image
    >>> im=Image.open('foo1.png')
    >>> a=numpy.array(p)
    # do something with a ...
    >>> im = Image.fromarray(a)
    >>> im.save( 'foo2.png' )
    



--------------

CategoryCookbook

