Operating on color vectors
==========================

Ever wanted to reverse a colormap, or to desaturate one ? Here is a
routine to apply a function to the look up table of a colormap:



.. code-block:: python

    def cmap_map(function,cmap):
        """ Applies function (which should operate on vectors of shape 3:
        [r, g, b], on colormap cmap. This routine will break any discontinuous     p
    oints in a colormap.
        """
        cdict = cmap._segmentdata
        step_dict = {}
        # Firt get the list of points where the segments start or end
        for key in ('red','green','blue'):         step_dict[key] = map(lambda x: x[
    0], cdict[key])
        step_list = sum(step_dict.values(), [])
        step_list = array(list(set(step_list)))
        # Then compute the LUT, and apply the function to the LUT
        reduced_cmap = lambda step : array(cmap(step)[0:3])
        old_LUT = array(map( reduced_cmap, step_list))
        new_LUT = array(map( function, old_LUT))
        # Now try to make a minimal segment definition of the new LUT
        cdict = {}
        for i,key in enumerate(('red','green','blue')):
            this_cdict = {}
            for j,step in enumerate(step_list):
                if step in step_dict[key]:
                    this_cdict[step] = new_LUT[j,i]
                elif new_LUT[j,i]!=old_LUT[j,i]:
                    this_cdict[step] = new_LUT[j,i]
            colorvector=  map(lambda x: x + (x[1], ), this_cdict.items())
            colorvector.sort()
            cdict[key] = colorvector
    
        return matplotlib.colors.LinearSegmentedColormap('colormap',cdict,1024)
    



Lets try it out: I want a jet colormap, but lighter, so that I can plot
things on top of it:



.. code-block:: python

    light_jet = cmap_map(lambda x: x/2+0.5, cm.jet)
    x,y=mgrid[1:2,1:10:0.1]
    imshow(y, cmap=light_jet)
    



.. image:: Matplotlib(2f)ColormapTransformations_attachments/light_jet4.png

As a comparison, this is what the original jet looks like:
.. image:: Matplotlib(2f)ColormapTransformations_attachments/jet.png

Operating on indices
====================

OK, but what if you want to change the indices of a colormap, but not
its colors.



.. code-block:: python

    def cmap_xmap(function,cmap):
        """ Applies function, on the indices of colormap cmap. Beware, function
        should map the [0, 1] segment to itself, or you are in for surprises.
    
        See also cmap_xmap.
        """
        cdict = cmap._segmentdata
        function_to_map = lambda x : (function(x[0]), x[1], x[2])
        for key in ('red','green','blue'):         cdict[key] = map(function_to_map,
     cdict[key])
            cdict[key].sort()
            assert (cdict[key][0]<0 or cdict[key][-1]>1), "Resulting indices extend 
    out of the [0, 1] segment."
    
    
        return matplotlib.colors.LinearSegmentedColormap('colormap',cdict,1024)
    



Discrete colormap
=================

Here is how you can discretize a continuous colormap.



.. code-block:: python

    def cmap_discretize(cmap, N):
        """Return a discrete colormap from the continuous colormap cmap.
        
            cmap: colormap instance, eg. cm.jet. 
            N: number of colors.
        
        Example
            x = resize(arange(100), (5,100))
            djet = cmap_discretize(cm.jet, 5)
            imshow(x, cmap=djet)
        """
        
        if type(cmap) == str:
            cmap = get_cmap(cmap)
        colors_i = concatenate((linspace(0, 1., N), (0.,0.,0.,0.)))
        colors_rgba = cmap(colors_i)
        indices = linspace(0, 1., N+1)
        cdict = {}
        for ki,key in enumerate(('red','green','blue')):
            cdict[key] = [ (indices[i], colors_rgba[i-1,ki], colors_rgba[i,ki]) for 
    i in xrange(N+1) ]
        # Return colormap object.
        return matplotlib.colors.LinearSegmentedColormap(cmap.name + "_%d"%N, cdict,
     1024)
    



So for instance, this is what you would get by doing .
.. image:: Matplotlib(2f)ColormapTransformations_attachments/dicrete_jet1.png

