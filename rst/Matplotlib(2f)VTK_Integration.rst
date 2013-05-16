Just in case you would ever like to incorporate matplotlib plots into
your vtk application, vtk provides a really easy way to import them.

Here is a full example for now:



.. code-block:: python

    from vtk import *
    
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    import pylab as p
    
    # The vtkImageImporter will treat a python string as a void pointer
    importer = vtkImageImport()
    importer.SetDataScalarTypeToUnsignedChar()
    importer.SetNumberOfScalarComponents(4)
    
    # It's upside-down when loaded, so add a flip filter
    imflip = vtkImageFlip()
    imflip.SetInput(importer.GetOutput())
    imflip.SetFilteredAxis(1)
    
    # Map the plot as a texture on a cube
    cube = vtkCubeSource()
    
    cubeMapper = vtkPolyDataMapper()
    cubeMapper.SetInput(cube.GetOutput())
    
    cubeActor = vtkActor()
    cubeActor.SetMapper(cubeMapper)
    
    # Create a texture based off of the image
    cubeTexture = vtkTexture()
    cubeTexture.InterpolateOn()
    cubeTexture.SetInput(imflip.GetOutput())
    cubeActor.SetTexture(cubeTexture)
    
    ren = vtkRenderer()
    ren.AddActor(cubeActor)
    
    renWin = vtkRenderWindow()
    renWin.AddRenderer(ren)
    
    iren = vtkRenderWindowInteractor()
    iren.SetRenderWindow(renWin)
    
    # Now create our plot
    fig = Figure()
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.grid(True)
    ax.set_xlabel('Hello from VTK!', size=16)
    ax.bar(xrange(10), p.rand(10))
    
    # Powers of 2 image to be clean
    w,h = 1024, 1024
    dpi = canvas.figure.get_dpi()
    fig.set_figsize_inches(w / dpi, h / dpi)
    canvas.draw() # force a draw
    
    # This is where we tell the image importer about the mpl image
    extent = (0, w - 1, 0, h - 1, 0, 0)
    importer.SetWholeExtent(extent)
    importer.SetDataExtent(extent)
    importer.SetImportVoidPointer(canvas.buffer_rgba(0,0), 1)
    importer.Update()
    
    iren.Initialize()
    iren.Start()
    



To have the plot be a billboard:



.. code-block:: python

    bbmap = vtkImageMapper()
    bbmap.SetColorWindow(255.5)
    bbmap.SetColorLevel(127.5)
    bbmap.SetInput(imflip.GetOutput())
    
    bbact = vtkActor2D()
    bbact.SetMapper(hmap)
    



Comments
========




.. code-block:: python

    From zunzun Fri Aug 19 07:06:44 -0500 2005
    From: zunzun
    Date: Fri, 19 Aug 2005 07:06:44 -0500
    Subject: 
    Message-ID: <20050819070644-0500@www.scipy.org>
    
    from http://sourceforge.net/mailarchive/forum.php?thread_id=7884469&forum_id=334
    05
    
    If pylab is imported before vtk, everything works fine:
     
    import pylab, vtkpython
    pylab.ylabel("Frequency\n", multialignment="center", rotation=90)
    n, bins, patches = pylab.hist([1,1,1,2,2,3,4,5,5,5,8,8,8,8], 5)
    pylab.show()
     
    If however vtk is imported first:
     
    import vtkpython, pylab
    pylab.ylabel("Frequency\n", multialignment="center", rotation=90)
    n, bins, patches = pylab.hist([1,1,1,2,2,3,4,5,5,5,8,8,8,8], 5)
    pylab.show()
     
    then the Y axis label is positioned incorrectly on the plots.
    







.. code-block:: python

    From earthman Tue Oct 25 15:21:14 -0500 2005
    From: earthman
    Date: Tue, 25 Oct 2005 15:21:14 -0500
    Subject: 
    Message-ID: <20051025152114-0500@www.scipy.org>
    
    The reason for this is that vtk comes with it's own freetype library, and this i
    s the one being used if vtk is loaded first. Worse symptoms could be errors abou
    t fonts not being found. This is typically solved by importing vtk after other p
    ackages which might use freetype (pylab, wxPython, etc).
    







.. code-block:: python

    From mroublic Tue Jan 10 11:26:45 -0600 2006
    From: mroublic
    Date: Tue, 10 Jan 2006 11:26:45 -0600
    Subject: One more change I had to make
    Message-ID: <20060110112645-0600@www.scipy.org>
    In-reply-to: <20050819070644-0500@www.scipy.org>
    
    When I first tried this, I had the error:
    
    Traceback (most recent call last):
      File "MatplotlibToVTK.py", line 61, in ?
        importer.SetImportVoidPointer(canvas.buffer_rgba(), 1)
    TypeError: buffer_rgba() takes exactly 3 arguments (1 given)
    
    I had to add 0,0 to the import line:
     importer.SetImportVoidPointer(canvas.buffer_rgba(0,0), 1)
    
    I'm using VTK from CVS using the 5_0 Branch from around November 2005
    



The above code didn't run on my system. I had to change the following
line: fig.set\_figsize\_inches(w / dpi, h / dpi) into:
fig.set\_figsize\_inches(1.0\*w / dpi, 1.0\*h / dpi)

--------------

CategoryCookbookMatplotlib

