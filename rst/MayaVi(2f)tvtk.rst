\|\|<#80FF80> This page is not the main source of documentation. You are
invited to refer to the `Mayavi2 home
page <http://enthought.github.com/mayavi/mayavi>`__ for up-to-date
documentation on TVTK. In particular, bear in mind that if you are
looking for a high-level Python 3D plotting library, Mayavi also
provides the right API, and can be embedded (see the `user
guide <http://enthought.github.com/mayavi/mayavi/building_applications.html>`__).
\|\|



.. code-block:: python

    
    What is tvtk?
    =============
    
    ``tvtk`` is a `traits <http://www.enthought.com/traits>`_ 
    enabled version of `VTK <http://www.vtk.org>`_
    for 3D graphics and visualization.
    
    It provides an exact match to the VTK objects, but with a Pythonic feel, unlike 
    `Mayavi <http://code.enthought.com/projects/mayavi>`_ which aims to provide new 
    APIs
    
    Most important features are
    
    * All VTK classes are wrapped.
    * Support for traits.
    * Elementary pickle support.
    * Pythonic feel.
    * Handles numpy/Numeric/numarray arrays/Python lists transparently.
    * Support for a pipeline browser, `ivtk` 
    * High-level `mlab` module.
    * Envisage plugins for a tvtk scene and the pipeline browser.
    * MayaVi2 is built on top of tvtk
    
    See the `enthought TVTK page <http://www.enthought.com/enthought/wiki/TVTK>`_ fo
    r more details,
    in particular the `tvtk introduction <http://www.enthought.com/enthought/wiki/TV
    TKIntroduction>`_.
    
    tvtk examples
    -------------
    
    cone
    ````
    
    The following example displays a cone which can be rotated/scaled/... with the m
    ouse.
    `Simple_tvtk_cone.py
    
    <.. image:: MayaVi(2f)tvtk_attachments/Simple_tvtk_cone.py>`_
    
    quadrics
    ````````
    
    A more interesting example is the generation of some contour surfaces of an impl
    icit function.
    `Vis_quad.py
    
    <.. image:: MayaVi(2f)tvtk_attachments/Vis_quad.py>`_
    
    This displays the following scene on screen and also saves it to a file.
    
    .. image:: .. image:: MayaVi(2f)tvtk_attachments/vis_quad.png
    
    
    
    ivtk
    ----
    
    The module `tools.ivtk` makes VTK/TVTK easier to use from the Python
    interpreter. 
    For example, start IPython with::
    
      ipython -wthread 
    
    (if you have both wxPython 2.4 and wxPython 2.6 installed you will need a recent
     IPython
    and do ``ipython -wthread  -wxversion 2.6``).
    



Then you can paste the following lines:



.. code-block:: python

      from enthought.tvtk.tools import ivtk
      from enthought.tvtk.api import tvtk
      # Create a cone:
      cs = tvtk.ConeSource(resolution=100)
      mapper = tvtk.PolyDataMapper(input=cs.output)
      actor = tvtk.Actor(mapper=mapper)
    
      # Now create the viewer:
      v = ivtk.IVTKWithCrustAndBrowser(size=(600,600))
      v.open()
      v.scene.add_actors(actor)  # or v.scene.add_actor(a)
    







.. code-block:: python

    You can then explore the visualization pipeline and modify any settings.
    
    .. image:: .. image:: MayaVi(2f)tvtk_attachments/ivtk_example.png
    
    
    For creating the viewer there are different options:
    
    - ``v = ivtk.viewer()`` - this one does not need the ``v.open()`` and the ``size
    =(600,600)``
    - ``v = ivtk.IVTK()``
    - ``v = ivtk.IVTKWithCrust()``
    - ``v = ivtk.IVTKWithBrowser()``
    - ``v = ivtk.IVTKWithCrustAndBrowser)``
    
    For viewers with `Crust` you can use the python command line window to modify th
    e pipeline.
    



--------------

CategoryCookbook

