#. 

   #. page was renamed from Cookbook/MayaVi/UsingMayavi2

TableOfContents

\|\|<#80FF80> This page presents scripting Mayavi2 using the advanced,
object-oriented API. Mayavi2 has recently acquired an easy-to-use,
thought maybe not as powerful, scripting module: mlab. You are invited
to refer to the
`section <http://code.enthought.com/projects/mayavi/docs/development/mayavi/html/mlab.html>`__
of `Mayavi2 user
guide <http://code.enthought.com/projects/mayavi/docs/development/mayavi/html/>`__.
Reading this page will give you a deeper understanding of how Mayavi2
works, and it complements the user guide. \|\|

Introduction
============

To script !MayaVi2, you need (at least):

| ``* your favorite text editor;``
| ``* python installed ;-)``
| ``* !MayaVi2 installed ;-)``

Scripting !MayaVi2 is quite simple because !MayaVi2 is written in python
and based on TVTK, which eases the uses of all VTK objects.

In the following, you'll be learned how to script and use !MayaVi2
modules and filters.

Modules can be split in two parts:

``* modules which do not interact with VTK data, and are seldom modified/handled (Outline, Axes, !OrientationAxes and Text). These are called the "basic" modules. Although color bar is not strictly speaking a module, it will be presented here. Setting a few parameters for rendering your scene will be also presented.``

``* modules which do interact with VTK data, and those you want to play with (i.e. all the remainder modules ;-).``

Before starting, let's see the "main template" of a !MayaVi2 script
written in python.

Main template: create your MayaVi2 class
========================================

A !MayaVi2 script should contain at least the following few lines:



.. code-block:: python

    #! /usr/bin/env python
    
    from enthought.mayavi.app import Mayavi
    
    class MyClass(Mayavi):
        
        def run(self):
            script = self.script
            # `self.script` is the MayaVi Script interface (an instance of
            # enthought.mayavi.script.Script) that is created by the
            # base `Mayavi` class.  Here we save a local reference for
            # convenience.
      
            ## import any Mayavi modules and filters you want (they must be done her
    #e!)
            .../...
    
            script.new_scene()                      # to create the rendering scene
    
            ## your stuff (modules, filters, etc) here
            .../...
    
    if __name__ == '__main__':
        
        mc = MyClass()
        mc.main()
    



Adding modules or filters is quite simple: you have to import it, and
then add it to your !MayaVi2 class.

To add a module, type:



.. code-block:: python

    from enthought.mayavi.modules.foo_module import FooModule
    .../...
    mymodule = FooModule()
    script.add_module(mymodule)
    



To add a filter, type:



.. code-block:: python

    from enthought.mayavi.filters.bar_filter import BarFilter
    .../...
    myfilter = BarFilter()
    script.add_filter(myfilter)
    



Notice the used syntax: for modules for example, foo\_module is the
foo\_module python file (without the extension .py) in the subdirectory
module/ of mayavi/ directory (lower case, underscore); this file
contains the class FooModule (no underscore, capitalized name).

But first of all, before rendering your scene with the modules and the
filters you want to use, you have to load some data, of course.

Loading data
============

You have the choice between:

``* create a 3D data array, for scalars data (for vectors data, you have to use a 4D scalars data, i.e. a 3D scalar data for each component) and load it with !ArraySource method;``

``* load a data file with !FileReader methods.``

Loading data from array using ArraySource method
------------------------------------------------

For example, we will create a 50\*50\*50 3D (scalar) array of a product
of cosinus & sinus functions.

To do this, we need to load the appropriate modules:



.. code-block:: python

    import scipy
    from scipy import ogrid, sin, cos, sqrt, pi
    
    from enthought.mayavi.sources.array_source import ArraySource
    
    Nx = 50
    Ny = 50
    Nz = 50
    
    Lx = 1
    Ly = 1
    Lz = 1
    
    x, y, z = ogrid[0:Lx:(Nx+1)*1j,0:Ly:(Ny+1)*1j,0:Lz:(Nz+1)*1j]
    
    # Strictly speaking, H is the magnetic field of the "transverse electric" eigenm
    #ode m=n=p=1
    # of a rectangular resonator cavity, with dimensions Lx, Ly, Lz
    Hx = sin(x*pi/Lx)*cos(y*pi/Ly)*cos(z*pi/Lz)
    Hy = cos(x*pi/Lx)*sin(y*pi/Ly)*cos(z*pi/Lz)
    Hz = cos(x*pi/Lx)*cos(y*pi/Ly)*sin(z*pi/Lz)
    Hv_scal = sqrt(Hx**2 + Hy**2 + Hz**2)
    
    # We want to load a scalars data (Hv_scal) as magnitude of a given 3D vector (Hv
    # = {Hx, Hy, Hz})
    # Hv_scal is a 3D scalars data, Hv is a 4D scalars data
    src = ArraySource()
    src.scalar_data = Hv_scal # load scalars data
    
    # To load vectors data
    # src.vector_data = Hv
    



Loading data from file using FileReader methods
-----------------------------------------------

To load a VTK data file, say heart.vtk file in mayavi/examples/data/
directory, simply type:



.. code-block:: python

    from enthought.mayavi.sources.vtk_file_reader import VTKFileReader
    
    src = VTKFileReader()
    src.initialize("heart.vtk")
    



Note: Files with .vtk extension are called "legacy VTK" files. !MayaVi2
can read a lot of other files formats (XML, files from Ensight, Plot3D
and so on). For example, you can load an XML file (with extension .vti,
.vtp, .vtr, .vts, .vtu, etc) using VTKXML!FileReader method.

Add the source to your MayaVi2 class
------------------------------------

Then, once your data are loaded using one of the two methods above, add
the source with the add\_source() method in the body of the class
!MyClass (after script.new\_scene):



.. code-block:: python

    script.add_source(src)
    



The four basic modules Outline, Axes, !OrientationAxes and Text will be
presented now.

Basic Modules
=============

See the [:Cookbook/MayaVi/ScriptingMayavi2/BasicModules: Basic Modules]
wiki page.

Main Modules
============

See the [:Cookbook/MayaVi/ScriptingMayavi2/MainModules: Main Modules]
wiki page.

Filters
=======

See the [:Cookbook/MayaVi/ScriptingMayavi2/Filters: Filters] wiki page.

--------------

CategoryCookbook

