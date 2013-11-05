#!/usr/bin/env python
"""
This example demonstrates the use of the contour filter, and the use of
the vtkSampleFunction to generate a volume of data samples from an
implicit function.

Conversion of VisQuad.py of the VTK example to tvtk
(and some additions).
"""

from enthought.tvtk import tvtk

# General form for a quadric to create an elliptical data field.
quadric = tvtk.Quadric(coefficients=(.5, 1, .2, 0, .1, 0, 0, .2, 0, 0))

# Sample implicit function over a specified x-y-z range.
# (here it defaults to -1,1 in the x,y,z directions).
sample = tvtk.SampleFunction(implicit_function=quadric,
                             sample_dimensions=(30, 30, 30))

# Create five surfaces F(x,y,z) = constant between range specified:
contours = tvtk.ContourFilter(input=sample.output)
contours.generate_values(5, 0.0, 1.2)

contMapper = tvtk.PolyDataMapper(input=contours.output, scalar_range=(0.0, 1.2))
contActor = tvtk.Actor(mapper=contMapper)

# outline around the data.
outline = tvtk.OutlineFilter(input=sample.output)
outlineMapper = tvtk.PolyDataMapper(input=outline.output)
outlineActor = tvtk.Actor(mapper=outlineMapper)
outlineActor.property.color=(0, 0, 0)

# The usual rendering stuff.
ren = tvtk.Renderer(background=(0.95, 0.95, 1.0))
renWin = tvtk.RenderWindow()
renWin.add_renderer(ren)
iren = tvtk.RenderWindowInteractor(render_window=renWin)
ren.add_actor(contActor)
ren.add_actor(outlineActor)

# some nice view:
ren.active_camera.elevation(10)

iren.initialize()
renWin.render()

# save scene as png:
renderLarge = tvtk.RenderLargeImage(input=ren, magnification=1)
writer = tvtk.PNGWriter(input=renderLarge.output, file_name="vis_quad.png")
writer.write()

# interactive part starts here:
iren.start()
