from enthought.tvtk import tvtk
cs = tvtk.ConeSource(resolution=100)
mapper = tvtk.PolyDataMapper(input=cs.output)
actor = tvtk.Actor(mapper=mapper)

# create a renderer:
renderer = tvtk.Renderer()
# create a render window and hand it the renderer:
render_window = tvtk.RenderWindow(size=(400,400))
render_window.add_renderer(renderer)

# create interactor and hand it the render window
# This handles mouse interaction with window.
interactor = tvtk.RenderWindowInteractor(render_window=render_window)
renderer.add_actor(actor)
interactor.initialize()
interactor.start()
