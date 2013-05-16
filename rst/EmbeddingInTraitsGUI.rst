#. 

   #. page was renamed from EmbeddingInTraitsGUI

Embedding a Matplotlib Figure in a Traits App
=============================================

Traits, part of the\ `Enthought Tools
Suit <http://code.enthought.com/>`__, provides a great framework for
creating GUI Apps without a lot of the normal boilerplate required to
connect the UI the rest of the application logic. A brief introduction
to Traits can be found [:TraitsUI:here]. Although ETS comes with it's
own traits-aware plotting framework (Chaco), if you already know
matplotlib it is just as easy to embed this instead. The advantages of
Chaco (IMHO) are its interactive "tools", an (in development) OpenGL
rendering backend and an easy-to-understand codebase. However,
matplotlib has more and better documentation and better defaults; it
just works. The key to getting TraitsUI and matplotlib to play nice is
to use the mpl object-oriented API, rather than pylab / pyplot. This
recipe requires the following packages:

| ``* numpy``
| ``* wxPython``
| ``* matplotlib``
| ``* Traits > 3.0``
| ``* TraitsGUI > 3.0``
| ``* TraitsBackendWX > 3.0``
``For this example, we will display a function (y, a sine wave) of one variable (x, a numpy ndarray) and one parameter (scale, a float value with bounds). We want to be able to vary the parameter from the UI and see the resulting changes to y in a plot window. Here's what the final result looks like: ``\ ```.. image:: EmbeddingInTraitsGUI_attachments/mpl_in_traits_view.png`` <.. image:: EmbeddingInTraitsGUI_attachments/mpl_in_traits_view.png>`__\ `` The TraitsUI "!CustomEditor" can be used to display any wxPython window as the editor for the object. You simply pass the !CustomEditor a callable which, when called, returns the wxPython window you want to display. In this case, our !MakePlot() function returns a wxPanel containing the mpl !FigureCanvas and Navigation toolbar. This example exploits a few of Traits' features. We use "dynamic initialisation" to create the Axes and Line2D objects on demand (using the _xxx_default methods).  We use Traits "notification", to call update_line(...) whenever the x- or y-data is changed. Further, the y-data is declared as a Property trait which depends on both the 'scale' parameter and the x-data. 'y' is then recalculated on demand, whenever either 'scale' or 'x' change. The 'cached_property' decorator prevents recalculation of y if it's dependancies ``\ *``are``
``not``*\ `` modified.``

Finally, there's a bit of wx-magic in the redraw() method to limit the
redraw rate by delaying the actual drawing by 50ms. This uses the
wx.!CallLater class. This prevents excessive redrawing as the slider is
dragged, keeping the UI from
lagging.\ `Here's <.. image:: EmbeddingInTraitsGUI_attachments/mpl_editor.py>`__ the full listing:



.. code-block:: python

    #!python
    """
    A simple demonstration of embedding a matplotlib plot window in
    a traits-application. The CustomEditor allow any wxPython window
    to be used as an editor. The demo also illustrates Property traits,
    which provide nice dependency-handling and dynamic initialisation, using
    the _xxx_default(...) method.
    """
    from enthought.traits.api import HasTraits, Instance, Range,\
                                    Array, on_trait_change, Property,\
                                    cached_property, Bool
    from enthought.traits.ui.api import View, Item
    from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg
    from matplotlib.backends.backend_wx import NavigationToolbar2Wx
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    from matplotlib.lines import Line2D
    from enthought.traits.ui.api import CustomEditor
    import wx
    import numpy
    def MakePlot(parent, editor):
        """
        Builds the Canvas window for displaying the mpl-figure
        """
        fig = editor.object.figure
        panel = wx.Panel(parent, -1)
        canvas = FigureCanvasWxAgg(panel, -1, fig)
        toolbar = NavigationToolbar2Wx(canvas)
        toolbar.Realize()
        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(canvas,1,wx.EXPAND|wx.ALL,1)
        sizer.Add(toolbar,0,wx.EXPAND|wx.ALL,1)
        panel.SetSizer(sizer)
        return panel
    class PlotModel(HasTraits):
        """A Model for displaying a matplotlib figure"""
        #we need instances of a Figure, a Axes and a Line2D
        figure = Instance(Figure, ())
        axes = Instance(Axes)
        line = Instance(Line2D)
        _draw_pending = Bool(False) #a flag to throttle the redraw rate
        #a variable paremeter
        scale = Range(0.1,10.0)
        #an independent variable
        x = Array(value=numpy.linspace(-5,5,512))
        #a dependent variable
        y = Property(Array, depends_on=['scale','x'])
        traits_view = View(
                        Item('figure',
                             editor=CustomEditor(MakePlot),
                             resizable=True),
                        Item('scale'),
                        resizable=True
                        )
        def _axes_default(self):
            return self.figure.add_subplot(111)
        def _line_default(self):
            return self.axes.plot(self.x, self.y)[0]
        @cached_property
        def _get_y(self):
            return numpy.sin(self.scale * self.x)
        @on_trait_change("x, y")
        def update_line(self, obj, name, val):
            attr = {'x': "set_xdata", 'y': "set_ydata"}[name]
            getattr(self.line, attr)(val)
            self.redraw()
        def redraw(self):
            if self._draw_pending:
                return
            canvas = self.figure.canvas
            if canvas is None:
                return
            def _draw():
                canvas.draw()
                self._draw_pending = False
            wx.CallLater(50, _draw).Start()
            self._draw_pending = True
    if __name__=="__main__":
        model = PlotModel(scale=2.0)
        model.configure_traits()
    





