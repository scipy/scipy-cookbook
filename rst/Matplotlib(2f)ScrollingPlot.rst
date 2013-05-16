Controlling an Embedded Plot with wx Scrollbars
===============================================

When plotting a very long sequence in a matplotlib canvas embedded in a
wxPython application, it sometimes is useful to be able to display a
portion of the sequence without resorting to a scrollable window so that
both axes remain visible. Here is how to do so:



.. code-block:: python

    from numpy import arange, sin, pi, float, size
    
    import matplotlib
    matplotlib.use('WXAgg')
    from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg
    from matplotlib.figure import Figure
    
    import wx
    
    class MyFrame(wx.Frame):
        def __init__(self, parent, id):
            wx.Frame.__init__(self,parent, id, 'scrollable plot',
                    style=wx.DEFAULT_FRAME_STYLE ^ wx.RESIZE_BORDER,
                    size=(800, 400))
            self.panel = wx.Panel(self, -1)
    
            self.fig = Figure((5, 4), 75)
            self.canvas = FigureCanvasWxAgg(self.panel, -1, self.fig)
            self.scroll_range = 400
            self.canvas.SetScrollbar(wx.HORIZONTAL, 0, 5, 
                                     self.scroll_range)
            
            sizer = wx.BoxSizer(wx.VERTICAL)
            sizer.Add(self.canvas, -1, wx.EXPAND)
    
            self.panel.SetSizer(sizer)
            self.panel.Fit()
    
            self.init_data()
            self.init_plot()
    
            self.canvas.Bind(wx.EVT_SCROLLWIN, self.OnScrollEvt)
    
        def init_data(self):
            
            # Generate some data to plot:
            self.dt = 0.01
            self.t = arange(0,5,self.dt)
            self.x = sin(2*pi*self.t)
    
            # Extents of data sequence: 
            self.i_min = 0
            self.i_max = len(self.t)
    
            # Size of plot window:       
            self.i_window = 100
    
            # Indices of data interval to be plotted:
            self.i_start = 0
            self.i_end = self.i_start + self.i_window
    
        def init_plot(self):
            self.axes = self.fig.add_subplot(111)
            self.plot_data = \
                      self.axes.plot(self.t[self.i_start:self.i_end],
                                     self.x[self.i_start:self.i_end])[0]
    
        def draw_plot(self):
    
            # Update data in plot:
            self.plot_data.set_xdata(self.t[self.i_start:self.i_end])
            self.plot_data.set_ydata(self.x[self.i_start:self.i_end])
    
            # Adjust plot limits:
            self.axes.set_xlim((min(self.t[self.i_start:self.i_end]),
                               max(self.t[self.i_start:self.i_end])))
            self.axes.set_ylim((min(self.x[self.i_start:self.i_end]),
                                max(self.x[self.i_start:self.i_end])))
    
            # Redraw:                  
            self.canvas.draw()
    
        def OnScrollEvt(self, event):
    
    	# Update the indices of the plot:
            self.i_start = self.i_min + event.GetPosition()
            self.i_end = self.i_min + self.i_window + event.GetPosition()
            self.draw_plot()
    
    class MyApp(wx.App):
        def OnInit(self):
            self.frame = MyFrame(parent=None,id=-1)
            self.frame.Show()
            self.SetTopWindow(self.frame)
            return True
    
    if __name__ == '__main__':
        app = MyApp()
        app.MainLoop()
    





