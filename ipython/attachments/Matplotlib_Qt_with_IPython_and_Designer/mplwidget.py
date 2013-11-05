
import sys, os, random
from qt import *

import matplotlib
matplotlib.use('Agg')
import pylab

from matplotlib.numerix import arange, sin, pi

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class MatplotlibWidget(FigureCanvas):
    """Ultimately, this is a QWidget (as well as a FigureCanvasAgg, etc.)."""
    def __init__(self, parent=None, name=None, width=5, height=4, dpi=100, bgcolor=None):
	self.parent = parent
	if self.parent:
		bgc = parent.backgroundBrush().color()
		bgcolor = float(bgc.red())/255.0, float(bgc.green())/255.0, float(bgc.blue())/255.0
		#bgcolor = "#%02X%02X%02X" % (bgc.red(), bgc.green(), bgc.blue())

        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor=bgcolor, edgecolor=bgcolor)
        self.axes = self.fig.add_subplot(111)
        # We want the axes cleared every time plot() is called
        self.axes.hold(False)

        self.compute_initial_figure()
        
        FigureCanvas.__init__(self, self.fig)
        self.reparent(parent, QPoint(0, 0))

        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def sizeHint(self):
        w = self.fig.get_figwidth()
        h = self.fig.get_figheight()
        return QSize(w, h)

    def minimumSizeHint(self):
        return QSize(10, 10)

    def compute_initial_figure(self):
       	t = arange(0.0, 3.0, 0.01)
	s = sin(2*pi*t)
	self.axes.plot(t, s) 

