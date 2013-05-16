This is a very basic example showing how to display a matplotlib plot
within a Qt application using PySide. In case of problems try to change
the rcParam entry “backend.qt4” to "PySide" (e.g. by in the matplotlibrc
file).



.. code-block:: python

    #!/usr/bin/env python
    import sys
    import matplotlib
    matplotlib.use('Qt4Agg')
    import pylab
    
    from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
    
    from matplotlib.figure import Figure
    
    from PySide import QtCore, QtGui
    
    if __name__ == '__main__':
        app = QtGui.QApplication(sys.argv)
        
        # generate the plot
        fig = Figure(figsize=(600,600), dpi=72, facecolor=(1,1,1), edgecolor=(0,0,0)
    )
        ax = fig.add_subplot(111)
        ax.plot([0,1])
        # generate the canvas to display the plot
        canvas = FigureCanvas(fig)
        
        win = QtGui.QMainWindow()
        # add the plot canvas to a window
        win.setCentralWidget(canvas)
        
        win.show()
        
        sys.exit(app.exec_())
    





