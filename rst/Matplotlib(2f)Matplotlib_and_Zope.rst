``0. Prerequisites: You need to have the following installed to successfully run this example: Zope, Matplotlib (on top of Zope's Python), Python Image Library (PIL). And one more thing - probably every body does this right, but just in case - zope instance home directory has to be writable, for following to work.``

``1. Create a file (e.g. mpl.py) in INSTANCEHOME\Extensions:``



.. code-block:: python

    import matplotlib
    matplotlib.use('Agg')
    from pylab import *
    from os import *
    from StringIO import StringIO
    from PIL import Image as PILImage
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    def chart(self):
        clf()
        img_dpi=72
        width=400
        height=300
        fig=figure(dpi=img_dpi, figsize=(width/img_dpi, height/img_dpi))
        x=arange(0, 2*pi+0.1, 0.1)
        sine=plot(x, sin(x))
        legend(sine, "y=sin x", "upper right")
        xlabel('x')
        ylabel('y=sin x')
        grid(True)
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        size = (int(canvas.figure.get_figwidth())*img_dpi, int(canvas.figure.get_fig
    height())*img_dpi)
        buf=canvas.tostring_rgb()
        im=PILImage.fromstring('RGB', size, buf, 'raw', 'RGB', 0, 1)
        imgdata=StringIO()
        im.save(imgdata, 'PNG')
        self.REQUEST.RESPONSE.setHeader('Pragma', 'no-cache')
        self.REQUEST.RESPONSE.setHeader('Content-Type', 'image/png')
        return imgdata.getvalue()
    



2. Then create an external method in ZMI (e.g. Id -> mplchart, module
name -> mpl, function name -> chart).

3. Click the Test tab and you should see the sine plot.

--------------

CategoryCookbookMatplotlib

