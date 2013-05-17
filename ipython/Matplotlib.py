# <markdowncell>

# The cookbook is a place for community contributions of recipes, howtos
# and examples.
# 
# Complete documentation and tutorials for matplotlib can be found at
# [matplotlib's webpage](http://matplotlib.sourceforge.net/)
# 
# <BR> <TableOfContents>
# 
# Simple Plotting
# ===============
# 
# `* [:Cookbook/Matplotlib/SigmoidalFunctions:Sigmoidal Functions] - plotting simple functions`\
# ` . `[`![](files/Matplotlib_attachments/sigmoids_small.png`](![](files/Matplotlib_attachments/sigmoids_small.png)\
)# `* [:Cookbook/Matplotlib/MultilinePlots:Multiline Plots] - how to plot multiple lines over one another`\
# ` . `[`![](files/Matplotlib_attachments/multiline.png`](![](files/Matplotlib_attachments/multiline.png)\
)# `* [:Cookbook/Matplotlib/BarCharts:Bar Charts] - how to make a bar chart`\
# ` . `[`![](files/Matplotlib_attachments/barchartscaled.png`](![](files/Matplotlib_attachments/barchartscaled.png)\
)# `* [:Cookbook/Matplotlib/Common Errors:Common Errors] - Compilation of common errors that can cause erroneous behavior. Check before emailing mailing lists.`\
# `* ["/Animations"] - how to animate your figures.`\
# `* [:Cookbook/Matplotlib/MulticoloredLine:Multicolored Line] - different colors for different parts of a line`\
# ` . `[`![](files/Matplotlib_attachments/colored_line.png`](![](files/Matplotlib_attachments/colored_line.png)\
)# `* [:Cookbook/Matplotlib/ShadedRegions:Shaded Regions] - how to plot grey shaded regions using transparency.`\
# ` . `[`![](files/Matplotlib_attachments/shaded_small.png`](![](files/Matplotlib_attachments/shaded_small.png)\
)# `* ["/Arrows"] - how to plot arrows`\
# ` . `[`![](files/Matplotlib_attachments/plot_arrow_small.png`](![](files/Matplotlib_attachments/plot_arrow_small.png)\
)# `* [:Cookbook/Matplotlib/UnfilledHistograms:Unfilled Histograms] - how to plot histograms that are un-filled and don't look like bar charts.`\
# ` . `[`![](files/Matplotlib_attachments/hist_outline_small.png`](![](files/Matplotlib_attachments/hist_outline_small.png)\
)# `* ["Cookbook/Histograms"] - 2D histograms with variable bin width.`\
# ` . `[`![](files/Matplotlib_attachments/Cookbook/Histograms/histogram2d.png`](![](files/Matplotlib_attachments/Cookbook/Histograms/histogram2d.png)\
)# `* [:Cookbook/Matplotlib/CustomLogLabels:Custom Log Plot Labels] - plotting log plots with custom tick labels that are formatted as integer numbers rather than exponents as is the default.`\
# ` . `[`![](files/Matplotlib_attachments/log_labels_small.png`](![](files/Matplotlib_attachments/log_labels_small.png)\
)# `* [:Cookbook/Matplotlib/ThickAxes:Thick Axes] - how to make thick axes lines and bold fonts.`\
# ` . `[`![](files/Matplotlib_attachments/thick_axes.png`](![](files/Matplotlib_attachments/thick_axes.png)\
)# `* ["/Maps"] - how to plot data on map projections`\
# ` . `[`![](files/Matplotlib_attachments/basemap1.png`](![](files/Matplotlib_attachments/basemap1.png)\
)# `* [:Cookbook/Matplotlib/Plotting values with masked arrays:Plotting values with masked arrays] - How to plot only selected values of an array, because some values are meaningless (detector malfunction), out of range, etc. etc.`\
# `* ["/Transformations"] - Using transformations to convert between different coordinate systems.`\
# `* TreeMap - classic treemap style plots`\
# `* ["/Legend"] - Adding a legend to your plot`\
# `* [:Cookbook/Matplotlib/HintonDiagrams:Hinton Diagrams] - A way of visualizing weight matrices`\
# ` . `[`![](files/Matplotlib_attachments/hinton-small.png`](![](files/Matplotlib_attachments/hinton-small.png)
)# 
# Pseudo Color Plots
# ==================
# 
# `* [:Cookbook/Matplotlib/Loading a colormap dynamically:Loading a colormap dynamically] - How to load a color map from a GMT (Generic Mapping Tools) file.`\
# `* [:Cookbook/Matplotlib/Show colormaps:Show colormaps] - Small script to display all of the Matplotlib colormaps, and an exampleshowing how to create a new one.`\
# `* [:Cookbook/Matplotlib/converting a matrix to a raster image:Converting a matrix to a raster image] - A replacement for scipy's imsave command`\
# `* [:Cookbook/Matplotlib/Gridding irregularly spaced `[`data:Gridding`](data:Gridding)` irregularly spaced data] - how to grid scattered data points in order to make a contour or image plot.`\
# `* [:Cookbook/Matplotlib/Plotting Images with Special Values:Plotting Images with Special Values] - how to plot an image with special values mapped to specific colors, e.g. missing values or data extrema`\
# ` . `[`![](files/Matplotlib_attachments/sentinel.png`](![](files/Matplotlib_attachments/sentinel.png)\
)# `* [:Cookbook/Matplotlib/ColormapTransformations:Transformations on Colormaps] - how to apply a function to the look up table of a colormap and turn it into another one.`
# 
# Typesetting
# ===========
# 
# `* [:Cookbook/Matplotlib/UsingTex:Using TeX] - formatting matplotlib text with LaTeX`\
# ` . `[`![](files/Matplotlib_attachments/tex_demo.png`](![](files/Matplotlib_attachments/tex_demo.png)\
)# `* [:Cookbook/Matplotlib/LaTeX Examples:LaTeX Examples] - Complete examples for generating publication quality figures using LaTeX.`
# 
# 3D Plotting
# ===========
# 
# ||**NOTE:** ***Experimental work has been going on to integrate 3D
# plotting functionality into matplotlib***. Please see the related
# [mplot3d
# documentation](http://matplotlib.sourceforge.net/mpl_toolkits/mplot3d/index.html?highlight=mplot3d)
# or take a look at [matplotlib
# gallery](http://matplotlib.sourceforge.net/gallery.html) for example 3D
# plots. For a more sophisticated 3D visualization and plotting interface,
# you can try [Mayavi](http://code.enthought.com/projects/mayavi/) which
# is actively maintained and features an 'mlab' interface similar to
# matplotlib's 'pylab'. || \* [:Cookbook/Matplotlib/mplot3D:3D plots] -
# Simple 3D plots using matplotlibs built-in 3D functions (which were
# originally provided by John Porter's mplot3d add-on module). .
# <![](files/Matplotlib_attachments/contourf3D.jpg> \* [:Cookbook/Matplotlib/VTK Integration:VTK
)# Integration] - How to import plots into VTK. . <![](files/Matplotlib_attachments/mpl_vtk.png>
)# 
# Embedding Plots in Apps
# =======================
# 
# `* [:Cookbook/Matplotlib/EmbeddingInWx:Embedding in WX] - Advice on how to embed matplotlib figures in `[`wxPython`](http://www.wxpython.org)` applications.`\
# `* `[`WxMpl`](http://agni.phys.iit.edu/~kmcivor/wxmpl)` - Python module for integrating matplotlib into wxPython GUIs.`\
# `* ["Cookbook/Matplotlib/ScrollingPlot"] - Demonstrates how to control a matplotlib plot embedded in a wxPython application with scrollbars.`\
# `* `[`Gael` `Varoquax's` `scientific` `GUI`
# `tutorial`](http://code.enthought.com/projects/traits/docs/html/tutorials/traits_ui_scientific_app.html)` - Includes an instructive example of embedding matplotlib in a `[`Traits`
# `GUI`](http://code.enthought.com/projects/traits_gui/)`.`\
# `* ["Cookbook/Matplotlib/PySide"] - Demonstrates how to display a matplotlib plot embedded in a PySide (Qt) application`
# 
# Misc
# ====
# 
# `* [:Cookbook/Matplotlib/LoadImage:Load and display an image] - shows a simple way to import a PNG image to a numpy array`\
# `* [:Cookbook/Matplotlib/Interactive Plotting:Interactive Plotting] - Adding mouse interaction to identify data annotations.`\
# `* [:Cookbook/Matplotlib/Matplotlib and Zope:Matplotlib and Zope] - How to use Matplotlib within the application server `[`Zope`](http://www.zope.org)`.`\
# `* [:Cookbook/Matplotlib/Qt with IPython and Designer:Qt with IPython and Designer] - How to design a GUI using Qt's Designer tool using Matplotlib widgets, and that can be interactively controlled from the IPython command line.`\
# `* [:Cookbook/Matplotlib/CompilingMatPlotLibOnSolaris10:Compiling Matplotlib on Solaris 10] - how to compile the thing on Solaris 10, using gcc/g++`\
# `* [:Cookbook/Matplotlib/Using MatPlotLib in a CGI script:Using MatPlotLib in a CGI script] - steps needed to be able to use matplotlib from a python cgi script`\
# `* `[`Making` `Dynamic` `Charts` `for` `your`
# `Webpage`](http://www.answermysearches.com/index.php/making-dynamic-charts-and-graphs-for-your-webpage/135/)` - Complete CGI script example.`\
# `* `[`matplotlib` `without`
# `GUI`](http://www.dalkescientific.com/writings/diary/archive/2005/04/23/matplotlib_without_gui.html)` by Andrew Dalke.`\
# `* `[`Andrew` `Straw's` `Apt`
# `Repository`](http://debs.astraw.com/dapper/)` - Bleeding edge deb packages for Debian, Ubuntu (also has packages for numpy/scipy etc.).`\
# `* [:Cookbook/Matplotlib/AdjustingImageSize:Adjusting Image Size] - a brief discussion of how to adjust the size of figures -- for printing, web, etc.`\
# `* [:Cookbook/Matplotlib/DeletingAnExistingDataSeries:Deleting An Existing Data Series] - a quick example showing how to remove one data series from an already existing plot.`\
# `* [:Cookbook/Matplotlib/Django:Embedding in Django] - example on how to use matplotlib with Django.`\
# `* `[`timeseries`
# `scikit`](http://pytseries.sourceforge.net)``  - The documentation contains a section on plotting `TimeSeries` objects using matplotlib ``\
# `* /TreeMap - A compact way of showing weighted tree information.`\
# `* [:Cookbook/Matplotlib/Multiple Subplots with One Axis Label:Multiple Subplots with One Axis Label] - how to use one centered label to annotate several subplots`\
# `* `[`Multiple`
# `Y-axis`](http://www.nabble.com/Multiple-Y-axis-td10734643.html)` - How to plot different variables on the same plot but different Y-Axis (one left and one right)`\
# `* `[`Creating` `video` `of` `3D` `graph` `plotting` `using`
# `matplotlib` `and`
# `mencoder`](http://debtechandstuff.blogspot.com/2009/10/creating-video-of-3d-graph-plotting.html)` by Ilya Zakreuski`
# 
# * * * * *
# 
# `. CategoryCookbookMatplotlib CategoryCookbook`
# 