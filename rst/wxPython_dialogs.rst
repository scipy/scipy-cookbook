Overview
========

Something I enjoy in matlab is the ease in which simple file selector
dialogs, and status bars can be made. Now that I use nothing but scipy,
I have wanted to have simliar functions for my scipy code. Thanks to the
great reference `"wxPython in
Action" <http://www.manning.com/rappin/>`__ I have learned some of the
basics again, with the promise of making very fancy GUIs if I ever find
the urge! (Check out the sample chapters, they have given the entire
section on making dialogs, which is how I initially got started with
wxPython).

File Selector Dialog
====================

I often write simple translation scripts that convert some data, into
another form. I like to use these for a series of data, and share them
with some coworkers who do not program. the wxPython FileSelector
function comes to the rescue.



.. code-block:: python

    import wx
    
    # setup the GUI main loop
    app = wx.App()
    
    filename = wx.FileSelector()
    



With this basic code, filename contains a string pathname (may be
unicode depending on your installation of wxPython, more on this latter)
of the selected file.

Some of the spiffing up will include using the current directory the
script was started in, we do this easily



.. code-block:: python

    import wx
    import os
    
    # setup the GUI main loop
    app = wx.App()
    
    filename = wx.FileSelector(default_path=os.getcwd())
    



If one runs such script repeatedly, it might be a good idea to do some
basic clean-up after each run.



.. code-block:: python

    # ...
    app.Destroy()
    



To be continued . . .

Status Bar
==========

Coming soon . . .

