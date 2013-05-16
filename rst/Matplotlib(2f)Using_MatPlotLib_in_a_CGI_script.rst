Trying to use matplotlib in a python CGI script naïvely will most likely
result in the following error:



.. code-block:: python

    ...
    352, in _get_configdir
    raise RuntimeError("'%s' is not a writable dir; you must set
    environment variable HOME to be a writable dir "%h)
    RuntimeError: '<WebServer DocumentRoot>' is not a writable dir; you must set
    environment variable HOME to be a writable dir
    



Matplotlib needs the environment variable HOME to point to a writable
directory. One way to accomplish this is to set this environment
variable from within the CGI script on runtime (another way would be to
modify the file but that would be not as portable). The following
template can be used for a cgi that uses matplotlib to create a png
image:



.. code-block:: python

    #!/usr/bin/python
    import os,sys
    import cgi
    import cgitb; cgitb.enable()
    
    # set HOME environment variable to a directory the httpd server can write to
    os.environ[ 'HOME' ] = '/tmp/'
    
    import matplotlib
    # chose a non-GUI backend
    matplotlib.use( 'Agg' )
    
    import pylab
    
    #Deals with inputing data into python from the html form
    form = cgi.FieldStorage()
    
    # construct your plot
    pylab.plot([1,2,3])
    
    print "Content-Type: image/png\n"
    
    # save the plot as a png and output directly to webserver
    pylab.savefig( sys.stdout, format='png' )
    



This image can then be accessed with a URL such as:
http://localhost/showpng.py

As documented,some backends will not allow the output to be sent to
sys.stdout. It is possible to replace the last line with the following
to work around this:



.. code-block:: python

    pylab.savefig( "tempfile.png", format='png' )
    import shutil
    shutil.copyfileobj(open("tempfile.png",'rb'), sys.stdout)
    



(Of course it is necessary to create and delete proper temp files to use
this in production.)

--------------

CategoryCookbookMatplotlib

