Reading SPE file from CCD camera
================================

Some `charge-coupled device
(CCD) <http://en.wikipedia.org/wiki/CCD_camera>`__ cameras (Princeton
and like) produce SPE files. This page suggests how to read such binary
files with Numpy but the code is not robust. The following code is only
able to read files having the same format as the example,
'lampe\_dt.spe' (unfortuanetly the only SPE file on the wiki).

Loading SPE file with numpy
---------------------------

Only Numpy is required for loading SPE file, the result will be an array
made of colors. The image size is at position 42 and 656 and the data at
4100. There are then many other data in a SPE file header, one must be
the data type (you are welcome to edit this page if you know where).
Finally note that the image is always made of colors coded on unsigned
integer of 16 bits but it might not be the case in your input file.



.. code-block:: python

    #!python numbers=disabled
    # read_spe.py
    import numpy as N
    
    class File(object):
    
        def __init__(self, fname):
            self._fid = open(fname, 'rb')
            self._load_size()
    
        def _load_size(self):
            self._xdim = N.int64(self.read_at(42, 1, N.int16)[0])
            self._ydim = N.int64(self.read_at(656, 1, N.int16)[0])
    
        def _load_date_time(self):
            rawdate = self.read_at(20, 9, N.int8)
            rawtime = self.read_at(172, 6, N.int8)
            strdate = ''
            for ch in rawdate :
                strdate += chr(ch)
            for ch in rawtime:
                strdate += chr(ch)
            self._date_time = time.strptime(strdate,"%d%b%Y%H%M%S")
    
        def get_size(self):
            return (self._xdim, self._ydim)
            
        def read_at(self, pos, size, ntype):
            self._fid.seek(pos)
            return N.fromfile(self._fid, ntype, size)
    
        def load_img(self):
            img = self.read_at(4100, self._xdim * self._ydim, N.uint16)
            return img.reshape((self._ydim, self._xdim))
    
        def close(self):
            self._fid.close()
    
    def load(fname):
        fid = File(fname)
        img = fid.load_img()
        fid.close()
        return img
    
    if __name__ == "__main__":
        import sys
        img = load(sys.argv[-1])
    



Viewing the image with matplotlib and ipython
---------------------------------------------

The 'read\_spe.py' script from above and the 'lampe\_dt.spe' example are
provided in the archive .. image:: Reading_SPE_files_attachments/read_spe.zip. Once decompresesed, you
can then start ipython in the directory where the script lives:



.. code-block:: python

    ipython -pylab read_spe.py lampe_dt.spe
    



The following first line will show the image in a new window. The second
line will change the colormap (try 'help(pylab.colormaps)' for listing
them).



.. code-block:: python

    #!python
    >>> pylab.imshow(img)
    >>> pylab.hot()
    



\|\|<:>.. image:: Reading_SPE_files_attachments/lampe_dt.png\|\|

