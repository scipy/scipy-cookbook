Fortran I/O Formats
===================

Files written by Fortran programs can be written using one of two
formats: formatted or unformatted. Formatted files are written in
human-readable formats and it should be possible to load them using
numpy.fromfile. Unformatted files are written using a binary format that
is unspecified by the Fortran standard. In practice, most
compilers/runtimes use a record-based format with an integer header
consisting of the length of the record in bytes, then the record itself
followed by an integer footer with the length of the preceeding in
bytes.

Given that the precision and endian-ness of the headers and the data are
unspecified, there are a large number of possible combinations that may
be seen in the wild. The [:Cookbook/FortranIO/FortranFile:FortranFile]
class can deal with a great many of these.

The following is an example of how to read a particular unformatted
output file. Note the presence of the 'i4' elements of the dtype
representing the header and the footer.

Reading FORTRAN "unformatted IO" files
======================================

Lots of scientific code is written in FORTRAN. One of the most
convenient file formats to create in FORTRAN is the so-called
"`unformatted binary
file <http://local.wasp.uwa.edu.au/~pbourke/dataformats/fortran/>`__\ ".
These files have all the disadvantages of raw binary IO - no metadata,
data depends on host endianness, floating-point representation, and
possibly word size - but are not simply raw binary. They are organized
into "records", which are padded with size information. Nevertheless,
one does encounter such files from time to time. No prewritten code
appears to be available to read them in numpy/scipy, but it can be done
with relative ease using numpy's record arrays:



.. code-block:: python

    >>> A = N.fromfile("/tmp/tmp_i7j_a/resid2.tmp",
    ...   N.dtype([('pad1','i4'),
    ...    ('TOA','f8'),
    ...    ('resid_p','f8'),
    ...    ('resid_s','f8'),
    ...    ('orb_p','f8'),
    ...    ('f','f8'),
    ...    ('wt','f8'),
    ...    ('sig','f8'),
    ...    ('preres_s','f8'),
    ...    ('pad3','i8'),
    ...    ('pad2','i4')]))
    



This example is designed to read `a
file <http://www.atnf.csiro.au/research/pulsar/tempo/ref_man_sections/output.txt>`__
output by `TEMPO <http://www.atnf.csiro.au/research/pulsar/tempo/>`__.
Most of the fields, "TOA" up to "preres\_s", are fields that are present
and of interest in the file. The field "pad3" is either an undocumented
addition to the file format or some kind of padding (it is always zero
in my test file). The FORTRAN unformatted I/O adds the fields "pad1" and
"pad2". Each should contain the length, in bytes, of each record (so the
presence of the extra "pad3" field could be deduced). This code ignores
tProxy-Connection: keep-alive Cache-Con

