Data Frames
===========

The .. image:: DataFrame_attachments/DataFrame.py class, posted by Andrew Straw on the
scipy-user mailing
list[\ http://thread.gmane.org/gmane.comp.python.scientific.user/6860,
original link], is an extremely useful tool for using alphanumerical
tabular data, as often found in databases. Some data which might be
ingested into a data frame could be: \|\| **ID** \|\| **LOCATION** \|\|
**VAL\_1** \|\| **VAL\_2** \|\| \|\| 01 \|\| Somewhere \|\| 0.1 \|\| 0.6
\|\| \|\| 02 \|\| Somewhere Else \|\| 0.2 \|\| 0.5 \|\| \|\| 03 \|\|
Elsewhere \|\| 0.3 \|\| 0.4 \|\|

The .. image:: DataFrame_attachments/DataFrame.py class can be populated from data from a CSV
file (comman-separated values). In its current implementation, these
files are read with Python's own CSV module, which allows for a great
deal of customisation.

Example Usage
-------------

A sample file CSV file from Access2000 is in .. image:: DataFrame_attachments/CSVSample.csv .
We first import the module:



.. code-block:: python

    #!python
    import DataFrame
    



and read the file in using our desired CVS dialect:



.. code-block:: python

    #!python
    df=DataFrame.read_csv ("CSVSample.csv",dialect=DataFrame.access2000)
    



(note that the dialect is actually defined in the DataFrame class). It
is often useful to filter the data according to some criterion.

Compatibility with Python 2.6 and above
---------------------------------------

Starting with Python 2.6, the sets module is deprecated, in order to get
rid of the warning, replace



.. code-block:: python

    #!python
     imports sets
    



with



.. code-block:: python

    #!python
     try:
         set
     except NameError:
         from sets import Set as set
    



Then replace all instances of sets.Set() with set().

``CategoryCookbook CategoryCookbook``

