**Recipies and FAQ for the Timeseries Scikit**

TableOfContents(4)

NOTE: The official documentation and important remarks from the
developers can be found at the `timseries scikit sourceforge
page <http://pytseries.sourceforge.net>`__.

FAQ
===

General threads
---------------

| ``1. time series analysis - ``\ ```http://article.gmane.org/gmane.comp.python.scientific.user/13949`` <http://article.gmane.org/gmane.comp.python.scientific.user/13949>`__
| ``1. time series: Python vs. R URL missing!!!``
| ``1. roadmap/plans for timeseries package -  ``\ ```http://permalink.gmane.org/gmane.comp.python.scientific.user/14599`` <http://permalink.gmane.org/gmane.comp.python.scientific.user/14599>`__

Reading data and creating timeseries objects
--------------------------------------------

masking NoData values
~~~~~~~~~~~~~~~~~~~~~

Question
^^^^^^^^

In my original data nodata values are marked with "-999". How can I
import the data or create the time series and exclude these no data
points from further processing? (flagging no data in timeseries -
http://permalink.gmane.org/gmane.comp.python.scientific.user/14455)

Answer
^^^^^^

``* use masked_where from maskedarray``



.. code-block:: python

    #!python
    myvalues_ts_hourly = masked_where(myvalues_ts_hourly , -999)
    



``* Use indexing``



.. code-block:: python

    #!python
    myvalues_ts_hourly[myvalues_ts_hourly==-999] = M.masked
    



More extensive answer
^^^^^^^^^^^^^^^^^^^^^

**\* START SAMPLE DATA (tmp.txt) \***



.. code-block:: python

    date;hour_of_day;value
    01.02.2004;1;247
    01.02.2004;2;889
    01.02.2004;3;914
    01.02.2004;4;292
    01.02.2004;5;183
    01.02.2004;6;251
    01.02.2004;7;953
    01.02.2004;8;156
    01.02.2004;9;991
    01.02.2004;10;557
    01.02.2004;11;581
    01.02.2004;12;354
    01.02.2004;13;485
    01.02.2004;14;655
    01.02.2004;15;-999
    01.02.2004;16;-999
    01.02.2004;17;-999
    01.02.2004;18;744
    01.02.2004;19;445
    01.02.2004;20;374
    01.02.2004;21;168
    01.02.2004;22;995
    01.02.2004;23;943
    01.02.2004;24;326
    02.02.2004;1;83.98
    02.02.2004;2;302.26
    02.02.2004;3;310.76
    02.02.2004;4;-999
    02.02.2004;5;62.22
    02.02.2004;6;85.34
    02.02.2004;7;324.02
    02.02.2004;8;53.04
    02.02.2004;9;336.94
    02.02.2004;10;189.38
    02.02.2004;11;197.54
    02.02.2004;12;120.36
    02.02.2004;13;164.9
    02.02.2004;14;222.7
    02.02.2004;15;34.74
    02.02.2004;16;85.34
    02.02.2004;17;53.04
    02.02.2004;18;252.96
    02.02.2004;19;151.3
    02.02.2004;20;-999
    02.02.2004;21;57.12
    02.02.2004;22;338.3
    02.02.2004;23;320.62
    02.02.2004;24;110.84}}}
    
    '''* END SAMPLE DATA *'''
    
    {{{
    #!python
    import numpy as N
    import maskedarray as M
    import timeseries as ts
    data = N.loadtxt("tmp.txt", dtype='|S10', skiprows=2)
    dates = ts.date_array([ts.Date(freq='H',string="%s %s:00" %
    (d[0],int(d[1])-1))
                           for d in data],
                          freq='H')
    series = ts.time_series(data[:,-1].astype(N.float_),
                            dates,
                            mask=(data[:,-1]=='-999'))
    



frequencies
~~~~~~~~~~~

Question
^^^^^^^^

Is there a example data set for at least one year on a high temporal
resolution: 15min or at least 1h. Having such a common data set one
could set up tutorials examples and debug or ask questions easier
because all will have the same (non-confidetial) data on the disk.

Answer
^^^^^^

For hours, you have the 'hourly' frequency. For 15min, you have the
'minutely' frequency, from which you can select every other 15th point.

(cf. Re: roadmap/plans for timeseries package -
http://permalink.gmane.org/gmane.comp.python.scientific.user/1459)

hour of the day
~~~~~~~~~~~~~~~

(cf.: assignment of hours of day in time series -
http://permalink.gmane.org/gmane.comp.python.scientific.user/14597) When
exchanging agrregated data sets (e.g. with hourly frequency) the data is
often presented as follows: desired report output



.. code-block:: python

     date; hour_of_day; value
     1-Feb-2004;1:00;247
     1-Feb-2004;2:00;889
     1-Feb-2004;3:00;914
     1-Feb-2004;4:00;292
     1-Feb-2004;5:00;183
     1-Feb-2004;6:00;251
     1-Feb-2004;7:00;953
     1-Feb-2004;8:00;156
     1-Feb-2004;9:00;991
     1-Feb-2004;10:00;557
     1-Feb-2004;11:00;581
     1-Feb-2004;12:00;354
     1-Feb-2004;13:00;485
     1-Feb-2004;14:00;655
     1-Feb-2004;15:00;862
     1-Feb-2004;16:00;399
     1-Feb-2004;17:00;598
     1-Feb-2004;18:00;744
     1-Feb-2004;19:00;445
     1-Feb-2004;20:00;374
     1-Feb-2004;21:00;168
     1-Feb-2004;22:00;995
     1-Feb-2004;23:00;943
     1-Feb-2004;24:00;326
    



This formatting may be the result of some logging devices which for
instance record 5 minutes averaged values which have been taken with a
device using a sample rate of 16 sec. As well, syntetically generated
data sets which have been created by scientifc models or from remote
sensing information can have such a format. When creating a timeseries
object the start hour should be set to zero (0) internally to achieve a
correct assignment of the hours (01:00 h is the end of the period 00:00
h - 01:00 h => data for this period starts at 00:00 h). For the output
one can be customized as shown below in the answer. The python built-in
module datetime can help here.

Question
^^^^^^^^

I have hourly measurements where hour 1 represents the end of the period
0:00-1:00, 2 the end of the period 1:00-2:00, ... , 24 the end of the
period 23:00 to 24:00.

When I plot these hourly time series from February to November the curve
is continued into December because of that matter. time series then
assumes that the value for hour 0:00 of dec, 01 is 0 which then leads do
a wrong plotting behaviour.

I want to achieve that hour 24 is accounted as the last measurement
period of a day and not as the first measurement of the next day (like
0:00).

Answer
^^^^^^

Since the time "24:00" doesn't actually exist (as far as I am aware
anyway), you will have to rely on somewhat of a hack to get your desired
output. Try this:



.. code-block:: python

    #!python
    import timeseries as ts
    series = ts.time_series(range(400, 430), start_date=ts.now('hourly'))
    hours = ts.time_series(series.hour + 1, dates=series.dates)
    hour_fmtfunc = lambda x : '%i:00' % x
    ts.Report(hours, series, datefmt='%d-%b-%Y', delim='  ', fmtfunc=[None hour_fmtf
    unc,])()
    







.. code-block:: python

    date time; value
    06-Jan-2008  23:00;  400
    06-Jan-2008  24:00;  401
    07-Jan-2008   1:00;  402
    07-Jan-2008   2:00;  403
    07-Jan-2008   3:00;  404
    07-Jan-2008   4:00;  405
    07-Jan-2008   5:00;  406
    07-Jan-2008   6:00;  407
    07-Jan-2008   7:00;  408
    07-Jan-2008   8:00;  409
    07-Jan-2008   9:00;  410
    07-Jan-2008  10:00;  411
    07-Jan-2008  11:00;  412
    07-Jan-2008  12:00;  413
    07-Jan-2008  13:00;  414
    07-Jan-2008  14:00;  415
    07-Jan-2008  15:00;  416
    07-Jan-2008  16:00;  417
    07-Jan-2008  17:00;  418
    07-Jan-2008  18:00;  419
    07-Jan-2008  19:00;  420
    07-Jan-2008  20:00;  421
    07-Jan-2008  21:00;  422
    07-Jan-2008  22:00;  423
    07-Jan-2008  23:00;  424
    07-Jan-2008  24:00;  425
    08-Jan-2008   1:00;  426
    08-Jan-2008   2:00;  427
    08-Jan-2008   3:00;  428
    08-Jan-2008   4:00;  429
    



Manipulations & Operations with time series

use the datetime information of the time series
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

(Re: roadmap/plans for timeseries package -
http://permalink.gmane.org/gmane.comp.python.scientific.user/14598) A
example:

Question
^^^^^^^^

One has to get rainfall intensity during early morning hours. For such a
filter the information on the corresponding hours are neccessary.

Answer
^^^^^^




.. code-block:: python

    import timeseries as ts
    data = ts.time_series(range(100), start_date=ts.today('hourly'))
    hours = data.hour
    filtered_data = data[(hours < 7) & (hours > 3)]
    filtered_data
    timeseries([80  6  7  8 30 31 32 54 55 56 78 79],
    dates = [07-Jan-2008 04:00 07-Jan-2008 05:00 07-Jan-2008 06:00
    08-Jan-2008 04:00 08-Jan-2008 05:00 08-Jan-2008 06:00 09-Jan-2008 04:00
    09-Jan-2008 05:00 09-Jan-2008 06:00 10-Jan-2008 04:00 10-Jan-2008 05:00
    10-Jan-2008 06:00],
              freq  = H)
    



using the result of time series operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Question
^^^^^^^^

How can one save the read the result of time series operations into a
array?

For instance, if I convert data in an hourly frequency to daily averages
how to I read the daily averages into a array for further processing?

when I print out my daily timeseries converted from hourly data I get
something like this:



.. code-block:: python

    #!python
    In: myvalues_ts_daily
    Out:
    timeseries([  1.4   89.4  3.5 ...,  11.5  1.6
         0.        ],
                dates = [01-Dec-2006 01-Feb-1995 ...],
                freq  = D)
    



What I would like is an array with just the values of the daily averages
. Additional a report-like array output with the format day value



.. code-block:: python

    1   3
    2   11
    



Answer
^^^^^^

> For instance, if I convert data in an hourly frequency to daily
averages > > how to I read the daily averages into a array for further
processing?

``1. possibility #1: use the keyword func while converting.``



.. code-block:: python

    
     1. possibility #2:
    If you don't use the keyword func, you end up with a 2d array, each row being  a
     day, each column an hour. Just use maskedarray.mean on each row avgdata = conve
    rt(data,'D').mean(-1)
    
    If you only want the values, use the .series attribute, it will give you a  view
     of the array as a MaskedArray.
    
    == Plotting ==
    Word of caution... the timeseries plotting stuff does not currently support freq
    uencies higher than daily (eg. hourly, minutely, etc...). Support for these freq
    uencies could be added without too much trouble, but just haven't got around to 
    it yet. (Cf. Re: roadmap/plans for timeseries package - http://permalink.gmane.o
    rg/gmane.comp.python.scientific.user/14598)
    
    = About this page =
    == Source ==
     * Most information presented here has been compiled from discussions at the sci
    py mailing list.
    == todo ==
     * Use one data set consistently for the examples
     * offer the code for download
    


