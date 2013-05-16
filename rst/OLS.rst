OLS
---

OLS is an abbreviation for ordinary least squares.

The class estimates a multi-variate regression model and provides a
variety of fit-statistics. To see the class in action download the
`ols.py <.. image:: OLS_attachments/ols.0.2.py>`__ file and run it (python ols.py). This
will estimate a multi-variate regression using simulated data and
provide output. It will also provide output from R to validate the
results if you have rpy installed (http://rpy.sourceforge.net/).

To import the class:



.. code-block:: python

    #!python
    import ols
    



After importing the class you can estimate a model by passing data to it
as follows



.. code-block:: python

    #!python
    mymodel = ols.ols(y,x,y_varnm,x_varnm)
    



where y is an array with data for the dependent variable, x contains the
independent variables, y\_varnm, is a string with the variable label for
the dependent variable, and x\_varnm is a list of variable labels for
the independent variables. Note: An intercept term and variable label is
automatically added to the model.

Example Usage
-------------




.. code-block:: python

    #!python
    >>> import ols
    >>> from numpy.random import randn
    >>> data = randn(100,5)
    >>> y = data[:,0]
    >>> x = data[:,1:]
    >>> mymodel = ols.ols(y,x,'y',['x1','x2','x3','x4'])
    >>> mymodel.p               # return coefficient p-values
    array([ 0.31883448,  0.7450663 ,  0.95372471,  0.97437927,  0.09993078])
    >>> mymodel.summary()       # print results
    ==============================================================================
    Dependent Variable: y
    Method: Least Squares
    Date: Thu, 28 Feb 2008
    Time: 22:32:24
    # obs:             100
    # variables:         5
    ==============================================================================
    variable     coefficient     std. Error      t-statistic     prob.
    ==============================================================================
    const           0.107348      0.107121      1.002113      0.318834
    x1             -0.037116      0.113819     -0.326100      0.745066
    x2              0.006657      0.114407      0.058183      0.953725
    x3              0.003617      0.112318      0.032201      0.974379
    x4              0.186022      0.111967      1.661396      0.099931
    ==============================================================================
    Models stats                         Residual stats
    ==============================================================================
    R-squared             0.033047         Durbin-Watson stat   2.012949
    Adjusted R-squared   -0.007667         Omnibus stat         5.664393
    F-statistic           0.811684         Prob(Omnibus stat)   0.058883
    Prob (F-statistic)    0.520770	       JB stat              6.109005
    Log likelihood       -145.182795       Prob(JB)             0.047146
    AIC criterion         3.003656         Skew                 0.327103
    BIC criterion         3.133914         Kurtosis             4.018910
    ==============================================================================
    



Note
----

Library function
`numpy.linalg.lstsq() <http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.lstsq.html>`__
performs basic OLS estimation.

