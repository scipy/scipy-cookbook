The attached file ( .. image:: RANSAC_attachments/ransac.py ) implements the `RANSAC
algorithm <http://en.wikipedia.org/wiki/RANSAC>`__. An example image:

.. image:: RANSAC_attachments/ransac.png

To run the file, save it to your computer, start IPython



.. code-block:: python

    ipython -wthread
    



Import the module and run the test program



.. code-block:: python

    import ransac 
    ransac.test()
    



To use the module you need to create a model class with two methods



.. code-block:: python

    def fit(self, data):
      """Given the data fit the data with your model and return the model (a vector)
    """
    def get_error(self, data, model):
      """Given a set of data and a model, what is the error of using this model to e
    stimate the data """
    



An example of such model is the class LinearLeastSquaresModel as seen
the file source (below)

.. image:: RANSAC_attachments/ransac.py

--------------

``CategoryCookbook``

