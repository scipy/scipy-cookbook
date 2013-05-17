# <markdowncell>

# The attached file ( <![](files/RANSAC_attachments/ransac.py> ) implements the [RANSAC
)# algorithm](http://en.wikipedia.org/wiki/RANSAC). An example image:
# 
# <![](files/RANSAC_attachments/ransac.png>
)# 
# To run the file, save it to your computer, start IPython
# 
# <codecell>


ipython -wthread

# <markdowncell>

# Import the module and run the test program
# 
# <codecell>


import ransac 
ransac.test()

# <markdowncell>

# To use the module you need to create a model class with two methods
# 
# <codecell>


def fit(self, data):
  """Given the data fit the data with your model and return the model (a vector)"""
def get_error(self, data, model):
  """Given a set of data and a model, what is the error of using this model to estimate the data """

# <markdowncell>

# An example of such model is the class LinearLeastSquaresModel as seen
# the file source (below)
# 
# ![](files/RANSAC_attachments/ransac.py
# 
# * * * * *
# 
# `CategoryCookbook`
# 