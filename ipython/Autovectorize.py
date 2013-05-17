# <markdowncell>

# Autovectorization
# =================
# 
# There are instances where it is very convenient to have a function
# defined in the language of scalars that can operate on arrays.
# [numpy.vectorize](http://docs.scipy.org/doc/numpy/reference/generated/numpy.vectorize.html)
# provides such a conversion.
# 
# In simplier language: This function basically makes a functions which
# calculate single values (e. g. math.sin) operate on array.
# 
# Some links and threads on this:
# 
# `* optimising single value functions for array calculations -  `[`http://article.gmane.org/gmane.comp.python.numeric.general/26543`](http://article.gmane.org/gmane.comp.python.numeric.general/26543)\
# `* vectorized function inside a class -  `[`http://article.gmane.org/gmane.comp.python.numeric.general/16438`](http://article.gmane.org/gmane.comp.python.numeric.general/16438)\
# `* numpy.vectorize performance - `[`http://article.gmane.org/gmane.comp.python.numeric.general/6867`](http://article.gmane.org/gmane.comp.python.numeric.general/6867)\
# `* vectorize() - `[`http://www.scipy.org/Numpy_Example_List_With_Doc#head-fbff061fdb843209707a8fa537d9b24b6a91245e`](http://www.scipy.org/Numpy_Example_List_With_Doc#head-fbff061fdb843209707a8fa537d9b24b6a91245e)\
# `* NumPy: vectorization - `[`http://folk.uio.no/hpl/PyUiT/PyUiT-split/slide218.html`](http://folk.uio.no/hpl/PyUiT/PyUiT-split/slide218.html)\
# `* vectorizing loops - `[`http://article.gmane.org/gmane.comp.python.numeric.general/17266`](http://article.gmane.org/gmane.comp.python.numeric.general/17266)
# 
# See also
# --------
# 
# `* ["SciPyPackages/NumExpr"]`
# 