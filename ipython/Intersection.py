# <markdowncell>

# Find the points at which two given functions intersect
# ------------------------------------------------------
# 
# Consider the example of finding the intersection of a polynomial and a
# line:
# 
# <codecell>


y1=x1^2
y2=x2+1

# <markdowncell>

# 
# 
# <codecell>


from scipy.optimize import fsolve

import numpy as np

def f(xy):
   x, y = xy
   z = np.array([y - x**2, y - x - 1.0])
   return z

fsolve(f, [1.0, 2.0])

# <markdowncell>

# The result of this should be:
# 
# <codecell>


array([ 1.61803399,  2.61803399])

# <markdowncell>

# See also:
# <http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fsolve.html#scipy.optimize.fsolve>
# 