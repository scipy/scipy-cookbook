# <markdowncell>

# This page demonstrates two functions in scipy.signal for generating
# frequency-swept signals: \`chirp\` and \`sweep\_poly\`.
# 
# Some of these require SciPy 0.8.
# 
# To run the code samples, you will need the following imports:
# 
# <codecell>


    import numpy as np
    from scipy.signal import chirp, sweep_poly

# <markdowncell>

# Linear Chirp
# ============
# 
# Sample code:
# 
# <codecell>


    t = np.linspace(0, 10, 5001)
    w = chirp(t, f0=12.5, f1=2.5, t1=10, method='linear')

# <markdowncell>

# <![](files/FrequencySweptDemo_attachments/chirp_linear.png>
)# 
# Quadratic Chirp
# ===============
# 
# Sample code:
# 
# <codecell>


    t = np.linspace(0, 10, 5001)
    w = chirp(t, f0=12.5, f1=2.5, t1=10, method='quadratic')

# <markdowncell>

# <![](files/FrequencySweptDemo_attachments/chirp_quadratic.png>
)# 
# Sample code using \`vertex\_zero\`:
# 
# <codecell>


    t = np.linspace(0, 10, 5001)
    w = chirp(t, f0=12.5, f1=2.5, t1=10, method='quadratic', vertex_zero=False)

# <markdowncell>

# <![](files/FrequencySweptDemo_attachments/chirp_quadratic_v0false.png>
)# 
# Logarithmic Chirp
# =================
# 
# Sample code:
# 
# <codecell>


    t = np.linspace(0, 10, 5001)
    w = chirp(t, f0=12.5, f1=2.5, t1=10, method='logarithmic')

# <markdowncell>

# <![](files/FrequencySweptDemo_attachments/chirp_logarithmic.png>
)# 
# Hyperbolic Chirp
# ================
# 
# Sample code:
# 
# <codecell>


    t = np.linspace(0, 10, 5001)
    w = chirp(t, f0=12.5, f1=2.5, t1=10, method='hyperbolic')

# <markdowncell>

# <![](files/FrequencySweptDemo_attachments/chirp_hyperbolic.png>
)# 
# Sweep Poly
# ==========
# 
# Sample code:
# 
# <codecell>


    p = poly1d([0.05, -0.75, 2.5, 5.0])
    t = np.linspace(0, 10, 5001)
    w = sweep_poly(t, p)

# <markdowncell>

# <![](files/FrequencySweptDemo_attachments/sweep_poly.png>
)# 
# The script that generated the plots is here:
# 
# <![](files/FrequencySweptDemo_attachments/chirp_plot.py>
)# 