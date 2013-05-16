TableOfContents

Finding the minimum point in the convex hull of a finite set of points
======================================================================

Based on the work of Philip Wolf [1] and the recursive algorithm of
Kazuyuki Sekitani and Yoshitsugu Yamamoto [2].

The algorithm in [2] has 3 epsilon to avoid comparison problems in three
parts of the algorithm. The code below has few changes and only one
epsilon. The aim of the change is to avoid infinite loops.

Code
====




.. code-block:: python

    
    from numpy import array, matrix, sin, sqrt, dot, cos, ix_, zeros, concatenate, a
    bs, log10, exp, ones
    from numpy.linalg import norm
    
    from mpmath import mpf, mp
    mp.dps=80
    
    def find_min_point(P):
    #    print "Calling find_min with P: ", P
    
        if len(P) == 1:
            return P[0]
    
        eps = mpf(10)**-40
    
        P = [array([mpf(i) for i in p]) for p in P]
        
        # Step 0. Choose a point from C(P)
        x  = P[array([dot(p,p) for p in P]).argmin()]
    
        while True:
    
            # Step 1. \alpha_k := min{x_{k-1}^T p | p \in P}
            p_alpha = P[array([dot(x,p) for p in P]).argmin()]
    
            if dot(x,x-p_alpha) < eps:
                return array([float(i) for i in x]) 
            
            Pk = [p for p in P if abs(dot(x,p-p_alpha)) < eps]
    
            # Step 2. P_k := { p | p \in P and x_{k-1}^T p = \alpha_k}
            P_Pk = [p for p in P if not array([(p == q).all() for q in Pk]).any()]
    
            if len(Pk) == len(P):
                return array([float(i) for i in x]) 
    
            y = find_min_point(Pk)
    
    
            p_beta = P_Pk[array([dot(y,p) for p in P_Pk]).argmin()]
            
            if dot(y,y-p_beta) < eps:
                return array([float(i) for i in y]) 
    
            
            # Step 4. 
            P_aux = [p for p in P_Pk if (dot(y-x,y-p)>eps) and (dot(x,y-p)!=0)]
            p_lambda = P_aux[array([dot(y,y-p)/dot(x,y-p) for p in P_aux]).argmin()]
    
            lam = dot(x,p_lambda-y) / dot(y-x,y-p_lambda)
    
            x += lam * (y-x)
    
    
    
    if __name__ == '__main__':
        print find_min_point( [array([ -4.83907292e+00,   2.22438863e+04,  -2.674967
    63e+04]), array([   9.71147604, -351.46404195, -292.18064276]), array([  4.60452
    808e+00,   1.07020174e+05,  -1.25310230e+05]), array([  2.16080134e+00,   5.1201
    9937e+04,  -5.96167833e+04]), array([  2.65472146e+00,   6.70546443e+04,  -7.716
    19656e+04]), array([  1.55775358e+00,  -1.34347516e+05,   1.53209265e+05]), arra
    y([   13.22464295,  1869.01251292, -2137.61850989])])
    
    
        print find_min_point( [array([ -4.83907292e+00,   2.22438863e+04,  -2.674967
    63e+04]), array([   9.71147604, -351.46404195, -292.18064276]), array([  4.60452
    808e+00,   1.07020174e+05,  -1.25310230e+05]), array([  2.16080134e+00,   5.1201
    9937e+04,  -5.96167833e+04]), array([  2.65472146e+00,   6.70546443e+04,  -7.716
    19656e+04]), array([  1.55775358e+00,  -1.34347516e+05,   1.53209265e+05]), arra
    y([   13.22464295,  1869.01251292, -2137.61850989]), array([ 12273.18670123,  -1
    233.32015854,  61690.10864825])])
    



References
==========

| ``1. ``\ ```Finding`` ``the`` ``nearest`` ``point`` ``in`` ``A``
``polytope`` <http://www.springerlink.com/content/hw0l2n1271260604/>`__
| ``2. ``\ ```A`` ``recursive`` ``algorithm`` ``for`` ``finding``
``the`` ``minimum`` ``norm`` ``point`` ``in`` ``a`` ``polytope`` ``and``
``a`` ``pair`` ``of`` ``closest`` ``points`` ``in`` ``two``
``polytopes`` <http://www.springerlink.com/content/j25702174115q68x/>`__

