Modeling a Zombie Apocalypse
----------------------------

This example demonstrates how to solve a system of first order ODEs
using SciPy. Note that a Nth order equation can also be solved using
SciPy by transforming it into `a system of first order
equations <http://en.wikipedia.org/wiki/Ordinary_differential_equation#Reduction_to_a_first_order_system>`__.
In a this lighthearted example, a system of ODEs can be used to model a
"zombie invasion", using the equations specified in `Munz et al.
2009 <http://mysite.science.uottawa.ca/rsmith43/Zombies.pdf>`__.

The system is given as:



.. code-block:: python

    dS/dt = P - B*S*Z - d*S
    dZ/dt = B*S*Z + G*R - A*S*Z
    dR/dt = d*S + A*S*Z - G*R
    







.. code-block:: python

    with the following notations:
    
    *  S: the number of susceptible victims
    *  Z: the number of zombies
    *  R: the number of people "killed"
    *  P: the population birth rate
    *  d: the chance of a natural death
    *  B: the chance the "zombie disease" is transmitted (an alive person becomes a 
    zombie)
    *  G: the chance a dead person is resurrected into a zombie
    *  A: the chance a zombie is totally destroyed
    



This involves solving a system of first order ODEs given by: d\ **y**/dt
= **f**\ (**y**, t)

Where **y** = [S, Z, R].

The code used to solve this system is below:



.. code-block:: python

    # zombie apocalypse modeling
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.integrate import odeint
    plt.ion()
    
    P = 0	    # birth rate
    d = 0.0001  # natural death percent (per day)
    B = 0.0095  # transmission percent  (per day)
    G = 0.0001  # resurect percent (per day)
    A = 0.0001  # destroy percent  (per day)
    
    # solve the system dy/dt = f(y, t)
    def f(y, t):
    	Si = y[0]
    	Zi = y[1]
    	Ri = y[2]
    	# the model equations (see Munz et al. 2009)
    	f0 = P - B*Si*Zi - d*Si
    	f1 = B*Si*Zi + G*Ri - A*Si*Zi
    	f2 = d*Si + A*Si*Zi - G*Ri
    	return [f0, f1, f2]
    
    # initial conditions
    S0 = 500.     		# initial population
    Z0 = 0        		# initial zombie population
    R0 = 0        		# initial death population
    y0 = [S0, Z0, R0]	# initial condition vector
    t  = np.linspace(0, 5., 1000) 	# time grid
    
    # solve the DEs
    soln = odeint(f, y0, t)
    S = soln[:, 0]
    Z = soln[:, 1]
    R = soln[:, 2]
    
    # plot results
    plt.figure()
    plt.plot(t, S, label='Living')
    plt.plot(t, Z, label='Zombies')
    plt.xlabel('Days from outbreak')
    plt.ylabel('Population')
    plt.title('Zombie Apocalypse - No Init. Dead Pop.; No New Births.')
    plt.legend(loc=0)
    
    # change the initial conditions
    R0 = 0.01*S0   # 1% of initial pop is dead
    y0 = [S0, Z0, R0]
    
    # solve the DEs
    soln = odeint(f, y0, t)
    S = soln[:, 0]
    Z = soln[:, 1]
    R = soln[:, 2]
    
    plt.figure()
    plt.plot(t, S, label='Living')
    plt.plot(t, Z, label='Zombies')
    plt.xlabel('Days from outbreak')
    plt.ylabel('Population')
    plt.title('Zombie Apocalypse - 1% Init. Pop. is Dead; No New Births.')
    plt.legend(loc=0)
    
    # change the initial conditions
    R0 = 0.01*S0   # 1% of initial pop is dead
    P  = 10        # 10 new births daily
    y0 = [S0, Z0, R0]
    
    # solve the DEs
    soln = odeint(f, y0, t)
    S = soln[:, 0]
    Z = soln[:, 1]
    R = soln[:, 2]
    
    plt.figure()
    plt.plot(t, S, label='Living')
    plt.plot(t, Z, label='Zombies')
    plt.xlabel('Days from outbreak')
    plt.ylabel('Population')
    plt.title('Zombie Apocalypse - 1% Init. Pop. is Dead; 10 Daily Births')
    plt.legend(loc=0)
    



.. image:: Zombie_Apocalypse_ODEINT_attachments/zombie_nodead_nobirths.png

.. image:: Zombie_Apocalypse_ODEINT_attachments/zombie_somedead_nobirth.png

.. image:: Zombie_Apocalypse_ODEINT_attachments/zombie_somedead_10birth.png

