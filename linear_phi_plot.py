import numpy as np
import matplotlib.pyplot as plt
from analytical_solutions import *
from plotting_functions import *

# Model parameters
a = 0.001
R = 1e3
B = 1.0/ (R + 4.0/3.0)
phi0 = 0.01
Vinf = np.array([0.0, 0.0, 0.0])*a
gradVinf = np.array([[0.0, 1.0, 0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]])
#gradVinf2D = np.array([[0.0, 1.0],[0.0,0.0]])

# Plotting parameters
maxxy = 2.0
n = 128
outputfile = 'strain_phi_a_'+str(a)+'_R'+str(R)+'_phi0'+str(phi0)+'.eps'

# Create solution object
flow = LinearFlow(Vinf, gradVinf, a, B)
#flow = LinearFlow2D(gradVinf2D, a, B)
flow.phi0 = phi0

#tvals = np.array([1.0, 2.0, 3.0, 4.0])
tvals = np.array([0.5, 1.0, 1.5, 2.0])*np.pi
fig = plt.figure(figsize=(5.25,4.0*len(tvals)))
fig.set_tight_layout(True)

nrow = len(tvals)
ncol = 1
for i, t in enumerate(tvals):
    print(t)
    flow.t = t
    ax = plt.subplot(nrow, ncol, i+1, aspect='equal')
    phi_plot(ax, a, flow.porosity, phi0, maxxy, n, flow.ndim)
#    plt.title('t = '+str(t))
    plt.title('$t = '+str(t/np.pi) +'\pi$')

plt.savefig(outputfile)
plt.show()


