import numpy as np
import matplotlib.pyplot as plt
from analytical_solutions import *
from plotting_functions import *

# Model parameters
a = 10.0
R = 1.0
B = 1.0/ (R + 4.0/3.0)
Vinf = np.array([1.0, 0.0, 0.0])*a
gradVinf = np.array([[0.0, 0.0, 0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]])
#gradVinf2D = np.array([[0.0, 1.0],[0.0,0.0]])

# Plotting parameters
maxxy = 2.0
n_scalar = 256
n_vector = 24
outputfile = 'translation_a_'+str(a)+'_R'+str(R)+'.eps'

# Create solution object
flow = LinearFlow(Vinf, gradVinf, a, B)
#flow = LinearFlow2D(gradVinf2D, a, B)

w, h = plt.figaspect(1.0)
fig = plt.figure(figsize=(w,h))
fig.set_tight_layout(True)
ax = plt.subplot(221, aspect='equal')
scalar_plot(ax, a, flow.pressure, maxxy, n_scalar, flow.ndim)
plt.title('Pressure $P$')

ax = plt.subplot(222, aspect='equal')
streamtracer(a, flow.velocity, flow.grad_velocity, maxxy, flow.ndim)
vector_plot(ax, a, flow.velocity, maxxy, n_vector, flow.ndim)
plt.title('Solid velocity $\mathbf{v}_s$')

ax = plt.subplot(223, aspect='equal')
scalar_plot(ax, a, flow.compaction_rate, maxxy, n_scalar, flow.ndim)
plt.title('Compaction rate $\mathcal{C}$')

ax = plt.subplot(224, aspect='equal')
vector_plot(ax, a, flow.darcy_flux, maxxy, n_vector, flow.ndim)
plt.title('Darcy flux $\mathbf{q}$')

plt.savefig(outputfile)
plt.show()
