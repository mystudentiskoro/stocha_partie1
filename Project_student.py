#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LINMA1731 Stochastic Processes

Code for the project

@author: Philémon Beghin and Amir Mehrnoosh
"""

"""
LORENZ SYSTEM
"""

# from https://en.wikipedia.org/wiki/Lorenz_system

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D
from numpy import pi
import random

sigma = 10.0
rho = 28.0
beta = 8.0/3.0

# Lorenz model

def Lorenz(state,t):
    x, y, z = state  # Unpack the state vector
    return sigma * (y - x), x * (rho - z) - y, x * y - beta * z  # Derivatives

state0 = [1.0, 1.0, 1.0] # initial condition
t = np.arange(0.0, 100.0, 0.02) # time vector

states = odeint(Lorenz, state0, t) # vector containing the (x,y,z) positions for each time step    en gros la résollution du système d'équations différentielles

fig = plt.figure()
plt.rcParams['font.family'] = 'serif'
ax = fig.add_subplot(projection="3d")
ax.plot(states[:, 0], states[:, 1], states[:, 2])
plt.xlabel('x')
plt.ylabel('y')
plt.legend(['True system'])
plt.draw()
plt.show()


"""
PLOTLY : TRUE SYSTEM
"""

# Uncomment this section once you've installed the "Plotly" package


# import plotly.graph_objects as go
# import plotly.io as pio
# pio.renderers.default = "browser"


# fig = go.Figure(data=[go.Scatter3d(x=states[:, 0],y=states[:, 1],z=states[:, 2],
#                                    mode='markers',
#                                    marker=dict(
#                                        size=2,
#                                        opacity=0.8
#     )                        
#                                    )])
# fig.update_layout(
#     title='True system')
# fig.update_scenes(aspectmode='data')
# fig.show()


x_lim = [-20, 20]
y_lim = [-30, 30]
z_lim = [0, 50]
box_size = 5

#nombre de box
nx = int((x_lim[1] - x_lim[0]) / box_size)
ny = int((y_lim[1] - y_lim[0]) / box_size)
nz = int((z_lim[1] - z_lim[0]) / box_size)


matrice = np.zeros((nx, ny, nz))

for state in states :
    x,y,z = state     #solution de l'équation à chaque t
    if(x_lim[0] <= x <= x_lim[1] and y_lim[0] <= y <= y_lim[1] and z_lim[0] <= z <= z_lim[1]):
        i = int((x - x_lim[0]) / box_size)
        j = int((y - y_lim[0]) / box_size)
        k = int((z - z_lim[0]) / box_size)
        matrice[i, j, k] += 1

#normaliser pour la pdf
sum = np.sum(matrice)
pdf = matrice / sum

# print("PDF : ", pdf)   matrice 12x10x8  donc 8 prints

#projection sur le plan xy
proj_xy = np.sum(pdf, axis=2)  #somme de toutes les valeurs de z
proj_yz = np.sum(pdf, axis=0)
proj_xz = np.sum(pdf, axis=1)

print("Projection XY : \n", proj_xy)
print("Projection YZ : \n", proj_yz)
print("Projection XZ : \n", proj_xz)



# Visualiser les projections   chat gpt oeoe
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
im1 = axs[0].imshow(proj_xy.T, origin='lower', extent=[x_lim[0], x_lim[1], y_lim[0], y_lim[1]])
axs[0].set_title('Projection XY')
axs[0].set_xlabel('X')
axs[0].set_ylabel('Y')
plt.colorbar(im1, ax=axs[0])

im2 = axs[1].imshow(proj_yz.T, origin='lower', extent=[y_lim[0], y_lim[1], z_lim[0], z_lim[1]])
axs[1].set_title('Projection YZ')
axs[1].set_xlabel('Y')
axs[1].set_ylabel('Z')
plt.colorbar(im2, ax=axs[1])

im3 = axs[2].imshow(proj_xz.T, origin='lower', extent=[x_lim[0], x_lim[1], z_lim[0], z_lim[1]])
axs[2].set_title('Projection XZ')
axs[2].set_xlabel('X')
axs[2].set_ylabel('Z')
plt.colorbar(im3, ax=axs[2])

plt.tight_layout()
plt.show()

#implémentation de la distance