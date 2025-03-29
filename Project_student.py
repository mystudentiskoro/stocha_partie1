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




# Définir les limites du domaine spatial
x_lim = [-20, 20]
y_lim = [-30, 30]
z_lim = [0, 50]
box_size = 5


# Calculer le nombre de boîtes dans chaque dimension
nx = int((x_lim[1] - x_lim[0]) / box_size)
ny = int((y_lim[1] - y_lim[0]) / box_size)
nz = int((z_lim[1] - z_lim[0]) / box_size)

# Créer une grille de boîtes
matrice = np.zeros((nx, ny, nz))

for state in states :
    x,y,z = state     #solution de l'équation à chaque t
    if(x_lim[0] <= x <= x_lim[1] and y_lim[0] <= y <= y_lim[1] and z_lim[0] <= z <= z_lim[1]):
        i = int((x - x_lim[0]) / box_size)
        j = int((y - y_lim[0]) / box_size)
        k = int((z - z_lim[0]) / box_size)
        matrice[i, j, k] += 1

# Normaliser pour obtenir une PDF
sum = np.sum(matrice)
pdf = matrice / sum

print("PDF : ", pdf)
 