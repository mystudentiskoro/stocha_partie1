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
from scipy.spatial.distance import jensenshannon
import matplotlib.gridspec as gridspec

# Définition des limites et taille des boîtes pour la PDF
x_lim = [-20, 20]
y_lim = [-30, 30]
z_lim = [0, 50]
box_size = 1

# Nombre de boîtes
nx = int((x_lim[1] - x_lim[0]) / box_size)
ny = int((y_lim[1] - y_lim[0]) / box_size)
nz = int((z_lim[1] - z_lim[0]) / box_size)

# Fonction pour visualiser les projections de la PDF
def projection(pdf, title="PDF Projections"):
    # Projection sur le plan xy
    proj_xy = np.sum(pdf, axis=2)  # Somme de toutes les valeurs de z
    proj_yz = np.sum(pdf, axis=0)
    proj_xz = np.sum(pdf, axis=1)

    # print("Projection XY : \n", proj_xy)
    # print("Projection YZ : \n", proj_yz)
    # print("Projection XZ : \n", proj_xz)

    # Visualiser les projections
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(title)
    
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

# Fonction pour calculer et visualiser le système de Lorenz
def simulate_lorenz(sigma, rho, beta, state0, title="Lorenz System", print=True):
    # Définition de la fonction Lorenz avec les paramètres fournis
    def lorenz_model(state, t):
        x, y, z = state  # Unpack the state vector
        return sigma * (y - x), x * (rho - z) - y, x * y - beta * z  # Derivatives
    
    # Vecteur temps
    t = np.arange(0.0, 100.0, 0.02)
    
    # Résolution du système d'équations différentielles
    states = odeint(lorenz_model, state0, t)
    
    # Visualisation 3D du système

    if print:
        fig = plt.figure()
        plt.rcParams['font.family'] = 'serif'
        ax = fig.add_subplot(projection="3d")
        ax.plot(states[:, 0], states[:, 1], states[:, 2])
        plt.xlabel('x')
        plt.ylabel('y')
        ax.set_zlabel('z')
        plt.title(title)
        plt.legend(['Lorenz System'])
        plt.show()
    
    # Calcul de la PDF empirique
    pdf_matrix = np.zeros((nx, ny, nz))
    
    for state in states:
        x, y, z = state
        if x_lim[0] <= x <= x_lim[1] and y_lim[0] <= y <= y_lim[1] and z_lim[0] <= z <= z_lim[1]:
            i = int((x - x_lim[0]) / box_size)
            j = int((y - y_lim[0]) / box_size)
            k = int((z - z_lim[0]) / box_size)
            pdf_matrix[i, j, k] += 1
    
    # Normalisation pour obtenir une PDF
    total_sum = np.sum(pdf_matrix)
    if total_sum > 0:  # Éviter division par zéro
        pdf = pdf_matrix / total_sum
    else:
        pdf = pdf_matrix
    
    return states, pdf

# Premiers paramètres - Système chaotique standard
if False:
    sigma_1 = 10.0
    rho_1 = 28.0
    beta_1 = 8.0/3.0
    initial_state = [1.0, 1.0, 1.0]

    # Simulation avec les premiers paramètres
    print("Simulation avec paramètres standard (σ=10, ρ=28, β=8/3)")
    states_1, pdf_1 = simulate_lorenz(sigma_1, rho_1, beta_1, initial_state, "Système de Lorenz standard")
    projection(pdf_1, "Projections PDF - Paramètres standard")

    # Seconds paramètres - Système modifié
    sigma_2 = 5.0
    rho_2 = 18.0
    beta_2 = 8.0

    # Simulation avec les seconds paramètres
    print("\nSimulation avec paramètres modifiés (σ=5, ρ=18, β=8)")
    states_2, pdf_2 = simulate_lorenz(sigma_2, rho_2, beta_2, initial_state, "Système de Lorenz modifié")
    projection(pdf_2, "Projections PDF - Paramètres modifiés")

    # Calculer les distances entre les PDFs
    # Aplatir les PDF pour utiliser les méthodes de distance
    flat_pdf_1 = pdf_1.flatten()
    flat_pdf_2 = pdf_2.flatten()

    # Jensen-Shannon distance
    js_distance = jensenshannon(flat_pdf_1, flat_pdf_2)
    print(f"\nDistance de Jensen-Shannon entre les deux PDFs: {js_distance}")

    # Troisième paramètre - Condition initiale différente
    different_initial_state = [10.0, 10.0, 10.0]

    # Simulation avec condition initiale différente
    print("\nSimulation avec condition initiale différente (10,10,10)")
    states_3, pdf_3 = simulate_lorenz(sigma_1, rho_1, beta_1, different_initial_state, "Système de Lorenz - Condition initiale différente")
    projection(pdf_3, "Projections PDF - Condition initiale différente")

    # Calculer la distance avec la condition initiale différente
    flat_pdf_3 = pdf_3.flatten()
    js_distance_init = jensenshannon(flat_pdf_1, flat_pdf_3)
    print(f"\nDistance de Jensen-Shannon entre condition initiale standard et modifiée: {js_distance_init}")



sigma_1 = 10.0
rho_1 = 28.0
beta_1 = 8.0/3.0
initial_state = [1.0, 1.0, 1.0]

# Simulation avec les premiers paramètres
print("Simulation avec paramètres standard (σ=10, ρ=28, β=8/3)")
states_1, pdf_base = simulate_lorenz(sigma_1, rho_1, beta_1, initial_state, "Système de Lorenz standard")
flat_base_pdf = pdf_base.flatten()


def impact_parameter_graph(sigma, rho, beta, initial_state, delta=5):
    # Création des plages de valeurs
    sigma_values = np.linspace(sigma-delta, sigma+delta, 100)  # 100 valeurs entre sigma-delta et sigma+delta
    rho_values = np.linspace(rho-delta, rho+delta, 100)        # Inclut rho±5
    beta_values = np.linspace(max(0.1, beta-delta * 2/5), beta+delta * 2/5 + 0.1, 100)  # Plage adaptée pour beta
    
    # Listes pour stocker les distances
    sigma_distances = []
    rho_distances = []
    beta_distances = []

    # Étude de sigma
    print("Étude de l'impact de sigma:")
    for new_sigma in sigma_values:
        state, new_pdf = simulate_lorenz(new_sigma, rho, beta, initial_state, 
                                         f"Système de Lorenz avec σ={new_sigma}", False)
        flat_pdf = new_pdf.flatten()
        distance = jensenshannon(flat_base_pdf, flat_pdf)
        print(f"Distance de Jensen-Shannon avec σ={new_sigma}: {distance}")
        sigma_distances.append(distance)

    # Étude de rho
    print("\nÉtude de l'impact de rho:")
    for new_rho in rho_values:
        state, new_pdf = simulate_lorenz(sigma, new_rho, beta, initial_state, 
                                         f"Système de Lorenz avec ρ={new_rho}", False)
        flat_pdf = new_pdf.flatten()
        distance = jensenshannon(flat_base_pdf, flat_pdf)
        print(f"Distance de Jensen-Shannon avec ρ={new_rho}: {distance}")
        rho_distances.append(distance)

    # Étude de beta
    print("\nÉtude de l'impact de beta:")
    for new_beta in beta_values:
        state, new_pdf = simulate_lorenz(sigma, rho, new_beta, initial_state, 
                                         f"Système de Lorenz avec β={new_beta}", False)
        flat_pdf = new_pdf.flatten()
        distance = jensenshannon(flat_base_pdf, flat_pdf)
        print(f"Distance de Jensen-Shannon avec β={new_beta}: {distance}")
        beta_distances.append(distance)
    
    # Visualisation des résultats
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(sigma_values, sigma_distances, 'o-')
    plt.xlabel('Sigma (σ)')
    plt.ylabel('Distance JS')
    plt.title('Sigma sensibility')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(rho_values, rho_distances, 'o-')
    plt.xlabel('Rho (ρ)')
    plt.ylabel('Distance JS')
    plt.title('Rho sensibility')
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(beta_values, beta_distances, 'o-')
    plt.xlabel('Beta (β)')
    plt.ylabel('Distance JS')
    plt.title('Beta sensibility')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return {
        'sigma': {'values': sigma_values, 'distances': sigma_distances},
        'rho': {'values': rho_values, 'distances': rho_distances},
        'beta': {'values': beta_values, 'distances': beta_distances}
    }

# impact_parameter_graph(sigma_1, rho_1, beta_1, initial_state, 7.5)

#visualisation des effets des paramètres

def lorenz_parameter_variation(param_name, sigma_base=10.0, rho_base=28.0, beta_base=8.0/3.0, 
                               initial_state=[1.0, 1.0, 1.0], delta=7.5, n_plots=10):
    # Définition de la fonction Lorenz
    def lorenz_model(state, t, sigma, rho, beta):
        x, y, z = state
        return sigma * (y - x), x * (rho - z) - y, x * y - beta * z
    
    # Vecteur temps
    t = np.arange(0.0, 40.0, 0.01)
    
    # Déterminer les valeurs du paramètre à utiliser
    if param_name == 'sigma':
        param_values = np.linspace(sigma_base - delta, sigma_base+2, int(n_plots/2))
        base_value = sigma_base
    elif param_name == 'rho':
        param_values = np.linspace(24, 32, n_plots) #totalement pris au hasard
        base_value = rho_base
    elif param_name == 'beta':
        param_values = np.linspace(max(0.1, beta_base - delta/3), beta_base + delta/3, n_plots)
        base_value = beta_base
    else:
        raise ValueError("param_name doit être 'sigma', 'rho' ou 'beta'")
    
    # Créer une figure avec une grille de sous-graphiques
    fig = plt.figure(figsize=(20, 15))
    gs = gridspec.GridSpec(3, 4)  # 3 lignes, 4 colonnes pour une disposition plus compacte
    
    # Créer un suptitle pour expliquer la visualisation
    param_symbol = {'sigma': 'σ', 'rho': 'ρ', 'beta': 'β'}
    fig.suptitle(f"Variation of the Lorenz system around {param_symbol[param_name]} = {base_value} ± {delta/3}", 
                 fontsize=16)
    
    # Générer les graphiques
    for i, param_val in enumerate(param_values):
        if i < n_plots:  # Assurez-vous de ne pas dépasser le nombre de graphiques demandés
            # Calculer la position du sous-graphique
            row = i // 4
            col = i % 4
            
            # Créer le sous-graphique 3D
            ax = fig.add_subplot(gs[row, col], projection='3d')
            
            # Définir les paramètres pour cette itération
            if param_name == 'sigma':
                sigma, rho, beta = param_val, rho_base, beta_base
                param_text = f'σ = {param_val:.2f}'
            elif param_name == 'rho':
                sigma, rho, beta = sigma_base, param_val, beta_base
                param_text = f'ρ = {param_val:.2f}'
            else:  # beta
                sigma, rho, beta = sigma_base, rho_base, param_val
                param_text = f'β = {param_val:.2f}'
            
            # Résoudre le système
            states = odeint(lorenz_model, initial_state, t, args=(sigma, rho, beta))
            
            # Tracer la solution
            ax.plot(states[:, 0], states[:, 1], states[:, 2], linewidth=0.7)
            
            # Personnaliser le graphique
            ax.set_title(param_text, fontsize=12)
            ax.set_xlabel('X', fontsize=8)
            ax.set_ylabel('Y', fontsize=8)
            ax.set_zlabel('Z', fontsize=8)
            ax.tick_params(axis='x', labelsize=6)
            ax.tick_params(axis='y', labelsize=6)
            ax.tick_params(axis='z', labelsize=6)
            
            # Définir la même vue pour tous les graphiques
            ax.view_init(elev=30, azim=45)
            
            # Optionnel: définir les mêmes limites pour tous les graphiques
            ax.set_xlim([-25, 25])
            ax.set_ylim([-25, 25])
            ax.set_zlim([0, 50])
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Ajuster la mise en page (rect laisse de l'espace pour le suptitle)
    plt.show()

    print(f"Visualization completed with 8 variations of {param_name} in range {base_value-delta} à {base_value+delta}")


# lorenz_parameter_variation('sigma', delta=7.5, n_plots=8)

# # Pour varier rho autour de 28 ± 7.5
# lorenz_parameter_variation('rho', delta=7.5, n_plots=8)

# # Pour varier beta autour de 8/3 ± 7.5
# lorenz_parameter_variation('beta', delta=7.5, n_plots=8)

def condition_initial():
    states_1, pdf_1 = simulate_lorenz(sigma_1, rho_1, beta_1, initial_state, "Système de Lorenz standard", False)
    flat_pdf_1 = pdf_1.flatten()


    # Simulation avec condition initiale différente
    different_initial_state = [10.0, 10.0, 10.0]
    print("\nSimulation avec condition initiale différente (10,10,10)")
    states_3, pdf_3 = simulate_lorenz(sigma_1, rho_1, beta_1, different_initial_state, "Système de Lorenz - Condition initiale différente")
    projection(pdf_3, "Projections PDF - Condition initiale différente")
    flat_pdf_3 = pdf_3.flatten()
    js_distance_init = jensenshannon(flat_pdf_1, flat_pdf_3)
    print(f"\nDistance de Jensen-Shannon entre condition initiale standard et modifiée: {js_distance_init}")

    different_initial_state = [100.0, 100.0, 100.0]
    print("\nSimulation avec condition initiale différente (10,10,10)")
    states_3, pdf_3 = simulate_lorenz(sigma_1, rho_1, beta_1, different_initial_state, "Système de Lorenz - Condition initiale différente")
    projection(pdf_3, "Projections PDF - Condition initiale différente")
    flat_pdf_3 = pdf_3.flatten()
    js_distance_init = jensenshannon(flat_pdf_1, flat_pdf_3)
    print(f"\nDistance de Jensen-Shannon entre condition initiale standard et modifiée: {js_distance_init}")

condition_initial()





    


# Étude de l'impact des paramètres
# impact_parameter(sigma_1, rho_1, beta_1, initial_state, 100)

# simulate_lorenz(sigma_1, rho_1, -0.00001, initial_state, f"Système de Lorenz avec β={-0.00001}", True)
















# PLOTLY : Décommenter cette section si vous avez installé Plotly
"""
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"

# Visualisation du premier système
fig = go.Figure(data=[go.Scatter3d(
    x=states_1[:, 0], y=states_1[:, 1], z=states_1[:, 2],
    mode='markers',
    marker=dict(size=2, opacity=0.8)
)])
fig.update_layout(title='Système de Lorenz standard')
fig.update_scenes(aspectmode='data')
fig.show()

# Visualisation du second système
fig = go.Figure(data=[go.Scatter3d(
    x=states_2[:, 0], y=states_2[:, 1], z=states_2[:, 2],
    mode='markers',
    marker=dict(size=2, opacity=0.8)
)])
fig.update_layout(title='Système de Lorenz modifié')
fig.update_scenes(aspectmode='data')
fig.show()

# Visualisation du troisième système
fig = go.Figure(data=[go.Scatter3d(
    x=states_3[:, 0], y=states_3[:, 1], z=states_3[:, 2],
    mode='markers',
    marker=dict(size=2, opacity=0.8)
)])
fig.update_layout(title='Système de Lorenz - Condition initiale différente')
fig.update_scenes(aspectmode='data')
fig.show()
"""