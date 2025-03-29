import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon

# Création de trois distributions 1D
x = np.linspace(0, 10, 100)

# Distribution 1: gaussienne centrée en x=4
P = np.exp(-0.5 * ((x - 4) / 1.0)**2)
P = P / np.sum(P)  # Normalisation

# Distribution 2: gaussienne centrée en x=6 (légèrement déplacée)
Q = np.exp(-0.5 * ((x - 6) / 1.0)**2)
Q = Q / np.sum(Q)  # Normalisation

# Distribution 3: deux pics (structure différente)
R = np.exp(-0.5 * ((x - 2) / 0.5)**2) + np.exp(-0.5 * ((x - 8) / 0.5)**2)
R = R / np.sum(R)  # Normalisation

# Calcul des distances de Jensen-Shannon
js_distance_PQ = jensenshannon(P, Q, base=2)
js_distance_PR = jensenshannon(P, R, base=2)

# Visualisation des distributions
plt.figure(figsize=(15, 6))

# Graphique des distributions
plt.subplot(2, 1, 1)
plt.plot(x, P, 'b-', label='Distribution P (gaussienne à x=4)')
plt.plot(x, Q, 'r-', label='Distribution Q (gaussienne à x=6)')
# plt.plot(x, R, 'g-', label='Distribution R (deux pics)')
plt.legend()
plt.title('Comparaison de trois distributions 1D')
plt.grid(True)

# Graphique des différences
plt.subplot(2, 1, 2)
plt.plot(x, np.abs(P - Q), 'r-', label=f'|P-Q| (JS Dist = {js_distance_PQ:.4f})')
# plt.plot(x, np.abs(P - R), 'g-', label=f'|P-R| (JS Dist = {js_distance_PR:.4f})')
plt.legend()
plt.title('Différences absolues entre les distributions')
plt.grid(True)

plt.tight_layout()
plt.show()

print(f"Distance de Jensen-Shannon entre P et Q: {js_distance_PQ:.4f}")
print(f"Distance de Jensen-Shannon entre P et R: {js_distance_PR:.4f}")