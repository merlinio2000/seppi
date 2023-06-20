import numpy as np
from src.hm2.dgl import runge_kutta_4stufig_vec


# Aufgabe b)
def dz(t, z):
    return np.array([z[1], -0.1*z[1] * abs(z[1]) - 10], dtype=np.float64)

z0 = np.array([20., 0.])

a, b = 0, 3

t_grenzen = a, b

h = 0.05
n = int(np.ceil((b-a)/h))

y = runge_kutta_4stufig_vec(t_grenzen, n, dz, z0)


import matplotlib.pyplot as plt

x_pos = y[:, 0]
dx_vel = y[:, 1]

t = np.arange(a,b+h, step=h)
plt.title('Aufgabe b)')
plt.plot(t, x_pos, 'r', label='Position x(t)')
plt.plot(t, dx_vel, 'g', label='Geschwindigkeit dx/dt(t)')
plt.legend()
plt.grid()
plt.show()


# Aufgabe c)

def dz_refl(t, z):
    z1, z2 = z # x, dx/dt
    if z1 < 0 and z2 < 0:
        z2 = -z2
        return np.array([z1 , -0.1*z2 * abs(z2) - 10])
    return np.array([z2, -0.1*z2 * abs(z2) - 10], dtype=np.float64)

a, b = 0, 8
n = int(np.ceil((b-a)/h))

y = runge_kutta_4stufig_vec((a, b), n, dz_refl, z0)


x_pos = y[:, 0]
dx_vel = y[:, 1]

t = np.arange(a,b+h, step=h)
plt.title('Aufgabe c)')
plt.plot(t, x_pos, 'r', label='Position x(t)')
plt.plot(t, dx_vel, 'g', label='Geschwindigkeit dx/dt(t)')
plt.legend()
plt.grid()
plt.show()
