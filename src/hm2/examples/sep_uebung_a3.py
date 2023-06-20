import numpy as np
from scipy.interpolate import CubicSpline
from src.hm2.interpolation import nat_spline_interpolation


x = np.array([0., 2, 6])
y = np.array([0.1, 0.9, 0.1])

x_int = np.linspace(0, 6)
y_int = nat_spline_interpolation(x, y, x_int)


import matplotlib.pyplot as plt

sci_spline = CubicSpline(x, y, bc_type='natural')

plt.plot(x, y, 'bo', label='Messpunkte')
plt.plot(x_int, y_int, 'r', label='Spline')
plt.plot(x_int, sci_spline(x_int), 'g', label='SciPy')
plt.legend()
plt.grid()
plt.show()
