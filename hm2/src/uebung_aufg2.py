import numpy as np
import matplotlib.pyplot as plt

def f1(x):
    return np.sqrt(1 - x**2)
"""
(y-1)^2/b = 1 - (x-2)^2/a
(y-1)^2 = b - ((x-2)^2 * b) / a
y = sqrt(b - ((x-2)^2 * b) / a) + 1
"""

a = 2
b = 4

def f2(x):
    return np.sqrt(b - ((x-2)**2 * b) / a) + 1




xs = np.linspace(-50, 50, num=100_000)

y1 = f1(xs)
y2 = f2(xs)

# TODO how to plot the full circle
plt.plot(xs, y1, 'b', label='f1')
plt.plot(xs, y2, 'g', label='f2')

plt.legend()
plt.grid()
plt.show()
