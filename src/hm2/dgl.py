from typing import Callable
import numpy as np

from src.util import types


def richtungsfeld_plot(f: Callable[[np.ndarray, np.ndarray], np.ndarray], \
        x_grenzen: tuple[float, float], y_grenzen: tuple[float, float], hx: float, hy: float):
    
    x_min, x_max = x_grenzen
    x_min, x_max = float(x_min), float(x_max)
    assert x_max > x_min
    y_min, y_max = y_grenzen
    y_min, y_max = float(y_min), float(y_max)
    assert y_max > y_min

    x = np.arange(x_min,x_max, hx, dtype=np.float64)
    y = np.arange(y_min, y_max, hy, dtype=np.float64)
    x, y = np.meshgrid(x, y, indexing='xy')

    dx = np.ones_like(x)
    dy = f(x, y)
    
    # pfeil vektoren normieren
    l = np.sqrt(dx**2 + dy**2)
    dx /= l
    dy /= l

    import matplotlib.pyplot as plt
    plt.title('Richtungsfeld der DGL')
    plt.quiver(x, y, dx, dy, color='blue', width=0.003, angles='xy')
    plt.grid()
    plt.show()


def allg_runge_kutta(x_grenzen: tuple[float, float], n:int, \
        f: Callable[[float, np.ndarray], np.ndarray], y0: np.ndarray, A, b, c):
    x_min, x_max = x_grenzen
    x_min, x_max = float(x_min), float(x_max)
    assert x_max > x_min
    
    assert len(b.shape) == 1
    assert b.shape == c.shape
    assert len(A.shape) == 2
    assert A.shape[0] == A.shape[1]
    assert A.shape[0] == len(b)

    A = A.astype(np.float64)
    b = b.astype(np.float64)
    c = c.astype(np.float64)

    assert np.isclose(np.sum(b), 1, atol=1e-10)
    assert all([c_i <= 1. for c_i in c])
    assert np.array_equal(A, np.tril(A))

    s = len(b)

    h = abs((x_max - x_min)/n)

    y_i = y0.astype(np.float64)

    for i in range(n):
        x_i = x_min + i * h
        k = np.zeros_like(b)

        for stufen_idx in range(s):
            k[stufen_idx] = f(x_i + c[stufen_idx]*h, \
                    y_i + h \
                    * np.sum(A[stufen_idx, :stufen_idx] * k[stufen_idx, :stufen_idx]))
        
        y_i += h * np.sum(b * k)

