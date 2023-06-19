import numbers
import numpy as np
from typing import Callable

def allg_runge_kutta(x_grenzen: tuple[float, float], n:int, \
        df: Callable[[float, float], np.ndarray], \
        y0: float, A: np.ndarray, b: np.ndarray, c: np.ndarray):
    '''
    Löst das allgemeine s-Stufige Runge-Kutta Verfahren

    Parameters:
        x_grenzen: des zu bestimmenden Intervalls
        n: Anzahl Teilintervalle
        df: Funktion die die Steigung gibt, y' = f(x, y)
        y0: Startwert
        A: s x s Matrix der Vorkfaktoren für k_s
        b: Vektor mit länge = s der Gewichte der Einzelnen k_s
        c: Vektor mit länge = s der h Vorfaktoren für die x_i der k_s
    Returns:
        y: die Approximierten y Werte im intervall, länge n+1
    '''
    x_min, x_max = x_grenzen
    x_min, x_max = float(x_min), float(x_max)
    assert x_max > x_min
    
    assert isinstance(y0, numbers.Real)
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
    
    y = np.zeros(n+1)
    y[0] = y0

    for i in range(n):
        x_i = x_min + i * h
        k = np.zeros_like(b)

        for stufen_idx in range(s):
            k[stufen_idx] = df(x_i + c[stufen_idx]*h, \
                    y[i] + h \
                    * np.sum(A[stufen_idx, :stufen_idx] * k[:stufen_idx]))
        
        y[i+1] = y[i] + h * np.sum(b * k)
    
    return y


def Aufg5b():
    def dy(t, y):
        return t/y

    A = np.array([[0, 0, 0], [1/3, 0, 0], [0, 2/3, 0]], dtype=np.float64)
    c = np.array([0., 1./3, 2./3], dtype=np.float64)
    b = np.array([0.25, 0, 0.75], dtype=np.float64)
    
    t_min, t_max = 2., 5.
    y0 =  1
    h = 0.1
    n = int(np.ceil((t_max - t_min) / h))

    t = np.arange(t_min, t_max + h, step=h)
    y = allg_runge_kutta((t_min, t_max), n, dy, y0, A, b, c)

    import matplotlib.pyplot as plt

    def y_exakt(t):
        return np.sqrt(t**2 - 3)
    plt.plot(t, y, label='Runge-Kutta')
    plt.plot(t, y_exakt(t), label='Exakt')
    plt.legend()
    plt.grid()
    plt.show()

    assert np.isclose(y[-1], 4.69046273, atol=1e-7)


if __name__ == '__main__':
    Aufg5b()


