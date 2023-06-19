import numbers
from typing import Callable
import numpy as np


def richtungsfeld_plot_bsp(f: Callable[[np.ndarray, np.ndarray], np.ndarray], \
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
        df: Callable[[float, float], np.ndarray], \
        y0: float, A: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
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



def allg_runge_kutta_vec(x_grenzen: tuple[float, float], n:int, \
        df: Callable[[float, np.ndarray], np.ndarray], \
        y0: np.ndarray, A: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    '''
    Löst das allgemeine s-Stufige Runge-Kutta Verfahren für ein System von DGL

    Hauptunterschied zu allg_runge_kutta ist das y0 ein Vektor ist und entsprechend
    eine Matrix mit allen y Vektoren returned wird

    Parameters:
        x_grenzen: des zu bestimmenden Intervalls
        n: Anzahl Teilintervalle
        df: Funktion die die Steigung gibt, y' = f(x, y)
        y0: Startzeilenvektor länge m
        A: s x s Matrix der Vorkfaktoren für k_s
        b: Vektor mit länge = s der Gewichte der Einzelnen k_s
        c: Vektor mit länge = s der h Vorfaktoren für die x_i der k_s
    Returns:
        y: n+1 x m Matrix die Approximierten y Werte im intervall, 
            einzelne Vektoren stehen in den Zeilen 
    '''
    x_min, x_max = x_grenzen
    x_min, x_max = float(x_min), float(x_max)
    assert x_max > x_min
    
    assert len(y0.shape) == 1
    assert len(b.shape) == 1
    assert b.shape == c.shape
    assert len(A.shape) == 2
    assert A.shape[0] == A.shape[1]
    assert A.shape[0] == len(b)

    y0 = y0.astype(np.float64)
    A = A.astype(np.float64)
    b = b.astype(np.float64)
    c = c.astype(np.float64)

    assert np.isclose(np.sum(b), 1, atol=1e-10)
    assert all([c_i <= 1. for c_i in c])
    assert np.array_equal(A, np.tril(A))

    s = len(b)

    h = abs((x_max - x_min)/n)
    
    y = np.zeros((n+1, len(y0)))
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




import unittest

class DglTest(unittest.TestCase):
    def test_allg_runge_kutta_FS21_A5(self):
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

        # import matplotlib.pyplot as plt
        #
        # def y_exakt(t):
        #     return np.sqrt(t**2 - 3)
        #
        # plt.subplot(121)
        # plt.plot(t, y, label='Runge-Kutta')
        # plt.xlabel('Zeit t')
        # plt.ylabel('y')
        # plt.grid()
        # plt.legend()
        #
        # plt.subplot(122)
        # plt.plot(t, y_exakt(t), label='Exakt')
        # plt.legend()
        # plt.grid()
        # plt.show()

        self.assertAlmostEqual(y[-1], 4.69046273, places=7)
