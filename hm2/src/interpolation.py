import numpy as np
import matplotlib.pyplot as plt
import utl

# TODO HM2 add spline


def lagrange_int(x: np.ndarray, y: np.ndarray, x_int: np.ndarray) -> np.ndarray:
    '''
    Lagrange Interpolation für ein Polynom vom Grad n

    Parameters:
        x: Zeilenvektor mit x der Stützpunkte, länge = n + 1
        y: Zeilenvektor mit y der Stützpunkte, länge = n + 1
        x_int: Die x für die die Interpolation berechnet werden soll
    Returns:
        y_int: Die für x_int interpolierten y
    '''
    utl.assert_is_vec(x)
    utl.assert_eq_shape(x, y)
    utl.assert_is_vec(x_int)

    x = x.astype(np.float64)
    y = y.astype(np.float64)
    x_int = x_int.astype(np.float64)
    y_int = np.zeros_like(x_int)

    # n + 1 Stützpunkte!
    n = len(x) - 1

    for i in range(n+1):
        li = np.ones_like(x_int)
        for j in range(n+1):
            if i == j: continue
            li *= (x_int - x[j]) / (x[i] - x[j])
        
        y_int += li * y[i]

    return y_int

def np_polyval_fit_scaling_bsp():
    '''
    Beispiel aus Serie 4 Aufgabe 3 b)
    zur Interpolation mittels numpy's polyfit/polyval
    '''
    x = np.array([1981, 1984, 1989, 1993, 1997, 2000, 2001, 2003, 2004, 2010], dtype=np.float64)
    y = np.array([0.5, 8.2, 15, 22.9, 36.6, 51, 56.3, 61.8, 65, 76.7], dtype=np.float64)

    assert len(x) == len(y)
    '''
    Hier sind die Daten noch zusätzlich Skaliert (x - mean(x))
    Dadurch ist das Problem besser konditioniert

    Die Kurve an sich hat eine grössere Varianz,
    bei den einzelnen Stützpunkten dafür ist die Kurve exakt,
    was bei a) nicht der Fall ist
    '''
    # n + 1 Stützpunkte
    n = len(x) - 1
    x_nrm = x - np.mean(x)
    # DIE REIHENFOLGE der returned Koeffizienten ist beginnend
    # mit dem für den höchsten Exponent x^n danach absteigend
    coeff = np.polyfit(x_nrm, y, n)

    x_int = np.arange(1975, 2020.1, step=0.1)
    y_int = np.polyval(coeff, x_int - np.mean(x))

    plt.plot(x, y, label='original')
    plt.plot(x_int, y_int, label='interpolated')
    plt.xlim(1975, 2020)
    plt.ylim(-100, 250)
    plt.legend()
    plt.grid()
    plt.show()

