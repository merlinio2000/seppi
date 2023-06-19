import numpy as np
import matplotlib.pyplot as plt
import src.util.utl as utl

def nat_spline_interpolation(x: np.ndarray, y: np.ndarray, x_int: np.ndarray) \
    -> np.ndarray:
    '''
    natürliche kubische Spline Interpolation für n+1 Stützpunkte 

    Parameters:
        x: Zeilenvektor mit x der Stützpunkte, länge = n + 1
        y: Zeilenvektor mit y der Stützpunkte, länge = n + 1
        x_int: Die x für die die Interpolation berechnet werden soll
    Returns:
        y_int: Die für x_int interpolierten y
    '''
    assert len(x.shape) == 1
    assert x.shape == y.shape
    n = len(x) - 1 
    assert n >= 2

    x, y, x_int = x.astype(np.float64), y.astype(
        np.float64), x_int.astype(np.float64)

    print('natürliche kubische Spline Interpolation')

    a = y[:-1]

    print('Koeffizienten a_i aus y_i:')
    utl.np_pprint(a)

    h = x[1:] - x[:-1]

    c = np.zeros_like(x)

    if n >= 2:
        A = np.diag(2 * (h[:-1] + h[1:])) + \
            np.diag(h[1:-1], -1) + np.diag(h[1:-1], 1)
        print('A-Matrix für die c_i:')
        utl.np_pprint(A) 

        z = 3 * (y[2:] - y[1:-1]) / h[1:] - \
            3 * (y[1:-1] - y[0:-2]) / h[:-1]
        print('z-Vektor für die c_i:')
        utl.np_pprint(z)

        c[1:-1] = np.linalg.solve(A, z)
        print('Berechnete Koeffizienten c_i aus Ac = z:')
        utl.np_pprint(c)

    b = (y[1:] - y[:-1]) / h[:] \
            - h[:] / 3 * (c[1:] + 2 * c[:-1])
    print('Berechnete Koeffizienten b_i:')
    utl.np_pprint(b)

    d = (c[1:] - c[:-1]) / (3*h[:])
    print('Berechnete Koeffizienten d_i:')
    utl.np_pprint(d)

    yy = np.zeros_like(x_int)

    # x werte mit Funktion des korrekten Intervalls interpolieren
    # n+1 Stützpunkte -> n Intervalle (der reihe nach)
    for k in range(n):
        idx = np.where(np.logical_and(x_int >= x[k], x_int <= x[k+1]))

        dx = x_int[idx] - x[k]
        yy[idx] = a[k] + b[k] * dx + c[k] * dx**2 + d[k] * dx**3

    return yy


def lagrange_interpolation(x: np.ndarray, y: np.ndarray, x_int: np.ndarray) \
    -> np.ndarray:
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




import unittest

class InterpolationTest(unittest.TestCase):
    def test_lagrange_int_S4_A1(self):
        x = np.array([0, 2_500, 5_000, 10_000], dtype=np.float64)
        y = np.array([1_013, 747, 540, 226], dtype=np.float64)
        
        x_gesucht = np.array([3_750], dtype=np.float64)

        y_int = lagrange_interpolation(x, y, x_gesucht)
        actual = y_int[0]
        
        self.assertAlmostEqual(actual, 637.328125)
    
    def test_spline_int_S5_A2(self):
        x = np.array([4., 6, 8, 10])
        y = np.array([6., 3, 9, 0])

        _ = nat_spline_interpolation(x, y, np.array([]))

    def test_spline_int_S5_A3(self):
        x = np.array([1_900, 1_910, 1_920, 1_930, 1_940, 1_950,
                     1_960, 1_970, 1_980, 1_990, 2_000, 2_010])
        y = np.array([75.995, 91.972, 105.711, 123.203, 131.669, 150.697,
                     179.323, 203.212, 226.505, 249.633, 281.422, 308.745])
        
        y_int = nat_spline_interpolation(x, y, x)
        
        self.assertTrue(np.allclose(y_int, y))

        # x_int = np.linspace(x[0], x[-1], num=100_000)
        # y_int = nat_spline_interpolation(x, y, x_int)
        # plt.plot(x, y, 'bo', label='Messpunkte')
        # plt.plot(x_int, y_int, 'r', label='Spline Interpoliert')
        # plt.legend()
        # plt.grid()
        # plt.show()




