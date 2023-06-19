import numpy as np

from src.util import utl
from ..util import types
from typing import Tuple


def romberg(x_grenzen: Tuple[float, float], f: types.NPValueToScalarFn, m: int)\
        -> float:
    '''
    Romberg-Extrapolation für eine Funktion f im Intervall a,b

    Parameters:
        x_grenzen: a,b inklusive Intervallgrenzen
        f: zu integriende funktion
        m: "tiefe" der Rekursion
    Returns:
        float: Annäherung des Integrals (Schritt T_0m)
    '''
    def n(j):
        return 2**j

    a, b = float(x_grenzen[0]), float(x_grenzen[1])
    assert b > a

    T = np.zeros((m+1, m+1))
    for j in range(m+1):
        T[j, 0] = sum_Tf(x_grenzen, n(j), f, log=False)

    for i in range(1, m+1):
        for j in range(0, m-i+1):
            T[j, i] = (4**i * T[j+1, i-1] - T[j, i-1]) / (4**i - 1)

    print('Rekursionsschritte T_ji:')
    utl.np_pprint(T)

    return T[0, m]

def sum_Rf(x_grenzen: Tuple[float, float], n: int, f: types.NPValueToScalarFn) -> float:
    '''
    integriert f nach der summierten Rechteckregel

    Parameters:
        x_grenzen: Die untere/obere Grenze des gesuchten Intervalls
        n: Anzahl stützstellen ZWINGEND INTEGER
        f: Zu integrierende Funktion
    Returns:
        float: Annäherung des Integrals
    '''
    assert isinstance(n, int) 

    a, b = x_grenzen
    a, b = float(a), float(b)
    assert b > a

    h = abs((b-a)/n)

    res = 0.
    print(f'Summierte Rechteckregel im Intervall [{a}, {b}], h={h}')

    for i in range(n):
        x_i = a + i*h
        y_i = f(x_i + h/2)
        print(f'\ti={i}: x_i={x_i} -> f(x_i + h/2) = {y_i}')
        res += y_i

    print(f'Summierte Funktionswerte: {res}')
    res *= h
    print(f'Endresulat (*= h): {res}')

    return res

def sum_Tf_variable_h(x: np.ndarray, y: np.ndarray) -> float:
    '''
    Integriert tabellierte Daten mit variabler Schrittweite nach der Trapezregel

    Parameters:
        x: alle x der n+1 Messpunkte
        y: alle y der n+1 Messpunkte
    Returns:
        float: Annäherung ans Integral
    '''
    utl.assert_is_vec(x)
    utl.assert_eq_shape(x, y)
    
    n = len(x) - 1 
    print(f'Summierte Trapezregel für tabellierte Daten mit variabler Schrittweite, n={n}')
    res = 0
    for i in range(n):
       step = (y[i] + y[i+1]) / 2 * (x[i+1] - x[i])
       print(f'\ti={i}; (y_{i} + y_{i+1}) / 2 * (x_{i+1} - x_i) = {step}')
       res += (y[i] + y[i+1]) / 2 * (x[i+1] - x[i])

    return res

def sum_Tf(x_grenzen: Tuple[float, float], n: int, f: types.NPValueToScalarFn, log = True) -> float:
    '''
    integriert f nach der summierten Trapezregel

    Parameters:
        x_grenzen: Die untere/obere Grenze des gesuchten Intervalls
        n: Anzahl stützstellen ZWINGEND INTEGER
        f: Zu integrierende Funktion
        log=True: ob geloggt werden soll
    Returns:
        float: Annäherung des Integrals
    '''
    assert isinstance(n, int) 

    a, b = x_grenzen
    a, b = float(a), float(b)
    assert b > a

    dprint = lambda s: print(s) if log else None

    h = abs((b-a)/n)

    res = (f(a) + f(b)) / 2
    dprint(f'Summierte Trapezregel im Intervall [{a}, {b}], h={h}')
    dprint(f'(f(a) + f(b)) / 2 = {res}')

    for i in range(1, n):
        x_i = a + i*h
        f_xi = f(x_i)
        dprint(f'\ti={i}: x_i={x_i} -> f(x_i) = {f_xi}')
        res += f_xi

    dprint(f'Nach summieren der f(x_i): {res}')
    res *= h
    dprint(f'Endresulat (*= h): {res}')

    return res

def sum_Sf(x_grenzen: Tuple[float, float], n: int, f: types.NPValueToScalarFn) -> float:
    '''
    integriert f nach der summierten Simpsonregel

    Parameters:
        x_grenzen: Die untere/obere Grenze des gesuchten Intervalls
        n: Anzahl stützstellen ZWINGEND INTEGER
        f: Zu integrierende Funktion
    Returns:
        float: Annäherung des Integrals
    '''
    assert isinstance(n, int) 

    a, b = x_grenzen
    a, b = float(a), float(b)
    assert b > a

    h = abs((b-a)/n)

    trapez_teil = sum_Tf(x_grenzen, n, f) 
    rechteck_teil = 2 * sum_Rf(x_grenzen, n, f)
    
    res = 1/3 * (trapez_teil + rechteck_teil)

    print(f'Summierte Simpsonregel im Intervall [{a}, {b}], h={h}')
    print(f'Aus Trapezregel: {trapez_teil}')
    print(f' + 2 * Rechteckregel: {rechteck_teil}')
    print(f' * 1/3 = {res}')

    return res




import unittest

class IntegrationTest(unittest.TestCase):
    def test_sum_Tf_FS20_A1b(self):
        u = 2_000
        m_0 = 10_000
        q = 100
        g = 9.8

        def v(t):
            return u * np.log(m_0 / (m_0 - q*t)) - g*t

        h = 10

        a = 0.
        T = 30.

        n = int(np.ceil((T-a)/h))
        assert n == 3

        actual = sum_Tf((a, T), n, v)
        self.assertAlmostEqual(actual, 5726.8, places=1)

    def test_sum_Rf_Tf_Sf_S8_A2(self):
        m = 10 # kg

        def R(v):
            return -v * np.sqrt(v)
        def f_orig(v):
            return m/R(v)
        # grenzen a=20, b=5
        # -> invertieren und funktion *-1
        x_grenzen = 5, 20
        def f(v):
            return -f_orig(v)
        
        n = 5

        res_rf = sum_Rf(x_grenzen, n, f)
        res_tf = sum_Tf(x_grenzen, n, f)
        res_sf = sum_Sf(x_grenzen, n, f)
        self.assertAlmostEqual(res_rf, 4.3823144)
        self.assertAlmostEqual(res_tf, 4.6581815)
        self.assertAlmostEqual(res_sf, 4.4742701)

    def test_sum_Tf_var_h_S8_A3(self):
        r = np.array([0, 800, 1_200, 1_400, 2_000, 3_000, 3_400, 3_600, 4_000, 5_000,
                      5_500, 6_370], dtype=np.float64)
        r = r * 1_000 # km -> m
        rho = np.array([13_000, 12_900, 12_700, 12_000, 11_650, 10_600,
                       9_900, 5_500, 5_300, 4_750, 4_500, 3_300], dtype=np.float64)

        def m(r, rho):
            return rho * 4 * np.pi * r**2

        y = m(r, rho)

        m_actual = sum_Tf_variable_h(r, y)
        m_expected = 6.026e24
        self.assertAlmostEqual(m_actual, m_expected, delta=1e20) 

    def test_romberg_S9_A3(self):
        def f(x):
            return np.cos(x**2)
        
        m = 4

        x_grenzen = 0, np.pi

        actual = romberg(x_grenzen, f, m)

        self.assertAlmostEqual(actual, 0.564187600278486)

        
    def test_romberg_S10_A1(self):
        m = 97_000 # kg
        v0 = 100 # m/s
        
        def f(v):
            return m / (5 * v**2 + 570_000)

        actual = romberg((0, v0), f, 4)

        self.assertAlmostEqual(actual, 16.5446, places=4)

