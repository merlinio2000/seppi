import numpy as np
from ..util import types
from typing import Tuple

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

def sum_Tf(x_grenzen: Tuple[float, float], n: int, f: types.NPValueToScalarFn) -> float:
    '''
    integriert f nach der summierten Trapezregel

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

    res = (f(a) + f(b)) / 2
    print(f'Summierte Trapezregel im Intervall [{a}, {b}], h={h}')
    print(f'(f(a) + f(b)) / 2 = {res}')

    for i in range(1, n):
        x_i = a + i*h
        f_xi = f(x_i)
        print(f'\ti={i}: x_i={x_i} -> f(x_i) = {f_xi}')
        res += f_xi

    print(f'Nach summieren der f(x_i): {res}')
    res *= h
    print(f'Endresulat (*= h): {res}')

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
        self.assertAlmostEquals(actual, 5726.8, places=1)

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
