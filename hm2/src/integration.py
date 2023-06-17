import numpy as np
from typing import Tuple
from ...utl import NPValueToScalarFn

def sum_Rf(x_grenzen: Tuple[float, float], n: int, f: NPValueToScalarFn) -> float:
    '''
    integriert f nach der summierten Rechteckregel

    Parameters:
        x_grenzen: Die untere/obere Grenze des gesuchten Intervalls
        n: Anzahl stützstellen
        f: Zu integrierende Funktion
    Returns:
        float: Annäherung des Integrals
    '''

    a, b = x_grenzen
    a, b = float(a), float(b)
    assert b > a

    h = abs((b-a)/n)

    res = 0.
    print(f'Summierte Rechteckregel im Intervall [{a}, {b}]')

    for i in range(n):
        x_i = a + i*h
        y_i = f(x_i + h/2)
        print(f'i={i}: x_i={x_i} -> f(x_i + h/2) = {y_i}')
        res += y_i

    print(f'Summierte Funktionswerte: {res}')
    res *= h
    print(f'Endresulat (*= h): {res}')

    return res

def sum_Tf(x_grenzen: Tuple[float, float], n: int, f: NPValueToScalarFn) -> float:
    '''
    integriert f nach der summierten Trapezregel

    Parameters:
        x_grenzen: Die untere/obere Grenze des gesuchten Intervalls
        n: Anzahl stützstellen
        f: Zu integrierende Funktion
    Returns:
        float: Annäherung des Integrals
    '''

    a, b = x_grenzen
    a, b = float(a), float(b)
    assert b > a

    h = abs((b-a)/n)

    res = (f(a) + f(b)) / 2
    print(f'Summierte Trapezregel im Intervall [{a}, {b}]')
    print(f'(f(a) + f(b)) / 2 = {res}')

    for i in range(n):
        x_i = a + i*h
        f_xi = f(x_i)
        print(f'i={i}: x_i={x_i} -> f(x_i) = {f_xi}')
        res += f_xi

    print(f'Nach summieren der f(x_i): {res}')
    res *= h
    print(f'Endresulat (*= h): {res}')

    return res




import unittest

class IntegrationTest(unittest.TestCase):
    def test_Rf_FS20_A1b(self):
        u = 2_000
        m_0 = 10_000
        q = 100
        g = 9.8

        def v(t):
            return u * np.log(m_0 / (m_0 - q*t)) - g*t

        h = 10

        a = 0.
        T = 30.

        n = np.ceil((T-a)/h)
        assert n == 3

        actual = sum_Tf((a, T), n, v)
        self.assertAlmostEquals(actual, 5726.8, places=1)


if __name__ == '__main__':
    unittest.main()


