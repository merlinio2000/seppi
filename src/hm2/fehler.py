import math
import numpy as np
from typing import Tuple

'''
--------------------------------------------------
Interpolation/Ausgleichsrechnung
--------------------------------------------------
'''

def absfehler_lagrange_interpolation(x: float, x_messpunkte: np.ndarray, max_nplus1te_ableitung: float):
    '''
    REIN THEORETISCH weil n+1-te ableitung der tatsächlichen Funktion f bekannt
    sein muss
    
    Parameters:
        x: der Punkt an dem der Fehler gesucht ist
        x_messpunkte: Alle x der Messpunkte mit länge n+1
        max_nplus1te_ableitung: das Maximum der n+1 ten Ableitung der tatsächlichen
             Funktion f auf [x_0, x_n]
    Returns:
        float: maximaler absoluter fehler an x
    '''
    n = len(x_messpunkte) - 1
    res = 1.
    for x_i in x_messpunkte:
        res *= x - x_i
    res = abs(res) / math.factorial(n+1)
    return res * abs(max_nplus1te_ableitung)

'''
--------------------------------------------------
Integration
--------------------------------------------------
'''

def absfehler_sum_Rf(intervall: Tuple[float, float], h: float, max_2te_ableitung: float) -> float:
    '''
    Berechnet den maximalen Fehler bei der Rechteckregel im Intervall

    Parameters:
        intervall: inklusive Grenzen [a,b] in denen die Regel angewandt wurde
        h: verwendete Schrittweite
        max_2te_ableitung: das Maximum der 2. Ableitung der Funktion 
                        im Intervall [a,b]
    Returns:
        float: Der maximale Fehler
    '''
    a, b = intervall
    assert b > a
    return h**2/24 * (b-a) * abs(max_2te_ableitung)

def max_schrittweite_fuer_fehler_sum_Rf(intervall: Tuple[float, float], max_fehler: float, \
        max_2te_ableitung: float) -> float:
    '''
    Rechteckregel: 
    Rechnet die maximale Schrittweite h aus damit der Fehler <= max_fehler ist


    Parameters:
        intervall: inklusive Grenzen [a,b] in denen die Regel angewandt wird
        max_fehler: maximaler Fehler der erreicht werden soll
        max_2te_ableitung: das Maximum der 2. Ableitung der Funktion 
                        im Intervall [a,b]
    Returns:
        float: maximale Schrittweite
    '''
    max_2te_ableitung = abs(max_2te_ableitung)
    a, b = intervall
    assert b > a
    return np.sqrt(24 * max_fehler / ((b-a) * max_2te_ableitung))








def absfehler_sum_Tf(intervall: Tuple[float, float], h: float, max_2te_ableitung: float) -> float:
    '''
    Berechnet den maximalen Fehler bei der Trapezregel im Intervall

    Parameters:
        intervall: inklusive Grenzen [a,b] in denen die Regel angewandt wurde
        h: verwendete Schrittweite
        max_2te_ableitung: das Maximum der 2. Ableitung der Funktion 
                        im Intervall [a,b]
    Returns:
        float: Der maximale Fehler
    '''
    a, b = intervall
    assert b > a
    return h**2/12 * (b-a) * abs(max_2te_ableitung)

def max_schrittweite_fuer_fehler_sum_Tf(intervall: Tuple[float, float], max_fehler: float, \
        max_2te_ableitung: float) -> float:
    '''
    Trapezregel: 
    Rechnet die maximale Schrittweite h aus damit der Fehler <= max_fehler ist


    Parameters:
        intervall: inklusive Grenzen [a,b] in denen die Regel angewandt wird
        max_fehler: maximaler Fehler der erreicht werden soll
        max_2te_ableitung: das Maximum der 2. Ableitung der Funktion 
                        im Intervall [a,b]
    Returns:
        float: maximale Schrittweite
    '''
    max_2te_ableitung = abs(max_2te_ableitung)
    a, b = intervall
    assert b > a
    return np.sqrt(12 * max_fehler / ((b-a) * max_2te_ableitung))








def absfehler_sum_Sf(intervall: Tuple[float, float], h: float, max_4te_ableitung: float) -> float:
    '''
    Berechnet den maximalen Fehler bei der Simpsonregel im Intervall

    Parameters:
        intervall: inklusive Grenzen [a,b] in denen die Regel angewandt wurde
        h: verwendete Schrittweite
        max_4te_ableitung: das Maximum der 4. Ableitung der Funktion 
                        im Intervall [a,b]
    Returns:
        float: Der maximale Fehler
    '''
    a, b = intervall
    assert b > a
    return h**4/2_880 * (b-a) * abs(max_4te_ableitung)


def max_schrittweite_fuer_fehler_sum_Sf(intervall: Tuple[float, float], max_fehler: float, \
        max_4te_ableitung: float) -> float:
    '''
    Simpsonregel: 
    Rechnet die maximale Schrittweite h aus damit der Fehler <= max_fehler ist


    Parameters:
        intervall: inklusive Grenzen [a,b] in denen die Regel angewandt wird
        max_fehler: maximaler Fehler der erreicht werden soll
        max_2te_ableitung: das Maximum der 4. Ableitung der Funktion 
                        im Intervall [a,b]
    Returns:
        float: maximale Schrittweite
    '''
    max_4te_ableitung = abs(max_4te_ableitung)
    a, b = intervall
    assert b > a
    return np.power(2_880 * max_fehler / ((b-a) * max_4te_ableitung), 1./4)






import unittest

class IntegrationTest(unittest.TestCase):
    def test_max_schrittweite_fuer_fehler_Tf_FS20_A1c(self):
        u = 2_000
        m_0 = 10_000
        q = 100
        g = 9.8

        def ddv(t):
            return u * q**2 / (m_0 - q*t)**2
        h = 10

        a = 0.
        T = 30.

        actual = max_schrittweite_fuer_fehler_sum_Tf((a, T), 0.1, ddv(T))

        self.assertAlmostEqual(actual, 0.3130, places=4)
