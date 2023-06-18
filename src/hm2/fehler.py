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

def max_h_min_n_fuer_fehler_sum_Rf(intervall: Tuple[float, float], max_fehler: float, \
        max_2te_ableitung: float, naiv=False) -> Tuple[float, int]:
    '''
    Rechteckregel: 
    Berücksichtigt den Fakt das n eine int anzahl Intervalle sein muss (h=(b-a)/n)
    Rechnet die maximale Schrittweite h aus damit der Fehler <= max_fehler ist


    Parameters:
        intervall: inklusive Grenzen [a,b] in denen die Regel angewandt wird
        max_fehler: maximaler Fehler der erreicht werden soll
        max_2te_ableitung: das Maximum der 2. Ableitung der Funktion 
                        im Intervall [a,b]
        naiv=False: berücksichtigt nicht das n ein int sein muss
    Returns:
        max_h: maximale Schrittweite
        min_n: minimale Anzahle Teilintervalle, 0 falls naiv
    '''
    max_2te_ableitung = abs(max_2te_ableitung)
    a, b = intervall
    assert b > a
    h_naiv = np.sqrt(24 * max_fehler / ((b-a) * max_2te_ableitung))
    if naiv: return h_naiv, 0
    # ab hier wird geschaut dass n ein int ist
    min_n = int(np.ceil((b-a)/h_naiv))
    max_h = (b-a)/min_n

    return max_h, min_n








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

def max_h_min_n_fuer_fehler_sum_Tf(intervall: Tuple[float, float], max_fehler: float, \
        max_2te_ableitung: float, naiv=False) -> Tuple[float, int]:
    '''
    Trapezregel: 
    Berücksichtigt den Fakt das n eine int anzahl Intervalle sein muss (h=(b-a)/n)
    Rechnet die maximale Schrittweite h aus damit der Fehler <= max_fehler ist


    Parameters:
        intervall: inklusive Grenzen [a,b] in denen die Regel angewandt wird
        max_fehler: maximaler Fehler der erreicht werden soll
        max_2te_ableitung: das Maximum der 2. Ableitung der Funktion 
                        im Intervall [a,b]
        naiv=False: berücksichtigt nicht das n ein int sein muss
    Returns:
        max_h: maximale Schrittweite
        min_n: minimale Anzahle Teilintervalle, 0 falls naiv
    '''
    max_2te_ableitung = abs(max_2te_ableitung)
    a, b = intervall
    assert b > a
    h_naiv = np.sqrt(12 * max_fehler / ((b-a) * max_2te_ableitung))
    if naiv: return h_naiv, 0
    # ab hier wird geschaut dass n ein int ist
    min_n = int(np.ceil((b-a)/h_naiv))
    max_h = (b-a)/min_n

    return max_h, min_n







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


def max_h_min_n_fuer_fehler_sum_Sf(intervall: Tuple[float, float], max_fehler: float, \
        max_4te_ableitung: float, naiv = False) -> Tuple[float, int]:
    '''
    Berücksichtigt den Fakt das n eine int anzahl Intervalle sein muss (h=(b-a)/n)
    Simpsonregel:
    Rechnet die maximale Schrittweite h aus damit der Fehler <= max_fehler ist


    Parameters:
        intervall: inklusive Grenzen [a,b] in denen die Regel angewandt wird
        max_fehler: maximaler Fehler der erreicht werden soll
        max_4te_ableitung: das Maximum der 4. Ableitung der Funktion 
                        im Intervall [a,b]
        naiv = False: berücksichtigt nicht das n ein int sein muss
    Returns:
        max_h: maximale Schrittweite
        min_n: minimale Anzahle Teilintervalle, 0 falls naiv
    '''
    max_4te_ableitung = abs(max_4te_ableitung)
    a, b = intervall
    assert b > a
    h_naiv = np.power(2_880 * max_fehler / ((b-a) * max_4te_ableitung), 1./4)
    if naiv: return h_naiv, 0
    # ab hier wird geschaut dass n ein int ist
    min_n = int(np.ceil((b-a)/h_naiv))
    max_h = (b-a)/min_n

    return max_h, min_n





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

        actual, _ = max_h_min_n_fuer_fehler_sum_Tf((a, T), 0.1, ddv(T), naiv=True)

        self.assertAlmostEqual(actual, 0.3130, places=4)

    def test_max_schrittweite_fuer_fehler_Rf_Tf_Sf_S9_A1(self):
        from sympy import symbols, ln, pprint, lambdify

        x = symbols('x')

        x_grenzen = 1, 2
        max_fehler = 1e-5

        sym_f = ln(x**2)

        # -2/x^2 -> maximal für x -> 0
        sym_ddf = sym_f.diff(x, 2) 
        ddf = lambdify([x], sym_ddf)
        # -12/x^4 -> maximal für x -> 0
        sym_d4f = sym_f.diff(x, 4) 
        d4f = lambdify([x], sym_d4f)
        # pprint(sym_ddf)
        # pprint(sym_4df)

        max_2te_ableitung = ddf(x_grenzen[0])
        max_4te_ableitung = d4f(x_grenzen[0])



        max_h_rf, min_n_rf = max_h_min_n_fuer_fehler_sum_Rf(x_grenzen, max_fehler, \
                max_2te_ableitung)
        max_h_tf, min_n_tf = max_h_min_n_fuer_fehler_sum_Tf(x_grenzen, max_fehler, \
                max_2te_ableitung)
        max_h_sf, min_n_sf = max_h_min_n_fuer_fehler_sum_Sf(x_grenzen, max_fehler, \
                max_4te_ableitung)
        max_h_rf_naiv, _ = max_h_min_n_fuer_fehler_sum_Rf(x_grenzen, max_fehler, \
                max_2te_ableitung, naiv=True)
        max_h_tf_naiv, _ = max_h_min_n_fuer_fehler_sum_Tf(x_grenzen, max_fehler, \
                max_2te_ableitung, naiv=True)

        self.assertAlmostEqual(min_n_rf, 92)
        self.assertAlmostEqual(min_n_tf, 130)
        self.assertAlmostEqual(min_n_sf, 5)

        self.assertAlmostEqual(max_h_rf_naiv, 1.0954e-2, places=6)
        self.assertAlmostEqual(max_h_tf_naiv, 7.7460e-3)
